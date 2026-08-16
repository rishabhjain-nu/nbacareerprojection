"""Coherent joint availability system (Session-2 candidate, behind a flag).

The shipping path draws a possession total from the availability latent state
and then *re-derives* games and minutes-per-game from it with a post-hoc
regression (`derive.MinutesSplit`).  GP and MPG are therefore a deterministic
split of one possession draw, so their dispersions cannot be set independently
-- MPG intervals are too wide for durable stars while GP is simultaneously
over-projected for iron men and over-confident for fragile players -- and the
bounds (MPG<=48, GP<=schedule) are enforced by clipping, which piles mass at the
cap.

This module models the two quantities people read as first-class bounded
distributions and lets possessions be their coherent product:

    appears | active   ~  the existing absence sub-model  (P(plays))
    severity | appears ~  Multinomial(healthy / moderate / severe)   [injury states]
    GP | severity      ~  BetaBinomial(schedule, mean_s, kappa_s)     in [1, schedule]
    MPG | severity      ~  48 * sigmoid(N(mu(X), s(X)^2))             in [0, 48]
    possessions        =  GP * MPG * PACE                            (exact identity)

`PACE = 2.02` is definitional here (minutes := possessions / 2.02), so
possessions-per-minute is a constant.  GP conditional on appearing is genuinely
bimodal -- a player projected around 65 games either stays healthy near 75 or is
hurt near 30 -- which a single beta-binomial cannot represent; the severity
mixture is what makes GP calibrated and is exactly the injury-state structure
Session-2 asks for.  Missed whole seasons are the absence sub-model, not
re-modelled here (no double count).  A season *drawn* severe flips the next
season into a return-from-injury state, which lowers both the severity odds and
the MPG mean -- the recovery dynamic, through a covariate rather than a rule.

MPG keeps the availability latent anchor, so the Session-1 work (own-record EB,
aging, age x quality) still flows into minutes.  GP is deliberately driven by
the player's own recent *game count*, not by log-possessions (which is high for
a high-minute player and would push GP toward the cap even at 65 games).

All behind `use_joint_availability`; with the flag off the shipping path is
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AVAIL_IDX, S

PACE = 2.02
MAX_MPG = 48.0
DEFAULT_SCHEDULE = 82.0

# Severity of a *played* season, from games_share = games / schedule.  A 48-game
# season (share 0.59) is "moderate", a 75-game season (0.91) is "healthy".
SEV_HEALTHY, SEV_MODERATE, SEV_SEVERE = 0, 1, 2
SEV_NAMES = ["healthy", "moderate", "severe"]
HEALTHY_CUT, MODERATE_CUT = 0.75, 0.45


def classify_severity(games_share) -> np.ndarray:
    gs = np.asarray(games_share, float)
    out = np.full(gs.shape, SEV_HEALTHY, dtype=int)
    out[gs < HEALTHY_CUT] = SEV_MODERATE
    out[gs < MODERATE_CUT] = SEV_SEVERE
    return out


def build_schedule_lookup(panel):
    sub = panel[["player_id", "season_year", "schedule_length"]].dropna()
    return {(int(r.player_id), int(r.season_year)): float(r.schedule_length)
            for r in sub.itertuples()}


def _logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class PlayerContext:
    recent_mpg_logit: float
    recent_gp_logit: float
    mpg_trend: float
    gp_trend: float
    durability_mpg: float
    durability_gp: float          # shifts the severity linear predictor toward healthy
    last_severity: int
    recurrent: float


@dataclass
class AvailabilitySystem:
    mpg_beta: np.ndarray          # logit(MPG/48) mean, incl. severity indicators
    mpg_sd_coef: np.ndarray       # heteroskedastic log-sd on [1, anchor]
    sev_beta: np.ndarray          # (n_feat, 2) multinomial logits for moderate/severe vs healthy
    gp_mean: np.ndarray           # per-class mean gp_frac (3,)
    gp_kappa: np.ndarray          # per-class beta-binomial concentration (3,)
    anchor_mean: float
    anchor_sd: float
    quality_mean: float
    quality_sd: float
    gamma_skill: np.ndarray
    gp_cal: float = 0.0           # load-management trend on logit(gp_frac) per decade
    year_ref: float = 2015.0
    sched_lookup: dict = field(default_factory=dict)
    dur_mpg_by_pid: dict = field(default_factory=dict)
    dur_gp_by_pid: dict = field(default_factory=dict)
    offset_half_life: float = 2.0
    mpg_shrink_k: float = 3.0
    gp_shrink_k: float = 3.0

    def schedule_for(self, grid, i, t):
        return self.sched_lookup.get(
            (int(grid.player_ids[i]), int(grid.season_years[i, t])), DEFAULT_SCHEDULE)

    def _std_anchor(self, a):
        return (a - self.anchor_mean) / self.anchor_sd

    def _std_quality(self, q):
        return (q - self.quality_mean) / self.quality_sd

    def _mpg_features(self, anchor_z, age_c, quality_z, recent_mpg, mpg_trend,
                      sev_mod, sev_sev, ret_injury, year_c):
        one = np.ones_like(anchor_z)
        return np.stack([one, anchor_z, age_c * one, (age_c ** 2) * one,
                         recent_mpg * one, mpg_trend * one, quality_z,
                         sev_mod, sev_sev, ret_injury, year_c * one], axis=-1)

    def _sev_features(self, age_c, quality_z, recent_gp, gp_trend, ret_injury,
                      recurrent, dur_gp, year_c):
        one = np.ones_like(quality_z)
        return np.stack([one, age_c * one, (age_c ** 2) * one, recent_gp * one,
                         gp_trend * one, quality_z, ret_injury, recurrent * one,
                         dur_gp * one, year_c * one], axis=-1)

    def draw_season(self, rng, anchor, age, cur_state, ctx: PlayerContext,
                    ret_injury, year, schedule=DEFAULT_SCHEDULE):
        n = anchor.shape[0]
        anchor_z = self._std_anchor(anchor)
        age_c = (float(age) - 27.0) / 5.0
        year_c = (float(year) - self.year_ref) / 10.0
        quality_z = self._std_quality(cur_state @ self.gamma_skill)
        ri = ret_injury.astype(float)

        # ---- severity mixture (injury state), conditional on appearing ----
        Zs = self._sev_features(age_c, quality_z, ctx.recent_gp_logit, ctx.gp_trend,
                                ri, ctx.recurrent, ctx.durability_gp, year_c)
        eta = Zs @ self.sev_beta                     # (n, 2): moderate, severe
        logits = np.column_stack([np.zeros(n), eta])
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        u = rng.random(n)
        cum = np.cumsum(probs, axis=1)
        sev = (u[:, None] > cum[:, :-1]).sum(axis=1)  # 0/1/2

        # ---- GP | severity: beta-binomial per class, shifted by the
        # player's own within-class durability (a load-managed star plays
        # fewer "healthy" games than an iron man) -------------------------
        mean_s = _sigmoid(_logit(self.gp_mean[sev]) + ctx.durability_gp
                          + self.gp_cal * year_c)
        kappa_s = self.gp_kappa[sev]
        a = np.clip(mean_s * kappa_s, 1e-3, None)
        b = np.clip((1 - mean_s) * kappa_s, 1e-3, None)
        q = rng.beta(a, b)
        games = np.maximum(rng.binomial(int(schedule), q).astype(float), 1.0)

        # ---- MPG | severity: logistic-normal, heteroskedastic ------------
        Xm = self._mpg_features(anchor_z, age_c, quality_z, ctx.recent_mpg_logit,
                                ctx.mpg_trend, (sev == SEV_MODERATE).astype(float),
                                (sev == SEV_SEVERE).astype(float), ri, year_c)
        mu_mpg = Xm @ self.mpg_beta + ctx.durability_mpg
        sd_mpg = np.clip(np.exp(self.mpg_sd_coef[0] + self.mpg_sd_coef[1] * anchor_z),
                         0.05, 1.2)
        mpg = MAX_MPG * _sigmoid(mu_mpg + sd_mpg * rng.standard_normal(n))

        possessions = games * mpg * PACE
        next_ret = sev == SEV_SEVERE
        return games, mpg, possessions, next_ret


# ---------------------------------------------------------------------------
def _build_frame(ds, filt, sched_lookup):
    import pandas as pd
    grid = ds.grid
    rows = []
    for i in range(grid.n_players):
        obs = np.flatnonzero(grid.observed[i])
        prev_mpg = prev_gp = prev2_mpg = prev2_gp = np.nan
        sev_hist = []
        for t in obs:
            poss, g = grid.exposure[i, t], grid.games[i, t]
            sched = sched_lookup.get(
                (int(grid.player_ids[i]), int(grid.season_years[i, t])), DEFAULT_SCHEDULE)
            if not (poss > 0 and g > 0 and sched > 0):
                sev_hist.append(SEV_SEVERE)
                continue
            mpg = (poss / PACE) / g
            gp_frac = min(g / sched, 0.999)
            sev = int(classify_severity(np.array([g / sched]))[0])
            ret = 1.0 if (sev_hist and sev_hist[-1] == SEV_SEVERE) else 0.0
            recurrent = 1.0 if sum(s in (SEV_MODERATE, SEV_SEVERE)
                                   for s in sev_hist[-3:]) >= 2 else 0.0
            rows.append({
                "player_id": int(grid.player_ids[i]), "t": t,
                "season_year": int(grid.season_years[i, t]),
                "age": float(grid.age[i, t]),
                "anchor": float(filt.x1[i, t][AVAIL_IDX]), "state": filt.x1[i, t],
                "mpg": mpg, "gp": g, "schedule": sched, "gp_frac": gp_frac,
                "severity": sev,
                "mpg_logit": _logit(min(mpg / MAX_MPG, 0.999)),
                "recent_mpg": _logit(prev_mpg / MAX_MPG) if np.isfinite(prev_mpg) else 0.0,
                "recent_gp": _logit(min(prev_gp, 0.999)) if np.isfinite(prev_gp) else 0.0,
                "mpg_trend": (_logit(prev_mpg / MAX_MPG) - _logit(prev2_mpg / MAX_MPG))
                             if np.isfinite(prev_mpg) and np.isfinite(prev2_mpg) else 0.0,
                "gp_trend": (_logit(min(prev_gp, .999)) - _logit(min(prev2_gp, .999)))
                            if np.isfinite(prev_gp) and np.isfinite(prev2_gp) else 0.0,
                "ret_injury": ret, "recurrent": recurrent,
            })
            prev2_mpg, prev2_gp = prev_mpg, prev_gp
            prev_mpg, prev_gp = mpg, gp_frac
            sev_hist.append(sev)
    return pd.DataFrame(rows)


def _softmax_mnl(X, y, n_classes=3, iters=200, ridge=1e-3, lr=0.5):
    """Multinomial logistic regression (reference class 0), gradient descent."""
    n, d = X.shape
    beta = np.zeros((d, n_classes - 1))
    Y = np.eye(n_classes)[y][:, 1:]
    for _ in range(iters):
        eta = np.column_stack([np.zeros(n), X @ beta])
        P = np.exp(eta - eta.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        grad = X.T @ (Y - P[:, 1:]) - ridge * beta
        beta = beta + lr * grad / n
    return beta


def fit_availability(ds, filt, hazard) -> AvailabilitySystem:
    gamma = np.asarray(hazard.coef[3:3 + S], float).copy()
    gamma[AVAIL_IDX] = 0.0
    sched_lookup = build_schedule_lookup(ds.panel)
    df = _build_frame(ds, filt, sched_lookup)
    df = df[np.isfinite(df["mpg"]) & (df["mpg"] > 0.5) & (df["schedule"] > 0)].reset_index(drop=True)

    anchor = df["anchor"].to_numpy()
    amean, asd = float(anchor.mean()), float(anchor.std() or 1.0)
    states = np.stack(df["state"].to_numpy())
    quality = states @ gamma
    qmean, qsd = float(quality.mean()), float(quality.std() or 1.0)

    sysm = AvailabilitySystem(
        mpg_beta=np.zeros(11), mpg_sd_coef=np.zeros(2),
        sev_beta=np.zeros((10, 2)), gp_mean=np.zeros(3), gp_kappa=np.zeros(3),
        anchor_mean=amean, anchor_sd=asd, quality_mean=qmean, quality_sd=qsd,
        gamma_skill=gamma, sched_lookup=sched_lookup)

    az = sysm._std_anchor(anchor)
    age_c = (df["age"].to_numpy() - 27.0) / 5.0
    qz = sysm._std_quality(quality)
    year_c = (df["season_year"].to_numpy() - sysm.year_ref) / 10.0
    ret = df["ret_injury"].to_numpy()
    recur = df["recurrent"].to_numpy()
    sev = df["severity"].to_numpy()

    # ---- MPG mean + heteroskedastic sd -------------------------------------
    Xm = sysm._mpg_features(az, age_c, qz, df["recent_mpg"].to_numpy(),
                            df["mpg_trend"].to_numpy(),
                            (sev == SEV_MODERATE).astype(float),
                            (sev == SEV_SEVERE).astype(float), ret, year_c)
    ym = df["mpg_logit"].to_numpy()
    beta_m, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    resid = ym - Xm @ beta_m
    Xs = np.column_stack([np.ones_like(az), az])
    sd_coef, *_ = np.linalg.lstsq(Xs, np.log(np.maximum(resid ** 2, 1e-6)) + 1.2704, rcond=None)
    sysm.mpg_beta, sysm.mpg_sd_coef = beta_m, sd_coef / 2.0

    pid = df["player_id"].to_numpy()
    ages = df["age"].to_numpy()
    gp_frac = df["gp_frac"].to_numpy()

    # ---- GP | severity: per-class mean + beta-binomial concentration ------
    for c in (SEV_HEALTHY, SEV_MODERATE, SEV_SEVERE):
        fr = gp_frac[sev == c]
        sysm.gp_mean[c] = float(np.clip(fr.mean(), 0.02, 0.98))
        v = max(fr.var(), 1e-4)
        mm = sysm.gp_mean[c]
        sysm.gp_kappa[c] = float(np.clip(mm * (1 - mm) / v - 1.0, 2.0, 200.0))

    # ---- calendar (load-management) trend on within-class logit(gp_frac) ---
    gp_resid0 = _logit(gp_frac) - _logit(sysm.gp_mean[sev])
    Xc = np.column_stack([np.ones_like(year_c), year_c])
    cal, *_ = np.linalg.lstsq(Xc, gp_resid0, rcond=None)
    sysm.gp_cal = float(cal[1])

    # ---- GP durability offset = residual net of the calendar trend --------
    gp_resid = gp_resid0 - sysm.gp_cal * year_c
    sysm.dur_gp_by_pid = _prov_offsets(pid, ages, gp_resid, sysm.offset_half_life,
                                       sysm.gp_shrink_k)
    dgp = np.array([sysm.dur_gp_by_pid.get(int(p), 0.0) for p in pid])

    # ---- severity multinomial (uses the corrected durability_gp) ----------
    Zs = sysm._sev_features(age_c, qz, df["recent_gp"].to_numpy(),
                            df["gp_trend"].to_numpy(), ret, recur, dgp, year_c)
    sysm.sev_beta = _softmax_mnl(Zs, sev)

    # ---- MPG durability offset (full-model residual) ----------------------
    sysm.dur_mpg_by_pid = _prov_offsets(pid, ages, resid, sysm.offset_half_life,
                                        sysm.mpg_shrink_k)
    return sysm


def _prov_offsets(pid, age, resid, half_life, k):
    import pandas as pd
    df = pd.DataFrame({"pid": pid, "age": age, "r": resid})
    out = {}
    for p, g in df.groupby("pid"):
        a = g["age"].to_numpy()
        w = 0.5 ** ((a.max() - a) / half_life)
        off = float(np.sum(w * g["r"].to_numpy()) / np.sum(w))
        n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))
        out[int(p)] = off * n_eff / (n_eff + k)
    return out


def player_context(sysm: AvailabilitySystem, ds, filt, row) -> PlayerContext:
    grid = ds.grid
    obs = np.flatnonzero(grid.observed[row])
    mpgs, gpfr, sev = [], [], []
    for t in obs:
        poss, g = grid.exposure[row, t], grid.games[row, t]
        sched = sysm.schedule_for(grid, row, t)
        if poss > 0 and g > 0 and sched > 0:
            mpgs.append((poss / PACE) / g); gpfr.append(min(g / sched, 0.999))
            sev.append(int(classify_severity(np.array([g / sched]))[0]))
        else:
            mpgs.append(np.nan); gpfr.append(np.nan); sev.append(SEV_SEVERE)
    mpgs, gpfr = np.array(mpgs), np.array(gpfr)
    recent_mpg = _logit(mpgs[-1] / MAX_MPG) if np.isfinite(mpgs[-1]) else 0.0
    recent_gp = _logit(gpfr[-1]) if np.isfinite(gpfr[-1]) else 0.0
    mpg_trend = (recent_mpg - _logit(mpgs[-2] / MAX_MPG)) if len(mpgs) >= 2 and np.isfinite(mpgs[-2]) else 0.0
    gp_trend = (recent_gp - _logit(gpfr[-2])) if len(gpfr) >= 2 and np.isfinite(gpfr[-2]) else 0.0
    recurrent = 1.0 if sum(s in (SEV_MODERATE, SEV_SEVERE) for s in sev[-3:]) >= 2 else 0.0
    pid = int(grid.player_ids[row])
    return PlayerContext(recent_mpg, recent_gp, mpg_trend, gp_trend,
                         sysm.dur_mpg_by_pid.get(pid, 0.0),
                         sysm.dur_gp_by_pid.get(pid, 0.0),
                         sev[-1] if sev else SEV_HEALTHY, recurrent)
