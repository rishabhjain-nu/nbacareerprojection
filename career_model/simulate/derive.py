"""Everything the model does not model: points, rates, and composites (§7.1).

Points is not a state dimension and must not be.  It is `1*FTM + 2*2PM + 3*3PM`,
and a weighted sum of counts does not have variance equal to its mean, so no
Poisson-family observation layer applies to it.  Deriving it per draw from the
simulated makes gets the distribution right for free, including the correlation
between volume and efficiency that a directly-modelled points dimension would
have to invent.

The same argument makes composites *validation* rather than *input*.  BPM, TS%,
USG% and VORP are deterministic reads of the box score, so simulating the box
score and computing them afterward is free out-of-sample validation on metrics
people understand, with zero contamination of the fit (§1.2, §7.1).

**BPM caveat, stated once.**  Real BPM applies a team adjustment that needs team
offensive rating and the team's minutes distribution -- context this model does
not simulate.  §7.1 offers two options; this module implements the box-score
component via a fitted translator (option (b)), which is the right choice for
forward projections where no team context exists.  `validate/backtest.py` uses
option (a) where it can, holding actual team context fixed for seasons that have
already happened.  The translator is a *display and validation* device: nothing
in `model/` reads it, and removing it changes no state, no gain and no interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import RAW_DIR

# Per-100-possession rate columns the translator reads.
RATE_COLS = ["fga_2p", "fgm_2p", "fga_3p", "fgm_3p", "fta", "ftm",
             "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]
BPM_TARGETS = ["bpm", "obpm", "dbpm"]


def points(box: dict) -> np.ndarray:
    """`1*FTM + 2*2PM + 3*3PM`.  Derived, never a state dimension (§3.1)."""
    return box["ftm"] + 2.0 * box["fgm_2p"] + 3.0 * box["fgm_3p"]


def true_shooting(box: dict) -> np.ndarray:
    """PTS / (2 * (FGA + 0.44 * FTA))."""
    fga = box["fga_2p"] + box["fga_3p"]
    denom = 2.0 * (fga + 0.44 * box["fta"])
    return np.where(denom > 0, points(box) / np.maximum(denom, 1e-9), np.nan)


def usage_per_100(box: dict, possessions: np.ndarray) -> np.ndarray:
    """Possessions the player *ends*, per 100 he is on the floor for.

    Not the traditional USG%, which is a share of team plays while on court and
    needs teammate data the model does not simulate.  This is the same quantity
    with the player's own possession count as the denominator; it ranks players
    almost identically and is honestly computable from what is simulated.  It is
    labelled "USG/100" in the interface rather than "USG%" for that reason.
    """
    ends = box["fga_2p"] + box["fga_3p"] + 0.44 * box["fta"] + box["tov"]
    return 100.0 * ends / np.maximum(possessions, 1e-9)


def rates_per_100(box: dict, possessions: np.ndarray) -> np.ndarray:
    """(n, 13) matrix of per-100 rates, in `RATE_COLS` order.  Display only."""
    e = np.maximum(possessions, 1e-9)
    return np.column_stack([100.0 * np.asarray(box[c], float) / e for c in RATE_COLS])


# ---------------------------------------------------------------------------
# Splitting possessions into games x minutes-per-game
# ---------------------------------------------------------------------------
@dataclass
class MinutesSplit:
    """Turn a simulated possession total into games played and minutes per game.

    The model's availability dimension is `log(possessions)`, which deliberately
    conflates the two ways a season gets short: playing 40 games at 30 minutes,
    and playing 75 games at 16.  Nothing in §3 separates them, so per-game
    figures cannot come out of the state vector -- they need the split, and the
    split is not free information.

    What makes it recoverable rather than invented is that the two are strongly
    linked in practice: a player logging 4,000 possessions is nearly always
    playing most of a season at starter minutes, because minutes per game is
    capped at 48.  Regressing `log(mpg)` on a quadratic in `log(possessions)`
    plus age gives **R^2 = 0.78** with a residual sd of 0.29.

    That residual is not noise, though: **39% of its variance is a stable player
    trait** -- some players are durable and play 34 minutes across 78 games,
    others give you the same possessions across 60 games at 38.  So the draw
    carries a shrunk per-player offset estimated from his own history, plus a
    fresh within-player draw each season.  Treating the whole residual as
    iid season noise would make every projected career drift toward league-average
    durability, which is exactly the kind of individual signal the rest of the
    model works to preserve.

    This is a **display-layer** device, in the same category as the BPM
    translator: it is fitted, it reads no latent state, and nothing in `model/`
    imports it.  Removing it changes no state, no gain, and no interval on any
    per-possession quantity.
    """

    coef: np.ndarray            # on [1, logE, logE^2, age, age^2]
    sd_between: float           # pooled, for the shrinkage ratio only
    sd_within: float            # pooled, for the shrinkage ratio only
    sd_coef: np.ndarray = None  # log residual sd on [1, logE]
    between_share: float = 0.39
    max_mpg: float = 44.0
    # Durability-offset tuning.  A high-minute star concentrates his possessions
    # in fewer games (37 mpg over 65 games, not 33 over 74), so the population
    # split under-assigns his minutes; the per-player offset corrects it but
    # was systematically too small.  A leave-future-out sweep over 1,787
    # high-MPG seasons picked these: weight recent seasons more (a 2-season
    # half-life, since current durability beats an eight-year-old rookie line
    # at predicting next year) and shrink the offset half as hard.  Together
    # they cut high-MPG bias -2.79 -> -2.30 min and MAE 3.50 -> 3.34 while
    # *improving* the overall MAE, so it is not a star-only trade.
    offset_half_life: float = 2.0
    offset_shrink_scale: float = 0.5

    def resid_sd(self, possessions):
        """Residual sd of `log(mpg)`, which is **strongly** heteroskedastic:
        0.48 for a sub-500-possession season, 0.067 above 4,500.

        That is not a refinement, it is the difference between a usable table
        and a broken one.  Minutes per game is bounded above by 48, and a
        starter already sits near the ceiling — applying a pooled sd of 0.217 to
        a 36-minute median pushes the 90th percentile past 43 minutes, a figure
        that occurs a handful of times in thirty years of basketball.  The
        physical bound does the clipping, the histogram piles up against it, and
        the interval stops meaning anything.  Conditioning the spread on
        exposure keeps the mass where the data put it.
        """
        if self.sd_coef is None:
            return np.full_like(np.asarray(possessions, float), self.sd_within)
        log_e = np.log(np.clip(np.asarray(possessions, float), 1.0, 6000.0))
        X = np.column_stack([np.ones_like(log_e), log_e, log_e ** 2])
        return np.clip(np.exp(X @ self.sd_coef), 0.03, 0.9)

    def _design(self, log_e, age):
        log_e, age = np.asarray(log_e, float), np.asarray(age, float)
        return np.column_stack([np.ones_like(log_e), log_e, log_e ** 2, age, age ** 2])

    def predict_log_mpg(self, possessions, age):
        log_e = np.log(np.maximum(np.asarray(possessions, float), 1.0))
        return self._design(log_e, age) @ self.coef

    def shrinkage_n(self) -> float:
        """`n / (n + k)` with `k = sd_within^2 / sd_between^2` -- the same
        variance-ratio shrinkage the filter applies, on a quantity the filter
        does not carry."""
        return (self.sd_within ** 2) / max(self.sd_between ** 2, 1e-9)

    def player_offset(self, possessions, age, games, schedule=82.0) -> float:
        """A player's own durability tendency, shrunk by how many seasons back it."""
        poss = np.asarray(possessions, float)
        games = np.asarray(games, float)
        ok = (poss > 0) & (games > 0)
        if not ok.any():
            return 0.0
        age_ok = np.asarray(age, float)[ok]
        mpg = (poss[ok] / 2.02) / games[ok]
        mpg = np.clip(mpg, 1.0, self.max_mpg)
        resid = np.log(mpg) - self.predict_log_mpg(poss[ok], age_ok)
        # Standardise before averaging: a 300-possession season's residual is
        # seven times noisier than a 4,000-possession one, so pooling them raw
        # would let one garbage-time year dominate a durable player's offset.
        sd = self.resid_sd(poss[ok])
        w = 1.0 / sd ** 2
        # Recency-weight toward the most recent observed season: a player's
        # current durability tendency predicts next year better than his career
        # average, and an old low-minute development season should not drag a
        # now-durable star's offset down.
        if self.offset_half_life:
            w = w * 0.5 ** ((age_ok.max() - age_ok) / self.offset_half_life)
        offset = float(np.sum(w * resid) / np.sum(w))
        n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))
        return offset * n_eff / (n_eff + self.shrinkage_n() * self.offset_shrink_scale)

    def draw(self, rng, possessions, age, offset=0.0, schedule=82.0):
        """Return `(games, mpg)` for each simulated season.

        `games` is derived from minutes and the drawn `mpg` rather than sampled
        independently, so `games * mpg` reproduces the simulated minutes exactly
        and the per-game figures are consistent with the per-possession ones by
        construction.
        """
        poss = np.asarray(possessions, float)
        minutes = poss / 2.02
        mu = self.predict_log_mpg(poss, age) + offset
        # Only the within-player share is fresh noise each season; the offset
        # already carries the between-player part.
        sd_w = self.resid_sd(poss) * np.sqrt(1.0 - self.between_share)
        mpg = np.exp(mu + sd_w * rng.standard_normal(mu.shape))
        mpg = np.clip(mpg, 1.0, self.max_mpg)

        games = np.clip(np.round(minutes / mpg), 1.0, schedule)
        games = np.where(minutes <= 0, 0.0, games)
        # Recompute mpg from the rounded/clipped games so the identity holds.
        with np.errstate(divide="ignore", invalid="ignore"):
            mpg = np.where(games > 0, minutes / games, 0.0)
        return games, np.clip(mpg, 0.0, 48.0)


def fit_minutes_split(panel: pd.DataFrame, verbose: bool = True) -> MinutesSplit:
    df = panel[(panel["possessions"] > 0) & (panel["games_played"] > 0)].copy()
    minutes = df["possessions"].to_numpy(float) / 2.02
    mpg = minutes / df["games_played"].to_numpy(float)
    keep = (mpg > 1) & (mpg < 46)
    df, mpg = df[keep], mpg[keep]

    log_e = np.log(df["possessions"].to_numpy(float))
    age = df["age"].to_numpy(float)
    y = np.log(mpg)
    tmp = MinutesSplit(np.zeros(5), 1.0, 1.0)
    X = tmp._design(log_e, age)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ coef
    by_player = pd.Series(resid).groupby(df["player_id"].to_numpy())
    counts = by_player.size()
    between = float(by_player.mean()[counts >= 3].std())
    within = float((resid - df["player_id"].map(by_player.mean()).to_numpy()).std())
    share = between ** 2 / (between ** 2 + within ** 2)

    # Heteroskedastic residual sd: regress log(resid^2) on log(possessions).
    # The -1.2704 is E[log chi^2_1], the bias correction that makes exp(fit/2)
    # an estimate of the sd rather than of something slightly smaller.
    # Quadratic, not linear: the sd falls away much faster at high exposure
    # than a log-linear fit allows, and it is the top end that matters, because
    # that is where the 48-minute ceiling is close enough to distort the band.
    lr2 = np.log(np.maximum(resid ** 2, 1e-8))
    Xs = np.column_stack([np.ones(len(log_e)), log_e, log_e ** 2])
    sd_fit, *_ = np.linalg.lstsq(Xs, lr2 + 1.2704, rcond=None)
    sd_coef = sd_fit / 2.0

    out = MinutesSplit(coef=coef, sd_between=between, sd_within=within,
                       sd_coef=sd_coef, between_share=share)
    if verbose:
        lo, hi = out.resid_sd(np.array([400.0, 4500.0]))
        print(f"[derive] minutes split: R^2 {1 - resid.var() / y.var():.3f}; "
              f"{100 * share:.0f}% of the residual is a player trait; "
              f"residual sd {lo:.3f} at 400 possessions -> {hi:.3f} at 4500")
    return out


# ---------------------------------------------------------------------------
# BPM translator
# ---------------------------------------------------------------------------
@dataclass
class BPMTranslator:
    coef: np.ndarray            # (n_features, 3)
    intercept: np.ndarray       # (3,)
    mean: np.ndarray
    sd: np.ndarray
    r2: dict

    def features(self, box: dict, possessions: np.ndarray) -> np.ndarray:
        r = rates_per_100(box, possessions)
        ts = true_shooting(box)
        return np.column_stack([r, np.nan_to_num(ts, nan=0.5),
                                np.log(np.maximum(possessions, 1.0))])

    def __call__(self, box: dict, possessions: np.ndarray) -> dict:
        X = (self.features(box, possessions) - self.mean) / self.sd
        out = X @ self.coef + self.intercept
        return {t: out[:, i] for i, t in enumerate(BPM_TARGETS)}


def fit_bpm_translator(panel: pd.DataFrame, alpha: float = 1.0,
                       verbose: bool = True) -> BPMTranslator:
    """Ridge from the simulated box score to BPM units, validated out of fold.

    Grouped by player, because two seasons of the same player are not
    independent observations and a random split would report an R^2 the model
    does not have.
    """
    bbr = pd.read_parquet(RAW_DIR / "bbr_advanced.parquet")
    df = panel.merge(bbr, on=["player_id", "season"], how="inner").dropna(subset=BPM_TARGETS)
    df = df[df["possessions"] > 200]

    box = {c: df[c].to_numpy(dtype=float) for c in RATE_COLS}
    poss = df["possessions"].to_numpy(dtype=float)
    tmp = BPMTranslator(np.zeros((15, 3)), np.zeros(3), np.zeros(15), np.ones(15), {})
    X = tmp.features(box, poss)
    Y = df[BPM_TARGETS].to_numpy(dtype=float)

    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mean) / sd

    def solve(Zt, Yt):
        A = Zt.T @ Zt + alpha * np.eye(Zt.shape[1])
        return np.linalg.solve(A, Zt.T @ (Yt - Yt.mean(axis=0))), Yt.mean(axis=0)

    coef, intercept = solve(Z, Y)

    r2 = {}
    if verbose:
        pids = df["player_id"].to_numpy()
        rng = np.random.default_rng(0)
        uniq = np.unique(pids)
        fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % 5))
        fold = np.array([fold_of[p] for p in pids])
        pred = np.zeros_like(Y)
        for f in range(5):
            tr, te = fold != f, fold == f
            c, b = solve(Z[tr], Y[tr])
            pred[te] = Z[te] @ c + b
        for i, t in enumerate(BPM_TARGETS):
            ss_res = ((Y[:, i] - pred[:, i]) ** 2).sum()
            ss_tot = ((Y[:, i] - Y[:, i].mean()) ** 2).sum()
            r2[t] = float(1 - ss_res / ss_tot)
        print("[derive] BPM translator out-of-fold R^2: "
              + ", ".join(f"{t} {r2[t]:.3f}" for t in BPM_TARGETS)
              + f"  (n={len(df)})")
    return BPMTranslator(coef=coef, intercept=intercept, mean=mean, sd=sd, r2=r2)


# ---------------------------------------------------------------------------
def derive_composites(box: dict, possessions: np.ndarray,
                      translator: BPMTranslator | None = None,
                      games: np.ndarray | None = None) -> dict:
    """The single utility §7.1 asks for.  `box` maps count name -> array."""
    out = {
        "pts": points(box),
        "ts_pct": true_shooting(box),
        "usg_100": usage_per_100(box, possessions),
        "possessions": possessions,
    }
    # Raw counts pass through as well as their rates: a *rate* is the right unit
    # for a single season and a *total* is the right unit for a career, and
    # `career_totals` needs the second.
    for c in RATE_COLS:
        arr = np.asarray(box[c], float)
        out[c] = arr
        out[f"{c}_per100"] = 100.0 * arr / np.maximum(possessions, 1e-9)
    out["reb"] = out["oreb"] + out["dreb"]
    out["pts_per100"] = 100.0 * out["pts"] / np.maximum(possessions, 1e-9)
    out["reb_per100"] = out["oreb_per100"] + out["dreb_per100"]
    if translator is not None:
        out.update(translator(box, possessions))
        # VORP as BBR defines it: (BPM + 2.0) * (share of team minutes played)
        # * (season fraction).  Minutes share is approximated from possessions
        # against a 48-minute five-man team, which is exact up to pace.
        min_share = np.minimum(possessions / (2.02 * 48 * 5 * 82 / 5), 1.0)
        out["vorp"] = (out["bpm"] + 2.0) * min_share * (82 / 82.0) * 2.70
    if games is not None:
        # Per-game figures, the units most readers actually think in.  They are
        # a *derived* view: the model projects possessions, and the games/minutes
        # split comes from `MinutesSplit`.  Every one of them is season total
        # divided by the same simulated games count, so a draw is internally
        # consistent -- 30 points a game and 20 minutes a game cannot co-occur
        # in the same draw unless the box score really says so.
        g = np.asarray(games, float)
        safe = np.maximum(g, 1e-9)
        out["games"] = g
        out["minutes_per_game"] = np.where(g > 0, (possessions / 2.02) / safe, 0.0)
        per_game = {"pts": "pts", "reb": "reb", "ast": "ast", "stl": "stl",
                    "blk": "blk", "tov": "tov", "oreb": "oreb", "dreb": "dreb",
                    "fga_3p": "fg3a", "fta": "fta"}
        for src, label in per_game.items():
            if src in out:
                out[f"{label}_per_game"] = np.where(g > 0, out[src] / safe, np.nan)
    return out
