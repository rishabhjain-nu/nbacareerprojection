"""Turning box score counts into Gaussian pseudo-observations (§3.2, §3.3).

The exact observation model is negative-binomial for volume, binomial for
accuracy, Gaussian for availability.  None of those are linear-Gaussian, so the
v1 inference path uses the delta-method approximation:

    volume    z = log((Y + 0.5) / E)        R = 1/(Y + 0.5) + 1/phi_s
    accuracy  z = logit((M + 0.5)/(A + 1))  R = 1/(A p (1-p))
    avail     z = log(E)                    R = sigma_poss^2

Two things about `R` carry the whole model.

First it is **known per observation and heteroskedastic**.  That is what makes
minutes-weighting automatic: a 200-possession rookie arrives with an `R` twenty
times a starter's, so the Kalman gain discounts him without anyone writing a
minimum-minutes filter.

Second, the `1/phi_s` term is a **floor** that does not vanish as `E` grows.
Drop it and `R -> 0` for a 4000-possession season, the gain goes to 1, and the
model becomes certain that a healthy veteran's one season *is* his true talent.
Overdispersion from role volatility, matchups and injury is real and does not
average away.  §3.3 says do not omit it; this module does not let you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    ACCURACY_PAIRS, ACCURACY_STATS, AVAIL_IDX, IDX, MISSING_R, S, STATE_NAMES,
    VOLUME_STATS,
)

CONTINUITY = 0.5


@dataclass
class PanelGrid:
    """The panel reshaped to a dense (player x career-season) grid.

    A career occupies a contiguous block from a player's first observed season
    to his last.  Seasons he missed inside that span appear as rows with
    `observed=False` -- present in the grid so the state keeps aging, absent
    from the likelihood so nothing is inferred from them (§5.1).
    """

    player_ids: np.ndarray        # (N,)
    season_years: np.ndarray      # (N, T)  int, 0 where padded
    z: np.ndarray                 # (N, T, S)
    R: np.ndarray                 # (N, T, S)
    obs_mask: np.ndarray          # (N, T, S)  bool
    age: np.ndarray               # (N, T)  years
    in_span: np.ndarray           # (N, T)  bool: inside the career span
    observed: np.ndarray          # (N, T)  bool: a real season row exists
    exposure: np.ndarray          # (N, T)  possessions, NaN when unobserved
    games: np.ndarray             # (N, T)  games played, NaN when unobserved
    counts: dict[str, np.ndarray] # raw counts, (N, T), NaN when unobserved
    n_history: np.ndarray         # (N,) number of observed seasons
    last_index: np.ndarray        # (N,) grid index of the last observed season
    n_active: np.ndarray          # (T,) players still inside their span at step t
    injury: np.ndarray            # (N, T) bool, §4.2 R-inflation mask
    r_inflation: float = 1.5

    @property
    def n_players(self) -> int:
        return len(self.player_ids)

    @property
    def n_steps(self) -> int:
        return self.z.shape[1]


def volume_pseudo_obs(count: np.ndarray, exposure: np.ndarray,
                      phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z and R for the log per-possession rate of a count stat.

    `phi` is the per-stat negative-binomial overdispersion; `1/phi` is the
    irreducible part of the observation variance.
    """
    y = np.asarray(count, dtype=float) + CONTINUITY
    z = np.log(y / exposure)
    R = 1.0 / y + 1.0 / phi
    return z, R


def accuracy_pseudo_obs(made: np.ndarray, attempts: np.ndarray,
                        floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z and R for a shooting percentage, conditional on realized attempts.

    `floor` is the accuracy analogue of `1/phi`.  §3.3 does not mandate it and
    the pure binomial `R` is the spec's letter; it is included because the same
    argument that motivates the volume floor applies -- a 600-attempt shooter is
    not measured to binomial precision, since defensive attention and shot
    quality move his true rate within a season.  It is an estimated parameter,
    so if the data want it at zero they get it at zero.
    """
    a = np.asarray(attempts, dtype=float)
    m = np.asarray(made, dtype=float)
    p = (m + CONTINUITY) / (a + 1.0)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.log(p / (1.0 - p))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = 1.0 / np.maximum(a * p * (1.0 - p), 1e-8) + floor
    return z, R


def default_phi(panel: pd.DataFrame) -> np.ndarray:
    """Method-of-moments start for the volume overdispersion.

    For each stat, regress the empirical variance of the log rate on 1/(Y+0.5)
    across player-seasons; the intercept is 1/phi.  Only a starting value -- the
    likelihood re-estimates it -- but a sane one keeps the optimiser off the
    boundary.
    """
    phi = np.zeros(len(VOLUME_STATS))
    e = panel["possessions"].to_numpy(dtype=float)
    for k, stat in enumerate(VOLUME_STATS):
        y = panel[stat].to_numpy(dtype=float) + CONTINUITY
        lr = np.log(y / e)
        # Season-over-season change within a player is (2 x observation noise)
        # plus process noise; the part that survives at large Y is the floor.
        df = pd.DataFrame({"pid": panel["player_id"].to_numpy(), "lr": lr, "y": y})
        d = df.groupby("pid")["lr"].diff()
        w = df.groupby("pid")["y"].shift()
        ok = d.notna() & (w > 200) & (df["y"] > 200)
        resid_var = float(np.var(d[ok])) if ok.sum() > 50 else 0.05
        binom_part = float(np.mean(1.0 / df.loc[ok, "y"] + 1.0 / w[ok])) if ok.sum() else 0.0
        floor = max((resid_var - binom_part) / 2.0, 0.002)
        phi[k] = 1.0 / min(floor, 0.5)
    return phi


def build_grid(panel: pd.DataFrame, phi: np.ndarray | None = None,
               accuracy_floor: np.ndarray | None = None,
               sigma_poss: float = 0.15,
               injury_r_inflation: float = 1.5) -> PanelGrid:
    """Reshape the panel into the aligned grid the filter runs over."""
    if phi is None:
        phi = default_phi(panel)
    if accuracy_floor is None:
        accuracy_floor = np.full(len(ACCURACY_STATS), 0.005)

    panel = panel.sort_values(["player_id", "season_year"]).reset_index(drop=True)
    pids = panel["player_id"].to_numpy()

    starts = panel.groupby("player_id")["season_year"].min()
    ends = panel.groupby("player_id")["season_year"].max()
    span = (ends - starts + 1)

    # Players are ordered by career span, longest first.  That makes the set of
    # players still active at career-season `t` a contiguous prefix, so the
    # filter's hot loop slices `[:n_active]` instead of gathering rows -- careers
    # are short and right-skewed, and skipping the padding is a 4x saving over
    # filtering a rectangular block.
    uniq = span.sort_values(ascending=False, kind="stable").index.to_numpy()
    pid_to_row = {int(p): i for i, p in enumerate(uniq)}
    T = int(span.max())
    N = len(uniq)

    z = np.zeros((N, T, S))
    R = np.full((N, T, S), MISSING_R)
    mask = np.zeros((N, T, S), dtype=bool)
    age = np.zeros((N, T))
    in_span = np.zeros((N, T), dtype=bool)
    observed = np.zeros((N, T), dtype=bool)
    exposure = np.full((N, T), np.nan)
    games_played = np.full((N, T), np.nan)
    years = np.zeros((N, T), dtype=np.int64)

    start_year = np.array([starts[p] for p in uniq])
    span_len = np.array([span[p] for p in uniq])
    for i in range(N):
        in_span[i, : span_len[i]] = True
        years[i, : span_len[i]] = start_year[i] + np.arange(span_len[i])

    row = np.array([pid_to_row[int(p)] for p in pids])
    col = panel["season_year"].to_numpy() - start_year[row]
    observed[row, col] = True
    exposure[row, col] = panel["possessions"].to_numpy(dtype=float)
    games_played[row, col] = panel["games_played"].to_numpy(dtype=float)
    age[row, col] = panel["age"].to_numpy(dtype=float)

    # Ages for missed seasons interpolate off the observed ones: a player does
    # not stop getting older because he tore an ACL.
    for i in range(N):
        n = span_len[i]
        obs = observed[i, :n]
        if obs.sum() == 0:
            continue
        idx = np.arange(n)
        age[i, :n] = np.interp(idx, idx[obs], age[i, :n][obs])

    counts: dict[str, np.ndarray] = {}
    for c in ("fga_2p", "fgm_2p", "fga_3p", "fgm_3p", "fta", "ftm",
              "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"):
        arr = np.full((N, T), np.nan)
        arr[row, col] = panel[c].to_numpy(dtype=float)
        counts[c] = arr

    # ---- volume dimensions -------------------------------------------------
    e = exposure
    for k, stat in enumerate(VOLUME_STATS):
        y = counts[stat]
        zz, rr = volume_pseudo_obs(np.nan_to_num(y), np.where(np.isnan(e), 1.0, e), phi[k])
        j = IDX[stat]
        ok = observed & ~np.isnan(y)
        z[:, :, j] = np.where(ok, zz, 0.0)
        R[:, :, j] = np.where(ok, rr, MISSING_R)
        mask[:, :, j] = ok

    # ---- accuracy dimensions ----------------------------------------------
    for k, stat in enumerate(ACCURACY_STATS):
        made_c, att_c = ACCURACY_PAIRS[stat]
        m_, a_ = counts[made_c], counts[att_c]
        zz, rr = accuracy_pseudo_obs(np.nan_to_num(m_), np.nan_to_num(a_), accuracy_floor[k])
        j = IDX[stat]
        # Zero attempts is not a percentage.  Volume already recorded that he
        # did not shoot; inventing a rate here would be making something up.
        ok = observed & ~np.isnan(a_) & (np.nan_to_num(a_) > 0)
        z[:, :, j] = np.where(ok, zz, 0.0)
        R[:, :, j] = np.where(ok, rr, MISSING_R)
        mask[:, :, j] = ok

    # ---- availability ------------------------------------------------------
    j = AVAIL_IDX
    z[:, :, j] = np.where(observed, np.log(np.where(np.isnan(e), 1.0, e)), 0.0)
    R[:, :, j] = np.where(observed, sigma_poss ** 2, MISSING_R)
    mask[:, :, j] = observed

    # ---- injury seasons (§4.2) --------------------------------------------
    # A 22-game season is a worse measurement of talent than its possession
    # count alone implies -- the player was hurt for some of the possessions he
    # did log.  Inflate R rather than dropping the row.
    injury = np.zeros((N, T), dtype=bool)
    injury[row, col] = panel["injury_season_flag"].to_numpy()
    R = np.where(injury[:, :, None] & mask, R * injury_r_inflation, R)
    # The mask is carried on the grid because `refresh_R` rebuilds `R` from
    # scratch on every likelihood evaluation, and it has to apply the same
    # inflation -- otherwise this adjustment exists only in the array that
    # nothing downstream reads.

    n_history = observed.sum(axis=1)
    last_index = np.array([np.max(np.flatnonzero(observed[i])) if observed[i].any() else -1
                           for i in range(N)])

    n_active = in_span.sum(axis=0)
    assert np.all(in_span == (np.arange(N)[:, None] < n_active[None, :])), \
        "player ordering must make the active set a contiguous prefix"

    return PanelGrid(
        player_ids=uniq, season_years=years, z=z, R=R, obs_mask=mask, age=age,
        in_span=in_span, observed=observed, exposure=exposure,
        games=games_played, counts=counts,
        n_history=n_history, last_index=last_index, n_active=n_active,
        injury=injury, r_inflation=float(injury_r_inflation),
    )


def refresh_R(grid: PanelGrid, phi: np.ndarray, accuracy_floor: np.ndarray,
              sigma_poss: float, sigma_poss_inj: float | None = None,
              injury_infl: float | None = None) -> np.ndarray:
    """Recompute `R` for new noise hyperparameters without rebuilding the grid.

    Only the additive floors and `sigma_poss` change, so the count-dependent
    part is reusable.  This runs once per likelihood evaluation, so it matters.
    """
    R = np.full_like(grid.R, MISSING_R)
    for k, stat in enumerate(VOLUME_STATS):
        j = IDX[stat]
        y = np.nan_to_num(grid.counts[stat]) + CONTINUITY
        R[:, :, j] = np.where(grid.obs_mask[:, :, j], 1.0 / y + 1.0 / phi[k], MISSING_R)
    for k, stat in enumerate(ACCURACY_STATS):
        j = IDX[stat]
        made_c, att_c = ACCURACY_PAIRS[stat]
        a_ = np.nan_to_num(grid.counts[att_c])
        p = np.clip((np.nan_to_num(grid.counts[made_c]) + CONTINUITY) / (a_ + 1.0), 1e-4, 1 - 1e-4)
        val = 1.0 / np.maximum(a_ * p * (1.0 - p), 1e-8) + accuracy_floor[k]
        R[:, :, j] = np.where(grid.obs_mask[:, :, j], val, MISSING_R)
    # §4.2: an injury season is a worse measurement of talent than its
    # possession count alone implies -- the player was hurt for some of the
    # possessions he did log.  Inflate R rather than dropping the row.
    #
    # Availability gets its **own two-regime variance** rather than a shared
    # multiplier, because the two regimes are not a little different, they are
    # a different distribution: year-over-year sd of log possessions is 0.59
    # between two healthy seasons and 1.60 when either is injury-flagged.  A
    # single Gaussian split the difference at 1.05, which meant a durable
    # player's decade of consistent seasons was read through an observation
    # variance calibrated mostly by other players' injuries -- and the filter
    # duly refused to believe him.  Both scales are estimated.
    infl = grid.r_inflation if injury_infl is None else injury_infl
    R = np.where(grid.injury[:, :, None] & grid.obs_mask, R * infl, R)

    j = AVAIL_IDX
    s_inj = sigma_poss if sigma_poss_inj is None else sigma_poss_inj
    var = np.where(grid.injury, s_inj ** 2, sigma_poss ** 2)
    R[:, :, j] = np.where(grid.obs_mask[:, :, j], var, MISSING_R)
    return R


def observed_rate_frame(grid: PanelGrid) -> pd.DataFrame:
    """Raw observed rates, for plotting the filter against (§10.3).

    Display only -- nothing in `model/` reads this.
    """
    rows = []
    for i in range(grid.n_players):
        for t in range(grid.n_steps):
            if not grid.observed[i, t]:
                continue
            rec = {"player_id": int(grid.player_ids[i]), "t": t,
                   "season_year": int(grid.season_years[i, t]),
                   "age": grid.age[i, t], "possessions": grid.exposure[i, t]}
            for s in STATE_NAMES:
                rec[s] = grid.z[i, t, IDX[s]]
            rows.append(rec)
    return pd.DataFrame(rows)
