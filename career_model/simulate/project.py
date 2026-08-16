"""Monte Carlo forward simulation (§6).

Three nested sources of variance, all of them required:

  1. **parameter** -- `(Q, delta, A, Sigma, gamma, phi_s)` drawn from the
     posterior, so the bands do not pretend the hyperparameters are known;
  2. **state** -- `theta_{i,T}` drawn from the filter's `P_{T|T}` and then
     propagated by `theta_{t+1} = m + A(theta_t - m) + eta`, `eta ~ N(0, Q)`.
     Because `A < 1` this is *mean-reverting*, so the h-step covariance is
     `P_h = A^h P_0 (A^h)' + sum_{j<h} A^j Q (A^j)'` -- it grows sub-linearly
     and saturates toward the stationary variance `Q/(1-A^2)`.  (It is NOT
     `P_0 + hQ`; that random-walk approximation, `A = I`, over-states the
     variance several-fold at long horizons.)  The Monte Carlo propagates `A`
     exactly, so it carries the correct covariance for free;
     `tests/test_simulation.py::test_covariance_propagation` pins it against the
     augmented-state analytic and direct Monte Carlo.
  3. **sampling** -- the negative-binomial and binomial draws at the end.

The old residual-bootstrap stack captured (3) and part of (1).  Getting (2)
right is why the predictive interval grows correctly with horizon here without
anyone tuning a widening factor.  Note that because the process mean-reverts, a
*displayed* band need not widen monotonically -- a declining veteran's band can
hold flat or narrow as his level drifts down while the latent variance still
grows -- so a non-widening band is not a bug.

Rookies enter through the same machinery from a different starting point: no
filtered state exists, so `theta` is drawn from the prior `m_i ~ N(f_GBM(x_i),
Sigma_player)` plus the debut-age offset and the stationary spread.  The result
is a year-1 band about as wide as a veteran's year-5 band, which is the truth
and which §8.1 forbids the interface from hiding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    ACCURACY_PAIRS, ACCURACY_STATS, AVAIL_IDX, IDX, S, STATE_NAMES, VOLUME_IDX,
    VOLUME_STATS,
)
from ..model import hierarchy as hier
from . import derive

# The hazard's age quadratic turns *upward* past its vertex (~age 40): the few
# 40-plus player-seasons in the panel belong to exactly the iron men who kept
# going, so the polynomial extrapolates to near-immortality.  Rather than the
# old hard wall at 42 -- which gave a 41-year-old logging 4,000 possessions a
# 0% chance of playing next season, a cliff no data supports either -- survival
# is multiplied by a ramp that leaves the fitted hazard untouched through
# AGE_RAMP_START and declines linearly to zero at MAX_AGE.  Nobody in the
# modern era has played past 44; the ramp encodes that judgment smoothly.
MAX_AGE = 46.0
AGE_RAMP_START = 41.0
COUNT_NAMES = ["fga_2p", "fgm_2p", "fga_3p", "fgm_3p", "fta", "ftm",
               "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]
MIN_PER_POSSESSION = 1.0 / 2.02      # league-stable; minutes are possessions rescaled
# Physical ceiling on a season: 82 games at the 44-minute sustainable maximum
# (the same ceiling `MinutesSplit` uses), at league pace.  ~7,288 -- just above
# the panel maximum of 7,195 (mid-2000s Iverson), unlike the old 6,000
# constant, which sat *below* the data and pinned the median of every
# high-availability young player's draw distribution to itself.
MAX_POSSESSIONS = 82.0 * 44.0 / MIN_PER_POSSESSION
# Where compression begins: the modern-era extreme (Beal 2019 logged 6,486,
# nobody since 2020 has cracked 6,200).  Draws below it pass through untouched.
SOFT_CAP_SHOULDER = 6100.0


def _soft_cap(e: np.ndarray, cap: float = MAX_POSSESSIONS,
              shoulder: float = SOFT_CAP_SHOULDER) -> np.ndarray:
    """Smooth ceiling: identity below `shoulder`, asymptoting at `cap`.

    A hard clip piles probability mass into an atom at the cap and the
    percentiles read straight off the artifact (a median of exactly 6,000 was
    the visible symptom).  This map is C1 -- identity through the whole
    observed bulk, then a tanh that compresses the availability state's
    over-heavy upper tail (a known model weakness, see the README) into the
    range a season can physically hold.  No draw exceeds `cap`, and no draw a
    real season could produce is distorted by more than a few possessions.
    """
    e = np.asarray(e, float)
    span = cap - shoulder
    over = np.maximum(e - shoulder, 0.0)
    return np.where(e <= shoulder, e, shoulder + span * np.tanh(over / span))


@dataclass
class Projection:
    player_id: int
    ages: np.ndarray               # (H,)
    season_years: np.ndarray       # (H,)
    alive: np.ndarray              # (n_draws, H) bool -- career still active
    theta: np.ndarray              # (n_draws, H, S)
    box: dict                      # count name -> (n_draws, H)
    possessions: np.ndarray        # (n_draws, H)
    games: np.ndarray              # (n_draws, H) simulated games played
    is_rookie: bool
    n_history: int
    # (n_draws, H) bool -- career active AND played that season (deviation #4).
    # A missed season is `alive & ~played`: the career continues, the box score
    # is zero.  Everything the reader thinks of as "this season" -- the per-game
    # table, career totals, the peak -- conditions on `played`, while longevity
    # conditions on `alive`.  Defaulted to `alive` so a Projection built without
    # it (older call sites, tests) behaves as before.
    played: np.ndarray = None

    def __post_init__(self):
        if self.played is None:
            self.played = self.alive

    @property
    def n_draws(self) -> int:
        return self.alive.shape[0]


def _draw_joint_state(rng, mean1, mean2, P11, P12, P22, n):
    """Draw `[theta; m]` jointly -- they are correlated and drawing them
    separately would understate the spread of the whole trajectory."""
    mu = np.concatenate([mean1, mean2])
    C = np.block([[P11, P12], [P12.T, P22]])
    C = 0.5 * (C + C.T) + 1e-8 * np.eye(2 * S)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
    draws = mu + rng.standard_normal((n, 2 * S)) @ L.T
    return draws[:, :S], draws[:, S:]


def _param_blocks(model, n_draws: int, n_param_sets: int, rng):
    """Assign draws to parameter sets (§6 step 1).

    Re-unpacking the hyperparameters for every one of 2000 draws costs more than
    it buys -- the parameter posterior is far smoother than the state and
    sampling noise -- so draws are grouped into `n_param_sets` blocks, each with
    its own `(A, Q, phi, gamma, ...)`.  With the default 64 blocks the Monte
    Carlo error in the parameter dimension is well under the width of the bands
    it contributes to.
    """
    base = model.fit.params
    if model.posterior is None or len(model.posterior) == 0:
        return [base] * 1, [model.hazard.coef], np.zeros(n_draws, dtype=int)
    idx = rng.integers(0, len(model.posterior), size=n_param_sets)
    sets = [hier.unpack(model.posterior[j], base) for j in idx]
    hz = [model.hazard.draw_coef(rng) for _ in range(n_param_sets)]
    assign = rng.integers(0, n_param_sets, size=n_draws)
    return sets, hz, assign


def _volume_counts(rng, theta_vol, exposure, phi):
    """`Y ~ NegBin(mu = E exp(theta), phi)`, parameterized so
    `Var = mu + mu^2/phi` exactly as §3.2 specifies.

    `phi` may be (n_vol,) or (n_draws, n_vol) -- the latter when each draw
    carries its own posterior parameter set.
    """
    mu = np.clip(exposure[:, None] * np.exp(np.clip(theta_vol, -12, 3)), 1e-6, 1e6)
    ph = np.broadcast_to(np.atleast_2d(phi), mu.shape)
    return rng.negative_binomial(ph, ph / (ph + mu))


# ---------------------------------------------------------------------------
# Age x quality availability aging (the star minute-drop fix)
# ---------------------------------------------------------------------------
# The fitted availability aging curve is a single population-average decline,
# and the population is dominated by role players who lose their rotation spot.
# Empirically, within-player minute retention is strongly quality-dependent: at
# age 31-32 an average player loses ~33%/yr of his possessions and an elite one
# ~11%; at 35-36, ~46% vs ~22% (n well into the hundreds per cell, not survivor
# noise).  A regression of within-player Delta-log(possessions) on the model's
# own state quality index confirms it -- the quality x age interaction is +0.082
# per year of age, t = 4.3, while the quality main effect is ~0: quality does
# not change how fast a *young* player's minutes move, but it strongly slows an
# *old* star's decline.  The model carries none of this (the aging spline is
# position-grouped, quality-blind), so it applies the role-player decline to
# MVPs and their minutes collapse while their per-100 rates hold.
#
# This is NOT the state-dependent-aging idea that was rejected: that used the
# player's own availability slope (survivor-selected noise, double-counting the
# level).  This uses his *quality* -- well-identified from thousands of
# possessions, exogenous to the availability-aging question, with a plain
# basketball mechanism (coaches keep their best players on the floor).  It is
# the availability analogue of the fix-6 hazard interaction.
#
# Applied as a projection-layer adjustment, mean-centred so an average-quality
# player is unchanged (the validated aggregate curve is preserved) and only the
# tails move.  Capped both ways: it moderates a star's decline, it never
# reverses it into a gain, and it cannot rescue a genuinely faded player.
from dataclasses import dataclass as _dataclass


HINGE_AGE = 31.0               # quality starts protecting minutes here, not before


@_dataclass
class AvailQualityAging:
    gamma: np.ndarray          # SKILL quality readout (hazard coupling, availability zeroed)
    qmean: float               # quality standardisation, over the transitions
    qsd: float
    c_step: float              # protection /sd at the onset age (31)
    c_slope: float             # per-year change in protection past 31 (near zero empirically)
    cap_lo: float = -0.10      # per-year adjustment floor (weak old players decline faster)
    cap_hi: float = 0.18       # ceiling (stars decline slower, never turn into a gain)

    def increment(self, age: float, theta: np.ndarray) -> np.ndarray:
        """Extra availability increment for this age and state, per draw.

        `theta` is (n_draws, S); returns (n_draws,).  Only the quality x age
        *interaction* is added -- the age main effect is already in the fitted
        delta.  The quality index deliberately excludes the availability
        dimension itself (`gamma[AVAIL_IDX] == 0`): including it would make the
        index partly measure current minutes, whose mean-reversion masks the
        skill-protection effect (contaminated t 1.6 -> clean t 3.2).

        The onset is a **step at 31**, not a ramp.  The empirical quality
        protection jumps to ~+0.09/sd at 31-32 and stays roughly flat
        (measured +0.09, +0.07, +0.08 across 31-32 / 33-34 / 35-37); a linear
        ramp from 31 -- the first version shipped -- gave only +0.02/sd at the
        first step (so a just-turned-32 star got almost no year-1 protection
        and his minutes still dropped) while over-protecting at 38.  The step
        matches the data and fixes both ends.  Zero below 31, where the data
        show no quality gap, so young stars are untouched.
        """
        if age < HINGE_AGE:
            return np.zeros(theta.shape[0])
        qz = np.clip((theta @ self.gamma - self.qmean) / self.qsd, -2.5, 2.5)
        s = self.c_step + self.c_slope * (age - HINGE_AGE) / 3.0
        return np.clip(qz * s, self.cap_lo, self.cap_hi)


def fit_avail_quality_aging(ds, filt, hazard) -> AvailQualityAging:
    """Estimate the quality x age interaction on within-player availability.

    Regresses within-player Delta-log(possessions) on a centred age quadratic
    plus quality x step(age>=31) and quality x slope past 31, over consecutive
    observed seasons in the 28-37 window where the decline lives.  Quality is
    the hazard's state coupling with the availability dimension removed (see
    `increment`).  Keeps only the two interaction coefficients for projection;
    the age terms are the model's existing curve.
    """
    grid = ds.grid
    gamma = np.asarray(hazard.coef[3:3 + S], float).copy()
    gamma[AVAIL_IDX] = 0.0
    ages, dlog, q = [], [], []
    for i in range(grid.n_players):
        obs = np.flatnonzero(grid.observed[i])
        for a, b in zip(obs[:-1], obs[1:]):
            if b != a + 1:
                continue
            pa, pb = grid.exposure[i, a], grid.exposure[i, b]
            if not (pa > 0 and pb > 0):
                continue
            ag = float(grid.age[i, a])
            if not (28.0 <= ag <= 37.0):
                continue
            ages.append(ag)
            dlog.append(float(np.log(pb) - np.log(pa)))
            q.append(float(filt.x1[i, a] @ gamma))
    ages, dlog, q = np.array(ages), np.array(dlog), np.array(q)
    qmean, qsd = float(q.mean()), float(q.std() or 1.0)
    qz = (q - qmean) / qsd
    ac = (ages - 32.0) / 3.0
    step = (ages >= HINGE_AGE).astype(float)
    hinge = np.maximum(ages - HINGE_AGE, 0.0) / 3.0
    X = np.column_stack([np.ones_like(ac), ac, ac ** 2, qz * step, qz * step * hinge])
    coef, *_ = np.linalg.lstsq(X, dlog, rcond=None)
    return AvailQualityAging(gamma=gamma, qmean=qmean, qsd=qsd,
                             c_step=float(coef[3]), c_slope=float(coef[4]))


# ---------------------------------------------------------------------------
# Injury-regime availability noise (§4.2, propagated forward)
# ---------------------------------------------------------------------------
# The fit estimates *two* availability observation scales -- `sigma_poss` for
# healthy seasons and `sigma_poss_inj` (~4x larger) for injury-flagged ones --
# but the simulation used to draw every future season with the healthy scale
# only, projecting a league in which nobody ever loses a season to injury.
# Forward simulation now draws a per-season injury indicator and, for those
# seasons, applies the injury-regime scale to the *downside half* of the noise
# (the flag marks sub-half-schedule seasons, which by construction shorten a
# season, never stretch it).  The visible effect is the left tail -- the lost
# seasons the old draws were missing -- with the median draw untouched.
INJURY_RATE_DEFAULT = 0.21     # pooled panel rate, the fallback when no fit is passed


def fit_injury_rate(panel) -> np.ndarray:
    """Logit-quadratic in age for P(injury-flagged season), fitted by IRLS.

    The flag (`build_panel` §4.2) marks seasons with under half the schedule
    played, debut seasons excluded.  The fit further conditions on the player
    having had a real role the season before (> 500 possessions), because for
    fringe players a sub-half-schedule season is usually *role* -- G-League
    stints, DNPs -- and their low availability state already carries that.
    Feeding the pooled rate (26.5%, vs 20.4% role-conditioned, and heavily
    inflated at young ages) into the mixture would double-count it.  What is
    left is the thing the state cannot see coming: a rotation player losing
    most of a season.  The rate rises from ~15% in the early 20s to ~26% past
    34, which is the injury curve everyone knows.
    Returns the coefficient triple on [1, a, a^2] with a = (age - 27) / 5.
    """
    p = panel.sort_values(["player_id", "season_year"]).copy()
    p["prev_poss"] = p.groupby("player_id")["possessions"].shift(1)
    df = p[(p["possessions"].notna()) & (p["possessions"] > 0)
           & (p["season_index"] > 0) & (p["prev_poss"] > 500)]
    a = (df["age"].to_numpy(float) - 27.0) / 5.0
    y = df["injury_season_flag"].to_numpy(float)
    X = np.column_stack([np.ones_like(a), a, a * a])
    beta = np.zeros(3)
    beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    for _ in range(30):
        eta = np.clip(X @ beta, -8.0, 8.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        step = np.linalg.solve((X * w[:, None]).T @ X, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def _injury_prob(beta, age: float) -> float:
    if beta is None:
        return INJURY_RATE_DEFAULT
    a = (float(age) - 27.0) / 5.0
    eta = beta[0] + beta[1] * a + beta[2] * a * a
    return float(1.0 / (1.0 + np.exp(-np.clip(eta, -8.0, 8.0))))


# ---------------------------------------------------------------------------
# Role-change innovations (S3-B)
# ---------------------------------------------------------------------------
# The dimensions a role change actually moves: shot profile (2PA/3PA/FTA) and
# playmaking (AST/TOV).  Accuracy and rebounding are not "role" in this sense.
ROLE_DIMS = ["fga_2p", "fga_3p", "fta", "ast", "tov"]
ROLE_DIMS_MASK = np.array([1.0 if n in ROLE_DIMS else 0.0 for n in STATE_NAMES])
# A season is a "role change" if its filtered volume state jumped more than this
# (log-rate; ~0.35 ~= a 42% move) from the prior season on any role dimension.
ROLE_CHANGE_CUT = 0.35


def fit_role_change(ds, filt):
    """Logistic P(role change next season | cutoff covariates).

    Cutoff-available covariates only: age, experience, recent volume volatility
    on the role dimensions, and recent playmaking/shot-volume swings.  This is
    the `pi_t` of the two-component innovation mixture; deliberately a simple
    logistic, not a hidden Markov model.
    """
    grid = ds.grid
    idx = [IDX[n] for n in ROLE_DIMS]
    age, vol, swing_ast, swing_shot, y = [], [], [], [], []
    ast_i, shot_i = IDX["ast"], IDX["fga_3p"]
    for i in range(grid.n_players):
        obs = np.flatnonzero(grid.observed[i])
        for k in range(1, len(obs) - 1):
            t0, t1, t2 = obs[k - 1], obs[k], obs[k + 1]
            th0, th1, th2 = filt.x1[i, t0], filt.x1[i, t1], filt.x1[i, t2]
            age.append(float(grid.age[i, t1]))
            vol.append(float(np.max(np.abs(th1[idx] - th0[idx]))))
            swing_ast.append(abs(th1[ast_i] - th0[ast_i]))
            swing_shot.append(abs(th1[shot_i] - th0[shot_i]))
            y.append(1.0 if np.max(np.abs(th2[idx] - th1[idx])) > ROLE_CHANGE_CUT else 0.0)
    age = np.array(age); vol = np.array(vol)
    X = np.column_stack([np.ones_like(age), (age - 27) / 5.0, vol,
                         np.array(swing_ast), np.array(swing_shot)])
    y = np.array(y)
    beta = np.zeros(X.shape[1]); beta[0] = np.log(max(y.mean(), 1e-3) / (1 - y.mean() + 1e-3))
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-8, None)
        step = np.linalg.solve((X * w[:, None]).T @ X + 1e-3 * np.eye(X.shape[1]),
                               X.T @ (y - p) - 1e-3 * beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return {"beta": beta, "base_rate": float(y.mean())}


def role_change_pi(role_model, ds, filt, row):
    grid = ds.grid
    idx = [IDX[n] for n in ROLE_DIMS]
    obs = np.flatnonzero(grid.observed[row])
    last = obs[-1]
    if len(obs) >= 2:
        th1, th0 = filt.x1[row, last], filt.x1[row, obs[-2]]
        vol = float(np.max(np.abs(th1[idx] - th0[idx])))
        s_ast = abs(th1[IDX["ast"]] - th0[IDX["ast"]])
        s_shot = abs(th1[IDX["fga_3p"]] - th0[IDX["fga_3p"]])
    else:
        vol = s_ast = s_shot = 0.0
    age_c = (float(grid.age[row, last]) - 27) / 5.0
    x = np.array([1.0, age_c, vol, s_ast, s_shot])
    eta = float(np.clip(x @ role_model["beta"], -30, 30))
    return float(np.clip(1.0 / (1.0 + np.exp(-eta)), 0.0, 0.6))


# Career length at which a player's own record and the cross-sectional prior get
# equal weight in the reversion target (see `_eb_reversion_target`).
EB_HALF_LIFE = 4.0


def _eb_reversion_target(model, ds, filt, player_row, ref_age=19.0,
                         K=EB_HALF_LIFE):
    """Empirical-Bayes reversion target for the projection.

    The model reverts `theta` toward `m_i`, the player's long-run level.  But
    `m_i` is badly identified for a certain kind of player: a late draft pick who
    turned into a star gets a low GBM prior (the trees read pick 41 as marginal),
    and `m_i` is observed only through noisy AR realizations, so even a decade of
    data does not fully overturn the prior.  Jokic's fitted `m_i` sits near 1,200
    possessions against a decade above 4,800, and mean reversion toward it eats
    ten minutes a game off his projection -- the failure the user reported.

    The fix does not touch the fit.  It replaces the reversion *target* with a
    career-length-weighted blend of the model's `m_i` and the level implied by
    the player's own **filtered current state**, which -- unlike `m_i` -- is
    well-measured.  `m_own = theta_{T|T} - L(age_T)` de-ages the current state to
    a common reference through the same level path the projection ages along, so
    reverting toward it makes the trajectory start where the player actually is
    and decline down the curve rather than collapse toward a mis-estimated mean.

        m_eff = w * m_own + (1 - w) * m_model,   w = n / (n + K)

    A rookie (n small) keeps the prior; an established star trusts his own
    record.  `K` is the only knob and is set by the backtest, not by eye.
    """
    grid = ds.grid
    p = model.fit.params
    last = int(grid.last_index[player_row])
    age_T = float(grid.age[player_row, last])
    n = int(grid.n_history[player_row])

    coefs = _delta_coefs_for(model, ds, player_row)
    L = _level_at(model, p.A, coefs, age_T, ref_age)
    m_model = filt.x2[player_row, last]
    m_own = filt.x1[player_row, last] - L
    w = n / (n + K)
    return w * m_own + (1.0 - w) * m_model, m_model


def grid_pos_of(ds, player_row):
    return ds.pos_idx[player_row]


def _delta_coefs_for(model, ds, player_row):
    """This player's aging-increment coefficients: league + position, scaled by
    his own fitted aging rate where one exists."""
    coefs = model.fit.delta_league + model.fit.delta_pos[grid_pos_of(ds, player_row)]
    if model.fit.player_scale is not None:
        coefs = coefs * float(model.fit.player_scale[player_row])
    return coefs


def _level_at(model, A, coefs, age: float, ref_age: float = 19.0) -> np.ndarray:
    """Level path L(age), anchored L(ref_age)=0, via L_{k+1} = A L_k + delta(a_k).

    With `A < 1` the state mean-reverts, so the level a constant increment
    settles at is the recursion's fixed point, not a cumulative sum -- this is
    the same transform `_eb_reversion_target` has always used, factored out so
    the availability estimator below de-ages through the identical path.
    """
    ages = np.append(np.arange(ref_age, age, 1.0), age)
    L = np.zeros(S)
    for k in range(1, len(ages)):
        d = model.age_basis(np.array([ages[k - 1]]))[0] @ coefs
        L = A * L + d
    return L


# ---------------------------------------------------------------------------
# Per-player availability EB (the durable-star fix)
# ---------------------------------------------------------------------------
# Availability is special among the fourteen dimensions: log(possessions) is
# *directly observed* each season.  There is no sampling-noise ladder between
# the box score and the state -- a 5,000-possession season is a 5,000-possession
# season -- only circumstance (injury, role) separates the observation from the
# latent level.  So a long consistent record is close to a direct measurement of
# that level, and it is exactly the long-record players whom the filter
# under-serves: their `m_i` anchor is under-identified (README, "Star decline"),
# and the availability posterior it drags down feeds every per-game number.
#
# Two corrections, both projection-layer, both per-player, both shrunk by the
# information actually in the record:
#
#   1. **Own-record level.**  Each observed season's log-possessions is de-aged
#      to the current age through the fitted level path, weighted by the fitted
#      per-regime observation variance (an injury-flagged season is a bad
#      measurement of the level, exactly as the filter's R-inflation says) plus
#      the process-noise drift accumulated since (an old season speaks about a
#      level that has since diffused).  The precision-weighted mean replaces
#      `w_own` of the filtered starting state and of the reversion target's
#      availability component.  A short or erratic record gets a small weight
#      and keeps the filter's answer.
#   2. **Injury propensity.**  The fix-3 mixture uses a league age curve for
#      P(injury-type season).  A decade of never being flagged is evidence; so
#      is a career of half-seasons.  The per-season rate becomes the age curve
#      shrunk toward the player's own flag share, with the strength of a
#      beta-binomial prior worth `INJ_PROPENSITY_STRENGTH` pseudo-seasons.
EB_AVAIL_K = 2.0              # pseudo-seasons; own-record weight = n_eff/(n_eff+K)
INJ_PROPENSITY_STRENGTH = 8.0


def _avail_own_level(model, ds, filt, player_row, ref_age=19.0, K=EB_AVAIL_K):
    """The player's own availability level at his current age, from his record.

    The estimator is a scalar, anchor-free Kalman filter over the record's
    *level residual against the aging curve*, `r_k = log(poss_k) - L(age_k)`:
    random-walk transition with the fitted availability process noise, and the
    fitted per-regime observation variance per season.  That is the main
    filter's own recursion with everything that hurts durable stars removed --
    no `m_i` anchor to drag the level and no cross-dimension terms -- so a
    decade of consistent workloads reads as exactly that.  Recency weighting
    falls out of the gain rather than a hand-tuned decay, and an injury-flagged
    season is nearly ignored as a measurement (`sigma_poss_inj` is ~10x the
    healthy sd), which is what gives a star coming off a lost season a
    bounce-back level rather than a collapsed one.

    A naive precision-weighted mean of de-aged seasons was tried first and
    moved stars the *wrong* way: it pools a career toward one constant
    residual, so a player who beats the curve for years -- the residual
    trending up -- gets his current level dragged down by his own past.  The
    filter tracks the trend; the mean erases it.

    Returns `(level, weight, L_T)` -- the level estimate at the current age,
    the blend weight `n / (n + K)`, and the level-path value at the current age
    (to express the level in `m`-space) -- or `None` with no usable record.
    """
    grid = ds.grid
    p = model.fit.params
    last = int(grid.last_index[player_row])
    age_T = float(grid.age[player_row, last])
    obs = np.flatnonzero(grid.observed[player_row])
    poss = grid.exposure[player_row, obs].astype(float)
    keep = poss > 0
    obs, poss = obs[keep], poss[keep]
    if len(obs) == 0:
        return None
    ages_k = grid.age[player_row, obs].astype(float)
    inj = grid.injury[player_row, obs].astype(bool)

    coefs = _delta_coefs_for(model, ds, player_row)
    L_T = float(_level_at(model, p.A, coefs, age_T, ref_age)[AVAIL_IDX])
    L_k = np.array([float(_level_at(model, p.A, coefs, float(a), ref_age)[AVAIL_IDX])
                    for a in ages_k])

    r_obs = np.log(np.maximum(poss, 1.0)) - L_k
    sig2 = np.where(inj, p.sigma_poss_inj ** 2, p.sigma_poss ** 2)
    q = float(p.Q()[AVAIL_IDX, AVAIL_IDX])

    r, P = float(r_obs[0]), float(sig2[0])
    for k in range(1, len(r_obs)):
        gap = max(float(ages_k[k] - ages_k[k - 1]), 1.0)
        P_pred = P + q * gap
        gain = P_pred / (P_pred + float(sig2[k]))
        r = r + gain * (float(r_obs[k]) - r)
        P = (1.0 - gain) * P_pred

    n = len(r_obs)
    return float(r + L_T), n / (n + K), L_T


# --- state-dependent aging (tried, REJECTED by its own A/B; default off) ----
# The level correction fixes where a durable star *starts*; his projected
# decline still follows the league+position curve, which for the players the
# fix targets is demonstrably steeper than their own history -- that is what
# "beating the curve" means.  This estimator measures the player's personal
# drift against the curve (the slope of his availability residual over his
# recent seasons) and adds the shrunk slope to every projected availability
# increment.
#
# **The A/B rejected it, at two clean cutoffs** (`outputs/ab_slope_*.log`):
# possessions CRPS 896 -> 936 (2018) and 910 -> 932 (2016), coverage down,
# top-bucket h=1 bias +5.4% -> +11.8% and -1.1% -> +3.8%.  The mechanism is
# worth recording: once the own-record *level* correction has put a player at
# his record, his past drift against the curve is already priced in -- the
# veterans' residual slopes are positive mostly because of survivor selection,
# not personal signal, so extrapolating them double-counts the same evidence
# and over-projects exactly the durable players the fix was meant to serve.
# This is the third independent attempt at per-player aging to fail its own
# validation (fit-level scalar: README deviation #3; permanent/transient
# split: tried externally, inconclusive).  The information is not there.
# The flag stays for reproducibility; nothing sets it.
AVAIL_SLOPE_K = 4.0        # pseudo-seasons of shrinkage on the personal slope
AVAIL_SLOPE_CAP = 0.15     # |log-possessions per year|; no runaway extrapolation
AVAIL_SLOPE_MAX_SEASONS = 8   # early-career role growth is not anti-aging


def _avail_own_slope(model, ds, filt, player_row, ref_age=19.0,
                     K=AVAIL_SLOPE_K) -> float:
    """Shrunk personal drift of the availability residual, per year.

    Weighted least squares of `r_k = log(poss_k) - L(age_k)` on age over the
    last `AVAIL_SLOPE_MAX_SEASONS` observed seasons, with the same
    regime-variance + process-drift weights the level estimator uses, shrunk
    by `(n-3)/((n-3)+K)` -- a slope through fewer than four points is noise
    and gets zero.  Capped so an erratic record cannot extrapolate a collapse
    or an ascension the panel has never seen.
    """
    grid = ds.grid
    p = model.fit.params
    last = int(grid.last_index[player_row])
    age_T = float(grid.age[player_row, last])
    obs = np.flatnonzero(grid.observed[player_row])
    poss = grid.exposure[player_row, obs].astype(float)
    keep = poss > 0
    obs, poss = obs[keep], poss[keep]
    if len(obs) < 4:
        return 0.0
    obs, poss = obs[-AVAIL_SLOPE_MAX_SEASONS:], poss[-AVAIL_SLOPE_MAX_SEASONS:]
    ages_k = grid.age[player_row, obs].astype(float)
    inj = grid.injury[player_row, obs].astype(bool)

    coefs = _delta_coefs_for(model, ds, player_row)
    L_k = np.array([float(_level_at(model, p.A, coefs, float(a), ref_age)[AVAIL_IDX])
                    for a in ages_k])
    r = np.log(np.maximum(poss, 1.0)) - L_k
    sig2 = np.where(inj, p.sigma_poss_inj ** 2, p.sigma_poss ** 2)
    q = float(p.Q()[AVAIL_IDX, AVAIL_IDX])
    w = 1.0 / (sig2 + q * np.maximum(age_T - ages_k, 0.0))

    x = ages_k - np.average(ages_k, weights=w)
    denom = float(np.sum(w * x * x))
    if denom < 1e-9:
        return 0.0
    slope = float(np.sum(w * x * (r - np.average(r, weights=w))) / denom)
    n = len(r)
    shrunk = slope * (n - 3) / ((n - 3) + K)
    return float(np.clip(shrunk, -AVAIL_SLOPE_CAP, AVAIL_SLOPE_CAP))


def _injury_record(ds, player_row):
    """(flagged seasons, observed seasons) -- the propensity evidence."""
    grid = ds.grid
    obs = np.flatnonzero(grid.observed[player_row])
    if not len(obs):
        return None
    return int(grid.injury[player_row, obs].sum()), int(len(obs))


def simulate(model, ds, player_row: int, filt, n_draws: int = 2000,
             horizon: int = 12, seed: int = 0, n_param_sets: int = 64,
             minutes_split=None, use_eb: bool = True, extra_gap: int = 0,
             injury_beta=None, use_avail_eb: bool = True,
             use_avail_slope: bool = False, avail_quality=None,
             avail_system=None, innovation="gaussian", t_nu=6.0,
             role_model=None, role_scale=3.0) -> Projection:
    """Project one established player forward from `theta_{T|T}` and `P_{T|T}`.

    `avail_system` (Session-2 candidate, default None = off) replaces the
    log-possessions -> MinutesSplit chain with a joint GP/MPG/possessions model.

    `innovation` (S3-B, default "gaussian") selects the forward innovation law:
    "student_t" (heavier tails, `t_nu`) or "mixture" (a `role_model`-driven
    fraction of role-change draws with `role_scale` extra variance on the
    shot/playmaking dimensions).  Gaussian reproduces the shipping path.

    `extra_gap` is the number of *known missed seasons* between the player's
    last observed season and the season being projected from -- a player who
    sat out 2026 with an Achilles tear projects from his 2025 state, but his
    next possible season is 2027, not 2026.  The gap years are rolled through
    the transition (survival flips and fresh process noise, no observation), so
    the projection resumes at the current season with the widened uncertainty a
    silent year deserves, and `P(active)` already prices in the chance the
    player never returns.

    `filt` is required and must be causal filter output.  The guard below is
    the enforcement point for the rule that projections never launch from a
    smoothed state: `theta_{t|T}` conditions on the whole panel, so using it as
    the starting point of a backtested forecast means the forecast has already
    seen the seasons it is being scored against.  The state read here is
    `x1[player_row, last]` -- the filtered value at the player's last observed
    season, which by construction saw nothing after it.
    """
    if filt is None or getattr(filt, "conditioning", None) != "filtered":
        raise ValueError(
            "simulate() requires causal filter output (conditioning='filtered'); "
            f"got {getattr(filt, 'conditioning', type(filt).__name__)!r}. "
            "Projections must start from theta_{T|T}, never from a smoothed state.")
    grid = ds.grid
    last = int(grid.last_index[player_row])
    if last < 0:
        raise ValueError("player has no observed seasons; use simulate_rookie")

    rng = np.random.default_rng(seed + int(grid.player_ids[player_row]))
    sets, hz_coefs, assign = _param_blocks(model, n_draws, n_param_sets, rng)

    theta, m = _draw_joint_state(
        rng, filt.x1[player_row, last], filt.x2[player_row, last],
        filt.P11[player_row, last], filt.P12[player_row, last],
        filt.P22[player_row, last], n_draws)

    # Recentre the reversion target on the empirical-Bayes blend, keeping the
    # posterior spread of the draw.  This is the fix for the star-decline
    # problem; `use_eb=False` recovers the raw model behaviour for comparison.
    target_mean = filt.x2[player_row, last]
    if use_eb and int(grid.n_history[player_row]) >= 1:
        m_eff, m_model = _eb_reversion_target(model, ds, filt, player_row)
        m = m + (m_eff - m_model)
        target_mean = m_eff

    # Availability gets its own, stronger correction (the durable-star fix):
    # the record measures the level directly, so both the starting state and
    # the reversion target are mean-shifted toward the de-aged own-record
    # level, by a weight that reflects how much record there is.  Spreads are
    # untouched -- this moves centres, not confidence.
    inj_record = None
    if use_avail_eb:
        own = _avail_own_level(model, ds, filt, player_row)
        if own is not None:
            level, w_own, L_T = own
            x1_avail = float(filt.x1[player_row, last][AVAIL_IDX])
            theta[:, AVAIL_IDX] += w_own * (level - x1_avail)
            m[:, AVAIL_IDX] += w_own * ((level - L_T) - float(target_mean[AVAIL_IDX]))
        inj_record = _injury_record(ds, player_row)

    # Experimental state-dependent aging: bend the projected availability
    # increments by the player's own shrunk drift against the curve.
    avail_slope = 0.0
    if use_avail_slope:
        avail_slope = _avail_own_slope(model, ds, filt, player_row)

    age0 = float(grid.age[player_row, last])
    year0 = int(grid.season_years[player_row, last])

    # The player's own durability tendency, from his observed seasons.
    offset = 0.0
    if minutes_split is not None:
        obs = np.flatnonzero(grid.observed[player_row])
        offset = minutes_split.player_offset(
            grid.exposure[player_row, obs], grid.age[player_row, obs],
            grid.games[player_row, obs])

    # His own rate-of-aging multiplier, estimated from his own trajectory.
    ps = model.fit.player_scale
    aging_scale = 1.0 if ps is None else float(ps[player_row])

    # Joint availability candidate: build the player's cutoff context once.
    avail_context = None
    if avail_system is not None:
        from . import availability as _av
        avail_context = _av.player_context(avail_system, ds, filt, player_row)

    # Role-change mixture (S3-B): per-player probability from cutoff covariates.
    role_pi = 0.0
    role_dims = None
    if innovation == "mixture" and role_model is not None:
        role_pi = float(role_change_pi(role_model, ds, filt, player_row))
        role_dims = ROLE_DIMS_MASK

    return _roll_forward(model, ds, rng, theta, m, age0, year0,
                         int(grid.player_ids[player_row]), ds.pos_idx[player_row],
                         sets, hz_coefs, assign, n_draws, horizon,
                         start_is_first_season=False,
                         n_history=int(grid.n_history[player_row]),
                         is_rookie=False, minutes_split=minutes_split,
                         mpg_offset=offset, aging_scale=aging_scale,
                         extra_gap=max(int(extra_gap), 0),
                         injury_beta=injury_beta, inj_record=inj_record,
                         avail_slope=avail_slope, avail_quality=avail_quality,
                         avail_system=avail_system, avail_context=avail_context,
                         innovation=innovation, t_nu=t_nu, role_pi=role_pi,
                         role_scale=role_scale, role_dims=role_dims)


def simulate_rookie(model, ds, x_row: np.ndarray, gbm_offset: np.ndarray,
                    age0: float, year0: int, pos_idx: int, player_id: int,
                    n_draws: int = 2000, horizon: int = 12, seed: int = 0,
                    n_param_sets: int = 64, minutes_split=None,
                    injury_beta=None) -> Projection:
    """Project a player with no NBA seasons at all (§8.1).

    Everything comes from the prior.  There is nothing to filter, so the only
    honest year-1 distribution is `m_i ~ N(f_GBM(x_i) + beta' x_i,
    Sigma_player)` shifted by the debut-age offset and spread by the stationary
    AR dispersion.  It is wide, and it is supposed to be.
    """
    rng = np.random.default_rng(seed + player_id)
    sets, hz_coefs, assign = _param_blocks(model, n_draws, n_param_sets, rng)
    p = model.fit.params

    m_mean = x_row @ p.beta + gbm_offset
    # theta_0 = m + debut-age offset (its own spline in the increment form).
    from ..model.fit_kf import LEVEL_PARAM
    if LEVEL_PARAM:
        _coefs = model.fit.delta_league + model.fit.delta_pos[pos_idx]
        theta_mean = m_mean + model.age_basis(np.array([age0]))[0] @ _coefs
    else:
        theta_mean = m_mean + model.init_basis(np.array([age0]))[0] @ model.fit.c_init_coefs
    D = p.stationary_dispersion()
    theta, m = _draw_joint_state(rng, theta_mean, m_mean,
                                 p.Sigma_p + D, p.Sigma_p, p.Sigma_p, n_draws)

    # A rookie has no durability history, so he starts at the league tendency.
    return _roll_forward(model, None, rng, theta, m, age0, year0, player_id, pos_idx,
                         sets, hz_coefs, assign, n_draws, horizon,
                         minutes_split=minutes_split, mpg_offset=0.0,
                         start_is_first_season=True, n_history=0, is_rookie=True,
                         injury_beta=injury_beta)


def _roll_forward(model, ds, rng, theta, m, age0, year0, player_id, pos_idx,
                  sets, hz_coefs, assign, n_draws, horizon,
                  start_is_first_season, n_history, is_rookie,
                  minutes_split=None, mpg_offset=0.0,
                  aging_scale: float = 1.0, extra_gap: int = 0,
                  injury_beta=None, inj_record=None,
                  avail_slope: float = 0.0, avail_quality=None,
                  avail_system=None, avail_context=None,
                  innovation="gaussian", t_nu=6.0, role_pi=0.0,
                  role_scale=3.0, role_dims=None) -> Projection:
    from ..model.fit_kf import LEVEL_PARAM
    basis = model.age_basis
    # League + position curve, scaled by this player's own aging rate.  A
    # rookie has no trajectory to estimate it from and gets the league rate.
    delta_coefs = (model.fit.delta_league + model.fit.delta_pos[pos_idx]) * aging_scale

    hz_inter = _hazard_inter(model.hazard)
    # Role-change innovations (S3-B).  `role_mask` marks the volume dimensions a
    # role change actually moves (shot profile, playmaking) so a mixture draw
    # widens those and not, say, free-throw accuracy.
    role_mask = np.ones(S) if role_dims is None else role_dims

    def _draw_eta():
        """Innovation eta ~ N(0,Q), or a heavier-tailed / mixture variant."""
        z = rng.standard_normal((n_draws, S))
        if innovation == "student_t":
            # multivariate-t: common per-draw scale sqrt(nu / chi2(nu)).
            g = rng.chisquare(t_nu, size=n_draws) / t_nu
            z = z / np.sqrt(g)[:, None]
        eta = np.einsum("nij,nj->ni", QL, z)
        if innovation == "mixture" and role_pi > 0.0:
            # a fraction of draws get a role-change kick: extra variance on the
            # role dimensions only, on top of the normal innovation.
            rc = rng.random(n_draws) < role_pi
            extra = np.einsum("nij,nj->ni", QL, rng.standard_normal((n_draws, S)))
            extra = extra * (np.sqrt(max(role_scale - 1.0, 0.0)) * role_mask)[None, :]
            eta = eta + np.where(rc[:, None], extra, 0.0)
        return eta

    def _step(cur, age, live):
        """One transition: survival flip on the state being left, then
        propagate.  Shared between the gap pre-roll and the projection loop so
        the two cannot drift apart."""
        surv_p = _p_survive(hz, age, cur, hz_inter)
        live = live & (rng.random(n_draws) < surv_p)
        # Must match the fitted parameterization.  Increment form (the base):
        # theta_{t+1} = m + A(theta_t - m) + delta(a).  Level form adds the
        # c(a+1) - A c(a) offset instead.
        if LEVEL_PARAM:
            c_now = basis(np.array([age[0]]))[0] @ delta_coefs
            c_nxt = basis(np.array([age[0] + 1.0]))[0] @ delta_coefs
            d = c_nxt - A * c_now
        else:
            d = basis(np.array([age[0]]))[0] @ delta_coefs
        eta = _draw_eta()
        nxt = m + A * (cur - m) + d + eta
        if avail_slope != 0.0:
            # Experimental state-dependent aging: the player's own shrunk
            # drift against the curve enters exactly like delta does, so its
            # long-run effect compounds through the same recursion.
            nxt[:, AVAIL_IDX] += avail_slope
        if avail_quality is not None:
            # Age x quality availability aging: quality slows an old star's
            # minute decline.  Uses the *current* state, so the protection
            # fades as a player's projected skill fades -- a faded star is not
            # rescued.  Enters like delta and compounds through the recursion.
            nxt[:, AVAIL_IDX] += avail_quality.increment(age[0] + 1.0, cur)
        return nxt, age + 1.0, live

    # Ages and seasons are deterministic and are filled in up front: the draw
    # loop can exit early once every draw has retired, and leaving trailing
    # zeros here would put blank ticks on the longevity axis.
    step0 = (0 if start_is_first_season else 1) + extra_gap
    ages = age0 + step0 + np.arange(horizon, dtype=float)
    years = year0 + step0 + np.arange(horizon, dtype=int)
    alive = np.zeros((n_draws, horizon), dtype=bool)
    played = np.zeros((n_draws, horizon), dtype=bool)
    theta_out = np.zeros((n_draws, horizon, S))
    poss_out = np.zeros((n_draws, horizon))
    games_out = np.zeros((n_draws, horizon))
    box = {c: np.zeros((n_draws, horizon)) for c in COUNT_NAMES}
    absence = getattr(model, "absence", None)
    # Return-from-injury state for the joint availability system: the first
    # projected season inherits it from the player's last observed severity.
    avail_ret = np.zeros(n_draws, dtype=bool)
    if avail_system is not None and avail_context is not None:
        from . import availability as _av
        avail_ret = np.full(n_draws, avail_context.last_severity == _av.SEV_SEVERE)

    # Per-draw parameter views, materialised once.
    A = np.stack([s.A for s in sets])[assign]                      # (n_draws, S)
    phi = np.stack([s.phi for s in sets])[assign]
    sigp = np.array([s.sigma_poss for s in sets])[assign]
    sigp_inj = np.array([getattr(s, "sigma_poss_inj", 0.0) or s.sigma_poss
                         for s in sets])[assign]
    Qs = [np.linalg.cholesky(s.Q() + 1e-10 * np.eye(S)) for s in sets]
    QL = np.stack(Qs)[assign]                                      # (n_draws, S, S)
    hz = np.stack(hz_coefs)[assign] if len(hz_coefs) > 1 else \
        np.broadcast_to(hz_coefs[0], (n_draws, len(hz_coefs[0])))

    live = np.ones(n_draws, dtype=bool)
    age = np.full(n_draws, age0)
    cur = theta.copy()

    # Known missed seasons between the last observed season and now: roll the
    # state through them with no observation.  Each gap year costs a survival
    # flip (the player may in fact be done) and adds a full Q of process noise
    # (nothing was learned about him while he was out).
    for _ in range(extra_gap):
        cur, age, live = _step(cur, age, live)

    for h in range(horizon):
        if h > 0 or not start_is_first_season:
            # ---- a+b. survival on the state being left (§3.5), propagate --
            cur, age, live = _step(cur, age, live)

        assert abs(age[0] - ages[h]) < 1e-9, "age bookkeeping diverged from the grid"
        alive[:, h] = live
        theta_out[:, h] = cur

        # ---- within-career absence (deviation #4) -------------------------
        # The career hazard has kept `live` alive; a live draw can still miss
        # this specific season.  Draw a per-season play indicator coupled to
        # the current state, so a fringe player's projected years carry the
        # real chance of a gap and a durable star's do not.  A missed season
        # keeps the career alive (it does not touch `live`) but contributes a
        # zero box score -- the explicit "missed the whole year" state the old
        # simulation lacked.
        if absence is not None:
            plays = live & (rng.random(n_draws)
                            < absence.p_survive(np.full(n_draws, ages[h]), cur))
        else:
            plays = live
        played[:, h] = plays

        # ---- c. possessions from the availability dimension ---------------
        # Two-regime observation noise, mirroring the fit (§4.2): most seasons
        # scatter around the latent availability with the healthy sd, but an
        # age-dependent fraction are injury-type seasons drawn with the much
        # larger injury-regime sd -- the left tail where a lost season lives.
        # The injury scale applies to the *downside only*: the flag marks
        # seasons with under half the schedule played, so by construction it
        # can shorten a season relative to the latent level, never stretch it.
        # The upside keeps the healthy scale, and the median draw is untouched.
        # The rate is the league age curve shrunk toward the player's own flag
        # share -- an iron man has earned a lower rate, a chronically injured
        # player a higher one, and a rookie has earned nothing either way.
        if avail_system is not None:
            # Joint availability candidate: severity -> GP (beta-binomial,
            # bounded by schedule) and MPG (logistic-normal, bounded 0-48);
            # possessions = GP x MPG x pace, reconciled by construction.  No
            # injury mixture, no soft cap, no MinutesSplit -- this path replaces
            # all three.  `avail_ret` carries the per-draw return-from-injury
            # state forward for the recovery dynamic.
            g_draw, mpg_draw, e, next_ret = avail_system.draw_season(
                rng, cur[:, AVAIL_IDX], ages[h], cur, avail_context,
                avail_ret, years[h])
            avail_ret = next_ret
            poss_out[:, h] = np.where(plays, e, 0.0)
            games_out[:, h] = np.where(plays, g_draw, 0.0)
        else:
            p_inj = _injury_prob(injury_beta, ages[h])
            if inj_record is not None:
                k_flag, n_obs = inj_record
                p_inj = np.clip((INJ_PROPENSITY_STRENGTH * p_inj + k_flag)
                                / (INJ_PROPENSITY_STRENGTH + n_obs), 0.03, 0.7)
            inj = rng.random(n_draws) < p_inj
            z = rng.standard_normal(n_draws)
            sig = np.where(inj & (z < 0), sigp_inj, sigp)
            log_e = cur[:, AVAIL_IDX] + sig * z
            e_raw = np.exp(np.clip(log_e, 0.0, np.log(MAX_POSSESSIONS) + 1.0))
            e = np.clip(_soft_cap(e_raw), 1.0, MAX_POSSESSIONS)
            # Masked by `plays`, not `live`: a missed season is a zero box score.
            poss_out[:, h] = np.where(plays, e, 0.0)
            if minutes_split is not None:
                g_draw, _ = minutes_split.draw(rng, e, np.full(n_draws, ages[h]),
                                               offset=mpg_offset)
                games_out[:, h] = np.where(plays, g_draw, 0.0)

        # ---- d. counts ----------------------------------------------------
        counts = _volume_counts(rng, cur[:, VOLUME_IDX], e, phi)
        for k, stat in enumerate(VOLUME_STATS):
            box[stat][:, h] = np.where(plays, counts[:, k], 0)
        for stat in ACCURACY_STATS:
            made_c, att_c = ACCURACY_PAIRS[stat]
            p_make = 1.0 / (1.0 + np.exp(-np.clip(cur[:, IDX[stat]], -12, 12)))
            att = box[att_c][:, h].astype(np.int64)
            box[made_c][:, h] = rng.binomial(att, p_make)

        if not live.any():
            break

    return Projection(player_id=player_id, ages=ages, season_years=years, alive=alive,
                      theta=theta_out, box=box, possessions=poss_out,
                      games=games_out, is_rookie=is_rookie, n_history=n_history,
                      played=played)


def _hazard_inter(h):
    """The age x quality interaction basis carried by a fix-6 hazard, or None
    for the original additive design (including any pre-fix pickle)."""
    g = getattr(h, "inter_gamma", None)
    if g is None:
        return None
    return np.asarray(g, float), float(h.inter_mu), float(h.inter_sd)


def _p_survive(hz_coef: np.ndarray, age: np.ndarray, theta: np.ndarray,
               inter=None) -> np.ndarray:
    from ..model.hazard import AGE_CENTRE, AGE_SCALE
    a = (age - AGE_CENTRE) / AGE_SCALE
    if inter is None:
        X = np.column_stack([np.ones_like(a), a, a ** 2, theta])
    else:
        g, mu, sd = inter
        X = np.column_stack([np.ones_like(a), a, a ** 2, theta,
                             a * ((theta @ g - mu) / sd)])
    p = 1.0 / (1.0 + np.exp(-np.clip(np.einsum("nj,nj->n", X, hz_coef), -30, 30)))
    # Old-age backstop.  The fitted quadratic is trusted through AGE_RAMP_START;
    # past it, the survivor-selected 40+ rows make the parabola turn upward, so
    # the ramp takes over and declines linearly to zero at MAX_AGE.  This
    # replaces the old hard `age <= 42` cliff with the same judgment made
    # smoothly: a great 41-year-old can play at 42, at longer and longer odds,
    # and nobody plays at 46.
    ramp = np.clip((MAX_AGE - (age + 1.0)) / (MAX_AGE - AGE_RAMP_START), 0.0, 1.0)
    return p * ramp


# ---------------------------------------------------------------------------
def summarise(proj: Projection, translator=None,
              percentiles=(5, 10, 25, 50, 75, 90, 95)) -> "pd.DataFrame":
    """Per-horizon percentile summary, conditional on the player being active.

    Conditioning matters: the median of a stat over all draws including the
    retired ones is a mixture of "his projected level" and "zero", which is a
    number describing nobody.  Survival is reported separately, and the career
    totals (which *should* mix in the zeros) are computed from the raw draws.
    """
    import pandas as pd

    rows = []
    box = proj.box
    for h in range(len(proj.ages)):
        # "If he plays" conditions on the seasons he actually played, not the
        # ones his career merely spanned -- a missed season's zero box score
        # would otherwise drag every conditional median down (deviation #4).
        live = proj.played[:, h]
        n_live = int(live.sum())
        rec_base = {"horizon": h + 1, "age": proj.ages[h],
                    "season_year": int(proj.season_years[h]),
                    "p_active": proj.alive[:, h].mean(),
                    "p_play": live.mean(), "n_live": n_live}
        if n_live < 10:
            continue
        cols = {c: box[c][live, h] for c in COUNT_NAMES}
        poss = proj.possessions[live, h]
        g = proj.games[live, h] if proj.games is not None else None
        comp = derive.derive_composites(cols, poss, translator,
                                        games=g if g is not None and g.any() else None)
        comp["minutes"] = poss * MIN_PER_POSSESSION
        for name, vals in comp.items():
            rec = dict(rec_base, stat=name)
            qs = np.nanpercentile(vals, percentiles)
            for q, val in zip(percentiles, qs):
                rec[f"p{q}"] = float(val)
            rec["mean"] = float(np.nanmean(vals))
            rows.append(rec)
        # State dimensions in their own units, for the fan chart's stat selector.
        for s, name in enumerate(STATE_NAMES):
            rec = dict(rec_base, stat=f"state:{name}")
            qs = np.nanpercentile(proj.theta[live, h, s], percentiles)
            for q, val in zip(percentiles, qs):
                rec[f"p{q}"] = float(val)
            rec["mean"] = float(np.nanmean(proj.theta[live, h, s]))
            rows.append(rec)
    if not rows:
        # Every horizon fell below the minimum surviving draws -- an ageing
        # player the hazard says is probably finished.  That is a real answer,
        # not a failure, so return an empty frame with the right columns and let
        # the survival curve carry the story.
        cols = ["horizon", "age", "season_year", "p_active", "p_play", "n_live",
                "stat", *[f"p{q}" for q in percentiles], "mean"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def career_totals(proj: Projection, translator=None,
                  percentiles=(5, 10, 25, 50, 75, 90, 95)) -> "pd.DataFrame":
    """Cumulative counts over the whole projected career, conditional on the
    survival draws.  This is where the hazard becomes visible to the user: a
    draw that exits at h=2 contributes two seasons and stops."""
    import pandas as pd

    # `played` masks the box score to zero on missed seasons already, so the
    # cumulative counts are unchanged by the choice, but "seasons" must count
    # seasons *played* -- a missed year is not a season on the stat line.
    box = {c: (proj.box[c] * proj.played).sum(axis=1) for c in COUNT_NAMES}
    poss = (proj.possessions * proj.played).sum(axis=1)
    comp = derive.derive_composites(box, poss, translator)
    comp["seasons"] = proj.played.sum(axis=1).astype(float)
    if proj.games is not None and proj.games.any():
        comp["games"] = (proj.games * proj.played).sum(axis=1)
    comp["minutes"] = poss * MIN_PER_POSSESSION

    rows = []
    for name, vals in comp.items():
        if (name.endswith("_per100") or name.endswith("_per_game")
                or name in ("ts_pct", "usg_100", "minutes_per_game")):
            continue                          # a career-total rate is not a total
        vals = np.asarray(vals, float)
        if not np.isfinite(vals).any():
            continue
        rec = {"stat": name, "mean": float(np.nanmean(vals))}
        for q, val in zip(percentiles, np.nanpercentile(vals, percentiles)):
            rec[f"p{q}"] = float(val)
        rows.append(rec)
    cols = ["stat", "mean", *[f"p{q}" for q in percentiles]]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def peak_distribution(proj: Projection, stats=("pts_per100", "reb_per100", "ast_per100"),
                      translator=None, percentiles=(5, 25, 50, 75, 95)) -> "pd.DataFrame":
    """Distribution of each draw's best projected season, and the age it lands."""
    import pandas as pd

    rows = []
    H = len(proj.ages)
    per_season = {}
    for h in range(H):
        cols = {c: proj.box[c][:, h] for c in COUNT_NAMES}
        comp = derive.derive_composites(cols, np.maximum(proj.possessions[:, h], 1.0),
                                        translator)
        for name in stats:
            if name in comp:
                # A peak season must be one he actually played.
                per_season.setdefault(name, []).append(
                    np.where(proj.played[:, h], comp[name], -np.inf))
    for name, seq in per_season.items():
        M = np.column_stack(seq)                       # (n_draws, H)
        has = np.isfinite(M).any(axis=1)
        if has.sum() < 10:
            continue
        best = M[has].max(axis=1)
        at = proj.ages[M[has].argmax(axis=1)]
        rec = {"stat": name}
        for q, v in zip(percentiles, np.nanpercentile(best, percentiles)):
            rec[f"value_p{q}"] = float(v)
        for q, v in zip(percentiles, np.nanpercentile(at, percentiles)):
            rec[f"age_p{q}"] = float(v)
        rec["peak_age_mean"] = float(np.mean(at))
        rows.append(rec)
    cols = ["stat", *[f"value_p{q}" for q in percentiles],
            *[f"age_p{q}" for q in percentiles], "peak_age_mean"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def survival_curve(proj: Projection) -> "pd.DataFrame":
    """Two distinct probabilities per season (deviation #4):

      * `p_active` -- the career is still going, straight from the hazard;
      * `p_play`   -- he actually appears that season, `p_active` times the
        within-career play probability.

    They differ by the chance of a mid-career gap, and keeping them apart is
    the whole point of the fix: the longevity view is a career statement, the
    season table's "will he play" column is an appearance statement, and the
    old code reported the first as if it were the second.
    """
    import pandas as pd
    return pd.DataFrame({
        "horizon": np.arange(1, len(proj.ages) + 1),
        "age": proj.ages,
        "season_year": proj.season_years,
        "p_active": proj.alive.mean(axis=0),
        "p_play": proj.played.mean(axis=0),
    })
