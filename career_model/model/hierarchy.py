"""Parameter container, priors, and the unconstrained parameterization (§4.1).

The hierarchy is league -> position group -> player, expressed in three places:

  * `beta` / `Sigma_player`   -- the player level of the *state*: `m_i` is drawn
                                 around a linear read of his pre-NBA covariates
                                 (§4.1, §5.3), and shrinks toward it exactly as
                                 far as his own sample size warrants.
  * `AgingCurves`             -- the league and position levels of `delta`.
  * `Q = Lambda Lambda' + Psi` -- shared across players, k=3 factors (§3.4).

Everything that has to stay positive or inside (0,1) is stored on an
unconstrained scale for the optimiser and the Metropolis sampler, and
transformed on the way out.  §4.1 also asks for a non-centred parameterization
on state innovations; in the Kalman path there are no explicit innovation
variables to centre -- they are integrated out analytically, which is the
strongest form of the same fix.  The reparameterization matters again in
`fit_numpyro`, where the states *are* parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config import ACCURACY_STATS, S, VOLUME_STATS

N_VOL = len(VOLUME_STATS)
N_ACC = len(ACCURACY_STATS)
Q_RANK = 3

# Persistence lives in (A_MIN, A_MAX) rather than (0,1): a stat with a=1 is a
# random walk with no pull to the player mean, which makes `m_i` unidentified.
A_MIN, A_MAX = 0.02, 0.995

# Floor on the availability observation sd, set from an **independent moment
# estimate** rather than from the likelihood or from any validation score.
#
# The unconstrained MLE goes to exactly zero: possessions really are observed
# exactly, and once injury seasons carry their own scale, almost all
# healthy-season movement is genuine level change rather than measurement
# error.  But sigma = 0 makes `P11` for that dimension identically zero, and
# that is a degeneracy, not a fit -- the filter becomes a pass-through of the
# last observation carrying no uncertainty at all, so the h=1 predictive
# interval loses its entire state component and comes out too narrow.  Measured:
# it dropped possessions coverage at h=1 to 0.74 against a nominal 0.80.
#
# The lag-1 autocorrelation of healthy-season changes in log(possessions) is
# -0.183.  Mean reversion at A = 0.83 accounts for about -(1-A)/2 = -0.087 of
# that on its own, leaving roughly -0.096 attributable to transient measurement
# error, which maps to sigma ~ 0.17.  That is where the floor sits.  It is a
# statement about the autocorrelation structure of the data, computed before the
# backtest was run and not adjusted after.
SIGMA_POSS_FLOOR = 0.17


def _softplus(x):
    return np.logaddexp(0.0, x)


def _inv_softplus(y):
    y = np.maximum(y, 1e-12)
    return np.where(y > 20, y, np.log(np.expm1(np.minimum(y, 20))))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


@dataclass
class Params:
    """Everything the filter needs, in natural (constrained) units."""

    A: np.ndarray                    # (S,)   diagonal persistence, in (0,1)
    Lam: np.ndarray                  # (S, k) low-rank process-noise loadings
    Psi: np.ndarray                  # (S,)   idiosyncratic process variance
    phi: np.ndarray                  # (N_VOL,) negative-binomial overdispersion
    acc_floor: np.ndarray            # (N_ACC,) accuracy observation-variance floor
    sigma_poss: float                # availability observation sd, healthy seasons
    sigma_poss_inj: float = 0.9      # availability observation sd, injury seasons
    injury_infl: float = 1.5         # R multiplier on the other dims, injury seasons
    beta: np.ndarray = field(default=None)        # (p_x, S) prior-mean coefficients
    Sigma_p: np.ndarray = field(default=None)     # (S, S) player-level covariance
    gbm_offset: np.ndarray = field(default=None)  # (N, S) fixed offset from §5.3

    def Q(self) -> np.ndarray:
        """Process covariance.  `Lambda Lambda'` is where cross-stat correlation
        lives -- the thing the old joint residual bootstrap was approximating."""
        return self.Lam @ self.Lam.T + np.diag(self.Psi)

    def stationary_dispersion(self) -> np.ndarray:
        """Var(theta - m) at stationarity: solves D = A D A' + Q.

        With A diagonal this is closed form, `D_ij = Q_ij / (1 - a_i a_j)`.  It
        is the right prior spread for a player's state at an arbitrary point in
        his career, which is exactly what the initial condition needs -- for
        rookies and for players whose careers began before the panel window.
        """
        Q = self.Q()
        denom = 1.0 - np.outer(self.A, self.A)
        return Q / np.maximum(denom, 1e-6)

    def copy(self, **kw) -> "Params":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Packing: natural <-> unconstrained
# ---------------------------------------------------------------------------
def pack(p: Params) -> np.ndarray:
    return np.concatenate([
        _logit((p.A - A_MIN) / (A_MAX - A_MIN)),
        p.Lam.reshape(-1),
        _inv_softplus(p.Psi),
        np.log(p.phi),
        _inv_softplus(p.acc_floor),
        [np.log(max(p.sigma_poss - SIGMA_POSS_FLOOR, 1e-6))],
        [np.log(max(p.sigma_poss_inj - SIGMA_POSS_FLOOR, 1e-6))],
        [np.log(max(p.injury_infl - 1.0, 1e-6))],
    ])


def unpack(v: np.ndarray, template: Params) -> Params:
    i = 0
    A = A_MIN + (A_MAX - A_MIN) * _sigmoid(v[i:i + S]); i += S
    Lam = v[i:i + S * Q_RANK].reshape(S, Q_RANK); i += S * Q_RANK
    Psi = _softplus(v[i:i + S]) + 1e-6; i += S
    phi = np.exp(np.clip(v[i:i + N_VOL], -8, 12)); i += N_VOL
    acc = _softplus(v[i:i + N_ACC]); i += N_ACC
    sig = SIGMA_POSS_FLOOR + float(np.exp(np.clip(v[i], -8, 3))); i += 1
    sig_inj = SIGMA_POSS_FLOOR + float(np.exp(np.clip(v[i], -8, 3))); i += 1
    infl = 1.0 + float(np.exp(np.clip(v[i], -8, 4))); i += 1
    assert i == len(v), f"packed vector length mismatch: consumed {i} of {len(v)}"
    return template.copy(A=A, Lam=Lam, Psi=Psi, phi=phi, acc_floor=acc,
                         sigma_poss=sig, sigma_poss_inj=sig_inj, injury_infl=infl)


def n_packed() -> int:
    return S + S * Q_RANK + S + N_VOL + N_ACC + 3


# ---------------------------------------------------------------------------
# Priors (§4.1).  Weakly informative, on the unconstrained scale.
# ---------------------------------------------------------------------------
def log_prior(p: Params) -> float:
    """Half-Normal(0,1) on scales, Normal(0,1) on the Q loadings, Gamma(2,0.1)
    on the overdispersions.  Deliberately weak: with 14k player-seasons the data
    dominate, and the priors are here to keep the optimiser off the boundary and
    the sampler out of the corners, not to express belief."""
    lp = 0.0
    lp += float(np.sum(-0.5 * (p.Lam / 1.0) ** 2))
    lp += float(np.sum(-0.5 * (np.sqrt(p.Psi) / 1.0) ** 2))
    lp += float(np.sum(-0.5 * (p.sigma_poss / 1.0) ** 2))
    lp += float(-0.5 * (p.sigma_poss_inj / 2.0) ** 2)
    # The injury inflation is a multiplier and must exceed 1: an injury season
    # cannot be a *better* measurement of talent than a healthy one.
    # Parameterized as 1 + exp(.), so it cannot go below 1 at all.  A soft
    # penalty was not enough: the likelihood fitted 0.945, i.e. an injury season
    # as a *better* measurement of talent than a healthy one, which is not a
    # thing.  What the data are really saying is that the other thirteen
    # dimensions need no extra inflation, because their count-based
    # `R ~ 1/count` already handles a short season -- so this sits at its bound
    # and only availability gets the separate treatment it needs.
    lp += float(-0.5 * (np.log(max(p.injury_infl - 1.0, 1e-6)) / 1.0) ** 2)
    # Gamma(2, 0.1): mean 20, generous right tail.  Keeps phi from running to
    # infinity, which would quietly delete the observation-noise floor.
    lp += float(np.sum(1.0 * np.log(p.phi) - 0.1 * p.phi))
    lp += float(np.sum(-0.5 * (np.sqrt(p.acc_floor) / 0.3) ** 2))
    # Mild pull of persistence toward the interior.
    lp += float(np.sum(-0.5 * ((_logit((p.A - A_MIN) / (A_MAX - A_MIN))) / 3.0) ** 2))
    return lp


def log_jacobian(v: np.ndarray) -> float:
    """Log |d natural / d unconstrained|, so the sampler targets the right
    density on the unconstrained scale."""
    i = 0
    lj = 0.0
    a = v[i:i + S]; i += S
    lj += float(np.sum(np.log(A_MAX - A_MIN) - a - 2 * np.logaddexp(0, -a)))
    i += S * Q_RANK                                     # identity
    psi = v[i:i + S]; i += S
    lj += float(np.sum(-np.logaddexp(0, -psi)))         # d softplus
    phi = v[i:i + N_VOL]; i += N_VOL
    lj += float(np.sum(phi))                            # d exp
    acc = v[i:i + N_ACC]; i += N_ACC
    lj += float(np.sum(-np.logaddexp(0, -acc)))
    lj += float(v[i]); i += 1                           # sigma_poss, d exp
    lj += float(v[i]); i += 1                           # sigma_poss_inj
    lj += float(v[i])                                   # injury_infl
    return lj


def default_params(phi0: np.ndarray | None = None, p_x: int = 1) -> Params:
    """A starting point that filters without blowing up.

    `A = 0.85` and modest `Q` say "a player's rate this year is mostly last
    year's" without asserting how much of the residual is real movement -- the
    fit decides that.
    """
    rng = np.random.default_rng(0)
    return Params(
        A=np.full(S, 0.85),
        Lam=0.02 * rng.standard_normal((S, Q_RANK)),
        Psi=np.full(S, 0.01),
        phi=np.full(N_VOL, 20.0) if phi0 is None else np.asarray(phi0, float),
        acc_floor=np.full(N_ACC, 0.005),
        sigma_poss=0.30,
        sigma_poss_inj=1.20,
        injury_infl=1.2,
        beta=np.zeros((p_x, S)),
        Sigma_p=np.eye(S) * 0.25,
    )
