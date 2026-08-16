"""Parameters for the permanent/transient state-space model (§3.4 v2).

The v1 model reverts `theta` toward a fixed player mean `m_i`.  That mean is
under-identified: an AR(1) mean gets under one effective observation even over a
long career (see the README's star-decline section), so a late-drafted star's
`m_i` stays pinned near a wrong draft-based prior and the projection reverts him
toward it.  The permanent/transient split removes the fixed mean entirely:

    theta_{i,t} = ell_{i,t} + u_{i,t}
    ell_{i,t+1} = ell_{i,t} + delta(age_{i,t}) + eta^ell     (random walk)
    u_{i,t+1}   = Phi u_{i,t} + eta^u                        (AR(1), reverts to 0)

`ell` is a random walk, so the filter *tracks* it from the data rather than
estimating a fixed target -- there is no mean to be under-identified.  Real
development and decline accumulate into `ell` and persist; fluke seasons, hot
shooting and role noise go into `u` and wash out.  The observation sums them,
`z_t = ell_t + u_t + noise`, so the observation matrix is `Z = [I, I]` -- the
one structural difference from v1's `Z = [I, 0]`.

Parameters, per stat `s`:
  * `Phi`      -- transient AR persistence, diagonal, in (0,1).
  * `Q_ell`    -- permanent innovation covariance.  How much a player's true
                  level can move year to year.  This is the lever that lets a
                  durable star's `ell` deviate from the population aging drift.
  * `Q_u`      -- transient innovation covariance.
  * `Sigma_ell0` -- spread of the initial level `ell_0` around its prior mean
                  `beta'x_i + f_GBM(x_i) + c_init(age_0)`.

`Q_ell` and `Q_u` both add to year-over-year variance; they are distinguished by
autocorrelation -- permanent changes persist, transient ones revert -- which is
exactly the lag-1 autocorrelation of first differences (−0.18 for possessions).
Both use the low-rank `Lambda Lambda' + Psi` structure of §3.4 so cross-stat
correlation is captured without 105 free covariance entries apiece.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config import ACCURACY_STATS, S, VOLUME_STATS
from .hierarchy import (  # reuse the observation-noise machinery unchanged
    SIGMA_POSS_FLOOR, _inv_softplus, _sigmoid, _softplus, _logit,
)

N_VOL = len(VOLUME_STATS)
N_ACC = len(ACCURACY_STATS)
Q_RANK = 3
# Transient persistence, capped well below 1.  The likelihood otherwise slides
# to a degenerate corner -- Phi -> 0.95, Q_ell -> 0 -- where the "transient"
# becomes a near-random-walk that impersonates the permanent level, `ell`
# freezes at its prior, and the model collapses back to v1's single AR process.
# The spec means `u` as genuinely transient (shooting variance, role noise that
# washes out in a season or two), so a half-life over ~1 year is out of scope.
# Capping at 0.5 forces the clean permanent/transient decomposition.
PHI_MIN, PHI_MAX = 0.02, 0.50


@dataclass
class ParamsPT:
    """Permanent/transient hyperparameters, in natural (constrained) units."""

    Phi: np.ndarray                  # (S,)   transient AR persistence, in (0,1)
    Lam_ell: np.ndarray              # (S, k) permanent innovation loadings
    Psi_ell: np.ndarray              # (S,)   permanent idiosyncratic variance
    Lam_u: np.ndarray                # (S, k) transient innovation loadings
    Psi_u: np.ndarray                # (S,)   transient idiosyncratic variance
    phi: np.ndarray                  # (N_VOL,) NB overdispersion (obs layer)
    acc_floor: np.ndarray            # (N_ACC,) accuracy obs-variance floor
    sigma_poss: float                # availability obs sd, healthy seasons
    sigma_poss_inj: float = 0.9      # availability obs sd, injury seasons
    injury_infl: float = 1.5         # R multiplier on other dims, injury seasons
    beta: np.ndarray = field(default=None)        # (p_x, S) prior-mean coefs
    Sigma_ell0: np.ndarray = field(default=None)  # (S, S) initial-level covariance
    gbm_offset: np.ndarray = field(default=None)  # (N, S) fixed offset (§5.3)

    def Q_ell(self) -> np.ndarray:
        return self.Lam_ell @ self.Lam_ell.T + np.diag(self.Psi_ell)

    def Q_u(self) -> np.ndarray:
        return self.Lam_u @ self.Lam_u.T + np.diag(self.Psi_u)

    def transient_stationary(self) -> np.ndarray:
        """Var(u) at stationarity: solves D = Phi D Phi' + Q_u.  With Phi
        diagonal, `D_ij = Q_u_ij / (1 - phi_i phi_j)`.  This is the initial
        spread of the transient component and its steady-state size."""
        Qu = self.Q_u()
        denom = 1.0 - np.outer(self.Phi, self.Phi)
        return Qu / np.maximum(denom, 1e-6)

    def copy(self, **kw) -> "ParamsPT":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Packing: natural <-> unconstrained (for the L-BFGS / Metropolis fit)
# ---------------------------------------------------------------------------
def pack(p: ParamsPT) -> np.ndarray:
    return np.concatenate([
        _logit((p.Phi - PHI_MIN) / (PHI_MAX - PHI_MIN)),
        p.Lam_ell.reshape(-1),
        _inv_softplus(p.Psi_ell),
        p.Lam_u.reshape(-1),
        _inv_softplus(p.Psi_u),
        np.log(p.phi),
        _inv_softplus(p.acc_floor),
        [np.log(max(p.sigma_poss - SIGMA_POSS_FLOOR, 1e-6))],
        [np.log(max(p.sigma_poss_inj - SIGMA_POSS_FLOOR, 1e-6))],
        [np.log(max(p.injury_infl - 1.0, 1e-6))],
    ])


def unpack(v: np.ndarray, template: ParamsPT) -> ParamsPT:
    i = 0
    Phi = PHI_MIN + (PHI_MAX - PHI_MIN) * _sigmoid(v[i:i + S]); i += S
    Lam_ell = v[i:i + S * Q_RANK].reshape(S, Q_RANK); i += S * Q_RANK
    Psi_ell = _softplus(v[i:i + S]) + 1e-7; i += S
    Lam_u = v[i:i + S * Q_RANK].reshape(S, Q_RANK); i += S * Q_RANK
    Psi_u = _softplus(v[i:i + S]) + 1e-7; i += S
    phi = np.exp(np.clip(v[i:i + N_VOL], -8, 12)); i += N_VOL
    acc = _softplus(v[i:i + N_ACC]); i += N_ACC
    sig = SIGMA_POSS_FLOOR + float(np.exp(np.clip(v[i], -8, 3))); i += 1
    sig_inj = SIGMA_POSS_FLOOR + float(np.exp(np.clip(v[i], -8, 3))); i += 1
    infl = 1.0 + float(np.exp(np.clip(v[i], -8, 4))); i += 1
    assert i == len(v), f"packed length mismatch: consumed {i} of {len(v)}"
    return template.copy(Phi=Phi, Lam_ell=Lam_ell, Psi_ell=Psi_ell,
                         Lam_u=Lam_u, Psi_u=Psi_u, phi=phi, acc_floor=acc,
                         sigma_poss=sig, sigma_poss_inj=sig_inj, injury_infl=infl)


def n_packed() -> int:
    return S + (S * Q_RANK + S) * 2 + N_VOL + N_ACC + 3


def log_prior(p: ParamsPT) -> float:
    """Weakly informative, matching v1's philosophy.  One deliberate addition:
    a mild penalty pulling `Q_ell` small relative to `Q_u`.  Permanent and
    transient variance are only separated by autocorrelation, a second-order
    feature, so the likelihood is flat along the ridge that trades them off;
    without a nudge the fit can dump everything into the permanent component and
    make every season a level change.  The prior says: prefer the transient
    explanation unless the data insist on the permanent one."""
    lp = 0.0
    # Symmetric on permanent and transient: with Phi capped the degenerate
    # corner is unreachable, so there is no need to bias the split, and biasing
    # it is what craters durable old stars (their sustained excess is forced
    # into the fast-decaying transient instead of the permanent level).  Let the
    # first-difference autocorrelation in the data set the ratio.
    lp += float(np.sum(-0.5 * (p.Lam_ell / 1.0) ** 2))
    lp += float(np.sum(-0.5 * (np.sqrt(p.Psi_ell) / 1.0) ** 2))
    lp += float(np.sum(-0.5 * (p.Lam_u / 1.0) ** 2))
    lp += float(np.sum(-0.5 * (np.sqrt(p.Psi_u) / 1.0) ** 2))
    lp += float(np.sum(1.0 * np.log(p.phi) - 0.1 * p.phi))
    lp += float(np.sum(-0.5 * (np.sqrt(p.acc_floor) / 0.3) ** 2))
    lp += float(np.sum(-0.5 * (p.sigma_poss / 1.0) ** 2))
    lp += float(-0.5 * (p.sigma_poss_inj / 2.0) ** 2)
    lp += float(-0.5 * (np.log(max(p.injury_infl - 1.0, 1e-6)) / 1.0) ** 2)
    lp += float(np.sum(-0.5 * (_logit((p.Phi - PHI_MIN) / (PHI_MAX - PHI_MIN)) / 3.0) ** 2))
    return lp


def default_params(phi0: np.ndarray | None = None, p_x: int = 1) -> ParamsPT:
    rng = np.random.default_rng(0)
    return ParamsPT(
        Phi=np.full(S, 0.5),
        Lam_ell=0.02 * rng.standard_normal((S, Q_RANK)),
        Psi_ell=np.full(S, 0.02),
        Lam_u=0.02 * rng.standard_normal((S, Q_RANK)),
        Psi_u=np.full(S, 0.05),
        phi=np.full(N_VOL, 20.0) if phi0 is None else np.asarray(phi0, float),
        acc_floor=np.full(N_ACC, 0.005),
        sigma_poss=0.30, sigma_poss_inj=1.20, injury_infl=1.2,
        beta=np.zeros((p_x, S)),
        Sigma_ell0=np.eye(S) * 0.4,
    )
