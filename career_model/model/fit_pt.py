"""Fit the permanent/transient model (§3.4 v2), same three-way split as v1.

  * the drift `delta(age)` and debut offset are profiled exactly
    (`state_space_pt.profile_drift_diag`);
  * `beta` and `Sigma_ell0` -- the prior on the initial level `ell_0` -- come
    from EM;
  * only `Phi`, `Q_ell`, `Q_u` and the observation-noise block need numerical
    optimisation, on the fast diagonal filter first, then the low-rank `Lambda`
    blocks on the full filter.

The one genuinely harder thing than v1 is separating `Q_ell` (permanent) from
`Q_u` (transient).  They both add year-over-year variance and are distinguished
only by autocorrelation, so the likelihood is nearly flat along the ridge that
trades them off.  Two things keep the fit on the right part of that ridge: the
prior in `hierarchy_pt.log_prior` mildly favours the transient explanation, and
the fit is seeded from a method-of-moments split of the observed first-difference
autocorrelation (`_moment_init`), so it starts near the answer rather than at a
corner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..config import POSITION_GROUPS, S
from . import hierarchy_pt as hpt
from . import observations as obs
from . import state_space_pt as sspt
from .hierarchy_pt import ParamsPT

BETA_PRIOR_SD = 5.0


@dataclass
class FitPT:
    params: ParamsPT
    c_init: np.ndarray                 # (N, S) evaluated debut offset
    c_init_coefs: np.ndarray           # (nbi, S) debut spline, for rookies
    delta_league: np.ndarray           # (nb, S) drift league curve
    delta_pos: np.ndarray              # (G, nb, S) drift position deviation
    loglik: float
    history: list


def _log(msg):
    print(f"[fit-pt] {msg}", flush=True)


def drift_offsets(ds, c_league, c_pos):
    """`delta(age)` per player-season -- the random-walk drift added to `ell`."""
    phi = ds.age_basis(ds.grid.age)
    coefs = c_league[None] + c_pos[ds.pos_idx]
    return np.einsum("ntk,nks->nts", phi, coefs)


def init_offset(ds, c_lin):
    return ds.init_basis(ds.grid.age[:, 0]) @ c_lin


# ---------------------------------------------------------------------------
# Method-of-moments seed for the permanent/transient variance split
# ---------------------------------------------------------------------------
def _moment_init(ds, R):
    """Seed Phi, Q_ell, Q_u per stat from first-difference autocorrelation.

    For `z_t = ell_t + u_t + e_t` with `ell` a random walk and `u` an AR(1),
    the first difference `dz_t = z_t - z_{t-1}` has a lag-1 autocorrelation that
    is negative and set by the transient share: a pure random walk gives ~0, a
    pure AR/noise gives ~-0.5.  We read the total first-difference variance and
    its lag-1 autocorrelation per stat and split them into a permanent piece
    (`q_ell`) and a transient piece, netting out the known observation variance.
    Rough, but it puts the optimiser on the right side of the ridge.
    """
    grid = ds.grid
    q_ell = np.zeros(S); q_u = np.zeros(S); phi = np.full(S, 0.5)
    for s in range(S):
        d1, d2 = [], []
        for i in range(grid.n_players):
            ob = np.flatnonzero(grid.obs_mask[i, :, s] & grid.observed[i])
            if len(ob) < 3:
                continue
            z = grid.z[i, ob, s]
            dz = np.diff(z)
            d1.extend(dz[:-1]); d2.extend(dz[1:])
        d1 = np.array(d1); d2 = np.array(d2)
        if len(d1) < 50:
            q_ell[s] = 0.02; q_u[s] = 0.05; continue
        var_dz = np.var(np.concatenate([d1, d2]))
        ac1 = np.corrcoef(d1, d2)[0, 1] if len(d1) > 2 else -0.2
        # Var(dz) = q_ell + 2(q_u + var_e)(1-phi... ) -- use a simple mapping:
        # permanent share rises as ac1 -> 0, transient as ac1 -> -0.5.
        perm_share = np.clip(1.0 + 2.0 * ac1, 0.05, 0.95)   # ac1=0 ->1, -0.5 ->0
        # net out observation variance (median R for this stat)
        med_R = np.median(R[grid.obs_mask[:, :, s], s]) if grid.obs_mask[:, :, s].any() else 0.05
        signal = max(var_dz / 2.0 - med_R, 1e-3)
        q_ell[s] = max(perm_share * signal, 1e-3)
        q_u[s] = max((1 - perm_share) * signal, 1e-3)
    return phi, q_ell, q_u


# ---------------------------------------------------------------------------
# EM for the initial-level prior
# ---------------------------------------------------------------------------
def update_prior_pt(ds, params, R, drift, c_init):
    """M-step for `beta` and `Sigma_ell0`.

    Targets the filtered level right after a player's debut season,
    `E[ell_0 | z_0]` -- the closest data-informed estimate of where he entered
    the league, which is what the prior must predict for a rookie.  A smoother
    would use later seasons too, but for the *rookie prior* the debut estimate
    is the relevant one, and this avoids a backward pass.
    """
    res = sspt.run_filter_pt_diag(ds.grid, params, ds.X, R, drift,
                                  ds.init_basis, c_init, keep_states=True)
    ell0 = res.ell[:, 0, :]                       # E[ell_0 | z_0], (N, S)
    P0 = res.Pee[:, 0, :]                         # its variance, (N, S)
    if params.gbm_offset is not None:
        ell0 = ell0 - params.gbm_offset
    X = ds.X
    penalty = np.eye(X.shape[1]) / BETA_PRIOR_SD ** 2
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ ell0)
    resid = ell0 - X @ beta
    Sigma = (resid.T @ resid) / len(X) + np.diag(P0.mean(axis=0))
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-5 * np.eye(S)
    return params.copy(beta=beta, Sigma_ell0=Sigma)


# ---------------------------------------------------------------------------
# Objective and optimiser
# ---------------------------------------------------------------------------
def _objective(v, ds, template, drift, c_init, diag):
    p = hpt.unpack(v, template)
    if not np.all(np.isfinite(v)):
        return 1e12
    try:
        R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss,
                          p.sigma_poss_inj, p.injury_infl)
        runner = sspt.run_filter_pt_diag if diag else sspt.run_filter_pt
        res = runner(ds.grid, p, ds.X, R, drift, ds.init_basis, c_init)
    except np.linalg.LinAlgError:
        return 1e12
    val = res.loglik + hpt.log_prior(p)
    return -val if np.isfinite(val) else 1e12


def _optimize(v0, ds, template, drift, c_init, diag, free, maxiter):
    v0 = np.asarray(v0, float)
    idx = np.flatnonzero(free)

    def f(sub):
        v = v0.copy(); v[idx] = sub
        return _objective(v, ds, template, drift, c_init, diag)

    res = minimize(f, v0[idx], method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 20, "eps": 1e-4, "ftol": 1e-10})
    out = v0.copy(); out[idx] = res.x
    return out, -res.fun


def _free_mask(blocks):
    k = hpt.Q_RANK
    spans = [("Phi", S), ("Lam_ell", S * k), ("Psi_ell", S), ("Lam_u", S * k),
             ("Psi_u", S), ("phi", hpt.N_VOL), ("acc_floor", hpt.N_ACC),
             ("sigma_poss", 1), ("sigma_poss_inj", 1), ("injury_infl", 1)]
    mask, i = np.zeros(hpt.n_packed(), dtype=bool), 0
    for name, w in spans:
        if name in blocks:
            mask[i:i + w] = True
        i += w
    return mask


# ---------------------------------------------------------------------------
def fit(ds, n_outer=3, maxiter_diag=200, maxiter_full=60, verbose=True) -> FitPT:
    nb, nbi, G = ds.age_basis.size, ds.init_basis.size, len(POSITION_GROUPS)
    c_lin = np.zeros((nbi, S)); c_init = np.zeros((ds.n_players, S))
    c_league = np.zeros((nb, S)); c_pos = np.zeros((G, nb, S))

    params = hpt.default_params(obs.default_phi(ds.panel), p_x=ds.X.shape[1])
    params = params.copy(Lam_ell=np.zeros((S, hpt.Q_RANK)), Lam_u=np.zeros((S, hpt.Q_RANK)))
    R0 = obs.refresh_R(ds.grid, params.phi, params.acc_floor, params.sigma_poss,
                       params.sigma_poss_inj, params.injury_infl)
    phi0, qell0, qu0 = _moment_init(ds, R0)
    params = params.copy(Phi=phi0, Psi_ell=qell0, Psi_u=qu0)
    if verbose:
        _log(f"moment seed: q_ell median {np.median(qell0):.3f}, q_u median {np.median(qu0):.3f}")
    history = []

    diag_blocks = _free_mask({"Phi", "Psi_ell", "Psi_u", "phi", "acc_floor",
                              "sigma_poss", "sigma_poss_inj", "injury_infl"})
    for it in range(n_outer):
        t0 = time.time()
        R = obs.refresh_R(ds.grid, params.phi, params.acc_floor, params.sigma_poss,
                          params.sigma_poss_inj, params.injury_infl)
        c_lin, c_league, c_pos = sspt.profile_drift_diag(
            ds.grid, params, ds.X, R, ds.age_basis, ds.init_basis, ds.pos_idx,
            c_league, c_pos, c_lin)
        c_init = init_offset(ds, c_lin)
        drift = drift_offsets(ds, c_league, c_pos)
        params = update_prior_pt(ds, params, R, drift, c_init)
        v = hpt.pack(params)
        v, ll = _optimize(v, ds, params, drift, c_init, True, diag_blocks, maxiter_diag)
        params = hpt.unpack(v, params)
        history.append({"stage": f"diag-{it}", "loglik": ll})
        if verbose:
            _log(f"outer {it}: diag loglik {ll:,.0f}  ({time.time() - t0:.1f}s)")

    drift = drift_offsets(ds, c_league, c_pos)
    rng = np.random.default_rng(1)
    params = params.copy(Lam_ell=1e-3 * rng.standard_normal((S, hpt.Q_RANK)),
                         Lam_u=1e-3 * rng.standard_normal((S, hpt.Q_RANK)))
    v = hpt.pack(params)
    t0 = time.time()
    v, ll = _optimize(v, ds, params, drift, c_init, False,
                      _free_mask({"Lam_ell", "Lam_u", "Psi_ell", "Psi_u"}), maxiter_full)
    params = hpt.unpack(v, params)
    history.append({"stage": "full-Q", "loglik": ll})
    if verbose:
        _log(f"full-Q loglik {ll:,.0f}  ({time.time() - t0:.1f}s)")

    R = obs.refresh_R(ds.grid, params.phi, params.acc_floor, params.sigma_poss,
                      params.sigma_poss_inj, params.injury_infl)
    c_lin, c_league, c_pos = sspt.profile_drift_diag(
        ds.grid, params, ds.X, R, ds.age_basis, ds.init_basis, ds.pos_idx,
        c_league, c_pos, c_lin)
    c_init = init_offset(ds, c_lin)
    drift = drift_offsets(ds, c_league, c_pos)
    params = update_prior_pt(ds, params, R, drift, c_init)
    final = sspt.run_filter_pt(ds.grid, params, ds.X, R, drift, ds.init_basis, c_init)
    if verbose:
        _log(f"final loglik {final.loglik:,.0f}")
    return FitPT(params=params, c_init=c_init, c_init_coefs=c_lin,
                 delta_league=c_league, delta_pos=c_pos,
                 loglik=final.loglik, history=history)
