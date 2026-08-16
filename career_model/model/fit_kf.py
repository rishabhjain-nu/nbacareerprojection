"""v1 inference: MCMC over the Kalman marginal likelihood (§5.2).

The estimation problem splits three ways, and the split is what makes it run in
minutes instead of days.

**Linear terms are profiled exactly.**  The aging coefficients and the debut-age
offset enter the state mean linearly, so the marginal likelihood is an exact
quadratic in them and `profile_linear_diag` jumps straight to the optimum.  No
iteration, no smoother, no attenuation.

**`beta` and `Sigma_player` come from EM.**  `m_i` is part of the augmented
state, so the filter already returns `E[m_i | data]` and `Var(m_i | data)` --
that is the E-step, free.  The M-step is a ridge regression of the posterior
means on `x_i` plus the usual variance correction.  Exact EM, two lines.

**Only the noise and persistence hyperparameters need numerical optimisation**
-- `A`, `Psi`, `Lambda`, `phi_s`, the accuracy floors, `sigma_poss`.  Those are
fitted by L-BFGS on the diagonal filter first (milliseconds per evaluation),
then the `Lambda` block that carries cross-stat correlation is fitted on the
full filter.

`sample_posterior` then runs adaptive Metropolis around the mode so that step 1
of §6 -- parameter uncertainty -- draws from something real rather than being
pinned at a point estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..config import POSITION_GROUPS, S
from .hierarchy import N_VOL
from . import hierarchy as hier
from . import observations as obs
from . import state_space as ss
from .hierarchy import Params

BETA_PRIOR_SD = 5.0          # §4.1, on standardized covariates


@dataclass
class Fit:
    params: Params
    c_init: np.ndarray                 # (n_basis_init, S) debut-age offset
    delta_league: np.ndarray           # (n_basis, S)
    delta_pos: np.ndarray              # (G, n_basis, S)
    loglik: float
    history: list
    player_scale: np.ndarray = None    # (N,) per-player rate-of-aging multiplier
    c_init_coefs: np.ndarray = None    # (n_basis_init, S) debut-age spline, for rookies


# Parameterization of the aging curve, chosen once here.
#
# LEVEL form -- `theta = m + c(age)` with `m` the player's level -- identifies
# `m_i` directly and was tried to fix the star-decline problem.  It failed: the
# only version of it that respects the within-player-aging rule (§1.5) leaves
# `m_i` even less identified than the increment form (the filter's likelihood
# cannot pin it down from noisy AR realizations -- flattening the prior 10x
# barely moves it), and it drove the persistence `A` down, making mean reversion
# *stronger*.  So the base model is the INCREMENT form of §3.4, which is what was
# validated, and the star-decline problem is fixed at the projection layer by an
# empirical-Bayes reversion target (see `simulate.project`).
LEVEL_PARAM = False


def _offset_A(params):
    """The `A` passed to `offsets_from`: the persistence in level mode, `None`
    (increment mode) otherwise.  One switch, so the whole fit stays consistent."""
    return params.A if LEVEL_PARAM else None


def _log(msg: str) -> None:
    print(f"[fit] {msg}", flush=True)


def level_coefs(ds, c_league, c_pos, player_scale=None):
    """(N, nb, S) level-curve coefficients, one set per player."""
    coefs = c_league[None] + c_pos[ds.pos_idx]
    if player_scale is not None:
        coefs = coefs * np.asarray(player_scale)[:, None, None]
    return coefs


def offsets_from(ds, c_league, c_pos, player_scale=None, A=None) -> np.ndarray:
    """Transition offset in the level parameterization.

    `theta_{t+1} = m_i + c(age_{t+1}) + A(theta_t - m_i - c(age_t))` gives an
    offset of `c(age_{t+1}) - A c(age_t)`.  Passing `A=None` falls back to the
    spec's increment form, `delta(age_t)`, which the tests still exercise.
    """
    coefs = level_coefs(ds, c_league, c_pos, player_scale)
    phi = ds.age_basis(ds.grid.age)
    cur = np.einsum("ntk,nks->nts", phi, coefs)
    if A is None:
        return cur
    nxt = np.einsum("ntk,nks->nts", ds.age_basis(ds.grid.age + 1.0), coefs)
    return nxt - np.asarray(A)[None, None, :] * cur


def init_offset_from(ds, c_lin) -> np.ndarray:
    """Debut-age offset, from its **own** spline (§1.5).

    Deliberately not `c(age_0)` off the level curve: debut age is confounded
    with player quality, so letting the aging curve place the initial state
    identifies it cross-sectionally instead of within-player.
    """
    return ds.init_basis(ds.grid.age[:, 0]) @ c_lin


# ---------------------------------------------------------------------------
# EM step for the player level of the hierarchy
# ---------------------------------------------------------------------------
def update_prior(ds, params: Params, R, offsets, c_init) -> Params:
    """M-step for `beta` and `Sigma_player` (§4.1).

    `m_i` is a constant component of the augmented state, so the filtered value
    after a player's last season already conditions on his whole career -- it is
    the smoothed estimate, and no backward pass is needed to get it.
    """
    res = ss.run_filter(ds.grid, params, ds.X, R, offsets,
                        ds.init_basis, c_init, keep_states=True)
    T = ds.grid.z.shape[1]
    m_hat = res.x2[:, T, :]                                 # (N, S)
    V_m = res.P22[:, T]                                     # (N, S, S)
    if params.gbm_offset is not None:
        m_hat = m_hat - params.gbm_offset

    X = ds.X
    XtX = X.T @ X
    penalty = np.eye(X.shape[1]) / BETA_PRIOR_SD ** 2
    penalty[0, 0] = 0.0                                     # never shrink the intercept
    beta = np.linalg.solve(XtX + penalty, X.T @ m_hat)

    resid = m_hat - X @ beta
    Sigma = (resid.T @ resid + V_m.sum(axis=0)) / len(X)
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-6 * np.eye(S)
    return params.copy(beta=beta, Sigma_p=Sigma)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def _objective(v, ds, template, offsets, c_init, diag: bool):
    p = hier.unpack(v, template)
    if not np.all(np.isfinite(v)):
        return 1e12
    try:
        R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss,
                              p.sigma_poss_inj, p.injury_infl)
        runner = ss.run_filter_diag if diag else ss.run_filter
        res = runner(ds.grid, p, ds.X, R, offsets, ds.init_basis, c_init)
    except np.linalg.LinAlgError:
        return 1e12
    val = res.loglik + hier.log_prior(p)
    if not np.isfinite(val):
        return 1e12
    return -val


def _optimize(v0, ds, template, offsets, c_init, diag: bool, free: np.ndarray,
              maxiter: int):
    """L-BFGS over the free coordinates only, others held at `v0`."""
    v0 = np.asarray(v0, float)
    idx = np.flatnonzero(free)

    def f(sub):
        v = v0.copy()
        v[idx] = sub
        return _objective(v, ds, template, offsets, c_init, diag)

    res = minimize(f, v0[idx], method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 20, "eps": 1e-4, "ftol": 1e-10})
    out = v0.copy()
    out[idx] = res.x
    return out, -res.fun


def _free_mask(blocks: set[str]) -> np.ndarray:
    """Boolean mask over the packed vector selecting named parameter blocks."""
    n_vol, n_acc, k = hier.N_VOL, hier.N_ACC, hier.Q_RANK
    spans = [("A", S), ("Lam", S * k), ("Psi", S), ("phi", n_vol),
             ("acc_floor", n_acc), ("sigma_poss", 1), ("sigma_poss_inj", 1),
             ("injury_infl", 1)]
    mask, i = np.zeros(hier.n_packed(), dtype=bool), 0
    for name, width in spans:
        if name in blocks:
            mask[i:i + width] = True
        i += width
    return mask


# ---------------------------------------------------------------------------
# Staged fit
# ---------------------------------------------------------------------------
def fit(ds, n_outer: int = 3, maxiter_diag: int = 200, maxiter_full: int = 60,
        verbose: bool = True) -> Fit:
    """Run the M1 -> M3 ladder and return the mode."""
    nb, nbi, G = ds.age_basis.size, ds.init_basis.size, len(POSITION_GROUPS)
    c_lin = np.zeros((nbi, S))          # profile block, unused in level form
    c_init = np.zeros((ds.n_players, S))  # evaluated per-player init offset
    delta_league = np.zeros((nb, S))
    delta_pos = np.zeros((G, nb, S))
    player_scale = np.ones(ds.n_players)

    params = hier.default_params(obs.default_phi(ds.panel), p_x=ds.X.shape[1])
    params = params.copy(Lam=np.zeros((S, hier.Q_RANK)))
    history = []

    v = hier.pack(params)
    diag_blocks = _free_mask({"A", "Psi", "phi", "acc_floor", "sigma_poss",
                              "sigma_poss_inj", "injury_infl"})

    for it in range(n_outer):
        t0 = time.time()
        offsets = offsets_from(ds, delta_league, delta_pos, None, _offset_A(params))
        R = obs.refresh_R(ds.grid, params.phi, params.acc_floor, params.sigma_poss,
                          params.sigma_poss_inj, params.injury_infl)

        # --- 1. exact profile for the linear terms (M4) --------------------
        c_lin, delta_league, delta_pos = ss.profile_linear_diag(
            ds.grid, params, ds.X, R, ds.age_basis, ds.init_basis, ds.pos_idx,
            delta_league, delta_pos, c_lin, level_param=LEVEL_PARAM)
        c_init = init_offset_from(ds, c_lin)
        offsets = offsets_from(ds, delta_league, delta_pos, None, _offset_A(params))

        # --- 2. player-level aging rate (§4.1, third level) ----------------
        # The per-player aging rate is *collinear with `m_i`* in the level
        # parameterization: over an eight-season window `s_i * c(age)` is close
        # to linear in age, so it trades off against the level intercept almost
        # freely.  Fitted jointly it collapsed to 0.64 for long-career players
        # and pushed `m_i` to physically impossible values (Durant at 9,600
        # possessions).  The level form already fixes what the scalar was added
        # for, so it is held at the league rate here.
        player_scale = np.ones(ds.n_players)
        c_init = init_offset_from(ds, c_lin)
        offsets = offsets_from(ds, delta_league, delta_pos, player_scale, _offset_A(params))

        # --- 3. EM for the player level (M2) -------------------------------
        params = update_prior(ds, params, R, offsets, c_init)

        # --- 3. noise and persistence, diagonal filter ---------------------
        v = hier.pack(params)
        v, ll = _optimize(v, ds, params, offsets, c_init, True, diag_blocks, maxiter_diag)
        params = hier.unpack(v, params)
        history.append({"stage": f"diag-{it}", "loglik": ll})
        if verbose:
            _log(f"outer {it}: diag loglik {ll:,.0f}  ({time.time() - t0:.1f}s)")

    # --- 4. cross-stat correlation on the full filter (M3) -----------------
    offsets = offsets_from(ds, delta_league, delta_pos, player_scale, _offset_A(params))
    params = params.copy(Lam=1e-3 * np.random.default_rng(1).standard_normal((S, hier.Q_RANK)))
    v = hier.pack(params)
    t0 = time.time()
    v, ll = _optimize(v, ds, params, offsets, c_init, False,
                      _free_mask({"Lam", "Psi"}), maxiter_full)
    params = hier.unpack(v, params)
    if verbose:
        _log(f"full-Q loglik {ll:,.0f}  ({time.time() - t0:.1f}s)")
    history.append({"stage": "full-Q", "loglik": ll})

    # Re-profile and re-EM under the fitted Q, now on the full filter.
    R = obs.refresh_R(ds.grid, params.phi, params.acc_floor, params.sigma_poss,
                          params.sigma_poss_inj, params.injury_infl)
    c_lin, delta_league, delta_pos = ss.profile_linear_diag(
        ds.grid, params, ds.X, R, ds.age_basis, ds.init_basis, ds.pos_idx,
        delta_league, delta_pos, c_lin, level_param=LEVEL_PARAM)
    c_init = init_offset_from(ds, c_lin)
    offsets = offsets_from(ds, delta_league, delta_pos, None, _offset_A(params))
    player_scale = np.ones(ds.n_players)
    c_init = init_offset_from(ds, c_lin)
    offsets = offsets_from(ds, delta_league, delta_pos, player_scale, _offset_A(params))
    params = update_prior(ds, params, R, offsets, c_init)

    final = ss.run_filter(ds.grid, params, ds.X, R, offsets, ds.init_basis, c_init)
    if verbose:
        _log(f"final loglik {final.loglik:,.0f}")
    if verbose:
        _log(f"player aging scale: median {np.median(player_scale):.2f}, "
             f"10-90% {np.percentile(player_scale, 10):.2f}-"
             f"{np.percentile(player_scale, 90):.2f}")
    return Fit(params=params, c_init=c_init, delta_league=delta_league,
               delta_pos=delta_pos, loglik=final.loglik, history=history,
               player_scale=player_scale)


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------
def sample_posterior(ds, fit_result: Fit, n_draws: int = 400, burn: int = 200,
                     seed: int = 0, verbose: bool = True) -> np.ndarray:
    """Adaptive Metropolis over the KF marginal likelihood.

    Returns draws of the *packed* hyperparameter vector.  These feed step 1 of
    §6: without them the simulation would propagate state and sampling
    uncertainty but treat `Q`, `A` and the overdispersions as known, which
    understates the horizon-5 bands in exactly the way the architecture exists
    to avoid.

    The proposal is seeded from a finite-difference curvature estimate at the
    mode and then adapted on the running covariance (Haario et al.), which is
    what makes 84 correlated coordinates mix at all under a random walk.
    """
    rng = np.random.default_rng(seed)
    offsets = offsets_from(ds, fit_result.delta_league, fit_result.delta_pos,
                           fit_result.player_scale, _offset_A(fit_result.params))
    c_init = fit_result.c_init
    template = fit_result.params

    def logpost(v):
        val = -_objective(v, ds, template, offsets, c_init, diag=False)
        if val <= -1e11:
            return -np.inf
        return val + hier.log_jacobian(v)

    v0 = hier.pack(template)
    d = len(v0)
    scale = _curvature_scale(logpost, v0)
    C = np.diag(scale ** 2)
    chol = np.linalg.cholesky(C)

    cur, cur_lp = v0.copy(), logpost(v0)
    draws = np.zeros((n_draws, d))
    kept, accepted, total = 0, 0, 0
    step = 2.4 / np.sqrt(d)
    mean = v0.copy()
    cov = C.copy()

    t0 = time.time()
    n_total = burn + n_draws
    for i in range(n_total):
        prop = cur + step * (chol @ rng.standard_normal(d))
        lp = logpost(prop)
        total += 1
        if np.log(rng.random()) < lp - cur_lp:
            cur, cur_lp = prop, lp
            accepted += 1
        if i >= burn:
            draws[kept] = cur
            kept += 1
        # Adaptation: running covariance during burn-in only, so the chain is
        # a proper Markov chain over the retained draws.
        if i < burn:
            delta = cur - mean
            mean = mean + delta / (i + 1)
            cov = cov + (np.outer(delta, cur - mean) - cov) / (i + 1)
            if i > 50 and i % 25 == 0:
                reg = cov + 1e-8 * np.eye(d)
                try:
                    chol = np.linalg.cholesky(reg)
                except np.linalg.LinAlgError:
                    pass
                rate = accepted / total
                step *= np.exp((rate - 0.234) * 0.5)
        if verbose and (i + 1) % 100 == 0:
            _log(f"  MH {i + 1}/{n_total}  accept {accepted / total:.2f}  "
                 f"lp {cur_lp:,.0f}  ({time.time() - t0:.0f}s)")

    if verbose:
        _log(f"posterior: {kept} draws, acceptance {accepted / total:.2f}")
    return draws


def _curvature_scale(logpost, v0, eps: float = 1e-3) -> np.ndarray:
    """Per-coordinate sd from the diagonal of the numerical Hessian."""
    f0 = logpost(v0)
    scale = np.zeros(len(v0))
    for j in range(len(v0)):
        vp, vm = v0.copy(), v0.copy()
        vp[j] += eps
        vm[j] -= eps
        curv = (logpost(vp) - 2 * f0 + logpost(vm)) / eps ** 2
        scale[j] = 1.0 / np.sqrt(-curv) if curv < -1e-8 else 0.1
    return np.clip(scale, 1e-4, 2.0)


def rhat_ess(draws: np.ndarray, n_chains: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """R-hat and effective sample size (§5.2, M6 acceptance check).

    `draws` may be (n, d) -- one chain, split into `n_chains` pseudo-chains,
    which detects non-stationarity within a run but not multimodality across
    independent starts -- or (chains, n, d), which is the real diagnostic.
    """
    if draws.ndim == 3:
        x = draws
        n_chains, m, d = x.shape
    else:
        n, d = draws.shape
        m = n // n_chains
        if m < 4:
            return np.full(d, np.nan), np.full(d, np.nan)
        x = draws[: m * n_chains].reshape(n_chains, m, d)
    mean_c = x.mean(axis=1)
    var_c = x.var(axis=1, ddof=1)
    W = var_c.mean(axis=0)
    B = m * mean_c.var(axis=0, ddof=1) if n_chains > 1 else np.zeros(d)
    var_hat = ((m - 1) * W + B) / m
    rhat = np.sqrt(np.where(W > 0, var_hat / W, np.nan))

    ess = np.zeros(d)
    for j in range(d):
        acf, s = [], x[:, :, j] - mean_c[:, j:j + 1]
        denom = (s ** 2).mean()
        if denom <= 0:
            ess[j] = np.nan
            continue
        for lag in range(1, min(m // 2, 500)):
            r = (s[:, :-lag] * s[:, lag:]).mean() / denom
            if r < 0.05:
                break
            acf.append(r)
        ess[j] = n_chains * m / (1 + 2 * sum(acf))
    return rhat, ess


def laplace_draws(chain: np.ndarray, mode: np.ndarray, n_draws: int = 4000,
                  shrink: float = 0.10, seed: int = 0) -> np.ndarray:
    """Independent draws from a Gaussian fitted to the chain, centred at the mode.

    Why this rather than a longer chain.  A random-walk Metropolis over 84
    correlated coordinates delivers an effective sample size of roughly `n/d`,
    so buying ESS > 400 costs ~35,000 filter evaluations, and running chains in
    parallel to get there turned out to be memory-bound rather than CPU-bound
    on this panel.

    The shape of the posterior is far easier to estimate than its tail
    behaviour: with 14.5k player-seasons informing 84 hyperparameters, the
    marginal posterior is close to Gaussian near the mode (Bernstein-von Mises),
    and the sample covariance of 3,000 *correlated* draws is a consistent
    estimator of it even when the ESS for any single coordinate is 40.  The mode
    itself comes from the optimiser exactly, so the location does not depend on
    the chain mixing at all.

    Draws from `N(mode, Sigma_hat)` are then **independent by construction** --
    ESS equals the number of draws for every function of the parameters, which
    is strictly better for §6 step 1 than a poorly-mixed chain of the same
    length.  What is given up is any non-Gaussian shape in the posterior: skew
    in the variance parameters, and the rotational ridge in `Lambda`.  Both are
    reasons this is labelled an approximation and M6 remains the correct fix.

    `shrink` pulls the covariance toward its diagonal, which keeps the draw
    well-conditioned when the chain has not resolved every off-diagonal.
    """
    Sigma = np.cov(chain, rowvar=False)
    Sigma = (1 - shrink) * Sigma + shrink * np.diag(np.diag(Sigma))
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-12, None)
    L = V @ np.diag(np.sqrt(w))
    rng = np.random.default_rng(seed)
    return mode + rng.standard_normal((n_draws, len(mode))) @ L.T


def identified_quantities(draws: np.ndarray, template: Params) -> dict:
    """Map packed draws to the quantities that are actually identified.

    `Lambda` is only pinned down up to an orthogonal rotation -- `(Lambda R)`
    and `Lambda` give the same `Q` for any orthogonal `R` -- so R-hat computed
    on individual `Lambda` entries is measuring the chain wandering around a
    rotation manifold, not a convergence failure.  Diagnostics belong on `Q`
    itself, on `A`, on `phi` and on `sigma_poss`, all of which are identified.

    Returns a dict of (chains, n, k) or (n, k) arrays, matching the input shape.
    """
    flat = draws.reshape(-1, draws.shape[-1])
    iu = np.triu_indices(S)
    Q = np.zeros((len(flat), len(iu[0])))
    A = np.zeros((len(flat), S))
    phi = np.zeros((len(flat), N_VOL))
    sig = np.zeros((len(flat), 1))
    for i, v in enumerate(flat):
        p = hier.unpack(v, template)
        Q[i] = p.Q()[iu]
        A[i] = p.A
        phi[i] = p.phi
        sig[i] = p.sigma_poss
    shape = draws.shape[:-1]
    return {"A": A.reshape(*shape, -1), "Q": Q.reshape(*shape, -1),
            "phi": phi.reshape(*shape, -1), "sigma_poss": sig.reshape(*shape, -1)}
