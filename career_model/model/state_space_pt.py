"""Kalman recursions for the permanent/transient model (§3.4 v2).

State per player-season is `[ell ; u]` (permanent level, transient deviation),
dimension 2S = 28.  The observation sums them, so unlike v1 the observation
matrix is `Z = [I, I]` rather than `[I, 0]`:

    ell_{t+1} = ell_t + delta(age_t) + eta^ell,   eta^ell ~ N(0, Q_ell)
    u_{t+1}   = Phi u_t + eta^u,                   eta^u   ~ N(0, Q_u)
    z_t       = ell_t + u_t + obs_noise

`ell` being a random walk is the whole point: the filter tracks it from the data
instead of estimating a fixed mean, so the star-level identification failure of
v1 does not arise (Phase-1 validation: Jokic's filtered level went from ~1,400
to ~4,900, tracking his actual ~5,300).

**Two paths, as in v1.**  With `Q_ell`, `Q_u` and `Sigma_ell0` diagonal the S
dimensions decouple into S independent 2-state filters -- pure elementwise
arithmetic, milliseconds -- and that is where the drift, the persistence and the
variances are fitted.  The full path carries the low-rank cross-stat structure
and costs a batched Cholesky per season; it is what projection runs on.

**Missing seasons** are handled exactly as in v1: a gap sets every dimension's
`R` to `MISSING_R`, the update gain against it is zero, and both components keep
evolving (the level random-walks, the transient decays), so uncertainty grows
and the filter trusts new data more on reappearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import JITTER, MISSING_R, S
from .hierarchy_pt import ParamsPT

LOG_2PI = float(np.log(2 * np.pi))
LOG_MISSING_R = float(np.log(MISSING_R))


@dataclass
class FilterResultPT:
    loglik: float
    ell: np.ndarray | None = None      # (N, T+1, S) filtered permanent level
    u: np.ndarray | None = None        # (N, T+1, S) filtered transient
    Pee: np.ndarray | None = None      # covariance blocks (diag: (N,T+1,S))
    Peu: np.ndarray | None = None
    Puu: np.ndarray | None = None
    per_player: np.ndarray | None = None
    conditioning: str = "filtered"     # same leakage guard as v1


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
def prior_level(params: ParamsPT, X, age0, init_basis, c_init):
    """Prior mean of `ell_0`: `beta'x + f_GBM + c_init(age_0)`.

    The debut-age offset places the initial *level* on the aging curve, exactly
    as in v1; after that `ell` random-walks and the prior no longer pulls on it,
    which is what frees a star's level from a wrong draft prior.
    """
    b = X @ params.beta
    if params.gbm_offset is not None:
        b = b + params.gbm_offset
    if c_init is not None:
        arr = np.asarray(c_init)
        if arr.ndim == 2 and arr.shape[0] == X.shape[0]:
            b = b + arr
        elif init_basis is not None:
            b = b + init_basis(age0) @ arr
    return b


# ---------------------------------------------------------------------------
# Diagonal path -- fitting workhorse
# ---------------------------------------------------------------------------
def run_filter_pt_diag(grid, params: ParamsPT, X, R, drift, init_basis=None,
                       c_init=None, keep_states=False, dims=None) -> FilterResultPT:
    """Per-stat 2-state filter with diagonal `Q_ell`, `Q_u`, `Sigma_ell0`."""
    N, T = grid.z.shape[0], grid.z.shape[1]
    phi_t = params.Phi[None, :]
    q_ell = np.diag(params.Q_ell())[None, :]
    q_u = np.diag(params.Q_u())[None, :]
    d_u = np.diag(params.transient_stationary())[None, :]
    sig0 = np.diag(params.Sigma_ell0)[None, :]

    ell = prior_level(params, X, grid.age[:, 0], init_basis, c_init).copy()
    u = np.zeros((N, S))
    Pee = np.broadcast_to(sig0, (N, S)).copy()
    Puu = np.broadcast_to(d_u, (N, S)).copy()
    Peu = np.zeros((N, S))

    use = np.zeros(S, dtype=bool)
    use[np.arange(S) if dims is None else np.atleast_1d(dims)] = True

    if keep_states:
        ELL = np.zeros((N, T + 1, S)); U = np.zeros((N, T + 1, S))
        PEE = np.zeros((N, T + 1, S)); PEU = np.zeros((N, T + 1, S)); PUU = np.zeros((N, T + 1, S))

    ll = np.zeros(N)
    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :] & use
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - (ell[sl] + u[sl]), 0.0)
        F = Pee[sl] + 2 * Peu[sl] + Puu[sl] + Rt

        ll[sl] += np.sum(np.where(mask, -0.5 * (np.log(F) + v * v / F + LOG_2PI), 0.0), axis=1)

        ZPe = Pee[sl] + Peu[sl]
        ZPu = Peu[sl] + Puu[sl]
        Ke = ZPe / F
        Ku = ZPu / F
        ell[sl] = ell[sl] + Ke * v
        u[sl] = u[sl] + Ku * v
        Pee[sl] = Pee[sl] - Ke * ZPe
        Peu[sl] = Peu[sl] - Ke * ZPu
        Puu[sl] = Puu[sl] - Ku * ZPu

        if keep_states:
            ELL[:, t] = ell; U[:, t] = u
            PEE[:, t] = Pee; PEU[:, t] = Peu; PUU[:, t] = Puu

        # predict
        ell[sl] = ell[sl] + drift[sl, t, :]
        Pee[sl] = Pee[sl] + q_ell
        Peu[sl] = phi_t * Peu[sl]
        u[sl] = phi_t * u[sl]
        Puu[sl] = phi_t ** 2 * Puu[sl] + q_u

    if keep_states:
        ELL[:, T] = ell; U[:, T] = u
        PEE[:, T] = Pee; PEU[:, T] = Peu; PUU[:, T] = Puu
        return FilterResultPT(float(ll.sum()), ELL, U, PEE, PEU, PUU, ll)
    return FilterResultPT(float(ll.sum()), per_player=ll)


# ---------------------------------------------------------------------------
# Full path -- projection and cross-stat correlation
# ---------------------------------------------------------------------------
def _chol(H):
    L = np.linalg.cholesky(H)
    logdet = 2.0 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)
    return L, logdet


def _solve(L, B):
    return np.linalg.solve(np.swapaxes(L, -1, -2), np.linalg.solve(L, B))


def run_filter_pt(grid, params: ParamsPT, X, R, drift, init_basis=None,
                  c_init=None, keep_states=False) -> FilterResultPT:
    """Full-covariance permanent/transient filter (projection runs on this)."""
    N, T = grid.z.shape[0], grid.z.shape[1]
    Phi = params.Phi
    Q_ell = params.Q_ell()
    Q_u = params.Q_u()
    eye = np.eye(S)

    ell = prior_level(params, X, grid.age[:, 0], init_basis, c_init).copy()
    u = np.zeros((N, S))
    Pee = np.broadcast_to(params.Sigma_ell0, (N, S, S)).copy()
    Puu = np.broadcast_to(params.transient_stationary(), (N, S, S)).copy()
    Peu = np.zeros((N, S, S))

    if keep_states:
        ELL = np.zeros((N, T + 1, S)); U = np.zeros((N, T + 1, S))
        PEE = np.zeros((N, T + 1, S, S)); PEU = np.zeros((N, T + 1, S, S)); PUU = np.zeros((N, T + 1, S, S))

    ll = np.zeros(N)
    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :]
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - (ell[sl] + u[sl]), 0.0)
        n_obs = mask.sum(axis=1)

        Pue = np.swapaxes(Peu[sl], -1, -2)
        F = Pee[sl] + Peu[sl] + Pue + Puu[sl] + Rt[:, :, None] * eye
        F = 0.5 * (F + np.swapaxes(F, -1, -2)) + JITTER * eye
        L, logdet = _chol(F)

        ll[sl] += -0.5 * (logdet - (S - n_obs) * LOG_MISSING_R
                          + np.einsum("ns,ns->n", v, _solve(L, v[:, :, None])[:, :, 0])
                          + n_obs * LOG_2PI)

        ZPe = Pee[sl] + Pue          # (Pee + Pue)  -> gain numerator for ell:  Pee + Peu
        # careful: K = P Z' F^-1, P Z' = [Pee+Peu ; Pue+Puu]
        PZt_e = Pee[sl] + Peu[sl]
        PZt_u = Pue + Puu[sl]
        Ke = _solve(L, np.swapaxes(PZt_e, -1, -2))
        Ke = np.swapaxes(Ke, -1, -2)                     # PZt_e F^-1
        Ku = np.swapaxes(_solve(L, np.swapaxes(PZt_u, -1, -2)), -1, -2)

        ell[sl] = ell[sl] + np.einsum("nij,nj->ni", Ke, v)
        u[sl] = u[sl] + np.einsum("nij,nj->ni", Ku, v)
        # Z P = [Pee+Pue, Peu+Puu]
        ZP_e = Pee[sl] + Pue
        ZP_u = Peu[sl] + Puu[sl]
        newPee = Pee[sl] - Ke @ ZP_e
        newPeu = Peu[sl] - Ke @ ZP_u
        newPuu = Puu[sl] - Ku @ ZP_u
        Pee[sl] = 0.5 * (newPee + np.swapaxes(newPee, -1, -2))
        Puu[sl] = 0.5 * (newPuu + np.swapaxes(newPuu, -1, -2))
        Peu[sl] = newPeu
        _ = ZPe

        if keep_states:
            ELL[:, t] = ell; U[:, t] = u
            PEE[:, t] = Pee; PEU[:, t] = Peu; PUU[:, t] = Puu

        # predict
        ell[sl] = ell[sl] + drift[sl, t, :]
        Pee[sl] = Pee[sl] + Q_ell
        Peu[sl] = Peu[sl] * Phi[None, None, :]
        u[sl] = Phi * u[sl]
        Puu[sl] = Phi[:, None] * Puu[sl] * Phi[None, :] + Q_u

    if keep_states:
        ELL[:, T] = ell; U[:, T] = u
        PEE[:, T] = Pee; PEU[:, T] = Peu; PUU[:, T] = Puu
        return FilterResultPT(float(ll.sum()), ELL, U, PEE, PEU, PUU, ll)
    return FilterResultPT(float(ll.sum()), per_player=ll)


# ---------------------------------------------------------------------------
# Exact profiling of the within-player drift (§1.5)
# ---------------------------------------------------------------------------
def profile_drift_diag(grid, params: ParamsPT, X, R, age_basis, init_basis,
                       pos_idx, c_league, c_pos, c_lin, ridge_pos=20.0, ridge=1e-3):
    """Profile the random-walk drift `delta(age)` and the debut offset exactly.

    The drift enters `ell`'s mean linearly, so the marginal likelihood is a
    quadratic in the coefficients and the augmented recursion reaches the
    optimum in one step -- identical machinery to v1's `profile_linear_diag`,
    only the state carried is `[ell, u]` and the derivative flows through the
    permanent component.  Within-player by construction (§1.5): the debut offset
    keeps its own spline so the drift is never identified by comparing players
    of different debut ages.
    """
    N, T = grid.z.shape[0], grid.z.shape[1]
    nb, nbi = age_basis.size, init_basis.size
    n_groups = c_pos.shape[0]
    W = nbi + nb

    phi_t = params.Phi[None, :]
    q_ell = np.diag(params.Q_ell())[None, :]
    q_u = np.diag(params.Q_u())[None, :]
    d_u = np.diag(params.transient_stationary())[None, :]
    sig0 = np.diag(params.Sigma_ell0)[None, :]

    ell = prior_level(params, X, grid.age[:, 0], init_basis, c_lin).copy()
    u = np.zeros((N, S))
    Pee = np.broadcast_to(sig0, (N, S)).copy()
    Puu = np.broadcast_to(d_u, (N, S)).copy()
    Peu = np.zeros((N, S))

    # derivatives of [ell, u] wrt the drift/init coefficients
    Ve = np.zeros((N, S, W)); Vu = np.zeros((N, S, W))
    Ve[:, :, :nbi] = init_basis(grid.age[:, 0])[:, None, :]   # ell_0 = ... + init@c_lin

    M = np.zeros((n_groups, S, W, W)); rhs = np.zeros((n_groups, S, W))
    group_rows = [np.flatnonzero(pos_idx == g) for g in range(n_groups)]
    phi_age = age_basis(grid.age)
    coefs = c_league[None] + c_pos[pos_idx]

    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :]
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - (ell[sl] + u[sl]), 0.0)
        F = Pee[sl] + 2 * Peu[sl] + Puu[sl] + Rt
        # d(z_pred)/d(coef) = Ve + Vu
        Vz = Ve[sl] + Vu[sl]
        w = Vz / np.sqrt(F)[:, :, None]
        rv = v / F
        for g, r in enumerate(group_rows):
            r = r[r < n]
            if not len(r):
                continue
            M[g] += np.einsum("nsw,nsx->swx", w[r], w[r])
            rhs[g] += np.einsum("nsw,ns->sw", Vz[r], rv[r])

        ZPe = Pee[sl] + Peu[sl]; ZPu = Peu[sl] + Puu[sl]
        Ke = ZPe / F; Ku = ZPu / F
        ell[sl] += Ke * v; u[sl] += Ku * v
        Ve[sl] = Ve[sl] - Ke[:, :, None] * Vz
        Vu[sl] = Vu[sl] - Ku[:, :, None] * Vz
        Pee[sl] -= Ke * ZPe; Peu[sl] -= Ke * ZPu; Puu[sl] -= Ku * ZPu

        d = np.einsum("nk,nks->ns", phi_age[sl, t, :], coefs[sl])
        ell[sl] = ell[sl] + d
        Ve[sl, :, nbi:] += phi_age[sl, t, :][:, None, :]     # d(ell)/d(drift coef)
        Pee[sl] += q_ell
        Peu[sl] = phi_t * Peu[sl]
        u[sl] = phi_t * u[sl]
        Puu[sl] = phi_t ** 2 * Puu[sl] + q_u

    return _solve_drift(M, rhs, nbi, nb, n_groups, c_lin, c_league, c_pos, ridge_pos, ridge)


def profile_drift_scalar(grid, params: ParamsPT, X, R, drift, init_basis, c_init,
                         prior_sd=0.4, clip=(0.15, 2.0)):
    """Per-player multiplier `s_i` on the aging drift (the player level of §4.1).

    `ell_{t+1} = ell_t + s_i * drift(age) + eta`.  The population drift is too
    steep for a durable star -- it is the average, and includes the sharp
    end-of-career drops of players heading for retirement -- so a 37-year-old who
    is still producing has his sustained level pushed off `ell` (which follows
    the steep drift down) and into the fast-decaying transient, which then
    craters his projection.

    This failed in v1 because there the scalar was collinear with the level
    `m_i`.  Here it is not: `ell` is a random walk that tracks the *level*
    regardless of `s_i`, so the scalar only has to explain the *slope* of a
    player's decline relative to his position group -- a separable signal, and
    best identified for exactly the long-career old players the crater afflicts.

    Given the drift, `s_i * drift(age)` is linear in `s_i`, so one derivative
    column carried through the 2-state recursion gives the exact GLS estimate,
    shrunk toward 1 by `prior_sd`.
    """
    N, T = grid.z.shape[0], grid.z.shape[1]
    phi_t = params.Phi[None, :]
    q_ell = np.diag(params.Q_ell())[None, :]
    q_u = np.diag(params.Q_u())[None, :]
    d_u = np.diag(params.transient_stationary())[None, :]
    sig0 = np.diag(params.Sigma_ell0)[None, :]

    ell = prior_level(params, X, grid.age[:, 0], init_basis, c_init).copy()
    u = np.zeros((N, S))
    Pee = np.broadcast_to(sig0, (N, S)).copy()
    Puu = np.broadcast_to(d_u, (N, S)).copy()
    Peu = np.zeros((N, S))

    Ve = np.zeros((N, S)); Vu = np.zeros((N, S))     # d[ell,u]/d s_i (scalar per player)
    M = np.zeros(N); rhs = np.zeros(N)

    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :]
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - (ell[sl] + u[sl]), 0.0)
        F = Pee[sl] + 2 * Peu[sl] + Puu[sl] + Rt
        Vz = Ve[sl] + Vu[sl]
        M[sl] += np.sum(Vz * Vz / F, axis=1)
        rhs[sl] += np.sum(Vz * v / F, axis=1)

        ZPe = Pee[sl] + Peu[sl]; ZPu = Peu[sl] + Puu[sl]
        Ke = ZPe / F; Ku = ZPu / F
        ell[sl] += Ke * v; u[sl] += Ku * v
        Ve[sl] = Ve[sl] - Ke * Vz
        Vu[sl] = Vu[sl] - Ku * Vz
        Pee[sl] -= Ke * ZPe; Peu[sl] -= Ke * ZPu; Puu[sl] -= Ku * ZPu

        ell[sl] = ell[sl] + drift[sl, t, :]
        Ve[sl] = Ve[sl] + drift[sl, t, :]            # d(s_i*drift)/d s_i = drift
        Pee[sl] += q_ell
        Peu[sl] = phi_t * Peu[sl]
        u[sl] = phi_t * u[sl]
        Puu[sl] = phi_t ** 2 * Puu[sl] + q_u

    ridge = 1.0 / prior_sd ** 2
    return np.clip(1.0 + rhs / (M + ridge), *clip)


def _solve_drift(M, rhs, nbi, nb, n_groups, c_lin, c_league, c_pos, ridge_pos, ridge):
    P = nbi + nb * (1 + n_groups)
    ii, dd = slice(0, nbi), slice(nbi, nbi + nb)

    def sl(block):
        if block == "init":
            return slice(0, nbi)
        if block == "league":
            return slice(nbi, nbi + nb)
        return slice(nbi + nb * (1 + block), nbi + nb * (2 + block))

    new_lin = np.zeros_like(c_lin)
    new_league = np.zeros_like(c_league)
    new_pos = np.zeros_like(c_pos)
    for s in range(M.shape[1]):
        A = np.zeros((P, P)); b = np.zeros(P)
        for g in range(n_groups):
            Mg, ug = M[g, s], rhs[g, s]
            blocks = ["init", "league", g]; parts = [ii, dd, dd]
            for br, pr in zip(blocks, parts):
                b[sl(br)] += ug[pr]
                for cbl, pc in zip(blocks, parts):
                    A[sl(br), sl(cbl)] += Mg[pr, pc]
        A[np.diag_indices(P)] += ridge
        for g in range(n_groups):
            A[sl(g), sl(g)] += ridge_pos * np.eye(nb)
        step = np.linalg.solve(A, b)
        new_lin[:, s] = c_lin[:, s] + step[:nbi]
        new_league[:, s] = c_league[:, s] + step[nbi:nbi + nb]
        for g in range(n_groups):
            new_pos[g, :, s] = c_pos[g, :, s] + step[sl(g)]
    return new_lin, new_league, new_pos
