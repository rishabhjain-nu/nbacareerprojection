"""Kalman recursions over the augmented state `[theta_t ; m_i]` (§5.1).

The state the spec writes down is `theta`, with `m_i` a hierarchical parameter
sitting in the transition.  Filtering it as *part of the state* is the same
model and is strictly better behaved: `m_i` is a constant component, so the
filter's estimate of it after the last season is automatically the estimate
given the player's whole career, and the shrinkage of `m_i` toward `beta' x_i`
falls out of the same gain that shrinks `theta` toward its prediction.  One `R`,
one mechanism, and the `K = 1 - B` identity of §9 holds by construction rather
than by two layers happening to agree.

    x_t = [theta_t ; m]     T = [[A, I-A], [0, I]]     Z = [I, 0]
    c_t = [delta(age_t) ; 0]                           Q_aug = blkdiag(Q, 0)

Because `A` is diagonal, every block of `T P T'` is an elementwise rescaling of
a 14x14 block -- there is not a single 28x28 matmul in the hot loop.

**Two paths.**  `run_filter_diag` is the model with `Q` and `Sigma_player`
diagonal.  The 14 dimensions then decouple into 14 independent 2-state filters
that vectorize to pure elementwise arithmetic, no factorization anywhere, and it
runs in milliseconds.  That is stages M1, M2 and M4 exactly, and it is where the
persistence, process variance and aging coefficients get fitted.  `run_filter`
is the full model with the low-rank `Q` of §3.4 -- it is what M3 onward and
every projection uses, and it costs a batched Cholesky per season.

**Missing seasons.**  A gap is a grid row with every dimension's `R` set to
`MISSING_R`.  In the limit that is exactly "skip the update": the gain against
those rows is zero, the state keeps aging through `c_t`, and `P` keeps growing
by `Q`, so on reappearance the filter trusts the new data more.  Handling it
through `R` rather than a branch keeps the batch rectangular and reduces the
likelihood bookkeeping to a single subtraction (see `_loglik_terms`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import JITTER, MISSING_R, S
from .hierarchy import Params

LOG_2PI = float(np.log(2 * np.pi))
LOG_MISSING_R = float(np.log(MISSING_R))


@dataclass
class FilterResult:
    """`x1[:, t]` is `theta_{t|t}`; index `T` holds the one-step-ahead
    prediction past the end of the grid, which is where projection starts.

    `conditioning` is a **load-bearing** tag, not documentation.  Everything in
    this module is a causal forward pass: `theta_{t|t}` sees data through `t`
    and nothing after, and the innovation `v_t = z_t - theta_{t|t-1}` is formed
    against a prediction made before `z_t` was read.

    If an RTS smoother is ever added -- it is legitimate for fitting
    hyperparameters and for drawing a player's historical trajectory in the UI
    -- it must set `conditioning="smoothed"`, because `theta_{t|T}` has seen the
    whole panel including seasons after `t`.  Launching a backtested projection
    from a smoothed state means forecasting from a state that already read the
    answer, and the resulting CRPS is not a forecast score at all.
    `simulate.project` rejects anything not tagged `"filtered"`.
    """

    loglik: float
    x1: np.ndarray | None = None       # (N, T+1, S)
    x2: np.ndarray | None = None       # (N, T+1, S)
    P11: np.ndarray | None = None      # (N, T+1, S, S)  or (N, T+1, S) if diag
    P12: np.ndarray | None = None
    P22: np.ndarray | None = None
    gain: np.ndarray | None = None     # K1, for the M2 shrinkage check
    per_player: np.ndarray | None = None
    conditioning: str = "filtered"     # "filtered" | "smoothed" -- see above


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
def prior_mean(params: Params, X: np.ndarray, age0: np.ndarray,
               init_basis=None, c_init=None):
    """Prior at each player's first season: `theta_0 = m_i + c(age_0)`.

    `m_i ~ N(beta' x_i + f_GBM(x_i), Sigma_player)` (§4.1, §5.3) is the player's
    **level**, and `c(age_0)` places him on the age curve at his debut.  That
    offset is not cosmetic: without it a 19-year-old's prior mean would be his
    eventual peak, which is the easiest way to make a draft model look brilliant
    and be wrong.

    `c_init` may be an (N, S) array of per-player offsets -- which is what the
    level parameterization needs, since the curve carries a position deviation
    and a per-player rate -- or a (n_basis, S) coefficient matrix to be applied
    through `init_basis`, which is the older increment form.
    """
    b = X @ params.beta
    if params.gbm_offset is not None:
        b = b + params.gbm_offset
    x1 = b.copy()
    if c_init is not None:
        arr = np.asarray(c_init)
        if arr.ndim == 2 and arr.shape[0] == X.shape[0]:
            x1 = x1 + arr                       # per-player offset, already evaluated
        elif init_basis is not None:
            x1 = x1 + init_basis(age0) @ arr
    return x1, b


# ---------------------------------------------------------------------------
# Diagonal path -- M1, M2, M4
# ---------------------------------------------------------------------------
def run_filter_diag(grid, params: Params, X: np.ndarray, R: np.ndarray,
                    offsets: np.ndarray, init_basis=None,
                    c_init: np.ndarray | None = None,
                    keep_states: bool = False,
                    dims: np.ndarray | None = None) -> FilterResult:
    """Filter with `Q` and `Sigma_player` treated as diagonal.

    `dims` restricts the likelihood to a subset of state dimensions.  Passing a
    single index is stage M1: one stat, filtered on its own, which is the check
    the spec wants passed by eye before anything else is attempted.
    """
    N, T = grid.z.shape[0], grid.z.shape[1]
    a = params.A[None, :]
    c = 1.0 - a
    q = np.diag(params.Q())[None, :]
    sig_p = np.diag(params.Sigma_p)[None, :]
    disp = np.diag(params.stationary_dispersion())[None, :]

    x1, x2 = prior_mean(params, X, grid.age[:, 0], init_basis, c_init)
    x1 = x1.copy()
    p11 = np.broadcast_to(sig_p + disp, (N, S)).copy()
    p12 = np.broadcast_to(sig_p, (N, S)).copy()
    p22 = np.broadcast_to(sig_p, (N, S)).copy()

    use = np.zeros(S, dtype=bool)
    use[np.arange(S) if dims is None else np.atleast_1d(dims)] = True

    if keep_states:
        X1 = np.zeros((N, T + 1, S)); X2 = np.zeros((N, T + 1, S))
        P11 = np.zeros((N, T + 1, S)); P12 = np.zeros((N, T + 1, S))
        P22 = np.zeros((N, T + 1, S)); GAIN = np.zeros((N, T, S))

    ll = np.zeros(N)
    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :] & use
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - x1[sl], 0.0)
        h = p11[sl] + Rt

        ll[sl] += np.sum(np.where(mask, -0.5 * (np.log(h) + v * v / h + LOG_2PI), 0.0), axis=1)

        k1 = p11[sl] / h
        k2 = p12[sl] / h
        x1[sl] += k1 * v
        x2[sl] += k2 * v
        p22[sl] -= k2 * p12[sl]
        p12[sl] -= k1 * p12[sl]
        p11[sl] -= k1 * p11[sl]

        if keep_states:
            X1[:, t] = x1; X2[:, t] = x2
            P11[:, t] = p11; P12[:, t] = p12; P22[:, t] = p22
            GAIN[:, t] = 0.0; GAIN[sl, t] = k1

        d = offsets[sl, t, :]
        nx1 = a * x1[sl] + c * x2[sl] + d
        np11 = a * a * p11[sl] + 2 * a * c * p12[sl] + c * c * p22[sl] + q
        np12 = a * p12[sl] + c * p22[sl]
        x1[sl], p11[sl], p12[sl] = nx1, np11, np12

    if keep_states:
        X1[:, T] = x1; X2[:, T] = x2
        P11[:, T] = p11; P12[:, T] = p12; P22[:, T] = p22
        return FilterResult(float(ll.sum()), X1, X2, P11, P12, P22, GAIN, ll)
    return FilterResult(float(ll.sum()), per_player=ll)


# ---------------------------------------------------------------------------
# Full path -- M3 onward
# ---------------------------------------------------------------------------
def _chol(H):
    L = np.linalg.cholesky(H)
    logdet = 2.0 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)
    return L, logdet


def _chol_solve(L, B):
    y = np.linalg.solve(L, B)
    return np.linalg.solve(np.swapaxes(L, -1, -2), y)


def run_filter(grid, params: Params, X: np.ndarray, R: np.ndarray,
               offsets: np.ndarray, init_basis=None,
               c_init: np.ndarray | None = None,
               keep_states: bool = False,
               m_prior_scale: float = 1.0,
               m_prior_mean: np.ndarray | None = None) -> FilterResult:
    """Full-covariance filter.  This is the one projections run on.

    `m_prior_scale` (S3-A "diffuse-m") inflates the initial m-block covariance
    (P22) but leaves m's prior *mean* at f_GBM.  `m_prior_mean` (S3 closure
    "decoupled-m") instead REPLACES the m-block prior mean with an
    f_GBM-independent vector, so the pre-NBA prediction informs only theta_0
    (the entry state), not the permanent reversion destination m -- the
    structurally-distinct formulation Session-3 actually requested.  theta_0's
    prior (P11 and x1) is unchanged in both.
    """
    N, T = grid.z.shape[0], grid.z.shape[1]
    a = params.A
    c = 1.0 - a
    Q = params.Q()
    aa = np.outer(a, a); ac = np.outer(a, c); ca = ac.T; cc = np.outer(c, c)
    eye = np.eye(S)

    x1, x2 = prior_mean(params, X, grid.age[:, 0], init_basis, c_init)
    x1 = x1.copy()
    if m_prior_mean is not None:
        # Decoupled-m: theta_0 keeps its f_GBM-based prior (x1); only m's prior
        # mean is replaced by the f_GBM-independent target.
        x2 = np.asarray(m_prior_mean, float).copy()
    Sig = params.Sigma_p
    D = params.stationary_dispersion()
    P11 = np.broadcast_to(Sig + D, (N, S, S)).copy()
    P12 = np.broadcast_to(Sig, (N, S, S)).copy()
    # S3-A: a diffuse m-prior inflates only the m-block so NBA evidence learns
    # the persistent level.  theta_0 (P11) and the theta_0<->m coupling (P12,
    # = Cov(m+drift, m) = Sig) are left as-is; the joint init stays PSD.
    P22 = np.broadcast_to(Sig * m_prior_scale, (N, S, S)).copy()

    if keep_states:
        X1 = np.zeros((N, T + 1, S)); X2 = np.zeros((N, T + 1, S))
        PP11 = np.zeros((N, T + 1, S, S)); PP12 = np.zeros((N, T + 1, S, S))
        PP22 = np.zeros((N, T + 1, S, S)); GAIN = np.zeros((N, T, S, S))

    ll = np.zeros(N)
    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :]
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - x1[sl], 0.0)
        n_obs = mask.sum(axis=1)

        H = P11[sl] + Rt[:, :, None] * eye
        H = 0.5 * (H + np.swapaxes(H, -1, -2)) + JITTER * eye
        L, logdet = _chol(H)

        rhs = np.concatenate([P11[sl], np.swapaxes(P12[sl], -1, -2),
                              v[:, :, None].transpose(0, 2, 1)], axis=1)   # (n, 2S+1, S)
        sol = _chol_solve(L, np.swapaxes(rhs, -1, -2))                     # (n, S, 2S+1)
        K1 = np.swapaxes(sol[:, :, :S], -1, -2)
        K2 = np.swapaxes(sol[:, :, S:2 * S], -1, -2)
        Hv = sol[:, :, 2 * S]

        ll[sl] += -0.5 * (logdet - (S - n_obs) * LOG_MISSING_R
                          + np.einsum("ns,ns->n", v, Hv) + n_obs * LOG_2PI)

        x1[sl] += np.einsum("nij,nj->ni", K1, v)
        x2[sl] += np.einsum("nij,nj->ni", K2, v)
        P22[sl] -= K2 @ P12[sl]
        newP12 = P12[sl] - K1 @ P12[sl]
        P11[sl] -= K1 @ P11[sl]
        P12[sl] = newP12
        P11[sl] = 0.5 * (P11[sl] + np.swapaxes(P11[sl], -1, -2))
        P22[sl] = 0.5 * (P22[sl] + np.swapaxes(P22[sl], -1, -2))

        if keep_states:
            X1[:, t] = x1; X2[:, t] = x2
            PP11[:, t] = P11; PP12[:, t] = P12; PP22[:, t] = P22
            GAIN[sl, t] = K1

        # ---- predict -----------------------------------------------------
        d = offsets[sl, t, :]
        nx1 = a * x1[sl] + c * x2[sl] + d
        nP11 = (aa * P11[sl] + ac * P12[sl] + ca * np.swapaxes(P12[sl], -1, -2)
                + cc * P22[sl] + Q)
        nP12 = a[:, None] * P12[sl] + c[:, None] * P22[sl]
        x1[sl] = nx1
        P11[sl] = 0.5 * (nP11 + np.swapaxes(nP11, -1, -2))
        P12[sl] = nP12

    if keep_states:
        X1[:, T] = x1; X2[:, T] = x2
        PP11[:, T] = P11; PP12[:, T] = P12; PP22[:, T] = P22
        return FilterResult(float(ll.sum()), X1, X2, PP11, PP12, PP22, GAIN, ll)
    return FilterResult(float(ll.sum()), per_player=ll)


def assert_positive_definite(grid, params: Params, X: np.ndarray, R: np.ndarray,
                             offsets: np.ndarray, init_basis=None, c_init=None) -> None:
    """§5.1: assert `F_t` is PD at every step.

    A failure here is almost never a numerical accident -- it means two state
    dimensions are carrying the same information, which is what §1.2 forbids.
    `tests/test_no_redundant_dims.py` is the standing version of this check.
    """
    res = run_filter(grid, params, X, R, offsets, init_basis, c_init, keep_states=True)
    for t in range(grid.z.shape[1]):
        n = int(grid.n_active[t])
        if n == 0:
            break
        H = res.P11[:n, t] + np.where(grid.obs_mask[:n, t], R[:n, t], MISSING_R)[:, :, None] * np.eye(S)
        w = np.linalg.eigvalsh(0.5 * (H + np.swapaxes(H, -1, -2)))
        if w.min() <= 0:
            raise AssertionError(f"F_t not positive definite at career-season {t}: "
                                 f"min eigenvalue {w.min():.3e}")


# ---------------------------------------------------------------------------
# Exact profiling of the linear terms (Durbin & Koopman ch. 6)
# ---------------------------------------------------------------------------
def profile_linear_diag(grid, params: Params, X: np.ndarray, R: np.ndarray,
                        age_basis, init_basis, pos_idx: np.ndarray,
                        c_delta_league: np.ndarray, c_delta_pos: np.ndarray,
                        c_init: np.ndarray, ridge_pos: float = 20.0,
                        ridge: float = 1e-3, level_param: bool = True):
    """Solve exactly for the aging curve, in the **level** parameterization.

    §3.4 writes the transition with `delta(age)` as an increment:

        theta_{t+1} = m_i + A(theta_t - m_i) + delta(age_t)

    which makes `m_i` an AR *intercept*.  That is algebraically fine and
    numerically poor.  Rearranged, `m_i` is identified by
    `(1-A) m_i = theta_{t+1} - A theta_t - delta`, so estimating it divides by
    `1 - A`.  For log-possessions `A = 0.87`, which amplifies the estimation
    variance by `1/(1-A)^2 = 63`.  The data then say almost nothing about
    `m_i`, the `N(beta'x_i, Sigma_player)` prior supplies ~74% of it, and a
    player the GBM reads as marginal keeps a marginal `m_i` no matter how many
    seasons contradict it.  Jokic -- a 41st pick who has logged ~5,000
    possessions a year for a decade -- came out with `m_i` at 1,100
    possessions, and the projection then reverted him toward it every year.

    So the curve is carried as a **level path** `c(age)` instead:

        theta_{t+1} = m_i + c(age_{t+1}) + A(theta_t - m_i - c(age_t))

    identical to the spec's form with `delta(a) = c(a+1) - A c(a)`, but now
    `m_i` is the player's own level, identified straight off his observations
    with no amplification, and the prior shrinks a level rather than an
    intercept.  It also merges the debut-age offset into the same spline: the
    initial state is just `m_i + c(age_0)`, which is what §3.4's increment
    formulation needed a separate `c_init` block to express.

    Still exactly linear in the coefficients, so the same augmented recursion
    reaches the optimum in one Newton step -- only the derivative increments
    change, from `Phi(age_t)` to `Phi(age_{t+1}) - A Phi(age_t)`.

    `delta` and the debut-age offset enter the state mean **linearly**, so the
    innovations are affine in their coefficients and the marginal log-likelihood
    is an exact quadratic in them.  Carrying the derivative of the state mean
    alongside the state through the same recursion therefore reaches the global
    optimum in one Newton step -- no EM iteration, no smoother, and none of the
    attenuation you get from regressing filtered states on age.

    That it is a *within-player* estimator is not a detail (§1.5).  The
    derivative accumulates over a player's own consecutive-age transitions.  A
    player who washes out at 30 contributes no age-31 transition, so he cannot
    push the age-31 increment upward by his absence -- which is exactly the
    survivor bias that fitting an aging curve on population age means walks into.

    Under the diagonal model the dimensions separate, so this runs as `S`
    independent GLS problems solved in one batched pass.
    """
    N, T = grid.z.shape[0], grid.z.shape[1]
    nb, nbi = age_basis.size, init_basis.size
    n_groups = c_delta_pos.shape[0]
    W = nbi + nb                       # derivative columns per dimension
    # In the level parameterization the initial-state block and the transition
    # block are the *same* spline, so the init columns carry no separate
    # coefficient; they are folded into the level block below.

    a = params.A[None, :, None]
    c = 1.0 - a
    q = np.diag(params.Q())[None, :]
    sig_p = np.diag(params.Sigma_p)[None, :]
    disp = np.diag(params.stationary_dispersion())[None, :]

    x1, x2 = prior_mean(params, X, grid.age[:, 0], init_basis, c_init)
    x1 = x1.copy()
    p11 = np.broadcast_to(sig_p + disp, (N, S)).copy()
    p12 = np.broadcast_to(sig_p, (N, S)).copy()
    p22 = np.broadcast_to(sig_p, (N, S)).copy()

    V1 = np.zeros((N, S, W))
    V2 = np.zeros((N, S, W))
    # **The initial state gets its own block, always.**  It is tempting to let
    # the level curve place the initial state too -- `theta_0 = m + c(age_0)` is
    # the self-consistent form -- and that is exactly the trap.  Debut age is
    # confounded with player quality: a 22-year-old rookie is a worse prospect
    # than a 19-year-old one, so letting `c` explain the initial state lets it
    # be identified by *comparing players of different ages*, which §1.5
    # forbids in the strongest terms.  Doing it produced a curve peaking at 29
    # -- still rising at 26, where the within-player data already fall 9.5% a
    # year -- because it had absorbed the survivor composition of the panel.
    #
    # So `c` is identified from within-player transitions only, and a separate
    # debut-age spline carries wherever a player enters the league.
    V1[:, :, :nbi] = init_basis(grid.age[:, 0])[:, None, :]

    M = np.zeros((n_groups, S, W, W))
    u = np.zeros((n_groups, S, W))
    group_rows = [np.flatnonzero(pos_idx == g) for g in range(n_groups)]

    phi_age = age_basis(grid.age)                                  # (N, T, nb)
    phi_next = age_basis(grid.age + 1.0)                           # ages one season on
    coefs = c_delta_league[None] + c_delta_pos[pos_idx]             # (N, nb, S)

    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :]
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - x1[sl], 0.0)
        h = p11[sl] + Rt

        w = V1[sl] / np.sqrt(h)[:, :, None]
        rv = v / h
        for g, rows in enumerate(group_rows):
            r = rows[rows < n]
            if not len(r):
                continue
            M[g] += np.einsum("nsw,nsx->swx", w[r], w[r])
            u[g] += np.einsum("nsw,ns->sw", V1[r], rv[r])

        k1 = p11[sl] / h
        k2 = p12[sl] / h
        x1[sl] += k1 * v
        x2[sl] += k2 * v
        V1[sl], V2[sl] = (V1[sl] - k1[:, :, None] * V1[sl],
                          V2[sl] - k2[:, :, None] * V1[sl])
        p22[sl] -= k2 * p12[sl]
        p12[sl] -= k1 * p12[sl]
        p11[sl] -= k1 * p11[sl]

        # Level form: the transition offset is c(age_{t+1}) - A c(age_t), so the
        # derivative increment is Phi(age_{t+1}) - A Phi(age_t).
        if level_param:
            nxt = phi_next[sl, t, :]
            inc = nxt[:, None, :] - params.A[None, :, None] * phi_age[sl, t, :][:, None, :]
            d = (np.einsum("nk,nks->ns", nxt, coefs[sl])
                 - params.A[None, :] * np.einsum("nk,nks->ns", phi_age[sl, t, :], coefs[sl]))
        else:
            inc = np.broadcast_to(phi_age[sl, t, :][:, None, :], (n, S, nb))
            d = np.einsum("nk,nks->ns", phi_age[sl, t, :], coefs[sl])
        nV1 = a * V1[sl] + c * V2[sl]
        nV1[:, :, nbi:] += inc
        x1[sl] = a[:, :, 0] * x1[sl] + c[:, :, 0] * x2[sl] + d
        np11 = (a[:, :, 0] ** 2 * p11[sl] + 2 * a[:, :, 0] * c[:, :, 0] * p12[sl]
                + c[:, :, 0] ** 2 * p22[sl] + q)
        p12[sl] = a[:, :, 0] * p12[sl] + c[:, :, 0] * p22[sl]
        p11[sl] = np11
        V1[sl] = nV1

    return _solve_linear_terms(M, u, nbi, nb, n_groups, c_init,
                               c_delta_league, c_delta_pos, ridge_pos, ridge)


def profile_player_aging(grid, params: Params, X: np.ndarray, R: np.ndarray,
                         offsets: np.ndarray, init_basis, c_init: np.ndarray,
                         prior_sd: float = 0.45, clip=(0.0, 2.2)) -> np.ndarray:
    """Per-player rate-of-aging multiplier -- the player level of §4.1's `delta`.

    UNUSED in the shipped model.  It was collinear with `m_i` (over an eight-
    season window `s_i * delta(age)` is near-linear in age and trades off against
    the level intercept), which drove `m_i` to physically impossible values.  The
    star-decline problem it targeted is handled instead by the empirical-Bayes
    reversion target in `simulate.project`.  Kept for reference and because the
    exact-profiling machinery it demonstrates is reusable; call sites were
    removed from `fit_kf` and `pipeline`.

    The league and position curves say how the *average* player at each age
    moves.  Applied alone they say every 38-year-old declines at the same rate,
    and that is visibly false: of the players who were logging 4,000+
    possessions at 37, the ones still in the league at 41 are playing ~2,000,
    while the league curve compounded over four years predicts ~400.  The
    hazard cannot repair this, because it can only remove players -- it cannot
    make the survivors decline more slowly than the single curve everyone
    shares.  The result is a model that is right about a 38-year-old and absurd
    about a 41-year-old.

    So each player carries one scalar `s_i` multiplying his whole aging curve.
    Given `delta`, the offset `s_i * delta(age)` is **linear in `s_i`**, so the
    same augmented recursion that profiles `delta` estimates it exactly -- one
    extra derivative column, accumulated per player.  It is deliberately a
    single scalar rather than a curve: eight degrees of freedom per stat cannot
    be recovered from one career, but "ages faster or slower than his position"
    can be, and it is the part that matters.

    Shrunk toward 1 by `prior_sd`, so a two-season player gets the league rate
    and an eighteen-season player gets his own.  Clipped to keep a freak history
    from producing a negative multiplier (which would age a player *backwards*).
    """
    N, T = grid.z.shape[0], grid.z.shape[1]
    a = params.A[None, :]
    c = 1.0 - a
    q = np.diag(params.Q())[None, :]
    sig_p = np.diag(params.Sigma_p)[None, :]
    disp = np.diag(params.stationary_dispersion())[None, :]

    x1, x2 = prior_mean(params, X, grid.age[:, 0], init_basis, c_init)
    x1 = x1.copy()
    p11 = np.broadcast_to(sig_p + disp, (N, S)).copy()
    p12 = np.broadcast_to(sig_p, (N, S)).copy()
    p22 = np.broadcast_to(sig_p, (N, S)).copy()

    V1 = np.zeros((N, S))          # d(theta)/d(s_i)
    V2 = np.zeros((N, S))
    M = np.zeros(N)
    u = np.zeros(N)

    for t in range(T):
        n = int(grid.n_active[t])
        if n == 0:
            break
        sl = slice(0, n)
        mask = grid.obs_mask[sl, t, :]
        Rt = np.where(mask, R[sl, t, :], MISSING_R)
        v = np.where(mask, grid.z[sl, t, :] - x1[sl], 0.0)
        h = p11[sl] + Rt

        M[sl] += np.sum(V1[sl] * V1[sl] / h, axis=1)
        u[sl] += np.sum(V1[sl] * v / h, axis=1)

        k1 = p11[sl] / h
        k2 = p12[sl] / h
        x1[sl] += k1 * v
        x2[sl] += k2 * v
        V1[sl], V2[sl] = V1[sl] - k1 * V1[sl], V2[sl] - k2 * V1[sl]
        p22[sl] -= k2 * p12[sl]
        p12[sl] -= k1 * p12[sl]
        p11[sl] -= k1 * p11[sl]

        d = offsets[sl, t, :]
        nV1 = a * V1[sl] + c * V2[sl] + d      # the derivative of s_i*delta is delta
        x1[sl] = a * x1[sl] + c * x2[sl] + d
        np11 = a * a * p11[sl] + 2 * a * c * p12[sl] + c * c * p22[sl] + q
        p12[sl] = a * p12[sl] + c * p22[sl]
        p11[sl] = np11
        V1[sl] = nV1

    ridge = 1.0 / prior_sd ** 2
    scale = 1.0 + u / (M + ridge)
    return np.clip(scale, *clip)


def _solve_linear_terms(M, u, nbi, nb, n_groups, c_init, c_league, c_pos,
                        ridge_pos, ridge):
    """Assemble and solve the normal equations, one dimension at a time.

    Parameter order per dimension: `[c_init | c_league | c_pos_0 ... ]`.  Player
    `i`'s effective coefficient vector is `[c_init ; c_league + c_pos_{g_i}]`,
    so the cross-blocks are copies of the per-group accumulators.  The position
    deviations carry a ridge -- that *is* the position level of the hierarchy,
    shrinking each group toward the league curve instead of letting a thin group
    chase its own noise.
    """
    P = nbi + nb * (1 + n_groups)
    ii, dd = slice(0, nbi), slice(nbi, nbi + nb)

    def sl(block):
        if block == "init":
            return slice(0, nbi)
        if block == "league":
            return slice(nbi, nbi + nb)
        return slice(nbi + nb * (1 + block), nbi + nb * (2 + block))

    new_init = np.zeros_like(c_init)
    new_league = np.zeros_like(c_league)
    new_pos = np.zeros_like(c_pos)

    for s in range(M.shape[1]):
        A = np.zeros((P, P))
        b = np.zeros(P)
        for g in range(n_groups):
            Mg, ug = M[g, s], u[g, s]
            blocks = ["init", "league", g]
            parts = [ii, dd, dd]
            for br, pr in zip(blocks, parts):
                b[sl(br)] += ug[pr]
                for cbl, pc in zip(blocks, parts):
                    A[sl(br), sl(cbl)] += Mg[pr, pc]
        A[np.diag_indices(P)] += ridge
        for g in range(n_groups):
            sg = sl(g)
            A[sg, sg] += ridge_pos * np.eye(nb)
        step = np.linalg.solve(A, b)
        new_init[:, s] = c_init[:, s] + step[:nbi]
        new_league[:, s] = c_league[:, s] + step[nbi:nbi + nb]
        for g in range(n_groups):
            new_pos[g, :, s] = c_pos[g, :, s] + step[sl(g)]

    return new_init, new_league, new_pos


# ---------------------------------------------------------------------------
def shrinkage_factor(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """`B = R / (P + R)` -- the hierarchical shrinkage weight (§9).

    Its complement is the Kalman gain.  `test_filter_identities` asserts they
    are computed from the same `R`; that identity is the tripwire for the
    observation layer disagreeing with itself.
    """
    return R / (P + R)
