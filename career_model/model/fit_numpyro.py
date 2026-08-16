"""M6: full Bayes with the latent states as parameters (§5.2, §5.3).

The v1 path integrates the states out analytically through the Kalman recursion
and samples only the hyperparameters.  That is exact *given* the Gaussian
approximation of §3.3, and it is the reason v1 runs in minutes.  What it cannot
do is drop that approximation: the delta method is a second-order expansion, and
it is at its worst exactly where the data are thinnest -- a 40-possession
call-up, a player who took four threes all year -- which is where the intervals
matter most.

This module states the exact model instead: negative binomial on counts,
binomial on makes, Gaussian on log possessions, states sampled jointly with the
hyperparameters by NUTS.  Latent dimension is roughly `players x seasons x 14`,
about 560k for this panel -- feasible in NumPyro on a GPU, not in Stan.

Three things are worth flagging before running it.

**Non-centred parameterization is mandatory, not stylistic.**  `theta` is drawn
around `m_i` whose scale `Sigma_player` is itself being sampled; centred, that is
a funnel and NUTS will produce divergences that no amount of `target_accept`
will fix.  Every state innovation and every player-level deviation below is
written as a standard normal times a scale.

**The v1 mode is the initialisation.**  Starting NUTS from the KF mode and the
filtered states, rather than from the prior, is worth more than any tuning
parameter here.

**Acceptance is R-hat < 1.01, ESS > 400, zero divergences.**  A run that clears
CRPS but shows divergences has not fit this model; it has fit some other one.

NumPyro is an optional dependency and is deliberately not in the base
requirements -- v1 does not need it, and JAX wheels lag new Python releases.
"""

from __future__ import annotations

import numpy as np

from ..config import ACCURACY_PAIRS, ACCURACY_STATS, AVAIL_IDX, S, VOLUME_STATS
from .hierarchy import Q_RANK


def _require():
    try:
        import jax  # noqa: F401
        import numpyro  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError(
            "M6 needs numpyro and jax:  pip install 'numpyro[cpu]'  (or the CUDA "
            "wheel for GPU).  They are optional -- the v1 path in fit_kf.py has no "
            "such dependency and is what every other stage uses."
        ) from exc


def build_arrays(ds) -> dict:
    """Flatten the grid to the ragged-free layout NumPyro wants.

    One row per (player, career-season) that is inside a career span, with an
    `observed` mask.  Gaps stay in the array so the transition still fires for
    them; they are simply excluded from every likelihood term (§5.1).
    """
    grid = ds.grid
    N, T = grid.z.shape[0], grid.z.shape[1]
    counts = {c: np.nan_to_num(v) for c, v in grid.counts.items()}
    return {
        "n_players": N,
        "n_steps": T,
        "in_span": grid.in_span,
        "observed": grid.observed,
        "exposure": np.nan_to_num(grid.exposure, nan=1.0),
        "age": grid.age,
        "counts": counts,
        "attempt_mask": {s: (counts[ACCURACY_PAIRS[s][1]] > 0) & grid.observed
                         for s in ACCURACY_STATS},
        "X": ds.X,
        "pos_idx": ds.pos_idx,
        "age_basis": ds.age_basis(grid.age),
        "init_basis": ds.init_basis(grid.age[:, 0]),
        "gbm_offset": None,
    }


def model_fn(data, gbm_offset=None):
    """The exact §3 model.  No Gaussian approximation anywhere."""
    _require()
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    N, T = data["n_players"], data["n_steps"]
    p_x = data["X"].shape[1]
    nb = data["age_basis"].shape[-1]
    nbi = data["init_basis"].shape[-1]

    # ---- hyperparameters (§4.1) -------------------------------------------
    A = numpyro.sample("A", dist.Beta(5.0, 1.5).expand([S]))
    Lam = numpyro.sample("Lam", dist.Normal(0.0, 1.0).expand([S, Q_RANK]))
    psi = numpyro.sample("psi", dist.HalfNormal(1.0).expand([S]))
    phi = numpyro.sample("phi", dist.Gamma(2.0, 0.1).expand([len(VOLUME_STATS)]))
    sigma_poss = numpyro.sample("sigma_poss", dist.HalfNormal(1.0))

    beta = numpyro.sample("beta", dist.Normal(0.0, 5.0).expand([p_x, S]))
    sigma_m = numpyro.sample("sigma_m", dist.HalfNormal(1.0).expand([S]))
    L_m = numpyro.sample("L_m", dist.LKJCholesky(S, concentration=2.0))

    delta_league = numpyro.sample("delta_league", dist.Normal(0.0, 0.5).expand([nb, S]))
    tau_pos = numpyro.sample("tau_pos", dist.HalfNormal(0.2))
    c_init = numpyro.sample("c_init", dist.Normal(0.0, 0.5).expand([nbi, S]))

    # ---- non-centred player level -----------------------------------------
    # m_i = beta'x_i + f_GBM(x_i) + diag(sigma_m) L_m z_i, with z_i ~ N(0, I).
    # Written centred, sigma_m and z_i form a funnel and NUTS diverges.
    z_m = numpyro.sample("z_m", dist.Normal(0.0, 1.0).expand([N, S]))
    m = jnp.asarray(data["X"]) @ beta + (z_m @ (sigma_m[:, None] * L_m).T)
    if gbm_offset is not None:
        m = m + jnp.asarray(gbm_offset)        # §5.3, a fixed offset

    z_pos = numpyro.sample("z_pos", dist.Normal(0.0, 1.0).expand([3, nb, S]))
    delta_pos = tau_pos * z_pos
    coefs = delta_league[None] + delta_pos[jnp.asarray(data["pos_idx"])]

    Q_chol = jnp.linalg.cholesky(Lam @ Lam.T + jnp.diag(psi ** 2) + 1e-8 * jnp.eye(S))

    # ---- states, non-centred ----------------------------------------------
    z_eta = numpyro.sample("z_eta", dist.Normal(0.0, 1.0).expand([N, T, S]))
    offsets = jnp.einsum("ntk,nks->nts", jnp.asarray(data["age_basis"]), coefs)
    eta = jnp.einsum("ij,ntj->nti", Q_chol, z_eta)

    def step(theta_prev, t):
        theta = m + A * (theta_prev - m) + offsets[:, t] + eta[:, t]
        return theta, theta

    theta0 = m + jnp.asarray(data["init_basis"]) @ c_init + eta[:, 0]
    thetas = [theta0]
    for t in range(1, T):
        thetas.append(m + A * (thetas[-1] - m) + offsets[:, t - 1] + eta[:, t])
    theta = jnp.stack(thetas, axis=1)                       # (N, T, S)

    # ---- observation layers (§3.2), exact ---------------------------------
    E = jnp.asarray(data["exposure"])
    obs = jnp.asarray(data["observed"])

    for k, stat in enumerate(VOLUME_STATS):
        mu = E * jnp.exp(jnp.clip(theta[:, :, k], -12, 3))
        with numpyro.handlers.mask(mask=obs):
            numpyro.sample(f"y_{stat}",
                           dist.GammaPoisson(phi[k], phi[k] / jnp.maximum(mu, 1e-9)),
                           obs=jnp.asarray(data["counts"][stat]))

    for stat in ACCURACY_STATS:
        made_c, att_c = ACCURACY_PAIRS[stat]
        j = S - 1 - (len(ACCURACY_STATS) - ACCURACY_STATS.index(stat))
        att = jnp.asarray(data["counts"][att_c]).astype("int32")
        with numpyro.handlers.mask(mask=jnp.asarray(data["attempt_mask"][stat])):
            numpyro.sample(f"y_{stat}",
                           dist.Binomial(total_count=att, logits=theta[:, :, j]),
                           obs=jnp.asarray(data["counts"][made_c]).astype("int32"))

    with numpyro.handlers.mask(mask=obs):
        numpyro.sample("y_poss", dist.Normal(theta[:, :, AVAIL_IDX], sigma_poss),
                       obs=jnp.log(jnp.maximum(E, 1.0)))


def fit(ds, init_from=None, num_warmup: int = 800, num_samples: int = 800,
        num_chains: int = 4, target_accept: float = 0.9, seed: int = 0):
    """Run NUTS.  `init_from` should be the v1 `FittedModel` -- initialising at
    the KF mode is worth more here than any sampler tuning."""
    _require()
    import jax
    from numpyro.infer import MCMC, NUTS

    data = build_arrays(ds)
    gbm = None
    if init_from is not None and init_from.fit.params.gbm_offset is not None:
        gbm = init_from.fit.params.gbm_offset

    kernel = NUTS(model_fn, target_accept_prob=target_accept,
                  init_strategy=_init_strategy(init_from))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), data, gbm_offset=gbm)
    mcmc.print_summary()

    extra = mcmc.get_extra_fields(group_by_chain=False)
    divergences = int(np.sum(extra.get("diverging", np.zeros(1))))
    if divergences:
        print(f"\n!! {divergences} divergent transitions.  §5.2 M6 requires zero. "
              "Check that every state innovation and player deviation is "
              "non-centred before touching target_accept.")
    return mcmc


def _init_strategy(init_from):
    from numpyro.infer import init_to_median, init_to_value
    if init_from is None:
        return init_to_median()
    p = init_from.fit.params
    values = {
        "A": np.asarray(p.A),
        "Lam": np.asarray(p.Lam),
        "psi": np.sqrt(np.asarray(p.Psi)),
        "phi": np.asarray(p.phi),
        "sigma_poss": float(p.sigma_poss),
        "beta": np.asarray(p.beta),
        "delta_league": np.asarray(init_from.fit.delta_league),
        "c_init": np.asarray(init_from.fit.c_init),
    }
    return init_to_value(values=values)
