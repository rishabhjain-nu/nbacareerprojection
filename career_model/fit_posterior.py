"""Add hyperparameter posterior draws to an already-fitted artifact.

Separate from `pipeline` because the mode is what every stage check reads and
the posterior is what §6 step 1 needs -- and the second costs an order of
magnitude more wall clock than the first.

Two steps, and on this panel you want both:

    python -m career_model.fit_posterior --chains 1 --draws 3000 --burn 1500
    python -m career_model.fit_posterior --laplace 4000

Without either, the simulation still propagates state and sampling uncertainty
but treats `Q`, `A` and `phi_s` as known.  That is not a small omission:
parameter uncertainty is the component that does *not* shrink as a player
accumulates seasons, so leaving it out makes the veterans -- the players the
interface is most confident about -- the ones whose intervals are most wrong.

**What the chain is for, and what it is not.**  A random-walk Metropolis over
`d` correlated coordinates yields an effective sample size of roughly `n/d`
however well the proposal is tuned.  At `d = 84`, 3,000 draws buy an ESS around
11 on `Q`; clearing the §5.2 ESS > 400 bar needs ~35,000.  The `--chains` path
runs dispersed starts in separate processes to get a real between-chain R-hat
and more independent draws, but on this panel it is **memory**-bound, not
CPU-bound: four resident copies of the grid push the per-iteration cost from
0.16 s to 11 s, and it does not finish.

So the chain is used for what it estimates well -- the *shape* of the posterior
-- and `--laplace` then draws independently from `N(mode, Sigma_hat)`.  The mode
comes from the optimiser, so the location never depended on mixing at all.  See
`fit_kf.laplace_draws` for why that is sound here and what it gives up.

**Read diagnostics on identified quantities.**  `Lambda` is pinned down only up
to an orthogonal rotation, so R-hat on its entries tracks the chain sliding
along a rotation manifold rather than any convergence failure.
`fit_kf.identified_quantities` maps draws onto `Q`, `A`, `phi` and
`sigma_poss`, which are identified.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import ARTIFACT_DIR
from .model import fit_kf, hierarchy as hier
from .model.dataset import load as load_dataset
from .pipeline import ARTIFACT_PATH, load as load_model, save


def _run_chain(args):
    """One chain, in its own process."""
    path, seed, n_draws, burn, jitter = args
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "2"

    model = load_model(path)
    ds = load_dataset(max_season_year=model.train_cutoff)

    # Dispersed start: perturb the mode on the unconstrained scale.  Chains that
    # all start at the mode agree with each other by construction and make
    # R-hat look better than the sampler is.
    fit = model.fit
    if jitter > 0:
        rng = np.random.default_rng(1000 + seed)
        v0 = hier.pack(fit.params) + jitter * rng.standard_normal(hier.n_packed())
        fit = fit_kf.Fit(params=hier.unpack(v0, fit.params), c_init=fit.c_init,
                         delta_league=fit.delta_league, delta_pos=fit.delta_pos,
                         loglik=fit.loglik, history=fit.history)
    return fit_kf.sample_posterior(ds, fit, n_draws=n_draws, burn=burn,
                                   seed=seed, verbose=(seed == 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=6000, help="post-burn draws per chain")
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--jitter", type=float, default=0.15,
                    help="sd of the dispersion applied to each chain's start")
    ap.add_argument("--thin", type=int, default=0,
                    help="keep every k-th draw when storing (0 = auto from ESS)")
    ap.add_argument("--laplace", type=int, default=0, metavar="N",
                    help="replace the stored draws with N independent draws from "
                         "a Gaussian fitted to the existing chain, centred at the "
                         "mode; requires a posterior already in the artifact")
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()

    path = ARTIFACT_DIR / args.artifact if args.artifact else ARTIFACT_PATH

    if args.laplace:
        _laplace(path, args.laplace)
        return

    jobs = [(path, c, args.draws, args.burn, args.jitter if c else 0.0)
            for c in range(args.chains)]
    if args.chains == 1:
        chains = [_run_chain(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=args.chains) as pool:
            chains = list(pool.map(_run_chain, jobs))
    draws = np.stack(chains)                      # (chains, n, d)

    model = load_model(path)
    report(draws, model)

    # Store a thinned pool: the simulation samples at most a few dozen distinct
    # parameter sets per player, so keeping every draw costs memory for
    # resolution the Monte Carlo cannot use.
    flat = draws.reshape(-1, draws.shape[-1])
    thin = args.thin or max(1, len(flat) // 4000)
    model.posterior = flat[::thin]
    save(model, path)
    print(f"\nstored {len(model.posterior)} draws (thinned by {thin})")


def _laplace(path, n_draws: int) -> None:
    """Swap the stored chain for independent draws from a Gaussian fit to it."""
    model = load_model(path)
    if model.posterior is None or len(model.posterior) < 200:
        raise SystemExit("no chain in the artifact to fit a Gaussian to; run the "
                         "sampler first")
    chain = model.posterior
    mode = hier.pack(model.fit.params)

    print("=== before: the Metropolis chain ===")
    _summarise(chain[None], model)
    draws = fit_kf.laplace_draws(chain, mode, n_draws=n_draws)
    print(f"\n=== after: {n_draws} independent draws from N(mode, Sigma_hat) ===")
    _summarise(draws[None], model)
    print("\n  Independent by construction, so ESS equals the draw count for every\n"
          "  function of the parameters.  What is given up is non-Gaussian shape in\n"
          "  the posterior -- skew in the variance parameters, and the rotational\n"
          "  ridge in Lambda.  That is why this is an approximation and why M6\n"
          "  (`fit_numpyro`, gradient-based, states sampled jointly) remains the fix.")

    model.posterior = draws
    save(model, path)
    print(f"\nstored {len(draws)} independent draws")


def _summarise(draws: np.ndarray, model) -> None:
    """R-hat is reported split-chain (four pseudo-chains from one run).

    Passing a single chain to the multi-chain path would make the between-chain
    term zero and print a reassuring 1.000 next to an ESS of 11, which is worse
    than printing nothing.
    """
    base = model.fit.params
    for name, arr in fit_kf.identified_quantities(draws, base).items():
        flat = arr.reshape(-1, arr.shape[-1])
        r, e = fit_kf.rhat_ess(flat, n_chains=4)
        print(f"  {name:<12s} split R-hat max {np.nanmax(r):.3f}   "
              f"ESS min {np.nanmin(e):6.0f}  median {np.nanmedian(e):6.0f}  "
              f"(of {len(flat)} draws)")


def report(draws: np.ndarray, model) -> None:
    base = model.fit.params
    print(f"\n=== posterior over {draws.shape[-1]} hyperparameters: "
          f"{draws.shape[0]} chains x {draws.shape[1]} draws ===")

    print("\n-- identified quantities (the ones that mean anything) --")
    for name, arr in fit_kf.identified_quantities(draws, base).items():
        r, e = fit_kf.rhat_ess(arr)
        print(f"  {name:<12s} R-hat max {np.nanmax(r):.3f}  median {np.nanmedian(r):.3f}"
              f"   |  ESS min {np.nanmin(e):6.0f}  median {np.nanmedian(e):6.0f}")

    r, e = fit_kf.rhat_ess(draws)
    print("\n-- raw packed vector, for reference --")
    print(f"  R-hat < 1.01: {int(np.nansum(r < 1.01))}/{len(r)}   max {np.nanmax(r):.3f}")
    print(f"  ESS   > 400 : {int(np.nansum(e > 400))}/{len(e)}    min {np.nanmin(e):.0f}")

    print(
        "\n  `Lambda` is identified only up to an orthogonal rotation -- (Lambda R)\n"
        "  and Lambda give the same Q -- so R-hat on its individual entries tracks\n"
        "  the chain wandering a rotation manifold, not a convergence failure.\n"
        "  Read the Q, A, phi and sigma_poss rows above instead.  Where those\n"
        "  still fall short of R-hat < 1.01 and ESS > 400, the honest read is\n"
        "  that the parameter component of the interval is noisier than the state\n"
        "  and sampling components, which are exact.  The M6 fix is gradient-based\n"
        "  sampling in `fit_numpyro`, not a longer random walk.")


if __name__ == "__main__":
    main()
