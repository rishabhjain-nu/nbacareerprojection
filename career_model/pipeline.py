"""End-to-end fit: panel -> staged state-space fit -> GBM prior -> hazard.

Run it with `python -m career_model.pipeline`.  Everything downstream --
simulation, precompute, validation -- loads the single artifact this writes.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass

import numpy as np

from .config import ARTIFACT_DIR, MODEL_VERSION, S, STATE_NAMES
from .model import fit_kf, gbm_prior, hazard as hazard_mod
from .model import observations as obs
from .model import state_space as ss
from .model.dataset import Dataset, load as load_dataset

ARTIFACT_PATH = ARTIFACT_DIR / "career_model.pkl"


@dataclass
class FittedModel:
    fit: fit_kf.Fit
    gbm: gbm_prior.GBMPrior | None
    hazard: hazard_mod.Hazard
    posterior: np.ndarray | None          # (n_draws, n_packed)
    age_basis: object
    init_basis: object
    scaler: dict
    x_names: list
    train_cutoff: int
    model_version: str = MODEL_VERSION
    # Within-career absence model (deviation #4): P(plays | career active).
    # Defaulted so a pre-fix pickle still loads -- `None` means the projection
    # falls back to "career-alive implies plays", the old behaviour.
    absence: hazard_mod.Hazard | None = None

    def offsets_for(self, grid, pos_idx):
        """Transition offset.  Increment form (LEVEL_PARAM False): `delta(age)`."""
        coefs = self.fit.delta_league[None] + self.fit.delta_pos[pos_idx]
        if self.fit.player_scale is not None:
            coefs = coefs * np.asarray(self.fit.player_scale)[:, None, None]
        cur = np.einsum("ntk,nks->nts", self.age_basis(grid.age), coefs)
        if not fit_kf.LEVEL_PARAM:
            return cur
        nxt = np.einsum("ntk,nks->nts", self.age_basis(grid.age + 1.0), coefs)
        return nxt - np.asarray(self.fit.params.A)[None, None, :] * cur


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def filtered_states(ds: Dataset, fit: fit_kf.Fit, m_prior_scale: float = 1.0,
                    decouple_m: bool = False, decouple_scale: float = 5.0):
    """Run the full filter and return `theta_{t|t}`, `P_{t|t}` and `E[m_i]`.

    `m_prior_scale` (S3-A "diffuse-m") inflates the m-block prior variance,
    keeping m's prior mean at f_GBM.  `decouple_m` (S3 closure) is the
    structurally-different formulation: m's prior *mean* is set to the
    position-group average (f_GBM-independent), so the pre-NBA prediction
    informs only theta_0 and stops being the permanent reversion destination;
    `decouple_scale` sets how freely NBA evidence then moves m.
    """
    R = obs.refresh_R(ds.grid, fit.params.phi, fit.params.acc_floor,
                          fit.params.sigma_poss, fit.params.sigma_poss_inj,
                          fit.params.injury_infl)
    offsets = fit_kf.offsets_from(ds, fit.delta_league, fit.delta_pos,
                                  fit.player_scale, fit_kf._offset_A(fit.params))
    m_prior_mean = None
    if decouple_m:
        _, b = ss.prior_mean(fit.params, ds.X, ds.grid.age[:, 0],
                             ds.init_basis, fit.c_init)      # = f_GBM + beta x
        pos = ds.pos_idx
        m_prior_mean = np.zeros_like(b)
        for gidx in np.unique(pos):                          # position-group mean
            m_prior_mean[pos == gidx] = b[pos == gidx].mean(axis=0)
        m_prior_scale = decouple_scale
    return ss.run_filter(ds.grid, fit.params, ds.X, R, offsets,
                         ds.init_basis, fit.c_init, keep_states=True,
                         m_prior_scale=m_prior_scale, m_prior_mean=m_prior_mean)


def fit_everything(max_season_year: int | None = None, n_outer: int = 3,
                   maxiter_diag: int = 200, maxiter_full: int = 60,
                   use_gbm: bool = True, n_posterior: int = 0,
                   verbose: bool = True) -> tuple[FittedModel, Dataset]:
    t0 = time.time()
    ds = load_dataset(max_season_year=max_season_year)
    _log(f"dataset: {ds.n_players} players, {int(ds.grid.observed.sum())} seasons, "
         f"cutoff {max_season_year or 'none'}")

    fit = fit_kf.fit(ds, n_outer=n_outer, maxiter_diag=maxiter_diag,
                     maxiter_full=maxiter_full, verbose=verbose)

    gbm = None
    if use_gbm:
        res = filtered_states(ds, fit)
        m_hat = res.x2[:, ds.grid.z.shape[1], :]
        oof, gbm = gbm_prior.fit_offsets(ds, m_hat)
        fit.params = fit.params.copy(gbm_offset=oof)
        # With the trees carrying the cross-sectional signal, re-solve the linear
        # terms and re-run EM so `beta` explains only the residual.
        R = obs.refresh_R(ds.grid, fit.params.phi, fit.params.acc_floor,
                          fit.params.sigma_poss, fit.params.sigma_poss_inj,
                          fit.params.injury_infl)
        # `c_lin` is the profile's own (n_basis, S) block; `fit.c_init` is the
        # *evaluated* (N, S) per-player initial offset.  They are different
        # objects with different shapes -- passing the second where the first
        # belongs is what broke this stage when the level parameterization
        # landed, and it failed after the fit had already run.
        nbi = ds.init_basis.size
        c_lin = np.zeros((nbi, S))
        c_lin, fit.delta_league, fit.delta_pos = ss.profile_linear_diag(
            ds.grid, fit.params, ds.X, R, ds.age_basis, ds.init_basis, ds.pos_idx,
            fit.delta_league, fit.delta_pos, c_lin, level_param=fit_kf.LEVEL_PARAM)
        fit.player_scale = np.ones(ds.n_players)
        fit.c_init = fit_kf.init_offset_from(ds, c_lin)
        offsets = fit_kf.offsets_from(ds, fit.delta_league, fit.delta_pos,
                                      fit.player_scale, fit_kf._offset_A(fit.params))
        fit.params = fit_kf.update_prior(ds, fit.params, R, offsets, fit.c_init)
        fit.loglik = ss.run_filter(ds.grid, fit.params, ds.X, R, offsets,
                                   ds.init_basis, fit.c_init).loglik
        _log(f"after GBM prior: loglik {fit.loglik:,.0f}")

    res = filtered_states(ds, fit)
    # Age x quality interaction (fix 6): validated by
    # validate/ab_availability.py --compare hazard at cutoffs 2016 and 2018.
    haz = hazard_mod.fit(ds, res.x1, verbose=verbose, interaction=True)
    # Within-career absence (deviation #4): P(plays | career active), composed
    # with `haz` at projection time so appearance = continuation x plays.
    absence = hazard_mod.fit_absence(ds, res.x1, verbose=verbose)

    posterior = None
    if n_posterior:
        posterior = fit_kf.sample_posterior(ds, fit, n_draws=n_posterior,
                                            burn=max(100, n_posterior // 2),
                                            verbose=verbose)

    model = FittedModel(
        fit=fit, gbm=gbm, hazard=haz, absence=absence, posterior=posterior,
        age_basis=ds.age_basis, init_basis=ds.init_basis, scaler=ds.scaler,
        x_names=ds.x_names,
        train_cutoff=int(max_season_year or ds.panel["season_year"].max()),
    )
    _log(f"done in {time.time() - t0:.0f}s")
    return model, ds


def report(model: FittedModel, ds: Dataset) -> None:
    """The stage checks of §5.2, printed."""
    p = model.fit.params
    print("\n--- persistence A (per-stat, §3.4: expect assists sticky, 3P% less so) ---")
    for s in np.argsort(-p.A):
        print(f"  {STATE_NAMES[s]:>9s}  {p.A[s]:.3f}")

    Q = p.Q()
    d = np.sqrt(np.diag(Q))
    corr = Q / np.outer(d, d)
    print("\n--- Q factor loadings (§5.2 M3: expect usage/role and size) ---")
    for s in range(S):
        print(f"  {STATE_NAMES[s]:>9s}  " + "  ".join(f"{p.Lam[s, k]:+.3f}"
                                                     for k in range(p.Lam.shape[1])))
    print("\n--- strongest process-noise correlations ---")
    pairs = [(abs(corr[i, j]), i, j) for i in range(S) for j in range(i + 1, S)]
    for _, i, j in sorted(pairs, reverse=True)[:8]:
        print(f"  {STATE_NAMES[i]:>9s} ~ {STATE_NAMES[j]:<9s} {corr[i, j]:+.3f}")

    print("\n--- hazard gamma (coupling to the filtered state) ---")
    for n, c in sorted(zip(model.hazard.names, model.hazard.coef),
                       key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {n:>18s}  {c:+.3f}")


def save(model: FittedModel, path=ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    _log(f"wrote {path}")


class _Unpickler(pickle.Unpickler):
    """Resolve classes pickled while this module was running as `__main__`.

    `python -m career_model.pipeline` makes the dataclasses' `__module__` be
    `__main__`, and a later process cannot find them there.  Remapping on load
    keeps artifacts readable regardless of how the fit was launched.
    """

    _FALLBACKS = ("career_model.pipeline", "career_model.model.fit_kf",
                  "career_model.model.hierarchy", "career_model.model.hazard",
                  "career_model.model.gbm_prior", "career_model.model.aging",
                  "career_model.model.observations")

    def find_class(self, module, name):
        if module == "__main__":
            import importlib
            for mod in self._FALLBACKS:
                try:
                    return getattr(importlib.import_module(mod), name)
                except AttributeError:
                    continue
        return super().find_class(module, name)


def load(path=ARTIFACT_PATH) -> FittedModel:
    with open(path, "rb") as fh:
        return _Unpickler(fh).load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=None)
    ap.add_argument("--outer", type=int, default=3)
    ap.add_argument("--maxiter-diag", type=int, default=200)
    ap.add_argument("--maxiter-full", type=int, default=60)
    ap.add_argument("--posterior", type=int, default=0,
                    help="posterior draws over hyperparameters (0 = mode only)")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, ds = fit_everything(
        max_season_year=args.cutoff, n_outer=args.outer,
        maxiter_diag=args.maxiter_diag, maxiter_full=args.maxiter_full,
        use_gbm=not args.no_gbm, n_posterior=args.posterior)
    report(model, ds)
    save(model, ARTIFACT_DIR / args.out if args.out else ARTIFACT_PATH)


if __name__ == "__main__":
    main()
