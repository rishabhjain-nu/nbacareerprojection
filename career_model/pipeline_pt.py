"""End-to-end fit for the permanent/transient model (§3.4 v2).

Mirrors `pipeline.fit_everything`: staged PT fit, GBM prior on the initial level
`ell_0`, hazard coupled to the filtered `theta = ell + u`.  Produces a
`FittedModelPT` that the PT projection and precompute read.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

import numpy as np

from .config import ARTIFACT_DIR, MODEL_VERSION
from .model import fit_pt, gbm_prior, hazard as hazard_mod
from .model import observations as obs
from .model import state_space_pt as sspt
from .model.dataset import load as load_dataset

ARTIFACT_PT = ARTIFACT_DIR / "career_model_pt.pkl"


@dataclass
class FittedModelPT:
    fit: fit_pt.FitPT
    gbm: gbm_prior.GBMPrior | None
    hazard: hazard_mod.Hazard
    posterior: np.ndarray | None
    age_basis: object
    init_basis: object
    scaler: dict
    x_names: list
    train_cutoff: int
    model_version: str = MODEL_VERSION + "-pt"


def _log(msg):
    print(f"[pipeline-pt] {msg}", flush=True)


def _R(ds, p):
    return obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss,
                         p.sigma_poss_inj, p.injury_infl)


def filtered_states_pt(ds, fit):
    R = _R(ds, fit.params)
    drift = fit_pt.drift_offsets(ds, fit.delta_league, fit.delta_pos)
    return sspt.run_filter_pt(ds.grid, fit.params, ds.X, R, drift,
                              ds.init_basis, fit.c_init, keep_states=True)


def fit_everything_pt(max_season_year=None, n_outer=3, maxiter_diag=150,
                      maxiter_full=40, use_gbm=True, verbose=True):
    t0 = time.time()
    ds = load_dataset(max_season_year=max_season_year)
    _log(f"dataset: {ds.n_players} players, cutoff {max_season_year or 'none'}")
    fit = fit_pt.fit(ds, n_outer=n_outer, maxiter_diag=maxiter_diag,
                     maxiter_full=maxiter_full, verbose=verbose)

    gbm = None
    if use_gbm:
        # The GBM predicts the *debut level* ell_0 from draft/college -- the
        # natural cross-sectional target now that there is no career-mean.
        res = filtered_states_pt(ds, fit)
        ell0_hat = res.ell[:, 0, :]
        oof, gbm = gbm_prior.fit_offsets(ds, ell0_hat)
        fit.params = fit.params.copy(gbm_offset=oof)
        R = _R(ds, fit.params)
        c_lin, fit.delta_league, fit.delta_pos = sspt.profile_drift_diag(
            ds.grid, fit.params, ds.X, R, ds.age_basis, ds.init_basis, ds.pos_idx,
            fit.delta_league, fit.delta_pos, np.zeros((ds.init_basis.size, ell0_hat.shape[1])))
        fit.c_init_coefs = c_lin
        fit.c_init = fit_pt.init_offset(ds, c_lin)
        drift = fit_pt.drift_offsets(ds, fit.delta_league, fit.delta_pos)
        fit.params = fit_pt.update_prior_pt(ds, fit.params, R, drift, fit.c_init)
        fit.loglik = sspt.run_filter_pt(ds.grid, fit.params, ds.X, R, drift,
                                        ds.init_basis, fit.c_init).loglik
        _log(f"after GBM prior: loglik {fit.loglik:,.0f}")

    res = filtered_states_pt(ds, fit)
    theta = res.ell[:, :res.ell.shape[1] - 1, :] + res.u[:, :res.u.shape[1] - 1, :]
    haz = hazard_mod.fit(ds, theta, verbose=verbose)

    model = FittedModelPT(fit=fit, gbm=gbm, hazard=haz, posterior=None,
                          age_basis=ds.age_basis, init_basis=ds.init_basis,
                          scaler=ds.scaler, x_names=ds.x_names,
                          train_cutoff=int(max_season_year or ds.panel["season_year"].max()))
    _log(f"done in {time.time() - t0:.0f}s")
    return model, ds


def save(model, path=ARTIFACT_PT):
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    _log(f"wrote {path}")


class _UnpicklerPT(pickle.Unpickler):
    _FALLBACKS = ("career_model.pipeline_pt", "career_model.model.fit_pt",
                  "career_model.model.hierarchy_pt", "career_model.model.hazard",
                  "career_model.model.gbm_prior")

    def find_class(self, module, name):
        if module == "__main__":
            import importlib
            for mod in self._FALLBACKS:
                try:
                    return getattr(importlib.import_module(mod), name)
                except AttributeError:
                    continue
        return super().find_class(module, name)


def load(path=ARTIFACT_PT):
    with open(path, "rb") as fh:
        return _UnpicklerPT(fh).load()
