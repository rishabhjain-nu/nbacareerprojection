"""Part-B audit: is the parameter-uncertainty approximation sound?

The shipping artifact carries `posterior` = 4,000 draws produced by
`fit_kf.laplace_draws` -- independent draws from `N(mode, Sigma_hat)` where
`Sigma_hat` is the shrunk sample covariance of a random-walk Metropolis chain.
This module does not re-run inference; it *audits* the draws already shipped:

  1. every transformed draw satisfies the model's hard constraints
     (0 < A < 1, positive observation scales, positive-definite Q, valid hazard
     probabilities);
  2. the identified quantities (A, Q, phi, sigma_poss, aging, hazard) have
     believable marginal spread;
  3. the ESS is reported *honestly* -- Laplace draws are independent by
     construction, so ESS == n for functions of the parameters, but that is a
     property of the *sampler*, not evidence the Gaussian shape is correct.

What this audit deliberately does NOT claim: it does not re-derive the four
alternative approximations (Hessian-Laplace, career bootstrap, reduced-model
NUTS) as fresh inference runs -- those are heavier than one session's compute
budget.  It states which of the four the shipping path already realises
(1 == 2 here, because the stored draws *are* the Gaussian/Laplace approximation)
and checks that that realisation is internally valid.

Run:  python -m career_model.validate.param_uncertainty_audit
"""

from __future__ import annotations

import json

import numpy as np

from ..config import OUTPUT_DIR, STATE_NAMES
from ..model import fit_kf
from ..model.dataset import load as load_dataset
from ..pipeline import ARTIFACT_PATH, load as load_model


def _pct(x):
    return {"p05": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95))}


def audit(model, ds) -> dict:
    draws = model.posterior
    if draws is None:
        raise SystemExit("artifact has no posterior draws; run fit_posterior first")
    n, d = draws.shape
    iq = fit_kf.identified_quantities(draws, model.fit.params)
    A, Q, phi, sig = iq["A"], iq["Q"], iq["phi"], iq["sigma_poss"]

    report: dict = {"n_draws": int(n), "n_packed": int(d)}

    # ---- constraint satisfaction on transformed draws --------------------
    S = len(STATE_NAMES)
    iu = np.triu_indices(S)
    Qmin_eig = np.empty(n)
    for i in range(n):
        M = np.zeros((S, S))
        M[iu] = Q[i]
        M = M + M.T - np.diag(np.diag(M))
        Qmin_eig[i] = np.linalg.eigvalsh(M)[0]
    checks = {
        "A_in_open_unit_interval": bool(np.all((A > 0) & (A < 1))),
        "A_min": float(A.min()), "A_max": float(A.max()),
        "sigma_poss_positive": bool(np.all(sig > 0)),
        "sigma_poss_min": float(sig.min()),
        "phi_finite": bool(np.all(np.isfinite(phi))),
        "Q_positive_definite": bool(np.all(Qmin_eig > 0)),
        "Q_min_eigenvalue": float(Qmin_eig.min()),
    }

    # hazard probabilities: draw coefficients from the hazard's own covariance
    # and evaluate p_survive over the training design; must stay in [0, 1].
    hz = model.hazard
    rng = np.random.default_rng(0)
    res = fit_kf  # noqa: F841
    # reconstruct the standardized state score design the hazard was fit on
    from ..pipeline import filtered_states
    filt = filtered_states(ds, model.fit)
    theta = filt.x1[np.arange(ds.grid.n_players), ds.grid.last_index.clip(min=0), :]
    age = ds.grid.age[np.arange(ds.grid.n_players), ds.grid.last_index.clip(min=0)]
    p_all = []
    for _ in range(200):
        c = hz.draw_coef(rng)
        p = hz.p_survive(age, theta, coef=c)
        p_all.append(p)
    p_all = np.asarray(p_all)
    checks["hazard_p_in_unit_interval"] = bool(np.all((p_all >= 0) & (p_all <= 1)))
    checks["hazard_p_min"] = float(p_all.min())
    checks["hazard_p_max"] = float(p_all.max())

    # absence (P(plays | active)) hazard, same check
    if model.absence is not None:
        pa = []
        for _ in range(200):
            c = model.absence.draw_coef(rng)
            pa.append(model.absence.p_survive(age, theta, coef=c))
        pa = np.asarray(pa)
        checks["absence_p_in_unit_interval"] = bool(np.all((pa >= 0) & (pa <= 1)))

    report["constraint_checks"] = checks
    report["all_constraints_pass"] = bool(
        checks["A_in_open_unit_interval"] and checks["sigma_poss_positive"]
        and checks["phi_finite"] and checks["Q_positive_definite"]
        and checks["hazard_p_in_unit_interval"])

    # ---- marginal spread of identified quantities ------------------------
    report["identified_spread"] = {
        "A_cv": {STATE_NAMES[i]: float(A[:, i].std() / abs(A[:, i].mean()))
                 for i in range(S)},
        "Q_diag_rel_sd": {STATE_NAMES[i]: float(
            Q[:, np.where((iu[0] == i) & (iu[1] == i))[0][0]].std()
            / abs(Q[:, np.where((iu[0] == i) & (iu[1] == i))[0][0]].mean()))
            for i in range(S)},
        "sigma_poss": _pct(sig.ravel()),
    }

    # ---- ESS honesty -----------------------------------------------------
    # For a *function* g(theta), ESS of independent Laplace draws is n.  We
    # report that, and separately the between-arm caveat: this is not the ESS
    # of a converged Markov chain on the packed vector.
    report["ess_statement"] = {
        "sampler": "independent Laplace draws N(mode, Sigma_hat)",
        "ess_for_parameter_functions": int(n),
        "caveat": ("ESS == n because draws are i.i.d. from the fitted Gaussian, "
                   "NOT because a Markov chain reached that ESS; a random-walk "
                   "Metropolis over 84 correlated coords delivers ESS ~ n/d ~ "
                   f"{n // d}. What is approximated away is non-Gaussian shape "
                   "(skew in variance params, the Lambda rotation ridge)."),
    }

    # ---- which of the four approximations the shipping path realises -----
    report["method_map"] = {
        "1_gaussian_covariance": "SHIPPED -- posterior draws ARE N(mode, Sigma_hat)",
        "2_hessian_laplace": ("equivalent here: Sigma_hat is the shrunk chain "
                              "covariance, a consistent estimate of the Hessian "
                              "inverse near the mode (Bernstein-von Mises)"),
        "3_career_bootstrap": "NOT RE-RUN this session (compute); prior sessions' A/B stands",
        "4_reduced_model_nuts": "NOT RE-RUN this session (memory-bound on this panel)",
    }
    return report


def main() -> None:
    model = load_model(ARTIFACT_PATH)
    ds = load_dataset(max_season_year=model.train_cutoff)
    rep = audit(model, ds)
    out = OUTPUT_DIR / "param_uncertainty_audit.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in
                      ("n_draws", "all_constraints_pass", "constraint_checks")}, indent=2))
    print(f"[audit] wrote {out}")


if __name__ == "__main__":
    main()
