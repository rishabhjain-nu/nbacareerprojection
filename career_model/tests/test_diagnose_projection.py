"""The projection decomposition must reconstruct what the interface displays.

Two guarantees:

  1. **Additive latent exactness.**  The reported availability contributions
     (start + mean-reversion + aging + age-by-quality) sum to the deterministic
     next-season log-possessions, and that centre equals the simulator's own
     mean first-step latent availability within Monte Carlo error -- i.e. the
     decomposition mirrors the code path, not a parallel re-derivation.

  2. **Displayed-projection reconstruction.**  The tool's observable year-1
     medians (MPG, GP, PPG, RPG, APG, pts/100) equal the shipped store's
     `payload.json` for SGA, Zubac and Tatum within tolerance, because both come
     from `project.simulate` at the shipping settings (n_draws=2000, seed=0).

The real fitted model is loaded once; if it or the store is absent the store
half is skipped so the suite still runs on a fresh checkout.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from career_model.config import ARTIFACT_DIR, AVAIL_IDX, PROJECTION_DIR

pytestmark = pytest.mark.filterwarnings("ignore")

EXAMPLES = {"SGA": 1628983, "Zubac": 1627826, "Tatum": 1628369}


@pytest.fixture(scope="module")
def prep():
    if not (ARTIFACT_DIR / "career_model.pkl").exists():
        pytest.skip("no fitted model artifact")
    from career_model import diagnose_projection as dp
    return dp, *dp._prep()


def test_additive_latent_decomposition_matches_simulator(prep):
    dp, model, ds, filt, ms, ib, aq, rc = prep
    from career_model.simulate import project
    grid = ds.grid
    for pid in EXAMPLES.values():
        r = dp._row_of(grid, pid)
        d = dp.decompose(model, ds, filt, r, ms, ib, aq, n_draws=4000, role_model=rc)
        av = d["availability_decomposition_log_possessions"]
        # exact additivity: check_sum and the total are both round(next_av)
        assert abs(av["check_sum"] - av["8_deterministic_next_log_poss"]) < 1e-9
        # the displayed contributions sum to the total up to 3-decimal rounding
        parts = (av["3_effective_start"] + av["5_mean_reversion_contribution"]
                 + av["6_aging_contribution"] + av["7_age_by_quality_contribution"])
        assert abs(parts - av["8_deterministic_next_log_poss"]) < 3e-3
        # the deterministic centre equals the simulator's mean latent path
        proj = project.simulate(model, ds, r, filt, n_draws=4000, seed=0,
                                minutes_split=ms, injury_beta=ib, avail_quality=aq)
        mean_latent = float(np.mean(proj.theta[:, 0, AVAIL_IDX]))
        assert abs(mean_latent - av["8_deterministic_next_log_poss"]) < 0.03, pid


def test_reconstructs_displayed_projection(prep):
    dp, model, ds, filt, ms, ib, aq, rc = prep
    grid = ds.grid
    pairs = [("14_MPG", "minutes_per_game"), ("14_GP", "games"),
             ("14_PPG", "pts_per_game"), ("14_RPG", "reb_per_game"),
             ("14_APG", "ast_per_game"), ("13_pts_per100", "pts_per100")]
    checked = 0
    for name, pid in EXAMPLES.items():
        payload = PROJECTION_DIR / str(pid) / "payload.json"
        if not payload.exists():
            continue
        p = json.loads(payload.read_text())
        s = p["seasons"]
        d = dp.decompose(model, ds, filt, dp._row_of(grid, pid), ms, ib, aq,
                         n_draws=2000, role_model=rc)
        for dk, pk in pairs:
            store = s.get(pk)
            if not store:
                continue
            got = d["observable_medians_UI"][dk]
            want = store["p50"][0]
            # tolerance: 1% relative or 0.15 absolute, whichever is larger
            tol = max(0.01 * abs(want), 0.15)
            assert abs(got - want) <= tol, f"{name} {pk}: diagnose {got} vs store {want}"
        checked += 1
    if checked == 0:
        pytest.skip("no stored projections to reconstruct")
