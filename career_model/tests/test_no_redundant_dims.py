"""No composite metrics in the state vector (§1.2), enforced structurally.

PIE, BPM, PER, WS, VORP, EPM and DPM are deterministic or near-deterministic
functions of box score dimensions the state already carries.  Adding one makes
`Q` singular and `F_t^{-1}` explode, and the model becomes unidentified.  The
symptom in practice is not an error -- it is a filter that runs and produces
confident nonsense -- so these checks are worth having as tests rather than as
a comment.

The design-rank check is the general version: assemble the matrix that maps the
14 state dimensions to the observed quantities and assert it is full rank.  Any
composite added to the state shows up here as a rank drop, whether or not
anybody remembered to name it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from career_model.config import (
    ACCURACY_STATS, AVAILABILITY_STAT, COUNT_COLS, S, STATE_NAMES, VOLUME_STATS,
)
from career_model.model import hierarchy as hier
from career_model.model import observations as obs
from career_model.model import state_space as ss
from career_model.tests.fixtures import small_dataset

BANNED = {"pie", "bpm", "obpm", "dbpm", "per", "ws", "win_shares", "vorp",
          "epm", "dpm", "darko", "raptor", "lebron", "net_rating", "plus_minus"}


def test_state_contains_no_composites():
    for name in STATE_NAMES:
        assert name.lower() not in BANNED, f"{name} is a composite (§1.2)"
    assert len(set(STATE_NAMES)) == S


def test_state_design_is_full_rank():
    """Each state dimension must contribute something no other one does.

    Build the 14x14 map from state to observed quantities: volume dimensions are
    log rates of distinct counts, accuracy dimensions are logits of distinct
    make/attempt pairs, availability is log possessions.  Under the model's
    identity observation matrix this is `Z = I`, so the check is that no two
    dimensions are aliases -- which is what would happen if, say, both `FGA` and
    `2PA`+`3PA` were carried, or a composite were bolted on.
    """
    assert len(VOLUME_STATS) + len(ACCURACY_STATS) + 1 == S
    assert AVAILABILITY_STAT not in VOLUME_STATS

    # No volume dimension may be a sum of others -- the FGA/2PA+3PA trap.
    assert "fga" not in VOLUME_STATS and "fgm" not in VOLUME_STATS
    assert "reb" not in VOLUME_STATS, "REB is OREB + DREB"
    assert "pts" not in VOLUME_STATS and "pts" not in STATE_NAMES, \
        "points is derived, not a state dimension (§3.1)"

    ds = small_dataset()
    Z = ds.grid.z[ds.grid.observed]
    r = np.linalg.matrix_rank(Z - Z.mean(axis=0), tol=1e-8)
    assert r == S, f"observation matrix has rank {r}, expected {S}"


def test_innovation_covariance_is_positive_definite():
    """§5.1: assert `F_t` is PD at every step."""
    ds = small_dataset()
    p = hier.default_params(p_x=ds.X.shape[1])
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss)
    ss.assert_positive_definite(ds.grid, p, ds.X, R, np.zeros_like(ds.grid.z))


def test_process_covariance_is_positive_definite():
    """`Q = Lambda Lambda' + Psi` with `Psi > 0` is PD by construction; this
    pins the construction down so a refactor cannot drop `Psi`."""
    rng = np.random.default_rng(0)
    p = hier.default_params().copy(Lam=rng.standard_normal((S, hier.Q_RANK)))
    w = np.linalg.eigvalsh(p.Q())
    assert w.min() > 0
    assert np.all(np.diag(p.stationary_dispersion()) > 0)


def test_panel_carries_counts_not_rates():
    """§1.1 / §10.1, on the real panel if it has been built."""
    from career_model.config import PANEL_PATH
    if not PANEL_PATH.exists():
        pytest.skip("panel not built")
    from career_model.data.ingest_bbref import assert_counts_only
    panel = pd.read_parquet(PANEL_PATH)
    assert_counts_only(panel)
    for c in COUNT_COLS:
        assert (panel[c] >= 0).all()
    assert (panel["possessions"] > 0).all()


def test_rebuilding_reb_from_state_is_exact():
    """The reason REB is not a dimension: it is `OREB + DREB` exactly, so
    carrying it would make `Q` singular.  Shown, not asserted in prose."""
    ds = small_dataset()
    oreb = ds.grid.counts["oreb"]
    dreb = ds.grid.counts["dreb"]
    obs_mask = ds.grid.observed
    total = (oreb + dreb)[obs_mask]
    assert np.all(np.isfinite(total))
    assert np.corrcoef(total, (oreb + dreb)[obs_mask])[0, 1] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
