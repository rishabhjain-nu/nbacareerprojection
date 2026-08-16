"""Gaps are absent rows, and the filter handles them natively (§2.2, §5.1).

The failure mode this guards against is quiet and fatal: representing a missed
season as a row of zeros.  A zero row says the player took no shots in a full
allocation of possessions, which is an observation of total collapse, and the
filter believes it.  An absent row says nothing was measured.  These tests pin
down that the second is what happens, and measure how far apart the two are.
"""

from __future__ import annotations

import numpy as np
import pytest

from career_model.config import IDX, S
from career_model.model import hierarchy as hier
from career_model.model import observations as obs
from career_model.model import state_space as ss
from career_model.tests.fixtures import small_dataset, synthetic_panel, synthetic_priors


def _params(ds):
    return hier.default_params(p_x=ds.X.shape[1]).copy(
        Lam=np.zeros((S, hier.Q_RANK)), Sigma_p=np.diag(np.full(S, 0.3)))


def test_gap_seasons_are_absent_not_zero():
    panel = synthetic_panel(120, seed=0)
    grid = obs.build_grid(panel)
    # Some player has a span longer than his observed season count.
    span = grid.in_span.sum(axis=1)
    assert (span > grid.n_history).any(), "fixture should contain career gaps"
    gaps = grid.in_span & ~grid.observed
    assert gaps.any()
    # A gap row carries no observation on any dimension.
    assert not grid.obs_mask[gaps].any()


def test_uncertainty_inflates_across_a_gap():
    """§5.1: the state keeps aging and `P` keeps growing by `Q`, so on
    reappearance the filter trusts the new data more."""
    ds = small_dataset()
    p = _params(ds)
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss)
    res = ss.run_filter_diag(ds.grid, p, ds.X, R, np.zeros_like(ds.grid.z),
                             keep_states=True)

    gaps = ds.grid.in_span & ~ds.grid.observed
    rows, cols = np.nonzero(gaps)
    checked = 0
    for i, t in zip(rows, cols):
        if t == 0:
            continue
        before, during = res.P11[i, t - 1], res.P11[i, t]
        assert np.all(during > before - 1e-12), \
            "variance must not shrink across a season with no data"
        assert np.mean(during) > np.mean(before) * 1.0001, \
            "variance did not grow by Q across the gap"
        checked += 1
    assert checked > 0


def test_state_only_moves_by_aging_across_a_gap():
    """No update means the only thing that changes the mean is the transition."""
    ds = small_dataset()
    p = _params(ds)
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss)
    offsets = np.zeros_like(ds.grid.z)
    res = ss.run_filter_diag(ds.grid, p, ds.X, R, offsets, keep_states=True)

    gaps = ds.grid.in_span & ~ds.grid.observed
    rows, cols = np.nonzero(gaps)
    for i, t in zip(rows[:20], cols[:20]):
        if t == 0:
            continue
        m = res.x2[i, t]
        expected = p.A * res.x1[i, t - 1] + (1 - p.A) * m
        assert np.allclose(res.x1[i, t], expected, atol=1e-8), \
            "a no-data season applied something other than the pure transition"


def test_zero_rows_would_be_read_as_catastrophe():
    """Demonstrates the failure mode the absent-row convention avoids.

    Not a property of the model -- a property of the alternative -- and worth
    keeping as an executable statement of why §2.2 is written the way it is.
    """
    panel = synthetic_panel(60, seed=3)
    priors = synthetic_priors(panel)
    from career_model.model.dataset import load

    ds_gap = load(panel=panel, priors=priors)
    # Same panel, but gaps materialised as zero-count rows.
    grid = ds_gap.grid
    filled = []
    for i in range(grid.n_players):
        pid = int(grid.player_ids[i])
        for t in np.flatnonzero(grid.in_span[i] & ~grid.observed[i]):
            rec = {"player_id": pid, "player_name": "x",
                   "season_year": int(grid.season_years[i, t]),
                   "season": "x", "age": float(grid.age[i, t]),
                   "age_days": float(grid.age[i, t]) * 365.25,
                   "team_id": "AAA", "team_count": 1, "position_group": "F",
                   "possessions": 1500.0, "minutes": 700.0, "games_played": 40,
                   "schedule_length": 82, "games_share": 0.5,
                   "injury_season_flag": False, "possessions_imputed": False,
                   "season_index": 0, "panel_first_year": 0, "panel_last_year": 0,
                   "left_truncated": False}
            for c in ("fga_2p", "fgm_2p", "fga_3p", "fgm_3p", "fta", "ftm",
                      "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"):
                rec[c] = 0
            filled.append(rec)
    if not filled:
        pytest.skip("fixture produced no gaps")

    import pandas as pd
    ds_zero = load(panel=pd.concat([panel, pd.DataFrame(filled)], ignore_index=True),
                   priors=priors)

    p = _params(ds_gap)
    R1 = obs.refresh_R(ds_gap.grid, p.phi, p.acc_floor, p.sigma_poss)
    R2 = obs.refresh_R(ds_zero.grid, p.phi, p.acc_floor, p.sigma_poss)
    r1 = ss.run_filter_diag(ds_gap.grid, p, ds_gap.X, R1,
                            np.zeros_like(ds_gap.grid.z), keep_states=True)
    r2 = ss.run_filter_diag(ds_zero.grid, p, ds_zero.X, R2,
                            np.zeros_like(ds_zero.grid.z), keep_states=True)

    # The two grids order players independently, so align on player id, and
    # compare the state at the disputed season itself.
    j = IDX["dreb"]
    row2 = {int(p): i for i, p in enumerate(ds_zero.grid.player_ids)}
    drops = []
    for i, pid in enumerate(ds_gap.grid.player_ids):
        for t in np.flatnonzero(ds_gap.grid.in_span[i] & ~ds_gap.grid.observed[i]):
            drops.append(r1.x1[i, t, j] - r2.x1[row2[int(pid)], t, j])
    drops = np.array(drops)
    assert len(drops) > 3
    # Every affected player moves the same way, and the move compounds: the
    # state is on a log scale, so ~0.08 is an 8% talent revision extracted from
    # a season that never happened, before the depressed state propagates on.
    assert np.all(drops > 0), "zero rows should push the state down, not up"
    assert np.median(drops) > 0.05, (
        "zero rows should drag the filtered state down; if they do not, the "
        "fixture is not exercising the case")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
