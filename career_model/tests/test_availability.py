"""The joint availability candidate's structural guarantees (Session-2).

These hold regardless of whether the candidate is enabled by default: bounds are
enforced through the distributions (not clipping), possessions reconcile exactly
with games x mpg x pace, and the flag is inert when off.
"""

from __future__ import annotations

import numpy as np
import pytest

from career_model.config import ARTIFACT_DIR, S


@pytest.fixture(scope="module")
def prep():
    if not (ARTIFACT_DIR / "career_model.pkl").exists():
        pytest.skip("no fitted model artifact")
    from career_model.model.dataset import load as load_dataset
    from career_model.pipeline import filtered_states, load
    from career_model.simulate import availability as AV
    model = load()
    ds = load_dataset()
    filt = filtered_states(ds, model.fit)
    sysm = AV.fit_availability(ds, filt, model.hazard)
    return AV, model, ds, filt, sysm


def test_bounds_and_reconciliation(prep):
    AV, model, ds, filt, sysm = prep
    grid = ds.grid
    rng = np.random.default_rng(0)
    for pid in (1628983, 203999, 1628369, 101108):     # SGA, Jokic, Tatum, CP3
        r = int(np.flatnonzero(grid.player_ids == pid)[0])
        last = int(grid.last_index[r])
        ctx = AV.player_context(sysm, ds, filt, r)
        n = 5000
        anchor = np.full(n, filt.x1[r, last][AV.AVAIL_IDX])
        cur = np.tile(filt.x1[r, last], (n, 1))
        ret = np.zeros(n, bool)
        g, mpg, poss, _ = sysm.draw_season(rng, anchor, 29.0, cur, ctx, ret, 2027)
        # GP bounded by schedule, >=1 given appears; MPG strictly in (0,48)
        assert g.min() >= 1 and g.max() <= 82
        assert mpg.min() > 0 and mpg.max() < 48.0
        # possessions reconcile exactly with games * mpg * pace
        assert np.allclose(poss, g * mpg * AV.PACE, rtol=1e-9)
        # bounds come from the distribution, not a clip atom at the ceiling
        assert (g >= 81).mean() < 0.5
        assert (mpg > 47.5).mean() < 0.02


def test_flag_off_is_unchanged(prep):
    """simulate() with no avail_system must be bit-identical to the shipping
    path -- the candidate is inert when disabled."""
    AV, model, ds, filt, sysm = prep
    from career_model.simulate import project, derive
    ms = derive.fit_minutes_split(ds.panel, verbose=False)
    grid = ds.grid
    r = int(np.flatnonzero(grid.player_ids == 1628983)[0])
    a = project.simulate(model, ds, r, filt, n_draws=800, horizon=4, seed=0,
                         minutes_split=ms)
    b = project.simulate(model, ds, r, filt, n_draws=800, horizon=4, seed=0,
                         minutes_split=ms, avail_system=None)
    assert np.array_equal(a.games, b.games)
    assert np.array_equal(a.possessions, b.possessions)


def test_severity_states_distinct(prep):
    """48-game and 75-game seasons land in different injury states."""
    AV = prep[0]
    assert AV.classify_severity(np.array([48 / 82]))[0] == AV.SEV_MODERATE
    assert AV.classify_severity(np.array([75 / 82]))[0] == AV.SEV_HEALTHY
    assert AV.classify_severity(np.array([16 / 82]))[0] == AV.SEV_SEVERE
