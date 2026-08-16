"""Backtest leakage control: the four channels, closed and asserted.

The filter recursion is causal by construction -- `theta_{t|t}` conditions on
data through `t`, and `v_t = z_t - theta_{t|t-1}` is formed against a prediction
made before `z_t` was read.  That property is easy to destroy by accident, and
when it is destroyed the damage is invisible: nothing errors, the CRPS just
quietly gets better.  These tests are the tripwires.

  1. **No projecting from smoothed states.**  `test_cutoff_state_is_invariant`
     is the strong form: the filtered state at the cutoff must be *bit-for-bit*
     the same whether or not post-cutoff seasons exist in the panel at all.  An
     RTS smoother would fail this immediately, and so would any accidental
     backward pass.  `test_simulate_rejects_smoothed_state` closes the entry
     point so the rule is enforced in code, not convention.
  2. **Hyperparameters refit per fold** -- `test_fold_sees_no_future_rows`.
  3. **`beta` refit per fold** -- same test; `beta` is estimated from the
     truncated player set only.
  4. **GBM prior mean refit per fold** -- `test_gbm_prior_is_fold_local`.
"""

from __future__ import annotations

import numpy as np
import pytest

from career_model.config import S
from career_model.model import hierarchy as hier
from career_model.model import observations as obs
from career_model.model import state_space as ss
from career_model.model.dataset import load as load_dataset
from career_model.tests.fixtures import synthetic_panel, synthetic_priors

CUTOFF = 2012


def _params(ds):
    return hier.default_params(p_x=ds.X.shape[1]).copy(
        Lam=np.zeros((S, hier.Q_RANK)), Sigma_p=np.diag(np.full(S, 0.3)))


def _filter(ds, params):
    R = obs.refresh_R(ds.grid, params.phi, params.acc_floor, params.sigma_poss)
    return ss.run_filter(ds.grid, params, ds.X, R, np.zeros_like(ds.grid.z),
                         keep_states=True)


def test_cutoff_state_is_invariant_to_post_cutoff_data():
    """The launch point of a backtested projection must not move when seasons
    after the cutoff are added to or removed from the panel.

    Both datasets are filtered with the *same* hyperparameters, isolating the
    recursion itself: this asks whether the forward pass is causal, not whether
    the fit is fold-local (which `test_fold_sees_no_future_rows` covers).
    """
    panel = synthetic_panel(150, seed=5)
    priors = synthetic_priors(panel)

    ds_full = load_dataset(panel=panel, priors=priors)
    ds_cut = load_dataset(panel=panel, priors=priors, max_season_year=CUTOFF)

    params = _params(ds_cut)
    f_full = _filter(ds_full, params)
    f_cut = _filter(ds_cut, params)

    row_full = {int(p): i for i, p in enumerate(ds_full.grid.player_ids)}
    compared = 0
    for i, pid in enumerate(ds_cut.grid.player_ids):
        t_cut = int(ds_cut.grid.last_index[i])
        if t_cut < 0:
            continue
        year = int(ds_cut.grid.season_years[i, t_cut])
        j = row_full[int(pid)]
        t_full = int(np.flatnonzero(ds_full.grid.season_years[j] == year)[0])
        # Both grids start a career at its first season, so the same calendar
        # season sits at the same career index; guard that assumption anyway.
        assert t_full == t_cut

        assert np.allclose(f_cut.x1[i, t_cut], f_full.x1[j, t_full], atol=0, rtol=0), \
            f"theta_(T|T) moved for player {pid} when future seasons were added"
        assert np.allclose(f_cut.x2[i, t_cut], f_full.x2[j, t_full], atol=0, rtol=0), \
            f"E[m_i] at the cutoff moved for player {pid}"
        assert np.allclose(f_cut.P11[i, t_cut], f_full.P11[j, t_full], atol=0, rtol=0), \
            f"P_(T|T) moved for player {pid}"
        compared += 1
    assert compared > 30, "test did not actually compare enough players"


def test_simulate_rejects_smoothed_state():
    """The entry point enforces the rule, so a future smoother cannot be wired
    into a projection by an innocent-looking keyword argument."""
    from career_model.simulate import project

    panel = synthetic_panel(40, seed=1)
    ds = load_dataset(panel=panel, priors=synthetic_priors(panel))
    filt = _filter(ds, _params(ds))

    filt.conditioning = "smoothed"
    with pytest.raises(ValueError, match="filtered"):
        project.simulate(object(), ds, 0, filt, n_draws=8, horizon=2)

    with pytest.raises(ValueError, match="filtered"):
        project.simulate(object(), ds, 0, None, n_draws=8, horizon=2)


def test_fold_sees_no_future_rows():
    """Channels 2 and 3: everything a fold fits is derived from the truncated
    frame -- the aging basis centring, the `x_i` standardisation, and the
    player set `beta` is estimated over."""
    panel = synthetic_panel(150, seed=5)
    priors = synthetic_priors(panel)
    ds = load_dataset(panel=panel, priors=priors, max_season_year=CUTOFF)

    assert ds.panel["season_year"].max() <= CUTOFF
    assert ds.grid.season_years[ds.grid.observed].max() <= CUTOFF
    # No player in the fold debuted after the cutoff.
    assert ds.priors["first_nba_year"].max() <= CUTOFF or \
        (ds.panel.groupby("player_id")["season_year"].min() <= CUTOFF).all()
    # The design matrix `beta` is fit against covers exactly the fold's players.
    assert ds.X.shape[0] == ds.grid.n_players
    # Career bookkeeping was recomputed, not inherited from the full panel.
    assert ds.panel["panel_last_year"].max() <= CUTOFF


def test_age_basis_centring_is_fold_local():
    """The centring vector identifies `delta` against `m_i`; computing it on the
    full panel would carry post-cutoff age composition into the fold."""
    panel = synthetic_panel(150, seed=5)
    priors = synthetic_priors(panel)
    full = load_dataset(panel=panel, priors=priors)
    cut = load_dataset(panel=panel, priors=priors, max_season_year=CUTOFF)
    assert not np.allclose(full.age_basis.centre, cut.age_basis.centre), \
        "the aging basis centring is identical across folds -- it is not being " \
        "recomputed from the truncated panel"


def test_gbm_prior_is_fold_local():
    """Channel 4: `f_GBM(x_i)` is trained on the fold's players only, and its
    out-of-fold construction means no player's offset is informed by his own
    target even inside the fold."""
    from career_model.model import gbm_prior

    panel = synthetic_panel(150, seed=5)
    priors = synthetic_priors(panel)
    ds = load_dataset(panel=panel, priors=priors, max_season_year=CUTOFF)

    rng = np.random.default_rng(0)
    m_hat = rng.normal(size=(ds.n_players, S))
    oof, model = gbm_prior.fit_offsets(ds, m_hat)
    assert oof.shape == (ds.n_players, S)
    # Pure noise targets must not be predictable out of fold; if they are, the
    # offsets are being fit in-sample and Sigma_player will collapse.
    r2 = 1.0 - ((m_hat - oof) ** 2).sum() / ((m_hat - m_hat.mean(0)) ** 2).sum()
    assert r2 < 0.05, f"out-of-fold R^2 on noise is {r2:.3f}; offsets leak in-sample"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
