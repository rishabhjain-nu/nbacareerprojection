"""The algebraic identity that ties the model together (§9, priority test).

In scalar form the Kalman gain is `K = P/(P+R)` and the hierarchical shrinkage
factor is `B = R/(P+R)`, so `K = 1 - B`.  Both consume the **same** `R`, the one
the observation layer computed from the counts.  If this fails numerically it
means the two layers are computing observation variance differently -- the
classic way a hierarchical model ends up shrinking twice, or not at all, while
still producing plausible-looking output.
"""

from __future__ import annotations

import numpy as np
import pytest

from career_model.config import IDX, MISSING_R, S
from career_model.model import hierarchy as hier
from career_model.model import observations as obs
from career_model.model import state_space as ss
from career_model.tests.fixtures import small_dataset


def test_gain_is_one_minus_shrinkage():
    ds = small_dataset()
    p = hier.default_params(p_x=ds.X.shape[1]).copy(
        Lam=np.zeros((S, hier.Q_RANK)), Sigma_p=np.diag(np.full(S, 0.3)))
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss)
    offsets = np.zeros_like(ds.grid.z)

    res = ss.run_filter_diag(ds.grid, p, ds.X, R, offsets, keep_states=True)

    # The gain stored at step t was computed against P_{t|t-1}, which is the
    # prior variance the update saw.  Reconstruct it from the posterior:
    # P_post = (1 - K) P_prior  =>  P_prior = P_post / (1 - K).
    for t in range(3):
        n = int(ds.grid.n_active[t])
        K = res.gain[:n, t]
        mask = ds.grid.obs_mask[:n, t]
        r = np.where(mask, R[:n, t], MISSING_R)
        P_post = res.P11[:n, t]
        P_prior = P_post / np.maximum(1.0 - K, 1e-12)

        B = ss.shrinkage_factor(P_prior, r)
        assert np.allclose(K[mask], (1.0 - B)[mask], atol=1e-8), \
            "Kalman gain and shrinkage factor disagree -- R is inconsistent " \
            "between the observation layer and the hierarchy"


def test_gain_matches_closed_form_on_first_update():
    """At the very first season the prior variance is known in closed form,
    so the gain can be checked against arithmetic rather than against itself."""
    ds = small_dataset()
    p = hier.default_params(p_x=ds.X.shape[1]).copy(
        Lam=np.zeros((S, hier.Q_RANK)), Sigma_p=np.diag(np.full(S, 0.3)))
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss)
    res = ss.run_filter_diag(ds.grid, p, ds.X, R, np.zeros_like(ds.grid.z),
                             keep_states=True)

    P_prior = np.diag(p.Sigma_p) + np.diag(p.stationary_dispersion())
    mask = ds.grid.obs_mask[:, 0]
    expected = P_prior / (P_prior + np.where(mask, R[:, 0], MISSING_R))
    assert np.allclose(res.gain[:, 0][mask], expected[mask], atol=1e-10)


def test_shrinkage_behaves_by_sample_size():
    """§5.2 M2: a low-possession rookie shrinks hard, a full-season veteran
    barely at all.  These are the numbers the spec names as the check."""
    ds = small_dataset()
    p = hier.default_params(p_x=ds.X.shape[1]).copy(
        Lam=np.zeros((S, hier.Q_RANK)), Sigma_p=np.diag(np.full(S, 0.3)))
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss)

    j = IDX["dreb"]
    P_prior = p.Sigma_p[j, j] + p.stationary_dispersion()[j, j]
    poss = ds.grid.exposure[:, 0]
    counts = ds.grid.counts["dreb"][:, 0]
    ok = np.isfinite(poss) & np.isfinite(counts)

    B = ss.shrinkage_factor(P_prior, R[:, 0, j])
    thin = ok & (poss < 600)
    thick = ok & (poss > 3000)
    if thin.any():
        assert B[thin].mean() > 0.15, "small-sample seasons are not being shrunk"
    if thick.any():
        assert B[thick].mean() < 0.20, "full seasons are being over-shrunk"
    if thin.any() and thick.any():
        assert B[thin].mean() > B[thick].mean()


def test_observation_variance_has_a_floor():
    """§3.3: the `1/phi_s` term must not vanish as exposure grows.  Without it
    the model becomes certain a healthy veteran's one season is his talent."""
    e = np.array([4000.0])
    for phi in (10.0, 50.0):
        _, R_big = obs.volume_pseudo_obs(np.array([600.0]), e, phi)
        assert R_big[0] > 1.0 / phi * 0.99
        _, R_huge = obs.volume_pseudo_obs(np.array([1e7]), np.array([1e9]), phi)
        assert R_huge[0] >= 1.0 / phi - 1e-9, "floor disappeared at high volume"


def test_availability_noise_has_two_regimes():
    """§4.2, done properly: an injury season is a *different distribution* on
    the availability dimension, not the same one scaled.

    Year-over-year sd of log possessions is 0.59 between two healthy seasons and
    1.60 when either is injury-flagged.  A single Gaussian splits the difference
    at 1.05, and a durable player's decade of consistent seasons then gets read
    through an observation variance calibrated mostly by other players' injuries
    -- so the filter refuses to believe him.  Two estimated scales fix it.
    """
    from career_model.config import AVAIL_IDX
    ds = small_dataset()
    p = hier.default_params(p_x=ds.X.shape[1]).copy(
        sigma_poss=0.30, sigma_poss_inj=1.20, injury_infl=1.5)
    # The fixture has no injury flags, so set some by hand.
    ds.grid.injury[:, 1] = True

    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss,
                      p.sigma_poss_inj, p.injury_infl)
    j = AVAIL_IDX
    healthy = ds.grid.obs_mask[:, 0, j]
    hurt = ds.grid.obs_mask[:, 1, j]
    assert np.allclose(R[healthy, 0, j], 0.30 ** 2)
    assert np.allclose(R[hurt, 1, j], 1.20 ** 2)
    assert R[hurt, 1, j].mean() > 10 * R[healthy, 0, j].mean()

    # The general multiplier still applies to the other dimensions, and the
    # availability dimension must not be inflated twice.
    k = IDX["dreb"]
    R0 = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss,
                       p.sigma_poss_inj, injury_infl=1.0)
    both = ds.grid.obs_mask[:, 1, k]
    assert np.allclose(R[both, 1, k], 1.5 * R0[both, 1, k])
    assert np.allclose(R[hurt, 1, j], R0[hurt, 1, j]), \
        "availability was inflated by the shared multiplier as well as its own scale"


def test_diag_and_full_filters_agree_when_q_is_diagonal():
    """The fast path is the same model, not an approximation of it."""
    ds = small_dataset()
    p = hier.default_params(p_x=ds.X.shape[1]).copy(
        Lam=np.zeros((S, hier.Q_RANK)), Sigma_p=np.diag(np.full(S, 0.3)))
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss)
    offsets = 0.01 * np.ones_like(ds.grid.z)
    a = ss.run_filter(ds.grid, p, ds.X, R, offsets).loglik
    b = ss.run_filter_diag(ds.grid, p, ds.X, R, offsets).loglik
    assert abs(a - b) / abs(a) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
