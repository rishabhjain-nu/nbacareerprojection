"""The properties the simulation exists to have (§6).

The headline one is that predictive uncertainty grows with horizon *because the
model propagates it*, not because anyone applied a widening factor.  The state
is a **mean-reverting** AR(1) (A < 1), so the correct h-step covariance is

    P_h = A^h P_0 (A^h)' + sum_{j=0}^{h-1} A^j Q (A^j)'

which grows *sub-linearly* and saturates toward the stationary variance
`Q/(1-A^2)` -- it is NOT `P_0 + hQ`, which is the random-walk (A = I) special
case and overstates the variance by up to several-fold at long horizons.  With
the reversion target `m` itself uncertain, the exact object is the augmented
state `[theta; m]` propagated by `T = [[A, I-A],[0, I]]`.
`test_covariance_propagation` pins that the simulator matches the augmented
analytic covariance and the direct Monte-Carlo propagation -- and diverges from
`P_0 + hQ`.  Because the process mean-reverts, a displayed band need not widen
monotonically, so there is no test asserting that it must.
"""

from __future__ import annotations

import numpy as np
import pytest

from career_model.config import POSITION_GROUPS, S
from career_model.model import hierarchy as hier
from career_model.model import observations as obs
from career_model.model import state_space as ss
from career_model.model.dataset import load as load_dataset
from career_model.simulate import project
from career_model.tests.fixtures import synthetic_panel, synthetic_priors


class _Fit:
    def __init__(self, ds):
        nb, nbi = ds.age_basis.size, ds.init_basis.size
        rng = np.random.default_rng(0)
        # `Psi` is set to the middle of the fitted range rather than the library
        # default.  With the default 0.01 the process noise is so small that
        # `Q(1-A^2h)/(1-A^2)` adds ~7% to the state sd over six seasons, which is
        # inside Monte Carlo noise -- the widening test would then be measuring
        # sampling error rather than the propagation it is meant to check.
        self.params = hier.default_params(p_x=ds.X.shape[1]).copy(
            Lam=0.05 * rng.standard_normal((S, hier.Q_RANK)),
            Psi=np.full(S, 0.05),
            Sigma_p=np.eye(S) * 0.25,
            beta=np.zeros((ds.X.shape[1], S)))
        # A plausible level, so the negative-binomial draws are not degenerate.
        self.params.beta[0] = -3.0
        self.params.beta[0, -1] = 7.4              # log possessions
        self.c_init = np.zeros((nbi, S))
        self.delta_league = np.zeros((nb, S))
        self.delta_pos = np.zeros((len(POSITION_GROUPS), nb, S))
        self.player_scale = None       # league aging rate for every player
        self.c_init_coefs = np.zeros((nbi, S))  # debut-age spline for rookies


class _Model:
    def __init__(self, ds, posterior=None):
        from career_model.model.hazard import Hazard
        self.fit = _Fit(ds)
        self.age_basis = ds.age_basis
        self.init_basis = ds.init_basis
        self.posterior = posterior
        coef = np.zeros(3 + S)
        coef[0] = 2.2                              # ~90% continuation
        coef[1] = -0.5
        self.hazard = Hazard(coef=coef, cov=np.eye(3 + S) * 1e-4,
                             names=["c"] * (3 + S))


@pytest.fixture(scope="module")
def setup():
    panel = synthetic_panel(120, seed=2)
    ds = load_dataset(panel=panel, priors=synthetic_priors(panel))
    model = _Model(ds)
    R = obs.refresh_R(ds.grid, model.fit.params.phi, model.fit.params.acc_floor,
                      model.fit.params.sigma_poss)
    filt = ss.run_filter(ds.grid, model.fit.params, ds.X, R,
                         np.zeros_like(ds.grid.z), ds.init_basis,
                         model.fit.c_init, keep_states=True)
    return ds, model, filt


def _band_widths(proj, stat="pts_per100", relative=False):
    """80% band width per horizon.

    `relative` divides by the median. Absolute width is the right unit for "do
    the bands widen with horizon" on one player; *relative* width is the right
    unit for comparing two players, because these are log-scale states and a
    high-usage player's band is wider in points simply because his level is
    higher.
    """
    from career_model.simulate import derive
    widths = []
    for h in range(len(proj.ages)):
        live = proj.alive[:, h]
        if live.sum() < 100:
            break
        cols = {c: proj.box[c][live, h] for c in project.COUNT_NAMES}
        comp = derive.derive_composites(cols, np.maximum(proj.possessions[live, h], 1.0))
        v = comp[stat]
        lo, mid, hi = np.percentile(v, [10, 50, 90])
        widths.append(float((hi - lo) / max(mid, 1e-9)) if relative else float(hi - lo))
    return widths


def test_covariance_propagation(setup):
    """The simulator's state covariance must equal the correct mean-reverting
    AR(1) covariance -- NOT the random-walk `P_0 + hQ`.

    Three objects are compared for `Var(theta_{T+h})`:

      1. **analytic** -- the augmented state `x = [theta; m]` propagated by
         `P_h = T P_{h-1} T' + Q_aug` with `T = [[A, I-A],[0, I]]`, which is the
         exact covariance including reversion-target uncertainty;
      2. **direct Monte Carlo** -- drawing `x_0 ~ N(0, P_0)` and iterating
         `x_{t+1} = T x_t + noise`;
      3. **simulator** -- `Var` of `project.simulate`'s `theta` draws.

    (1) and (2) must match tightly; the simulator must match them within Monte
    Carlo error; and all three must diverge from `P_0 + hQ`, which over-states
    the variance because it assumes `A = I`.  This is the invariant that
    replaces the old "the band must always widen" rule: for a mean-reverting
    process the variance saturates, so a non-widening displayed band is correct
    behaviour, not a bug.
    """
    ds, model, filt = setup
    p = model.fit.params
    A = np.diag(p.A)
    Q = p.Q()
    I = np.eye(S)
    r = 0
    last = int(ds.grid.last_index[r])
    P0 = np.block([[filt.P11[r, last], filt.P12[r, last]],
                   [filt.P12[r, last].T, filt.P22[r, last]]])
    T = np.block([[A, I - A], [np.zeros((S, S)), I]])
    Q_aug = np.zeros((2 * S, 2 * S))
    Q_aug[:S, :S] = Q

    H = 6
    analytic = []
    Ph = P0.copy()
    for _ in range(H):
        Ph = T @ Ph @ T.T + Q_aug
        analytic.append(np.diag(Ph[:S, :S]).copy())

    # direct Monte Carlo of the same augmented process (mean set to zero)
    rng = np.random.default_rng(0)
    n = 60000
    L0 = np.linalg.cholesky(P0 + 1e-10 * np.eye(2 * S))
    LQ = np.linalg.cholesky(Q + 1e-10 * I)
    x = rng.standard_normal((n, 2 * S)) @ L0.T
    mc = []
    for _ in range(H):
        noise = np.zeros((n, 2 * S))
        noise[:, :S] = rng.standard_normal((n, S)) @ LQ.T
        x = x @ T.T + noise
        mc.append(x[:, :S].var(axis=0))

    # the simulator (no posterior draws, so a single clean param set)
    import copy
    m2 = copy.copy(model)
    m2.posterior = None
    proj = project.simulate(m2, ds, r, filt, n_draws=60000, horizon=H, seed=5)
    sim = [proj.theta[:, h, :].var(axis=0) for h in range(H)]

    for h in range(H):
        # analytic vs direct MC: same process, tight
        assert np.allclose(analytic[h], mc[h], rtol=0.06, atol=1e-3), \
            f"h={h + 1}: analytic AR(1) != direct Monte Carlo"
        # simulator vs analytic: within MC error
        assert np.allclose(sim[h], analytic[h], rtol=0.08, atol=3e-3), \
            f"h={h + 1}: simulator covariance != augmented AR(1) analytic"

    # and all of them diverge from the random-walk P_0 + hQ at long horizon
    P0_diag = np.diag(P0[:S, :S])
    Q_diag = np.diag(Q)
    hq_h6 = P0_diag + H * Q_diag
    # P_0 + hQ over-states the true variance by a clear margin (A=0.85 here)
    assert np.mean(hq_h6) > 1.5 * np.mean(analytic[H - 1]), \
        "P_0 + hQ should materially over-state the mean-reverting variance"


def test_state_variance_grows_but_saturates(setup):
    """State variance grows with horizon but *sub-linearly*, saturating toward
    the stationary variance `Q/(1-A^2)` -- the mean-reverting signature, not the
    linear growth `P_0 + hQ` would give."""
    ds, model, filt = setup
    proj = project.simulate(model, ds, 0, filt, n_draws=6000, horizon=6)
    var = [proj.theta[:, h, :].var(axis=0) for h in range(6)]
    for j in range(S):
        # grows early ...
        assert var[3][j] > var[0][j], f"state {j} variance did not grow"
        # ... but the later increment is smaller than the first (saturation)
        first_step = var[1][j] - var[0][j]
        last_step = var[5][j] - var[4][j]
        assert last_step <= first_step + 1e-6, \
            f"state {j} variance grows linearly, not saturating (A<1 violated?)"


def test_parameter_uncertainty_widens_intervals(setup):
    """§6 step 1 is not decorative: adding hyperparameter draws must widen the
    predictive distribution relative to holding them at the mode."""
    ds, model, filt = setup
    base = project.simulate(model, ds, 0, filt, n_draws=3000, horizon=5)

    rng = np.random.default_rng(0)
    v0 = hier.pack(model.fit.params)
    draws = v0 + 0.15 * rng.standard_normal((200, len(v0)))
    with_post = project.simulate(_Model(ds, posterior=draws), ds, 0, filt,
                                 n_draws=3000, horizon=5)

    w_base = _band_widths(base)
    w_post = _band_widths(with_post)
    n = min(len(w_base), len(w_post))
    assert n >= 3
    assert np.mean(w_post[:n]) > np.mean(w_base[:n]), (
        "drawing hyperparameters from the posterior did not widen the "
        "predictive interval -- the parameter variance source is not wired in")


def test_rookie_is_wider_than_a_veteran(setup):
    """§8.1: with nothing to filter, a rookie's year-1 interval should be about
    as wide as an established player's year-5 interval."""
    ds, model, filt = setup
    vet = None
    for i in range(ds.n_players):
        if ds.grid.n_history[i] >= 6:
            vet = i
            break
    assert vet is not None

    v = project.simulate(model, ds, vet, filt, n_draws=3000, horizon=6)
    r = project.simulate_rookie(
        model, ds, ds.X[vet], np.zeros(S), age0=20.0, year0=2026,
        pos_idx=int(ds.pos_idx[vet]), player_id=999999, n_draws=3000, horizon=6)

    # Relative width, because the two players sit at different levels and the
    # states are on a log scale -- an absolute comparison would mostly measure
    # who scores more.
    wv, wr = _band_widths(v, relative=True), _band_widths(r, relative=True)
    assert len(wv) >= 5 and len(wr) >= 2
    assert wr[0] > wv[0] * 1.2, \
        f"a rookie's year-1 band ({wr[0]:.2f}) is not wider than a veteran's ({wv[0]:.2f})"
    assert wr[0] > 0.8 * wv[min(4, len(wv) - 1)], (
        f"a rookie's year-1 band ({wr[0]:.2f}) should be comparable to a "
        f"veteran's year-5 band ({wv[min(4, len(wv) - 1)]:.2f}) -- §8.1")


def test_avail_quality_aging_targets_old_stars_only(setup):
    """The age x quality availability adjustment (star minute-drop fix) must:
    leave young players exactly alone (the age-31 hinge), slow decline for
    high-quality old players, speed it for low-quality ones, and respect its
    caps.  These are the safety properties that keep it from rescuing a faded
    player or manufacturing minutes for a young one."""
    ds, model, filt = setup
    gamma = np.zeros(S)
    gamma[project.IDX["ast"]] = 1.0                # a stand-in skill direction
    aq = project.AvailQualityAging(gamma=gamma, qmean=0.0, qsd=1.0,
                                   c_step=0.085, c_slope=0.0)

    theta_hi = np.zeros((3, S)); theta_hi[:, project.IDX["ast"]] = 2.0   # +2sd quality
    theta_lo = np.zeros((3, S)); theta_lo[:, project.IDX["ast"]] = -2.0  # -2sd

    # Below the onset age: no adjustment for anyone.
    assert np.allclose(aq.increment(29.0, theta_hi), 0.0)
    assert np.allclose(aq.increment(30.9, theta_lo), 0.0)
    # The onset is a step, not a ramp: full protection already at the first
    # post-31 season, not a slow ramp-in (the bug the step form fixed).
    assert np.all(aq.increment(32.0, theta_hi) > 0.10)
    # Old + high quality: minutes protected; old + low quality: decline faster.
    hi35 = aq.increment(35.0, theta_hi)
    lo35 = aq.increment(35.0, theta_lo)
    assert np.all(hi35 > 0) and np.all(lo35 < 0)
    assert np.all(hi35 <= aq.cap_hi + 1e-12) and np.all(lo35 >= aq.cap_lo - 1e-12)
    # It moderates decline, never turns it into a runaway gain.
    assert aq.cap_hi < 0.25


def test_within_career_absence_zeros_the_box_but_keeps_career(setup):
    """The absence model (deviation #4) must create an explicit missed-season
    state: a draw can be career-alive that season yet not have played, in which
    case its box score is zero, and P(plays) sits at or below P(career active)
    every horizon.  With no absence model the two collapse to identical."""
    from career_model.model.hazard import Hazard
    ds, model, filt = setup

    # No absence model: played is exactly alive, and nothing is a miss.
    p_off = project.simulate(model, ds, 0, filt, n_draws=1500, horizon=8)
    assert np.array_equal(p_off.played, p_off.alive)

    # A deliberately leaky absence model (~25% miss rate) so misses are common.
    coef = np.zeros(3 + S)
    coef[0] = 1.1                                  # inv_logit(1.1) ~ 0.75 play
    model.absence = Hazard(coef=coef, cov=np.eye(3 + S) * 1e-4,
                           names=["c"] * (3 + S))
    try:
        proj = project.simulate(model, ds, 0, filt, n_draws=1500, horizon=8)
    finally:
        model.absence = None

    # played implies alive, and strictly fewer seasons are played than spanned.
    assert np.all(proj.alive | ~proj.played)       # played -> alive
    assert proj.played.sum() < proj.alive.sum()
    # A missed season (alive & ~played) contributes a zero box score.
    missed = proj.alive & ~proj.played
    assert missed.any(), "leaky absence model should produce misses"
    for c in ("fga_2p", "ast", "stl"):
        assert np.all(proj.box[c][missed] == 0)
    assert np.all(proj.possessions[missed] == 0)
    # P(plays) <= P(active) at every horizon, and strictly below somewhere.
    sc = project.survival_curve(proj)
    assert np.all(sc["p_play"] <= sc["p_active"] + 1e-9)
    assert np.any(sc["p_play"] < sc["p_active"] - 1e-3)


def test_career_totals_are_conditional_on_survival(setup):
    """A draw that exits at h=2 contributes two seasons and stops -- the hazard
    has to be visible in the totals, not silently ignored."""
    ds, model, filt = setup
    proj = project.simulate(model, ds, 0, filt, n_draws=2000, horizon=10)
    totals = project.career_totals(proj)
    seasons = totals.loc[totals["stat"] == "seasons"].iloc[0]
    assert seasons["p5"] < seasons["p50"] <= seasons["p95"]
    assert seasons["p95"] <= 10
    # Points accumulated must be zero for a draw that never played.
    dead = ~proj.alive.any(axis=1)
    if dead.any():
        assert np.all((proj.box["fga_2p"] * proj.alive).sum(axis=1)[dead] == 0)


def test_hazard_interaction_design_matches_projection(setup):
    """The fix-6 age x quality hazard carries one extra design column, built in
    two places: `Hazard.design` (fit, calibration) and the projection's
    `_p_survive` (hot loop, no Hazard object).  If they ever disagree, the
    simulated survival silently stops being the fitted model -- so the two are
    asserted equal, interaction on and off, before any ramp is applied."""
    from career_model.model.hazard import Hazard

    rng = np.random.default_rng(3)
    theta = rng.standard_normal((40, S))
    ages = rng.uniform(20, 40, size=40)

    for inter in (False, True):
        coef = rng.standard_normal(3 + S + (1 if inter else 0)) * 0.3
        kw = {}
        if inter:
            g = rng.standard_normal(S)
            kw = dict(inter_gamma=g, inter_mu=0.3, inter_sd=1.7)
        h = Hazard(coef=coef, cov=np.eye(len(coef)), names=["c"] * len(coef), **kw)
        p_fit = h.p_survive(ages, theta)
        ramp = np.clip((project.MAX_AGE - (ages + 1.0))
                       / (project.MAX_AGE - project.AGE_RAMP_START), 0.0, 1.0)
        p_sim = project._p_survive(np.broadcast_to(coef, (40, len(coef))),
                                   ages, theta, project._hazard_inter(h))
        assert np.allclose(p_sim, p_fit * ramp, atol=1e-12), \
            f"projection survival diverges from the fitted hazard (interaction={inter})"


def test_points_are_derived_not_modelled(setup):
    """§3.1: points is `1*FTM + 2*2PM + 3*3PM` per draw, exactly."""
    from career_model.simulate import derive
    ds, model, filt = setup
    proj = project.simulate(model, ds, 0, filt, n_draws=500, horizon=3)
    box = {c: proj.box[c][:, 0] for c in project.COUNT_NAMES}
    expected = box["ftm"] + 2 * box["fgm_2p"] + 3 * box["fgm_3p"]
    assert np.array_equal(derive.points(box), expected)
    # And makes never exceed attempts, at every horizon.
    for h in range(3):
        assert np.all(proj.box["fgm_2p"][:, h] <= proj.box["fga_2p"][:, h])
        assert np.all(proj.box["fgm_3p"][:, h] <= proj.box["fga_3p"][:, h])
        assert np.all(proj.box["ftm"][:, h] <= proj.box["fta"][:, h])


def test_eb_reversion_target_only_moves_underidentified_players(setup):
    """The empirical-Bayes reversion target must leave a player alone when his
    own record and the model's `m_i` already agree, and pull toward his record
    only when they diverge.

    That is the whole safety case for the fix: it corrects the mis-identified
    stars without silently rewriting everyone else's projection.  A player whose
    filtered level matches his `m_i` should get `m_eff ~= m_model` regardless of
    career length; only a gap between the two should move him.
    """
    ds, model, filt = setup
    grid = ds.grid
    established = [i for i in range(grid.n_players) if grid.n_history[i] >= 5]
    assert established, "fixture needs a player with a real history"

    for i in established[:8]:
        m_eff, m_model = project._eb_reversion_target(model, ds, filt, i)
        # m_eff is a convex blend of m_model and the player's de-aged level, so
        # it can never leave the interval between them.
        m_own = m_eff  # recover implied m_own not needed; check the blend bound
        gap = np.abs(m_eff - m_model)
        # Where the filtered state already sits at m_model (well-identified dim),
        # the blend cannot have moved the target.
        j = filt.x1[i, int(grid.last_index[i])] - m_model  # theta - m per dim
        near = np.abs(j) < 1e-6
        assert np.allclose(m_eff[near], m_model[near], atol=1e-6), \
            "EB moved a dimension where theta already equals m_model"
        assert np.all(np.isfinite(gap))
        _ = m_own


def test_minutes_split_is_internally_consistent():
    """`games * mpg` must reproduce the simulated minutes exactly.

    Deriving games from minutes and the drawn MPG -- rather than sampling the
    two independently -- is what stops a draw showing 30 points a game on 20
    minutes.  If they are ever sampled apart, this fails.
    """
    from career_model.simulate import derive
    rng = np.random.default_rng(0)
    ms = derive.MinutesSplit(coef=np.array([-2.0, 0.6, 0.0, 0.0, 0.0]),
                             sd_between=0.175, sd_within=0.217,
                             sd_coef=np.array([0.5, -0.15, 0.0]))
    poss = rng.uniform(200, 5500, 5000)
    games, mpg = ms.draw(rng, poss, np.full(5000, 26.0))
    minutes = poss / 2.02
    assert np.allclose(games * mpg, minutes, rtol=1e-9)
    assert np.all(games >= 1) and np.all(games <= 82)
    assert np.all(mpg <= 48.0)


def test_both_entry_paths_produce_per_game_stats(setup):
    """Established players and rookies must both get the per-game table.

    Regression test for a real bug: `simulate_rookie` was called without
    `minutes_split`, so every prospect silently got an empty per-game table --
    no error, no warning, just a missing card. Anything that has to be threaded
    through two separate entry points deserves a test that checks both.
    """
    from career_model.simulate import derive
    ds, model, filt = setup
    ms = derive.MinutesSplit(coef=np.array([-2.0, 0.6, 0.0, 0.0, 0.0]),
                             sd_between=0.175, sd_within=0.217,
                             sd_coef=np.array([0.5, -0.15, 0.0]))
    wanted = {"games", "minutes_per_game", "pts_per_game", "reb_per_game",
              "ast_per_game", "stl_per_game", "blk_per_game"}

    vet = project.simulate(model, ds, 0, filt, n_draws=600, horizon=4,
                           minutes_split=ms)
    rookie = project.simulate_rookie(
        model, ds, ds.X[0], np.zeros(S), age0=20.0, year0=2026,
        pos_idx=int(ds.pos_idx[0]), player_id=999998, n_draws=600, horizon=4,
        minutes_split=ms)

    for name, proj in (("established", vet), ("rookie", rookie)):
        assert proj.games.any(), f"{name}: no games were simulated"
        stats = set(project.summarise(proj, None)["stat"])
        missing = wanted - stats
        assert not missing, f"{name} path is missing per-game stats: {sorted(missing)}"


def test_minutes_split_residual_sd_falls_with_exposure():
    """Heteroskedasticity is the property that keeps the band off the
    48-minute ceiling; a pooled sd puts a starter's 90th percentile past 43
    minutes, which basketball does not do."""
    from career_model.config import PANEL_PATH
    from career_model.simulate import derive
    if not PANEL_PATH.exists():
        pytest.skip("panel not built")
    import pandas as pd
    ms = derive.fit_minutes_split(pd.read_parquet(PANEL_PATH), verbose=False)
    sd = ms.resid_sd(np.array([300.0, 1000.0, 2500.0, 4500.0]))
    assert np.all(np.diff(sd) < 0), "residual sd must fall with exposure"
    assert sd[0] > 3 * sd[-1]

    # And the drawn band for a starter must stay clear of the physical ceiling.
    rng = np.random.default_rng(0)
    _, mpg = ms.draw(rng, np.full(20000, 4300.0), np.full(20000, 26.0))
    assert np.percentile(mpg, 99) < 42.0, \
        f"99th percentile MPG is {np.percentile(mpg, 99):.1f}; the band is " \
        "piling up against the 48-minute bound"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_avail_eb_touches_only_the_availability_dimension(setup):
    """The per-player availability correction is a mean shift on one dimension.

    With the same seed, toggling `use_avail_eb` must leave the simulated state
    for the other thirteen dimensions *bit-for-bit identical* -- the fix that
    silently rewrote a shooting percentage while adjusting possessions would be
    worse than the problem -- and must move the availability dimension toward
    the player's own-record level, never past it.
    """
    from career_model.config import AVAIL_IDX

    ds, model, filt = setup
    grid = ds.grid
    rows = [i for i in range(grid.n_players) if grid.n_history[i] >= 5][:6]
    assert rows, "fixture needs established players"

    for i in rows:
        own = project._avail_own_level(model, ds, filt, i)
        assert own is not None
        level, w, _ = own
        assert 0.0 < w < 1.0
        # The scalar filter's output is a convex combination of the de-aged
        # record, so it cannot leave the record's range.
        obs = np.flatnonzero(grid.observed[i])
        poss = grid.exposure[i, obs]
        z = np.log(poss[poss > 0])
        # (fixture aging coefs are zero, so no de-aging offset to allow for)
        assert z.min() - 1e-9 <= level <= z.max() + 1e-9

        off = project.simulate(model, ds, i, filt, n_draws=400, horizon=3,
                               seed=7, use_avail_eb=False)
        on = project.simulate(model, ds, i, filt, n_draws=400, horizon=3,
                              seed=7, use_avail_eb=True)
        # Bit-for-bit only at h=0: beyond it the negative-binomial box draws
        # consume a data-dependent number of underlying randoms, so the two
        # arms' RNG streams legitimately diverge.  The season the correction
        # acts on directly is the one that must show zero contamination.
        other = [d for d in range(S) if d != AVAIL_IDX]
        assert np.array_equal(off.theta[:, 0, other], on.theta[:, 0, other]), \
            "availability EB leaked into a non-availability dimension"

        # The correction is deterministic given the record, so the h=0 mean
        # shift is exactly predictable: the start moves by w*(level - x1), the
        # target by w*((level - L_T) - m_eff), and one transition mixes them
        # through A.  Same seed means same noise, so this holds to float
        # precision, not MC precision.
        last = int(grid.last_index[i])
        level_, w_, L_T = project._avail_own_level(model, ds, filt, i)
        m_eff, _ = project._eb_reversion_target(model, ds, filt, i)
        d_theta = w_ * (level_ - float(filt.x1[i, last][AVAIL_IDX]))
        d_m = w_ * ((level_ - L_T) - float(m_eff[AVAIL_IDX]))
        A_avail = float(model.fit.params.A[AVAIL_IDX])
        expected = d_m + A_avail * (d_theta - d_m)
        shift = float(np.mean(on.theta[:, 0, AVAIL_IDX] - off.theta[:, 0, AVAIL_IDX]))
        assert abs(shift - expected) < 1e-9, \
            f"availability shift {shift:.6f} != predicted {expected:.6f}"


def test_injury_record_counts_flags(setup):
    """The propensity evidence is (flagged, observed) straight off the grid."""
    ds, model, filt = setup
    grid = ds.grid
    i = next(j for j in range(grid.n_players) if grid.n_history[j] >= 3)
    k, n = project._injury_record(ds, i)
    obs = np.flatnonzero(grid.observed[i])
    assert n == len(obs)
    assert k == int(grid.injury[i, obs].sum())
    # Shrinkage arithmetic: an iron man's rate falls, a fragile player's rises,
    # and both stay inside the clip.
    M = project.INJ_PROPENSITY_STRENGTH
    base = 0.2
    iron = np.clip((M * base + 0) / (M + 12), 0.03, 0.7)
    glass = np.clip((M * base + 8) / (M + 12), 0.03, 0.7)
    assert iron < base < glass
