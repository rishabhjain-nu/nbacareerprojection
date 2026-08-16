"""Monte Carlo projection for the permanent/transient model (§3.4 v2).

Cleaner than the v1 projection and, crucially, needs no empirical-Bayes patch:
the permanent level `ell` is a random walk the filter already tracks, so there
is no mis-identified mean to revert toward.  Each draw carries the two
components separately and sums them:

    ell_{t+1} = ell_t + drift(age) + eta^ell    (permanent: development/decline)
    u_{t+1}   = Phi u_t + eta^u                  (transient: decays toward 0)
    theta     = ell + u

A star launches from his tracked `ell_T` (high) with a small transient, so the
projection starts at his demonstrated level and ages down the drift -- rather
than collapsing toward a draft-based prior.  The transient's fast decay (Phi
capped at 0.5) is the only reversion, and it reverts to `ell`, not to a
cross-sectional mean.

The three §6 variance sources are unchanged: parameter (posterior draws), state
(the joint `[ell_T, u_T]` draw plus fresh `eta` at every step), sampling (the
NB/binomial draws).  The horizon-widening now has the right structure by
construction -- permanent uncertainty grows linearly (random walk), transient
uncertainty is bounded (stationary) -- which is exactly what a career forecast
should say.
"""

from __future__ import annotations

import numpy as np

from ..config import ACCURACY_PAIRS, ACCURACY_STATS, AVAIL_IDX, IDX, S, VOLUME_IDX, VOLUME_STATS
from .project import (Projection, COUNT_NAMES, MAX_AGE, MAX_POSSESSIONS,
                      EB_HALF_LIFE, _hazard_inter, _injury_prob, _p_survive,
                      _soft_cap, _volume_counts)


def _draw_joint_pt(rng, ell_m, u_m, Pee, Peu, Puu, n):
    """Draw `[ell_T, u_T]` jointly from the filter's block covariance."""
    mu = np.concatenate([ell_m, u_m])
    C = np.block([[Pee, Peu], [Peu.T, Puu]])
    C = 0.5 * (C + C.T) + 1e-8 * np.eye(2 * S)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
    d = mu + rng.standard_normal((n, 2 * S)) @ L.T
    return d[:, :S], d[:, S:]


def _pt_param_blocks(model, n_draws, n_param_sets, rng):
    from ..model import hierarchy_pt as hpt
    base = model.fit.params
    if model.posterior is None or len(model.posterior) == 0:
        return [base], [model.hazard.coef], np.zeros(n_draws, dtype=int)
    idx = rng.integers(0, len(model.posterior), size=n_param_sets)
    sets = [hpt.unpack(model.posterior[k], base) for k in idx]
    hz = [model.hazard.draw_coef(rng) for _ in range(n_param_sets)]
    return sets, hz, rng.integers(0, n_param_sets, size=n_draws)


def simulate_pt(model, ds, player_row, filt, n_draws=2000, horizon=12,
                seed=0, n_param_sets=64, minutes_split=None) -> Projection:
    """Project an established player forward under the permanent/transient model."""
    if filt is None or getattr(filt, "conditioning", None) != "filtered":
        raise ValueError("simulate_pt requires causal filter output")
    grid = ds.grid
    last = int(grid.last_index[player_row])
    if last < 0:
        raise ValueError("player has no observed seasons")

    rng = np.random.default_rng(seed + int(grid.player_ids[player_row]))
    sets, hz_coefs, assign = _pt_param_blocks(model, n_draws, n_param_sets, rng)

    ell, u = _draw_joint_pt(
        rng, filt.ell[player_row, last], filt.u[player_row, last],
        filt.Pee[player_row, last], filt.Peu[player_row, last],
        filt.Puu[player_row, last], n_draws)

    # A transient that has been sustained across a long career is not really
    # transient -- it is level the filter could not attribute to `ell` because
    # the population aging drift is too steep for a durable star (see the README).
    # Left alone it decays toward the undershooting `ell` and craters the
    # early-horizon projection.  Fold the career-length-weighted share of the
    # current transient into the permanent level for the projection, exactly the
    # empirical-Bayes move the v1 model needed, here reading off the model's own
    # `u`.  A rookie (n small) keeps his transient; a 15-year vet's persistent
    # transient is treated as level.
    n_hist = int(grid.n_history[player_row])
    w = n_hist / (n_hist + EB_HALF_LIFE)
    ell = ell + w * u
    u = (1.0 - w) * u

    age0 = float(grid.age[player_row, last])
    year0 = int(grid.season_years[player_row, last])
    offset = 0.0
    if minutes_split is not None:
        ob = np.flatnonzero(grid.observed[player_row])
        offset = minutes_split.player_offset(
            grid.exposure[player_row, ob], grid.age[player_row, ob],
            grid.games[player_row, ob])

    return _roll_forward_pt(model, ds, rng, ell, u, age0, year0,
                            int(grid.player_ids[player_row]), ds.pos_idx[player_row],
                            sets, hz_coefs, assign, n_draws, horizon,
                            start_is_first_season=False,
                            n_history=int(grid.n_history[player_row]),
                            is_rookie=False, minutes_split=minutes_split,
                            mpg_offset=offset)


def simulate_rookie_pt(model, ds, x_row, gbm_offset, age0, year0, pos_idx,
                       player_id, n_draws=2000, horizon=12, seed=0,
                       n_param_sets=64, minutes_split=None) -> Projection:
    """Rookie: `ell_0` from the prior, `u_0` from its stationary spread."""
    rng = np.random.default_rng(seed + player_id)
    sets, hz_coefs, assign = _pt_param_blocks(model, n_draws, n_param_sets, rng)
    p = model.fit.params
    ell_mean = x_row @ p.beta + gbm_offset + model.init_basis(np.array([age0]))[0] @ model.fit.c_init_coefs
    Du = p.transient_stationary()
    ell, u = _draw_joint_pt(rng, ell_mean, np.zeros(S), p.Sigma_ell0,
                            np.zeros((S, S)), Du, n_draws)
    return _roll_forward_pt(model, None, rng, ell, u, age0, year0, player_id, pos_idx,
                            sets, hz_coefs, assign, n_draws, horizon,
                            start_is_first_season=True, n_history=0, is_rookie=True,
                            minutes_split=minutes_split, mpg_offset=0.0)


def _roll_forward_pt(model, ds, rng, ell, u, age0, year0, player_id, pos_idx,
                     sets, hz_coefs, assign, n_draws, horizon,
                     start_is_first_season, n_history, is_rookie,
                     minutes_split=None, mpg_offset=0.0) -> Projection:
    basis = model.age_basis
    drift_coefs = model.fit.delta_league + model.fit.delta_pos[pos_idx]   # (nb, S)

    step0 = 0 if start_is_first_season else 1
    ages = age0 + step0 + np.arange(horizon, dtype=float)
    years = year0 + step0 + np.arange(horizon, dtype=int)
    alive = np.zeros((n_draws, horizon), dtype=bool)
    played = np.zeros((n_draws, horizon), dtype=bool)
    theta_out = np.zeros((n_draws, horizon, S))
    poss_out = np.zeros((n_draws, horizon))
    games_out = np.zeros((n_draws, horizon))
    box = {c: np.zeros((n_draws, horizon)) for c in COUNT_NAMES}
    absence = getattr(model, "absence", None)

    Phi = np.stack([s.Phi for s in sets])[assign]
    phi_nb = np.stack([s.phi for s in sets])[assign]
    sigp = np.array([s.sigma_poss for s in sets])[assign]
    sigp_inj = np.array([getattr(s, "sigma_poss_inj", 0.0) or s.sigma_poss
                         for s in sets])[assign]
    QellL = np.stack([np.linalg.cholesky(s.Q_ell() + 1e-10 * np.eye(S)) for s in sets])[assign]
    QuL = np.stack([np.linalg.cholesky(s.Q_u() + 1e-10 * np.eye(S)) for s in sets])[assign]
    hz = np.stack(hz_coefs)[assign] if len(hz_coefs) > 1 else \
        np.broadcast_to(hz_coefs[0], (n_draws, len(hz_coefs[0])))

    live = np.ones(n_draws, dtype=bool)
    age = np.full(n_draws, age0)
    cur_ell, cur_u = ell.copy(), u.copy()
    hz_inter = _hazard_inter(model.hazard)

    for h in range(horizon):
        if h > 0 or not start_is_first_season:
            theta = cur_ell + cur_u
            # `_p_survive` carries the old-age ramp, so no hard age cliff here.
            live &= rng.random(n_draws) < _p_survive(hz, age, theta, hz_inter)
            d = basis(np.array([age[0]]))[0] @ drift_coefs
            eta_ell = np.einsum("nij,nj->ni", QellL, rng.standard_normal((n_draws, S)))
            eta_u = np.einsum("nij,nj->ni", QuL, rng.standard_normal((n_draws, S)))
            cur_ell = cur_ell + d + eta_ell
            cur_u = Phi * cur_u + eta_u
            age = age + 1.0

        cur = cur_ell + cur_u
        alive[:, h] = live
        theta_out[:, h] = cur

        # Within-career absence (deviation #4), mirroring project.py: a live
        # draw can still miss this season and contribute a zero box score.
        if absence is not None:
            plays = live & (rng.random(n_draws)
                            < absence.p_survive(np.full(n_draws, ages[h]), cur))
        else:
            plays = live
        played[:, h] = plays

        # Two-regime availability noise (downside-only injury scale) + smooth
        # ceiling, mirroring project.py.
        inj = rng.random(n_draws) < _injury_prob(None, ages[h])
        z = rng.standard_normal(n_draws)
        sig = np.where(inj & (z < 0), sigp_inj, sigp)
        log_e = cur[:, AVAIL_IDX] + sig * z
        e_raw = np.exp(np.clip(log_e, 0.0, np.log(MAX_POSSESSIONS) + 1.0))
        e = np.clip(_soft_cap(e_raw), 1.0, MAX_POSSESSIONS)
        poss_out[:, h] = np.where(plays, e, 0.0)
        if minutes_split is not None:
            g_draw, _ = minutes_split.draw(rng, e, np.full(n_draws, ages[h]), offset=mpg_offset)
            games_out[:, h] = np.where(plays, g_draw, 0.0)

        counts = _volume_counts(rng, cur[:, VOLUME_IDX], e, phi_nb)
        for k, stat in enumerate(VOLUME_STATS):
            box[stat][:, h] = np.where(plays, counts[:, k], 0)
        for stat in ACCURACY_STATS:
            made_c, att_c = ACCURACY_PAIRS[stat]
            p_make = 1.0 / (1.0 + np.exp(-np.clip(cur[:, IDX[stat]], -12, 12)))
            box[made_c][:, h] = rng.binomial(box[att_c][:, h].astype(np.int64), p_make)

        if not live.any():
            break

    return Projection(player_id=player_id, ages=ages, season_years=years, alive=alive,
                      theta=theta_out, box=box, possessions=poss_out,
                      games=games_out, is_rookie=is_rookie, n_history=n_history,
                      played=played)
