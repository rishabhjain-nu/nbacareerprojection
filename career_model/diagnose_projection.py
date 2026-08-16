"""First-year projection decomposition (`python -m career_model diagnose_projection`).

Answers "why is season T+1 what it is" by walking the same arithmetic the
simulator runs, one contribution at a time, for the first projected season of an
established player.

The availability latent state moves in log-possessions and every contribution is
**additive there**, so the decomposition is exact:

    log_poss_next = start
                  + mean_reversion          (A-1)(start - m_eff)
                  + aging                    delta(age)
                  + age_by_quality           quality increment
    possessions   = exp(log_poss_next)  then injury mixture + soft cap
    games, mpg    = MinutesSplit(possessions, age)
    per-100 rates = exp(volume state)               (production)
    PPG/RPG/...    = rates x possessions / games

`start` is itself the filtered state plus the availability empirical-Bayes shift,
and `m_eff` is the empirical-Bayes reversion target -- both reported.  The
observable medians (possessions, GP, MPG, PPG, ...) come from actually running
`project.simulate` with the shipping settings, so the tool reports exactly what
the interface shows; `tests/test_diagnose_projection.py` asserts that match.

The stochastic pieces (injury mixture, sampling noise, the soft cap, and the
median-vs-mean gap of the lognormal) are reported as measured shifts between the
deterministic centre and the simulated median, not folded into the additive
latent decomposition, so the additive part stays exact.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .config import AVAIL_IDX, OUTPUT_DIR, PROJECTION_DIR, VOLUME_STATS, STATE_NAMES
from .model.dataset import load as load_dataset
from .model.fit_kf import LEVEL_PARAM
from .pipeline import filtered_states, load as load_model
from .simulate import derive, project


def _round(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def _last_observed(grid, r):
    last = int(grid.last_index[r])
    return {
        "index": last,
        "season_year": int(grid.season_years[r, last]),
        "age": float(grid.age[r, last]),
        "possessions": float(grid.exposure[r, last]),
        "games": float(grid.games[r, last]),
        "points": float(grid.counts["ftm"][r, last]
                        + 2 * grid.counts["fgm_2p"][r, last]
                        + 3 * grid.counts["fgm_3p"][r, last]),
    }


def _median_conditional(proj, h=0):
    """Median per-game / total / rate quantities for horizon `h`, conditional on
    the player having played -- exactly the conditioning `summarise` uses."""
    played = proj.played[:, h]
    if played.sum() < 10:
        return None
    cols = {c: proj.box[c][played, h] for c in project.COUNT_NAMES}
    poss = proj.possessions[played, h]
    g = proj.games[played, h]
    comp = derive.derive_composites(cols, poss, translator=None,
                                    games=g if g.any() else None)
    comp["minutes"] = poss * project.MIN_PER_POSSESSION
    out = {}
    for k in ("pts_per_game", "reb_per_game", "ast_per_game", "stl_per_game",
              "blk_per_game", "minutes_per_game", "games", "possessions",
              "pts_per100", "reb_per100", "ast_per100", "pts"):
        if k in comp:
            out[k] = float(np.median(comp[k]))
    return out


def decompose(model, ds, filt, r, minutes_split, injury_beta, avail_quality,
              n_draws=2000, seed=0, role_model=None):
    """Full first-season decomposition for player row `r`."""
    grid = ds.grid
    p = model.fit.params
    last = int(grid.last_index[r])
    if last < 0:
        raise ValueError("player has no observed seasons")
    age_T = float(grid.age[r, last])
    n_hist = int(grid.n_history[r])
    x1 = filt.x1[r, last]
    A_av = float(p.A[AVAIL_IDX])

    # ---- 3. empirical-Bayes reversion target (§ star-decline) -------------
    m_eff = filt.x2[r, last].copy()
    if n_hist >= 1:
        m_eff, _m_model = project._eb_reversion_target(model, ds, filt, r)
    target_av = float(m_eff[AVAIL_IDX])

    # ---- 3/4. availability own-record EB (starting state + target shift) --
    start_av = float(x1[AVAIL_IDX])
    own = project._avail_own_level(model, ds, filt, r)
    w_own = 0.0
    own_level = None
    if own is not None:
        level, w_own, L_T = own
        own_level = float(np.exp(level))
        start_av = start_av + w_own * (level - float(x1[AVAIL_IDX]))
        target_av = target_av + w_own * ((level - L_T) - target_av)

    # ---- 6. aging increment on availability -------------------------------
    coefs = project._delta_coefs_for(model, ds, r)
    if LEVEL_PARAM:
        c_now = model.age_basis(np.array([age_T]))[0] @ coefs
        c_nxt = model.age_basis(np.array([age_T + 1.0]))[0] @ coefs
        aging_av = float((c_nxt - A_av * c_now)[AVAIL_IDX])
    else:
        aging_av = float((model.age_basis(np.array([age_T]))[0] @ coefs)[AVAIL_IDX])

    # ---- 7. age-by-quality availability contribution ----------------------
    quality_av = 0.0
    if avail_quality is not None:
        quality_av = float(avail_quality.increment(age_T + 1.0, x1[None, :])[0])

    # ---- additive latent decomposition (exact, in log-possessions) --------
    reversion_av = (A_av - 1.0) * (start_av - target_av)
    next_av = start_av + reversion_av + aging_av + quality_av
    center_poss = float(np.exp(min(next_av, np.log(project.MAX_POSSESSIONS))))

    # ---- 8/9. injury contribution + simulated possessions -----------------
    # Measured as the shift from the deterministic centre to the simulated
    # median, split into "injury mixture" (median with injury off vs on) and
    # "noise+cap+conditioning" (centre vs injury-off median).
    # Match the shipping config (Session 4): Gaussian state innovation.  The
    # S3-B mixture was disabled after it failed its aggregate CRPS/coverage gate,
    # so the decomposition must not add role-change bands the store no longer has.
    from .simulate.precompute import SHIP_CONFIG
    _mix = dict(innovation=SHIP_CONFIG["state_innovation"])
    proj_full = project.simulate(model, ds, r, filt, n_draws=n_draws, seed=seed,
                                 minutes_split=minutes_split, injury_beta=injury_beta,
                                 avail_quality=avail_quality, **_mix)
    proj_noinj = project.simulate(model, ds, r, filt, n_draws=n_draws, seed=seed,
                                  minutes_split=minutes_split, injury_beta=None,
                                  avail_quality=avail_quality, **_mix)
    played = proj_full.played[:, 0]
    med_poss = float(np.median(proj_full.possessions[played, 0])) if played.sum() >= 10 else None
    med_poss_noinj = (float(np.median(proj_noinj.possessions[proj_noinj.played[:, 0], 0]))
                      if proj_noinj.played[:, 0].sum() >= 10 else None)
    p_inj = project._injury_prob(injury_beta, age_T + 1.0)
    if injury_beta is not None:
        rec = project._injury_record(ds, r)
        if rec is not None:
            k_flag, n_obs = rec
            p_inj = float(np.clip((project.INJ_PROPENSITY_STRENGTH * p_inj + k_flag)
                                  / (project.INJ_PROPENSITY_STRENGTH + n_obs), 0.03, 0.7))
    injury_shift = (np.log(med_poss / med_poss_noinj)
                    if med_poss and med_poss_noinj else None)

    # ---- 10/11. survival + appearance -------------------------------------
    sc = project.survival_curve(proj_full)
    p_active = float(sc["p_active"].iloc[0])
    p_play = float(sc["p_play"].iloc[0])

    # ---- 12/13/14. observable medians (what the UI shows) -----------------
    med = _median_conditional(proj_full, 0)
    lastobs = _last_observed(grid, r)

    # ---- PPG difference decomposition -------------------------------------
    # PPG = scoring_rate(pts/poss) x poss_per_min(=2.02, fixed) x MPG.  GP and
    # appearance do NOT enter PPG (it is per game); they scale SEASON totals, so
    # they are reported against season points instead.
    ppg_decomp = None
    if med and lastobs["games"] > 0 and lastobs["possessions"] > 0:
        sr_last = lastobs["points"] / lastobs["possessions"]
        mpg_last = (lastobs["possessions"] * project.MIN_PER_POSSESSION) / lastobs["games"]
        ppg_last = lastobs["points"] / lastobs["games"]
        sr_proj = med["pts_per100"] / 100.0
        mpg_proj = med["minutes_per_game"]
        ppg_proj = med["pts_per_game"]
        gp_last, gp_proj = lastobs["games"], med["games"]

        def dlog(a, b):
            return float(np.log(a / b)) if (a > 0 and b > 0) else None
        ppg_decomp = {
            "ppg_last_observed": _round(ppg_last, 2),
            "ppg_projected_median": _round(ppg_proj, 2),
            "delta_ppg": _round(ppg_proj - ppg_last, 2),
            "log_decomposition_of_PPG": {
                "scoring_rate_pts_per_poss": _round(dlog(sr_proj, sr_last)),
                "poss_per_min_(fixed_2.02)": 0.0,
                "minutes_per_game": _round(dlog(mpg_proj, mpg_last)),
                "_note": "these three sum to delta-log(PPG); GP/appearance do "
                         "not affect PPG, only season totals",
                "sum": _round((dlog(sr_proj, sr_last) or 0) + (dlog(mpg_proj, mpg_last) or 0)),
                "actual_delta_log_ppg": _round(dlog(ppg_proj, ppg_last)),
            },
            "additional_factors_for_SEASON_POINTS": {
                "games_played": _round(dlog(gp_proj, gp_last)),
                "appearance_P(plays)": _round(np.log(p_play)) if p_play > 0 else None,
            },
        }

    return {
        "player_id": int(grid.player_ids[r]),
        "last_observed_season": {k: _round(v, 3) for k, v in lastobs.items()},
        "filtered_state_at_cutoff": {STATE_NAMES[j]: _round(float(x1[j])) for j in range(len(x1))},
        "availability_decomposition_log_possessions": {
            "1_filtered_start": _round(float(x1[AVAIL_IDX])),
            "2_eb_start_adjustment": _round(start_av - float(x1[AVAIL_IDX])),
            "3_effective_start": _round(start_av),
            "4_effective_reversion_target": _round(target_av),
            "5_mean_reversion_contribution": _round(reversion_av),
            "6_aging_contribution": _round(aging_av),
            "7_age_by_quality_contribution": _round(quality_av),
            "8_deterministic_next_log_poss": _round(next_av),
            "check_sum": _round(start_av + reversion_av + aging_av + quality_av),
            "own_record_level_possessions": _round(own_level, 1),
            "own_record_weight_w": _round(w_own),
        },
        "possessions": {
            "deterministic_center": _round(center_poss, 1),
            "9_projected_median": _round(med_poss, 1),
            "injury_probability": _round(p_inj),
            "injury_median_shift_logposs": _round(injury_shift),
            "noise_cap_conditioning_shift_logposs":
                _round(np.log(med_poss_noinj / center_poss)
                       if (med_poss_noinj and center_poss) else None),
        },
        "survival": {
            "10_P_career_active": _round(p_active),
            "11_P_appears_given_active": _round(p_play / p_active if p_active > 0 else None),
            "P_appears_unconditional": _round(p_play),
        },
        "observable_medians_UI": {
            "12_games": _round(med["games"], 1) if med else None,
            "12_minutes_per_game": _round(med["minutes_per_game"], 1) if med else None,
            "13_pts_per100": _round(med["pts_per100"], 1) if med else None,
            "13_reb_per100": _round(med["reb_per100"], 1) if med else None,
            "13_ast_per100": _round(med["ast_per100"], 1) if med else None,
            "14_PPG": _round(med["pts_per_game"], 1) if med else None,
            "14_RPG": _round(med["reb_per_game"], 1) if med else None,
            "14_APG": _round(med["ast_per_game"], 1) if med else None,
            "14_SPG": _round(med["stl_per_game"], 2) if med else None,
            "14_BPG": _round(med["blk_per_game"], 2) if med else None,
            "14_MPG": _round(med["minutes_per_game"], 1) if med else None,
            "14_GP": _round(med["games"], 1) if med else None,
        },
        "ppg_vs_last_observed": ppg_decomp,
    }


# ---------------------------------------------------------------------------
def _prep():
    model = load_model()
    ds = load_dataset()
    filt = filtered_states(ds, model.fit)
    minutes_split = derive.fit_minutes_split(ds.panel, verbose=False)
    injury_beta = project.fit_injury_rate(ds.panel)
    avail_quality = project.fit_avail_quality_aging(ds, filt, model.hazard)
    role_model = project.fit_role_change(ds, filt)   # S3-B, shipped
    return model, ds, filt, minutes_split, injury_beta, avail_quality, role_model


def _row_of(grid, pid):
    hit = np.flatnonzero(grid.player_ids == pid)
    if not len(hit):
        raise SystemExit(f"player_id {pid} not in the fitted panel")
    return int(hit[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", type=int, default=None, help="player_id")
    ap.add_argument("--batch", action="store_true",
                    help="decompose every active cutoff-season player")
    ap.add_argument("--out", type=str, default=None, help="write JSON here")
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    model, ds, filt, ms, ib, aq, rc = _prep()
    grid = ds.grid
    last_year = int(grid.season_years[grid.observed].max())

    if args.batch:
        rows = [i for i in range(grid.n_players)
                if grid.last_index[i] >= 0
                and int(grid.season_years[i, grid.last_index[i]]) == last_year]
        out = {}
        t0 = time.time()
        for n, r in enumerate(rows):
            try:
                d = decompose(model, ds, filt, r, ms, ib, aq, n_draws=args.draws, role_model=rc)
                out[str(d["player_id"])] = d
            except Exception as e:  # noqa: BLE001
                out[str(int(grid.player_ids[r]))] = {"error": str(e)}
            if (n + 1) % 100 == 0:
                print(f"[diagnose] {n + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        dest = args.out or str(OUTPUT_DIR / "diagnose_batch.json")
        with open(dest, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"[diagnose] wrote {len(out)} players to {dest}")
        return

    if args.player is None:
        raise SystemExit("pass --player <id> or --batch")
    d = decompose(model, ds, filt, _row_of(grid, args.player), ms, ib, aq,
                  n_draws=args.draws, role_model=rc)
    text = json.dumps(d, indent=2)
    print(text)
    if args.out:
        Path_out = args.out
        with open(Path_out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
