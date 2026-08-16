"""S3 candidates vs shipping, on identical folds and state draws.

Arms (all on the same fold fit, same seed; availability held at the shipping
Gaussian path so any rate-coverage change is from the state model, not from
injected availability variance -- Session-2 finding #6):

  shipping   Gaussian innovations, shipped EB reversion target
  diffuse_m  Candidate A: diffuse m-prior filter, revert to the learned level
  student_t  Candidate B: Student-t innovations (nu)
  mixture    Candidate B: two-component role-change innovation mixture

Scores the latent-driven per-100 rates (AST/2PA/3PA/FTA) and the per-game
outputs (PPG/RPG/APG/MPG), conditional on appearing, with CRPS / median bias /
50-80-95 coverage / randomized PIT, by horizon and by the Session-3 subgroups.

Run:  .venv/bin/python -m career_model.validate.state_dev_ab --cutoffs 2018,2023
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, PANEL_PATH, AVAIL_IDX, S, IDX
from ..pipeline import filtered_states
from ..simulate import project, derive
from .ab_availability import _fold
from .calibration import crps_ensemble, pit as pit_fn, pit_shape

RATES = {"ast_per100": "ast", "fga_2p_per100": "2pa", "fga_3p_per100": "3pa",
         "fta_per100": "fta"}
PERG = {"pts_per_game": "ppg", "reb_per_game": "rpg", "ast_per_game": "apg",
        "minutes_per_game": "mpg"}


def _truth(panel):
    p = panel[panel["possessions"] > 0].copy()
    e = p["possessions"].clip(lower=1)
    g = p["games_played"].clip(lower=1)
    t = {}
    for r in p.itertuples():
        ee = max(r.possessions, 1.0); gg = max(r.games_played, 1.0)
        pts = r.ftm + 2 * r.fgm_2p + 3 * r.fgm_3p
        t[(int(r.player_id), int(r.season_year))] = dict(
            ast=100 * r.ast / ee, **{"2pa": 100 * r.fga_2p / ee,
            "3pa": 100 * r.fga_3p / ee, "fta": 100 * r.fta / ee},
            ppg=pts / gg, rpg=(r.oreb + r.dreb) / gg, apg=r.ast / gg,
            mpg=(r.possessions / 2.02) / gg)
    return t, set(t.keys())


def _subgroups(ds, filt, grid, rows, gamma):
    """Cutoff-only subgroup tags per player row."""
    idxr = [IDX[n] for n in project.ROLE_DIMS]
    pts_i = [IDX["fga_2p"], IDX["fga_3p"], IDX["fta"]]
    quals, scorers = {}, {}
    tags = {}
    for i in rows:
        last = int(grid.last_index[i])
        st = filt.x1[i, last]
        quals[i] = float(st @ gamma)
        scorers[i] = float(st[IDX["fga_2p"]] + st[IDX["fga_3p"]] + st[IDX["fta"]])
    q_top = np.quantile(list(quals.values()), 0.9)
    s_top = np.quantile(list(scorers.values()), 0.9)
    try:
        picks = {int(r.player_id): float(r.draft_pick) for r in ds.priors.itertuples()}
    except Exception:
        picks = {}
    for i in rows:
        last = int(grid.last_index[i])
        obs = np.flatnonzero(grid.observed[i])
        st = filt.x1[i, last]
        d_shot = d_ast = 0.0
        outlier = False
        if len(obs) >= 2:
            prev = filt.x1[i, obs[-2]]
            d_shot = abs(st[IDX["fga_3p"]] - prev[IDX["fga_3p"]])
            d_ast = abs(st[IDX["ast"]] - prev[IDX["ast"]])
        if len(obs) >= 4:
            hist = np.stack([filt.x1[i, t] for t in obs[:-1]])
            base = hist[:, pts_i].sum(1).mean()
            outlier = abs(st[pts_i].sum() - base) > 0.30
        vol = 0.0
        if len(obs) >= 2:
            vol = float(np.max(np.abs(st[idxr] - filt.x1[i, obs[-2]][idxr])))
        pid = int(grid.player_ids[i])
        tags[i] = {
            "top_quality": quals[i] >= q_top,
            "top_scorer": scorers[i] >= s_top,
            "shot_breakout": d_shot > 0.30,
            "assist_breakout": d_ast > 0.30,
            "stable_vet": (float(grid.age[i, last]) >= 28 and len(obs) >= 5 and vol < 0.15),
            "one_season_outlier": outlier,
            "late_draft": picks.get(pid, 30) > 20,
            "high_draft": picks.get(pid, 30) <= 10,
        }
    return tags


def run(cutoffs, n_draws=600, max_h=5):
    panel = pd.read_parquet(PANEL_PATH)
    truth, appeared = _truth(panel)
    recs = []
    for cutoff in cutoffs:
        t0 = time.time()
        model, ds = _fold(cutoff, None)
        filt = filtered_states(ds, model.fit)
        filt_diffuse = filtered_states(ds, model.fit, m_prior_scale=25.0)
        ms = derive.fit_minutes_split(ds.panel, verbose=False)
        ib = project.fit_injury_rate(ds.panel)
        aq = project.fit_avail_quality_aging(ds, filt, model.hazard)
        rc = project.fit_role_change(ds, filt)
        gamma = model.hazard.coef[3:3 + S].copy(); gamma[AVAIL_IDX] = 0.0
        grid = ds.grid
        rows = [i for i in range(grid.n_players)
                if grid.last_index[i] >= 0
                and int(grid.season_years[i, grid.last_index[i]]) == cutoff]
        tags = _subgroups(ds, filt, grid, rows, gamma)
        arms = {
            "shipping": dict(filtx=filt, kw=dict()),
            "diffuse_m": dict(filtx=filt_diffuse, kw=dict(use_eb=False)),
            "student_t": dict(filtx=filt, kw=dict(innovation="student_t", t_nu=5.0)),
            "mixture": dict(filtx=filt, kw=dict(innovation="mixture", role_model=rc,
                                                role_scale=4.0)),
        }
        print(f"[state-ab] cutoff {cutoff}: {len(rows)} players", flush=True)
        for i in rows:
            pid = int(grid.player_ids[i])
            base = dict(minutes_split=ms, injury_beta=ib, avail_quality=aq)
            for arm, cfg in arms.items():
                proj = project.simulate(model, ds, i, cfg["filtx"], n_draws=n_draws,
                                        horizon=max_h, seed=0, **base, **cfg["kw"])
                for h in range(max_h):
                    key = (pid, cutoff + h + 1)
                    pl = proj.played[:, h]
                    if key not in truth or pl.sum() < 40:
                        continue
                    cols = {c: proj.box[c][pl, h] for c in project.COUNT_NAMES}
                    comp = derive.derive_composites(cols, proj.possessions[pl, h],
                                                    games=proj.games[pl, h])
                    for stat, short in {**RATES, **PERG}.items():
                        s = np.asarray(comp[stat], float); s = s[np.isfinite(s)]
                        y = truth[key][short]
                        if len(s) < 40 or not np.isfinite(y):
                            continue
                        lo80, hi80 = np.percentile(s, [10, 90])
                        lo50, hi50 = np.percentile(s, [25, 75])
                        lo95, hi95 = np.percentile(s, [2.5, 97.5])
                        recs.append({"cutoff": cutoff, "arm": arm, "horizon": h + 1,
                                     "stat": short, "y": y, "median": float(np.median(s)),
                                     "crps": crps_ensemble(s, y),
                                     "pit": float(pit_fn(s[None, :], np.array([y]))[0]),
                                     "in50": lo50 <= y <= hi50, "in80": lo80 <= y <= hi80,
                                     "in95": lo95 <= y <= hi95, **tags[i]})
        print(f"[state-ab] cutoff {cutoff} done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(recs)
    df.to_parquet(OUTPUT_DIR / "state_dev_ab_scores.parquet", index=False)
    _summarise(df)


def _summarise(df):
    def agg(g):
        return pd.Series({"n": len(g), "crps": g["crps"].mean(),
                          "median_bias": (g["median"] - g["y"]).median(),
                          "cover_50": g["in50"].mean(), "cover_80": g["in80"].mean(),
                          "cover_95": g["in95"].mean(),
                          "pit": pit_shape(g["pit"].to_numpy())})
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "n": int(len(df))}
    out["by_stat_arm"] = (df.groupby(["stat", "arm"]).apply(agg, include_groups=False)
                          .reset_index().round(4).to_dict("records"))
    out["by_stat_arm_horizon"] = (
        df[df.stat.isin(["apg", "ast", "ppg"])]
        .groupby(["stat", "arm", "horizon"]).apply(agg, include_groups=False)
        .reset_index().round(4).to_dict("records"))
    scale = df[df.arm == "shipping"].groupby("stat")["crps"].mean()
    df["crps_n"] = df["crps"] / df["stat"].map(scale)
    out["aggregate_norm_crps"] = df.groupby("arm")["crps_n"].mean().round(5).to_dict()
    out["subgroups"] = {}
    for sub in ["top_scorer", "top_quality", "shot_breakout", "assist_breakout",
                "stable_vet", "one_season_outlier", "late_draft", "high_draft"]:
        rows = []
        for stat in ["ppg", "apg", "ast", "3pa"]:
            d = df[(df.stat == stat) & (df[sub])]
            for arm in df.arm.unique():
                a = d[d.arm == arm]
                if len(a) < 25:
                    continue
                rows.append({"stat": stat, "arm": arm, "n": int(len(a)),
                             "median_bias": round(float((a["median"] - a["y"]).median()), 3),
                             "crps": round(float(a["crps"].mean()), 3),
                             "cover_80": round(float(a["in80"].mean()), 3)})
        out["subgroups"][sub] = rows
    (OUTPUT_DIR / "state_dev_ab_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[state-ab] wrote summary ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", default="2018,2023")
    ap.add_argument("--draws", type=int, default=600)
    args = ap.parse_args()
    run([int(c) for c in args.cutoffs.split(",")], n_draws=args.draws)


if __name__ == "__main__":
    main()
