"""Old vs candidate availability system, on identical folds and state draws.

Both arms use the same fold model, the same filtered states and the same per-
player RNG seed, so the only thing that differs is how the availability draw is
made: the shipping `log-possessions -> MinutesSplit` chain versus the joint
`severity -> GP/MPG` system.  Scored against the exact quantities the interface
displays, with the Session-2 acceptance targets and subgroups.

Run:  .venv/bin/python -m career_model.validate.availability_ab --cutoffs 2018,2023
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, PANEL_PATH, AVAIL_IDX, S
from ..pipeline import filtered_states
from ..simulate import project, derive, availability as AV
from .ab_availability import _fold
from .calibration import crps_ensemble, pit as pit_fn, pit_shape

PERG = {"pts_per_game": "ppg", "reb_per_game": "rpg", "ast_per_game": "apg",
        "minutes_per_game": "mpg", "games": "gp"}


def _truth(panel):
    p = panel[panel["possessions"] > 0].copy()
    p["mpg"] = (p["possessions"] / 2.02) / p["games_played"].clip(lower=1)
    p["ppg"] = (p["ftm"] + 2 * p["fgm_2p"] + 3 * p["fgm_3p"]) / p["games_played"].clip(lower=1)
    p["rpg"] = (p["oreb"] + p["dreb"]) / p["games_played"].clip(lower=1)
    p["apg"] = p["ast"] / p["games_played"].clip(lower=1)
    t = {}
    for r in p.itertuples():
        t[(int(r.player_id), int(r.season_year))] = dict(
            gp=float(r.games_played), mpg=float(r.mpg), ppg=float(r.ppg),
            rpg=float(r.rpg), apg=float(r.apg))
    appeared = set(t.keys())
    return t, appeared


def _subgroups(grid, i, last, quality, q_top_decile):
    poss, g = grid.exposure[i, last], grid.games[i, last]
    mpg = (poss / 2.02) / g if g > 0 else 0.0
    share = g / 82.0
    sev = int(AV.classify_severity(np.array([share]))[0])
    return {
        "mpg_gt30": mpg > 30, "mpg_gt34": mpg > 34, "gp_lt50": g < 50,
        "moderate_last": sev == AV.SEV_MODERATE, "severe_last": sev == AV.SEV_SEVERE,
        "return_injury": sev == AV.SEV_SEVERE,
        "age30plus": float(grid.age[i, last]) >= 30.0,
        "top_decile_quality": quality >= q_top_decile,
    }


def run(cutoffs, n_draws=800, max_h=5):
    panel = pd.read_parquet(PANEL_PATH)
    truth, appeared = _truth(panel)
    recs, appr = [], []
    for cutoff in cutoffs:
        t0 = time.time()
        model, ds = _fold(cutoff, None)
        filt = filtered_states(ds, model.fit)
        ms = derive.fit_minutes_split(ds.panel, verbose=False)
        ib = project.fit_injury_rate(ds.panel)
        aq = project.fit_avail_quality_aging(ds, filt, model.hazard)
        avail = AV.fit_availability(ds, filt, model.hazard)
        gamma = model.hazard.coef[3:3 + S].copy(); gamma[AVAIL_IDX] = 0.0
        grid = ds.grid
        rows = [i for i in range(grid.n_players)
                if grid.last_index[i] >= 0
                and int(grid.season_years[i, grid.last_index[i]]) == cutoff]
        quals = {i: float(filt.x1[i, int(grid.last_index[i])] @ gamma) for i in rows}
        q_top = float(np.quantile(list(quals.values()), 0.9))
        print(f"[avail-ab] cutoff {cutoff}: {len(rows)} players", flush=True)

        for i in rows:
            pid = int(grid.player_ids[i])
            sg = _subgroups(grid, i, int(grid.last_index[i]), quals[i], q_top)
            for arm, kw in (("old", dict(minutes_split=ms, injury_beta=ib, avail_quality=aq)),
                            ("new", dict(minutes_split=ms, injury_beta=ib,
                                         avail_quality=aq, avail_system=avail))):
                proj = project.simulate(model, ds, i, filt, n_draws=n_draws,
                                        horizon=max_h, seed=0, **kw)
                for h in range(max_h):
                    year = cutoff + h + 1
                    pl = proj.played[:, h]
                    surv = proj.alive[:, h]
                    # appearance / career-active
                    appr.append({"cutoff": cutoff, "arm": arm, "horizon": h + 1,
                                 "kind": "appears", "p": float(pl.mean()),
                                 "y": (pid, year) in appeared, **sg})
                    career_actual = any((pid, year + k) in appeared for k in range(0, 8))
                    appr.append({"cutoff": cutoff, "arm": arm, "horizon": h + 1,
                                 "kind": "career", "p": float(surv.mean()),
                                 "y": career_actual, **sg})
                    key = (pid, year)
                    if key not in truth or pl.sum() < 40:
                        continue
                    cols = {c: proj.box[c][pl, h] for c in project.COUNT_NAMES}
                    comp = derive.derive_composites(cols, proj.possessions[pl, h],
                                                    games=proj.games[pl, h])
                    comp["games"] = proj.games[pl, h]
                    cap_hit = float((proj.games[pl, h] >= 81).mean())
                    for stat, short in PERG.items():
                        s = np.asarray(comp[stat], float)
                        s = s[np.isfinite(s)]
                        y = truth[key][short]
                        if len(s) < 40 or not np.isfinite(y):
                            continue
                        lo50, hi50 = np.percentile(s, [25, 75])
                        lo80, hi80 = np.percentile(s, [10, 90])
                        lo95, hi95 = np.percentile(s, [2.5, 97.5])
                        recs.append({
                            "cutoff": cutoff, "arm": arm, "horizon": h + 1,
                            "stat": short, "y": y, "median": float(np.median(s)),
                            "crps": crps_ensemble(s, y),
                            "pit": float(pit_fn(s[None, :], np.array([y]))[0]),
                            "in50": lo50 <= y <= hi50, "in80": lo80 <= y <= hi80,
                            "in95": lo95 <= y <= hi95, "cap_hit": cap_hit, **sg})
        print(f"[avail-ab] cutoff {cutoff} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(recs); ap = pd.DataFrame(appr)
    df.to_parquet(OUTPUT_DIR / "availability_ab_scores.parquet", index=False)
    ap.to_parquet(OUTPUT_DIR / "availability_ab_appearance.parquet", index=False)
    _summarise(df, ap)


def _summarise(df, ap):
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_score_rows": int(len(df))}

    def agg(g):
        return pd.Series({"n": len(g), "crps": g["crps"].mean(),
                          "mae": (g["median"] - g["y"]).abs().mean(),
                          "median_bias": (g["median"] - g["y"]).median(),
                          "cover_50": g["in50"].mean(), "cover_80": g["in80"].mean(),
                          "cover_95": g["in95"].mean(),
                          "pit_shape": pit_shape(g["pit"].to_numpy())})

    out["by_stat_arm"] = (df.groupby(["stat", "arm"]).apply(agg, include_groups=False)
                          .reset_index().round(4).to_dict("records"))
    out["by_stat_arm_horizon"] = (
        df.groupby(["stat", "arm", "horizon"]).apply(agg, include_groups=False)
        .reset_index().round(4).to_dict("records"))
    # normalized aggregate CRPS per arm (each stat scaled by the old-arm mean)
    scale = df[df.arm == "old"].groupby("stat")["crps"].mean()
    df["crps_n"] = df["crps"] / df["stat"].map(scale)
    out["aggregate_norm_crps"] = df.groupby("arm")["crps_n"].mean().round(5).to_dict()
    out["cap_hit_freq"] = df[df.stat == "gp"].groupby("arm")["cap_hit"].mean().round(4).to_dict()

    # subgroups for GP and MPG
    subs = ["mpg_gt30", "mpg_gt34", "gp_lt50", "moderate_last", "severe_last",
            "return_injury", "age30plus", "top_decile_quality"]
    out["subgroups"] = {}
    for sub in subs:
        rows = []
        for stat in ("gp", "mpg", "ppg"):
            d = df[(df["stat"] == stat) & (df[sub])]
            for arm in ("old", "new"):
                a = d[d.arm == arm]
                if len(a) < 20:
                    continue
                rows.append({"stat": stat, "arm": arm, "n": int(len(a)),
                             "median_bias": round(float((a["median"] - a["y"]).median()), 3),
                             "crps": round(float(a["crps"].mean()), 3),
                             "cover_80": round(float(a["in80"].mean()), 3)})
        out["subgroups"][sub] = rows

    # appearance / career Brier by horizon
    ap["se"] = (ap["p"] - ap["y"].astype(float)) ** 2
    out["brier"] = {}
    for kind in ("appears", "career"):
        rows = []
        for (arm, h), g in ap[ap.kind == kind].groupby(["arm", "horizon"]):
            rows.append({"arm": arm, "horizon": int(h), "brier": round(g["se"].mean(), 4),
                         "pred": round(g["p"].mean(), 3), "act": round(g["y"].mean(), 3)})
        out["brier"][kind] = rows

    (OUTPUT_DIR / "availability_ab_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[avail-ab] wrote summary ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", default="2018,2023")
    ap.add_argument("--draws", type=int, default=800)
    args = ap.parse_args()
    run([int(c) for c in args.cutoffs.split(",")], n_draws=args.draws)


if __name__ == "__main__":
    main()
