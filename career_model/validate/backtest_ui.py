"""Rolling-origin backtest of the *exact quantities the interface displays*.

`backtest.py` scores the per-100 latent-rate outputs; this scores what a reader
actually sees on a player card -- PPG, RPG, APG, SPG, BPG, MPG, GP, season
points, season minutes -- plus the two probabilities, P(appears this season) and
P(career still active).

Leakage control is inherited: each cutoff is a full fold-local refit (reusing the
cached folds under `outputs/folds/`), every display device (minutes split, injury
rate, quality-aging, BPM translator) is fitted on the truncated frame, and only
players whose last training season is the cutoff are projected.

Every quantity is scored two ways:

  * **conditional on appearing** -- the "if he plays" number the card shows;
    samples are the played draws, actuals only the seasons the player appeared;
  * **unconditional** -- season totals and GP with a zero written in for a
    missed season or a career that has ended, so the hazard/absence models are
    on the hook for the mass they put on not playing.

Per-game rates are scored conditional only (a per-game rate is undefined for a
zero-game season); season totals and GP are scored both ways.

Recorded per (cutoff, player, horizon, quantity, mode): CRPS, MAE, median bias,
50/80/95 coverage, randomized PIT, plus a rich set of pre-cutoff covariates so
every breakout the report needs is a groupby.  Appearance and career-active
probabilities are scored separately as Brier + calibration.  Short horizons from
recent cutoffs are kept even when the five-year horizon is not yet observable.

Output (machine-readable):
  outputs/backtest_ui_scores.parquet       one row per (cutoff,player,h,quantity,mode)
  outputs/backtest_ui_appearance.parquet   one row per (cutoff,player,h,kind)
  outputs/backtest_ui_summary.json         headline breakouts

Run:  .venv/bin/python -m career_model backtest_ui --cutoffs 2016,2018
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..config import AVAIL_IDX, OUTPUT_DIR, PANEL_PATH, RAW_DIR
from ..pipeline import filtered_states
from ..simulate import derive, project
from . import calibration as cal
from .ab_availability import _fold

# The displayed per-game rates (conditional-on-appearing only).
PER_GAME = ["pts_per_game", "reb_per_game", "ast_per_game", "stl_per_game",
            "blk_per_game", "minutes_per_game"]
# Totals + GP: scored conditional AND unconditional.
TOTALS = ["season_points", "season_minutes", "games"]

POS = {0: "G", 1: "F", 2: "C"}


def _log(m):
    print(f"[backtest_ui] {m}", flush=True)


# ---------------------------------------------------------------------------
def _actuals(panel: pd.DataFrame) -> dict:
    """Per (player, year): appeared, per-game rates, season totals, GP, MPG."""
    df = panel[panel["possessions"] > 0].copy()
    pts = df["ftm"] + 2 * df["fgm_2p"] + 3 * df["fgm_3p"]
    reb = df["oreb"] + df["dreb"]
    g = df["games_played"].to_numpy(float)
    mins = df["possessions"].to_numpy(float) / 2.02
    out = {}
    for i, row in enumerate(df.itertuples()):
        gp = g[i]
        if gp <= 0:
            continue
        out[(int(row.player_id), int(row.season_year))] = {
            "appeared": True,
            "pts_per_game": pts.iloc[i] / gp, "reb_per_game": reb.iloc[i] / gp,
            "ast_per_game": row.ast / gp, "stl_per_game": row.stl / gp,
            "blk_per_game": row.blk / gp,
            "minutes_per_game": mins[i] / gp,
            "season_points": float(pts.iloc[i]), "season_minutes": float(mins[i]),
            "games": float(gp),
        }
    return out


def _covariates(grid, filt, i, gamma_skill, pts100_cut, pos_idx):
    """Pre-cutoff covariates for player row `i` (all knowable at the cutoff)."""
    last = int(grid.last_index[i])
    poss = float(grid.exposure[i, last]); gp = float(grid.games[i, last])
    mpg = (poss / 2.02) / gp if gp > 0 else np.nan
    # recent deltas: need the prior observed season
    obs = np.flatnonzero(grid.observed[i])
    prev = obs[-2] if len(obs) >= 2 else None

    def per100(t, stat):
        e = grid.exposure[i, t]
        return 100.0 * grid.counts[stat][i, t] / e if e > 0 else np.nan
    d_mpg = d_fga = d_ast = np.nan
    if prev is not None:
        pg = float(grid.games[i, prev]); pp = float(grid.exposure[i, prev])
        if pg > 0:
            d_mpg = mpg - (pp / 2.02) / pg
        d_fga = (per100(last, "fga_2p") + per100(last, "fga_3p")) - \
                (per100(prev, "fga_2p") + per100(prev, "fga_3p"))
        d_ast = per100(last, "ast") - per100(prev, "ast")
    pts100 = per100(last, "fgm_2p") * 2 + per100(last, "fgm_3p") * 3 + per100(last, "ftm")
    quality = float(filt.x1[i, last] @ gamma_skill)
    return {
        "position": POS.get(int(pos_idx), "F"),
        "last_mpg": mpg, "last_gp": gp, "last_poss": poss,
        "quality": quality, "injured_last": bool(grid.injury[i, last]),
        "recent_mpg_change": d_mpg, "recent_fga_change": d_fga,
        "recent_ast_change": d_ast,
        "top_decile_scorer": bool(pts100 >= pts100_cut),
        "gt30mpg": bool(mpg > 30), "gt34mpg": bool(mpg > 34), "lt50gp": bool(gp < 50),
    }


def score_cutoff(cutoff, panel, actuals, career_last, max_h=5, n_draws=800,
                 fit_kwargs=None):
    model, ds = _fold(cutoff, fit_kwargs)
    assert int(ds.panel["season_year"].max()) <= cutoff
    filt = filtered_states(ds, model.fit)
    ms = derive.fit_minutes_split(ds.panel, verbose=False)
    ib = project.fit_injury_rate(ds.panel)
    aq = project.fit_avail_quality_aging(ds, filt, model.hazard)
    gamma_skill = model.hazard.coef[3:3 + 14].copy(); gamma_skill[AVAIL_IDX] = 0.0
    grid = ds.grid

    rows = [i for i in range(grid.n_players)
            if grid.last_index[i] >= 0
            and int(grid.season_years[i, grid.last_index[i]]) == cutoff]
    # top-decile scorer cut, among cutoff-active players
    p100 = []
    for i in rows:
        last = int(grid.last_index[i]); e = grid.exposure[i, last]
        if e > 0:
            p100.append(100.0 * (grid.counts["fgm_2p"][i, last] * 2
                        + grid.counts["fgm_3p"][i, last] * 3
                        + grid.counts["ftm"][i, last]) / e)
    pts100_cut = np.percentile(p100, 90) if p100 else np.inf
    _log(f"cutoff {cutoff}: {len(rows)} active players")

    score_rows, appear_rows = [], []
    for i in rows:
        pid = int(grid.player_ids[i])
        proj = project.simulate(model, ds, i, filt, n_draws=n_draws, horizon=max_h,
                                seed=0, minutes_split=ms, injury_beta=ib,
                                avail_quality=aq)
        cov = _covariates(grid, filt, i, gamma_skill, pts100_cut, ds.pos_idx[i])
        for h in range(max_h):
            year = cutoff + h + 1
            if year > career_last:          # outcome not observed yet -> skip
                continue
            played = proj.played[:, h]
            appeared = (pid, year) in actuals
            # career active actual = player appears in this year or any later year
            still = any((pid, y) in actuals for y in range(year, career_last + 1))
            base = {"cutoff": cutoff, "player_id": pid, "horizon": h + 1,
                    "age": float(proj.ages[h]), **cov}

            # ---- appearance + career-active probabilities (Brier) ---------
            appear_rows.append(dict(base, kind="appears",
                                    p=float(played.mean()), actual=float(appeared)))
            appear_rows.append(dict(base, kind="career_active",
                                    p=float(proj.alive[:, h].mean()),
                                    actual=float(still)))

            # ---- box-score quantities -------------------------------------
            box = {c: proj.box[c][:, h] for c in project.COUNT_NAMES}
            pts_all = derive.points(box)                       # 0 where not played
            reb_all = box["oreb"] + box["dreb"]
            poss_all = proj.possessions[:, h]
            g_all = proj.games[:, h]
            mins_all = poss_all * project.MIN_PER_POSSESSION

            def add(quantity, samples, actual, mode):
                samples = np.asarray(samples, float)
                samples = samples[np.isfinite(samples)]
                if len(samples) < 30 or actual is None or not np.isfinite(actual):
                    return
                lo50, hi50 = np.percentile(samples, [25, 75])
                lo80, hi80 = np.percentile(samples, [10, 90])
                lo95, hi95 = np.percentile(samples, [2.5, 97.5])
                med = float(np.median(samples))
                score_rows.append(dict(
                    base, quantity=quantity, mode=mode, actual=float(actual),
                    median=med, crps=cal.crps_ensemble(samples, actual),
                    pit=float(cal.pit(samples[None, :], np.array([actual]))[0]),
                    abs_err=abs(med - actual), bias=med - actual,
                    inside_50=bool(lo50 <= actual <= hi50),
                    inside_80=bool(lo80 <= actual <= hi80),
                    inside_95=bool(lo95 <= actual <= hi95),
                    width_80=float(hi80 - lo80)))

            act = actuals.get((pid, year))
            # per-game rates: conditional on appearing only
            if appeared and played.sum() >= 30:
                pg = {"pts_per_game": pts_all[played] / g_all[played],
                      "reb_per_game": reb_all[played] / g_all[played],
                      "ast_per_game": box["ast"][played] / g_all[played],
                      "stl_per_game": box["stl"][played] / g_all[played],
                      "blk_per_game": box["blk"][played] / g_all[played],
                      "minutes_per_game": mins_all[played] / g_all[played]}
                for q in PER_GAME:
                    add(q, pg[q], act[q], "cond")
                add("season_points", pts_all[played], act["season_points"], "cond")
                add("season_minutes", mins_all[played], act["season_minutes"], "cond")
                add("games", g_all[played], act["games"], "cond")

            # totals + GP: unconditional, zero for missed/exit
            act_pts = act["season_points"] if appeared else 0.0
            act_min = act["season_minutes"] if appeared else 0.0
            act_gp = act["games"] if appeared else 0.0
            add("season_points", pts_all, act_pts, "uncond")
            add("season_minutes", mins_all, act_min, "uncond")
            add("games", g_all, act_gp, "uncond")

    return pd.DataFrame(score_rows), pd.DataFrame(appear_rows)


# ---------------------------------------------------------------------------
def _summary(scores: pd.DataFrame, appear: pd.DataFrame) -> dict:
    out = {"n_score_rows": int(len(scores)), "n_appear_rows": int(len(appear))}

    def agg(g):
        return pd.Series({
            "n": len(g), "crps": g["crps"].mean(), "mae": g["abs_err"].mean(),
            "median_bias": g["bias"].mean(),
            "cover_50": g["inside_50"].mean(), "cover_80": g["inside_80"].mean(),
            "cover_95": g["inside_95"].mean(),
            "pit_shape": cal.pit_shape(g["pit"].to_numpy())})
    # headline: by quantity x mode x horizon
    head = (scores.groupby(["quantity", "mode", "horizon"], observed=True)
            .apply(agg, include_groups=False).reset_index())
    out["by_quantity_mode_horizon"] = head.round(4).to_dict("records")

    # breakouts (conditional per-game + uncond totals pooled over horizon)
    breakdowns = {}
    scores = scores.copy()
    scores["age_bucket"] = pd.cut(scores["age"], [0, 24, 27, 30, 33, 99],
                                  labels=["<=24", "25-27", "28-30", "31-33", "34+"])
    scores["mpg_bucket"] = pd.cut(scores["last_mpg"], [0, 20, 28, 32, 34, 48],
                                  labels=["<20", "20-28", "28-32", "32-34", "34+"])
    scores["gp_bucket"] = pd.cut(scores["last_gp"], [0, 50, 65, 75, 82],
                                 labels=["<50", "50-65", "65-75", "75+"])
    scores["poss_bucket"] = pd.cut(scores["last_poss"], [0, 1500, 3000, 4500, 9999],
                                   labels=["<1500", "1500-3000", "3000-4500", "4500+"])
    scores["quality_bucket"] = pd.qcut(scores["quality"], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                                       duplicates="drop")
    for col in ["age_bucket", "position", "mpg_bucket", "gp_bucket", "poss_bucket",
                "quality_bucket", "injured_last", "top_decile_scorer",
                "gt30mpg", "gt34mpg", "lt50gp"]:
        sub = scores[scores["quantity"].isin(["pts_per_game", "minutes_per_game", "games"])
                     & (scores["mode"] == "cond")]
        b = (sub.groupby([col, "quantity"], observed=True)
             .apply(agg, include_groups=False).reset_index())
        breakdowns[col] = b.round(4).to_dict("records")
    out["breakouts_conditional"] = breakdowns

    # appearance / career-active Brier + calibration
    ap = {}
    for kind in ["appears", "career_active"]:
        d = appear[appear["kind"] == kind]
        by_h = (d.groupby("horizon")
                .apply(lambda g: pd.Series({
                    "n": len(g), "brier": ((g["p"] - g["actual"]) ** 2).mean(),
                    "pred": g["p"].mean(), "actual": g["actual"].mean()}),
                        include_groups=False).reset_index())
        ap[kind] = by_h.round(4).to_dict("records")
    out["appearance_brier_by_horizon"] = ap
    return out


def run(cutoffs, max_h=5, n_draws=800, fit_kwargs=None):
    panel = pd.read_parquet(PANEL_PATH)
    career_last = int(panel.loc[panel["possessions"] > 0, "season_year"].max())
    actuals = _actuals(panel)
    all_s, all_a = [], []
    t0 = time.time()
    for c in cutoffs:
        s, a = score_cutoff(c, panel, actuals, career_last, max_h=max_h,
                            n_draws=n_draws, fit_kwargs=fit_kwargs)
        all_s.append(s); all_a.append(a)
        _log(f"cutoff {c}: {len(s)} score rows ({time.time() - t0:.0f}s)")
    scores = pd.concat(all_s, ignore_index=True)
    appear = pd.concat(all_a, ignore_index=True)
    scores.to_parquet(OUTPUT_DIR / "backtest_ui_scores.parquet", index=False)
    appear.to_parquet(OUTPUT_DIR / "backtest_ui_appearance.parquet", index=False)
    summary = _summary(scores, appear)
    (OUTPUT_DIR / "backtest_ui_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"wrote scores ({len(scores)}), appearance ({len(appear)}), summary")
    return scores, appear, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", default="2016,2018")
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--max-h", type=int, default=5)
    args = ap.parse_args()
    cutoffs = [int(c) for c in args.cutoffs.split(",")]
    run(cutoffs, max_h=args.max_h, n_draws=args.draws)


if __name__ == "__main__":
    main()
