"""Batch-run every player and write the static projection store (§8.5).

Monte Carlo at request time is not on the table -- 2000 draws x 12 seasons x 14
dimensions per player is seconds, not milliseconds.  So the whole corpus is
simulated once and stored as **percentile summaries**, which is small enough to
serve as flat files from any static host with no database and no API layer.

The cost of storing percentiles rather than draws is stated plainly in §8.5 and
worth repeating: a query like "P(25 ppg AND 8 apg)" needs the joint distribution
across stats, and percentiles cannot answer it.  The storage layer is laid out
so retaining draws for a specific query path is additive -- `--keep-draws`
writes `draws.npz` alongside the summaries for the players that need it,
and nothing else changes.

Every file is stamped with `model_version` and the training cutoff.  When the
model is refit the version changes, and `index.json` carries the same stamp, so
a client holding a stale cached payload can tell.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    DISPLAY_LABEL, MODEL_VERSION, PROJECTION_DIR, PRIOR_COVARIATES, STATE_NAMES,
    RAW_DIR,
)
from ..model.dataset import load as load_dataset
from ..pipeline import filtered_states, load as load_model
from . import derive, project

# What the client can chart.  The 14 state dimensions plus the derived
# quantities §8.3 asks for -- no more, because every extra stat is payload.
CHART_STATS = [
    "pts_per100", "reb_per100", "oreb_per100", "dreb_per100", "ast_per100",
    "stl_per100", "blk_per100", "tov_per100", "pf_per100",
    "fga_2p_per100", "fga_3p_per100", "fta_per100",
    "ts_pct", "usg_100", "possessions", "minutes", "bpm", "obpm", "dbpm",
    # Per-game figures, derived via `MinutesSplit`.  The model projects
    # possessions; the games/minutes split is a fitted display-layer device.
    "minutes_per_game", "pts_per_game", "reb_per_game", "ast_per_game",
    "stl_per_game", "blk_per_game", "tov_per_game", "fg3a_per_game", "games",
] + [f"state:{s}" for s in STATE_NAMES]

# The stats the season-by-season per-game table shows, in order.
PER_GAME_TABLE = ["minutes_per_game", "pts_per_game", "reb_per_game",
                  "ast_per_game", "stl_per_game", "blk_per_game", "games"]

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

# ---------------------------------------------------------------------------
# Frozen Session-4 shipping configuration (§ Final integration).
#
# Every candidate tested across Sessions 1-4 is enumerated here with its gate
# verdict, and only the ones that PASSED are enabled.  This dict is the single
# source of truth for the flags, and its hash is stamped into every artifact
# (index.json + each player meta.json) as `config_fingerprint`, so a client can
# tell exactly which configuration produced a payload.
#
# Integration decision (Session 4): the S3-B role-change innovation *mixture*
# improves a few shot-volume / assist-role subgroups but FAILS its aggregate
# acceptance gate -- rolling-origin normalized CRPS 1.000 -> 1.017 and 80%
# coverage 0.801 (calibrated) -> 0.839 (over-covered), see
# outputs/state_dev_ab_summary.json.  Per "enable only candidates that passed
# their gate", the shipping state innovation is GAUSSIAN.
# ---------------------------------------------------------------------------
SHIP_CONFIG = {
    # enabled (passed their gates / are the baseline behaviour)
    "state_innovation": "gaussian",        # S3-B mixture/student-t/diffuse-m all failed
    "use_eb_reversion_target": True,
    "within_career_absence": True,         # deviation #4: P(plays | active)
    "injury_regime": True,
    "age_quality_availability_aging": True,
    "hazard_age_quality_interaction": True,
    # disabled (failed their acceptance gates -- kept behind flags, not shipped)
    "use_joint_availability": False,       # S2: durable-star GP over-projection worse
    "diffuse_m": False,                    # S3-A: aggregate CRPS +15%
    "student_t_innovation": False,         # S3-B: dominated by (and worse than) gaussian
    "role_change_mixture": False,          # S3-B: aggregate CRPS +1.7%, over-covers
    "permanent_transient_state": False,    # pre-S3: failed, do not retry
    "fold_local_era_component": False,     # S4-A: candidate/diagnostic, not yet fold-validated
}


def config_fingerprint() -> str:
    """Stable short hash of SHIP_CONFIG -- stamped into every artifact."""
    import hashlib
    blob = json.dumps(SHIP_CONFIG, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _log(msg: str) -> None:
    print(f"[precompute] {msg}", flush=True)


def _round(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


# ---------------------------------------------------------------------------
# Incoming draft class
# ---------------------------------------------------------------------------
def load_prospects(model, draft_year: int | None = None) -> pd.DataFrame:
    """The incoming class: players with college/combine/draft data and no NBA
    seasons at all (§8.1).  Their entire projection comes from `x_i`."""
    path = RAW_DIR / "prospects.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if draft_year is not None:
        df = df[df["draft_year"] == draft_year]
    df = df.copy()
    df["has_college_data"] = df["college_bpm_sos"].notna()
    df["has_combine_data"] = df["wingspan_in"].notna()
    df["undrafted"] = df.get("undrafted", 0)
    df["age_at_draft_days"] = pd.to_numeric(df["age_at_draft_days"], errors="coerce")
    for c in PRIOR_COVARIATES:
        if c not in df.columns:
            df[c] = np.nan
    return df


def prospect_design(prospects: pd.DataFrame, scaler: dict) -> np.ndarray:
    """Apply the *training* standardisation.  Re-standardising on the draft
    class alone would silently rescale every covariate against a 60-player
    reference distribution."""
    cols, mean, sd = scaler["cols"], scaler["mean"], scaler["sd"]
    raw = prospects[cols].to_numpy(dtype=float)
    z = (raw - mean) / sd
    z[~np.isfinite(z)] = 0.0
    pick = np.clip(prospects["draft_pick"].fillna(61).to_numpy(dtype=float), 1, None)
    log_pick = (np.log(pick) - scaler["log_pick_mean"]) / scaler["log_pick_sd"]
    ind_col = prospects["has_college_data"].to_numpy(dtype=float)
    ind_comb = prospects["has_combine_data"].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(prospects)), z, log_pick, ind_col, ind_comb])


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------
def payload_for(proj: project.Projection, translator, meta: dict) -> dict:
    seasons = project.summarise(proj, translator, PERCENTILES)
    seasons = seasons[seasons["stat"].isin(CHART_STATS)]
    totals = project.career_totals(proj, translator, PERCENTILES)
    peak = project.peak_distribution(
        proj, stats=("pts_per100", "reb_per100", "ast_per100", "bpm"),
        translator=translator, percentiles=(5, 25, 50, 75, 95))
    surv = project.survival_curve(proj)

    by_stat: dict[str, dict] = {}
    for stat, g in seasons.groupby("stat", observed=True):
        g = g.sort_values("horizon")
        by_stat[stat] = {
            "horizon": g["horizon"].astype(int).tolist(),
            "age": [_round(a, 1) for a in g["age"]],
            "season_year": g["season_year"].astype(int).tolist(),
            **{f"p{q}": [_round(v, 3) for v in g[f"p{q}"]] for q in PERCENTILES},
        }

    return {
        "meta": meta,
        "per_game_table": [s for s in PER_GAME_TABLE if s in by_stat],
        "seasons": by_stat,
        "survival": {"horizon": surv["horizon"].astype(int).tolist(),
                     "age": [_round(a, 1) for a in surv["age"]],
                     "season_year": surv["season_year"].astype(int).tolist(),
                     # p_active = career still active; p_play = actually appears
                     # that season (deviation #4).  p_play is absent from a store
                     # built before the fix; the interface falls back to p_active.
                     "p_active": [_round(p, 4) for p in surv["p_active"]],
                     "p_play": [_round(p, 4) for p in surv.get("p_play", surv["p_active"])]},
        "totals": [{k: (_round(v, 2) if isinstance(v, (int, float, np.number)) else v)
                    for k, v in r.items()} for r in totals.to_dict("records")],
        "peak": [{k: (_round(v, 3) if isinstance(v, (int, float, np.number)) else v)
                  for k, v in r.items()} for r in peak.to_dict("records")],
    }


def write_player(out_dir: Path, pid: int, proj, translator, meta: dict,
                 write_parquet: bool, keep_draws: bool,
                 history: list | None = None) -> dict:
    """Write one player's static payload.

    Observed history rides in the per-player file rather than in `index.json`:
    the index is loaded once on page init for the whole corpus, and carrying
    three thousand players' season histories in it would make the first paint
    pay for data the reader will never look at.
    """
    d = out_dir / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    payload = payload_for(proj, translator, dict(meta, history=history or []))
    (d / "payload.json").write_text(json.dumps(payload, separators=(",", ":")))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))

    if write_parquet:
        # The layout §8.5 specifies, for analysis; the client reads payload.json.
        seasons = project.summarise(proj, translator, PERCENTILES)
        seasons[seasons["stat"].isin(CHART_STATS)].to_parquet(d / "seasons.parquet",
                                                              index=False)
        project.career_totals(proj, translator, PERCENTILES).to_parquet(
            d / "totals.parquet", index=False)
        project.peak_distribution(proj, translator=translator).to_parquet(
            d / "peak.parquet", index=False)
        project.survival_curve(proj).to_parquet(d / "survival.parquet", index=False)
    if keep_draws:
        np.savez_compressed(d / "draws.npz", alive=proj.alive, theta=proj.theta,
                            possessions=proj.possessions,
                            **{f"box_{k}": v for k, v in proj.box.items()})
    return payload


def summary_row(payload: dict, stat: str = "pts_per100") -> dict:
    """The one-glance distributional summary each search result row carries
    (§8.2).  A median with a band -- never a bare number."""
    s = payload["seasons"].get(stat)
    if not s or not s["p50"]:
        return {}
    return {"stat": stat, "p10": s["p10"][0], "p50": s["p50"][0], "p90": s["p90"][0]}


# ---------------------------------------------------------------------------
def run(n_draws: int = 2000, horizon: int = 12, out_dir: Path = PROJECTION_DIR,
        limit: int | None = None, write_parquet: bool = False,
        keep_draws: bool = False, draft_year: int | None = 2026) -> None:
    t0 = time.time()
    model = load_model()
    ds = load_dataset()
    filt = filtered_states(ds, model.fit)
    translator = derive.fit_bpm_translator(ds.panel)
    minutes_split = derive.fit_minutes_split(ds.panel)
    injury_beta = project.fit_injury_rate(ds.panel)
    # Session-2 joint availability candidate (severity -> GP/MPG).  DISABLED:
    # it failed its acceptance rule (durable/high-quality GP over-projection got
    # worse, per-game rate coverage regressed) -- see validate/availability_ab
    # and the Session-2 report.  Flip this to True to enable the fitted system;
    # `avail_system=None` keeps the shipping path.
    from ..simulate import availability as _availability
    USE_JOINT_AVAILABILITY = False
    avail_system = (_availability.fit_availability(ds, filt, model.hazard)
                    if USE_JOINT_AVAILABILITY else None)
    # Age x quality availability aging: slows an old star's minute decline
    # (the star minute-drop fix).  Fitted from the same filtered states.
    avail_quality = project.fit_avail_quality_aging(ds, filt, model.hazard)
    # S3-B role-change innovation mixture (DISABLED as of Session 4).  It widens
    # a few shot-volume / assist-role subgroups nicely, but on the rolling-origin
    # backtest it FAILED its aggregate gate: normalized CRPS 1.000 -> 1.017 and
    # 80% coverage went from a calibrated 0.801 to an over-covered 0.839
    # (outputs/state_dev_ab_summary.json).  The shipping state innovation is
    # Gaussian (SHIP_CONFIG["state_innovation"]).  The mixture, Student-t and
    # diffuse-m variants remain reachable via simulate()'s `innovation=` /
    # `filtered_states(m_prior_scale=)` for A/B work, but none ships.
    role_model = project.fit_role_change(ds, filt)   # kept for A/B; unused when gaussian
    cfg_fp = config_fingerprint()

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)     # §8.5: stale projections are invalidated, not mixed
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = ds.grid
    last_year = int(grid.season_years[grid.observed].max())
    corpus = []

    rows = range(grid.n_players) if limit is None else range(min(limit, grid.n_players))
    for n, i in enumerate(rows):
        if grid.last_index[i] < 0:
            continue
        pid = int(grid.player_ids[i])
        last = int(grid.last_index[i])
        final_year = int(grid.season_years[i, last])
        info = ds.priors.iloc[i]
        team = ds.panel.loc[(ds.panel["player_id"] == pid)
                            & (ds.panel["season_year"] == final_year), "team_id"]
        # One missed season is not retirement -- ~400 players in this panel
        # miss a year and come back, and an Achilles rehab is not an exit.
        # `inactive` marks exactly one missed season; his projection rolls the
        # filtered state through the gap and resumes at the *current* season.
        # Two or more missed seasons is called retired, and the projection
        # stays a counterfactual from the point of exit.
        gap = last_year - final_year
        status = "active" if gap == 0 else ("inactive" if gap == 1 else "retired")
        meta = {
            "player_id": pid,
            "name": str(info["player_name"]),
            "position": str(info.get("position_group", "F")),
            "age": _round(float(grid.age[i, last]) + (last_year - final_year), 1),
            "team": str(team.iloc[0]) if len(team) else "",
            "status": status,
            "last_season": final_year,
            "n_history_seasons": int(grid.n_history[i]),
            "is_rookie": False,
            "has_college_data": bool(info.get("has_college_data", False)),
            "has_combine_data": bool(info.get("has_combine_data", False)),
            "model_version": MODEL_VERSION,
            "train_cutoff": model.train_cutoff,
            "config_fingerprint": cfg_fp,
        }
        proj = project.simulate(model, ds, i, filt, n_draws=n_draws, horizon=horizon,
                                minutes_split=minutes_split,
                                extra_gap=gap if status == "inactive" else 0,
                                injury_beta=injury_beta, avail_quality=avail_quality,
                                avail_system=avail_system,
                                innovation=SHIP_CONFIG["state_innovation"],
                                role_model=role_model, role_scale=4.0)
        payload = write_player(out_dir, pid, proj, translator, meta,
                               write_parquet, keep_draws,
                               history=observed_history(ds, i))
        meta["summary"] = summary_row(payload)
        corpus.append(meta)
        if (n + 1) % 250 == 0:
            _log(f"{n + 1} players  ({time.time() - t0:.0f}s)")

    # ---- incoming draft class (§8.1) --------------------------------------
    prospects = load_prospects(model, draft_year)
    if len(prospects) and model.gbm is not None:
        Xp = prospect_design(prospects, model.scaler)
        gbm_off = model.gbm.predict(prospects)
        pos_map = {"G": 0, "F": 1, "C": 2}
        for k, row in enumerate(prospects.itertuples()):
            # Prospect ids are negative in the source table; map them into a
            # high positive block so they cannot collide with an NBA player id
            # and can still be used as a directory name and an RNG seed.
            raw = int(getattr(row, "prospect_id", 0) or 0) or -(k + 1)
            pid = 9_000_000 + abs(raw) % 1_000_000
            age0 = float(getattr(row, "rookie_age", 20.0) or 20.0)
            meta = {
                "player_id": pid,
                "name": str(row.player_name),
                "position": _guess_position(row),
                "age": _round(age0, 1),
                "team": f"{int(row.draft_year)} draft class",
                "status": "prospect",
                "last_season": None,
                "n_history_seasons": 0,
                "is_rookie": True,
                "has_college_data": bool(row.has_college_data),
                "has_combine_data": bool(row.has_combine_data),
                "draft_pick": None if pd.isna(row.draft_pick) else int(row.draft_pick),
                "model_version": MODEL_VERSION,
                "train_cutoff": model.train_cutoff,
            "config_fingerprint": cfg_fp,
            }
            proj = project.simulate_rookie(
                model, ds, Xp[k], gbm_off[k], age0=age0,
                year0=int(row.rookie_season_year), pos_idx=pos_map[meta["position"]],
                player_id=pid, n_draws=n_draws, horizon=horizon,
                minutes_split=minutes_split, injury_beta=injury_beta)
            payload = write_player(out_dir, pid, proj, translator, meta,
                                   write_parquet, keep_draws, history=[])
            meta["summary"] = summary_row(payload)
            corpus.append(meta)
        _log(f"{len(prospects)} incoming prospects")

    index = {
        "model_version": MODEL_VERSION,
        "train_cutoff": model.train_cutoff,
        "config_fingerprint": cfg_fp,
        "enabled_flags": SHIP_CONFIG,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "last_season": last_year,
        "stats": [{"key": s, "label": _label(s)} for s in CHART_STATS],
        "players": corpus,
    }
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    _log(f"wrote {len(corpus)} players to {out_dir} in {time.time() - t0:.0f}s")


def _guess_position(row) -> str:
    h = getattr(row, "height_in", np.nan)
    if not np.isfinite(h):
        return "F"
    return "G" if h < 78 else ("C" if h >= 82 else "F")


def observed_history(ds, i: int) -> list:
    """Actual seasons, so the chart can render history to the left of the
    projection boundary in a visually distinct treatment (§8.3)."""
    grid = ds.grid
    out = []
    for t in np.flatnonzero(grid.observed[i]):
        poss = float(grid.exposure[i, t])
        games = float(grid.games[i, t])
        box = {c: float(grid.counts[c][i, t]) for c in derive.RATE_COLS}
        # Actual games played, so the observed half of the per-game table is the
        # real box score rather than a re-derivation through `MinutesSplit`.
        comp = derive.derive_composites({k: np.array([v]) for k, v in box.items()},
                                        np.array([poss]),
                                        games=np.array([games]) if games > 0 else None)
        rec = {"season_year": int(grid.season_years[i, t]),
               "age": _round(float(grid.age[i, t]), 1),
               "possessions": _round(poss, 0)}
        for stat in CHART_STATS:
            if stat in comp:
                rec[stat] = _round(float(comp[stat][0]), 3)
        rec["minutes"] = _round(poss * project.MIN_PER_POSSESSION, 0)
        for s, name in enumerate(STATE_NAMES):
            rec[f"state:{name}"] = _round(float(grid.z[i, t, s]), 3)
        out.append(rec)
    return out


_EXTRA_LABELS = {
    "pts_per100": "Points / 100", "reb_per100": "Rebounds / 100",
    "oreb_per100": "Off. rebounds / 100", "dreb_per100": "Def. rebounds / 100",
    "ast_per100": "Assists / 100", "stl_per100": "Steals / 100",
    "blk_per100": "Blocks / 100", "tov_per100": "Turnovers / 100",
    "pf_per100": "Fouls / 100", "fga_2p_per100": "2PA / 100",
    "fga_3p_per100": "3PA / 100", "fta_per100": "FTA / 100",
    "ts_pct": "True shooting %", "usg_100": "Usage / 100 poss",
    "possessions": "Possessions", "minutes": "Minutes",
    "bpm": "BPM (derived)", "obpm": "OBPM (derived)", "dbpm": "DBPM (derived)",
    "minutes_per_game": "Minutes per game", "pts_per_game": "Points per game",
    "reb_per_game": "Rebounds per game", "ast_per_game": "Assists per game",
    "stl_per_game": "Steals per game", "blk_per_game": "Blocks per game",
    "tov_per_game": "Turnovers per game", "fg3a_per_game": "3PA per game",
    "games": "Games played",
}


def _label(key: str) -> str:
    if key.startswith("state:"):
        return f"{DISPLAY_LABEL.get(key[6:], key[6:])} (state)"
    return _EXTRA_LABELS.get(key, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parquet", action="store_true",
                    help="also write the per-player parquet layout of §8.5")
    ap.add_argument("--keep-draws", action="store_true",
                    help="retain raw draws for joint-distribution queries")
    args = ap.parse_args()
    run(n_draws=args.draws, horizon=args.horizon, limit=args.limit,
        write_parquet=args.parquet, keep_draws=args.keep_draws)


if __name__ == "__main__":
    main()
