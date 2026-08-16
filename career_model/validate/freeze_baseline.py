"""Freeze a reproducible snapshot of the current shipping configuration.

This does not touch projection behaviour.  It records, into
`career_model/outputs/baseline/`:

  config.json            every enabled feature flag + model/projection config
  metrics.json           existing backtest metrics (flagged stale where they
                         predate the shipping model) + this session's A/B logs
  example_players.json   current stored projections for SGA, Zubac, Tatum
  test_results.txt       pytest output
  model_fingerprint.txt  content hashes (this is not a git repo, so a hash of
                         the model pickle + the source files that define
                         behaviour stands in for a commit)

Run:  .venv/bin/python -m career_model.validate.freeze_baseline
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from ..config import (
    AGE_BOUNDARY, AGE_KNOTS, MODEL_VERSION, PROJECTION_DIR, S, STATE_NAMES,
    TRAIN_CUTOFF_YEAR,
)
from .. import config as cfg
from ..simulate import project, derive
from ..model import hazard as hazard_mod
from ..pipeline import load as load_model

BASELINE = cfg.OUTPUT_DIR / "baseline"
EXAMPLE_IDS = {"Shai Gilgeous-Alexander": 1628983,
               "Ivica Zubac": 1627826, "Jayson Tatum": 1628369}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _flags() -> dict:
    """The feature flags that are actually in force in the shipping pipeline.

    Values are read from the code defaults and from what `precompute.run`
    passes, so this is the *effective* configuration, not merely the defaults.
    """
    ms = derive.MinutesSplit  # dataclass defaults
    return {
        # --- projection sub-models, as wired in simulate.precompute.run ---
        "use_eb_reversion_target": True,          # simulate() default, not overridden
        "use_avail_eb": True,                      # availability EB, default on
        "use_avail_slope": False,                  # rejected experiment, default off
        "injury_regime": True,                     # injury_beta fitted + passed
        "injury_downside_only": True,
        "within_career_absence": True,             # model.absence fitted + passed
        "age_quality_availability_aging": True,    # avail_quality fitted + passed
        "hazard_age_quality_interaction":
            getattr(load_model().hazard, "inter_gamma", None) is not None,
        "gap_preroll_for_inactive": True,
        # --- constants that shape the projection ---
        "MAX_AGE": project.MAX_AGE,
        "AGE_RAMP_START": project.AGE_RAMP_START,
        "MAX_POSSESSIONS": project.MAX_POSSESSIONS,
        "SOFT_CAP_SHOULDER": project.SOFT_CAP_SHOULDER,
        "HINGE_AGE": project.HINGE_AGE,
        "EB_HALF_LIFE": project.EB_HALF_LIFE,
        "EB_AVAIL_K": project.EB_AVAIL_K,
        "INJ_PROPENSITY_STRENGTH": project.INJ_PROPENSITY_STRENGTH,
        "INJURY_RATE_DEFAULT": project.INJURY_RATE_DEFAULT,
        "MIN_PER_POSSESSION": project.MIN_PER_POSSESSION,
        # --- minutes/games split (display layer) ---
        "minutes_split_offset_half_life":
            derive.MinutesSplit.__dataclass_fields__["offset_half_life"].default,
        "minutes_split_offset_shrink_scale":
            derive.MinutesSplit.__dataclass_fields__["offset_shrink_scale"].default,
        "minutes_split_max_mpg":
            derive.MinutesSplit.__dataclass_fields__["max_mpg"].default,
        "minutes_split_between_share":
            derive.MinutesSplit.__dataclass_fields__["between_share"].default,
    }


def _model_config(model) -> dict:
    p = model.fit.params
    return {
        "model_version": model.model_version,
        "train_cutoff": int(model.train_cutoff),
        "state_dim": int(S),
        "state_names": list(STATE_NAMES),
        "age_knots": list(AGE_KNOTS),
        "age_boundary": list(AGE_BOUNDARY),
        "posterior_draws": 0 if model.posterior is None else int(len(model.posterior)),
        "hazard_coef_len": int(len(model.hazard.coef)),
        "hazard_names": list(model.hazard.names),
        "A_diag_avail": float(p.A[cfg.AVAIL_IDX]),
        "sigma_poss": float(p.sigma_poss),
        "sigma_poss_inj": float(p.sigma_poss_inj),
        "default_n_draws": 2000,
        "default_horizon": 12,
        "default_n_param_sets": 64,
    }


def freeze() -> None:
    BASELINE.mkdir(parents=True, exist_ok=True)
    model = load_model()

    # ---- config.json ------------------------------------------------------
    config = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Effective shipping configuration; values reflect code defaults "
                "AND what simulate.precompute.run actually passes.",
        "flags": _flags(),
        "model": _model_config(model),
    }
    (BASELINE / "config.json").write_text(json.dumps(config, indent=2))

    # ---- model_fingerprint.txt -------------------------------------------
    # No git here, so hash the pickle plus the source files that define
    # filtering / simulation / derived outputs / validation behaviour.
    behaviour_files = [
        "config.py", "pipeline.py",
        "model/state_space.py", "model/hazard.py", "model/observations.py",
        "model/hierarchy.py", "model/aging.py",
        "simulate/project.py", "simulate/derive.py", "simulate/precompute.py",
        "validate/backtest.py", "validate/calibration.py",
    ]
    fp_lines = [f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "repo is NOT a git repository; fingerprint is content-hash based",
                "", "artifacts:"]
    for name in ("career_model.pkl",):
        fp_lines.append(f"  {name}  {_sha(cfg.ARTIFACT_DIR / name)}")
    idx = PROJECTION_DIR / "index.json"
    if idx.exists():
        fp_lines.append(f"  projections/index.json  {_sha(idx)}")
        fp_lines.append(f"  store_generated  {json.loads(idx.read_text())['generated']}")
    fp_lines += ["", "source files:"]
    root = Path(cfg.ROOT)
    for rel in behaviour_files:
        f = root / rel
        fp_lines.append(f"  {rel}  {_sha(f) if f.exists() else 'MISSING'}")
    combined = hashlib.sha256(
        "".join(l for l in fp_lines if "  " in l).encode()).hexdigest()
    fp_lines += ["", f"combined_fingerprint  {combined}"]
    (BASELINE / "model_fingerprint.txt").write_text("\n".join(fp_lines) + "\n")

    # ---- test_results.txt -------------------------------------------------
    proc = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "career_model/tests", "-q"],
        cwd=str(root.parent), capture_output=True, text=True)
    (BASELINE / "test_results.txt").write_text(
        f"$ pytest career_model/tests -q\n(exit {proc.returncode})\n\n"
        + proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))

    # ---- example_players.json --------------------------------------------
    examples = {}
    for name, pid in EXAMPLE_IDS.items():
        payload = PROJECTION_DIR / str(pid) / "payload.json"
        if not payload.exists():
            examples[name] = {"player_id": pid, "error": "no stored projection"}
            continue
        p = json.loads(payload.read_text())
        s = p["seasons"]
        surv = p["survival"]
        # The exact rows the UI shows, plus survival, plus recent history.
        pg = ["minutes_per_game", "pts_per_game", "reb_per_game", "ast_per_game",
              "stl_per_game", "blk_per_game", "games"]

        def series(stat):
            d = s.get(stat)
            if not d:
                return None
            return {"age": d["age"], "season_year": d["season_year"],
                    "p10": d["p10"], "p50": d["p50"], "p90": d["p90"]}
        examples[name] = {
            "player_id": pid,
            "meta": {k: p["meta"].get(k) for k in
                     ("name", "position", "age", "team", "status",
                      "last_season", "n_history_seasons")},
            "history_last3": (p["meta"].get("history") or [])[-3:],
            "per_game": {stat: series(stat) for stat in pg if s.get(stat)},
            "survival": {"season_year": surv["season_year"],
                         "p_active": surv["p_active"],
                         "p_play": surv.get("p_play")},
        }
    (BASELINE / "example_players.json").write_text(json.dumps(examples, indent=2))

    # ---- metrics.json -----------------------------------------------------
    metrics = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warning": "Stored backtest parquets predate this session's shipping "
                   "fixes (absence, hazard interaction, quality-aging, offset). "
                   "They describe an OLDER model. The authoritative current-config "
                   "metrics come from this session's A/B logs and from part C "
                   "(validate.backtest_ui). Treated as historical here.",
        "stored_parquets": {},
        "session_ab_logs": {},
    }
    for pq in sorted(cfg.OUTPUT_DIR.glob("backtest_scores*.parquet")):
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            by = (df.groupby(["stat", "horizon"])
                  .agg(n=("crps", "size"), crps=("crps", "mean"),
                       cover_80=("inside_80", "mean")).reset_index())
            metrics["stored_parquets"][pq.name] = {
                "mtime": time.strftime("%Y-%m-%d", time.localtime(pq.stat().st_mtime)),
                "n_rows": int(len(df)),
                "summary_by_stat_horizon": by.round(4).to_dict("records"),
            }
        except Exception as e:  # noqa: BLE001
            metrics["stored_parquets"][pq.name] = {"error": str(e)}
    for log in sorted(cfg.OUTPUT_DIR.glob("ab_*2018.log")) + \
            sorted(cfg.OUTPUT_DIR.glob("ab_*2016.log")):
        txt = log.read_text()
        # keep the summary block (from the CLEAN A/B header onward)
        i = txt.find("=== CLEAN A/B")
        metrics["session_ab_logs"][log.name] = txt[i:] if i >= 0 else txt[-1500:]
    (BASELINE / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[freeze] wrote baseline to {BASELINE}")
    for f in sorted(BASELINE.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


def main() -> None:
    freeze()


if __name__ == "__main__":
    main()
