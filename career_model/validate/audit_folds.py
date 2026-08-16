"""Preflight audit of the cached backtest folds (`outputs/folds/fold_<cutoff>.pkl`).

A cached fold is a pickled `(FittedModel, Dataset)` from
`pipeline.fit_everything(max_season_year=cutoff)`.  Because this repo is not
under version control and the folds were written at different times across a
session in which `fit_everything` itself changed (the age x quality hazard
interaction and the within-career absence model were both added mid-session), a
cached fold can silently encode an *older* configuration than the frozen
Session-1 shipping model.  Running an old-vs-new availability comparison on top
of a mixed fold set would confound the very thing it is trying to measure.

This module records, per fold: cutoff, fit date, a model fingerprint (hash of
the packed state parameters + hazard + absence coefficients), the enabled
model-side feature flags it can be *read off the pickle* (hazard interaction,
absence model, posterior draws), and the training-data cutoff actually present
in the dataset -- then compares each against the frozen baseline config and
writes `outputs/baseline/fold_manifest.json`.

It does not change model behaviour.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, OUTPUT_DIR
from ..model import hierarchy as hier

FOLDS_DIR = OUTPUT_DIR / "folds"
BASELINE = OUTPUT_DIR / "baseline"


def _fit_fingerprint(model) -> str:
    """Stable hash of everything that defines this fold's projection behaviour:
    the packed state parameters plus the hazard and absence coefficients."""
    h = hashlib.sha256()
    try:
        h.update(np.asarray(hier.pack(model.fit.params), float).tobytes())
    except Exception:  # noqa: BLE001
        h.update(b"unpackable_fit")
    h.update(np.asarray(model.hazard.coef, float).tobytes())
    ab = getattr(model, "absence", None)
    if ab is not None:
        h.update(np.asarray(ab.coef, float).tobytes())
    return h.hexdigest()[:16]


def audit_one(path: Path, baseline: dict) -> dict:
    with open(path, "rb") as f:
        model, ds = pickle.load(f)
    cutoff_name = int(path.stem.split("_")[1])
    panel_max = int(ds.panel["season_year"].max())
    grid_max = int(ds.grid.season_years[ds.grid.observed].max())
    coef_len = int(len(model.hazard.coef))
    has_interaction = getattr(model.hazard, "inter_gamma", None) is not None
    has_absence = getattr(model, "absence", None) is not None
    n_post = 0 if model.posterior is None else int(len(model.posterior))

    want_coef = baseline["model"]["hazard_coef_len"]
    reasons = []
    if coef_len != want_coef or not has_interaction:
        reasons.append(f"hazard coef_len={coef_len} interaction={has_interaction} "
                       f"(frozen expects {want_coef}, interaction=True)")
    if not has_absence:
        reasons.append("no within-career absence model (frozen has one)")
    if int(model.train_cutoff) != cutoff_name:
        reasons.append(f"train_cutoff {model.train_cutoff} != filename {cutoff_name}")
    if panel_max > cutoff_name:
        reasons.append(f"LEAKAGE: panel season_year max {panel_max} > cutoff {cutoff_name}")
    if model.model_version != baseline["model"]["model_version"]:
        reasons.append(f"model_version {model.model_version} != "
                       f"{baseline['model']['model_version']}")

    return {
        "file": path.name,
        "cutoff": cutoff_name,
        "fit_date": time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(path.stat().st_mtime)),
        "model_fingerprint": _fit_fingerprint(model),
        "source_fingerprint": "not stored in fold; see baseline "
                              "model_fingerprint.txt for current source hashes",
        "model_version": model.model_version,
        "train_cutoff_declared": int(model.train_cutoff),
        "training_data_cutoff_panel": panel_max,
        "training_data_cutoff_grid": grid_max,
        "enabled_flags": {
            "hazard_age_quality_interaction": bool(has_interaction),
            "hazard_coef_len": coef_len,
            "within_career_absence": bool(has_absence),
            "posterior_draws": n_post,
            "_note": "display-layer devices (minutes_split, injury_beta, "
                     "avail_quality) are NOT stored in the fold; they are re-fit "
                     "from ds.panel at backtest time, so they always reflect "
                     "current code. posterior_draws=0 means the backtest omits "
                     "the parameter-uncertainty variance source (uniform across "
                     "folds, so not a confound).",
        },
        "compatible_with_frozen_baseline": len(reasons) == 0,
        "incompatibility_reasons": reasons,
    }


def audit(regenerated: dict | None = None) -> dict:
    baseline = json.loads((BASELINE / "config.json").read_text())
    folds = sorted(FOLDS_DIR.glob("fold_*.pkl"))
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_is_git": False,
        "frozen_baseline": {
            "model_pickle_sha256_prefix":
                hashlib.sha256((ARTIFACT_DIR / "career_model.pkl").read_bytes())
                .hexdigest()[:16],
            "hazard_coef_len": baseline["model"]["hazard_coef_len"],
            "within_career_absence": True,
            "model_version": baseline["model"]["model_version"],
            "source_combined_fingerprint": _read_combined_fp(),
        },
        "folds": [audit_one(p, baseline) for p in folds],
    }
    if regenerated:
        manifest["regenerated_this_preflight"] = regenerated
    compat = [f["cutoff"] for f in manifest["folds"] if f["compatible_with_frozen_baseline"]]
    incompat = [f["cutoff"] for f in manifest["folds"] if not f["compatible_with_frozen_baseline"]]
    manifest["summary"] = {
        "compatible_cutoffs": compat,
        "incompatible_cutoffs": incompat,
        "safe_for_old_vs_new_availability_comparison":
            {"2018", "2023"}.issubset({str(c) for c in compat}),
    }
    BASELINE.mkdir(parents=True, exist_ok=True)
    (BASELINE / "fold_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _read_combined_fp() -> str:
    fp = BASELINE / "model_fingerprint.txt"
    if not fp.exists():
        return "n/a"
    for line in fp.read_text().splitlines():
        if line.startswith("combined_fingerprint"):
            return line.split()[-1]
    return "n/a"


def main() -> None:
    m = audit()
    print(json.dumps(m["summary"], indent=2))
    for f in m["folds"]:
        flag = "OK " if f["compatible_with_frozen_baseline"] else "STALE"
        print(f"  [{flag}] {f['file']}  fit {f['fit_date']}  "
              f"coef={f['enabled_flags']['hazard_coef_len']} "
              f"absence={f['enabled_flags']['within_career_absence']} "
              f"fp={f['model_fingerprint']}"
              + ("" if f["compatible_with_frozen_baseline"]
                 else "  <- " + "; ".join(f["incompatibility_reasons"])))


if __name__ == "__main__":
    main()
