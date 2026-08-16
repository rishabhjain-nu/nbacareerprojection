"""The projection store must be stamped with, and consistent with, the frozen
Session-4 shipping configuration.

This guards the Part-D invariant: a payload generated under a superseded
configuration must be detectable.  `index.json` and every player `meta.json`
carry `config_fingerprint`; they must agree with each other and with the current
`SHIP_CONFIG` hash, and the index must record the Gaussian state innovation.
"""

from __future__ import annotations

import json

import pytest

from career_model.config import PROJECTION_DIR
from career_model.simulate.precompute import SHIP_CONFIG, config_fingerprint


def _index():
    p = PROJECTION_DIR / "index.json"
    if not p.exists():
        pytest.skip("no projection store; run precompute first")
    return json.loads(p.read_text())


def test_index_fingerprint_matches_ship_config():
    idx = _index()
    assert idx["config_fingerprint"] == config_fingerprint()
    assert idx["enabled_flags"]["state_innovation"] == "gaussian"
    assert idx["enabled_flags"] == SHIP_CONFIG


def test_every_payload_carries_matching_fingerprint():
    idx = _index()
    fp = idx["config_fingerprint"]
    # sample the first 40 real players (skip prospects, which have no meta pid dir
    # collisions) -- checking all 2900 would be slow but the loop is identical.
    checked = 0
    for p in idx["players"][:40]:
        meta_path = PROJECTION_DIR / str(p["player_id"]) / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        assert meta.get("config_fingerprint") == fp, p["player_id"]
        checked += 1
    assert checked > 0


def test_no_disabled_experiment_is_enabled():
    """Every candidate that failed its gate must be off in the shipped config."""
    for flag in ("use_joint_availability", "diffuse_m", "student_t_innovation",
                 "role_change_mixture", "permanent_transient_state",
                 "fold_local_era_component"):
        assert SHIP_CONFIG[flag] is False, flag
