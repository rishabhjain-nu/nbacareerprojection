"""Anthropometrics and draft position -- the time-invariant half of `x_i`.

Three sources, cached under `data_raw/`:

  * `draftcombineplayeranthro` (2000+) -- wingspan and standing reach, the two
    measurements that are not on any roster page and that matter most for the
    defensive dimensions of the state.
  * `playerindex`                      -- listed height/weight and position, for
                                          everyone including non-combine players.
  * `drafthistory`                     -- pick number.

Combine data is keyed by NBA player id when the endpoint supplies one and by
name otherwise; both paths are handled here and reconciled in `reconcile_ids`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RAW_DIR
from .ingest_college import normalize_name

ANTHRO_DIR = RAW_DIR / "anthro"
INDEX_DIR = RAW_DIR / "playerindex"
DRAFT_PATH = RAW_DIR / "draft_history.csv"
UNDRAFTED_PICK = 61.0     # one slot past the last pick of a 60-pick draft


def _log(msg: str) -> None:
    print(f"[anthro] {msg}", flush=True)


def parse_height(h) -> float:
    """'6-9' -> 81.0 inches."""
    if isinstance(h, (int, float, np.floating)) and not pd.isna(h):
        return float(h)
    if not isinstance(h, str) or "-" not in h:
        return np.nan
    try:
        ft, inch = h.split("-")
        return float(ft) * 12 + float(inch)
    except ValueError:
        return np.nan


def load_combine() -> pd.DataFrame:
    frames = []
    for path in sorted(ANTHRO_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if "PLAYER_NAME" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        _log(f"no combine files under {ANTHRO_DIR}")
        return pd.DataFrame(columns=["name_key"])
    df = pd.concat(frames, ignore_index=True)
    df["name_key"] = df["PLAYER_NAME"].map(normalize_name)

    out = pd.DataFrame({
        "name_key": df["name_key"],
        "combine_height_in": pd.to_numeric(df.get("HEIGHT_WO_SHOES"), errors="coerce"),
        "weight_lb": pd.to_numeric(df.get("WEIGHT"), errors="coerce"),
        "wingspan_in": pd.to_numeric(df.get("WINGSPAN"), errors="coerce"),
        "standing_reach_in": pd.to_numeric(df.get("STANDING_REACH"), errors="coerce"),
    })
    out = out[out["name_key"] != ""]
    # A player can appear at more than one combine; take the first non-null of each.
    out = out.groupby("name_key").first().reset_index()
    _log(f"{len(out)} combine participants")
    return out


def load_player_index() -> pd.DataFrame:
    """Listed height/weight, position, draft slot, and true debut year."""
    frames = []
    for path in sorted(INDEX_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if "PERSON_ID" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        _log(f"no playerindex files under {INDEX_DIR}")
        return pd.DataFrame(columns=["player_id"])
    df = pd.concat(frames, ignore_index=True)

    df["player_id"] = pd.to_numeric(df["PERSON_ID"], errors="coerce")
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype("int64")
    df["player_name"] = (df["PLAYER_FIRST_NAME"].fillna("").astype(str) + " "
                         + df["PLAYER_LAST_NAME"].fillna("").astype(str)).str.strip()
    df["name_key"] = df["player_name"].map(normalize_name)
    df["index_height_in"] = df["HEIGHT"].map(parse_height)
    df["index_weight_lb"] = pd.to_numeric(df["WEIGHT"], errors="coerce")
    df["position_raw"] = df["POSITION"].astype(str)
    df["from_year"] = pd.to_numeric(df["FROM_YEAR"], errors="coerce")
    df["country"] = df["COUNTRY"].astype(str)
    df["college_name"] = df["COLLEGE"].astype(str)

    cols = ["player_id", "player_name", "name_key", "index_height_in", "index_weight_lb",
            "position_raw", "from_year", "country", "college_name"]
    # Later files carry corrections; keep the most recent non-null per player.
    out = df[cols].groupby("player_id").last().reset_index()
    _log(f"{len(out)} players in the index")
    return out


def load_draft() -> pd.DataFrame:
    if not DRAFT_PATH.exists():
        _log(f"no draft history at {DRAFT_PATH}")
        return pd.DataFrame(columns=["player_id"])
    df = pd.read_csv(DRAFT_PATH)
    df["player_id"] = pd.to_numeric(df["PERSON_ID"], errors="coerce")
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype("int64")
    df["draft_pick"] = pd.to_numeric(df["OVERALL_PICK"], errors="coerce")
    df["draft_year"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df["name_key"] = df["PLAYER_NAME"].map(normalize_name)
    out = (df[["player_id", "name_key", "draft_pick", "draft_year"]]
           .sort_values("draft_year").groupby("player_id").first().reset_index())
    _log(f"{len(out)} drafted players")
    return out


POSITION_MAP = {
    "G": "G", "PG": "G", "SG": "G", "G-F": "G", "Guard": "G",
    "F": "F", "SF": "F", "PF": "F", "F-G": "F", "F-C": "F", "Forward": "F",
    "C": "C", "C-F": "C", "Center": "C",
}


def position_group(raw: str) -> str:
    """Collapse listed position to G / F / C.

    Three groups, not five: the hierarchy in §4.1 needs enough players per level
    for the position-deviation aging curves to be estimable, and the PG/SG and
    PF/C splits are noisier than the guard/wing/big split they sit inside.
    """
    if not isinstance(raw, str):
        return "F"
    raw = raw.strip()
    if raw in POSITION_MAP:
        return POSITION_MAP[raw]
    head = raw.split("-")[0].strip()
    return POSITION_MAP.get(head, "F")


if __name__ == "__main__":
    print(load_combine().head().to_string())
    print(load_player_index().head().to_string())
    print(load_draft().head().to_string())
