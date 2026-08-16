"""Season-total box score counts plus the possession exposure denominator.

§1.1 is the whole point of this module: what leaves here is **integer counts and
an exposure column**, never a rate.  A 5.0 AST/100 could be 15 assists or 250,
and the filter runs on the difference.

Source is the cached `leaguedashplayerstats` pull:

  * `Base / Totals`               -> the counts
  * `Advanced / Per100Possessions` -> `POSS`, the possessions the player was on
                                      the floor for, which is the exposure

Both are cached per season under `data_raw/leaguedash/`.  Nothing here talks to
the network; re-pulling is the reference project's job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ALL_SEASONS, RAW_DIR

LEAGUEDASH_DIR = RAW_DIR / "leaguedash"

# Totals columns we keep, mapped to panel names.  FGA/FGM are *split* below --
# they never survive as aggregates (§2.2).
_TOTAL_MAP = {
    "FGA": "_fga_all", "FGM": "_fgm_all", "FG3A": "fga_3p", "FG3M": "fgm_3p",
    "FTA": "fta", "FTM": "ftm", "OREB": "oreb", "DREB": "dreb", "AST": "ast",
    "TOV": "tov", "STL": "stl", "BLK": "blk", "PF": "pf",
}


def _log(msg: str) -> None:
    print(f"[bbref] {msg}", flush=True)


def _read(season: str, measure: str, per: str) -> pd.DataFrame | None:
    path = LEAGUEDASH_DIR / f"{season}_{measure}_{per}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_season(season: str) -> pd.DataFrame | None:
    totals = _read(season, "Base", "Totals")
    adv = _read(season, "Advanced", "Per100Possessions")
    if totals is None or adv is None:
        _log(f"{season}: missing cache file, skipped")
        return None

    keep = ["PLAYER_ID", "PLAYER_NAME", "AGE", "GP", "MIN", "TEAM_ABBREVIATION",
            "TEAM_COUNT", *(_TOTAL_MAP)]
    keep = [c for c in keep if c in totals.columns]
    df = totals[keep].rename(columns=_TOTAL_MAP).rename(
        columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
                 "AGE": "age_reported", "GP": "games_played", "MIN": "minutes",
                 "TEAM_ABBREVIATION": "team_id", "TEAM_COUNT": "team_count"}
    )

    # Exposure.  POSS from the Advanced table is the player's own on-floor
    # possession count -- exactly the E_{i,t} of §3.
    poss = adv[["PLAYER_ID", "POSS"]].rename(
        columns={"PLAYER_ID": "player_id", "POSS": "possessions"})
    df = df.merge(poss, on="player_id", how="left")

    # 2-pointers are FGA minus 3PA.  A player moving midrange attempts behind
    # the arc is a real state change and must not vanish into an aggregate.
    df["fga_2p"] = df["_fga_all"] - df["fga_3p"]
    df["fgm_2p"] = df["_fgm_all"] - df["fgm_3p"]
    df = df.drop(columns=["_fga_all", "_fgm_all"])

    df["season"] = season
    df["season_year"] = int(season[:4]) + 1
    return df


def load_counts(seasons: list[str] | None = None) -> pd.DataFrame:
    seasons = seasons or ALL_SEASONS
    frames = [d for d in (load_season(s) for s in seasons) if d is not None]
    if not frames:
        raise FileNotFoundError(f"no leaguedash cache found under {LEAGUEDASH_DIR}")
    df = pd.concat(frames, ignore_index=True)

    count_cols = ["fga_2p", "fgm_2p", "fga_3p", "fgm_3p", "fta", "ftm",
                  "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round().astype("int64")

    df["possessions"] = pd.to_numeric(df["possessions"], errors="coerce").astype("float64")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").astype("float64")

    # A handful of very old seasons predate the POSS field.  Possessions run at
    # a stable ~2.1 per player-minute (the league pace band is narrow); fall
    # back to that rather than dropping the season, and flag it.
    bad = df["possessions"].isna() | (df["possessions"] <= 0)
    if bad.any():
        rate = float((df.loc[~bad, "possessions"] / df.loc[~bad, "minutes"]).median())
        df.loc[bad, "possessions"] = df.loc[bad, "minutes"] * rate
        _log(f"imputed possessions for {int(bad.sum())} rows at {rate:.3f} poss/min")
    df["possessions_imputed"] = bad

    # Zero-minute rows carry no information and break log(E).
    n0 = int((df["possessions"] < 1).sum())
    if n0:
        df = df[df["possessions"] >= 1].reset_index(drop=True)
        _log(f"dropped {n0} rows with <1 possession")

    _log(f"{len(df)} player-seasons across {df['season'].nunique()} seasons")
    return df


def assert_counts_only(df: pd.DataFrame) -> None:
    """§10.1 -- fail loudly if a rate column ever sneaks into the panel."""
    banned = [c for c in df.columns
              if any(t in c.lower() for t in ("_100", "per36", "per_36", "_pg", "_pct"))
              and not c.startswith("college_")]
    if banned:
        raise AssertionError(f"rate columns must not enter the model panel: {banned}")
    counts = ["fga_2p", "fgm_2p", "fga_3p", "fgm_3p", "fta", "ftm",
              "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]
    for c in counts:
        if not pd.api.types.is_integer_dtype(df[c]):
            raise AssertionError(f"{c} is {df[c].dtype}, must be an integer count")
    if df["possessions"].isna().any() or (df["possessions"] <= 0).any():
        raise AssertionError("every row needs a positive possessions exposure")
    # Makes can never exceed attempts.
    for made, att in (("fgm_2p", "fga_2p"), ("fgm_3p", "fga_3p"), ("ftm", "fta")):
        if (df[made] > df[att]).any():
            raise AssertionError(f"{made} exceeds {att} on some rows")
    if np.isfinite(df["possessions"]).all() is np.False_:
        raise AssertionError("non-finite possessions")


if __name__ == "__main__":
    d = load_counts()
    assert_counts_only(d)
    print(d.head().to_string())
