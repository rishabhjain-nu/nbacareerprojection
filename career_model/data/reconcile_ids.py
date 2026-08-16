"""One canonical player identity across four sources that agree on nothing.

The NBA stats endpoints key on `PERSON_ID`; barttorvik and the combine key on
name.  Name matching alone is not safe -- there are two Mike Dunleavys, two
Gary Paytons, a Marcus Morris and a Markieff Morris -- so a match is only
accepted when a second field corroborates it.

Rules, in order:

  1. Normalise the name (accents, punctuation, generational suffixes stripped).
  2. If exactly one candidate carries that key, accept it.
  3. If several do, break the tie on birthdate (college source carries one) or,
     failing that, on the year the college career ended versus the year the NBA
     career started -- a college senior does not debut six years later.
  4. If still ambiguous, reject.  A NaN prior covariate is honest; a covariate
     belonging to a different person is not.

`MANUAL_OVERRIDES` is the escape hatch for the residue.  It maps NBA player id
to college `name_key` and is expected to grow; that is the intended workflow,
not a failure of it.
"""

from __future__ import annotations

import pandas as pd

# NBA player_id -> college name_key.  Populated as mismatches are found by hand.
MANUAL_OVERRIDES: dict[int, str] = {
    # 203507: "giannisantetokounmpo",   # example shape; Giannis has no college row
}

# NBA player_id values whose name collides with a *different* college player and
# must never be auto-matched.
BLOCKLIST: set[int] = set()


def _log(msg: str) -> None:
    print(f"[reconcile] {msg}", flush=True)


def attach_college(players: pd.DataFrame, college: pd.DataFrame) -> pd.DataFrame:
    """Join the college table onto a per-player NBA frame.

    `players` needs: player_id, name_key, birthdate (may be NaT), first_nba_year.
    Returns `players` with the college columns added and a `college_matched` flag.
    """
    if college.empty:
        out = players.copy()
        out["college_matched"] = False
        return out

    counts = college["name_key"].value_counts()
    unique_keys = set(counts[counts == 1].index)
    dup_keys = set(counts[counts > 1].index)

    chosen: dict[int, int] = {}   # player_id -> positional index into `college`
    college = college.reset_index(drop=True)
    dup_rows = college[college["name_key"].isin(dup_keys)]

    n_unique = n_tiebreak = n_reject = n_manual = 0
    for row in players.itertuples(index=False):
        pid = int(row.player_id)
        if pid in BLOCKLIST:
            continue
        key = MANUAL_OVERRIDES.get(pid, row.name_key)
        if pid in MANUAL_OVERRIDES:
            hit = college.index[college["name_key"] == key]
            if len(hit):
                chosen[pid] = int(hit[0])
                n_manual += 1
            continue
        if key in unique_keys:
            chosen[pid] = int(college.index[college["name_key"] == key][0])
            n_unique += 1
        elif key in dup_keys:
            cands = dup_rows[dup_rows["name_key"] == key]
            pick = _break_tie(row, cands)
            if pick is not None:
                chosen[pid] = pick
                n_tiebreak += 1
            else:
                n_reject += 1

    college_cols = [c for c in college.columns if c != "name_key"]
    attached = pd.DataFrame(index=players.index, columns=college_cols, dtype="object")
    pos = {int(p): i for i, p in enumerate(players["player_id"])}
    for pid, cidx in chosen.items():
        attached.iloc[pos[pid]] = college.loc[cidx, college_cols].to_numpy()

    out = players.copy()
    for c in college_cols:
        if c == "college_birthdate":
            out[c] = pd.to_datetime(attached[c], errors="coerce")
        else:
            out[c] = pd.to_numeric(attached[c], errors="coerce")
    out["college_matched"] = out.index.isin([pos[p] for p in chosen])

    _log(f"college matched {len(chosen)}/{len(players)} players "
         f"({n_unique} unique, {n_tiebreak} tie-broken, {n_manual} manual, "
         f"{n_reject} rejected as ambiguous)")
    return out


def _break_tie(player, candidates: pd.DataFrame) -> int | None:
    """Return the positional index of the one candidate that corroborates."""
    bd = getattr(player, "birthdate", None)
    if bd is not None and not pd.isna(bd):
        cb = pd.to_datetime(candidates["college_birthdate"], errors="coerce")
        same = candidates.index[(cb - pd.Timestamp(bd)).abs() < pd.Timedelta(days=2)]
        if len(same) == 1:
            return int(same[0])
        if len(same) > 1:
            return None

    first_nba = getattr(player, "first_nba_year", None)
    if first_nba is not None and not pd.isna(first_nba):
        last_col = pd.to_numeric(candidates["college_last_year"], errors="coerce")
        gap = (first_nba - last_col).abs()
        plausible = candidates.index[gap <= 2]
        if len(plausible) == 1:
            return int(plausible[0])
    return None


def attach_combine(players: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    """Combine measurements join on name only -- but the combine pool is the
    draft class, so a collision needs the two players to have entered the same
    draft.  Rare enough that a plain unique-key join is safe; duplicates drop.
    """
    if combine.empty:
        out = players.copy()
        for c in ("combine_height_in", "weight_lb", "wingspan_in", "standing_reach_in"):
            out[c] = float("nan")
        return out
    counts = combine["name_key"].value_counts()
    safe = combine[combine["name_key"].isin(counts[counts == 1].index)]
    n_before = players["player_id"].nunique()
    out = players.merge(safe, on="name_key", how="left")
    assert out["player_id"].nunique() == n_before, "combine join changed the player set"
    _log(f"combine matched {int(out['wingspan_in'].notna().sum())}/{len(out)} players")
    return out


def load_birthdates(player_ids) -> pd.DataFrame:
    """Per-player birthdate from the cached `commonplayerinfo` responses."""
    from ..config import RAW_DIR
    info_dir = RAW_DIR / "playerinfo"
    rows = []
    wanted = set(int(p) for p in player_ids)
    for path in info_dir.glob("*.csv"):
        try:
            pid = int(path.stem)
        except ValueError:
            continue
        if pid not in wanted:
            continue
        try:
            d = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if "BIRTHDATE" not in d.columns or not len(d):
            continue
        rows.append({"player_id": pid,
                     "birthdate": pd.to_datetime(d["BIRTHDATE"].iloc[0], errors="coerce")})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["player_id", "birthdate"])
    out["birthdate"] = out["birthdate"].dt.tz_localize(None)
    _log(f"{out['birthdate'].notna().sum()}/{len(wanted)} birthdates resolved")
    return out
