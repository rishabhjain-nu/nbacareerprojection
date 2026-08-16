"""Assemble the player-season panel (§2.2).  One row per player-season, counts
plus exposure, and **no row at all** for a season the player did not play.

That last point is load-bearing.  A zero row says "he took 0 shots in 4000
possessions", which is an observation of catastrophic failure.  An absent row
says "we know nothing about this year", which is what an ACL tear, a season in
Barcelona, or a G-League stint actually is.  The filter reads the second
correctly and nothing can rescue it from the first (§5.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    ALL_SEASONS, CACHE_DIR, COUNT_COLS, PANEL_PATH, PRIORS_PATH, PRIOR_COVARIATES,
)
from . import ingest_anthro, ingest_bbref, ingest_college, reconcile_ids

# Age is evaluated at the same point of every season so the aging spline is not
# picking up calendar drift.  1 February is roughly the midpoint of a season.
AGE_REFERENCE_MMDD = (2, 1)
DRAFT_REFERENCE_MMDD = (6, 26)     # the draft is held in late June
UNDRAFTED_PICK = 61.0


def _log(msg: str) -> None:
    print(f"[panel] {msg}", flush=True)


def _season_reference_date(season_year: pd.Series) -> pd.Series:
    m, d = AGE_REFERENCE_MMDD
    return pd.to_datetime(
        season_year.astype(int).astype(str) + f"-{m:02d}-{d:02d}", errors="coerce")


def build_panel(seasons: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = seasons or ALL_SEASONS

    # ---- counts + exposure -------------------------------------------------
    counts = ingest_bbref.load_counts(seasons)

    # A player traded mid-season appears once per team plus a TOT row in some
    # feeds; this pull is already season-aggregated, but guard anyway.
    dup = counts.duplicated(["player_id", "season"], keep=False)
    if dup.any():
        agg = {c: "sum" for c in COUNT_COLS}
        agg.update({"possessions": "sum", "minutes": "sum", "games_played": "sum",
                    "player_name": "first", "age_reported": "first",
                    "team_id": "last", "team_count": "max",
                    "season_year": "first", "possessions_imputed": "any"})
        counts = counts.groupby(["player_id", "season"], as_index=False).agg(agg)
        _log(f"collapsed {int(dup.sum())} duplicated player-season rows")

    # ---- schedule length ---------------------------------------------------
    # Lockouts and the 2020 bubble mean 82 is not always right, so the schedule
    # is observed rather than asserted: the most games any *single-team* player
    # managed.  Restricting to one-team players matters -- a player traded
    # mid-season can appear in more games than the schedule has.
    single = counts[counts["team_count"] <= 1]
    sched = single.groupby("season_year")["games_played"].max().rename("schedule_length")
    counts = counts.merge(sched, on="season_year", how="left")

    # ---- identity, age, position ------------------------------------------
    index = ingest_anthro.load_player_index()
    counts = counts.merge(index[["player_id", "position_raw", "index_height_in",
                                 "index_weight_lb", "country", "from_year"]],
                          on="player_id", how="left")
    counts["position_group"] = counts["position_raw"].map(ingest_anthro.position_group)

    births = reconcile_ids.load_birthdates(counts["player_id"].unique())
    counts = counts.merge(births, on="player_id", how="left")
    ref = _season_reference_date(counts["season_year"])
    counts["age_days"] = (ref - counts["birthdate"]).dt.days.astype("Float64")

    # Fall back to the endpoint's own integer age where a birthdate is missing.
    missing_age = counts["age_days"].isna()
    if missing_age.any():
        counts.loc[missing_age, "age_days"] = (
            pd.to_numeric(counts.loc[missing_age, "age_reported"], errors="coerce") * 365.25)
        _log(f"{int(missing_age.sum())} rows fell back to reported integer age")
    counts["age_days"] = pd.to_numeric(counts["age_days"], errors="coerce")
    counts["age"] = counts["age_days"] / 365.25
    counts = counts[counts["age"].between(15, 50)].reset_index(drop=True)

    # ---- career bookkeeping ------------------------------------------------
    counts = counts.sort_values(["player_id", "season_year"]).reset_index(drop=True)
    first = counts.groupby("player_id")["season_year"].min().rename("panel_first_year")
    last = counts.groupby("player_id")["season_year"].max().rename("panel_last_year")
    counts = counts.merge(first, on="player_id").merge(last, on="player_id")
    counts["season_index"] = counts.groupby("player_id").cumcount()

    # Careers that began before the panel window are missing their early years.
    # Their initial state must be diffuse, not prior-driven, so flag them.
    first_year = int(min(int(s[:4]) + 1 for s in seasons))
    counts["left_truncated"] = counts["panel_first_year"] <= first_year

    # ---- injury flag (§4.2: feeds the hazard, and inflates R) ---------------
    played_share = counts["games_played"] / counts["schedule_length"]
    counts["injury_season_flag"] = (played_share < 0.50) & (counts["season_index"] > 0)
    counts["games_share"] = played_share

    # ---- static prior covariates ------------------------------------------
    priors = build_priors(counts)

    keep = [
        "player_id", "player_name", "season", "season_year", "age_days", "age",
        "team_id", "team_count", "position_group",
        "possessions", "minutes", "games_played", "schedule_length", "games_share",
        *COUNT_COLS,
        "injury_season_flag", "possessions_imputed", "season_index",
        "panel_first_year", "panel_last_year", "left_truncated",
    ]
    panel = counts[keep].copy()
    ingest_bbref.assert_counts_only(panel)
    _log(f"panel: {len(panel)} rows, {panel['player_id'].nunique()} players")
    return panel, priors


def build_priors(counts: pd.DataFrame) -> pd.DataFrame:
    """The `x_i` block: one row per player, pre-NBA and time-invariant only."""
    players = (counts.sort_values("season_year")
               .groupby("player_id")
               .agg(player_name=("player_name", "first"),
                    position_group=("position_group", "first"),
                    birthdate=("birthdate", "first"),
                    first_nba_year=("season_year", "min"),
                    index_height_in=("index_height_in", "first"),
                    index_weight_lb=("index_weight_lb", "first"),
                    country=("country", "first"))
               .reset_index())
    players["name_key"] = players["player_name"].map(ingest_college.normalize_name)

    players = reconcile_ids.attach_college(players, ingest_college.build_player_table())
    players = reconcile_ids.attach_combine(players, ingest_anthro.load_combine())

    draft = ingest_anthro.load_draft()
    players = players.merge(draft[["player_id", "draft_pick", "draft_year"]],
                            on="player_id", how="left")
    players["undrafted"] = players["draft_pick"].isna().astype(int)
    players["draft_pick"] = players["draft_pick"].fillna(UNDRAFTED_PICK)
    players["draft_year"] = players["draft_year"].fillna(players["first_nba_year"] - 1)

    # Age at draft, in days.  The exact draft date is not in any endpoint, but a
    # fixed late-June reference is constant within a class, so the within-class
    # ordering the covariate actually encodes is exact.
    m, d = DRAFT_REFERENCE_MMDD
    draft_ref = pd.to_datetime(
        players["draft_year"].astype("Int64").astype(str) + f"-{m:02d}-{d:02d}",
        errors="coerce")
    players["age_at_draft_days"] = (draft_ref - players["birthdate"]).dt.days

    # Height: combine (without shoes) is the honest number; listed height is the
    # fallback and runs about an inch generous, which the standardisation absorbs.
    players["height_in"] = players["combine_height_in"].fillna(players["index_height_in"])
    players["weight_lb"] = players["weight_lb"].fillna(players["index_weight_lb"])

    for c in PRIOR_COVARIATES:
        if c not in players.columns:
            players[c] = np.nan
        players[c] = pd.to_numeric(players[c], errors="coerce")

    # §4.3: missing college data gets an explicit indicator and its own prior
    # mean, never an imputation to zero.  Same for the combine measurements,
    # which simply do not exist before 2000.
    players["has_college_data"] = players["college_bpm_sos"].notna()
    players["has_combine_data"] = players["wingspan_in"].notna()

    _log(f"priors: {len(players)} players, "
         f"{int(players['has_college_data'].sum())} with college, "
         f"{int(players['has_combine_data'].sum())} with combine")
    return players


def standardize_priors(priors: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    """Return the standardized design matrix `X` for `m_i ~ N(beta' x_i, ...)`.

    Layout, per §4.3 plus the missingness handling it mandates:
      [1, standardized covariates (NaN -> 0), has_college, has_combine]

    Zeroing a standardized covariate puts that player at the population mean for
    it, and the accompanying indicator lets `beta` learn a *separate intercept*
    for the missing group.  That is the "separate prior mean" the spec asks for,
    expressed as a linear model rather than a second model.
    """
    cols = list(PRIOR_COVARIATES)
    raw = priors[cols].to_numpy(dtype=float)
    mean = np.nanmean(raw, axis=0)
    sd = np.nanstd(raw, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    z = (raw - mean) / sd
    z[~np.isfinite(z)] = 0.0
    # Draft pick is the one covariate whose effect is plainly nonlinear -- the
    # gap between pick 1 and pick 10 dwarfs the gap between 40 and 50 -- so it
    # enters on a log scale as well.
    pick = priors["draft_pick"].to_numpy(dtype=float)
    log_pick = np.log(np.clip(pick, 1, None))
    log_pick = (log_pick - log_pick.mean()) / (log_pick.std() or 1.0)

    ind_college = priors["has_college_data"].to_numpy(dtype=float)
    ind_combine = priors["has_combine_data"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(priors)), z, log_pick, ind_college, ind_combine])
    names = ["intercept", *cols, "log_draft_pick", "has_college", "has_combine"]
    scaler = {"cols": cols, "mean": mean, "sd": sd,
              "log_pick_mean": float(np.log(np.clip(pick, 1, None)).mean()),
              "log_pick_sd": float(np.log(np.clip(pick, 1, None)).std() or 1.0)}
    return X, names, scaler


def main() -> None:
    panel, priors = build_panel()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    priors.drop(columns=[c for c in ("name_key",) if c in priors.columns]).to_parquet(
        PRIORS_PATH, index=False)
    _log(f"wrote {PANEL_PATH}")
    _log(f"wrote {PRIORS_PATH}")


if __name__ == "__main__":
    main()
