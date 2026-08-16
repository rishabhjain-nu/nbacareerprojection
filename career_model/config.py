"""Paths, season coverage, and the canonical state-vector definition.

Everything downstream imports the state layout from here, so the 14 dimensions
are declared exactly once.  §3.1 of the spec.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data_raw"
CACHE_DIR = ROOT / "cache"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"
PROJECTION_DIR = ROOT / "app" / "projections"

for _d in (CACHE_DIR, ARTIFACT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PANEL_PATH = CACHE_DIR / "panel.parquet"
PRIORS_PATH = CACHE_DIR / "priors.parquet"

MODEL_VERSION = "ssm-v1.0"

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
FIRST_SEASON_YEAR = 1997          # season *end* year: 1996-97
LAST_SEASON_YEAR = 2026           # 2025-26
TRAIN_CUTOFF_YEAR = 2026          # last season fed to the filter


def season_str(end_year: int) -> str:
    """2019 -> '2018-19'."""
    return f"{end_year - 1}-{str(end_year)[-2:]}"


ALL_SEASONS = [season_str(y) for y in range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1)]

# ---------------------------------------------------------------------------
# State vector (§3.1).  Order matters and is fixed forever.
# ---------------------------------------------------------------------------
# Volume dimensions: log(count / possession).  Each names the count column and
# the exposure column in the panel.
VOLUME_STATS = [
    "fga_2p", "fga_3p", "fta", "oreb", "dreb", "ast", "tov", "stl", "blk", "pf",
]
# Accuracy dimensions: logit(makes / attempts).  (made_col, attempt_col)
ACCURACY_STATS = ["pct_2p", "pct_3p", "pct_ft"]
ACCURACY_PAIRS = {
    "pct_2p": ("fgm_2p", "fga_2p"),
    "pct_3p": ("fgm_3p", "fga_3p"),
    "pct_ft": ("ftm", "fta"),
}
AVAILABILITY_STAT = "log_poss"

STATE_NAMES = VOLUME_STATS + ACCURACY_STATS + [AVAILABILITY_STAT]
S = len(STATE_NAMES)                                   # 14
IDX = {name: i for i, name in enumerate(STATE_NAMES)}

VOLUME_IDX = [IDX[s] for s in VOLUME_STATS]
ACCURACY_IDX = [IDX[s] for s in ACCURACY_STATS]
AVAIL_IDX = IDX[AVAILABILITY_STAT]

DISPLAY_LABEL = {
    "fga_2p": "2PA", "fga_3p": "3PA", "fta": "FTA", "oreb": "OREB", "dreb": "DREB",
    "ast": "AST", "tov": "TOV", "stl": "STL", "blk": "BLK", "pf": "PF",
    "pct_2p": "2P%", "pct_3p": "3P%", "pct_ft": "FT%", "log_poss": "Possessions",
}

# Raw count columns carried on every panel row (§2.2).
COUNT_COLS = [
    "fga_2p", "fgm_2p", "fga_3p", "fgm_3p", "fta", "ftm",
    "oreb", "dreb", "ast", "tov", "stl", "blk", "pf",
]

# ---------------------------------------------------------------------------
# Prior covariates x_i (§4.3) -- pre-NBA / time-invariant only.
# ---------------------------------------------------------------------------
COLLEGE_COVARIATES = [
    "college_bpm_sos", "college_obpm", "college_dbpm", "college_stl_pct",
    "college_blk_pct", "college_ft_pct", "college_tp_pct", "college_ts_pct",
    "college_usg", "college_ast_pct", "college_oreb_pct", "college_dreb_pct",
    "college_conf_strength", "college_age_adj_production", "college_min_pct",
    "college_seasons", "college_age_final_season",
]
ANTHRO_COVARIATES = ["height_in", "weight_lb", "wingspan_in", "standing_reach_in"]
DRAFT_COVARIATES = ["draft_pick", "undrafted", "age_at_draft_days"]

PRIOR_COVARIATES = COLLEGE_COVARIATES + ANTHRO_COVARIATES + DRAFT_COVARIATES

POSITION_GROUPS = ["G", "F", "C"]

# ---------------------------------------------------------------------------
# Aging spline (§3.4)
# ---------------------------------------------------------------------------
AGE_KNOTS = (21.0, 25.0, 29.0, 33.0)
AGE_BOUNDARY = (18.0, 42.0)

# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
MISSING_R = 1.0e10   # observation variance standing in for "not observed"
JITTER = 1.0e-9
