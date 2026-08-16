"""A small synthetic panel, so tests do not depend on the cached data pull."""

from __future__ import annotations

import numpy as np
import pandas as pd

from career_model.config import COUNT_COLS
from career_model.model.dataset import Dataset


def synthetic_panel(n_players: int = 120, seed: int = 0,
                    gap_players: tuple = (3, 7, 11)) -> pd.DataFrame:
    """Careers of varying length and volume, with deliberate mid-career gaps."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        pid = 1000 + i
        n = int(rng.integers(2, 13))
        start = int(rng.integers(1998, 2018))
        age = float(rng.integers(19, 26))
        level = rng.normal(0, 0.4, size=len(COUNT_COLS))
        skip = {start + 3} if i % 17 in gap_players else set()
        for t in range(n):
            year = start + t
            if year in skip or year > 2026:
                continue
            poss = float(np.clip(rng.gamma(4, 500), 60, 5200))
            rec = {"player_id": pid, "player_name": f"Player {i}",
                   "season": f"{year - 1}-{str(year)[-2:]}", "season_year": year,
                   "age": age + t, "age_days": (age + t) * 365.25,
                   "team_id": "AAA", "team_count": 1,
                   "position_group": ["G", "F", "C"][i % 3],
                   "possessions": poss, "minutes": poss / 2.02,
                   "games_played": int(min(82, max(5, poss / 30))),
                   "schedule_length": 82, "games_share": 0.8,
                   "injury_season_flag": False, "possessions_imputed": False,
                   "season_index": t, "panel_first_year": start,
                   "panel_last_year": start + n - 1, "left_truncated": False}
            base = {"fga_2p": 0.09, "fgm_2p": 0.045, "fga_3p": 0.05, "fgm_3p": 0.018,
                    "fta": 0.04, "ftm": 0.031, "oreb": 0.02, "dreb": 0.06,
                    "ast": 0.045, "tov": 0.025, "stl": 0.012, "blk": 0.008, "pf": 0.035}
            for k, c in enumerate(COUNT_COLS):
                rate = base[c] * np.exp(level[k] + rng.normal(0, 0.12))
                rec[c] = int(rng.poisson(rate * poss))
            rec["fgm_2p"] = min(rec["fgm_2p"], rec["fga_2p"])
            rec["fgm_3p"] = min(rec["fgm_3p"], rec["fga_3p"])
            rec["ftm"] = min(rec["ftm"], rec["fta"])
            rows.append(rec)
    return pd.DataFrame(rows)


def synthetic_priors(panel: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    from career_model.config import PRIOR_COVARIATES
    rng = np.random.default_rng(seed)
    pid = panel["player_id"].drop_duplicates().sort_values().to_numpy()
    out = pd.DataFrame({"player_id": pid})
    for c in PRIOR_COVARIATES:
        vals = rng.normal(0, 1, len(pid))
        vals[rng.random(len(pid)) < 0.2] = np.nan       # realistic missingness
        out[c] = vals
    out["draft_pick"] = rng.integers(1, 61, len(pid)).astype(float)
    out["undrafted"] = 0
    out["has_college_data"] = out["college_bpm_sos"].notna()
    out["has_combine_data"] = out["wingspan_in"].notna()
    out["player_name"] = [f"Player {p - 1000}" for p in pid]
    out["position_group"] = "F"
    out["first_nba_year"] = panel.groupby("player_id")["season_year"].min().to_numpy()
    return out


def small_dataset(n_players: int = 120, seed: int = 0) -> Dataset:
    from career_model.model.dataset import load
    panel = synthetic_panel(n_players, seed)
    return load(panel=panel, priors=synthetic_priors(panel))
