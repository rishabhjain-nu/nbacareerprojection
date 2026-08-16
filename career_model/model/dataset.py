"""One object holding everything a fit needs: grid, design matrix, positions.

Keeps `fit_kf`, `simulate` and `validate` from each re-deriving the same joins,
and guarantees they all index players in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PANEL_PATH, POSITION_GROUPS, PRIORS_PATH
from ..data.build_panel import standardize_priors
from . import observations
from .aging import fit_basis


@dataclass
class Dataset:
    grid: observations.PanelGrid
    X: np.ndarray                  # (N, p_x) standardized prior covariates
    x_names: list[str]
    scaler: dict
    pos_idx: np.ndarray            # (N,) index into POSITION_GROUPS
    priors: pd.DataFrame           # one row per player, aligned to grid order
    panel: pd.DataFrame
    age_basis: object              # for delta(.)
    init_basis: object             # for the debut-age offset

    @property
    def n_players(self) -> int:
        return self.grid.n_players


def load(panel: pd.DataFrame | None = None, priors: pd.DataFrame | None = None,
         max_season_year: int | None = None) -> Dataset:
    """Build the fitting dataset.

    `max_season_year` is the backtest knob (§7.3): everything at or before it is
    filtered, everything after is unseen.  Expanding window, never k-fold --
    random folds put a player's age-30 season in the training set and ask the
    model about his age-27 season, which leaks the future into the past and
    flatters the model enormously.

    **Truncation is the single choke point for fold isolation.**  Every global
    quantity downstream is derived from the frame this function returns: the
    aging basis centring, the `x_i` standardisation, `phi_s`, `beta`,
    `Sigma_player`, the GBM prior mean, and the hazard's censoring rule.  Cut
    the panel here and all of them are confined to the fold; forget to, and the
    contamination is invisible in the output.  The assertion below is cheap
    insurance that the cut actually happened.
    """
    panel = pd.read_parquet(PANEL_PATH) if panel is None else panel
    priors = pd.read_parquet(PRIORS_PATH) if priors is None else priors
    if max_season_year is not None:
        panel = panel[panel["season_year"] <= max_season_year].copy()
        assert panel["season_year"].max() <= max_season_year
        # Career bookkeeping computed on the full panel is now wrong and would
        # be a back channel for post-cutoff information.  Recompute it, and drop
        # players left with nothing.
        panel = panel[panel.groupby("player_id")["player_id"].transform("size") > 0]
        g = panel.groupby("player_id")["season_year"]
        panel["panel_first_year"] = g.transform("min")
        panel["panel_last_year"] = g.transform("max")
        panel["season_index"] = panel.sort_values("season_year").groupby("player_id").cumcount()

    grid = observations.build_grid(panel)

    priors = priors.set_index("player_id").reindex(grid.player_ids).reset_index()
    X, names, scaler = standardize_priors(priors)

    pos_map = {g: i for i, g in enumerate(POSITION_GROUPS)}
    pos = (panel.sort_values("season_year").groupby("player_id")["position_group"]
           .first().reindex(grid.player_ids).fillna("F"))
    pos_idx = pos.map(pos_map).fillna(1).to_numpy().astype(int)

    # The aging basis is centred over the ages where transitions actually
    # happen; the debut basis over the ages players actually debut at.
    trans_ages = grid.age[grid.in_span]
    age_basis = fit_basis(trans_ages)
    init_basis = fit_basis(grid.age[:, 0])

    return Dataset(grid=grid, X=X, x_names=names, scaler=scaler, pos_idx=pos_idx,
                   priors=priors, panel=panel, age_basis=age_basis,
                   init_basis=init_basis)
