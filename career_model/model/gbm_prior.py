"""The LightGBM stack's role in this model (§5.3).

    m_i ~ N(f_GBM(x_i), Sigma_player)

The GBM is not discarded and it is not a competitor here.  It does
cross-sectional pattern-matching on college, combine and draft data -- where the
interactions are nonlinear (a 7-footer's block rate and a guard's block rate are
different functions of wingspan) and where native NaN handling beats any
imputation.  The state-space model owns temporal dynamics and uncertainty
propagation.  Each does the half it is better at.

`f_GBM(x_i)` enters as a **fixed offset**, not a fitted layer: the filter treats
it as known and `beta` then only has to explain what the trees left behind.

The targets are the filter's own `E[m_i | career]` from a preliminary fit --
the closest thing to an observation of `m_i` that exists.  They are produced
**out of fold, grouped by player**, so no player's offset is informed by his own
target.  Skipping that would make the prior mean look prescient in backtests and
collapse `Sigma_player` toward zero, which would then make every rookie's
interval too narrow -- the one failure this interface is built to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PRIOR_COVARIATES, S, STATE_NAMES

N_FOLDS = 5
GBM_PARAMS = dict(
    objective="regression", n_estimators=300, learning_rate=0.05,
    num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
)


@dataclass
class GBMPrior:
    """The trees are trained on `m_hat - centre`, so what they emit is already a
    **deviation** from the population mean of `m_i`, which is the units the
    filter's `gbm_offset` is added in (`beta` explains the rest).  `centre` is
    kept only for reconstructing an absolute level for inspection -- subtracting
    it again on the way out would double-count it and shift every prospect's
    prior by the league mean, which is a ~2.5 log-unit error in a quantity whose
    entire spread is 0.3."""

    models: list          # one booster per state dimension, trained on all players
    columns: list[str]
    centre: np.ndarray    # (S,) target means; NOT applied by `predict`

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Offset in deviation units, directly comparable to the `oof` array
        returned by `fit_offsets`."""
        X = _design(frame, self.columns)
        return np.column_stack([m.predict(X) for m in self.models])

    def predict_absolute(self, frame: pd.DataFrame) -> np.ndarray:
        """Implied level of `m_i` itself, for inspection only."""
        return self.predict(frame) + self.centre


def _log(msg: str) -> None:
    print(f"[gbm] {msg}", flush=True)


def _design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Raw, unstandardized, NaN preserved.  Trees want the original scale and
    LightGBM routes missing values down their own branch."""
    cols = []
    for c in columns:
        cols.append(pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
                    if c in frame.columns else np.full(len(frame), np.nan))
    return np.column_stack(cols)


def fit_offsets(ds, m_hat: np.ndarray, seed: int = 0) -> tuple[np.ndarray, GBMPrior]:
    """Return out-of-fold offsets for the panel players plus a full-data model.

    `m_hat` is `E[m_i | data]` from a preliminary fit -- shape (N, S).
    """
    import lightgbm as lgb

    columns = list(PRIOR_COVARIATES)
    X = _design(ds.priors, columns)
    n = len(X)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % N_FOLDS      # grouped by player: one row per player

    centre = m_hat.mean(axis=0)
    target = m_hat - centre

    oof = np.zeros_like(target)
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        for s in range(S):
            model = lgb.LGBMRegressor(**GBM_PARAMS, random_state=seed + s)
            model.fit(X[tr], target[tr, s])
            oof[te, s] = model.predict(X[te])

    models = []
    for s in range(S):
        model = lgb.LGBMRegressor(**GBM_PARAMS, random_state=seed + s)
        model.fit(X, target[:, s])
        models.append(model)

    r2 = 1.0 - ((target - oof) ** 2).sum(axis=0) / ((target - target.mean(0)) ** 2).sum(axis=0)
    _log("out-of-fold R^2 on E[m_i]: "
         + ", ".join(f"{STATE_NAMES[s]} {r2[s]:.2f}" for s in range(S)))
    return oof, GBMPrior(models=models, columns=columns, centre=centre)
