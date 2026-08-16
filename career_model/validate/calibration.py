"""Distributional scoring: CRPS, PIT, interval coverage (§7.2).

Calibration is the primary criterion, and the model is expected to *lose* to the
LightGBM stack on point-prediction RMSE for players with long histories.  That
is not a concession, it is the trade: RMSE rewards a confident centre and is
blind to whether the interval around it means anything.  A model whose 80%
intervals cover 62% of outcomes is wrong in a way RMSE cannot see and CRPS can.

  * **CRPS** is the headline.  Proper scoring rule for a full predictive
    distribution, in the units of the variable, reducing to absolute error when
    the forecast is a point mass -- so it is directly comparable against a point
    predictor without giving either side an advantage.
  * **PIT histograms** should be uniform.  U-shaped means the intervals are too
    narrow (`Q` underestimated); dome-shaped means too wide.  Counts are
    discrete, so the PIT is randomized -- without that, ties at low counts pile
    up at the histogram edges and a perfectly calibrated model looks U-shaped.
  * **Coverage** is checked at h=1 *and* h=5 separately, because an average over
    horizons hides a coverage that decays with horizon (the failure mode of the
    recursive-bootstrap benchmark).  Note the state mean-reverts, so the
    *latent* variance saturates rather than growing linearly; the test of
    correctness is that coverage stays near nominal at each horizon, not that a
    displayed band always widens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crps_ensemble(samples: np.ndarray, y: float) -> float:
    """CRPS of an empirical predictive distribution.

        CRPS = E|X - y| - 0.5 E|X - X'|

    The second term is computed from the sorted sample in O(n log n) rather than
    the O(n^2) double sum.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return np.nan
    term1 = np.abs(x - y).mean()
    i = np.arange(n)
    term2 = (2.0 * np.sum((2 * i - n + 1) * x)) / (n * n)
    return float(term1 - 0.5 * term2)


def crps_batch(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized CRPS over rows: `samples` is (n_obs, n_draws), `y` is (n_obs,)."""
    x = np.sort(np.asarray(samples, dtype=float), axis=1)
    n = x.shape[1]
    term1 = np.abs(x - np.asarray(y, float)[:, None]).mean(axis=1)
    i = np.arange(n)
    term2 = 2.0 * (x * (2 * i - n + 1)).sum(axis=1) / (n * n)
    return term1 - 0.5 * term2


def pit(samples: np.ndarray, y: np.ndarray, rng=None) -> np.ndarray:
    """Randomized probability integral transform, one value per observation.

    For a discrete forecast the naive rank PIT is not uniform even under a
    perfect model.  Randomizing within the tied mass fixes it, which matters
    here because low-count stats (blocks, steals for a wing) are mostly ties.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(samples, float)
    y = np.asarray(y, float)[:, None]
    below = (x < y).mean(axis=1)
    equal = (x == y).mean(axis=1)
    return below + rng.random(len(below)) * equal


def coverage(samples: np.ndarray, y: np.ndarray, level: float = 0.80) -> dict:
    lo = np.percentile(samples, 100 * (1 - level) / 2, axis=1)
    hi = np.percentile(samples, 100 * (1 + level) / 2, axis=1)
    inside = (y >= lo) & (y <= hi)
    return {"level": level, "coverage": float(inside.mean()),
            "mean_width": float(np.mean(hi - lo)), "n": int(len(y))}


def pit_histogram(pit_values: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    counts, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    expected = len(pit_values) / n_bins
    return pd.DataFrame({
        "bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts,
        "expected": expected,
        "ratio": counts / expected if expected > 0 else np.nan,
    })


def pit_shape(pit_values: np.ndarray) -> str:
    """One-word diagnosis, so the direction of a miscalibration is not left to
    the reader's eye on a histogram."""
    if len(pit_values) < 50:
        return "insufficient"
    edge = np.mean((pit_values < 0.1) | (pit_values > 0.9))
    if edge > 0.28:
        return "U-shaped (intervals too narrow: Q underestimated)"
    if edge < 0.12:
        return "dome-shaped (intervals too wide)"
    lo, hi = np.mean(pit_values < 0.1), np.mean(pit_values > 0.9)
    if lo > 2 * hi and lo > 0.2:
        return "left-skewed (forecast biased high)"
    if hi > 2 * lo and hi > 0.2:
        return "right-skewed (forecast biased low)"
    return "approximately uniform"


def score_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Summarise a long frame of scored predictions.

    `records` needs: stat, horizon, crps, pit, inside_80, inside_50, abs_err,
    and `sq_err`.
    """
    def summarise(g):
        return pd.Series({
            "n": len(g),
            "crps": g["crps"].mean(),
            "rmse": np.sqrt(g["sq_err"].mean()),
            "mae": g["abs_err"].mean(),
            "cover_50": g["inside_50"].mean(),
            "cover_80": g["inside_80"].mean(),
            "mean_width_80": g["width_80"].mean(),
            "pit_shape": pit_shape(g["pit"].to_numpy()),
        })
    return (records.groupby(["stat", "horizon"], observed=True)
            .apply(summarise, include_groups=False).reset_index())


def survival_calibration(pred: np.ndarray, actual: np.ndarray, age: np.ndarray,
                         n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predicted vs actual continuation rates, bucketed by age and by prediction."""
    df = pd.DataFrame({"p": pred, "y": actual, "age": age})
    df["p_bucket"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    df["age_bucket"] = pd.cut(df["age"], [0, 22, 25, 28, 31, 34, 99],
                              labels=["<=22", "23-25", "26-28", "29-31", "32-34", "35+"])
    agg = dict(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return (df.groupby("p_bucket", observed=True).agg(**agg).reset_index(),
            df.groupby("age_bucket", observed=True).agg(**agg).reset_index())
