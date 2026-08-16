"""Head-to-head against the LightGBM stack it replaces (§0, §7.2).

The benchmark is a compact rebuild of the three-sub-model framework: a survival
classifier, a minutes/possessions regressor, and one regressor per stat, all fed
the old feature vocabulary (last season, Marcel 5/4/3 weights, career shape,
age), run recursively for multi-season horizons, with a **joint** residual
bootstrap for uncertainty -- residuals resampled row-wise so cross-stat
correlation survives, which is the strongest version of the old approach.

Two things this comparison is set up to show, and one it is not.

It is *expected* to lose on RMSE for players with long histories.  Gradient
boosting on a rich lag vocabulary is very good at conditional means, and no
amount of state-space machinery changes that.

It is expected to win on CRPS, on coverage, and above all on how coverage
behaves as the horizon grows.  The bootstrap injects residual noise once per
recursive step from a pool fitted at h=1, so its intervals widen at roughly
sqrt(h); the state-space intervals grow with the correct mean-reverting AR(1)
covariance `P_h = A^h P_0 (A^h)' + sum_{j<h} A^j Q (A^j)'` (which saturates
toward the stationary variance, not the random-walk `P_0 + hQ`), with parameter
uncertainty on top.  At h=1 the two can look similar.  At h=5 they should not.

What this is *not* is a claim that the GBM is useless -- §5.3 puts it inside the
model as the prior mean, which is where its cross-sectional strength belongs.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, PANEL_PATH, PRIORS_PATH, PRIOR_COVARIATES
from ..simulate import derive
from . import calibration as cal
from .backtest import SCORED

RATE_STATS = derive.RATE_COLS
MARCEL_WEIGHTS = (5.0, 4.0, 3.0)


def _log(msg: str) -> None:
    print(f"[lgbm] {msg}", flush=True)


def build_features(panel: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """The old 75-ish feature vocabulary, rebuilt.

    Per-100 rates are legitimate *here* -- this is the benchmark, and rates as
    GBM inputs are exactly what §1.1 says the old framework did.  They are still
    forbidden anywhere in `model/`.
    """
    df = panel.sort_values(["player_id", "season_year"]).copy()
    e = df["possessions"].to_numpy(dtype=float)
    for c in RATE_STATS:
        df[f"{c}_r"] = 100.0 * df[c].to_numpy(dtype=float) / e
    df["pts_r"] = (df["ftm_r"] + 2 * df["fgm_2p_r"] + 3 * df["fgm_3p_r"])
    rate_cols = [f"{c}_r" for c in RATE_STATS] + ["pts_r"]

    g = df.groupby("player_id", sort=False)
    feats = {"player_id": df["player_id"], "season_year": df["season_year"],
             "age": df["age"], "possessions": df["possessions"],
             "minutes": df["minutes"], "games_played": df["games_played"],
             "exp": df.groupby("player_id").cumcount()}

    for c in rate_cols:
        feats[f"{c}_lag1"] = df[c]
        lag2, lag3 = g[c].shift(1), g[c].shift(2)
        w1, w2, w3 = MARCEL_WEIGHTS
        num = w1 * df[c] + w2 * lag2.fillna(df[c]) + w3 * lag3.fillna(df[c])
        feats[f"{c}_marcel"] = num / (w1 + w2 + w3)
        feats[f"{c}_career"] = g[c].transform(lambda s: s.expanding().mean())
        feats[f"{c}_max"] = g[c].transform(lambda s: s.expanding().max())
        feats[f"{c}_delta"] = df[c] - lag2
    feats["poss_lag1"] = df["possessions"]
    feats["poss_career"] = g["possessions"].transform(lambda s: s.expanding().mean())

    out = pd.DataFrame(feats)
    out = out.merge(priors[["player_id", *PRIOR_COVARIATES]], on="player_id", how="left")
    return out, df[["player_id", "season_year", *rate_cols, "possessions"]]


def fit_and_project(cutoff: int, max_h: int = 5, n_draws: int = 800,
                    seed: int = 0) -> pd.DataFrame:
    """Train through `cutoff`, project forward recursively, score."""
    import lightgbm as lgb

    panel = pd.read_parquet(PANEL_PATH)
    priors = pd.read_parquet(PRIORS_PATH)
    feats, rates = build_features(panel, priors)
    rate_cols = [c for c in rates.columns if c.endswith("_r")]

    # Targets are next season's values, aligned on (player, season+1).
    nxt = rates.copy()
    nxt["season_year"] = nxt["season_year"] - 1
    data = feats.merge(nxt, on=["player_id", "season_year"], suffixes=("", "_next"))
    train = data[data["season_year"] < cutoff]
    if not len(train):
        raise ValueError(f"no training rows before {cutoff}")

    x_cols = [c for c in feats.columns if c not in ("player_id", "season_year")]
    X = train[x_cols].to_numpy(dtype=float)

    params = dict(objective="regression", n_estimators=400, learning_rate=0.05,
                  num_leaves=31, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    models, resid = {}, {}
    for c in rate_cols + ["possessions_next"]:
        tgt = c if c.endswith("_next") else f"{c}_next"
        if tgt not in train.columns:
            tgt = c
        y = train[tgt].to_numpy(dtype=float)
        ok = np.isfinite(y)
        m = lgb.LGBMRegressor(**params, random_state=seed)
        m.fit(X[ok], y[ok])
        models[c] = m
        resid[c] = y[ok] - m.predict(X[ok])

    # Survival: appears in season+1 at all.
    played_next = data["player_id"].notna().to_numpy()
    surv_train = feats[feats["season_year"] < cutoff].copy()
    key = set(zip(panel["player_id"], panel["season_year"]))
    surv_y = np.array([(int(p), int(s) + 1) in key
                       for p, s in zip(surv_train["player_id"], surv_train["season_year"])],
                      dtype=float)
    surv = lgb.LGBMClassifier(**{**params, "objective": "binary"}, random_state=seed)
    surv.fit(surv_train[x_cols].to_numpy(dtype=float), surv_y)
    _ = played_next

    # Joint residual pool: resample whole rows so cross-stat correlation is kept.
    n_res = min(len(v) for v in resid.values())
    pool = np.column_stack([v[:n_res] for v in resid.values()])
    pool_cols = list(resid)

    rng = np.random.default_rng(seed)
    live_rows = feats[feats["season_year"] == cutoff]
    truth = rates.set_index(["player_id", "season_year"])

    records = []
    for _, row in live_rows.iterrows():
        pid = int(row["player_id"])
        state = row[x_cols].to_numpy(dtype=float).copy()
        draws = np.tile(state, (n_draws, 1))
        alive = np.ones(n_draws, dtype=bool)
        for h in range(max_h):
            year = cutoff + h + 1
            p_surv = surv.predict_proba(draws)[:, 1]
            alive &= rng.random(n_draws) < p_surv
            pred = {c: models[c].predict(draws) for c in pool_cols}
            idx = rng.integers(0, n_res, size=n_draws)
            sim = {c: pred[c] + pool[idx, k] for k, c in enumerate(pool_cols)}

            if (pid, year) in truth.index and alive.sum() > 50:
                actual = truth.loc[(pid, year)]
                for stat in SCORED:
                    src = _map_stat(stat)
                    if src is None or src not in sim:
                        continue
                    samples = sim[src][alive]
                    # The simulated key and the truth column differ for
                    # possessions: the target was aligned onto season-1 and so
                    # carries the `_next` suffix, while the truth frame holds it
                    # under its plain name.
                    truth_col = src[:-5] if src.endswith("_next") else src
                    y = float(actual[truth_col]) if truth_col in actual.index else np.nan
                    if not np.isfinite(y) or len(samples) < 50:
                        continue
                    lo80, hi80 = np.percentile(samples, [10, 90])
                    lo50, hi50 = np.percentile(samples, [25, 75])
                    med = float(np.median(samples))
                    records.append({
                        "cutoff": cutoff, "player_id": pid, "horizon": h + 1,
                        "stat": stat, "actual": y, "median": med,
                        "crps": cal.crps_ensemble(samples, y),
                        "pit": float(cal.pit(samples[None, :], np.array([y]))[0]),
                        "inside_80": bool(lo80 <= y <= hi80),
                        "inside_50": bool(lo50 <= y <= hi50),
                        "width_80": float(hi80 - lo80),
                        "abs_err": abs(med - y), "sq_err": (med - y) ** 2,
                    })
            # Feed the simulated season back in as next step's input.
            draws = _advance(draws, sim, x_cols, pool_cols)
            if not alive.any():
                break
    return pd.DataFrame(records)


_STAT_SOURCE = {
    "pts_per100": "pts_r", "ast_per100": "ast_r", "stl_per100": "stl_r",
    "blk_per100": "blk_r", "tov_per100": "tov_r", "fga_3p_per100": "fga_3p_r",
    "possessions": "possessions_next",
}


def _map_stat(stat: str) -> str | None:
    return _STAT_SOURCE.get(stat)


def _advance(draws: np.ndarray, sim: dict, x_cols: list[str],
             pool_cols: list[str]) -> np.ndarray:
    """Recursive step: the simulated season becomes next season's lag features."""
    out = draws.copy()
    pos = {c: i for i, c in enumerate(x_cols)}
    if "age" in pos:
        out[:, pos["age"]] += 1.0
    if "exp" in pos:
        out[:, pos["exp"]] += 1.0
    for c in pool_cols:
        base = c.replace("_next", "")
        for suffix, weight in (("_lag1", 1.0), ("_marcel", 5 / 12.0)):
            col = f"{base}{suffix}"
            if col in pos:
                out[:, pos[col]] = (weight * sim[c]
                                    + (1 - weight) * draws[:, pos[col]])
    if "possessions" in pos and "possessions_next" in sim:
        out[:, pos["possessions"]] = sim["possessions_next"]
    return out


def compare(ssm_scores: pd.DataFrame, gbm_scores: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side on the metrics that decide it."""
    a = cal.score_frame(ssm_scores).assign(model="state-space")
    b = cal.score_frame(gbm_scores).assign(model="lightgbm")
    both = pd.concat([a, b], ignore_index=True)
    return both.pivot_table(index=["stat", "horizon"], columns="model",
                            values=["crps", "rmse", "cover_80", "mean_width_80"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", default="2015,2018")
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    frames = [fit_and_project(int(c), n_draws=args.draws)
              for c in args.cutoffs.split(",")]
    out = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_DIR / "lgbm_scores.parquet", index=False)
    print(cal.score_frame(out).to_string(index=False))


if __name__ == "__main__":
    main()
