"""Part-A: separate individual aging from league-era change.

The within-player aging curve (`model/aging.py`) already blocks cross-sectional
survivor bias -- it never compares an age-34 population mean to an age-27 one.
But age and calendar season still move together *within* a career: an aging
center in 2012->2024 shoots more threes every year, and a within-player curve
with no era term will read the league-wide three-point revolution as an aging
effect.

This module builds a **fold-local era component** (prompt option A):

    log_rate_{i,t} = era_{pos(i), season(t)} + resid_{i,t}

`era` is the league x position x season baseline; the player state is modelled
on the residual.  Every era quantity is refit inside a cutoff fold from seasons
<= cutoff ONLY, so no future league baseline leaks into an earlier fold.

Three things are checked here, none of which touch the shipping fit (this is a
diagnostic / candidate -- see the Session-4 report for why it does not ship):

  1. an explicit **leakage test**: appending an extreme fabricated future season
     must not move any baseline at season <= cutoff;
  2. **identifiability**: era baselines are anchored (league-season mean removed
     into a global level) so the residual aging curve keeps population-mean-zero
     increments, exactly the constraint `model/aging.py` relies on;
  3. the **center 3PA question**: does the apparent within-player increase in
     centre 3PA with age survive era control?

Run:  python -m career_model.validate.era_separation
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, TRAIN_CUTOFF_YEAR

RATE_COLS = ["fga_2p", "fga_3p", "fta", "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]
POSS_FLOOR = 100.0        # minimum possessions to enter era estimation


def _rates(panel: pd.DataFrame) -> pd.DataFrame:
    """log((count + 0.5) / possessions * 100) per row, one column per stat."""
    df = panel[panel["possessions"] >= POSS_FLOOR].copy()
    for c in RATE_COLS:
        df[f"lr_{c}"] = np.log((df[c] + 0.5) / df["possessions"] * 100.0)
    return df


def era_baselines(panel: pd.DataFrame, cutoff: int, shrink: float = 20.0) -> pd.DataFrame:
    """League x position x season mean log-rate, using seasons <= cutoff ONLY.

    Shrinkage: each (position, season) cell mean is pulled toward the
    position's all-season mean with weight `shrink` (pseudo-observations), so
    thin early-90s cells and short lockout seasons do not swing the baseline.
    Smoothness across seasons then falls out of the shared position anchor.
    """
    df = _rates(panel[panel["season_year"] <= cutoff])
    rows = []
    for c in RATE_COLS:
        col = f"lr_{c}"
        pos_mean = df.groupby("position_group")[col].mean()
        cell = df.groupby(["position_group", "season_year"])[col].agg(["mean", "count"])
        for (pos, season), r in cell.iterrows():
            m, k = r["mean"], r["count"]
            shrunk = (k * m + shrink * pos_mean[pos]) / (k + shrink)
            rows.append({"stat": c, "position_group": pos,
                         "season_year": int(season), "era": float(shrunk)})
    return pd.DataFrame(rows)


def residualize(panel: pd.DataFrame, base: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Attach era baseline and residual log-rate to each within-cutoff row."""
    df = _rates(panel[panel["season_year"] <= cutoff])
    long = df.melt(id_vars=["player_id", "season_year", "position_group", "age"],
                   value_vars=[f"lr_{c}" for c in RATE_COLS],
                   var_name="stat", value_name="lr")
    long["stat"] = long["stat"].str.slice(3)
    m = long.merge(base, on=["stat", "position_group", "season_year"], how="left")
    m["resid"] = m["lr"] - m["era"]
    return m


# ----------------------------------------------------------------------------
# 1. Leakage test
# ----------------------------------------------------------------------------
def leakage_test(panel: pd.DataFrame, cutoff: int) -> dict:
    """Appending a fabricated extreme future season must not move any baseline
    at season <= cutoff."""
    base = era_baselines(panel, cutoff)
    fake = panel[panel["season_year"] == panel["season_year"].max()].copy()
    fake["season_year"] = cutoff + 5
    for c in RATE_COLS:            # make it wildly out of distribution
        fake[c] = fake[c] * 10.0
    poisoned = pd.concat([panel, fake], ignore_index=True)
    # a naive (leaky) estimator would use all seasons; ours filters <= cutoff.
    base2 = era_baselines(poisoned, cutoff)
    merged = base.merge(base2, on=["stat", "position_group", "season_year"],
                        suffixes=("_clean", "_poisoned"))
    max_drift = float((merged["era_clean"] - merged["era_poisoned"]).abs().max())
    return {"cutoff": cutoff, "max_baseline_drift": max_drift,
            "leak_free": bool(max_drift < 1e-12)}


# ----------------------------------------------------------------------------
# 2. Identifiability check
# ----------------------------------------------------------------------------
def identifiability(res: pd.DataFrame) -> dict:
    """Within-player residual increments must stay ~mean-zero over the age
    distribution, the constraint the aging spline relies on."""
    res = res.sort_values(["player_id", "stat", "season_year"])
    res["dresid"] = res.groupby(["player_id", "stat"])["resid"].diff()
    out = {}
    for c in RATE_COLS:
        d = res[(res["stat"] == c) & res["dresid"].notna()]["dresid"]
        out[c] = {"mean_increment": float(d.mean()), "n": int(len(d))}
    return out


# ----------------------------------------------------------------------------
# 3. Centre 3PA: does the age effect survive era control?
# ----------------------------------------------------------------------------
def _within_player_age_slope(df: pd.DataFrame, value: str) -> dict:
    """OLS slope of `value` on age, within-player (player-demeaned), so it is a
    pure within-career trend -- no survivor bias from the cross-section."""
    d = df.dropna(subset=[value, "age"]).copy()
    d["age_c"] = d["age"] - d.groupby("player_id")["age"].transform("mean")
    d["y_c"] = d[value] - d.groupby("player_id")[value].transform("mean")
    x, y = d["age_c"].to_numpy(), d["y_c"].to_numpy()
    if len(x) < 30 or np.var(x) == 0:
        return {"slope": float("nan"), "n": int(len(x))}
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    # bootstrap SE over players
    rng = np.random.default_rng(0)
    pids = d["player_id"].unique()
    boots = []
    for _ in range(200):
        samp = rng.choice(pids, len(pids), replace=True)
        dd = d[d["player_id"].isin(samp)]
        xx, yy = dd["age_c"].to_numpy(), dd["y_c"].to_numpy()
        if np.var(xx) > 0:
            boots.append(np.cov(xx, yy, bias=True)[0, 1] / np.var(xx))
    return {"slope": slope, "se": float(np.std(boots)), "n": int(len(x))}


def centre_3pa(panel: pd.DataFrame, res: pd.DataFrame, cutoff: int) -> dict:
    """Uncontrolled vs era-controlled within-player age slope of centre 3PA/100."""
    raw = _rates(panel[(panel["season_year"] <= cutoff)
                       & (panel["position_group"] == "C")])
    uncontrolled = _within_player_age_slope(raw.rename(columns={"lr_fga_3p": "y"}), "y")
    ctr = res[(res["stat"] == "fga_3p") & (res["position_group"] == "C")]
    controlled = _within_player_age_slope(ctr.rename(columns={"resid": "y"}), "y")
    return {"uncontrolled_age_slope": uncontrolled,
            "era_controlled_age_slope": controlled,
            "interpretation": (
                "slope is per year of age in log(3PA/100). If the controlled "
                "slope is materially smaller / crosses zero, the apparent "
                "'centres shoot more threes as they age' effect was the league "
                "three-point era, not aging.")}


def main() -> None:
    from ..config import CACHE_DIR
    panel = pd.read_parquet(CACHE_DIR / "panel.parquet")
    report: dict = {"train_cutoff": TRAIN_CUTOFF_YEAR}

    # leakage over several folds
    report["leakage"] = [leakage_test(panel, y) for y in (2014, 2018, 2022, 2026)]

    base = era_baselines(panel, TRAIN_CUTOFF_YEAR)
    res = residualize(panel, base, TRAIN_CUTOFF_YEAR)
    report["identifiability_increment_means"] = identifiability(res)
    report["centre_3pa"] = centre_3pa(panel, res, TRAIN_CUTOFF_YEAR)

    # era baseline movement for centre 3PA across seasons (the era curve itself)
    c3 = base[(base["stat"] == "fga_3p") & (base["position_group"] == "C")]
    c3 = c3.sort_values("season_year")
    report["centre_3pa_era_curve"] = {
        "season_year": c3["season_year"].tolist(),
        "log_3pa_per100": [round(v, 3) for v in c3["era"].tolist()],
    }

    out = OUTPUT_DIR / "era_separation.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({"leakage": report["leakage"],
                      "centre_3pa": report["centre_3pa"]}, indent=2, default=float))
    print(f"[era] wrote {out}")


if __name__ == "__main__":
    main()
