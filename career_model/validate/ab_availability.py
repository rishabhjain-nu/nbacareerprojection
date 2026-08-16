"""Clean A/B for the per-player availability EB fix (durable stars).

Same protocol as the A/B that validated the EB reversion target: one full
fold fit through the cutoff, then every cutoff-active player simulated twice
from the same fitted model and the same filtered states -- `use_avail_eb`
toggled -- and scored against what actually happened.  The comparison is
paired, so fit quality cancels.

Two views of the result:

  * the standard per-stat CRPS / 80% coverage table, because a fix that helps
    possessions by breaking points is not a fix;
  * h=1 possession bias bucketed by *last observed* possessions -- a
    pre-cutoff quantity, so the bucketing is honest -- because the failure
    this fix targets lives in the top bucket (durable players projected down
    ~2-5%) and the bottom one (fringe players not expected to bounce back,
    -30%), and a pooled number would average the fix's effect away.

Run:  .venv/bin/python -m career_model.validate.ab_availability --cutoff 2019
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, PANEL_PATH
from ..pipeline import filtered_states, fit_everything
from ..simulate import derive, project
from . import calibration as cal
from .backtest import SCORED, actual_values

BUCKETS = [(4500, np.inf), (3000, 4500), (1500, 3000), (0, 1500)]

# Named comparisons.  Each arm is (label, kwargs passed to project.simulate).
# The fold fit is shared, so an experiment is a pair of cheap re-simulations.
COMPARISONS = {
    # the shipped per-player availability EB, against nothing
    "baseline": [("off", {"use_avail_eb": False}),
                 ("EB", {"use_avail_eb": True})],
    # state-dependent aging: EB alone vs EB + the personal drift bend
    "slope": [("EB", {"use_avail_eb": True}),
              ("EB+slope", {"use_avail_eb": True, "use_avail_slope": True})],
    # fix 6: additive hazard vs age x quality interaction.  The arms differ in
    # the *hazard object*, not simulate kwargs; built inside run() because the
    # interaction refit needs the fold's filtered states.
    "hazard": None,
    # deviation #4: career-continuation as appearance (old) vs the explicit
    # within-career absence model (new).  Built in run() because the absence
    # model is fitted from the fold's filtered states.
    "absence": None,
    # star minute-drop: shipped config vs + age x quality availability aging.
    # The adjustment object is fitted from the fold's filtered states in run().
    "qualaging": None,
}

SURV_AGE_BUCKETS = [(0, 28.5), (28.5, 31.5), (31.5, 34.5), (34.5, 99)]


def _log(msg: str) -> None:
    print(f"[ab-avail] {msg}", flush=True)


def _fold(cutoff: int, fit_kwargs: dict | None):
    """Fit-through-cutoff, cached.  The fold fit is ~10 minutes and identical
    across arms and experiments, so it is pickled once per cutoff; delete
    `outputs/folds/fold_<cutoff>.pkl` after any change to `model/` or the
    panel, since a stale fold silently invalidates every comparison run on it.
    """
    cache = OUTPUT_DIR / "folds" / f"fold_{cutoff}.pkl"
    if fit_kwargs is None and cache.exists():
        with open(cache, "rb") as f:
            model, ds = pickle.load(f)
        _log(f"cutoff {cutoff}: loaded cached fold fit")
        return model, ds
    model, ds = fit_everything(max_season_year=cutoff, verbose=False,
                               **(fit_kwargs or {}))
    if fit_kwargs is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache, "wb") as f:
                pickle.dump((model, ds), f)
        except Exception as e:      # a fold that cannot pickle is still usable
            _log(f"fold cache write failed ({e}); continuing uncached")
    return model, ds


def run(cutoff: int = 2019, max_h: int = 5, n_draws: int = 800,
        n_players: int | None = None, seed: int = 0,
        fit_kwargs: dict | None = None, compare: str = "baseline") -> None:
    t0 = time.time()
    panel = pd.read_parquet(PANEL_PATH)
    model, ds = _fold(cutoff, fit_kwargs)
    assert int(ds.panel["season_year"].max()) <= cutoff
    filt = filtered_states(ds, model.fit)
    assert filt.conditioning == "filtered"
    translator = derive.fit_bpm_translator(ds.panel, verbose=False)
    injury_beta = project.fit_injury_rate(ds.panel)
    _log(f"cutoff {cutoff}: model ready in {time.time() - t0:.0f}s")

    # Arms: (label, simulate kwargs, model override or None).
    if compare == "hazard":
        from ..model import hazard as hazard_mod
        import copy
        h_int = hazard_mod.fit(ds, filt.x1, verbose=True, interaction=True)
        m_base, m_int = copy.copy(model), copy.copy(model)
        m_base.hazard, m_int.hazard = model.hazard, h_int
        arms = [("base", {}, m_base), ("age*q", {}, m_int)]
    elif compare == "absence":
        from ..model import hazard as hazard_mod
        import copy
        # The cached fold predates the absence model; fit it here from the
        # fold's own filtered states (deterministic, cheap) so both arms share
        # every other parameter.  "career" scores career-continuation as
        # appearance (the old behaviour); "appear" scores the composed
        # appearance probability.
        ab = hazard_mod.fit_absence(ds, filt.x1, verbose=True)
        m_off, m_on = copy.copy(model), copy.copy(model)
        m_off.absence, m_on.absence = None, ab
        arms = [("career", {}, m_off), ("appear", {}, m_on)]
    elif compare == "qualaging":
        aq = project.fit_avail_quality_aging(ds, filt, model.hazard)
        _log(f"quality-aging onset c_step={aq.c_step:+.3f} c_slope={aq.c_slope:+.3f}")
        arms = [("base", {}, None), ("qual", {"avail_quality": aq}, None)]
    else:
        arms = [(nm, kw, None) for nm, kw in COMPARISONS[compare]]

    truth = actual_values(panel)
    truth = truth[truth["season_year"].between(cutoff + 1, cutoff + max_h)]
    truth_idx = {(int(r.player_id), int(r.season_year)): r for r in truth.itertuples()}

    grid = ds.grid
    rows = [i for i in range(grid.n_players)
            if grid.last_index[i] >= 0
            and int(grid.season_years[i, grid.last_index[i]]) == cutoff]
    if n_players:
        rows = list(np.random.default_rng(seed).permutation(rows)[:n_players])
    _log(f"cutoff {cutoff}: {len(rows)} players active in the cutoff season")

    # "Appeared that season" -- what survival is scored against.  Note the
    # definitional gap (README deviation #4): the hazard predicts *career
    # continuation*, so both arms over-predict against this target; the paired
    # deltas are what carry information.
    appeared = set(map(tuple, panel.loc[panel["possessions"] > 0,
                                        ["player_id", "season_year"]]
                       .astype(int).itertuples(index=False, name=None)))

    records, surv_records = [], []
    for arm, arm_kwargs, model_override in arms:
        sim_model = model if model_override is None else model_override
        for i in rows:
            proj = project.simulate(sim_model, ds, i, filt, n_draws=n_draws,
                                    horizon=max_h, seed=seed,
                                    injury_beta=injury_beta, **arm_kwargs)
            pid = int(grid.player_ids[i])
            last_poss = float(grid.exposure[i, int(grid.last_index[i])])
            for h in range(max_h):
                key = (pid, cutoff + h + 1)
                # Appearance is scored against "played the season"; condition
                # the box score on the same event so a missed-season zero does
                # not pollute the "if he plays" distribution.
                live = proj.played[:, h]
                surv_records.append({
                    "arm": arm, "player_id": pid, "horizon": h + 1,
                    "age": float(proj.ages[h]), "p": float(live.mean()),
                    "actual": key in appeared, "last_poss": last_poss,
                })
                if key not in truth_idx or live.sum() < 50:
                    continue
                actual = truth_idx[key]
                cols = {c: proj.box[c][live, h] for c in project.COUNT_NAMES}
                comp = derive.derive_composites(
                    cols, np.maximum(proj.possessions[live, h], 1.0), translator)
                for stat in SCORED:
                    if stat not in comp:
                        continue
                    samples = np.asarray(comp[stat], float)
                    samples = samples[np.isfinite(samples)]
                    y = float(getattr(actual, stat))
                    if not np.isfinite(y) or len(samples) < 50:
                        continue
                    lo, hi = np.percentile(samples, [10, 90])
                    records.append({
                        "arm": arm, "player_id": pid, "horizon": h + 1,
                        "stat": stat, "actual": y, "last_poss": last_poss,
                        "median": float(np.median(samples)),
                        "crps": cal.crps_ensemble(samples, y),
                        "inside_80": bool(lo <= y <= hi),
                    })
        _log(f"arm {arm}: scored ({time.time() - t0:.0f}s total)")

    df = pd.DataFrame(records)
    a0, a1 = arms[0][0], arms[1][0]
    print(f"\n=== CLEAN A/B ({compare}), cutoff {cutoff}, "
          f"same fitted model, {len(rows)} players ===")
    for stat in SCORED:
        d = df[df["stat"] == stat]
        if not len(d):
            continue
        o, n = d[d["arm"] == a0], d[d["arm"] == a1]
        print(f"{stat:<15} cover80: {a0} {o['inside_80'].mean():.3f}  "
              f"{a1} {n['inside_80'].mean():.3f}  |  "
              f"CRPS: {a0} {o['crps'].mean():.1f}  {a1} {n['crps'].mean():.1f}")
    o, n = df[df["arm"] == a0], df[df["arm"] == a1]
    # Normalised CRPS: each stat scaled by the first arm's mean so units cancel.
    scale = o.groupby("stat")["crps"].mean()
    print(f"{'ALL STATS':<15} cover80: {a0} {o['inside_80'].mean():.3f}  "
          f"{a1} {n['inside_80'].mean():.3f}  |  CRPS(norm): {a0} 1.000 "
          f"{a1} {(n['crps'] / n['stat'].map(scale)).mean() / (o['crps'] / o['stat'].map(scale)).mean():.3f}")

    print("\n=== h=1 possession bias by LAST OBSERVED possessions "
          "(pre-cutoff bucketing; the durable-star diagnostic) ===")
    if cutoff + 1 in (1999, 2012, 2020, 2021):
        print(f"  !! {cutoff + 1} was a shortened season: actuals are "
              f"mechanically deflated for every player, so the bias column "
              f"reads high in both arms.  Use a full-schedule cutoff for "
              f"this table; the paired per-stat comparison above is fine.")
    d = df[(df["stat"] == "possessions") & (df["horizon"] == 1)]
    for lo_b, hi_b in BUCKETS:
        b = d[(d["last_poss"] >= lo_b) & (d["last_poss"] < hi_b)]
        if not len(b):
            continue
        line = f"  last {lo_b:>5.0f}-{'inf' if np.isinf(hi_b) else f'{hi_b:.0f}':>5}: "
        for arm in (a0, a1):
            a = b[b["arm"] == arm]
            bias = (a["median"].mean() - a["actual"].mean()) / a["actual"].mean()
            line += (f"{arm} med {a['median'].mean():6.0f} vs act "
                     f"{a['actual'].mean():6.0f} (bias {100 * bias:+5.1f}%)   ")
        print(line + f"n={len(b) // 2}")

    # ---- durable-star possession trajectory: bias + CRPS by horizon -------
    # The star minute-drop is a multi-year problem, so pool the top two last-
    # possession buckets (> 3,000, i.e. genuine starters) and read the bias and
    # CRPS out to h=5.  A fix for it should shrink a *negative* (under-
    # projected) bias at long horizons without inflating CRPS.
    print("\n=== possessions for last-season > 3,000 (durable starters), by horizon ===")
    star = df[(df["stat"] == "possessions") & (df["last_poss"] >= 3000)]
    for h in range(1, max_h + 1):
        b = star[star["horizon"] == h]
        if len(b) < 4:
            continue
        line = f"  h={h}: "
        for arm in (a0, a1):
            a = b[b["arm"] == arm]
            bias = (a["median"].mean() - a["actual"].mean()) / a["actual"].mean()
            line += (f"{arm} bias {100 * bias:+5.1f}% CRPS {a['crps'].mean():5.0f}   ")
        print(line + f"n={len(b) // 2}")

    # ---- survival: p_active vs "appeared that season", pooled h=1..5 -------
    sv = pd.DataFrame(surv_records)
    sv["actual"] = sv["actual"].astype(float)
    print("\n=== survival, pooled h=1..5: p_active vs appeared "
          "(deviation #4 inflates both arms; read the deltas) ===")
    line = "  Brier: "
    for arm in (a0, a1):
        a = sv[sv["arm"] == arm]
        line += f"{arm} {((a['p'] - a['actual']) ** 2).mean():.4f}   "
    print(line)
    print("  by projected-season age:")
    for lo_a, hi_a in SURV_AGE_BUCKETS:
        b = sv[(sv["age"] >= lo_a) & (sv["age"] < hi_a)]
        if len(b) < 20:
            continue
        line = f"    age {lo_a:>4.1f}-{hi_a:>4.1f}: "
        for arm in (a0, a1):
            a = b[b["arm"] == arm]
            line += (f"{arm} pred {a['p'].mean():.3f} act {a['actual'].mean():.3f} "
                     f"(gap {a['p'].mean() - a['actual'].mean():+.3f})   ")
        print(line + f"n={len(b) // 2}")
    print("  by last observed possessions (the star-vs-fringe axis):")
    for lo_b, hi_b in BUCKETS:
        b = sv[(sv["last_poss"] >= lo_b) & (sv["last_poss"] < hi_b)]
        if len(b) < 20:
            continue
        line = f"    last {lo_b:>5.0f}-{'inf' if np.isinf(hi_b) else f'{hi_b:.0f}':>5}: "
        for arm in (a0, a1):
            a = b[b["arm"] == arm]
            line += (f"{arm} pred {a['p'].mean():.3f} act {a['actual'].mean():.3f} "
                     f"(gap {a['p'].mean() - a['actual'].mean():+.3f})   ")
        print(line + f"n={len(b) // 2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2019)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--players", type=int, default=None)
    ap.add_argument("--compare", choices=sorted(COMPARISONS), default="baseline")
    ap.add_argument("--quick", action="store_true",
                    help="fewer optimiser iterations in the fold fit (uncached)")
    args = ap.parse_args()
    fit_kwargs = ({"n_outer": 2, "maxiter_diag": 120, "maxiter_full": 25}
                  if args.quick else None)
    run(cutoff=args.cutoff, n_draws=args.draws, n_players=args.players,
        fit_kwargs=fit_kwargs, compare=args.compare)


if __name__ == "__main__":
    main()
