"""Expanding-window backtest (§7.3).

Fit through season `T`, project `T+1 .. T+5`, score.  `T` walks forward across
2010..2019 and the training window grows; it never slides, and folds are never
random.

**Why not k-fold.**  A random fold puts a player's age-30 season in training and
asks the model about his age-27 season.  The state at 27 would then be informed
by data from 30 -- the filter would have already seen the answer -- and the
model would look far better than it is.  The leak is not subtle: it is the
difference between forecasting and interpolating.  Nothing in this module ever
splits within a player's timeline.

**The four leakage channels, and where each is closed.**  The filter recursion
is causal on its own, but that is easy to destroy accidentally and the damage
never announces itself -- nothing errors, the score just improves.

1. *Never project from smoothed states.*  There is no RTS smoother in this
   codebase.  Projections launch from `theta_{T|T}` / `P_{T|T}`, and
   `simulate.project.simulate` **rejects** anything not tagged
   `conditioning="filtered"`, so a smoother added later for UI history
   rendering cannot be wired into a forecast by a stray keyword argument.
   `tests/test_no_smoother_leakage.py` asserts the cutoff state is bit-for-bit
   identical whether or not post-cutoff seasons exist in the panel.
2. *All hyperparameters refit per fold.*  `score_cutoff` calls
   `fit_everything(max_season_year=cutoff)`, which is a full hierarchical refit:
   `Q`, `A`, `delta(.)`, `Sigma_player`, `phi_s`, the accuracy floors,
   `sigma_poss`, and `gamma`.  Aging leaks hardest of all -- it is precisely
   what is being projected -- so the aging basis centring and the spline
   coefficients are both recomputed from the truncated frame.
3. *Prior-covariate coefficients refit per fold.*  `beta` is estimated by EM
   over the truncated player set only, i.e. players who debuted at or before
   `T`, and the `x_i` standardisation uses that same set.
4. *GBM prior mean refit per fold.*  `f_GBM(x_i)` is retrained inside
   `fit_everything`, out of fold and grouped by player, on pre-`T` outcomes.

`dataset.load` is the single choke point that makes 2-4 hold: every global
quantity is derived from the frame it returns, so cutting the panel there
confines all of them.  This makes a backtest expensive -- one full hierarchical
refit per cutoff, roughly fifteen minutes each -- and that cost is the point.  A
cheap backtest that reuses a global fit produces a number that is not measuring
what it claims to, and the error does not show up anywhere in the output.

**Sanity check.**  `crps_by_cutoff` reports the headline score per cutoff.  If
early cutoffs score dramatically better than late ones, assume leakage before
assuming drift and audit the four channels above.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, PANEL_PATH, RAW_DIR
from ..pipeline import fit_everything, filtered_states
from ..simulate import derive, project
from . import calibration as cal

# Stats scored.  Per-100 rates rather than counts, because a rate is what a
# reader compares across players -- but note the *model* never sees a rate; the
# rate is computed from the simulated counts and the simulated possessions, so
# both numerator and denominator carry their own uncertainty.
SCORED = ["pts_per100", "reb_per100", "ast_per100", "fga_3p_per100",
          "stl_per100", "blk_per100", "tov_per100", "ts_pct", "possessions",
          "bpm"]

# §7.1: BPM is scored by computing it *from the simulated box scores* and
# comparing to the real Basketball-Reference value.  It is free out-of-sample
# validation on a metric people understand, with zero contamination -- NBA BPM
# is never a model input, only a target here.
#
# One caveat to read the BPM row with.  Real BPM carries a team adjustment
# needing team context this model does not simulate, so the simulated value goes
# through the fitted translator (§7.1 option b), which is itself only 84%
# accurate *given the true box score*.  The BPM CRPS therefore includes
# translator error on top of forecast error and is an upper bound on the model's
# own error.  The coverage and PIT columns are the informative ones: translator
# error is roughly symmetric and unbiased, so it widens the effective target
# without shifting it.


def _log(msg: str) -> None:
    print(f"[backtest] {msg}", flush=True)


def actual_values(panel: pd.DataFrame) -> pd.DataFrame:
    """Observed per-100 rates, possessions and real BPM, keyed by (player, season).

    BPM comes from the Basketball-Reference advanced table -- the *actual*
    published value, not a re-derivation -- so the comparison in §7.1 is against
    the thing itself.
    """
    box = {c: panel[c].to_numpy(dtype=float) for c in derive.RATE_COLS}
    poss = panel["possessions"].to_numpy(dtype=float)
    comp = derive.derive_composites(box, poss)
    out = pd.DataFrame({k: v for k, v in comp.items() if k in SCORED})
    out["player_id"] = panel["player_id"].to_numpy()
    out["season_year"] = panel["season_year"].to_numpy()

    bbr_path = RAW_DIR / "bbr_advanced.parquet"
    if bbr_path.exists():
        bbr = pd.read_parquet(bbr_path)[["player_id", "season", "bpm"]]
        bbr["season_year"] = bbr["season"].str.slice(0, 4).astype(int) + 1
        out = out.merge(bbr[["player_id", "season_year", "bpm"]],
                        on=["player_id", "season_year"], how="left")
        # BPM on a 40-possession call-up is between -90 and +240 in the source
        # data.  Those are not outcomes any forecast should be scored against --
        # they are the metric's own small-sample behaviour, and a handful of them
        # would dominate a mean CRPS.  Scored only where the season is real.
        out.loc[out["possessions"] < 200, "bpm"] = np.nan
    else:
        out["bpm"] = np.nan
    return out


def score_cutoff(cutoff: int, panel: pd.DataFrame, max_h: int = 5,
                 n_draws: int = 800, n_players: int | None = None,
                 fit_kwargs: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Fit through `cutoff`, project forward, score against what happened."""
    t0 = time.time()
    model, ds = fit_everything(max_season_year=cutoff, verbose=False,
                               **(fit_kwargs or {}))
    # Channels 2-4: assert the refit really was confined to the fold before
    # anything is scored against it.
    assert int(ds.panel["season_year"].max()) <= cutoff, "fold saw post-cutoff rows"
    assert int(ds.grid.season_years[ds.grid.observed].max()) <= cutoff
    assert model.train_cutoff == cutoff

    filt = filtered_states(ds, model.fit)
    assert filt.conditioning == "filtered", "projection launch point is not causal"
    # The BPM translator is display-layer, but it is fitted, so it is fold-local
    # too -- `ds.panel` is the truncated frame.
    translator = derive.fit_bpm_translator(ds.panel, verbose=False)
    # Shipped-config projection devices, all fold-local (fitted from this fold's
    # panel/states, never the future) so the backtest measures what ships.
    injury_beta = project.fit_injury_rate(ds.panel)
    avail_quality = project.fit_avail_quality_aging(ds, filt, model.hazard)
    _log(f"cutoff {cutoff}: fitted in {time.time() - t0:.0f}s")

    truth = actual_values(panel)
    truth = truth[truth["season_year"].between(cutoff + 1, cutoff + max_h)]
    truth_idx = {(int(r.player_id), int(r.season_year)): r for r in truth.itertuples()}

    # Only players whose last training season is the cutoff -- projecting from a
    # state that is already two years stale is a different question.
    grid = ds.grid
    rows = [i for i in range(grid.n_players)
            if grid.last_index[i] >= 0
            and int(grid.season_years[i, grid.last_index[i]]) == cutoff]
    if n_players:
        rows = list(np.random.default_rng(seed).permutation(rows)[:n_players])
    _log(f"cutoff {cutoff}: {len(rows)} players active in the cutoff season")

    records, surv_rows = [], []
    for i in rows:
        proj = project.simulate(model, ds, i, n_draws=n_draws, horizon=max_h,
                                filt=filt, seed=seed, injury_beta=injury_beta,
                                avail_quality=avail_quality)
        pid = int(grid.player_ids[i])
        for h in range(max_h):
            year = cutoff + h + 1
            # Survival is scored against appearance ("did he play year T+h"), so
            # the prediction is P(plays) -- career active AND not a mid-career
            # gap (deviation #4).  `played` equals `alive` for a model with no
            # absence sub-model, so this stays correct for older fits too.  Box
            # scores condition on the same event.
            live = proj.played[:, h]
            surv_rows.append({"player_id": pid, "horizon": h + 1,
                              "age": proj.ages[h], "p_active": live.mean(),
                              "actual_active": (pid, year) in truth_idx})
            key = (pid, year)
            if key not in truth_idx or live.sum() < 50:
                continue
            actual = truth_idx[key]
            cols = {c: proj.box[c][live, h] for c in project.COUNT_NAMES}
            comp = derive.derive_composites(cols, np.maximum(proj.possessions[live, h], 1.0),
                                            translator)
            for stat in SCORED:
                if stat not in comp:
                    continue
                samples = np.asarray(comp[stat], float)
                samples = samples[np.isfinite(samples)]
                y = float(getattr(actual, stat))
                if not np.isfinite(y) or len(samples) < 50:
                    continue
                lo80, hi80 = np.percentile(samples, [10, 90])
                lo50, hi50 = np.percentile(samples, [25, 75])
                records.append({
                    "cutoff": cutoff, "player_id": pid, "horizon": h + 1, "stat": stat,
                    "actual": y, "median": float(np.median(samples)),
                    "crps": cal.crps_ensemble(samples, y),
                    "pit": float(cal.pit(samples[None, :], np.array([y]))[0]),
                    "inside_80": bool(lo80 <= y <= hi80),
                    "inside_50": bool(lo50 <= y <= hi50),
                    "width_80": float(hi80 - lo80),
                    "abs_err": abs(float(np.median(samples)) - y),
                    "sq_err": (float(np.median(samples)) - y) ** 2,
                })
    _log(f"cutoff {cutoff}: {len(records)} scored predictions "
         f"({time.time() - t0:.0f}s total)")
    return pd.DataFrame(records), pd.DataFrame(surv_rows)


def run(cutoffs=range(2010, 2020), max_h: int = 5, n_draws: int = 800,
        n_players: int | None = None, fit_kwargs: dict | None = None,
        resume: bool = True):
    """Score every cutoff, checkpointing after each one.

    Each cutoff is a complete hierarchical refit, so the full window is a
    multi-hour job.  Results are written after every cutoff rather than at the
    end: an interruption three hours in should cost the cutoff in flight, not
    all of them.  `resume=True` skips cutoffs already present in the checkpoint,
    so re-running the same command picks up where it stopped.
    """
    panel = pd.read_parquet(PANEL_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT_DIR / "backtest_scores.parquet"
    surv_path = OUTPUT_DIR / "backtest_survival.parquet"

    scored, surv, done = [], [], set()
    if resume and score_path.exists() and surv_path.exists():
        prev_s, prev_v = pd.read_parquet(score_path), pd.read_parquet(surv_path)
        done = set(prev_s["cutoff"].unique()) & set(cutoffs)
        if done:
            scored.append(prev_s[prev_s["cutoff"].isin(done)])
            surv.append(prev_v[prev_v["cutoff"].isin(done)]
                        if "cutoff" in prev_v.columns else prev_v)
            _log(f"resuming; {sorted(done)} already scored")

    for c in cutoffs:
        if c in done:
            continue
        s, v = score_cutoff(c, panel, max_h=max_h, n_draws=n_draws,
                            n_players=n_players, fit_kwargs=fit_kwargs)
        v["cutoff"] = c
        scored.append(s)
        surv.append(v)
        pd.concat(scored, ignore_index=True).to_parquet(score_path, index=False)
        pd.concat(surv, ignore_index=True).to_parquet(surv_path, index=False)
        _log(f"checkpointed through cutoff {c}")

    return (pd.concat(scored, ignore_index=True),
            pd.concat(surv, ignore_index=True))


def crps_by_cutoff(scored: pd.DataFrame) -> pd.DataFrame:
    """Headline score per cutoff -- the leakage sanity check.

    Every cutoff is an independent refit scored on genuinely unseen seasons, so
    the scores should be flat in the cutoff up to sample noise and real league
    drift (pace and three-point rate moved a lot over this window, which shows
    up as a mild trend on volume stats).  A *dramatic* advantage for early
    cutoffs is the signature of contamination, not of an easier era: earlier
    folds have more future data sitting in the panel for something to leak
    from.  If that pattern appears, audit the four channels in the module
    docstring before reaching for a drift explanation.
    """
    out = (scored.groupby(["cutoff", "horizon"], observed=True)
           .agg(n=("crps", "size"), crps=("crps", "mean"),
                cover_80=("inside_80", "mean"))
           .reset_index())
    return out.pivot(index="cutoff", columns="horizon",
                     values=["crps", "cover_80"]).round(3)


def summary(scored: pd.DataFrame, surv: pd.DataFrame | None = None) -> None:
    tbl = cal.score_frame(scored)
    print("\n=== calibration by stat and horizon ===")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tbl.to_string(index=False))

    print("\n=== the horizon check (§7.2): do 80% bands widen and keep covering? ===")
    piv = scored.pivot_table(index="stat", columns="horizon",
                             values=["inside_80", "width_80"], aggfunc="mean")
    print(piv.round(3).to_string())

    if scored["cutoff"].nunique() > 1:
        print("\n=== leakage sanity check: score by cutoff (should be flat) ===")
        print(crps_by_cutoff(scored).to_string())

    print("\n=== PIT shape by horizon (pooled) ===")
    for h in sorted(scored["horizon"].unique()):
        p = scored.loc[scored["horizon"] == h, "pit"].to_numpy()
        print(f"  h={h}: {cal.pit_shape(p)}   (n={len(p)})")

    if surv is not None and len(surv):
        by_p, by_age = cal.survival_calibration(
            surv["p_active"].to_numpy(), surv["actual_active"].to_numpy().astype(float),
            surv["age"].to_numpy())
        print("\n=== survival calibration by predicted bucket ===")
        print(by_p.to_string(index=False))
        print("\n=== survival calibration by age ===")
        print(by_age.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", default="2010-2019")
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--players", type=int, default=None,
                    help="subsample players per cutoff (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="three cutoffs, fewer optimiser iterations")
    ap.add_argument("--no-resume", action="store_true",
                    help="rescore cutoffs already present in the checkpoint")
    args = ap.parse_args()

    lo, hi = (args.cutoffs.split("-") + [args.cutoffs])[:2]
    cutoffs = range(int(lo), int(hi) + 1)
    fit_kwargs = None
    if args.quick:
        cutoffs = [2012, 2015, 2018]
        fit_kwargs = {"n_outer": 2, "maxiter_diag": 120, "maxiter_full": 25}

    scored, surv = run(cutoffs, n_draws=args.draws, n_players=args.players,
                       fit_kwargs=fit_kwargs, resume=not args.no_resume)
    summary(scored, surv)


if __name__ == "__main__":
    main()
