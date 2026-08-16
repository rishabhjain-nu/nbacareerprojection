"""The stage checks of §5.2 and §10, run against a fitted model.

`python -m career_model.diagnostics` writes plots to `outputs/` and prints the
tables.  These are gates, not decoration: §10 says do not proceed past M1 until
the filtered trajectories look right by eye, and this is the thing you look at.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import IDX, OUTPUT_DIR, POSITION_GROUPS, STATE_NAMES
from .model import hazard as hazard_mod
from .model import observations as obs
from .model import state_space as ss
from .model.dataset import load as load_dataset
from .pipeline import filtered_states, load as load_model

# §10.3: ten careers whose shapes are known in advance, so the filter can be
# judged against something other than its own output.
CHECK_PLAYERS = [
    ("LeBron James", "early bloomer"),
    ("Kevin Durant", "early bloomer"),
    ("Jimmy Butler", "late bloomer"),
    ("Pascal Siakam", "late bloomer"),
    ("Derrick Rose", "injury-disrupted"),
    ("Kawhi Leonard", "injury-disrupted"),
    ("Chris Paul", "steady veteran"),
    ("Mike Conley", "steady veteran"),
    ("Anthony Bennett", "flameout"),
    ("Michael Beasley", "flameout"),
]


def _rows_for(ds, names):
    """Match on the folded name, so "Jimmy Butler" finds "Jimmy Butler III"."""
    from .data.ingest_college import normalize_name
    lookup = {}
    for i, n in enumerate(ds.priors["player_name"]):
        lookup.setdefault(normalize_name(str(n)), i)
    out = []
    for name, shape in names:
        i = lookup.get(normalize_name(name))
        if i is not None and ds.grid.n_history[i] > 0:
            out.append((i, name, shape))
    return out


def plot_m1(ds, filt, stat: str = "dreb", path=None):
    """Filtered `theta` against the raw observed rate, ten known careers (§10.3).

    What to look for: the line tracks genuine movement, ignores small-sample
    spikes, and the shaded band visibly inflates across seasons the player
    missed.  If a 300-possession season yanks the line as hard as a 3000-
    possession one, `R` is wrong and nothing downstream is worth running.
    """
    rows = _rows_for(ds, CHECK_PLAYERS)
    j = IDX[stat]
    fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex=False)
    for ax, (i, name, shape) in zip(axes.ravel(), rows):
        span = np.flatnonzero(ds.grid.in_span[i])
        ages = ds.grid.age[i, span]
        theta = filt.x1[i, span, j]
        sd = np.sqrt(filt.P11[i, span, j, j])
        seen = ds.grid.observed[i, span]

        ax.fill_between(ages, theta - 1.96 * sd, theta + 1.96 * sd,
                        color="#4f9cf5", alpha=0.18, lw=0, label="95% on theta")
        ax.plot(ages, theta, color="#1f6fd1", lw=2, label="filtered theta")
        if seen.any():
            raw = ds.grid.z[i, span, j]
            poss = ds.grid.exposure[i, span]
            sizes = 12 + 60 * np.nan_to_num(poss) / 4000.0
            ax.scatter(ages[seen], raw[seen], s=sizes[seen], color="#e2554f",
                       zorder=3, label="observed log rate (size = possessions)")
        for a in ages[~seen]:
            ax.axvspan(a - 0.5, a + 0.5, color="#8b98a8", alpha=0.13, lw=0)
        ax.axhline(filt.x2[i, ds.grid.z.shape[1], j], color="#4ec9a8", ls="--",
                   lw=1.2, label="E[m_i]")
        ax.set_title(f"{name} — {shape}", fontsize=11)
        ax.set_xlabel("age")
        ax.set_ylabel(f"log({stat} / poss)")
        ax.grid(alpha=0.15)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"M1 gate — filtered state vs raw observation, {stat.upper()}; "
                 "grey bands are seasons with no data", y=0.995)
    fig.tight_layout()
    path = path or OUTPUT_DIR / f"m1_{stat}_trajectories.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def m2_shrinkage(ds, model) -> pd.DataFrame:
    """§5.2 M2: how hard is a player's *first* season pulled to the prior?

    M2 is the stage that adds the hierarchical prior on the initial state, so
    the check belongs at `t = 0`, where `P` is the full cross-player spread
    `Sigma_player + D` and `B = R/(P+R)` answers "how much of this rookie's line
    do we believe?".  The spec's targets are ~0.3 for a 200-minute rookie and
    under 0.05 for a 2400-minute season.

    The later-career column is a different quantity and is reported alongside so
    the two are not confused: by season five `P` is the filter's own converged
    one-step variance, not the population spread, so `B` is naturally larger --
    the model has a strong opinion by then and one season does not overturn it.
    """
    p = model.fit.params
    R = obs.refresh_R(ds.grid, p.phi, p.acc_floor, p.sigma_poss,
                              p.sigma_poss_inj, p.injury_infl)
    res = ss.run_filter(ds.grid, p, ds.X, R,
                        model.offsets_for(ds.grid, ds.pos_idx),
                        ds.init_basis, model.fit.c_init, keep_states=True)

    grid = ds.grid
    minutes = grid.exposure * (1.0 / 2.02)
    P0 = np.diag(p.Sigma_p) + np.diag(p.stationary_dispersion())
    big = 1e12
    rows = []
    for j, name in enumerate(STATE_NAMES):
        ok0 = grid.obs_mask[:, 0, j]
        r0 = np.where(ok0, R[:, 0, j], big)
        B0 = ss.shrinkage_factor(P0[j], r0)
        m0 = minutes[:, 0]

        K = res.gain[:, :, j, j]
        P_post = res.P11[:, :-1, j, j]
        P_prior = P_post / np.maximum(1 - K, 1e-12)
        Blate = ss.shrinkage_factor(P_prior, np.where(grid.obs_mask[:, :, j],
                                                      R[:, :, j], big))
        late = grid.obs_mask[:, :, j] & (np.arange(grid.z.shape[1])[None, :] >= 4)

        thin0 = ok0 & (m0 < 400)
        thick0 = ok0 & (m0 > 2400)
        rows.append({
            "stat": name,
            "B_debut_<400min": float(np.mean(B0[thin0])) if thin0.any() else np.nan,
            "B_debut_>2400min": float(np.mean(B0[thick0])) if thick0.any() else np.nan,
            "B_season5+": float(np.mean(Blate[late])) if late.any() else np.nan,
        })
    return pd.DataFrame(rows)


def plot_aging(model, path=None):
    """§5.2 M4: league and position level paths, one season per step."""
    ages = np.arange(19.0, 41.0)
    basis, A = model.age_basis, model.fit.params.A
    curves = {"league": basis.level_path(model.fit.delta_league, A, ages)}
    for g, name in enumerate(POSITION_GROUPS):
        curves[name] = basis.level_path(
            model.fit.delta_league + model.fit.delta_pos[g], A, ages)

    fig, axes = plt.subplots(4, 4, figsize=(15, 12))
    colors = {"league": "#131720", "G": "#1f6fd1", "F": "#17805f", "C": "#c2571f"}
    for j, name in enumerate(STATE_NAMES):
        ax = axes.ravel()[j]
        for k, c in curves.items():
            ax.plot(ages, c[:, j], color=colors[k], lw=2.2 if k == "league" else 1.4,
                    ls="-" if k == "league" else "--", label=k)
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.15)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(STATE_NAMES):]:
        ax.axis("off")
    fig.suptitle("M4 gate — aging level paths relative to m_i, league and position\n"
                 "(within-player identification only; units are log rate / logit)",
                 y=0.995)
    fig.tight_layout()
    path = path or OUTPUT_DIR / "m4_aging_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def m4_peak_ages(model) -> pd.DataFrame:
    """Where each level path tops out, per position.  The M4 check is that bigs
    decline on athleticism markers earlier than guards."""
    ages = np.arange(19.0, 41.0)
    rows = []
    for g, name in enumerate(POSITION_GROUPS):
        c = model.age_basis.level_path(
            model.fit.delta_league + model.fit.delta_pos[g], model.fit.params.A, ages)
        for j, stat in enumerate(STATE_NAMES):
            rows.append({"position": name, "stat": stat,
                         "peak_age": float(ages[np.argmax(c[:, j])])})
    return pd.DataFrame(rows).pivot(index="stat", columns="position", values="peak_age")


def m5_survival(ds, model, filt) -> None:
    by_p, by_age = hazard_mod.calibration_table(model.hazard, ds, filt.x1)
    print("\n=== M5: survival calibration by predicted bucket ===")
    print(by_p.to_string())
    print("\n=== M5: survival calibration by age ===")
    print(by_age.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", default="dreb")
    args = ap.parse_args()

    model = load_model()
    ds = load_dataset()
    filt = filtered_states(ds, model.fit)

    p1 = plot_m1(ds, filt, args.stat)
    p2 = plot_m1(ds, filt, "ast", OUTPUT_DIR / "m1_ast_trajectories.png")
    print(f"[diag] wrote {p1}\n[diag] wrote {p2}")

    print("\n=== M2: shrinkage factor B = R/(P+R) ===")
    print(m2_shrinkage(ds, model).to_string(index=False))

    p3 = plot_aging(model)
    print(f"\n[diag] wrote {p3}")
    print("\n=== M4: peak age of each cumulative curve, by position ===")
    print(m4_peak_ages(model).round(1).to_string())

    m5_survival(ds, model, filt)


if __name__ == "__main__":
    main()
