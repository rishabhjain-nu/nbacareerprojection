"""Discrete-time survival, coupled to the filtered latent state (§3.5).

    P(plays season t+1 | played t) = inv_logit(a0 + a1*age + a2*age^2 + gamma' theta_{i,t})

The coupling is to `theta_{i,t}` -- the **filtered** state -- and that is the
entire point of the sub-model.  A fringe big man who posts a terrible line in
200 possessions has, in raw box score terms, an alarming season; the filter has
already read most of that as sampling noise and barely moved his state.
Regressing exit on the raw line would double-count the noise and tell you he is
about to be cut.  Regressing it on the filtered state asks the right question:
given what we now believe about this player, does he stay in the league?

**What "exit" means here.**  The event is *career* termination -- no further NBA
season, ever -- not "absent next season".  Roughly 400 players in the panel miss
a season and come back, and calling those exits would both overstate the hazard
and make the model unable to represent the thing that actually happened to them.
So termination and within-career absence are **two separate events**, fitted by
two models: `fit` below is the career hazard, and `fit_absence` is the
within-career "does he play next season, given his career is still going"
model.  The projection composes them -- P(appears in season t) = P(career still
active) x P(plays | active) -- which is what closes the deviation-#4 gap: the
old code predicted only career continuation, so it over-predicted appearance by
exactly the within-career miss rate, and the backtest (which scores appearance)
duly showed survival over-predicted at every horizon beyond h=1.  A simulated
career now carries an explicit "missed the whole year" state, not just very low
possession draws.

Fitted two-stage in v1 (states first, then hazard on the posterior mean states),
which understates uncertainty in `gamma` because it conditions on the filtered
states as if they were known.  §3.5 permits this and asks that it be flagged;
the parameter draws in §6 cover `gamma`'s own sampling error via the asymptotic
covariance, but not the state uncertainty feeding into it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import S, STATE_NAMES

AGE_CENTRE = 27.0
AGE_SCALE = 5.0


@dataclass
class Hazard:
    coef: np.ndarray            # (2 + 1 + S [+1],) intercept, age, age^2, gamma [, a*q]
    cov: np.ndarray             # asymptotic covariance, for parameter draws
    names: list[str]
    # Age x quality interaction (§ fix-6).  When `inter_gamma` is set, the
    # design carries one extra column `a * q` with
    # `q = (inter_gamma' theta - inter_mu) / inter_sd` -- the hazard's own
    # state score, standardised over the training rows.  A single coefficient
    # then lets quality tilt the age slope: the age penalty a fringe player
    # feels at 33 is not the one an MVP feels, which the additive design
    # forced.  `None` (the default, and what any pre-fix pickle carries)
    # reproduces the original design exactly.
    inter_gamma: np.ndarray | None = None
    inter_mu: float = 0.0
    inter_sd: float = 1.0

    def _quality(self, theta: np.ndarray) -> np.ndarray:
        g = getattr(self, "inter_gamma", None)
        return (np.atleast_2d(theta) @ g - self.inter_mu) / self.inter_sd

    def design(self, age: np.ndarray, theta: np.ndarray) -> np.ndarray:
        a = (np.asarray(age, float) - AGE_CENTRE) / AGE_SCALE
        cols = [np.ones_like(a), a, a ** 2, np.atleast_2d(theta)]
        if getattr(self, "inter_gamma", None) is not None:
            cols.append(a * self._quality(theta))
        return np.column_stack(cols)

    def p_survive(self, age, theta, coef: np.ndarray | None = None) -> np.ndarray:
        c = self.coef if coef is None else coef
        eta = self.design(age, theta) @ c
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

    def draw_coef(self, rng) -> np.ndarray:
        L = np.linalg.cholesky(self.cov + 1e-10 * np.eye(len(self.coef)))
        return self.coef + L @ rng.standard_normal(len(self.coef))


def build_training_rows(ds, filtered_theta: np.ndarray):
    """One row per observed season that could be followed by another.

    The last season in the panel is dropped for everyone: we do not yet know
    whether a 2025-26 player comes back, and scoring him as an exit would put a
    fake cliff at the panel edge and bias every projection's longevity down.
    """
    grid = ds.grid
    max_year = int(grid.season_years[grid.observed].max())
    rows_theta, rows_age, y, pid, year = [], [], [], [], []
    for i in range(grid.n_players):
        obs = np.flatnonzero(grid.observed[i])
        if not len(obs):
            continue
        last = obs[-1]
        for t in obs:
            yr = int(grid.season_years[i, t])
            if yr >= max_year:
                continue          # right-censored: outcome not observed yet
            rows_theta.append(filtered_theta[i, t])
            rows_age.append(grid.age[i, t])
            y.append(1.0 if t < last else 0.0)
            pid.append(int(grid.player_ids[i]))
            year.append(yr)
    return (np.array(rows_age), np.array(rows_theta), np.array(y),
            np.array(pid), np.array(year))


def fit(ds, filtered_theta: np.ndarray, ridge: float = 1e-3,
        verbose: bool = True, interaction: bool = False) -> Hazard:
    """Newton-Raphson logistic regression (IRLS).

    With `interaction=True`, quality bends the age curve: a first pass fits
    the additive model, its state score `gamma' theta` (standardised) becomes
    the quality index `q`, and the model is refit with one extra column
    `a * q`.  The two-stage is iterated twice so the quality basis stabilises
    against its own refit.  The additive design forces every player onto the
    same age slope, which is exactly the failure the interaction targets: the
    exit risk age adds to a fringe 33-year-old is not the one it adds to an
    MVP, and the additive fit splits the difference between them.
    """
    age, theta, y, _, _ = build_training_rows(ds, filtered_theta)

    def _irls(h: Hazard):
        X = h.design(age, theta)
        # Standardize everything past the age block so the ridge means the
        # same thing across columns; undo at the end so `coef` applies to the
        # raw design.
        mu = X[:, 3:].mean(axis=0)
        sd = X[:, 3:].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xs = X.copy()
        Xs[:, 3:] = (X[:, 3:] - mu) / sd

        beta = np.zeros(Xs.shape[1])
        beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
        pen = ridge * np.eye(len(beta))
        pen[0, 0] = 0.0
        for _ in range(50):
            eta = np.clip(Xs @ beta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            w = np.clip(p * (1 - p), 1e-8, None)
            grad = Xs.T @ (y - p) - pen @ beta
            Hess = (Xs * w[:, None]).T @ Xs + pen
            step = np.linalg.solve(Hess, grad)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-9:
                break

        cov_s = np.linalg.inv(Hess)
        # Map back to the raw scale: b_raw = b_std / sd, intercept absorbs it.
        J = np.eye(len(beta))
        J[3:, 3:] = np.diag(1.0 / sd)
        J[0, 3:] = -mu / sd
        return J @ beta, J @ cov_s @ J.T

    names = ["intercept", "age", "age^2", *[f"gamma[{n}]" for n in STATE_NAMES]]
    h = Hazard(coef=np.zeros(3 + S), cov=np.eye(3 + S), names=names)
    h.coef, h.cov = _irls(h)

    if interaction:
        for _ in range(2):
            g = h.coef[3:3 + S].copy()
            q_raw = theta @ g
            h2 = Hazard(coef=np.zeros(3 + S + 1), cov=np.eye(3 + S + 1),
                        names=[*names, "age*q"], inter_gamma=g,
                        inter_mu=float(q_raw.mean()),
                        inter_sd=float(q_raw.std()) or 1.0)
            h2.coef, h2.cov = _irls(h2)
            h = h2

    if verbose:
        p = h.p_survive(age, theta)
        ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) +
                          (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
        base = y.mean()
        ll0 = float(len(y) * (base * np.log(base) + (1 - base) * np.log(1 - base)))
        extra = ""
        if interaction:
            se = float(np.sqrt(h.cov[-1, -1]))
            extra = f", age*q coef {h.coef[-1]:+.3f} (se {se:.3f})"
        print(f"[hazard] {len(y)} transitions, survival rate {base:.3f}, "
              f"pseudo-R2 {1 - ll / ll0:.3f}{extra}")
    return h


def build_absence_rows(ds, filtered_theta: np.ndarray):
    """One row per within-career season transition (deviation #4).

    For every in-span season `t` with `1 <= t <= last_index` -- i.e. the career
    is known to continue through `t`, because the player has a later observed
    season -- the outcome is `1` if he actually played `t` (an observed row) and
    `0` if `t` was a gap.  The predictor is the filtered state at `t-1`, the
    last state the model carried into the season, coupled the same way the
    hazard couples: a fringe player with a thin, low-availability state is the
    one who shuttles to the G-League, not the star.

    The panel stores no row for a missed season, so a gap is an in-span season
    with `observed == False`.  Debut seasons (`t == 0`) are excluded: you cannot
    miss a season before your career starts, and there is no prior state to
    condition on.
    """
    grid = ds.grid
    rows_theta, rows_age, y, pid, year = [], [], [], [], []
    for i in range(grid.n_players):
        last = int(grid.last_index[i])
        if last < 1:
            continue
        for t in range(1, last + 1):
            if not grid.in_span[i, t]:
                continue
            rows_theta.append(filtered_theta[i, t - 1])
            rows_age.append(grid.age[i, t])
            y.append(1.0 if grid.observed[i, t] else 0.0)
            pid.append(int(grid.player_ids[i]))
            year.append(int(grid.season_years[i, t]))
    return (np.array(rows_age), np.array(rows_theta), np.array(y),
            np.array(pid), np.array(year))


def fit_absence(ds, filtered_theta: np.ndarray, ridge: float = 1e-3,
                verbose: bool = True) -> Hazard:
    """P(plays season t | career still active), coupled to the filtered state.

    Same additive logistic form as the career hazard, fitted by the same IRLS,
    but on the within-career present/gap outcome.  Returned as a `Hazard` so the
    projection can score it through the identical `p_survive` path; there is no
    age x quality interaction here (the miss rate is non-monotonic in age but
    modest, and the state coupling carries the star-vs-fringe signal directly).
    """
    age, theta, y, _, _ = build_training_rows_from(
        build_absence_rows(ds, filtered_theta))
    names = ["intercept", "age", "age^2", *[f"gamma[{n}]" for n in STATE_NAMES]]
    h = Hazard(coef=np.zeros(3 + S), cov=np.eye(3 + S), names=names)
    X = h.design(age, theta)

    mu = X[:, 3:].mean(axis=0)
    sd = X[:, 3:].std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = X.copy()
    Xs[:, 3:] = (X[:, 3:] - mu) / sd

    beta = np.zeros(Xs.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    pen = ridge * np.eye(len(beta))
    pen[0, 0] = 0.0
    for _ in range(50):
        eta = np.clip(Xs @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-8, None)
        grad = Xs.T @ (y - p) - pen @ beta
        Hess = (Xs * w[:, None]).T @ Xs + pen
        step = np.linalg.solve(Hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov_s = np.linalg.inv(Hess)
    J = np.eye(len(beta))
    J[3:, 3:] = np.diag(1.0 / sd)
    J[0, 3:] = -mu / sd
    h.coef, h.cov = J @ beta, J @ cov_s @ J.T

    if verbose:
        p = h.p_survive(age, theta)
        ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) +
                          (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
        base = y.mean()
        ll0 = float(len(y) * (base * np.log(base) + (1 - base) * np.log(1 - base)))
        print(f"[absence] {len(y)} within-career transitions, play rate "
              f"{base:.3f}, pseudo-R2 {1 - ll / ll0:.3f}")
    return h


def build_training_rows_from(rows):
    """Adapter: `build_absence_rows` already returns the (age, theta, y, pid,
    year) tuple the fitters expect; this keeps the call site symmetric with the
    hazard's `build_training_rows` and makes the intent explicit."""
    age, theta, y, pid, year = rows
    return age, theta, y, pid, year


def calibration_table(h: Hazard, ds, filtered_theta: np.ndarray, n_bins: int = 10):
    """Predicted vs actual exit rates, bucketed by age and predicted state (§7.2)."""
    import pandas as pd
    age, theta, y, pid, year = build_training_rows(ds, filtered_theta)
    p = h.p_survive(age, theta)
    df = pd.DataFrame({"age": age, "p": p, "survived": y})
    df["age_bucket"] = pd.cut(df["age"], [0, 22, 25, 28, 31, 34, 99],
                              labels=["<=22", "23-25", "26-28", "29-31", "32-34", "35+"])
    df["p_bucket"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    by_p = df.groupby("p_bucket", observed=True).agg(
        n=("survived", "size"), predicted=("p", "mean"), actual=("survived", "mean"))
    by_age = df.groupby("age_bucket", observed=True).agg(
        n=("survived", "size"), predicted=("p", "mean"), actual=("survived", "mean"))
    return by_p, by_age
