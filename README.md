# NBA career projection — hierarchical Bayesian state-space model

A season-by-season career forecast that produces a full predictive distribution
for every player, from a 14-dimensional latent state filtered through a dynamic
generalized linear model, coupled to a survival sub-model, and propagated
forward by Monte Carlo.

This is an implementation of `career_model_spec_2.md`. It replaces the
three-sub-model LightGBM framework as the projection engine; the LightGBM stack
is retained and has a defined role (§5.3 — it supplies the prior mean) and is
the benchmark on calibration.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m career_model panel
```

```bash
.venv/bin/python -m career_model fit
```

```bash
.venv/bin/python -m career_model diagnostics
```

```bash
.venv/bin/python -m career_model posterior --chains 1 --draws 3000 --burn 1500
```

```bash
.venv/bin/python -m career_model posterior --laplace 4000
```

```bash
.venv/bin/python -m career_model precompute --draws 2000
```

Then serve the interface (it reads static files, so any static host works):

```bash
.venv/bin/python -m http.server 8000 --directory career_model/app
```

Validation:

```bash
.venv/bin/python -m career_model backtest --cutoffs 2010-2019
```

```bash
.venv/bin/python -m career_model compare --cutoffs 2015,2018
```

---

## What the model is

**State.** 14 dimensions per player-season: ten volume rates on the log
per-possession scale (`2PA 3PA FTA OREB DREB AST TOV STL BLK PF`), three
accuracy rates on the logit scale (`2P% 3P% FT%`), and log possessions.

Points is deliberately **not** a dimension. It is `1·FTM + 2·2PM + 3·3PM`,
derived per draw at simulation time. A weighted sum of counts does not have
variance equal to its mean, so no count-family observation layer applies to it.

No composite metric is in the state. PIE, BPM, PER, WS, VORP are deterministic
functions of dimensions already carried; adding one makes `Q` singular and the
model unidentified. `tests/test_no_redundant_dims.py` enforces this with a rank
check rather than a comment, so a composite added later fails a test whether or
not anyone remembers to blocklist it.

**Observations.** Negative binomial with a possession offset for volume,
binomial conditional on realized attempts for accuracy, Gaussian on log
possessions for availability. The v1 inference path uses the delta-method
Gaussian approximation of §3.3.

**Counts, never rates.** Every row of the panel is an integer count plus an
exposure column. `ingest_bbref.assert_counts_only` fails the build if a `_100`
or `_pg` column ever reaches the model panel. The reason is that observation
precision is `R ≈ 1/count`: a 5.0 AST/100 could be 15 assists or 250, and the
rate is sufficient for the mean but not for the precision, which is what the
entire filter runs on.

**Transition.**

```
theta_{i,t+1} = m_i + A(theta_{i,t} − m_i) + delta(age_{i,t}) + eta,   eta ~ N(0, Q)
Q = Lambda Lambda' + Psi,   Lambda: 14×3
```

`A` is diagonal and estimated. `delta(·)` is vector-valued — one cubic B-spline
curve per stat, knots at 21/25/29/33 — with a league level and a position-group
deviation. `Q`'s off-diagonal is where cross-stat correlation lives, replacing
the old joint residual bootstrap.

**Hazard.** `P(career continues | age, theta_{i,t})`, coupled to the *filtered*
state rather than the raw box score, so a fringe player's fluky 200-possession
line does not inflate his exit probability by noise the filter already
discounted.

---

## How it is fitted, and where that differs from the spec

The spec asks for hyperparameters fitted by MCMC over the Kalman marginal
likelihood. That is what happens, but the 84-dimensional problem is split three
ways first, because two of the three blocks have exact solutions and paying for
them with a random walk would be wasteful.

**The augmented state.** `m_i` is filtered as part of the state,
`x_t = [theta_t ; m]` with `T = [[A, I−A],[0, I]]`, rather than treated as a
parameter outside the recursion. Same model; better behaved. Because `m` is a
constant component, the filter's estimate of it after a player's last season
already conditions on his whole career — it *is* the smoothed estimate, with no
backward pass — and the shrinkage of `m_i` toward `beta'x_i` falls out of the
same gain that shrinks `theta` toward its prediction. One `R`, one mechanism.
This is why the `K = 1 − B` identity of §9 holds by construction here instead of
requiring two layers to agree.

**Linear terms are profiled exactly.** `delta(·)` and the debut-age offset enter
the state mean linearly, so the marginal likelihood is an exact quadratic in
them and the augmented (de Jong) recursion reaches the global optimum in one
Newton step — no EM, no smoother, and none of the attenuation you get from
regressing filtered states on age. It is a within-player estimator by
construction (§1.5): the derivative accumulates over a player's own
consecutive-age transitions, so a player who washes out at 30 contributes no
age-31 transition and cannot bias the age-31 increment by his absence.

**`beta` and `Sigma_player` come from EM,** with the E-step free (see above).

**Only the noise and persistence block needs numerical optimisation** — `A`,
`Psi`, `Lambda`, `phi_s`, the accuracy floors, `sigma_poss`. Fitted by L-BFGS on
the fast diagonal filter first, then the `Lambda` block on the full filter.

**Then adaptive Metropolis** over the same marginal likelihood, followed by a
Laplace step, for the parameter draws §6 step 1 requires — see "Parameter
uncertainty" below for why it ends in a Laplace step rather than the chain.

Two speed choices carry the whole thing. `A` is diagonal, so every block of
`T P T'` is an elementwise rescale of a 14×14 block and there is not a single
28×28 matmul in the hot loop. And players are ordered by career length, so the
active set at career-season `t` is a contiguous prefix and the filter slices
`[:n]` instead of filtering a rectangular block of mostly padding. A full-filter
pass over 2,858 players costs **0.15 s**; the diagonal path costs **5 ms**.

Actual wall clock for the point fit is **93 minutes**, and the shape of that
number is worth knowing before changing anything. The three diagonal outer
iterations — which do all the profiling, all the EM, and most of the likelihood
improvement — take **3.7 minutes combined**. The full-covariance `Lambda` stage
takes **55 minutes**, because L-BFGS with numerical gradients over 56
coordinates spends 56 full-filter passes per gradient. That stage is the only
place worth optimising, and the fix is analytic gradients or the M6 path, not a
faster filter.

### Flagged deviations

The spec asks for deviations to be flagged rather than made silently. These are
all of them.

| # | Deviation | Why |
|---|---|---|
| 1 | **Debut-age offset added to the initial state.** `theta_0 = m_i + Phi_init(age_0)·c_init`, a spline in debut age, not in the spec. | `m_i` is a career-average level by construction, and a 19-year-old's first season is not his career average. Without it every rookie's prior mean is his eventual peak — the easiest way to make a draft model look brilliant and be wrong. It matters most for §8.1 rookies, whose projection is *entirely* prior. |
| 2 | **Accuracy observations carry an estimated variance floor.** §3.3 specifies `R = 1/(A·p·(1−p))` with no floor. | The same argument that motivates `1/phi_s` on volume: a 600-attempt shooter is not measured to binomial precision, because defensive attention and shot quality move his true rate within a season. It is an estimated parameter, so if the data want it at zero they get it at zero. |
| 3 | **Aging hierarchy is league + position only; no per-player aging term is fitted.** The per-player *level* problem it was meant to address is instead handled at the projection layer by an empirical-Bayes reversion target. | A per-player aging scalar and a level reparameterization were both tried and both failed their own validation (collinearity with `m_i`; §1.5 violation). The real issue is that `m_i` is under-identified for late-drafted stars — an AR(1) mean gets < 1 effective observation even over a long career — which no re-fit can fix. See "Star decline, `m_i`, and the empirical-Bayes reversion target" below. |
| 4 | **Hazard event is career termination, not "absent next season".** *(Resolved — see "Within-career absence" below.)* | ~400 players in the panel miss a season and return. Calling those exits would both overstate the hazard and make the model unable to represent what actually happened to them. This is now handled by a **second** sub-model — `hazard.fit_absence`, a within-career P(plays \| career active) — composed with the career hazard at projection time so appearance = continuation × plays. The simulation carries an explicit missed-season state; the interface reports P(plays) and P(still in league) as distinct numbers. |
| 5 | **Hazard is fitted two-stage** (states, then hazard on filtered states). | §3.5 permits this for v1 and asks that it be flagged. It understates uncertainty in `gamma` by conditioning on filtered states as if known. Parameter draws cover `gamma`'s own sampling error via its asymptotic covariance, but not the state uncertainty feeding into it. |
| 6 | **`USG/100` instead of `USG%`.** | Real usage rate is a share of *team* plays while on court, which needs teammate data the model does not simulate. What is reported is possessions ended per 100 the player was on the floor for — it ranks players almost identically and is honestly computable. Labelled distinctly in the interface for that reason. |
| 7 | **BPM via a fitted translator** (§7.1 option b), not the published formula. | The team adjustment needs team offensive rating and minutes distribution, which are not simulated. The translator is a ridge from simulated per-100 rates to BPM units, validated out-of-fold grouped by player: **BPM R² 0.843, OBPM 0.888, DBPM 0.688**. It is display-and-validation only — nothing in `model/` reads it. |
| 8 | **M6 (`fit_numpyro.py`) is written but not run.** | JAX has no wheel for the Python 3.14 build this repo runs on. The module states the exact non-approximated model, non-centred throughout, initialised at the v1 mode. It is the correct next step and the place the §3.3 approximation gets dropped. |
| 9 | **v2 permanent/transient split not implemented.** | §3.4 says not to attempt it until v1 filters sensibly. v1 filters sensibly; this is the next thing to build, not something skipped. |
| 10 | **The projection store writes `payload.json` alongside the parquet layout.** | §8.5's tree specifies parquet; its serving paragraph specifies static JSON fetched by player id. A browser cannot read parquet. JSON is always written; the parquet layout is behind `--parquet`, since it doubles the store and only an analyst wants it. |
| 11 | **Parameter draws cover `A`, `Q`, `phi_s`, the accuracy floors and `sigma_poss`, plus `gamma` via its asymptotic covariance. `delta(·)`, `beta` and `Sigma_player` are held at their conditional modes.** | Those three are the blocks with exact conditional solutions (profiling and EM), which is what keeps the fit to 93 minutes rather than days — but a conditional mode is not a draw. The omitted uncertainty is smallest where it would matter most: `delta` and `beta` are global parameters estimated from 14k player-seasons, so their posteriors are tight relative to `Q`, which is drawn. M6 samples all of them jointly and is the principled fix. |
| 12 | **The stored parameter draws are independent draws from a Gaussian fitted to the Metropolis chain, not the chain itself.** | See "Parameter uncertainty" below. The chain reaches ESS ≈ 11 on `Q` after 3,000 draws, which is textbook for a random walk in 84 correlated dimensions and not fixable by running it longer at any affordable cost. The Gaussian is centred at the optimiser's mode and shaped by the chain's sample covariance, and gives ESS = 4,000 by construction. It gives up non-Gaussian shape — skew in the variance parameters, the rotational ridge in `Lambda`. |

---

## Results

### Data

14,568 player-seasons, 2,858 players, 1996-97 through 2025-26. The window starts
at 1996-97 because that is where the possession counts that form the exposure
denominator begin. 406 players have at least one mid-career gap (694 gap seasons
in total), all represented as **absent rows** — never zero rows. Schedule length
is observed rather than asserted, which recovers the 1999 (50), 2012 (66), 2020
(74) and 2021 (72) short seasons automatically.

College data reconciles to 1,358 of 1,622 players who debuted in 2009 or later
(coverage starts 2008); combine measurements to 1,199 players. Everyone else
gets an explicit missingness indicator and a separate prior mean, never an
imputation to zero.

Injury seasons — under half the schedule played, 21% of rows — are flagged
rather than dropped. The flag does two things per §4.2: it feeds the hazard, and
it inflates `R` for that season, because a player who was hurt for part of the
possessions he did log is a worse measurement of talent than his possession
count alone implies. The inflation is applied inside `refresh_R`, which is what
every likelihood evaluation calls — applying it only where the grid is first
built would leave it in an array nothing downstream reads.

**Availability gets two regimes, not one scaled regime.** The year-over-year sd
of `log(possessions)` is **0.59** between two healthy seasons and **1.60** when
either is injury-flagged. Those are not the same distribution with a multiplier
on it, and fitting a single Gaussian split the difference at 1.05 — which meant
a durable player's decade of consistent seasons was read through an observation
variance calibrated mostly by *other players'* injuries, and the filter duly
refused to believe him. The availability dimension therefore carries two
estimated scales, `sigma_poss` and `sigma_poss_inj`, selected per row by the
flag; the shared multiplier for the other thirteen dimensions is estimated too
rather than pinned at 1.5. See "The availability dimension" below for the
diagnosis this came from.

### M1 — the gate

`outputs/m1_dreb_trajectories.png` plots filtered `theta` against the raw
observed log rate for ten careers with known shapes. The properties to check,
and what they show:

- **Small-sample seasons do not move the state.** Derrick Rose's age-29 line sits
  far below every other observation of his career; the filter barely registers
  it, because it arrived with ~500 possessions and an `R` to match.
- **Uncertainty inflates across missing seasons.** Durant's 2019-20 Achilles year
  and Kawhi's age-30/31 gap both show the band visibly widening through the gap
  and re-tightening on his return.
- **Genuine movement is tracked.** LeBron rises from 19 to 25 and plateaus;
  Anthony Bennett's four thin seasons leave the band wide and *growing*, which
  is the correct answer for a player about whom almost nothing was learned.

### M2 — shrinkage

`B = R/(P+R)` at a player's first observed season, averaged over players. The
spec's benchmark is ~0.3 for a thin debut and under 0.05 for a full one:

| | `<400 min` debut | `>2400 min` debut |
|---|---|---|
| 2PA | 0.35 | **0.034** |
| 3PA | 0.25 | **0.045** |
| AST | 0.55 | 0.059 |
| OREB | 0.52 | 0.061 |
| TOV | 0.59 | 0.068 |
| DREB | 0.64 | 0.115 |
| 3P% | 0.90 | 0.46 |

The shot-volume dimensions land almost exactly on the spec's numbers. The rest
shrink harder, and the ordering is the informative part: **3P% is shrunk most**
(0.90 even on a thin debut, 0.46 on a full season), which is correct — one
season of three-point shooting is close to uninformative about true talent, and
a model that took it at face value would be the single most common way to be
wrong about a young shooter. Steals, blocks and free-throw percentage sit in
between, all rate stats with real sampling noise at NBA volumes.

Two structural notes. `Sigma_player` here is the spread *remaining after*
`f_GBM(x_i)` has spoken (§5.3), so `P` at `t=0` is a residual and the prior is
harder to overturn than it would be in a model without the trees — running with
`--no-gbm` widens `P` and lowers every `B`. And `log_poss` shows the same `B`
in both columns by construction: its observation variance is `sigma_poss^2`, a
fixed parameter rather than a count-driven one, so possessions are measured to
the same precision however many of them there are.

### M3 — cross-stat structure

Persistence, most to least sticky:

```
3PA 0.93 · BLK 0.87 · OREB 0.84 · STL 0.82 · FT% 0.82 · FTA 0.81 · PF 0.81
TOV 0.81 · 2PA 0.81 · POSS 0.75 · AST 0.74 · DREB 0.71 · 2P% 0.70 · 3P% 0.60
```

3P% is the least persistent dimension and assist rate is much stickier, exactly
as §3.4 predicts. Shot *profile* (3PA) is the stickiest thing about a player.

The three `Q` factors are interpretable, though not quite as the spec guessed:

1. **playing time / role** — loads +0.64 on log-possessions against negative
   loadings on most per-possession volume rates. A season where a player's
   minutes jump is a season where his per-possession counting rates dip. This is
   the bench-to-starter effect, and it is the dominant source of correlated
   year-over-year movement.
2. **shot profile** — 3PA positive against 2PA and FTA negative.
3. **size / foul-trouble** — blocks, fouls and offensive rebounds against
   possessions.

The spec expected factor 1 to look like usage/role (it does) and factor 2 like
size (that lands on factor 3).

### M4 — aging

`outputs/m4_aging_curves.png`. Level paths relative to `m_i`, run through the
recursion rather than cumulated — with `A < 1` the state mean-reverts, so a
constant `delta` settles at `delta/(1−A)` and a naive cumulative sum is simply
the wrong transform.

The curves reproduce known basketball without being told any of it:

- athleticism markers (OREB, STL, BLK) peak at **23** and decline from there;
- shooting percentages peak at **30–33**, long after athleticism;
- turnovers peak at 24 and fall steadily — veterans stop coughing it up;
- fouls dip to a minimum around 27 and rise again as players slow down;
- **guards** peak highest and fall hardest on OREB and STL, which is the
  position-level check §5.2 M4 asks for;
- **centres** raise their 3PA monotonically from 21 to 36 — the stretch-big era,
  recovered from within-player transitions alone.

### M5 — survival

13,986 transitions, 83.7% continuation rate, pseudo-R² 0.314. Calibration by age
bucket:

| age | predicted | actual | n |
|---|---|---|---|
| ≤22 | 0.920 | 0.938 | 1,404 |
| 23–25 | 0.860 | 0.851 | 3,963 |
| 26–28 | 0.853 | 0.844 | 3,428 |
| 29–31 | 0.846 | 0.857 | 2,483 |
| 32–34 | 0.771 | 0.783 | 1,650 |
| 35+ | 0.674 | 0.668 | 1,058 |

The largest coefficients are on 3P% and log-possessions: shooting and playing
time keep you in the league.

### Parameter uncertainty

§6 needs three nested variance sources, and the parameter one is the only one
that does not shrink as a player accumulates seasons — so omitting it would make
the *veterans*, the players the interface is most confident about, the ones whose
intervals are most wrong.

Getting it took a detour worth recording. Adaptive Metropolis over the marginal
likelihood mixes exactly as theory says a random walk in 84 correlated
dimensions will: after 3,000 draws, ESS is **11** on `Q` and **12** on `A`.
Reaching ESS > 400 needs roughly 35,000 draws, and running four chains in
parallel to get there turned out to be memory-bound rather than CPU-bound (four
resident copies of the panel drove the per-iteration cost from 0.16 s to 11 s).

The first diagnostic reflex — "R-hat is 2.65 on `Lambda`, the sampler is broken"
— is also wrong, and worth stating because it is an easy trap. `Lambda` is
identified only up to an orthogonal rotation: `(Lambda R)(Lambda R)' = Lambda
Lambda'` for any orthogonal `R`, so R-hat on individual `Lambda` entries measures
the chain sliding along a rotation manifold, not a convergence failure.
`fit_kf.identified_quantities` maps draws onto `Q`, `A`, `phi` and `sigma_poss`,
which *are* identified, and that is where diagnostics belong. (Those still showed
poor mixing — the rotation was not the whole story, just a distractor.)

What ships is a Laplace-style approximation: independent draws from
`N(mode, Sigma_hat)`, centred at the optimiser's mode and shaped by the chain's
sample covariance. The location does not depend on the chain mixing at all, and
with 14.5k player-seasons informing 84 hyperparameters the posterior is close to
Gaussian near the mode. The covariance of 3,000 correlated draws is a consistent
estimator of its shape even where the per-coordinate ESS is 11, and is shrunk 10%
toward its diagonal for conditioning. The result is **ESS = 4,000 for every
function of the parameters**, which is strictly better for §6 than a poorly-mixed
chain of the same length.

`test_simulation.py::test_parameter_uncertainty_widens_intervals` asserts the
component is actually wired in — that turning the draws on widens the predictive
interval relative to holding the hyperparameters at the mode.  The Session-4
`validate/param_uncertainty_audit.py` additionally verifies that every one of the
4,000 transformed draws is *valid* — 0 < A < 1, positive-definite Q, positive
observation scales, hazard probabilities in [0, 1] — and restates the ESS point
precisely: ESS = n because the draws are i.i.d. from the fitted Gaussian, which
is a property of the sampler, not evidence the Gaussian shape is exact.

### GBM prior mean (§5.3)

Out-of-fold R² predicting `E[m_i]` from college, combine and draft data:

```
DREB 0.78 · 3PA 0.77 · OREB 0.77 · AST 0.74 · BLK 0.69 · PF 0.63 · 2P% 0.61
POSS 0.58 · FT% 0.54 · 2PA 0.49 · STL 0.41 · TOV 0.39 · FTA 0.37 · 3P% 0.16
```

Size and shot profile translate from college; **3P% barely does** (0.16), which
is the correct and well-known answer and a good sign the offsets are honest.

### Backtest

Expanding window, `T ∈ {2010 … 2019}`, a full hierarchical refit per cutoff,
projecting `T+1 … T+5`. **143,000 scored predictions** across 4,700
player-cutoffs, on the shipped model. Full output in `outputs/backtest.log`;
regenerate with `python -m career_model backtest`, which takes about three hours
(each cutoff is a complete refit) and checkpoints after every cutoff.

**80% interval coverage, and whether the bands widen** (the check §7.2 says
matters most):

| stat | h=1 | h=3 | h=5 | width h=1 → h=5 |
|---|---|---|---|---|
| 3PA/100 | 0.848 | 0.835 | 0.823 | 4.95 → 9.27 |
| possessions | 0.860 | 0.869 | 0.867 | 4390 → 5070 |
| PTS/100 | 0.799 | 0.818 | 0.810 | 10.1 → 15.5 |
| TOV/100 | 0.788 | 0.775 | 0.773 | 1.77 → 2.07 |
| BLK/100 | 0.786 | 0.791 | 0.788 | 1.14 → 1.22 |
| STL/100 | 0.783 | 0.785 | 0.785 | 1.16 → 1.19 |
| TS% | 0.759 | 0.742 | 0.736 | 0.112 → 0.116 |
| REB/100 | 0.755 | 0.751 | 0.728 | 3.45 → 3.92 |
| **BPM** | 0.753 | 0.762 | 0.756 | 4.87 → 6.21 |
| AST/100 | 0.731 | 0.693 | **0.659** | 2.61 → 3.34 |

Bands widen with horizon for **every** stat, and **pooled PIT is approximately
uniform at every horizon from h=1 to h=5**. That is the property the
architecture exists to deliver, and it comes from propagating the state's own
mean-reverting AR(1) covariance — `P_h = A^h P_0 (A^h)' + Σ_{j<h} A^j Q (A^j)'`,
which grows sub-linearly and saturates toward `Q/(1−A²)` — rather than from any
tuned widening factor. (An earlier version of this note wrote the random-walk
approximation `P_{T|T} + hQ`; that is the `A = I` special case and over-states
the variance several-fold at long horizons. The simulator always propagated `A`
correctly; only the formula in the prose was wrong. See
`tests/test_simulation.py::test_covariance_propagation`.)

**BPM is the §7.1 check, and it passes.** It is computed from the simulated box
scores and scored against the actual Basketball-Reference value — a metric that
is never a model input, only a target. Coverage 0.75–0.76 and uniform PIT out to
five years. It is under-covered by ~4 points, and some of that is not the model:
the simulated value goes through a translator that is itself only 84% accurate
given the *true* box score, so the BPM row carries translator error on top of
forecast error.

**Assists remain the one badly-calibrated stat** — 0.731 → 0.659, U-shaped PIT
from h=2. The intervals are too narrow, meaning `Q` for that dimension is
underestimated. The likely cause is that assist rate does not move by Gaussian
increments: a player becoming his team's primary handler is a discrete role
change, and a Gaussian innovation cannot produce that jump at the rate the data
show. This is the strongest argument in the results for the v2
permanent/transient split.

Not patched. Reporting a coverage number for the stats that work and quietly
widening the one that does not is exactly the failure mode CRPS and PIT exist to
catch.

**What the injury-`R` fix changed.** Before it, possessions were dome-shaped in
PIT at every horizon beyond h=1 with coverage drifting to 0.888 — intervals too
wide. Correcting §4.2's inflation so it survives into `refresh_R` moved that row
to **uniform PIT at all five horizons** and coverage 0.86–0.87. It is the
clearest evidence in the project that the fix was real and not cosmetic.

**Survival** is still over-predicted in the middle of the range (0.775 predicted
against 0.713 actual in the 0.73–0.81 bucket). Most of that gap is definitional:
the hazard predicts *career continuation* (deviation #4) while the backtest
scores it against "appeared in that specific season", so every injury year
inside a career counts as a miss. It remains the clearest case for splitting the
two.

#### Star decline, `m_i`, and the empirical-Bayes reversion target

The model reverts a player's state toward `m_i`, his long-run level, drawn
around a draft/college-based GBM prior (§5.3). For most players that works. For
a **late-drafted player who became a star** it fails, and the failure is not a
bug — it is a fundamental identification limit worth stating precisely.

`m_i` is the mean of an AR(1) process with persistence `A`. The effective number
of independent observations of that mean over `n` seasons is roughly
`n·(1−A)/(1+A)`. For possessions `A ≈ 0.85`, so **even an eleven-season career
gives under one effective observation of the long-run mean.** The data cannot
separate "his true level is high" from "he is currently high and will revert."
So Jokić — pick 41, whom the GBM reads as a 429-possession prior — keeps a low
`m_i` no matter how many 4,800-possession seasons he stacks, and mean reversion
toward it eats ten minutes a game off his projection. Loosening the prior 10×
barely moved him: the likelihood itself carries almost no information about `m`.

Two attempts to fix this in the *fit* both failed and are documented in the git
of this section's history rather than the model: a per-player aging-rate scalar
(collinear with `m_i`, drove `m` to physically impossible values) and a level
reparameterization (violated the within-player rule §1.5, or left `m` even less
identified). Neither survived its own validation. The lesson was that the
information is not there to be extracted by re-fitting.

The fix that shipped is at the **projection layer** and does not touch the fit.
`simulate._eb_reversion_target` replaces the reversion target with a
career-length-weighted blend of the model's `m_i` and the level implied by the
player's own well-measured filtered state:

```
m_eff = w · (theta_{T|T} − L(age_T)) + (1 − w) · m_i,     w = n / (n + 4)
```

`theta_{T|T} − L(age_T)` de-ages the current state through the same level path
the projection ages along, so reverting toward it starts the player where he
actually is and declines down the curve rather than collapsing toward a
mis-estimated mean. A rookie keeps the prior; an established star trusts his
record. `K = 4` is the only knob and was set by the backtest, not by eye.

**It is validated, and it is a mitigation rather than a cure.** A clean A/B on a
held-out cutoff — same fitted fold, EB toggled — improves out-of-sample CRPS by
~2%, holds aggregate coverage (0.770 → 0.773), and helps every stat the problem
touches (possessions coverage 0.776 → 0.780, points 0.819 → 0.830, assists
0.720 → 0.747) with a trivial rebound dip. A test
(`test_eb_reversion_target_only_moves_underidentified_players`) pins the safety
property: it leaves a player alone where his record and `m_i` already agree.

What it does *not* do is make Jokić hold 37 minutes. His shipped projection opens
at ~29 mpg — better than the ~27 without EB, but still a visible drop, because
his filtered state is itself depressed by the same weak identification and EB
only corrects the reversion target, not the starting point. That residual is the
model-class limit, and closing it would need information the box score does not
contain (that he is a durable MVP, not a player about to regress). The honest
framing on his card is that the model cannot know he is exceptional.

#### Projection-layer fixes (post-v1.0, `simulate/` and the store only — no refit)

Four changes to the forward simulation and the store, none touching `model/`:

1. **The hard age-42 wall is gone.** The hazard's age quadratic turns upward
   past ~40 (survivor-selected 40+ rows), so it cannot be extrapolated alone;
   instead of a cliff that gave a 41-year-old logging 4,000 possessions a 0%
   chance of another season, survival is multiplied by a ramp that is 1 through
   age 41 and declines linearly to 0 at 46. No player now shows a >30%
   survival probability collapsing to exactly zero in one step.
2. **One missed season is `inactive`, not `retired`** (~105 players at this
   build, Haliburton included). Their projection rolls the filtered state
   through the gap year — survival flip and a full `Q` of process noise, no
   observation — so it resumes at the *current* season with honestly widened
   bands. Two or more missed seasons is still called retired, and stays a
   counterfactual from the exit point.
3. **The injury regime is simulated.** The fit always had two availability
   observation scales (§4.2); the simulation used only the healthy one, so no
   simulated season was ever lost to injury. Each projected season now draws
   an injury indicator (logit-quadratic in age, fitted on players with a real
   prior-season role so fringe role-volatility is not double-counted) and
   applies `sigma_poss_inj` to the *downside half* of the noise — the flag
   marks sub-half-schedule seasons, which shorten a career year, never stretch
   it. Medians are untouched; the left tail is where the change lives.
4. **The 6,000-possession hard clip is replaced.** It sat below the panel
   maximum (7,195) and pinned the entire upper quartile — for the youngest
   stars, the *median* — into an atom at the cap. The ceiling is now the
   physical bound (82 games × 44 minutes at league pace ≈ 7,288) with a smooth
   tanh shoulder starting at the modern-era extreme (6,100): identity through
   the whole observed bulk, no atom, no draw beyond the bound. The visible
   cost: the availability state's over-heavy upper tail, which the old clip
   silently truncated, now shows (p90 minutes for durable stars reads high).
   That tail width is a fit problem, documented below, not a display choice.

The backtest has **not** been re-run over these changes; possessions coverage
and PIT should be re-checked, since (3) widens the availability tails the
backtest previously scored as slightly over-covered already.

#### The durable-star availability fix (per-player availability EB)

A fifth projection-layer change, aimed squarely at the failure documented
below: a decade of 4,800-possession seasons still projected to a sharp year-1
drop, because both the reversion target *and the filtered starting state* are
dragged by the under-identified `m_i`.  Two per-player corrections in
`simulate/project.py`, both gated by `use_avail_eb`:

1. **Own-record level.**  Availability is the one dimension the box score
   measures *directly* -- no sampling-noise ladder, only circumstance.  A
   scalar, anchor-free Kalman filter runs over the record's level residual
   against the aging curve (`log poss - L(age)`), with the fitted availability
   process noise as drift and the fitted per-regime observation variances per
   season -- the main filter's own recursion minus the two things that hurt
   durable stars (the `m_i` anchor and the cross-dimension terms).  Its output
   mean-shifts the starting state and the reversion target's availability
   component with weight `n / (n + 2)`.  Spreads are untouched.  A naive
   precision-weighted mean of de-aged seasons was tried first and moved stars
   the *wrong* way: pooling a career toward one constant residual drags a
   curve-beating player down by his own past; the filter tracks the trend.
   An injury-flagged season is nearly ignored as a measurement (its sd is ~10x
   healthy), so a star coming off a lost year gets a bounce-back level -- this
   moves Giannis-after-an-injury-season up ~28%, not just the iron men.
2. **Injury propensity.**  The forward mixture's league age-curve rate is
   shrunk toward the player's own injury-flag share with an
   8-pseudo-season prior: an iron man has earned a lower rate, a chronically
   injured player a higher one, a rookie has earned nothing either way.

**Validated the same way the reversion-target fix was** -- one fold fit, every
cutoff-active player simulated with the toggle off/on, paired scoring
(`validate/ab_availability.py`, logs in `outputs/ab_avail*.log`):

| cutoff | possessions CRPS | possessions cover80 | all-stat CRPS (norm) |
|---|---|---|---|
| 2018 | 920.9 → **896.2** (−2.7%) | 0.813 → 0.813 | 1.000 → 0.997 |
| 2019 | 910.9 → 902.6 (−0.9%) | 0.803 → 0.799 | 1.000 → 1.000 |

No other stat moves outside noise in either direction at either cutoff, and a
test asserts the correction cannot touch a non-availability dimension.  The
cutoff-2019 bias-by-bucket table is not interpretable (2019-20 was
COVID-shortened; the harness now says so in its output); at 2018 the middle
buckets move toward zero bias (+2.5 → +1.0, +0.7 → −0.4).  The honest cost:
the sub-1,500-possession bucket's known bounce-back bias worsens slightly
(−28.2% → −29.7%), because a low record now also speaks for itself.  On the
shipped store the visible effect is Jokić's year-1 median possessions moving
~+10%, Durant ~+9%, and injured-2026 stars (Giannis) getting bounce-back
medians instead of collapsed ones.

**State-dependent aging was tried next and rejected by the same harness.**
The natural follow-up -- estimate each player's personal drift against the
aging curve from his own residual slope, shrink it, and bend his projected
availability increments by it (`use_avail_slope`, default off) -- made things
worse at both clean cutoffs: possessions CRPS 896 → 936 (2018) and 910 → 932
(2016), coverage down, and the top-bucket h=1 bias moving from +5.4% to +11.8%
and from −1.1% to +3.8%.  The reason is instructive: once the level correction
has placed a player at his own record, his historical curve-beating is already
priced in; surviving veterans' residual slopes are positive largely by
survivor selection, so extrapolating them double-counts the same evidence.
This is the third independent per-player-aging attempt to fail its own
validation (the fit-level scalar of deviation #3, and a permanent/transient
split tried outside this repo, being the others).  The code and the A/B logs
(`outputs/ab_slope_*.log`) are kept so the negative result is reproducible;
nothing enables the flag.

#### Quality bends the hazard age curve (fix 6, shipped)

The hazard was additive in age and state: every player, star or fringe, was put
on the *same* age slope, and the survival calibration table hid it because it
averaged out.  It surfaced on the cards as a fringe-tier late-career decline
applied to MVPs -- SGA given ~31% of being out of the league at 35.  The fix
adds one interaction column, `a * q`, where `q = gamma' theta` (the hazard's own
state score) standardised over the training rows, fitted by iterating the
existing two-stage IRLS twice so the quality basis stabilises.  The coefficient
is **+0.240 (se 0.041)**, ~6 sigma: quality genuinely flattens the age slope.
At the fitted level a 95th-percentile state survives age 37 at 0.965 vs 0.939
additive, a 25th-percentile state at 0.445 vs 0.507 -- the fan opens in the
right direction.

Validated with `validate/ab_availability.py --compare hazard` (survival Brier
plus gap-to-actual by age and durability bucket, since box-score CRPS barely
moves it): **Brier 0.1648 → 0.1624 (2018) and 0.1501 → 0.1488 (2016)**, better
at both, with no box-score stat regressing.  Shipped: `pipeline.fit_everything`
now fits with `interaction=True`, and `test_hazard_interaction_design_matches_projection`
asserts the projection's hot-loop `_p_survive` reproduces `Hazard.design`
bit-for-bit (the interaction column is built in both places).

**The card-level effect is deliberately modest, and not uniformly upward.**
The interaction acts on a player's *projected* state at each future age -- which
has aged and mean-reverted -- not on his current reputation.  So a player the
model still believes is elite at 35 (Jokić, Giannis, Brunson) gains a couple of
points of survival there, while one whose projected-35 state is only above
average (SGA, after nine simulated seasons of reversion toward an
under-identified `m_i`) gains nothing or loses a hair.  That is the honest
answer: the fix rewards *demonstrated quality carried forward*, and it cannot
manufacture longevity for a player whose own projected state does not support
it -- that residual is the same `m_i` identification limit fix 5 mitigates but
does not cure.  What the interaction removes is the structural error of taxing
every 35-year-old at the fringe-player rate.

#### Within-career absence: appearance vs continuation (deviation #4, shipped)

The hazard predicts *career continuation* -- no further season ever -- but the
backtest scores *appearance* -- did he play season T+h.  The two differ by the
within-career gap rate (694 gap seasons, 406 players, ~5.6% of transitions),
and the old code reported continuation as if it were appearance, so survival was
over-predicted at every horizon beyond h=1.  The interface compounded it: the
per-game table's "P(active)" column was really P(career active), biased high by
exactly the miss rate.

The fix models the two events separately.  `hazard.fit_absence` fits a second
discrete-time logistic -- P(plays season t | career still active) -- on the
within-career present/gap outcome (`in_span & t <= last_index`, `observed` as
the label), coupled to the filtered state the same way the career hazard is, so
a thin-state churn player carries a real gap probability and a durable star does
not (Jokić ~0.997/season, the bottom decile ~0.79; pseudo-R² 0.151, calibrated
across deciles).  The projection now draws a per-season play indicator among the
career-alive draws: a missed season keeps the career alive but contributes a
**zero box score** -- the explicit "missed the whole year" state the old
simulation lacked.  `P(appears) = P(career active) x P(plays | active)`, and the
interface shows both numbers, distinctly labelled.

Validated with `validate/ab_availability.py --compare absence` (survival scored
against appearance, box scores conditioned on the played event):

| cutoff | survival Brier | age-≤28 gap | age 28–31 gap | all-stat CRPS |
|---|---|---|---|---|
| 2018 | 0.1648 → **0.1589** | +0.071 → **+0.033** | +0.066 → **+0.029** | 1.000 → 0.995 |
| 2016 | 0.1501 → **0.1464** | +0.043 → **+0.007** | +0.066 → **+0.027** | 1.000 → (flat) |

Brier improves at both cutoffs and every box-score stat is neutral.  The
over-prediction shrinks most where within-career gaps actually happen -- young,
mid-career, and low-possession players -- because that is where the churn is.
**The honest cost:** the top durability tier (last season > 4,500 possessions)
tips from slightly under-predicted to a touch more so (−0.046 → −0.066 at 2018),
because even an iron man now carries a small miss probability he rarely
realises, and that tier was already under-predicted for the separate `m_i`
reason.  It is a small tax on the tier the model is already too pessimistic
about, in exchange for fixing the much larger population the gap actually
described.  The 35-plus residual barely moves (+0.065 → +0.059), because *that*
over-prediction is genuine career-hazard optimism, not within-career gaps --
a real, separate weakness this fix correctly does not pretend to touch.

#### Age x quality availability aging: the star minute-drop (shipped)

The longest-standing complaint about this model: durable stars (Jokić, Siakam)
shed minutes fast with age while their per-100 rates barely move, so the cards
show an MVP projected to a third of his minutes but nearly his production.  This
turned out to be a **genuine model error, confirmed out-of-sample** -- not
reality and not the `m_i` limit.  On the backtest, high-skill players age 32+
have their actual possessions under-projected by **−10.6% (2018) / −17.2%
(2016)**, growing to −20% by horizon 5.

The cause is that the fitted availability aging curve is a single
population-average decline, and the population is mostly role players who lose
their rotation spot.  Within-player, minute retention is strongly
quality-dependent -- at 31-32 an average player loses ~33%/yr and an elite one
~11%; at 35-36, ~46% vs ~22% (hundreds of transitions per cell) -- and the model
applied the role-player curve to everyone.

The fix is an **age x quality interaction on the availability increment**
(`simulate.fit_avail_quality_aging`), the availability analogue of the fix-6
hazard interaction.  Three choices make it identify where the earlier
state-dependent-aging attempt failed: it keys on **quality** (well-measured from
thousands of possessions), not the player's own availability slope
(survivor-noise); the quality index **excludes the availability dimension**
itself, since including it lets current-minute mean-reversion mask the effect
(t 1.6 → 3.2); and it is **hinged at age 31**, because the data show no quality
gap in retention before then, so young stars are left exactly untouched.  It is
capped so it moderates a decline without reversing it, and uses the *current*
projected state, so protection fades as a player's skill fades -- a genuinely
declined star is not rescued.

Validated with `validate/ab_availability.py --compare qualaging`, bucketing by
skill rather than by minutes (the two are not the same, which is why the pooled
number hides the effect):

| population | 2018 base → fix | 2016 base → fix |
|---|---|---|
| **high-skill, age ≥32** (the target) | −10.6% → **−1.3%** | −17.2% → **−8.9%** |
| high-skill, age <32 | −0.4% → −0.3% *(untouched)* | −7.9% → −7.9% *(untouched)* |
| low-skill, age ≥32 | +14.0% → +5.2% | +5.1% → −2.2% |
| low-skill, age <32 | +13.0% → +12.9% *(untouched)* | +10.1% → +10.0% |

Both cutoffs correct the aging-star under-projection by roughly half to all of
it, leave young players exactly alone, and improve faded veterans' *over*-
projection as a bonus (their negative quality speeds the decline).

**Two honest caveats.**  *Pooled* possession CRPS is flat (896 → 895): this is a
bias correction concentrated in subpopulations whose errors have opposite signs
and cancel in the aggregate, and CRPS rewards spread over a median shift, so the
headline metric cannot see it -- only the skill-bucketed table can.

**Onset shape (revised).**  The first version ramped the protection in linearly
from age 31, which gave only ~+0.02/sd at the first post-31 step -- so a
just-turned-32 star like Jokić got almost no year-1 protection and still shed
minutes, while a 38-year-old got *over*-protected at year 1.  The empirical
protection is actually a **step**: it jumps to ~+0.09/sd at 31-32 and stays
roughly flat (measured +0.09, +0.07, +0.08 across 31-32 / 33-34 / 35-37).
Fitting a step-at-31 fixed both ends and, on the backtest, moved the high-skill
age-32+ deep-horizon under-projection from −18%/−20% at h=4/5 to −0%/−3%.  The
cost is a mild *over*-projection of the same group at short horizons (h=1
around +9%): the interaction is estimated on *survivors* (within-player
transitions condition on playing both seasons), so applying the full survivor
magnitude to a backtest group that includes the stars who declined reads as
over-projection.  That short-horizon overshoot is on the group, not on healthy
stars -- Jokić's shipped year-1 still lands below his own last season -- and the
multi-year trajectory is what the fix exists to get right.

#### The minutes/games split under-served durable stars (the visible year-1 drop)

The projection is in *possessions*; the per-game table needs a split into games
x minutes-per-game, which `MinutesSplit` regresses from possessions and age.
That split systematically **under-assigned minutes to high-minute stars**,
because a durable star concentrates his possessions in fewer games (Jokić: 37
mpg over 65 games, not 33 over 74), and the population regression pulls him
toward the average split.  It showed up as a discontinuity at the
history/projection boundary: the last *observed* row displayed his real 37 mpg,
and the first *projected* row dropped to ~33 before any decline at all -- the
larger half of the "why does Jokić lose 7 minutes in year one" question.  A
leave-future-out sweep over 1,787 high-mpg player-seasons measured the bias at
**−2.79 minutes** and picked the correction: weight recent seasons more (a
2-season half-life) and shrink the per-player durability offset half as hard.
That cut the high-mpg bias to **−2.30** and its MAE **3.50 → 3.34** while
*improving* the overall mpg MAE (2.94 → 2.91), so it is a genuine calibration
gain, not a star-only thumb on the scale.  Combined with the onset fix, Jokić's
projected year-1 mpg went from ~30.5 to ~33.5 (his last actual was 37), and the
decline from there is gentle rather than a cliff.

#### Latent development and role dynamics (Session 3)

Three candidates for the state model, validated old-vs-new on the frozen folds
2018 + 2023 with availability held at the Gaussian shipping path (so any
rate-coverage change is the state model's, not injected availability variance):

**Candidate A — diffuse-m prior (learn the NBA level from evidence).**  The
persistent level `m_i` is anchored to a tight pre-NBA prior; a late-drafted star
stays pinned low.  Candidate A inflates only the m-block of the filter's initial
covariance (`filtered_states(m_prior_scale=)`) so NBA seasons drive `m`.
Synthetic recovery (scalar augmented filter, `tests/test_state_dev.py`) confirms
the mechanism: an eight-elite-season late pick recovers +0.99 (truth +1.2) vs
+0.32 tight; an eight-poor-season #1 pick corrects to −0.15 (truth −0.4) vs
+0.62; a one-season outlier stays shrunk (+0.22, not +1.5).  **But it fails the
fold acceptance:** aggregate CRPS **+15%**, and it *over*-projects (PPG bias
+0.98, MPG +0.87) and over-covers (0.86–0.91) because a diffuse `m` chases
recent seasons.  This is the same fit-level relearning the README's deviation #3
already found unidentifiable; **rejected, left behind the flag.**

**Candidate B — role-change innovations.**  Two variants replace the Gaussian
forward innovation: **Student-t** (heavier tails, global) and a **two-component
mixture** — a fraction `pi_t` of draws get extra variance on the shot/playmaking
dimensions, with `pi_t` a simple logistic on cutoff-available recent volatility,
shot/assist swings and age (no HMM).  Both fix the Session-2 finding that the
production rates are under-dispersed: conditional rate coverage moves from
0.72–0.76 toward 0.80, PIT stays uniform, and **APG deep-horizon coverage
recovers (h=5 0.72 → 0.79–0.80)** — all at a small aggregate CRPS cost (+1.7–
1.9%), the expected sharpness price of correcting over-confident intervals.

Session 3 shipped the **mixture provisionally**; Student-t was left off.  On its
target subgroups the mixture wins, beating Student-t on exactly the abrupt
role-change cases while preserving stable veterans:

| subgroup (AST/100 cover80) | shipping | Student-t | **mixture** |
|---|---|---|---|
| shot-volume breakout | 0.69 | 0.73 | **0.78** |
| assist-role breakout | 0.67 | 0.73 | **0.76** |
| top-quality | 0.72 | 0.81 | **0.83** |
| stable veteran | 0.81 | 0.84 | 0.84 |

It keeps medians centred (unlike diffuse-m) and only widens the role dimensions
of volatile players.  What no candidate fixes acceptably is the **elite/
top-scorer PPG underprojection** (−1.5 bias): that is a *level* error, and the
only candidate that addresses it (diffuse-m) overshoots — so it remains a
documented model-class limitation, not solved this session.

**Session-4 verdict — the mixture is DISABLED and the shipping innovation is
Gaussian.**  Session 3 flagged the mixture as *provisional*, to be re-decided
against the aggregate at final integration.  The aggregate is decisive: on the
rolling-origin state A/B (`outputs/state_dev_ab_summary.json`) the mixture
raises normalized CRPS **1.000 → 1.017** and pushes 80% coverage from a
calibrated **0.801** to an over-covered **0.839** — it *over*-widens the whole
corpus to fix a few subgroups.  Per "enable only candidates that passed their
gate", the mixture fails its gate and does not ship.  Concretely it inflated the
next-season 3PA/100 80% band by **+28% (SGA), +21% (Tatum), +8% (Zubac)** —
bands the fresh exact-UI backtest confirms were too wide.  The mixture,
Student-t and diffuse-m code stays behind flags for A/B use; none is the
shipping path.  Note a latent inconsistency this fixed: the store was being
*generated* with the mixture while the exact-UI backtest that "validated" it
called `project.simulate` with the Gaussian default — so the shipped store had
never actually matched the backtested configuration until Session 4 reverted it.

#### Age vs era, and the parameter-uncertainty audit (Session 4)

**Separating individual aging from the league era.**  The within-player aging
curve (`model/aging.py`) blocks *cross-sectional* survivor bias — it never
compares an age-34 population mean to an age-27 one — but age and calendar
season still move together *inside* a career, so a within-player curve with no
era term reads the league-wide three-point revolution as an aging effect.
`validate/era_separation.py` builds a **fold-local era component** (option A:
league × position × season baselines, player modelled as the residual), refit
inside each cutoff from seasons ≤ cutoff only.  Three results:

- **Leakage test passes** at cutoffs 2014/2018/2022/2026: appending a fabricated
  extreme future season moves no baseline at season ≤ cutoff (max drift 0.0).
- **Identifiability holds**: residual within-player increments stay ≈ mean-zero
  (|mean| ≤ 0.03 per stat), the constraint the aging spline relies on.
- **The centre-3PA question is answered.**  The league centre 3PA/100 baseline
  rose ~4.6× over 1997→2026 (log −2.0 → −0.47).  The apparent within-player
  "centres shoot more threes as they age" slope is **+0.105/yr uncontrolled**
  but **+0.046/yr after era control** — so **~56% of it was the era, not
  aging** (the rest is a real, smaller within-career increase).

This is a genuine confound in the shipping aging curve, now *quantified*.  The
era component is a **candidate/diagnostic and does not ship** this session:
folding it into the transition changes the fit and would need its own
rolling-origin gate before it could be trusted, which is out of scope for a
final-integration pass.  It is documented as a known bias, not silently enabled.

**Parameter-uncertainty audit** (`validate/param_uncertainty_audit.py`).  The
shipping artifact carries 4,000 `laplace_draws` — independent draws from
`N(mode, Σ̂)`.  The audit confirms every transformed draw satisfies the model's
hard constraints: **0 < A < 1** (0.55–0.93), **positive-definite Q** (min
eigenvalue 8e-4 > 0), positive observation scales (σ_poss ≥ 0.17), and hazard /
absence survival probabilities in [0, 1].  On honesty: ESS **= n only because
the draws are i.i.d. from the fitted Gaussian**, not because a Markov chain
reached that ESS — a random-walk Metropolis over 84 correlated coordinates
delivers ESS ≈ n/d ≈ 47.  Of the four requested approximations, the shipping
path realises #1 (Gaussian covariance) ≡ #2 (Hessian-Laplace near the mode, by
Bernstein–von Mises); the career bootstrap (#3) and reduced-model NUTS (#4) were
not re-run this session (compute / memory bound), and the audit found no
calibration problem that would justify rewriting the inference backend.

#### Frozen Session-4 shipping configuration

`simulate/precompute.SHIP_CONFIG` is the single source of truth for enabled
features; its hash is stamped into every artifact as `config_fingerprint`
(current: **`e8210db080af4cd1`**) and surfaced in `index.json`, each player
`meta.json`, and the UI header.

| feature | verdict | shipped |
|---|---|---|
| Gaussian state innovation | baseline | **on** |
| EB reversion target | passed | **on** |
| within-career absence (P plays\|active) | passed | **on** |
| injury regime + downside-only | passed | **on** |
| age×quality availability aging | passed | **on** |
| hazard age×quality interaction | passed | **on** |
| joint availability (S2) | failed gate | off |
| diffuse-m (S3-A) | failed gate (CRPS +15%) | off |
| Student-t innovation (S3-B) | dominated | off |
| role-change mixture (S3-B) | failed gate (CRPS +1.7%) | off |
| permanent/transient state | failed (pre-S3) | off |
| fold-local era component (S4-A) | candidate, unvalidated | off |

#### The availability dimension, in detail

Possessions is the weakest part of the model and the one most worth
understanding, because everything else is a rate multiplied by it.

The aging curve is right. The model's implied within-player decline at age 31–34
is **−27% per year**; the empirical within-player mean over the same ages in
this panel is **−27.5%**. It is not guessing.

Bias, bucketed by *last observed possessions* — a pre-cutoff quantity, so the
comparison is honest:

| last season | model h=1 | actual | bias |
|---|---|---|---|
| > 4,500 | 4,129 | 4,206 | −1.8% |
| 3,000–4,500 | 3,145 | 3,307 | −4.9% |
| 1,500–3,000 | 2,077 | 2,271 | −8.5% |
| < 1,500 | 941 | 1,349 | **−30%** |

High-usage players are projected almost exactly right; the real miss is at the
bottom, where the model does not expect fringe players to bounce back as often
as they do. (Bucketing by the *outcome* instead reverses the apparent sign —
that is regression to the mean in the residual, an artifact of conditioning on
the thing being predicted, and it is worth stating because it is the easy way to
convince yourself of a bias that is not there.)

**Where it goes wrong is individual durable stars**, and this is mitigated by
the empirical-Bayes reversion target described below rather than fully fixed.
Because the reversion target `m_i` is under-identified for a late-drafted star,
a 31-year-old MVP is pulled toward a mean estimated mostly from his draft prior.
Jokić has played ~4,900 possessions a season for a decade; before the EB target
the model projected a year-1 median near 2,840, and with it ~3,500 — better, but
still a visible drop, because his filtered starting state is itself depressed by
the same weak identification. This is one player inside a tier that is unbiased
on average, so aggregate calibration never flags it.

**Survival, backtested**, is over-predicted at h≥2 — 0.747 predicted against
0.680 actual for ages 26–28, pooled over horizons. Most of that gap is
definitional rather than a modelling error: the hazard predicts *career
continuation* (deviation #4), while the backtest scores it against "appeared in
that specific season", so every injury year inside a career counts as a miss.
It is still the clearest case in the results for revisiting that choice.

### Head-to-head against the LightGBM stack

Same cutoffs (2013, 2016, 2019), same players, same scoring code. The benchmark
is a rebuild of the three-sub-model framework — survival classifier, possessions
regressor, one regressor per stat on the old lag/Marcel/career-shape vocabulary,
run recursively, with a **joint** residual bootstrap so cross-stat correlation
survives. That is the strongest honest version of the old approach.

**80% interval coverage** (nominal 0.80):

| stat | LGBM h=1 | SSM h=1 | LGBM h=5 | SSM h=5 |
|---|---|---|---|---|
| PTS/100 | 0.458 | **0.783** | 0.406 | **0.803** |
| 3PA/100 | 0.442 | **0.839** | 0.334 | **0.812** |
| AST/100 | 0.504 | **0.743** | 0.418 | **0.645** |
| BLK/100 | 0.576 | **0.784** | 0.569 | **0.758** |
| STL/100 | 0.557 | **0.761** | 0.507 | **0.768** |
| TOV/100 | 0.542 | **0.781** | 0.495 | **0.776** |
| possessions | 0.454 | 0.851 | 0.381 | 0.863 |

The gradient-boosted stack's 80% intervals cover **40–58%**, and every stat at
every horizon is U-shaped in PIT. Worse, its coverage *falls* as the horizon
grows — 3PA goes 0.442 → 0.334 — while the state-space coverage is flat. That is
exactly the failure §7.2 names as the reason this architecture exists. The
mechanism is visible in the widths: the bootstrap injects residual noise once per
recursive step from a pool fitted at h=1, so its bands grow about 40% from h=1 to
h=5 while starting two to four times too narrow.

**CRPS** — the headline. The state-space model wins **6 of 7** comparable stats
at h=1 and 6 of 7 at h=5. The single loss is possessions (738 vs 799 at h=1),
which is the availability dimension the PIT histogram flags independently — two
diagnostics agreeing on one real weakness.

**RMSE** is close to a wash, which is *better* than the spec anticipated.
LightGBM is genuinely better at h=1 on blocks (0.546 vs 0.701) and possessions
(1203 vs 1399), and on points (4.23 vs 4.52). By h=5 the state-space model has
the better RMSE on points (5.78 vs 6.02), turnovers and blocks, because a
recursive point forecast accumulates bias that a mean-reverting latent state
does not.

Two caveats. `reb_per100` and `ts_pct` are not in the benchmark's target set, so
they have no LightGBM column. And the comparison uses three cutoffs rather than
ten, for run time.

---

## Leakage control

The filter recursion is causal on its own — `theta_{t|t}` conditions on data
through `t`, and `v_t = z_t − theta_{t|t−1}` is formed against a prediction made
before `z_t` was read. That property is easy to destroy by accident, and when it
is destroyed nothing errors; the score just quietly improves. Four channels,
closed and asserted:

**1. Never project from smoothed states.** There is no RTS smoother in this
codebase. `FilterResult` carries a `conditioning` tag and
`simulate.project.simulate` **rejects** anything not marked `"filtered"`, so a
smoother added later for UI history rendering cannot be wired into a forecast by
a stray keyword argument. `test_no_smoother_leakage.py` asserts the stronger
property directly: the filtered state at the cutoff is **bit-for-bit identical**
whether or not post-cutoff seasons exist in the panel at all.

**2. All hyperparameters refit per fold.** Each cutoff runs a full hierarchical
refit — `Q`, `A`, `delta(·)`, `Sigma_player`, `phi_s`, the accuracy floors,
`sigma_poss`, `gamma`. Aging leaks hardest because it is precisely what is being
projected, so both the aging spline coefficients *and* the basis centring vector
are recomputed from the truncated frame. `test_age_basis_centring_is_fold_local`
fails if the centring is ever shared across folds.

**3. `beta` refit per fold**, over the truncated player set — players who
debuted at or before `T` — with the `x_i` standardisation computed on that same
set.

**4. GBM prior mean refit per fold**, out of fold and grouped by player, on
pre-`T` outcomes only. `test_gbm_prior_is_fold_local` feeds it pure noise and
asserts out-of-fold R² stays under 0.05; if the offsets were fit in-sample,
`Sigma_player` would collapse and every rookie's interval would be too narrow.

`dataset.load` is the single choke point that makes 2–4 hold: every global
quantity is derived from the frame it returns, so cutting the panel there
confines all of them, and `score_cutoff` asserts the cut actually happened
before scoring anything. This makes a backtest expensive — one full hierarchical
refit per cutoff — and that cost is the point. A cheap backtest that reuses a
global fit produces a number that is not measuring what it claims to.

`crps_by_cutoff` is the standing sanity check: scores should be flat in the
cutoff up to noise and real league drift. If early cutoffs look dramatically
better, assume leakage before assuming drift — earlier folds have more future
data in the panel for something to leak from — and audit the four channels
above.

**The check, run.** Pooled CRPS at h=1 by cutoff:

```
2010  84.1   2012  88.6   2014  85.2   2016  88.7   2018  91.3
2011  81.2   2013  92.8   2015  92.2   2017  93.1   2019  89.2
```

80% coverage over the same cutoffs stays in 0.786–0.819. There is no advantage
to the early folds — if anything a mild *dis*advantage to the later ones, which
is the direction real league drift predicts (the three-point revolution runs
right through this window and makes volume stats genuinely harder to project
forward) and the opposite of the direction leakage would push. Clean.

---

## Validation

Calibration is the primary criterion, and this model is *expected* to lose to
LightGBM on point-prediction RMSE for players with long histories. RMSE rewards
a confident centre and is blind to whether the interval around it means
anything. The metrics that decide it:

- **CRPS** as the headline — a proper scoring rule for distributional forecasts,
  in the units of the variable, reducing to absolute error against a point
  forecast so neither side is advantaged.
- **PIT histograms**, randomized (counts are discrete; the naive rank PIT is
  non-uniform even under a perfect model, and low-count stats are mostly ties).
  U-shaped means intervals too narrow, dome-shaped too wide.
- **Interval coverage at h=1 and h=5 separately.** The failure this architecture
  exists to fix is intervals that do not widen correctly with horizon, and an
  average over horizons hides exactly that.
- **Survival calibration** by age and by predicted state.

---

## The interface

`career_model/app/` — static HTML, no build step, no server-side compute. It
reads precomputed percentile files and renders them; it contains no modelling
logic, no threshold encoding basketball judgment, and no fallback computation.

Two entry paths, kept visibly distinct:

**Established players** project from `theta_{T|T}` and `P_{T|T}` — an informed
posterior with correspondingly tight bands.

**Incoming rookies have no NBA seasons at all.** Their projection is drawn
entirely from the prior, `m_i ~ N(f_GBM(x_i) + beta'x_i, Sigma_player)`, shifted
by the debut-age offset and propagated forward.

§8.1 asserts a rookie's year-1 interval should be about as wide as an
established player's year-5 interval. Measured on the generated store — 80%
band on points per 100, relative to the median, taken across the corpus:

| | relative 80% width |
|---|---|
| established player, year 1 | 0.52 |
| established player, year 5 | 0.89 |
| **incoming rookie, year 1** | **1.01** |

That lands where the spec says it should, and it is not something the interface
was told to do — it falls out of `Sigma_player` plus the stationary dispersion
being the only thing a rookie has. Presenting the top and bottom rows with
identical visual treatment would be the single most misleading thing this
interface could do, so rookie cards carry an explicit marker shown by default,
not behind a tooltip; a rookie with no college data on file gets a second one.

**Season by season, if he plays.** A box-score table — MPG, PTS, REB, AST, STL,
BLK, GP — one row per projected season, conditional on the player being in the
league that year, with `P(active)` in the same row so the condition is never
read separately from the numbers. Recent actual seasons sit above the
projections in a distinct treatment, the same history/forecast boundary the fan
chart draws.

The model projects *possessions*, not games, and §3.2's availability dimension
deliberately conflates the two ways a season gets short (40 games at 30 minutes;
75 games at 16). So per-game figures need a split, and
`derive.MinutesSplit` supplies it: `log(mpg)` on a quadratic in
`log(possessions)` plus age gives **R² 0.78**, and **39% of the residual variance
is a stable player trait**, so the draw carries a shrunk per-player durability
offset alongside a fresh within-player draw. Games are then derived from minutes
and the drawn MPG rather than sampled separately, so `games × mpg` reproduces the
simulated minutes exactly and a draw cannot show 30 points a game on 20 minutes
unless the box score really says so. Like the BPM translator, it is a fitted
display-layer device: nothing in `model/` imports it.

**The residual is strongly heteroskedastic**, and getting that wrong breaks the
table rather than merely blurring it. Residual sd runs 0.48 below 500
possessions down to 0.067 above 4,500. Minutes per game is bounded by 48 and a
starter already sits near the ceiling, so a pooled sd of 0.217 pushed
Wembanyama's 90th percentile to 43.7 minutes — a figure that occurs a handful of
times in thirty years — where the physical bound clipped it and the interval
stopped meaning anything. The sd is modelled as a quadratic in `log(possessions)`
(a log-linear fit misses the fall-off at the top, which is the end that matters),
and the same weighting is used for the player offset, so one garbage-time season
cannot dominate a durable player's durability estimate.

Validated out of sample in the configuration that ships — fit through 2023,
applied to 2024–26, holding possessions at their actual values so the split is
tested rather than the projection:

| | median bias | 80% coverage | MAE |
|---|---|---|---|
| games played | −0.23 | 0.746 | 6.9 |
| minutes per game | −0.82 | 0.746 | 3.4 |
| MPG, high-usage (>3,000 poss) | — | **0.837** | **2.5** |

Essentially unbiased, and tightest where the table is most read. Widening the
train/test gap to fit-through-2017 introduces a real +1.9 game bias, which is
era drift rather than a modelling error: load management increased sharply after
2018, so modern players deliver the same possessions across fewer games.

These rows also inherit whatever the availability dimension gets wrong, and the
card says so in words — see the possessions sub-section above.

Display rules that are enforced rather than encouraged:

- **No naked point estimates anywhere.** Every projected number carries an
  interval, in the fan chart, the peak table, the totals table and the search
  results.
- Default band 80%, with 50% and 95% as toggles; the 95% band is always drawn
  faintly behind the selected one so the tail stays visible.
- **Bands are not required to widen with horizon, and a non-widening band is
  not flagged as a bug.** The state is a *mean-reverting* AR(1) (`A < 1`), so
  its variance saturates toward the stationary bound `Q/(1−A²)` rather than
  growing without limit — the correct h-step covariance is
  `P_h = A^h P_0 (A^h)' + Σ_{j<h} A^j Q (A^j)'`, not `P_0 + hQ`. A declining
  veteran's band can legitimately hold flat, or narrow in points-per-100 as his
  level drifts down, while the latent variance still grows. An earlier build
  had `fan_chart.js` print a "band does not widen ⇒ Q-propagation bug" warning;
  that rule was mathematically unjustified for a mean-reverting process and has
  been **removed**. Covariance correctness is now asserted where it belongs — a
  unit test (`test_covariance_propagation`) comparing the simulator against the
  augmented-state analytic covariance and direct Monte Carlo — not by watching
  the chart.
- Career totals show median and band; the mean is not shown at all, because
  these distributions are heavily right-skewed and the mean describes almost
  nobody.
- Thin projections say so in words. Most readers look at the centre line and
  ignore the shading.

Not built, per §8.6: no compare-two-players view (it invites reading two medians
against each other, which is the point-estimate thinking the model exists to
replace), no user-adjustable model parameters, and no projections for players
outside the corpus — an unknown search returns "insufficient data" rather than a
maximally diffuse guess.

---

## Repo map

```
career_model/
  config.py                  state layout, paths, spline knots — declared once
  data/
    ingest_bbref.py          season totals: counts + possessions, rate-column guard
    ingest_college.py        SOS-adjusted college metrics
    ingest_anthro.py         combine, player index, draft
    reconcile_ids.py         canonical ids, corroborated matching, manual overrides
    build_panel.py           → player-season panel, gaps as absent rows
  model/
    observations.py          NB / binomial / Gaussian layers, R computation, the grid
    state_space.py           KF predict/update, missing seasons, exact linear profiling
    aging.py                 centred B-spline basis, level paths, hierarchy
    hierarchy.py             priors, packing, Q = ΛΛ' + Ψ
    hazard.py                coupled discrete-time survival + within-career absence
    gbm_prior.py             §5.3 — LightGBM as the prior mean
    dataset.py               the fold choke point
    fit_kf.py                v1: profiling + EM + L-BFGS + adaptive Metropolis
    fit_numpyro.py           M6: exact model, non-centred, states as parameters
  simulate/
    project.py               Monte Carlo forward simulation, three variance sources
    derive.py                points, TS%, USG/100, BPM translator
    precompute.py            batch → percentile store
  app/
    index.html  search.js  player_view.js  fan_chart.js
    projections/             generated, gitignored
  validate/
    backtest.py              expanding window + leakage assertions
    calibration.py           CRPS, PIT, coverage
    compare_lgbm.py          head-to-head vs the old stack
  tests/
    test_filter_identities.py     K = 1 − B, and the R floor
    test_missing_seasons.py       gaps vs zero rows
    test_no_redundant_dims.py     rank check, PD assertions
    test_no_smoother_leakage.py   the four backtest channels
  diagnostics.py             the M1–M5 stage gates
  pipeline.py                end-to-end fit
```

### The priority test

`test_filter_identities.py` asserts the identity that ties the model together:
in scalar form the Kalman gain `K = P/(P+R)` and the hierarchical shrinkage
factor `B = R/(P+R)` satisfy `K = 1 − B`, and both consume the same `R` supplied
by the observation layer. If that fails numerically, observation variance is
being computed inconsistently between the two layers — which is how a
hierarchical model ends up shrinking twice, or not at all, while still producing
plausible-looking output.
