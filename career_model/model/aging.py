"""The vector-valued aging function `delta(age)` (§3.4).

Three things about this module follow from the spec and none of them are
negotiable.

**It is vector-valued.**  One curve per stat, not a scalar multiplier on a
shared shape.  Assist rate and block rate do not peak at the same age and do not
decline at the same speed; a shared curve forces them to.

**It is an increment, not a level.**  `delta(age)` is what gets *added* going
from season t to season t+1, so cumulating it gives the familiar hump-shaped
career curve.  A constant increment is indistinguishable from a shift in the
player's stable level `m_i`, so the basis is centred over the empirical age
distribution: any coefficient vector then produces a curve with population mean
zero, which breaks the confound and leaves `m_i` identified.

**It is identified within-player only** (§1.5).  Nothing here ever compares an
age-34 population mean to an age-27 population mean.  The coefficients are
estimated from transitions -- the same player at consecutive ages -- through the
augmented Kalman recursion in `state_space.profile_linear_terms`.  Survivor bias
cannot enter, because a player who retires simply contributes no transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from ..config import AGE_BOUNDARY, AGE_KNOTS, POSITION_GROUPS, S

DEGREE = 3


def knot_vector(knots=AGE_KNOTS, boundary=AGE_BOUNDARY, degree: int = DEGREE) -> np.ndarray:
    lo, hi = boundary
    return np.concatenate([[lo] * (degree + 1), np.asarray(knots, float), [hi] * (degree + 1)])


def n_basis(knots=AGE_KNOTS, degree: int = DEGREE) -> int:
    return len(knot_vector(knots, degree=degree)) - degree - 1


def basis_matrix(age: np.ndarray, knots=AGE_KNOTS, boundary=AGE_BOUNDARY,
                 degree: int = DEGREE) -> np.ndarray:
    """Cubic B-spline design matrix, rows = ages, columns = basis functions.

    Ages outside the boundary knots are clamped rather than extrapolated: a
    cubic spline extrapolates violently, and there is no data past 42 to justify
    whatever it would say.
    """
    t = knot_vector(knots, boundary, degree)
    a = np.clip(np.asarray(age, float), boundary[0], boundary[1] - 1e-9)
    flat = np.ascontiguousarray(a.reshape(-1))
    design = np.asarray(BSpline.design_matrix(flat, t, degree, extrapolate=False).todense())
    return design.reshape(*a.shape, design.shape[1])


@dataclass
class AgeBasis:
    """A centred B-spline basis, plus the centring vector used to build it."""

    knots: tuple
    boundary: tuple
    centre: np.ndarray            # (n_basis,) population-weighted column mean

    @property
    def size(self) -> int:
        return len(self.centre)

    def __call__(self, age: np.ndarray) -> np.ndarray:
        return basis_matrix(age, self.knots, self.boundary) - self.centre

    def curve(self, coefs: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """Evaluate `delta` (increment per season) at a grid of ages -> (n_age, S)."""
        return self(ages) @ coefs

    def level_path(self, coefs: np.ndarray, A: np.ndarray, ages: np.ndarray,
                   start: np.ndarray | None = None) -> np.ndarray:
        """The readable aging curve: level relative to `m_i`, by age -> (n_age, S).

        Naively cumulating `delta` would be wrong.  The transition is
        `theta_{t+1} - m = A(theta_t - m) + delta`, so with `A < 1` the state
        mean-reverts and the increments do **not** accumulate without bound --
        a constant `delta` settles at `delta/(1-A)`, not at infinity.  The only
        correct way to turn increments into a level is to run the recursion,
        which is what this does:

            L_0 = start,   L_{k+1} = A L_k + delta(age_k)

        It reduces to the cumulative sum as `A -> 1` (the near-random-walk
        dimensions, like 3PA) and to `delta/(1-A)` where `A` is small (the
        noisy ones, like 3P%), which is why it has to be run rather than
        approximated either way.  `ages` must be spaced one season apart.
        """
        inc = self.curve(coefs, ages)
        out = np.zeros_like(inc)
        out[0] = 0.0 if start is None else start
        for k in range(1, len(ages)):
            out[k] = A * out[k - 1] + inc[k - 1]
        return out


def fit_basis(ages: np.ndarray, knots=AGE_KNOTS, boundary=AGE_BOUNDARY) -> AgeBasis:
    """Build the basis and centre it over the ages actually observed."""
    raw = basis_matrix(ages, knots, boundary)
    return AgeBasis(knots=tuple(knots), boundary=tuple(boundary), centre=raw.mean(axis=0))


@dataclass
class AgingCurves:
    """League curve plus per-position-group deviation (§4.1 levels 1 and 2).

    `league` is (n_basis, S); `position` is (3, n_basis, S) and is shrunk toward
    zero by a ridge in the GLS solve.  The player level of the hierarchy is
    carried as a scalar rate-of-aging multiplier per player -- see
    `player_scale` -- rather than a full per-player curve, because eight degrees
    of freedom per stat cannot be recovered from an eight-season career.  That
    is a deviation from the letter of §4.1 and is flagged as such.
    """

    basis: AgeBasis
    league: np.ndarray                       # (n_basis, S)
    position: np.ndarray                     # (3, n_basis, S)
    player_scale: dict[int, float] | None = None

    @classmethod
    def zeros(cls, basis: AgeBasis) -> "AgingCurves":
        return cls(basis=basis,
                   league=np.zeros((basis.size, S)),
                   position=np.zeros((len(POSITION_GROUPS), basis.size, S)))

    def coefs_for(self, pos_idx: np.ndarray) -> np.ndarray:
        """(N, n_basis, S) effective coefficients, one per player."""
        return self.league[None, :, :] + self.position[pos_idx]

    def offsets(self, age: np.ndarray, pos_idx: np.ndarray) -> np.ndarray:
        """delta at (player, season) -> (N, T, S)."""
        phi = self.basis(age)                       # (N, T, n_basis)
        coefs = self.coefs_for(pos_idx)             # (N, n_basis, S)
        out = np.einsum("ntk,nks->nts", phi, coefs)
        if self.player_scale is not None:
            scale = np.array([self.player_scale.get(i, 1.0) for i in range(len(pos_idx))])
            out = out * scale[:, None, None]
        return out

    def curve_table(self, A: np.ndarray, ages: np.ndarray) -> dict[str, np.ndarray]:
        """League and per-position level paths, for the M4 eyeball check."""
        out = {"league": self.basis.level_path(self.league, A, ages)}
        for g, name in enumerate(POSITION_GROUPS):
            out[name] = self.basis.level_path(self.league + self.position[g], A, ages)
        return out
