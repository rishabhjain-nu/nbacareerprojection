"""Synthetic recovery tests for the persistent-level candidate (S3-A).

The augmented state is `[theta_t ; m]` with `theta_{t+1} = m + A(theta_t - m) +
eta`.  These tests exercise the *m-learning* mechanism directly with a scalar
2x2 filter (the same math the full filter runs per dimension), on four
controlled careers with a KNOWN true level, comparing the shipping tight
m-prior against the diffuse m-prior (`m_prior_scale`).

Requirements checked:
  1. a late-drafted player with eight elite seasons -- diffuse recovers the
     high NBA level; the tight prior stays anchored to the low draft
     expectation;
  2. a highly-drafted player with eight poor seasons -- diffuse corrects
     downward off the high draft expectation;
  3. a role player with one elite outlier season -- both stay shrunk (the
     single season does not move the level much);
  4. a genuinely declining veteran -- the filtered state tracks the decline.
"""

from __future__ import annotations

import numpy as np
import pytest

# Representative availability-dimension dynamics from the shipped fit.
A, Q = 0.85, 0.29
D = Q / (1 - A ** 2)                 # stationary dispersion
SIG_M = 0.25                         # tight m-prior variance (Sigma_player diag)
DIFFUSE = 25.0                       # m_prior_scale for Candidate A


def _augmented_filter(z, r, m0, m_scale):
    """Scalar augmented Kalman filter over [theta, m].  Returns (theta_T, m_T)."""
    F = np.array([[A, 1 - A], [0.0, 1.0]])
    Qa = np.array([[Q, 0.0], [0.0, 0.0]])
    H = np.array([[1.0, 0.0]])
    x = np.array([m0, m0])
    P = np.array([[SIG_M + D, SIG_M], [SIG_M, SIG_M * m_scale]])
    for zt in z:
        # update
        S_ = (H @ P @ H.T)[0, 0] + r
        K = (P @ H.T)[:, 0] / S_
        x = x + K * (zt - (H @ x)[0])
        P = P - np.outer(K, H @ P)
        # predict
        x = F @ x
        P = F @ P @ F.T + Qa
    return float(x[0]), float(x[1])


def _recover(z, r, m0):
    """Return (m_tight, m_diffuse)."""
    return (_augmented_filter(z, r, m0, 1.0)[1],
            _augmented_filter(z, r, m0, DIFFUSE)[1])


def test_late_drafted_eight_elite_seasons():
    # draft expects average (m0=0); the player is truly elite (m*=+1.2).
    rng = np.random.default_rng(0)
    z = 1.2 + rng.normal(0, 0.05, 8)          # eight well-observed elite seasons
    m_tight, m_diffuse = _recover(z, r=0.03, m0=0.0)
    assert m_diffuse > m_tight + 0.3          # diffuse learns the NBA level
    assert m_diffuse > 0.9                     # recovers close to the true +1.2
    assert m_tight < 0.9                       # tight prior stays anchored low


def test_highly_drafted_eight_poor_seasons():
    # drafted #1 (m0=+1.0); the player is a bust (m*=-0.4).
    rng = np.random.default_rng(1)
    z = -0.4 + rng.normal(0, 0.05, 8)
    m_tight, m_diffuse = _recover(z, r=0.03, m0=1.0)
    assert m_diffuse < m_tight - 0.3          # diffuse corrects downward
    assert m_diffuse < -0.1                    # toward the true -0.4
    assert m_tight > -0.1                       # tight prior stays high


def test_role_player_one_outlier_stays_shrunk():
    # seven ordinary seasons (m*=0) and one fluky elite year.
    z = np.array([0.0, 0.02, -0.03, 0.01, 0.0, 1.5, -0.02, 0.01])
    m_tight, m_diffuse = _recover(z, r=0.15, m0=0.0)
    # even the diffuse prior must not be dragged near the outlier (1.5).
    assert m_diffuse < 0.4
    assert m_tight < 0.3


def test_declining_veteran_state_tracks():
    # a real decline in theta; the filtered state must follow it down.
    z = np.array([1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0])
    theta_T, _ = _augmented_filter(z, r=0.05, m0=1.0, m_scale=1.0)
    assert theta_T < 0.35                       # tracks the decline, not stuck high


def test_report_table(capsys):
    """Not an assertion -- prints the recovery table for the session report."""
    scen = {
        "late-draft/8 elite (m*=+1.2, prior 0)": (1.2 + np.zeros(8), 0.03, 0.0, 1.2),
        "high-draft/8 poor (m*=-0.4, prior +1)": (-0.4 + np.zeros(8), 0.03, 1.0, -0.4),
        "role+1 outlier (m*=0)": (np.array([0, 0, 0, 0, 0, 1.5, 0, 0.0]), 0.15, 0.0, 0.0),
    }
    print("\n[S3-A recovery]  scenario | m_tight | m_diffuse | truth")
    for name, (z, r, m0, truth) in scen.items():
        mt, md = _recover(np.asarray(z, float), r, m0)
        print(f"  {name:40s} {mt:+.2f}    {md:+.2f}     {truth:+.2f}")
