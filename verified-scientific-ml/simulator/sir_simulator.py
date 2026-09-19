"""
SIR epidemiological model simulator.

Physics
-------
The standard compartmental SIR model (Susceptible-Infected-Recovered)
describes the spread of an infectious disease through a fixed population N.
Normalised by N (so s=S/N, i=I/N, r=R/N), the ODEs are:

    ds/dt = -beta * s * i
    di/dt =  beta * s * i - gamma * i
    dr/dt =  gamma * i

Because d(s+i+r)/dt = 0 exactly, the total normalised population

    P = s + i + r

is conserved to 1 in continuous time. This is a strictly LINEAR conservation
law in the state variables (unlike, say, energy) -- the same feature that
makes the two-particle momentum constraint tractable for Marabou. For a
discrete one-step surrogate, the question is whether the learned map
approximately or exactly preserves P = 1.

Parameters used throughout: beta=0.3, gamma=0.1 (R0 = beta/gamma = 3.0,
a realistic pandemic scenario). Time step DT = 1 day.

State layout: [s, i, r] (length 3)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

# Default epidemiological parameters
BETA: float = 0.3    # transmission rate (1/day)
GAMMA: float = 0.1   # recovery rate (1/day), R0 = BETA/GAMMA = 3.0

# State vector layout: [s, i, r]
S, I, R = 0, 1, 2


def sir_equations(t: float, state: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    RHS of the normalised SIR ODE system.

    Parameters
    ----------
    t : float
        Current time (not used -- the system is autonomous, but solve_ivp
        requires this signature).
    state : np.ndarray, shape (3,)
        [s, i, r] -- normalised compartment fractions.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.

    Returns
    -------
    np.ndarray, shape (3,)
        [ds/dt, di/dt, dr/dt]
    """
    s, i, r = state[S], state[I], state[R]
    ds_dt = -beta * s * i
    di_dt = beta * s * i - gamma * i
    dr_dt = gamma * i
    return np.array([ds_dt, di_dt, dr_dt])


def simulate_sir(
    state0,
    beta: float = BETA,
    gamma: float = GAMMA,
    t_span: Tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
) -> dict:
    """
    Integrate the SIR system from an initial state using
    scipy.integrate.solve_ivp, evaluated at n_points evenly spaced times.

    Parameters
    ----------
    state0 : array-like, shape (3,)
        Initial [s, i, r]. Should satisfy s+i+r ≈ 1 and all components >= 0.
    beta : float
        Transmission rate (default BETA=0.3).
    gamma : float
        Recovery rate (default GAMMA=0.1).
    t_span : tuple of float
        (t_start, t_end) for integration.
    n_points : int
        Number of evaluation points (including t_start).

    Returns
    -------
    dict with keys "t" (shape (n_points,)) and "states" (shape (n_points, 3)).
    """
    state0 = np.asarray(state0, dtype=np.float64)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        sir_equations,
        t_span,
        y0=state0,
        args=(beta, gamma),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    states = sol.y.T  # (n_points, 3)

    # Sanity check: conservation law s+i+r = 1 should hold to integrator tolerance.
    pop = total_population(states)
    max_drift = float(np.max(np.abs(pop - 1.0)))
    assert max_drift < 1e-6, (
        f"Population conservation violated: max |s+i+r-1| = {max_drift:.3e}"
    )

    return {"t": sol.t, "states": states}


def total_population(states: np.ndarray) -> np.ndarray:
    """
    Total normalised population P = s + i + r.

    Works on a single state (shape (3,)) or a batch (shape (N, 3)); returns
    a scalar or shape-(N,) array accordingly. Should be ≈ 1.0 everywhere.
    """
    states = np.asarray(states)
    return states[..., S] + states[..., I] + states[..., R]
