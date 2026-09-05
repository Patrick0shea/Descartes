"""
Two-particle 2D spring system simulator.

Physics
-------
Two point particles of equal mass, connected by an ideal (Hookean) spring
with rest length L, moving in a 2D plane with no external forces.

State (length 8):
    [x1, y1, vx1, vy1, x2, y2, vx2, vy2]

Let r = (x2 - x1, y2 - y1) be the vector from particle 1 to particle 2,
and d = |r| the current separation. The spring exerts a restoring force
of magnitude k*(d - L) along r:

    F1 = k*(d - L) * (r / d)      (force on particle 1, pulling it toward
                                    particle 2 when stretched, d > L)
    F2 = -F1                      (Newton's third law: equal and opposite)

Equations of motion (mass m1 = m2 = 1 by default):
    dx1/dt = vx1
    dy1/dt = vy1
    dvx1/dt = F1x / m1
    dvy1/dt = F1y / m1
    dx2/dt = vx2
    dy2/dt = vy2
    dvx2/dt = F2x / m2
    dvy2/dt = F2y / m2

Because F1 = -F2 (no external forces), the total linear momentum

    Px = m1*vx1 + m2*vx2
    Py = m1*vy1 + m2*vy2

is conserved exactly in continuous time: d(Px)/dt = F1x + F2x = 0, and
likewise for Py. This is a strictly LINEAR conservation law (unlike
energy, which is quadratic in the state) -- that is what makes it
tractable for Marabou's piecewise-linear verification in Step 4.

As in Step 1, a numerical integrator only conserves momentum
approximately, so it is a useful sanity check on integration accuracy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

# State vector layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
X1, Y1, VX1, VY1, X2, Y2, VX2, VY2 = range(8)


def spring_force(state: np.ndarray, k: float, rest_length: float) -> Tuple[float, float]:
    """
    Force (F1x, F1y) exerted on particle 1 by the spring (force on
    particle 2 is the negative of this, by Newton's third law).
    """
    rx = state[X2] - state[X1]
    ry = state[Y2] - state[Y1]
    d = np.sqrt(rx**2 + ry**2)
    stretch = d - rest_length
    fx = k * stretch * (rx / d)
    fy = k * stretch * (ry / d)
    return fx, fy


def equations_of_motion(
    t: float, state: np.ndarray, k: float, rest_length: float, m1: float, m2: float
) -> np.ndarray:
    """RHS of the two-particle spring ODE system."""
    fx, fy = spring_force(state, k, rest_length)
    return np.array(
        [
            state[VX1],
            state[VY1],
            fx / m1,
            fy / m1,
            state[VX2],
            state[VY2],
            -fx / m2,
            -fy / m2,
        ]
    )


def simulate(
    state0,
    k: float = 1.0,
    rest_length: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
    t_span: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 200,
) -> dict:
    """
    Integrate the two-particle spring system from an initial state using
    scipy.integrate.solve_ivp, evaluated at n_points evenly spaced times.

    Returns a dict with keys "t" (shape (n_points,)), "states" (shape
    (n_points, 8)), and the physical parameters k, rest_length, m1, m2.
    """
    state0 = np.asarray(state0, dtype=np.float64)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        equations_of_motion,
        t_span,
        y0=state0,
        args=(k, rest_length, m1, m2),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return {
        "t": sol.t,
        "states": sol.y.T,  # (n_points, 8)
        "k": k,
        "rest_length": rest_length,
        "m1": m1,
        "m2": m2,
    }


def total_momentum(states: np.ndarray, m1: float = 1.0, m2: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Total linear momentum (Px, Py) = (m1*vx1 + m2*vx2, m1*vy1 + m2*vy2).

    Works on a single state (shape (8,)) or a batch of states
    (shape (N, 8)); returns scalars or shape-(N,) arrays accordingly.
    """
    states = np.asarray(states)
    vx1, vy1 = states[..., VX1], states[..., VY1]
    vx2, vy2 = states[..., VX2], states[..., VY2]
    px = m1 * vx1 + m2 * vx2
    py = m1 * vy1 + m2 * vy2
    return px, py
