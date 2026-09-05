"""
Simple harmonic oscillator simulator.

Physics
-------
A unit-mass point attached to an ideal, undamped spring obeys:

    dx/dt = v
    dv/dt = -k * x

where x is position, v is velocity, and k is the spring constant
(k = omega^2, with omega the angular frequency).

Total mechanical energy (unit mass):

    E = 0.5 * v^2 + 0.5 * k * x^2

is conserved exactly in continuous time. A numerical integrator only
conserves it approximately, so checking E(t) is a useful sanity check.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp


def equations_of_motion(t: float, state: np.ndarray, k: float) -> np.ndarray:
    """RHS of the ODE system: dx/dt = v, dv/dt = -k * x."""
    x, v = state
    return np.array([v, -k * x])


def simulate(
    x0: float,
    v0: float,
    k: float,
    t_span: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 200,
) -> dict:
    """
    Integrate the oscillator from initial state (x0, v0) using
    scipy.integrate.solve_ivp, evaluated at n_points evenly spaced times.

    Returns a dict with keys "t", "x", "v" (each a 1D array) and "k".
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        equations_of_motion,
        t_span,
        y0=[x0, v0],
        args=(k,),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    x, v = sol.y
    return {"t": sol.t, "x": x, "v": v, "k": k}


def total_energy(x: np.ndarray, v: np.ndarray, k: float) -> np.ndarray:
    """Total mechanical energy E = 0.5*v^2 + 0.5*k*x^2 (unit mass)."""
    return 0.5 * v**2 + 0.5 * k * x**2


def generate_transition_pairs(
    n_trajectories: int,
    k: float,
    dt: float,
    steps_per_trajectory: int,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    v_range: Tuple[float, float] = (-2.0, 2.0),
    seed: int = 42,
) -> dict:
    """
    Generate state-transition examples [x_t, v_t] -> [x_(t+1), v_(t+1)]
    for a fixed time step dt.

    For each of n_trajectories, samples a random initial state uniformly
    from x_range/v_range, integrates it for steps_per_trajectory steps of
    size dt, and records every consecutive pair of states.

    Returns a dict with:
        states_t, states_tp1 : (N, 2) arrays of [x, v]
        energy_t, energy_tp1 : (N,) arrays
        k, dt : scalars (metadata)
    """
    rng = np.random.default_rng(seed)
    states_t = []
    states_tp1 = []

    t_span = (0.0, dt * steps_per_trajectory)
    n_points = steps_per_trajectory + 1

    for _ in range(n_trajectories):
        x0 = rng.uniform(*x_range)
        v0 = rng.uniform(*v_range)
        result = simulate(x0, v0, k, t_span=t_span, n_points=n_points)
        traj = np.stack([result["x"], result["v"]], axis=1)
        states_t.append(traj[:-1])
        states_tp1.append(traj[1:])

    states_t = np.concatenate(states_t, axis=0)
    states_tp1 = np.concatenate(states_tp1, axis=0)

    return {
        "states_t": states_t,
        "states_tp1": states_tp1,
        "energy_t": total_energy(states_t[:, 0], states_t[:, 1], k),
        "energy_tp1": total_energy(states_tp1[:, 0], states_tp1[:, 1], k),
        "k": k,
        "dt": dt,
    }
