"""Tests for the harmonic oscillator simulator."""

import numpy as np

from simulator.harmonic_oscillator import (
    simulate,
    total_energy,
    generate_transition_pairs,
)


def test_known_solution_at_rest():
    """x0=0, v0=0 should stay at rest for all time."""
    result = simulate(x0=0.0, v0=0.0, k=1.0, t_span=(0.0, 10.0), n_points=50)
    assert np.allclose(result["x"], 0.0, atol=1e-8)
    assert np.allclose(result["v"], 0.0, atol=1e-8)


def test_known_solution_quarter_period():
    """For k=1, x0=1, v0=0: x(t) = cos(t). At t=pi/2, x~0 and v~-1."""
    result = simulate(x0=1.0, v0=0.0, k=1.0, t_span=(0.0, np.pi / 2), n_points=2)
    assert np.isclose(result["x"][-1], 0.0, atol=1e-4)
    assert np.isclose(result["v"][-1], -1.0, atol=1e-4)


def test_energy_conservation():
    result = simulate(x0=1.0, v0=0.5, k=2.0, t_span=(0.0, 20.0), n_points=500)
    energy = total_energy(result["x"], result["v"], result["k"])
    assert np.allclose(energy, energy[0], atol=1e-4)


def test_generate_transition_pairs_shapes():
    dataset = generate_transition_pairs(
        n_trajectories=10, k=1.0, dt=0.05, steps_per_trajectory=20, seed=0
    )
    n_expected = 10 * 20
    assert dataset["states_t"].shape == (n_expected, 2)
    assert dataset["states_tp1"].shape == (n_expected, 2)
    assert dataset["energy_t"].shape == (n_expected,)
    assert dataset["energy_tp1"].shape == (n_expected,)


def test_generate_transition_pairs_no_nan_or_inf():
    dataset = generate_transition_pairs(
        n_trajectories=20, k=1.5, dt=0.02, steps_per_trajectory=50, seed=123
    )
    for key in ("states_t", "states_tp1", "energy_t", "energy_tp1"):
        assert np.all(np.isfinite(dataset[key])), f"{key} contains NaN or inf"


def test_generate_transition_pairs_reproducible():
    d1 = generate_transition_pairs(
        n_trajectories=5, k=1.0, dt=0.1, steps_per_trajectory=10, seed=7
    )
    d2 = generate_transition_pairs(
        n_trajectories=5, k=1.0, dt=0.1, steps_per_trajectory=10, seed=7
    )
    assert np.allclose(d1["states_t"], d2["states_t"])
    assert np.allclose(d1["states_tp1"], d2["states_tp1"])
