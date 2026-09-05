"""Tests for the two-particle dataset generation (data/generate_particle_data.py)."""

import numpy as np

from data.generate_particle_data import (
    M1,
    M2,
    STATE_BOUNDS,
    generate_transition_pairs,
    sample_initial_state,
    split_dataset,
)
from simulator.two_particle_system import total_momentum


def test_sample_initial_state_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = sample_initial_state(rng)
        for value, (lo, hi) in zip(state, STATE_BOUNDS):
            assert lo - 1e-9 <= value <= hi + 1e-9


def test_sample_initial_state_respects_minimum_separation():
    """The polar sampling scheme must never place the particles on top of each other."""
    rng = np.random.default_rng(1)
    for _ in range(500):
        state = sample_initial_state(rng)
        d = np.hypot(state[4] - state[0], state[5] - state[1])
        assert d >= 0.6 - 1e-9


def test_generate_transition_pairs_shapes():
    pairs = generate_transition_pairs(
        n_trajectories=5, k=1.0, rest_length=1.0, m1=1.0, m2=1.0,
        dt=0.05, steps_per_trajectory=10, seed=0,
    )
    n_expected = 5 * 10
    assert pairs["states_t"].shape == (n_expected, 8)
    assert pairs["states_tp1"].shape == (n_expected, 8)


def test_generate_transition_pairs_no_nan_or_inf():
    pairs = generate_transition_pairs(
        n_trajectories=10, k=1.0, rest_length=1.0, m1=1.0, m2=1.0,
        dt=0.05, steps_per_trajectory=20, seed=123,
    )
    assert np.all(np.isfinite(pairs["states_t"]))
    assert np.all(np.isfinite(pairs["states_tp1"]))


def test_generate_transition_pairs_conserves_momentum():
    """The true simulator's generated transitions should conserve momentum almost exactly."""
    pairs = generate_transition_pairs(
        n_trajectories=20, k=1.0, rest_length=1.0, m1=1.0, m2=1.0,
        dt=0.05, steps_per_trajectory=20, seed=7,
    )
    px_t, py_t = total_momentum(pairs["states_t"], M1, M2)
    px_tp1, py_tp1 = total_momentum(pairs["states_tp1"], M1, M2)
    assert np.allclose(px_tp1, px_t, atol=1e-6)
    assert np.allclose(py_tp1, py_t, atol=1e-6)


def test_generate_transition_pairs_reproducible():
    p1 = generate_transition_pairs(
        n_trajectories=5, k=1.0, rest_length=1.0, m1=1.0, m2=1.0,
        dt=0.1, steps_per_trajectory=10, seed=42,
    )
    p2 = generate_transition_pairs(
        n_trajectories=5, k=1.0, rest_length=1.0, m1=1.0, m2=1.0,
        dt=0.1, steps_per_trajectory=10, seed=42,
    )
    assert np.allclose(p1["states_t"], p2["states_t"])


def test_split_dataset_fractions():
    states_t = np.zeros((1000, 8))
    states_tp1 = np.zeros((1000, 8))
    (train_x, _), (val_x, _), (test_x, _) = split_dataset(states_t, states_tp1, seed=1)
    assert train_x.shape[0] == 700
    assert val_x.shape[0] == 150
    assert test_x.shape[0] == 150
