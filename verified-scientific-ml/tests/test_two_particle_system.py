"""Tests for the two-particle spring simulator and its momentum invariant."""

import numpy as np

from simulator.two_particle_system import simulate, spring_force, total_momentum


def test_spring_force_is_zero_at_rest_length():
    """No force should act when the particles are exactly rest_length apart."""
    state = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # separation = 1 = rest_length
    fx, fy = spring_force(state, k=1.0, rest_length=1.0)
    assert np.isclose(fx, 0.0, atol=1e-12)
    assert np.isclose(fy, 0.0, atol=1e-12)


def test_spring_force_is_restoring_when_stretched():
    """When stretched beyond rest length, the spring should pull particle 1 toward particle 2."""
    state = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])  # separation = 2 > rest_length = 1
    fx, fy = spring_force(state, k=1.0, rest_length=1.0)
    assert fx > 0.0  # pulled in +x direction, toward particle 2
    assert np.isclose(fy, 0.0, atol=1e-12)


def test_forces_are_equal_and_opposite():
    """Newton's third law: the force on particle 2 must be exactly -F1."""
    state = np.array([0.3, -0.1, 0.0, 0.0, 1.4, 0.8, 0.0, 0.0])
    fx1, fy1 = spring_force(state, k=1.0, rest_length=1.0)
    # equations_of_motion applies -fx/-fy to particle 2's acceleration; check that directly
    from simulator.two_particle_system import equations_of_motion
    deriv = equations_of_motion(0.0, state, k=1.0, rest_length=1.0, m1=1.0, m2=1.0)
    assert np.isclose(deriv[2], fx1)  # dvx1/dt = fx1/m1
    assert np.isclose(deriv[6], -fx1)  # dvx2/dt = -fx1/m2


def test_static_equilibrium_stays_at_rest():
    """Two particles at rest length apart with zero velocity should not move."""
    state0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    result = simulate(state0, k=1.0, rest_length=1.0, t_span=(0.0, 5.0), n_points=20)
    assert np.allclose(result["states"], state0, atol=1e-8)


def test_momentum_conservation_in_true_simulator():
    """The true simulator should conserve total momentum to near machine precision."""
    state0 = np.array([-0.5, 0.0, 0.4, 0.1, 0.5, 0.0, 0.1, -0.2])
    result = simulate(state0, k=1.0, rest_length=1.0, t_span=(0.0, 20.0), n_points=400)
    px, py = total_momentum(result["states"], m1=1.0, m2=1.0)
    assert np.allclose(px, px[0], atol=1e-8)
    assert np.allclose(py, py[0], atol=1e-8)


def test_total_momentum_batch_and_single():
    state = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.5, -1.0])
    px, py = total_momentum(state, m1=1.0, m2=1.0)
    assert np.isclose(px, 1.5)  # vx1 + vx2 = 1.0 + 0.5
    assert np.isclose(py, 1.0)  # vy1 + vy2 = 2.0 + (-1.0)

    batch = np.stack([state, state * 0.5])
    px_batch, py_batch = total_momentum(batch, m1=1.0, m2=1.0)
    assert px_batch.shape == (2,)
    assert np.isclose(px_batch[0], 1.5)


def test_simulate_no_nan_or_inf():
    state0 = np.array([0.2, -0.3, 0.5, -0.5, 1.1, 0.4, -0.2, 0.3])
    result = simulate(state0, k=1.0, rest_length=1.0, t_span=(0.0, 10.0), n_points=100)
    assert np.all(np.isfinite(result["states"]))
