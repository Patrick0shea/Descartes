"""
Generate a reproducible dataset of state transitions for the two-particle
spring system (simulator/two_particle_system.py), split into
train/val/test, and save diagnostic plots.

Sampling scheme (defines the verification domain)
---------------------------------------------------
Each initial state samples particle 1's position uniformly, then places
particle 2 at a random separation d0 and angle theta around it -- this
guarantees, BY CONSTRUCTION, a minimum particle separation (no
division-by-zero singularities in the spring force), unlike sampling all
four position coordinates independently and hoping they don't collide:

    x1, y1        ~ Uniform(X1_RANGE)
    theta         ~ Uniform(0, 2*pi)
    d0            ~ Uniform(SEPARATION_RANGE)      (SEPARATION_RANGE brackets
                                                     the rest length L=1, so
                                                     both stretched and
                                                     compressed springs occur)
    x2 = x1 + d0*cos(theta), y2 = y1 + d0*sin(theta)
    vx1, vy1, vx2, vy2 ~ Uniform(V_RANGE), independently

STATE_BOUNDS below gives the resulting exact per-dimension box each state
coordinate is drawn from (x1, y1, vx1, vy1 directly from their ranges;
x2, y2 conservatively bounded by X1/Y1_RANGE +/- max separation). This
box is the domain later used, unchanged, as the formal verification
domain in verification/particle_properties.py -- it is not a separately
chosen or widened region, it is exactly what this script samples from.

Run from the verified-scientific-ml/ directory:
    python -m data.generate_particle_data
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from simulator.two_particle_system import simulate, total_momentum

SEED = 42
K = 1.0
REST_LENGTH = 1.0
M1 = 1.0
M2 = 1.0
DT = 0.05
STEPS_PER_TRAJECTORY = 100
N_TRAJECTORIES = 1000

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15  # remaining 0.15 is test

# Sampling ranges that define the initial-condition distribution AND the
# formal verification domain (see module docstring).
X1_RANGE = (-1.0, 1.0)
Y1_RANGE = (-1.0, 1.0)
SEPARATION_RANGE = (0.6, 1.4)  # brackets rest_length = 1.0
V_RANGE = (-1.0, 1.0)  # applied independently to vx1, vy1, vx2, vy2

_max_sep = SEPARATION_RANGE[1]
X2_RANGE = (X1_RANGE[0] - _max_sep, X1_RANGE[1] + _max_sep)
Y2_RANGE = (Y1_RANGE[0] - _max_sep, Y1_RANGE[1] + _max_sep)

# State layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
STATE_NAMES = ["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2"]
STATE_BOUNDS = [X1_RANGE, Y1_RANGE, V_RANGE, V_RANGE, X2_RANGE, Y2_RANGE, V_RANGE, V_RANGE]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "particle_dataset.npz")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    """Draw one initial state using the sampling scheme documented above."""
    x1 = rng.uniform(*X1_RANGE)
    y1 = rng.uniform(*Y1_RANGE)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    d0 = rng.uniform(*SEPARATION_RANGE)
    x2 = x1 + d0 * np.cos(theta)
    y2 = y1 + d0 * np.sin(theta)
    vx1 = rng.uniform(*V_RANGE)
    vy1 = rng.uniform(*V_RANGE)
    vx2 = rng.uniform(*V_RANGE)
    vy2 = rng.uniform(*V_RANGE)
    return np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])


def generate_transition_pairs(
    n_trajectories: int,
    k: float,
    rest_length: float,
    m1: float,
    m2: float,
    dt: float,
    steps_per_trajectory: int,
    seed: int,
) -> dict:
    """
    Generate state-transition examples state_t -> state_(t+1) for a fixed
    time step dt, by sampling n_trajectories random initial states and
    integrating each for steps_per_trajectory steps.
    """
    rng = np.random.default_rng(seed)
    states_t = []
    states_tp1 = []

    t_span = (0.0, dt * steps_per_trajectory)
    n_points = steps_per_trajectory + 1

    for _ in range(n_trajectories):
        state0 = sample_initial_state(rng)
        result = simulate(state0, k=k, rest_length=rest_length, m1=m1, m2=m2, t_span=t_span, n_points=n_points)
        traj = result["states"]
        states_t.append(traj[:-1])
        states_tp1.append(traj[1:])

    states_t = np.concatenate(states_t, axis=0)
    states_tp1 = np.concatenate(states_tp1, axis=0)
    return {"states_t": states_t, "states_tp1": states_tp1}


def split_dataset(states_t: np.ndarray, states_tp1: np.ndarray, seed: int):
    n = states_t.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (
        (states_t[train_idx], states_tp1[train_idx]),
        (states_t[val_idx], states_tp1[val_idx]),
        (states_t[test_idx], states_tp1[test_idx]),
    )


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    pairs = generate_transition_pairs(
        n_trajectories=N_TRAJECTORIES,
        k=K,
        rest_length=REST_LENGTH,
        m1=M1,
        m2=M2,
        dt=DT,
        steps_per_trajectory=STEPS_PER_TRAJECTORY,
        seed=SEED,
    )
    states_t, states_tp1 = pairs["states_t"], pairs["states_tp1"]

    # Sanity check: the true simulator should conserve momentum almost
    # exactly (up to integrator tolerance).
    px_t, py_t = total_momentum(states_t, M1, M2)
    px_tp1, py_tp1 = total_momentum(states_tp1, M1, M2)
    max_momentum_drift = float(np.max(np.abs(np.concatenate([px_tp1 - px_t, py_tp1 - py_t]))))
    print(f"Max |momentum drift| in true simulator data: {max_momentum_drift:.3e}")

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_dataset(states_t, states_tp1, SEED)

    domain_lower = np.array([b[0] for b in STATE_BOUNDS])
    domain_upper = np.array([b[1] for b in STATE_BOUNDS])

    np.savez(
        OUTPUT_PATH,
        train_states_t=train_x, train_states_tp1=train_y,
        val_states_t=val_x, val_states_tp1=val_y,
        test_states_t=test_x, test_states_tp1=test_y,
        k=K, rest_length=REST_LENGTH, m1=M1, m2=M2, dt=DT, seed=SEED,
        domain_lower=domain_lower, domain_upper=domain_upper,
        state_names=np.array(STATE_NAMES),
    )
    print(f"Saved {train_x.shape[0]} train / {val_x.shape[0]} val / {test_x.shape[0]} test pairs to {OUTPUT_PATH}")
    print("Domain (per state dimension):")
    for name, lo, hi in zip(STATE_NAMES, domain_lower, domain_upper):
        print(f"  {name}: [{lo:.2f}, {hi:.2f}]")

    plot_example_trajectory()


def plot_example_trajectory() -> None:
    """Sanity-check plot: particle paths and momentum conservation for one trajectory."""
    state0 = np.array([-0.5, 0.0, 0.4, 0.1, 0.5, 0.0, 0.1, -0.2])
    result = simulate(state0, k=K, rest_length=REST_LENGTH, m1=M1, m2=M2, t_span=(0.0, 20.0), n_points=400)
    t, states = result["t"], result["states"]
    px, py = total_momentum(states, M1, M2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(states[:, 0], states[:, 1], label="particle 1")
    axes[0].plot(states[:, 4], states[:, 5], label="particle 2")
    axes[0].set(xlabel="x", ylabel="y", title="Particle paths (2D)")
    axes[0].legend()
    axes[0].set_aspect("equal")

    axes[1].plot(t, px, label="Px")
    axes[1].plot(t, py, label="Py")
    axes[1].set(xlabel="time", ylabel="momentum", title="Total momentum vs time (true simulator)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "particle_example_trajectory.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
