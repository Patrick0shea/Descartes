"""
Generate a reproducible dataset of state transitions for the SIR
epidemiological model (simulator/sir_simulator.py), split into
train/val/test, and save to data/sir_dataset.npz.

Sampling scheme (defines the verification domain)
--------------------------------------------------
Each initial state samples s and i independently from their ranges, then
sets r = 1 - s - i. This enforces the conservation law s+i+r = 1 EXACTLY
in all training data -- no integrator noise. Rejection sampling discards
states where r < 0.01 or r > 0.95 (biologically implausible extremes):

    s     ~ Uniform(S_RANGE)      = Uniform(0.10, 0.90)
    i     ~ Uniform(I_RANGE)      = Uniform(0.01, 0.50)
    r     = 1 - s - i             (exact conservation by construction)
    reject if r < 0.01 or r > 0.95

STATE_BOUNDS below gives the resulting per-dimension box used as the formal
verification domain in verification/sir_properties.py. The SEPARATION_BOUND
(=0.02) encodes the |s+i+r - 1| <= SEPARATION_BOUND constraint added to
Marabou queries, tightening the plain box to the conservation manifold
(analogous to the particle separation bound in generate_particle_data.py).

Run from the verified-scientific-ml/ directory:
    python -m data.generate_sir_data
"""

from __future__ import annotations

import os

import numpy as np

from simulator.sir_simulator import simulate_sir, total_population, BETA, GAMMA

SEED = 42
DT = 1.0                      # one-day time step
STEPS_PER_TRAJECTORY = 100
N_TRAJECTORIES = 1000

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15  # remaining 0.15 is test

# Sampling ranges that define the initial-condition distribution AND the
# formal verification domain (see module docstring).
S_RANGE = (0.10, 0.90)   # susceptible fraction
I_RANGE = (0.01, 0.50)   # infected fraction
# r is computed as 1 - s - i (not sampled independently)

# Per-dimension box for the formal verification domain.  Conservative
# margins around S_RANGE / I_RANGE; r bounds follow from the constraint.
STATE_BOUNDS = [
    (0.05, 0.95),   # s
    (0.01, 0.60),   # i
    (0.01, 0.90),   # r
]

# SEPARATION_BOUND: tolerance |s+i+r - 1| <= SEPARATION_BOUND added as a
# Marabou linear constraint to tighten the box to the conservation manifold.
SEPARATION_BOUND = 0.02

# State layout: [s, i, r]
STATE_NAMES = ["s", "i", "r"]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "sir_dataset.npz")


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    """
    Draw one valid initial [s, i, r] state.

    Samples s ~ Uniform(S_RANGE), i ~ Uniform(I_RANGE), sets r = 1-s-i.
    Rejects states where r < 0.01 or r > 0.95 (biologically implausible).
    """
    while True:
        s = rng.uniform(*S_RANGE)
        i = rng.uniform(*I_RANGE)
        r = 1.0 - s - i
        if 0.01 <= r <= 0.95:
            return np.array([s, i, r])


def generate_transition_pairs(
    n_trajectories: int,
    beta: float,
    gamma: float,
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
        result = simulate_sir(state0, beta=beta, gamma=gamma, t_span=t_span, n_points=n_points)
        traj = result["states"]
        states_t.append(traj[:-1])
        states_tp1.append(traj[1:])

    states_t = np.concatenate(states_t, axis=0)
    states_tp1 = np.concatenate(states_tp1, axis=0)
    return {"states_t": states_t, "states_tp1": states_tp1}


def split_dataset(states_t: np.ndarray, states_tp1: np.ndarray, seed: int):
    """Split arrays into (train, val, test) using 70/15/15 fractions."""
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
    pairs = generate_transition_pairs(
        n_trajectories=N_TRAJECTORIES,
        beta=BETA,
        gamma=GAMMA,
        dt=DT,
        steps_per_trajectory=STEPS_PER_TRAJECTORY,
        seed=SEED,
    )
    states_t, states_tp1 = pairs["states_t"], pairs["states_tp1"]

    # Sanity check: the true simulator should conserve population almost
    # exactly (up to integrator tolerance).
    pop_t = total_population(states_t)
    pop_tp1 = total_population(states_tp1)
    max_pop_drift = float(np.max(np.abs(pop_tp1 - pop_t)))
    print(f"Max |population drift| in true simulator data: {max_pop_drift:.3e}")

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_dataset(states_t, states_tp1, SEED)

    domain_lower = np.array([b[0] for b in STATE_BOUNDS])
    domain_upper = np.array([b[1] for b in STATE_BOUNDS])

    np.savez(
        OUTPUT_PATH,
        train_states_t=train_x, train_states_tp1=train_y,
        val_states_t=val_x, val_states_tp1=val_y,
        test_states_t=test_x, test_states_tp1=test_y,
        beta=BETA, gamma=GAMMA, dt=DT, seed=SEED,
        domain_lower=domain_lower, domain_upper=domain_upper,
        state_names=np.array(STATE_NAMES),
    )
    print(f"Saved {train_x.shape[0]} train / {val_x.shape[0]} val / {test_x.shape[0]} test pairs to {OUTPUT_PATH}")
    print("Domain (per state dimension):")
    for name, lo, hi in zip(STATE_NAMES, domain_lower, domain_upper):
        print(f"  {name}: [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
