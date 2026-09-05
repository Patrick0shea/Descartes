"""
Generate a reproducible harmonic-oscillator state-transition dataset
and save diagnostic plots.

Run from the verified-scientific-ml/ directory with:
    python -m data.generate_data
"""

from __future__ import annotations

import os

import numpy as np

from simulator.harmonic_oscillator import simulate, generate_transition_pairs
from simulator.plotting import plot_trajectory

SEED = 42
K = 1.0
DT = 0.05
STEPS_PER_TRAJECTORY = 200
N_TRAJECTORIES = 500
X_RANGE = (-2.0, 2.0)
V_RANGE = (-2.0, 2.0)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "harmonic_oscillator_dataset.npz")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    dataset = generate_transition_pairs(
        n_trajectories=N_TRAJECTORIES,
        k=K,
        dt=DT,
        steps_per_trajectory=STEPS_PER_TRAJECTORY,
        x_range=X_RANGE,
        v_range=V_RANGE,
        seed=SEED,
    )

    np.savez(
        OUTPUT_PATH,
        states_t=dataset["states_t"],
        states_tp1=dataset["states_tp1"],
        energy_t=dataset["energy_t"],
        energy_tp1=dataset["energy_tp1"],
        k=dataset["k"],
        dt=dataset["dt"],
        seed=SEED,
    )
    print(f"Saved {dataset['states_t'].shape[0]} transition pairs to {OUTPUT_PATH}")

    example = simulate(x0=1.5, v0=0.0, k=K, t_span=(0.0, 20.0), n_points=400)
    plot_path = os.path.join(PLOTS_DIR, "example_trajectory.png")
    plot_trajectory(example, save_path=plot_path)
    print(f"Saved example trajectory plot to {plot_path}")


if __name__ == "__main__":
    main()
