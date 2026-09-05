"""Plotting helpers for a single simulated trajectory."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from .harmonic_oscillator import total_energy


def plot_trajectory(result: dict, save_path: Optional[str] = None) -> None:
    """
    Plot position vs time, velocity vs time, phase space (x vs v),
    and energy vs time for a trajectory from simulate().
    """
    t, x, v, k = result["t"], result["x"], result["v"], result["k"]
    energy = total_energy(x, v, k)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(t, x)
    axes[0, 0].set(xlabel="time", ylabel="position x", title="Position vs time")

    axes[0, 1].plot(t, v, color="tab:orange")
    axes[0, 1].set(xlabel="time", ylabel="velocity v", title="Velocity vs time")

    axes[1, 0].plot(x, v, color="tab:green")
    axes[1, 0].set(xlabel="position x", ylabel="velocity v", title="Phase space (x vs v)")

    axes[1, 1].plot(t, energy, color="tab:red")
    axes[1, 1].set(xlabel="time", ylabel="energy E", title="Energy vs time")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
