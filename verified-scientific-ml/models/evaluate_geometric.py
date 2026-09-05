"""
Evaluate the trained geometric surrogate (Step 5): test-set metrics,
momentum-conservation analysis, loss curve, and multi-step rollout vs.
the true simulator -- the exact same analysis models/evaluate_particle.py
runs for the plain MLP baseline, so the two are directly comparable.

Run from the verified-scientific-ml/ directory, after
models/train_geometric.py:
    python -m models.evaluate_geometric
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.particle_geometric import ParticleGeometricMLP
from simulator.two_particle_system import simulate, total_momentum

HIDDEN_DIM = 16  # must match models/train_geometric.py

# Same rollout initial condition as models/evaluate_particle.py, for a
# like-for-like comparison.
ROLLOUT_STATE0 = np.array([-0.5, 0.0, 0.4, 0.1, 0.5, 0.0, 0.1, -0.2])
ROLLOUT_STEPS = 200

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODELS_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "particle_dataset.npz")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")

TOLERANCES = [1e-3, 1e-4, 1e-5]


def load_artifacts():
    model = ParticleGeometricMLP(hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "particle_geometric.pt")))
    model.eval()

    loss_history = np.load(os.path.join(CHECKPOINT_DIR, "particle_geometric_loss_history.npz"))
    dataset = np.load(DATA_PATH)

    return model, loss_history, dataset


def predict(model: ParticleGeometricMLP, x: np.ndarray) -> np.ndarray:
    """Predict next physical state(s) from physical state(s) x (shape (N, 8)). No normalization."""
    with torch.no_grad():
        return model(torch.from_numpy(x.astype(np.float32))).numpy()


def compute_test_metrics(model, test_x, test_y) -> dict:
    pred_y = predict(model, test_x)
    error = pred_y - test_y
    return {
        "mse_overall": float(np.mean(error**2)),
        "mae_overall": float(np.mean(np.abs(error))),
        "max_err_overall": float(np.max(np.abs(error))),
        "pred_y": pred_y,
        "error": error,
    }


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean_abs": float(np.mean(np.abs(arr))),
        "max_abs": float(np.max(np.abs(arr))),
        "std": float(np.std(arr)),
    }


def momentum_analysis(test_x: np.ndarray, pred_y: np.ndarray, m1: float, m2: float) -> dict:
    px_before, py_before = total_momentum(test_x, m1, m2)
    px_after, py_after = total_momentum(pred_y, m1, m2)
    delta_px = px_after - px_before
    delta_py = py_after - py_before
    magnitude = np.sqrt(delta_px**2 + delta_py**2)

    pct_within_tol = {
        tol: {
            "px": float(np.mean(np.abs(delta_px) <= tol) * 100),
            "py": float(np.mean(np.abs(delta_py) <= tol) * 100),
            "magnitude": float(np.mean(magnitude <= tol) * 100),
        }
        for tol in TOLERANCES
    }

    return {
        "delta_px": _stats(delta_px),
        "delta_py": _stats(delta_py),
        "magnitude": _stats(magnitude),
        "pct_within_tol": pct_within_tol,
        "delta_px_arr": delta_px,
        "delta_py_arr": delta_py,
    }


def plot_loss_curve(train_losses: np.ndarray, val_losses: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train loss")
    ax.plot(epochs, val_losses, label="val loss")
    ax.set(xlabel="epoch", ylabel="MSE (physical units)", yscale="log",
           title="Geometric surrogate: training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "geometric_loss_curve.png"), dpi=150)
    plt.close(fig)


def plot_momentum_error_hist(delta_px: np.ndarray, delta_py: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(delta_px, bins=60, color="tab:green", alpha=0.8)
    axes[0].set(xlabel="delta_Px (predicted)", ylabel="count", title="Geometric: momentum error x")
    axes[1].hist(delta_py, bins=60, color="tab:red", alpha=0.8)
    axes[1].set(xlabel="delta_Py (predicted)", ylabel="count", title="Geometric: momentum error y")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "geometric_momentum_error_hist.png"), dpi=150)
    plt.close(fig)


def rollout_nn(model, state0: np.ndarray, n_steps: int) -> np.ndarray:
    states = np.zeros((n_steps + 1, 8), dtype=np.float64)
    states[0] = state0
    for i in range(n_steps):
        states[i + 1] = predict(model, states[i:i + 1])[0]
    return states


def plot_rollout_trajectories(true_states: np.ndarray, nn_states: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(true_states[:, 0], true_states[:, 1], label="particle 1 (true)")
    ax.plot(true_states[:, 4], true_states[:, 5], label="particle 2 (true)")
    ax.plot(nn_states[:, 0], nn_states[:, 1], "--", label="particle 1 (geometric NN)")
    ax.plot(nn_states[:, 4], nn_states[:, 5], "--", label="particle 2 (geometric NN)")
    ax.set(xlabel="x", ylabel="y", title="Rollout: particle paths, geometric NN vs true simulator")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "geometric_rollout_trajectories.png"), dpi=150)
    plt.close(fig)


def plot_momentum_drift(t, px_true, py_true, px_nn, py_nn) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(t, px_true, label="true (simulator)")
    axes[0].plot(t, px_nn, "--", label="geometric NN rollout")
    axes[0].set(xlabel="time", ylabel="Px", title="Total momentum (x): geometric NN vs true")
    axes[0].legend()

    axes[1].plot(t, py_true, label="true (simulator)")
    axes[1].plot(t, py_nn, "--", label="geometric NN rollout")
    axes[1].set(xlabel="time", ylabel="Py", title="Total momentum (y): geometric NN vs true")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "geometric_momentum_drift.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model, loss_history, dataset = load_artifacts()

    test_x = dataset["test_states_t"]
    test_y = dataset["test_states_tp1"]
    k, rest_length = float(dataset["k"]), float(dataset["rest_length"])
    m1, m2, dt = float(dataset["m1"]), float(dataset["m2"]), float(dataset["dt"])

    metrics = compute_test_metrics(model, test_x, test_y)

    momentum = momentum_analysis(test_x, metrics["pred_y"], m1, m2)
    plot_momentum_error_hist(momentum["delta_px_arr"], momentum["delta_py_arr"])

    plot_loss_curve(loss_history["train_losses"], loss_history["val_losses"])

    true_result = simulate(
        ROLLOUT_STATE0, k=k, rest_length=rest_length, m1=m1, m2=m2,
        t_span=(0.0, dt * ROLLOUT_STEPS), n_points=ROLLOUT_STEPS + 1,
    )
    t, true_states = true_result["t"], true_result["states"]
    nn_states = rollout_nn(model, ROLLOUT_STATE0, ROLLOUT_STEPS)

    plot_rollout_trajectories(true_states, nn_states)

    px_true, py_true = total_momentum(true_states, m1, m2)
    px_nn, py_nn = total_momentum(nn_states, m1, m2)
    plot_momentum_drift(t, px_true, py_true, px_nn, py_nn)

    rollout_final_drift = (float(abs(px_nn[-1] - px_true[0])), float(abs(py_nn[-1] - py_true[0])))
    rollout_max_drift = (float(np.max(np.abs(px_nn - px_true[0]))), float(np.max(np.abs(py_nn - py_true[0]))))

    lines = [
        "=== Test-set state-prediction metrics (physical units, all 8 dims) ===",
        f"MSE overall: {metrics['mse_overall']:.6e}",
        f"MAE overall: {metrics['mae_overall']:.6e}",
        f"Max error overall: {metrics['max_err_overall']:.6e}",
        "",
        "=== Momentum-conservation analysis (test set, N={}) ===".format(test_x.shape[0]),
        f"delta_Px: mean|.|={momentum['delta_px']['mean_abs']:.6e}  max|.|={momentum['delta_px']['max_abs']:.6e}  std={momentum['delta_px']['std']:.6e}",
        f"delta_Py: mean|.|={momentum['delta_py']['mean_abs']:.6e}  max|.|={momentum['delta_py']['max_abs']:.6e}  std={momentum['delta_py']['std']:.6e}",
        f"magnitude sqrt(dPx^2+dPy^2): mean={momentum['magnitude']['mean_abs']:.6e}  max={momentum['magnitude']['max_abs']:.6e}  std={momentum['magnitude']['std']:.6e}",
        "",
        "Percentage of test examples within tolerance:",
    ]
    for tol, pct in momentum["pct_within_tol"].items():
        lines.append(f"  tol={tol:.0e}:  Px within: {pct['px']:.2f}%   Py within: {pct['py']:.2f}%   magnitude within: {pct['magnitude']:.2f}%")

    lines += [
        "",
        f"=== Multi-step rollout ({ROLLOUT_STEPS} steps, dt={dt}) from state0={ROLLOUT_STATE0.tolist()} ===",
        f"True initial momentum: Px0={px_true[0]:.6f}, Py0={py_true[0]:.6f}",
        f"Final |momentum drift| geometric NN rollout: dPx={rollout_final_drift[0]:.6e}, dPy={rollout_final_drift[1]:.6e}",
        f"Max |momentum drift| over rollout, geometric NN: dPx={rollout_max_drift[0]:.6e}, dPy={rollout_max_drift[1]:.6e}",
        f"Max |momentum drift| over rollout, true simulator: "
        f"dPx={float(np.max(np.abs(px_true - px_true[0]))):.3e}, dPy={float(np.max(np.abs(py_true - py_true[0]))):.3e}",
    ]

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(CHECKPOINT_DIR, "particle_geometric_evaluation_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
