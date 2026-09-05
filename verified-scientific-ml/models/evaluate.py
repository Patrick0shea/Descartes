"""
Evaluate the trained MLP surrogate:
    - test-set metrics (MSE, MAE, max error)
    - predicted-vs-true and error plots
    - loss-vs-epoch plot
    - multi-step autoregressive rollout vs. the true numerical simulator
    - energy comparison between the NN rollout and the true trajectory

Run from the verified-scientific-ml/ directory, after models/train.py:
    python -m models.evaluate
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.normalization import Normalizer
from models.surrogate import MLPSurrogate
from simulator.harmonic_oscillator import simulate, total_energy

HIDDEN_DIM = 64  # must match models/train.py

# Rollout configuration
ROLLOUT_X0 = 1.5
ROLLOUT_V0 = 0.0
ROLLOUT_STEPS = 200

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")


def load_artifacts():
    normalizer = Normalizer.load(os.path.join(CHECKPOINT_DIR, "normalizer.npz"))

    model = MLPSurrogate(hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "surrogate.pt")))
    model.eval()

    splits = np.load(os.path.join(CHECKPOINT_DIR, "splits.npz"))
    loss_history = np.load(os.path.join(CHECKPOINT_DIR, "loss_history.npz"))

    return model, normalizer, splits, loss_history


def predict(model: MLPSurrogate, normalizer: Normalizer, x: np.ndarray) -> np.ndarray:
    """Predict next physical state(s) [x, v] from physical state(s) x (shape (N, 2))."""
    x_n = normalizer.transform(x).astype(np.float32)
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_n)).numpy()
    return normalizer.inverse_transform(pred_n)


def compute_test_metrics(model, normalizer, test_x, test_y) -> dict:
    pred_y = predict(model, normalizer, test_x)
    error = pred_y - test_y  # (N, 2), columns [x, v]

    mse = np.mean(error**2, axis=0)
    mae = np.mean(np.abs(error), axis=0)
    max_err = np.max(np.abs(error), axis=0)

    return {
        "mse_x": mse[0], "mse_v": mse[1], "mse_overall": np.mean(error**2),
        "mae_x": mae[0], "mae_v": mae[1], "mae_overall": np.mean(np.abs(error)),
        "max_err_x": max_err[0], "max_err_v": max_err[1],
        "max_err_overall": np.max(np.abs(error)),
        "pred_y": pred_y,
        "error": error,
    }


def plot_predictions(test_y: np.ndarray, pred_y: np.ndarray, error: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].scatter(test_y[:, 0], pred_y[:, 0], s=2, alpha=0.3)
    lims = [test_y[:, 0].min(), test_y[:, 0].max()]
    axes[0, 0].plot(lims, lims, "r--", linewidth=1)
    axes[0, 0].set(xlabel="true x_next", ylabel="predicted x_next",
                   title="Predicted vs true x_next")

    axes[0, 1].scatter(test_y[:, 1], pred_y[:, 1], s=2, alpha=0.3, color="tab:orange")
    lims = [test_y[:, 1].min(), test_y[:, 1].max()]
    axes[0, 1].plot(lims, lims, "r--", linewidth=1)
    axes[0, 1].set(xlabel="true v_next", ylabel="predicted v_next",
                   title="Predicted vs true v_next")

    axes[1, 0].hist(error[:, 0], bins=60, color="tab:green", alpha=0.8)
    axes[1, 0].set(xlabel="prediction error (x_next)", ylabel="count",
                   title="Position prediction error")

    axes[1, 1].hist(error[:, 1], bins=60, color="tab:red", alpha=0.8)
    axes[1, 1].set(xlabel="prediction error (v_next)", ylabel="count",
                   title="Velocity prediction error")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "test_predictions.png"), dpi=150)
    plt.close(fig)


def plot_loss_curve(train_losses: np.ndarray, val_losses: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train loss")
    ax.plot(epochs, val_losses, label="val loss")
    ax.set(xlabel="epoch", ylabel="MSE (normalized)", yscale="log",
           title="Training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)


def rollout_nn(model, normalizer, x0: float, v0: float, n_steps: int) -> np.ndarray:
    """Autoregressively roll out the NN: feed each prediction back as the next input."""
    states = np.zeros((n_steps + 1, 2), dtype=np.float64)
    states[0] = [x0, v0]
    for i in range(n_steps):
        states[i + 1] = predict(model, normalizer, states[i:i + 1])[0]
    return states


def plot_rollout_comparison(t, true_x, true_v, nn_x, nn_v) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(t, true_x, label="true (simulator)")
    axes[0, 0].plot(t, nn_x, "--", label="NN rollout")
    axes[0, 0].set(xlabel="time", ylabel="position x", title="Position: NN rollout vs true")
    axes[0, 0].legend()

    axes[0, 1].plot(t, true_v, label="true (simulator)")
    axes[0, 1].plot(t, nn_v, "--", label="NN rollout")
    axes[0, 1].set(xlabel="time", ylabel="velocity v", title="Velocity: NN rollout vs true")
    axes[0, 1].legend()

    axes[1, 0].plot(true_x, true_v, label="true (simulator)")
    axes[1, 0].plot(nn_x, nn_v, "--", label="NN rollout")
    axes[1, 0].set(xlabel="position x", ylabel="velocity v", title="Phase space: NN rollout vs true")
    axes[1, 0].legend()

    pos_error = np.abs(nn_x - true_x)
    axes[1, 1].plot(t, pos_error, color="tab:red")
    axes[1, 1].set(xlabel="time", ylabel="|x_nn - x_true|", title="Rollout position error growth")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rollout_comparison.png"), dpi=150)
    plt.close(fig)


def plot_energy_comparison(t, energy_true, energy_nn) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, energy_true, label="true (simulator) energy")
    ax.plot(t, energy_nn, "--", label="NN rollout energy")
    ax.set(xlabel="time", ylabel="energy E", title="Energy: NN rollout vs true simulator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "energy_comparison.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model, normalizer, splits, loss_history = load_artifacts()

    test_x, test_y = splits["test_x"], splits["test_y"]
    k, dt = float(splits["k"]), float(splits["dt"])

    # --- Test-set metrics ---
    metrics = compute_test_metrics(model, normalizer, test_x, test_y)
    plot_predictions(test_y, metrics["pred_y"], metrics["error"])

    # --- Loss curve ---
    plot_loss_curve(loss_history["train_losses"], loss_history["val_losses"])

    # --- Multi-step rollout vs. true simulator ---
    true_result = simulate(
        x0=ROLLOUT_X0, v0=ROLLOUT_V0, k=k,
        t_span=(0.0, dt * ROLLOUT_STEPS), n_points=ROLLOUT_STEPS + 1,
    )
    t, true_x, true_v = true_result["t"], true_result["x"], true_result["v"]

    nn_states = rollout_nn(model, normalizer, ROLLOUT_X0, ROLLOUT_V0, ROLLOUT_STEPS)
    nn_x, nn_v = nn_states[:, 0], nn_states[:, 1]

    plot_rollout_comparison(t, true_x, true_v, nn_x, nn_v)

    energy_true = total_energy(true_x, true_v, k)
    energy_nn = total_energy(nn_x, nn_v, k)
    plot_energy_comparison(t, energy_true, energy_nn)

    rollout_final_pos_error = float(np.abs(nn_x[-1] - true_x[-1]))
    rollout_max_pos_error = float(np.max(np.abs(nn_x - true_x)))
    energy_drift_true = float(energy_true[-1] - energy_true[0])
    energy_drift_nn = float(energy_nn[-1] - energy_nn[0])

    # --- Report ---
    report_lines = [
        "=== Test-set metrics (physical units) ===",
        f"MSE  x_next: {metrics['mse_x']:.6e}   v_next: {metrics['mse_v']:.6e}   overall: {metrics['mse_overall']:.6e}",
        f"MAE  x_next: {metrics['mae_x']:.6e}   v_next: {metrics['mae_v']:.6e}   overall: {metrics['mae_overall']:.6e}",
        f"Max error x_next: {metrics['max_err_x']:.6e}   v_next: {metrics['max_err_v']:.6e}   overall: {metrics['max_err_overall']:.6e}",
        "",
        f"=== Multi-step rollout ({ROLLOUT_STEPS} steps, dt={dt}) from x0={ROLLOUT_X0}, v0={ROLLOUT_V0} ===",
        f"Final position error |x_nn - x_true|: {rollout_final_pos_error:.6e}",
        f"Max position error over rollout: {rollout_max_pos_error:.6e}",
        "",
        "=== Energy drift over rollout ===",
        f"True simulator energy drift (E_final - E_initial): {energy_drift_true:.6e}",
        f"NN rollout energy drift (E_final - E_initial): {energy_drift_nn:.6e}",
    ]
    report = "\n".join(report_lines)
    print(report)

    with open(os.path.join(CHECKPOINT_DIR, "evaluation_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
