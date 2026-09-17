"""
Compare the VGT-trained Model C vs standard-trained Model C.

Evaluates both models on the held-out test set and a 200-step autoregressive
rollout, reporting side-by-side metrics for:
  - Test MSE
  - Momentum conservation error (mean and max |ΔPx|, |ΔPy|)
  - Rollout momentum drift

Run from the verified-scientific-ml/ directory (after training/run_vgt.py):
    python -m training.evaluate_vgt
"""

from __future__ import annotations

import os

import numpy as np
import torch

from models.particle_soft_geometric import ParticleSoftGeometricMLP
from simulator.two_particle_system import simulate, total_momentum
from data.generate_particle_data import DT, K, REST_LENGTH, M1, M2

HIDDEN_DIM = 16
ROLLOUT_STEPS = 200

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
DATA_PATH = os.path.join(ROOT_DIR, "data", "particle_dataset.npz")
REPORT_PATH = os.path.join(CHECKPOINT_DIR, "vgt_evaluation_report.txt")

STANDARD_CKPT = os.path.join(CHECKPOINT_DIR, "particle_soft_geometric.pt")
VGT_CKPT = os.path.join(CHECKPOINT_DIR, "particle_soft_geometric_vgt.pt")


def _load_model(path: str) -> ParticleSoftGeometricMLP:
    state_dict = torch.load(path, map_location="cpu")
    hidden_dim = state_dict["delta_net.0.weight"].shape[0]
    model = ParticleSoftGeometricMLP(hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict(model: ParticleSoftGeometricMLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x.astype(np.float32))).numpy()


def _evaluate_test_set(model, test_x, test_y):
    preds = _predict(model, test_x)
    mse = float(np.mean((preds - test_y) ** 2))
    mae = float(np.mean(np.abs(preds - test_y)))

    # Momentum errors
    dpx = (preds[:, 2] + preds[:, 6]) - (test_x[:, 2] + test_x[:, 6])
    dpy = (preds[:, 3] + preds[:, 7]) - (test_x[:, 3] + test_x[:, 7])
    mean_dpx = float(np.mean(np.abs(dpx)))
    mean_dpy = float(np.mean(np.abs(dpy)))
    max_dpx = float(np.max(np.abs(dpx)))
    max_dpy = float(np.max(np.abs(dpy)))

    return {
        "mse": mse, "mae": mae,
        "mean_abs_dpx": mean_dpx, "mean_abs_dpy": mean_dpy,
        "max_abs_dpx": max_dpx, "max_abs_dpy": max_dpy,
    }


def _rollout(model, state0: np.ndarray, steps: int):
    state = state0.copy().astype(np.float32)
    states = [state]
    for _ in range(steps):
        state = _predict(model, state[None])[0]
        states.append(state)
    return np.array(states)


def main() -> None:
    if not os.path.exists(STANDARD_CKPT):
        raise FileNotFoundError(f"Standard Model C not found: {STANDARD_CKPT}")
    if not os.path.exists(VGT_CKPT):
        raise FileNotFoundError(
            f"VGT Model C not found: {VGT_CKPT}. Run `python -m training.run_vgt` first."
        )

    data = np.load(DATA_PATH)
    test_x = data["test_states_t"].astype(np.float32)
    test_y = data["test_states_tp1"].astype(np.float32)

    standard = _load_model(STANDARD_CKPT)
    vgt = _load_model(VGT_CKPT)

    std_metrics = _evaluate_test_set(standard, test_x, test_y)
    vgt_metrics = _evaluate_test_set(vgt, test_x, test_y)

    # 200-step rollout from fixed initial state (same as evaluate_soft_geometric.py)
    state0 = np.array([-0.3, 0.2, 0.5, -0.3, 0.5, -0.4, -0.2, 0.4], dtype=np.float32)

    std_rollout = _rollout(standard, state0, ROLLOUT_STEPS)
    vgt_rollout = _rollout(vgt, state0, ROLLOUT_STEPS)

    # True simulator rollout for reference
    true_result = simulate(
        state0.astype(np.float64), k=K, rest_length=REST_LENGTH, m1=M1, m2=M2,
        t_span=(0.0, DT * ROLLOUT_STEPS), n_points=ROLLOUT_STEPS + 1,
    )
    true_states = true_result["states"].astype(np.float32)

    px0_true, py0_true = float(state0[2] + state0[6]), float(state0[3] + state0[7])

    def _drift(states):
        dpx = abs((states[-1, 2] + states[-1, 6]) - px0_true)
        dpy = abs((states[-1, 3] + states[-1, 7]) - py0_true)
        return dpx, dpy

    std_dpx, std_dpy = _drift(std_rollout)
    vgt_dpx, vgt_dpy = _drift(vgt_rollout)

    lines = [
        "=== VGT vs Standard Model C: Evaluation Report ===\n",
        f"Test set: {test_x.shape[0]} examples\n",
        f"Rollout: {ROLLOUT_STEPS} steps from fixed initial state\n",
        "",
        f"{'Metric':<40} {'Standard':>14} {'VGT':>14}",
        "-" * 70,
        f"{'Test MSE':<40} {std_metrics['mse']:>14.3e} {vgt_metrics['mse']:>14.3e}",
        f"{'Test MAE':<40} {std_metrics['mae']:>14.3e} {vgt_metrics['mae']:>14.3e}",
        f"{'Mean |dPx| (test set)':<40} {std_metrics['mean_abs_dpx']:>14.3e} {vgt_metrics['mean_abs_dpx']:>14.3e}",
        f"{'Mean |dPy| (test set)':<40} {std_metrics['mean_abs_dpy']:>14.3e} {vgt_metrics['mean_abs_dpy']:>14.3e}",
        f"{'Max |dPx| (test set)':<40} {std_metrics['max_abs_dpx']:>14.3e} {vgt_metrics['max_abs_dpx']:>14.3e}",
        f"{'Max |dPy| (test set)':<40} {std_metrics['max_abs_dpy']:>14.3e} {vgt_metrics['max_abs_dpy']:>14.3e}",
        f"{'Rollout final |dPx| (200 steps)':<40} {std_dpx:>14.3e} {vgt_dpx:>14.3e}",
        f"{'Rollout final |dPy| (200 steps)':<40} {std_dpy:>14.3e} {vgt_dpy:>14.3e}",
        "-" * 70,
        "",
        "Formal verification (Marabou epsilon sweep):",
        "  Standard Model C: UNSAT at epsilon=1e-2, SAT at epsilon=1e-3",
        "  VGT Model C:      see verification/artifacts/vgt_results.json",
        "",
    ]

    report = "\n".join(lines)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
