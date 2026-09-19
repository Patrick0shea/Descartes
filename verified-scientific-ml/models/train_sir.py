"""
Train all three SIR surrogate models (A: plain MLP, B: geometric/exact-
conservation, C: soft-geometric) on the same SIR dataset for direct
comparison.

All three learn the same one-step map s_t -> s_{t+1} with the same
train/val/test split, same optimizer/loss/seed/epoch budget. The only
differences are the architectures:
  - Model A (SirMLP): unconstrained 3->16->16->3 MLP
  - Model B (SirGeometricMLP): conservation exact by construction
  - Model C (SirSoftGeometricMLP): redundancy-aware input, no hard constraint

Training configuration (fixed, for reproducibility):
    seed              = 42
    train/val/test    = 70% / 15% / 15% (from data/generate_sir_data.py)
    normalization     = none
    loss              = MSE (on the full 3-dim next state)
    optimizer         = Adam, lr=1e-3
    batch size        = 256
    epochs            = 200
    model selection   = checkpoint with lowest validation loss

Saved checkpoints:
    models/checkpoints/sir_mlp.pt
    models/checkpoints/sir_geometric.pt
    models/checkpoints/sir_soft_geometric.pt

Run from the verified-scientific-ml/ directory:
    python -m models.train_sir
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.sir_mlp import SirMLP
from models.sir_geometric import SirGeometricMLP
from models.sir_soft_geometric import SirSoftGeometricMLP

SEED = 42
HIDDEN_DIM = 16
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 200

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODELS_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "sir_dataset.npz")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_model(
    model: nn.Module,
    train_x_t: torch.Tensor,
    train_y_t: torch.Tensor,
    val_x_t: torch.Tensor,
    val_y_t: torch.Tensor,
    name: str,
) -> float:
    """
    Train model for EPOCHS epochs using Adam + MSE. Returns best val loss.
    Prints progress at epoch 1 and every 20 epochs.
    """
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(train_x_t, train_y_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x_t), val_y_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0:
            print(f"  [{name}] epoch {epoch:4d}  train={train_loss:.6e}  val={val_loss:.6e}")

    model.load_state_dict(best_state)
    return best_val_loss


def _mean_conservation_error(model: nn.Module, test_x_t: torch.Tensor) -> float:
    """Mean |Δpop| = mean |(s_next+i_next+r_next) - 1| over test set."""
    model.eval()
    with torch.no_grad():
        next_state = model(test_x_t)
    pop_next = next_state[:, 0] + next_state[:, 1] + next_state[:, 2]
    return float(torch.mean(torch.abs(pop_next - 1.0)).item())


def main() -> None:
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python -m data.generate_sir_data` first."
        )

    data = np.load(DATA_PATH)
    train_x = data["train_states_t"].astype(np.float32)
    train_y = data["train_states_tp1"].astype(np.float32)
    val_x = data["val_states_t"].astype(np.float32)
    val_y = data["val_states_tp1"].astype(np.float32)
    test_x = data["test_states_t"].astype(np.float32)

    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y)
    val_x_t = torch.from_numpy(val_x)
    val_y_t = torch.from_numpy(val_y)
    test_x_t = torch.from_numpy(test_x)

    results = {}

    # Model A: plain MLP
    print("\n=== Model A: SirMLP ===")
    set_seed(SEED)
    model_a = SirMLP(hidden_dim=HIDDEN_DIM)
    val_loss_a = _train_model(model_a, train_x_t, train_y_t, val_x_t, val_y_t, "A-MLP")
    torch.save(model_a.state_dict(), os.path.join(CHECKPOINT_DIR, "sir_mlp.pt"))
    results["A (MLP)"] = {
        "val_loss": val_loss_a,
        "mean_dpop": _mean_conservation_error(model_a, test_x_t),
    }
    print(f"  Best val loss: {val_loss_a:.6e}")

    # Model B: geometric (exact conservation)
    print("\n=== Model B: SirGeometricMLP ===")
    set_seed(SEED)
    model_b = SirGeometricMLP(hidden_dim=HIDDEN_DIM)
    val_loss_b = _train_model(model_b, train_x_t, train_y_t, val_x_t, val_y_t, "B-Geom")
    torch.save(model_b.state_dict(), os.path.join(CHECKPOINT_DIR, "sir_geometric.pt"))
    results["B (geom)"] = {
        "val_loss": val_loss_b,
        "mean_dpop": _mean_conservation_error(model_b, test_x_t),
    }
    print(f"  Best val loss: {val_loss_b:.6e}")

    # Model C: soft-geometric
    print("\n=== Model C: SirSoftGeometricMLP ===")
    set_seed(SEED)
    model_c = SirSoftGeometricMLP(hidden_dim=HIDDEN_DIM)
    val_loss_c = _train_model(model_c, train_x_t, train_y_t, val_x_t, val_y_t, "C-Soft")
    torch.save(model_c.state_dict(), os.path.join(CHECKPOINT_DIR, "sir_soft_geometric.pt"))
    results["C (soft)"] = {
        "val_loss": val_loss_c,
        "mean_dpop": _mean_conservation_error(model_c, test_x_t),
    }
    print(f"  Best val loss: {val_loss_c:.6e}")

    # Comparison table
    print("\n")
    print(f"{'Model':<12} | {'val_loss':<14} | {'mean|Dpop| (test)'}")
    print("-" * 48)
    for model_name, r in results.items():
        print(f"{model_name:<12} | {r['val_loss']:<14.6e} | {r['mean_dpop']:.6e}")

    print(f"\nSaved checkpoints to {CHECKPOINT_DIR}/")
    print("  sir_mlp.pt")
    print("  sir_geometric.pt")
    print("  sir_soft_geometric.pt")


if __name__ == "__main__":
    main()
