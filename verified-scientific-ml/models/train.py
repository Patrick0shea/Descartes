"""
Train an MLP surrogate model on the Step 1 harmonic oscillator dataset.

Learns the one-step map:
    [x_t, v_t] -> [x_(t+1), v_(t+1)]

Training configuration (fixed, for reproducibility):
    seed              = 42
    train/val/test    = 70% / 15% / 15% (random split of the 100k pairs)
    normalization     = standardize (x, v) using TRAIN-set mean/std only,
                        same normalizer reused for inputs and targets
                        (x_t and x_(t+1) are drawn from the same physical
                        distribution)
    architecture      = MLP: 2 -> 64 -> 64 -> 2, ReLU activations
    loss              = MSE
    optimizer         = Adam, lr=1e-3
    batch size        = 256
    epochs            = 200
    model selection   = checkpoint with lowest validation loss

Run from the verified-scientific-ml/ directory:
    python -m models.train
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.normalization import Normalizer
from models.surrogate import MLPSurrogate

SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15  # remaining 0.15 is test

HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 200

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODELS_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "harmonic_oscillator_dataset.npz")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(states_t: np.ndarray, states_tp1: np.ndarray, seed: int):
    """Randomly split (states_t, states_tp1) pairs into train/val/test."""
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
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    data = np.load(DATA_PATH)
    states_t = data["states_t"].astype(np.float64)
    states_tp1 = data["states_tp1"].astype(np.float64)
    k = float(data["k"])
    dt = float(data["dt"])

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_dataset(
        states_t, states_tp1, SEED
    )

    # Normalizer fit on TRAIN inputs only, reused for outputs (same
    # physical quantities, stationary distribution over the orbit).
    normalizer = Normalizer.fit(train_x)
    normalizer.save(os.path.join(CHECKPOINT_DIR, "normalizer.npz"))

    def to_tensor(x: np.ndarray, y: np.ndarray):
        x_n = normalizer.transform(x).astype(np.float32)
        y_n = normalizer.transform(y).astype(np.float32)
        return torch.from_numpy(x_n), torch.from_numpy(y_n)

    train_x_t, train_y_t = to_tensor(train_x, train_y)
    val_x_t, val_y_t = to_tensor(val_x, val_y)

    # Save raw (un-normalized) val/test splits + metadata so evaluate.py
    # can reproduce everything without redoing the split RNG.
    np.savez(
        os.path.join(CHECKPOINT_DIR, "splits.npz"),
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        k=k, dt=dt, seed=SEED,
    )

    model = MLPSurrogate(hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(train_x_t, train_y_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    train_losses = []
    val_losses = []
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: p.clone() for name, p in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.6e}  val_loss={val_loss:.6e}")

    model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "surrogate.pt"))
    np.savez(
        os.path.join(CHECKPOINT_DIR, "loss_history.npz"),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        best_val_loss=best_val_loss,
    )

    print(f"\nBest validation loss (normalized MSE): {best_val_loss:.6e}")
    print(f"Saved model to {os.path.join(CHECKPOINT_DIR, 'surrogate.pt')}")
    print(f"Saved normalizer to {os.path.join(CHECKPOINT_DIR, 'normalizer.npz')}")


if __name__ == "__main__":
    main()
