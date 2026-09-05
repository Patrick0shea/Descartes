"""
Train the geometric (momentum-conserving-by-construction) surrogate on
the same two-particle spring system dataset as models/train_particle.py,
for direct comparison against that plain MLP baseline.

Learns the same one-step map, with the same train/val/test split, same
optimizer/loss/seed/epoch budget -- the only difference from
models/train_particle.py is the architecture (models/particle_geometric.py)
and that NO input/output normalization is applied here. Normalization is
unnecessary and would have to be threaded carefully through the model's
internal physical-unit arithmetic (the +F / -F velocity update relies on
operating in real physical units for the momentum identity to hold as
stated); the relative position/velocity features the network actually
sees are already O(1)-scaled over this domain, so plain unnormalized
regression converges fine.

Training configuration (fixed, for reproducibility):
    seed              = 42
    train/val/test    = 70% / 15% / 15% (from data/generate_particle_data.py)
    normalization     = none (see above)
    architecture      = ParticleGeometricMLP: force_net 4 -> 16 -> 16 -> 2,
                        ReLU activations, momentum-conserving by
                        construction (see models/particle_geometric.py)
    loss              = MSE (on the full 8-dim next state)
    optimizer         = Adam, lr=1e-3
    batch size        = 256
    epochs            = 200
    model selection   = checkpoint with lowest validation loss

Run from the verified-scientific-ml/ directory:
    python -m models.train_geometric
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.particle_geometric import ParticleGeometricMLP

SEED = 42
HIDDEN_DIM = 16
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 200

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODELS_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "particle_dataset.npz")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    data = np.load(DATA_PATH)
    train_x = data["train_states_t"].astype(np.float32)
    train_y = data["train_states_tp1"].astype(np.float32)
    val_x = data["val_states_t"].astype(np.float32)
    val_y = data["val_states_tp1"].astype(np.float32)

    train_x_t, train_y_t = torch.from_numpy(train_x), torch.from_numpy(train_y)
    val_x_t, val_y_t = torch.from_numpy(val_x), torch.from_numpy(val_y)

    model = ParticleGeometricMLP(hidden_dim=HIDDEN_DIM)
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

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "particle_geometric.pt"))
    np.savez(
        os.path.join(CHECKPOINT_DIR, "particle_geometric_loss_history.npz"),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        best_val_loss=best_val_loss,
    )

    print(f"\nBest validation loss (MSE, physical units): {best_val_loss:.6e}")
    print(f"Saved model to {os.path.join(CHECKPOINT_DIR, 'particle_geometric.pt')}")


if __name__ == "__main__":
    main()
