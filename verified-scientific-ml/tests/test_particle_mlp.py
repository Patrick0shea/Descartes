"""Tests for the particle MLP surrogate model and its training loop."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.normalization import Normalizer
from models.particle_mlp import ParticleMLP


def test_particle_mlp_output_shape():
    model = ParticleMLP(hidden_dim=16)
    x = torch.randn(5, 8)
    y = model(x)
    assert y.shape == (5, 8)


def test_particle_mlp_smoke_training_reduces_loss():
    """A few epochs on a tiny synthetic dataset should reduce the training loss."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Synthetic linear "physics": next_state = state + small fixed shift.
    x = rng.uniform(-1, 1, size=(256, 8)).astype(np.float32)
    y = x + 0.01

    normalizer = Normalizer.fit(x)
    x_n = normalizer.transform(x).astype(np.float32)
    y_n = normalizer.transform(y).astype(np.float32)

    model = ParticleMLP(hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_n), torch.from_numpy(y_n)), batch_size=64, shuffle=True)

    def epoch_loss():
        model.eval()
        with torch.no_grad():
            return loss_fn(model(torch.from_numpy(x_n)), torch.from_numpy(y_n)).item()

    loss_before = epoch_loss()
    model.train()
    for _ in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    loss_after = epoch_loss()

    assert loss_after < loss_before
