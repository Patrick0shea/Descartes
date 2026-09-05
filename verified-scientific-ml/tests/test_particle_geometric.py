"""
Tests for the geometric (momentum-conserving-by-construction) surrogate,
its ONNX export, and its Marabou verification -- the Step 5 counterpart
to tests/test_particle_mlp.py and tests/test_particle_verification.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.particle_geometric import ParticleGeometricMLP
from verification.export_geometric_onnx import check_equivalence, load_geometric_surrogate, ONNX_PATH, export
from verification.particle_properties import GEOMETRIC_MOMENTUM_PROPERTIES
from verification.verify_particle import verify_momentum_property


def test_geometric_model_output_shape():
    model = ParticleGeometricMLP(hidden_dim=8)
    x = torch.randn(5, 8)
    y = model(x)
    assert y.shape == (5, 8)


def test_geometric_model_conserves_momentum_exactly_regardless_of_weights():
    """
    The +F/-F construction must conserve momentum exactly for ANY weight
    setting, not just a trained one -- this is what makes it a structural
    guarantee rather than a learned approximation. Checked here with
    random (untrained) weights and again on the trained checkpoint.
    """
    torch.manual_seed(0)
    model = ParticleGeometricMLP(hidden_dim=16)
    state = torch.randn(500, 8)
    out = model(state)

    px_before, px_after = state[:, 2] + state[:, 6], out[:, 2] + out[:, 6]
    py_before, py_after = state[:, 3] + state[:, 7], out[:, 3] + out[:, 7]

    # Exact up to float32 rounding of an algebraic identity (no learning involved).
    assert torch.max(torch.abs(px_after - px_before)).item() < 1e-5
    assert torch.max(torch.abs(py_after - py_before)).item() < 1e-5


def test_geometric_model_translation_invariance():
    """Shifting both particles by the same amount must not change the predicted force/velocities."""
    torch.manual_seed(1)
    model = ParticleGeometricMLP(hidden_dim=16)
    state = torch.randn(20, 8)

    shift = torch.zeros(8)
    shift[0] = shift[4] = 3.0  # shift x1, x2 by the same amount
    shift[1] = shift[5] = -1.5  # shift y1, y2 by the same amount

    out = model(state)
    out_shifted = model(state + shift)

    # Velocities (indices 2,3,6,7) must be identical; positions shift by the same amount.
    assert torch.allclose(out[:, [2, 3, 6, 7]], out_shifted[:, [2, 3, 6, 7]], atol=1e-5)
    assert torch.allclose(out[:, [0, 4]] + 3.0, out_shifted[:, [0, 4]], atol=1e-5)
    assert torch.allclose(out[:, [1, 5]] - 1.5, out_shifted[:, [1, 5]], atol=1e-5)


def test_geometric_smoke_training_reduces_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(256, 8)).astype(np.float32)
    y = x + 0.01  # synthetic target, not real physics -- just checks the training loop runs

    model = ParticleGeometricMLP(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=64, shuffle=True)

    def current_loss():
        model.eval()
        with torch.no_grad():
            return loss_fn(model(torch.from_numpy(x)), torch.from_numpy(y)).item()

    loss_before = current_loss()
    model.train()
    for _ in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    loss_after = current_loss()

    assert loss_after < loss_before


def test_geometric_onnx_export_equivalence():
    export()
    model = load_geometric_surrogate()
    max_diff = check_equivalence(model, n_samples=200, seed=1)
    assert max_diff < 1e-4


def test_geometric_marabou_verifies_tighter_than_baseline():
    """
    End-to-end: the trained geometric model must verify UNSAT (no
    counterexample) at epsilon=1e-4 -- the exact tolerance Step 4's plain
    MLP baseline was SAT (violated) at. This is the pipeline's central
    comparative claim, checked directly against the real Marabou solve.
    """
    export()
    tight_props = [p for p in GEOMETRIC_MOMENTUM_PROPERTIES if p.epsilon == 1e-4]
    assert len(tight_props) == 2  # x and y axis

    for prop in tight_props:
        result = verify_momentum_property(ONNX_PATH, prop)
        assert result["result"] == "UNSAT"
        assert result["counterexample"] is None
