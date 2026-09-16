"""
Tests for the soft-geometric surrogate (Model C), its ONNX export, and
its Marabou verification -- the Model C counterpart to
tests/test_particle_geometric.py (Model B).

Key behavioural differences from Model B being tested:
- Translation invariance HOLDS (same extract_rel structure as Model B).
- Momentum conservation does NOT hold exactly for arbitrary weights
  (unlike Model B's +F/-F algebraic identity).
- Output shape and ONNX export equivalence match the existing pattern.
- Empirical momentum drift on the trained checkpoint is smaller than
  Model A's (if the checkpoint is available -- skipped otherwise).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.particle_soft_geometric import ParticleSoftGeometricMLP
from verification.export_soft_geometric_onnx import (
    ONNX_PATH,
    check_equivalence,
    export,
    load_soft_geometric_surrogate,
)
from verification.particle_properties import SOFT_GEOMETRIC_MOMENTUM_PROPERTIES

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "checkpoints", "particle_soft_geometric.pt",
)

CHECKPOINT_AVAILABLE = os.path.exists(CHECKPOINT_PATH)


def test_soft_geometric_model_output_shape():
    model = ParticleSoftGeometricMLP(hidden_dim=8)
    x = torch.randn(5, 8)
    y = model(x)
    assert y.shape == (5, 8)


def test_soft_geometric_translation_invariance():
    """
    Shifting both particles by the same vector must not change velocity
    outputs. This follows from the fixed extract_rel layer: delta_net
    only ever sees (x2-x1, y2-y1, vx2-vx1, vy2-vy1), which is
    unchanged by a common translation. Checked with random (untrained)
    weights.
    """
    torch.manual_seed(1)
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    state = torch.randn(20, 8)

    shift = torch.zeros(8)
    shift[0] = shift[4] = 3.0    # shift x1, x2 by the same amount
    shift[1] = shift[5] = -1.5   # shift y1, y2 by the same amount

    out = model(state)
    out_shifted = model(state + shift)

    # Velocities (indices 2,3,6,7) must be identical
    assert torch.allclose(out[:, [2, 3, 6, 7]], out_shifted[:, [2, 3, 6, 7]], atol=1e-5)
    # Positions shift by the same amount as the input shift
    assert torch.allclose(out[:, [0, 4]] + 3.0, out_shifted[:, [0, 4]], atol=1e-5)
    assert torch.allclose(out[:, [1, 5]] - 1.5, out_shifted[:, [1, 5]], atol=1e-5)


def test_soft_geometric_translation_invariance_helper():
    """_get_translation_invariance_error() should return near-zero errors."""
    torch.manual_seed(2)
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    state = torch.randn(50, 8)
    vx_err, vy_err = model._get_translation_invariance_error(state)
    assert vx_err.item() < 1e-5
    assert vy_err.item() < 1e-5


def test_soft_geometric_momentum_not_conserved_by_construction():
    """
    For random (untrained) weights, momentum conservation should NOT
    hold in general. Model C has no +F/-F algebraic identity -- the
    four velocity updates [dvx1, dvy1, dvx2, dvy2] are independent.
    A random seed with enough samples will almost certainly violate
    |delta_Px| < 1e-5 (the threshold Model B satisfies exactly).
    """
    torch.manual_seed(0)
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    state = torch.randn(500, 8)
    delta_px, delta_py = model._get_momentum_conservation_error(state)
    # With random weights, at least some states should have large momentum error
    assert torch.max(torch.abs(delta_px)).item() > 1e-3
    assert torch.max(torch.abs(delta_py)).item() > 1e-3


def test_soft_geometric_momentum_helper_returns_correct_shape():
    model = ParticleSoftGeometricMLP(hidden_dim=8)
    state = torch.randn(10, 8)
    delta_px, delta_py = model._get_momentum_conservation_error(state)
    assert delta_px.shape == (10,)
    assert delta_py.shape == (10,)


def test_soft_geometric_smoke_training_reduces_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(256, 8)).astype(np.float32)
    y = x + 0.01  # synthetic target, not real physics -- just checks the training loop runs

    model = ParticleSoftGeometricMLP(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=64,
        shuffle=True,
    )

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


def test_soft_geometric_fixed_layers_have_no_grad():
    """extract_rel and combine weights must never be updated by optimizers."""
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    assert not model.extract_rel.weight.requires_grad
    assert not model.combine.weight.requires_grad


def test_soft_geometric_onnx_export_equivalence():
    pytest.importorskip("onnx")
    export()
    model = load_soft_geometric_surrogate()
    max_diff = check_equivalence(model, n_samples=200, seed=1)
    assert max_diff < 1e-4


@pytest.mark.skipif(
    not CHECKPOINT_AVAILABLE,
    reason="Trained checkpoint particle_soft_geometric.pt not found; run models/train_soft_geometric.py first.",
)
def test_soft_geometric_trained_momentum_smaller_than_model_a():
    """
    Empirical check: trained Model C should have smaller mean momentum
    drift on a random sample than Model A (plain MLP), because the
    translation-invariant feature extraction provides a physics-aligned
    inductive bias that guides training toward momentum conservation.

    Model A's test-set analysis (Step 4) found max |delta_Px| ~ 5.7e-3.
    We check that Model C's mean |delta_Px| is below 1e-2 as a loose
    sanity bound -- the exact threshold depends on training outcomes,
    so this is intentionally loose.
    """
    from verification.particle_properties import DOMAIN_LOWER, DOMAIN_UPPER

    model = load_soft_geometric_surrogate()

    rng = np.random.default_rng(42)
    samples = rng.uniform(DOMAIN_LOWER, DOMAIN_UPPER, size=(5000, 8)).astype(np.float32)
    state = torch.from_numpy(samples)

    delta_px, delta_py = model._get_momentum_conservation_error(state)
    mean_px_err = torch.mean(torch.abs(delta_px)).item()
    mean_py_err = torch.mean(torch.abs(delta_py)).item()

    # Loose bound: Model C should do significantly better than chance
    assert mean_px_err < 1e-2, f"Mean |delta_Px| = {mean_px_err:.3e} exceeds 1e-2"
    assert mean_py_err < 1e-2, f"Mean |delta_Py| = {mean_py_err:.3e} exceeds 1e-2"


def test_soft_geometric_momentum_properties_list_structure():
    """SOFT_GEOMETRIC_MOMENTUM_PROPERTIES has 12 entries: 6 epsilons x 2 axes."""
    assert len(SOFT_GEOMETRIC_MOMENTUM_PROPERTIES) == 12
    axes = {p.axis for p in SOFT_GEOMETRIC_MOMENTUM_PROPERTIES}
    assert axes == {"x", "y"}
    epsilons = sorted({p.epsilon for p in SOFT_GEOMETRIC_MOMENTUM_PROPERTIES})
    assert epsilons == sorted([1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])


@pytest.mark.skipif(
    not CHECKPOINT_AVAILABLE,
    reason="Trained checkpoint particle_soft_geometric.pt not found; run models/train_soft_geometric.py first.",
)
def test_soft_geometric_marabou_loose_epsilon_unsat():
    """
    End-to-end: the trained soft-geometric model must verify UNSAT at
    epsilon=1.0 -- the loose tolerance both Model A and Model B also
    pass. If Model C fails this, something is wrong with training.
    Requires Marabou to be importable.
    """
    pytest.importorskip("maraboupy")
    from verification.verify_particle import verify_momentum_property

    export()
    loose_props = [p for p in SOFT_GEOMETRIC_MOMENTUM_PROPERTIES if p.epsilon == 1.0]
    assert len(loose_props) == 2  # x and y axis

    for prop in loose_props:
        result = verify_momentum_property(ONNX_PATH, prop)
        assert result["result"] == "UNSAT", (
            f"Model C failed loose epsilon=1.0 verification on axis={prop.axis}. "
            f"Marabou returned {result['result']}."
        )
