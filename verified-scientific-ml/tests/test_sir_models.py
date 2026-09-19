"""
Tests for the SIR surrogate models (A, B, C), their ONNX export, and the
SIR simulator -- the epidemiological-domain counterpart to
tests/test_particle_geometric.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.sir_mlp import SirMLP
from models.sir_geometric import SirGeometricMLP
from models.sir_soft_geometric import SirSoftGeometricMLP
from simulator.sir_simulator import simulate_sir, total_population


# ---------------------------------------------------------------------------
# Model A (SirMLP)
# ---------------------------------------------------------------------------


def test_sir_model_a_output_shape():
    model = SirMLP(hidden_dim=8)
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 3)


def test_sir_model_a_does_not_conserve_population():
    """
    Model A has no conservation structure, so its population errors should
    be non-trivial for random weights. We verify that the mean absolute
    error is greater than 1e-3 for a random initialisation.
    """
    torch.manual_seed(99)
    model = SirMLP(hidden_dim=16)
    # Random inputs -- not physically valid, but enough to check the model
    # has no built-in conservation constraint.
    state = torch.rand(500, 3) * 0.9 + 0.05
    errors = model._get_conservation_error(state)
    # For random weights this will typically be O(1) -- well above 1e-3.
    assert errors.abs().mean().item() > 1e-3


# ---------------------------------------------------------------------------
# Model B (SirGeometricMLP)
# ---------------------------------------------------------------------------


def test_sir_model_b_output_shape():
    model = SirGeometricMLP(hidden_dim=8)
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 3)


def test_sir_model_b_conserves_population_exactly_for_any_weights():
    """
    The r_next = r - Δs - Δi construction must conserve population exactly
    for ANY weight setting, not just a trained one -- this is what makes it
    a structural guarantee rather than a learned approximation. Checked here
    with random (untrained) weights.
    """
    torch.manual_seed(0)
    model = SirGeometricMLP(hidden_dim=16)
    state = torch.randn(500, 3)
    out = model(state)

    pop_before = state[:, 0] + state[:, 1] + state[:, 2]
    pop_after = out[:, 0] + out[:, 1] + out[:, 2]

    # Exact up to float32 rounding of an algebraic identity (no learning involved).
    assert torch.max(torch.abs(pop_after - pop_before)).item() < 1e-5


def test_sir_model_b_conservation_identity():
    """
    Algebraic proof via the combine layer weights: the row sums of the combine
    matrix for [s_next, i_next, r_next] in terms of [Δs, Δi] must be
    [+1, +1, -1] and [+1, +1, -1] respectively, so (Δs+Δi-Δs-Δi) = 0.

    Concretely: combine.weight has shape (3, 5); columns 3 and 4 correspond
    to Δs and Δi. Sum of rows (for the output) in those columns:
      row 0 (s_next): [+1,  0] -> contributes +Δs
      row 1 (i_next): [ 0, +1] -> contributes +Δi
      row 2 (r_next): [-1, -1] -> contributes -Δs - Δi
    Column sum of those two columns across all three rows = 0.
    """
    model = SirGeometricMLP(hidden_dim=16)
    W = model.combine.weight.detach()  # shape (3, 5)
    # Columns 3 and 4 are the Δs and Δi contributions
    delta_cols = W[:, 3:5]  # shape (3, 2)
    col_sums = delta_cols.sum(dim=0)  # sum over 3 output rows per delta column
    # Each delta column must sum to zero (Δs contributes +1 to s, -1 to r;
    # Δi contributes +1 to i, -1 to r).
    assert col_sums.abs().max().item() < 1e-6


def test_sir_model_b_extract_layer_is_fixed():
    """The extract layer's weights must have requires_grad=False."""
    model = SirGeometricMLP(hidden_dim=16)
    assert not model.extract.weight.requires_grad


def test_sir_model_b_combine_layer_is_fixed():
    """The combine layer's weights must have requires_grad=False."""
    model = SirGeometricMLP(hidden_dim=16)
    assert not model.combine.weight.requires_grad


# ---------------------------------------------------------------------------
# Model C (SirSoftGeometricMLP)
# ---------------------------------------------------------------------------


def test_sir_model_c_output_shape():
    model = SirSoftGeometricMLP(hidden_dim=8)
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 3)


def test_sir_model_c_does_not_conserve_by_construction():
    """
    Random weights for Model C should violate population conservation for
    at least some inputs -- this is the whole point of the soft-geometric
    design. We check that max |ΔPop| > 1e-3 for random weights.
    """
    torch.manual_seed(7)
    model = SirSoftGeometricMLP(hidden_dim=16)
    state = torch.rand(500, 3)
    errors = model._get_conservation_error(state)
    assert errors.abs().max().item() > 1e-3


def test_sir_model_c_extract_si_is_fixed():
    """The extract layer's weights must have requires_grad=False."""
    model = SirSoftGeometricMLP(hidden_dim=16)
    assert not model.extract.weight.requires_grad


def test_sir_model_c_combine_layer_is_fixed():
    """The combine layer's weights must have requires_grad=False."""
    model = SirSoftGeometricMLP(hidden_dim=16)
    assert not model.combine.weight.requires_grad


def test_sir_model_c_delta_net_is_learned():
    """All delta_net parameters must have requires_grad=True."""
    model = SirSoftGeometricMLP(hidden_dim=16)
    for name, param in model.delta_net.named_parameters():
        assert param.requires_grad, f"delta_net.{name} has requires_grad=False"


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_sir_simulator_conserves_population():
    """
    simulate_sir must preserve s+i+r=1 to integrator tolerance (< 1e-6)
    over 10 steps, starting from a valid initial condition.
    """
    state0 = np.array([0.70, 0.05, 0.25])  # s+i+r = 1
    result = simulate_sir(state0, beta=0.3, gamma=0.1, t_span=(0.0, 10.0), n_points=11)
    states = result["states"]  # (11, 3)
    pop = total_population(states)
    assert np.all(np.abs(pop - 1.0) < 1e-6), (
        f"Population drift exceeded 1e-6: max |s+i+r-1| = {np.max(np.abs(pop-1.0)):.3e}"
    )


def test_sir_simulator_monotone_s_decreasing():
    """Susceptible fraction s must be non-increasing along the trajectory."""
    state0 = np.array([0.80, 0.10, 0.10])
    result = simulate_sir(state0, beta=0.3, gamma=0.1, t_span=(0.0, 30.0), n_points=31)
    s = result["states"][:, 0]
    diffs = np.diff(s)
    # Allow small integrator noise but overall must be decreasing
    assert np.all(diffs < 1e-6)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_sir_dataset_generation():
    """
    sir_dataset.npz must contain train/val/test splits with shape (N, 3)
    and all states must satisfy s+i+r ≈ 1 (within integrator tolerance).
    """
    import os
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "sir_dataset.npz"
    )
    if not os.path.exists(data_path):
        pytest.skip("sir_dataset.npz not found -- run python -m data.generate_sir_data first")

    data = np.load(data_path)

    for split in ("train", "val", "test"):
        xt = data[f"{split}_states_t"]
        xtp1 = data[f"{split}_states_tp1"]
        assert xt.ndim == 2 and xt.shape[1] == 3, f"{split}_states_t shape wrong: {xt.shape}"
        assert xtp1.ndim == 2 and xtp1.shape[1] == 3, f"{split}_states_tp1 shape wrong: {xtp1.shape}"

        pop = xt[:, 0] + xt[:, 1] + xt[:, 2]
        assert np.all(np.abs(pop - 1.0) < 1e-5), (
            f"{split} states_t: max |s+i+r-1| = {np.max(np.abs(pop-1.0)):.3e}"
        )

        pop_tp1 = xtp1[:, 0] + xtp1[:, 1] + xtp1[:, 2]
        assert np.all(np.abs(pop_tp1 - 1.0) < 1e-5), (
            f"{split} states_tp1: max |s+i+r-1| = {np.max(np.abs(pop_tp1-1.0)):.3e}"
        )


# ---------------------------------------------------------------------------
# ONNX export equivalence
# ---------------------------------------------------------------------------


def test_sir_onnx_export_equivalence():
    """
    For each model, exporting to ONNX and running onnxruntime inference
    should give outputs that match PyTorch's direct forward pass to < 1e-4.
    """
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    import os
    import tempfile

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    samples = rng.uniform(0.05, 0.90, size=(50, 3)).astype(np.float32)
    # Normalise so s+i+r=1
    samples = samples / samples.sum(axis=1, keepdims=True)
    dummy = torch.zeros((1, 3), dtype=torch.float32)

    for ModelClass in [SirMLP, SirGeometricMLP, SirSoftGeometricMLP]:
        model = ModelClass(hidden_dim=16)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=["state"], output_names=["next_state"],
                opset_version=13, dynamo=False,
                dynamic_axes={"state": {0: "batch"}, "next_state": {0: "batch"}},
            )
            sess = ort.InferenceSession(onnx_path)
            ort_out = sess.run(None, {"state": samples})[0]

            with torch.no_grad():
                torch_out = model(torch.from_numpy(samples)).numpy()

            max_diff = float(np.max(np.abs(ort_out - torch_out)))
            assert max_diff < 1e-4, (
                f"{ModelClass.__name__}: max diff between PyTorch and ONNX = {max_diff:.3e}"
            )
        finally:
            os.unlink(onnx_path)


# ---------------------------------------------------------------------------
# Smoke: training loop reduces loss
# ---------------------------------------------------------------------------


def test_sir_model_b_smoke_training_reduces_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # Generate physically-valid synthetic data: s+i+r=1
    raw = rng.uniform(0.01, 1.0, size=(256, 3)).astype(np.float32)
    raw = raw / raw.sum(axis=1, keepdims=True)
    x = raw
    y = raw + rng.uniform(-0.01, 0.01, size=(256, 3)).astype(np.float32)
    y = np.clip(y, 0, 1)
    y = y / y.sum(axis=1, keepdims=True)

    model = SirGeometricMLP(hidden_dim=8)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=64, shuffle=True,
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
