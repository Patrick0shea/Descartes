"""
Tests for ConditionalGenerator.

These tests cover:
- Architecture shapes (no forward pass required for most)
- Forward pass output shapes and non-negativity
- input_bounds() contract
- encode_input / decode_energy helpers
- auto_LiRPA compatibility: model traces through BoundedModule
- IBP proves non-negativity (structural guarantee from final ReLU)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from case_studies.calorimeter.models.generator import (
    LOG_E_MAX,
    Z_DIM,
    ConditionalGenerator,
    decode_energy,
    encode_input,
)

# ── architecture tests ────────────────────────────────────────────────────────


class TestArchitecture:
    def test_photon_output_dim(self):
        g = ConditionalGenerator(n_voxels=368)
        x = torch.zeros(1, Z_DIM + 1)
        assert g(x).shape == (1, 368)

    def test_pion_output_dim(self):
        g = ConditionalGenerator(n_voxels=533)
        x = torch.zeros(1, Z_DIM + 1)
        assert g(x).shape == (1, 533)

    def test_batch_dimension(self):
        g = ConditionalGenerator(n_voxels=368)
        x = torch.zeros(16, Z_DIM + 1)
        assert g(x).shape == (16, 368)

    def test_n_hidden_layers(self):
        # With n_hidden=2 we get: Linear+ReLU, Linear+ReLU, Linear+ReLU (output)
        # → 3 Linear layers total
        g = ConditionalGenerator(n_voxels=368, hidden_dim=32, n_hidden=2)
        linears = [m for m in g.net if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 3  # input→hidden, hidden→hidden, hidden→output

    def test_n_hidden_one(self):
        g = ConditionalGenerator(n_voxels=368, hidden_dim=32, n_hidden=1)
        linears = [m for m in g.net if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 2  # input→hidden, hidden→output

    def test_final_layer_is_relu(self):
        g = ConditionalGenerator(n_voxels=368)
        assert isinstance(g.net[-1], torch.nn.ReLU)

    def test_no_batchnorm(self):
        g = ConditionalGenerator(n_voxels=368)
        for m in g.modules():
            assert not isinstance(m, torch.nn.BatchNorm1d), "No BatchNorm allowed"


# ── forward pass tests ────────────────────────────────────────────────────────


class TestForward:
    def test_outputs_nonneg(self):
        torch.manual_seed(0)
        g = ConditionalGenerator(n_voxels=368)
        # Random inputs spanning the full input range
        z = torch.rand(100, Z_DIM) * 2 - 1
        log_e = torch.rand(100, 1) * LOG_E_MAX
        x = torch.cat([z, log_e], dim=1)
        out = g(x)
        assert (out >= 0).all(), "All outputs should be >= 0 (final ReLU)"

    def test_outputs_nonneg_at_bounds(self):
        g = ConditionalGenerator(n_voxels=368)
        lb_list, ub_list = g.input_bounds()
        lb = torch.tensor([lb_list], dtype=torch.float32)
        ub = torch.tensor([ub_list], dtype=torch.float32)
        assert (g(lb) >= 0).all()
        assert (g(ub) >= 0).all()

    def test_deterministic(self):
        g = ConditionalGenerator(n_voxels=368)
        x = torch.rand(4, Z_DIM + 1)
        assert torch.equal(g(x), g(x))


# ── input_bounds tests ────────────────────────────────────────────────────────


class TestInputBounds:
    def test_length(self):
        g = ConditionalGenerator(n_voxels=368, z_dim=8)
        lb, ub = g.input_bounds()
        assert len(lb) == Z_DIM + 1
        assert len(ub) == Z_DIM + 1

    def test_z_bounds(self):
        g = ConditionalGenerator(n_voxels=368)
        lb, ub = g.input_bounds()
        assert lb[:Z_DIM] == [-1.0] * Z_DIM
        assert ub[:Z_DIM] == [1.0] * Z_DIM

    def test_energy_bounds(self):
        g = ConditionalGenerator(n_voxels=368)
        lb, ub = g.input_bounds()
        assert lb[Z_DIM] == 0.0
        assert abs(ub[Z_DIM] - LOG_E_MAX) < 1e-6

    def test_log_e_max_value(self):
        # E_MAX_MEV = 4_194_304 MeV (2^14 × E_MIN_MEV = 4 TeV, real data range)
        assert abs(LOG_E_MAX - math.log(4_194_304.0 / 256.0)) < 1e-6


# ── encode_input / decode_energy tests ───────────────────────────────────────


class TestHelpers:
    def test_encode_shape(self):
        z = torch.zeros(5, Z_DIM)
        e = torch.full((5,), 256.0)
        x = encode_input(z, e)
        assert x.shape == (5, Z_DIM + 1)

    def test_encode_log_energy_at_e_min(self):
        z = torch.zeros(1, Z_DIM)
        e = torch.tensor([256.0])
        x = encode_input(z, e)
        # log(256/256) = 0
        assert abs(x[0, Z_DIM].item()) < 1e-6

    def test_encode_log_energy_at_e_max(self):
        z = torch.zeros(1, Z_DIM)
        e = torch.tensor([4_194_304.0])   # actual E_MAX_MEV (4 TeV)
        x = encode_input(z, e)
        assert abs(x[0, Z_DIM].item() - LOG_E_MAX) < 1e-5

    def test_decode_roundtrip(self):
        for log_e in [0.0, 1.0, LOG_E_MAX]:
            assert abs(decode_energy(log_e) / (256.0 * math.exp(log_e)) - 1.0) < 1e-6


# ── auto_LiRPA compatibility ──────────────────────────────────────────────────


try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    HAS_LIRPA = True
except ImportError:
    HAS_LIRPA = False

lirpa_required = pytest.mark.skipif(not HAS_LIRPA, reason="auto_LiRPA not installed")


@lirpa_required
class TestLiRPACompatibility:
    def _make_bounded(self, n_voxels: int = 368):
        g = ConditionalGenerator(n_voxels=n_voxels)
        g.eval()
        dummy = torch.zeros(1, Z_DIM + 1)
        bounded = BoundedModule(g, dummy, device="cpu")
        bounded.eval()

        lb_list, ub_list = g.input_bounds()
        lb_np = np.array(lb_list, dtype=np.float32)
        ub_np = np.array(ub_list, dtype=np.float32)
        x_nom = torch.tensor((lb_np + ub_np) / 2).unsqueeze(0)
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=torch.tensor(lb_np).unsqueeze(0),
            x_U=torch.tensor(ub_np).unsqueeze(0),
        )
        return bounded, BoundedTensor(x_nom, ptb), n_voxels

    def test_traces_without_error(self):
        bounded, x_bounded, n_voxels = self._make_bounded()
        lb, ub = bounded.compute_bounds(x=(x_bounded,), method="IBP")
        assert lb.shape == (1, n_voxels)
        assert ub.shape == (1, n_voxels)

    def test_ibp_proves_nonnegativity(self):
        """IBP lower bounds should all be >= 0 (structural guarantee from final ReLU)."""
        bounded, x_bounded, _ = self._make_bounded()
        lb, ub = bounded.compute_bounds(x=(x_bounded,), method="IBP")
        lb_np = lb.detach().numpy()[0]
        assert (lb_np >= 0).all(), (
            f"IBP should prove non-negativity everywhere; "
            f"min lb = {lb_np.min():.6f}"
        )

    def test_ub_geq_lb(self):
        bounded, x_bounded, _ = self._make_bounded()
        lb, ub = bounded.compute_bounds(x=(x_bounded,), method="IBP")
        assert (ub >= lb).all()

    def test_crown_output_shape(self):
        bounded, x_bounded, n_voxels = self._make_bounded()
        lb, ub = bounded.compute_bounds(x=(x_bounded,), method="CROWN")
        assert lb.shape == (1, n_voxels)
        assert ub.shape == (1, n_voxels)
