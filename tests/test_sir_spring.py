"""
Tests for the sir_spring case study.

Covers:
- Model shapes and forward passes
- DeltaModel: output = f(x) - x empirically
- Model B algebraic conservation: delta sum ≈ 0 for all inputs
- Property factory functions: correct coefficients and types
- Checkpoint loading (uses real checkpoints from verified-scientific-ml/)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from case_studies.sir_spring.delta_model import DeltaModel
from case_studies.sir_spring.models.particle import (
    ParticleGeometricMLP,
    ParticleMLP,
    ParticleSoftGeometricMLP,
)
from case_studies.sir_spring.models.sir import (
    SirGeometricMLP,
    SirMLP,
    SirSoftGeometricMLP,
)
from case_studies.sir_spring.properties import (
    particle_x_momentum_conservation,
    particle_y_momentum_conservation,
    sir_population_conservation,
)
from verifier.property import Conservation

CKPT_DIR = (
    Path(__file__).parent.parent
    / "verified-scientific-ml" / "models" / "checkpoints"
)


# ── model shapes ──────────────────────────────────────────────────────────────

class TestSirShapes:
    @pytest.fixture(params=["mlp", "geometric", "soft"])
    def sir_model(self, request):
        return {"mlp": SirMLP(), "geometric": SirGeometricMLP(),
                "soft": SirSoftGeometricMLP()}[request.param]

    def test_output_shape(self, sir_model):
        x = torch.rand(8, 3)
        assert sir_model(x).shape == (8, 3)


class TestParticleShapes:
    @pytest.fixture(params=["mlp", "geometric", "soft"])
    def particle_model(self, request):
        return {"mlp": ParticleMLP(), "geometric": ParticleGeometricMLP(),
                "soft": ParticleSoftGeometricMLP()}[request.param]

    def test_output_shape(self, particle_model):
        x = torch.rand(8, 8)
        assert particle_model(x).shape == (8, 8)


# ── DeltaModel ────────────────────────────────────────────────────────────────

class TestDeltaModel:
    def test_output_equals_f_minus_x(self):
        torch.manual_seed(0)
        base = SirMLP()
        delta = DeltaModel(base, n_dims=3)
        x = torch.rand(16, 3)
        expected = base(x) - x
        torch.testing.assert_close(delta(x), expected)

    def test_zero_model_gives_neg_identity(self):
        """A base model that returns zeros should give delta = -x."""
        class ZeroModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)
        dm = DeltaModel(ZeroModel(), n_dims=4)
        x = torch.rand(5, 4)
        torch.testing.assert_close(dm(x), -x)

    def test_particle_shape(self):
        base = ParticleMLP()
        delta = DeltaModel(base, n_dims=8)
        x = torch.rand(4, 8)
        assert delta(x).shape == (4, 8)

    def test_neg_id_not_trainable(self):
        dm = DeltaModel(SirMLP(), n_dims=3)
        for p in dm.neg_id.parameters():
            assert not p.requires_grad

    def test_neg_id_weight_is_negative_identity(self):
        dm = DeltaModel(SirMLP(), n_dims=3)
        expected = -torch.eye(3)
        torch.testing.assert_close(dm.neg_id.weight, expected)


# ── Model B algebraic conservation ────────────────────────────────────────────

class TestSirGeometricConservation:
    def test_conservation_error_near_zero(self):
        """DeltaModel sum (Δs+Δi+Δr) should be ≈0 for SirGeometricMLP."""
        model = SirGeometricMLP()
        delta = DeltaModel(model, n_dims=3)
        torch.manual_seed(42)
        x = torch.rand(1000, 3) * torch.tensor([0.9, 0.59, 0.89]) + torch.tensor([0.05, 0.01, 0.01])
        d = delta(x)
        error = (d[:, 0] + d[:, 1] + d[:, 2]).abs()
        assert error.max().item() < 1e-5, f"max conservation error {error.max():.2e}"


class TestParticleGeometricConservation:
    def test_x_momentum_error_near_zero(self):
        """Δvx1 + Δvx2 should be ≈0 for ParticleGeometricMLP."""
        model = ParticleGeometricMLP()
        delta = DeltaModel(model, n_dims=8)
        torch.manual_seed(42)
        x = torch.rand(1000, 8) * 2 - 1
        d = delta(x)
        error = (d[:, 2] + d[:, 6]).abs()   # Δvx1 + Δvx2
        assert error.max().item() < 1e-5

    def test_y_momentum_error_near_zero(self):
        """Δvy1 + Δvy2 should be ≈0 for ParticleGeometricMLP."""
        model = ParticleGeometricMLP()
        delta = DeltaModel(model, n_dims=8)
        torch.manual_seed(42)
        x = torch.rand(1000, 8) * 2 - 1
        d = delta(x)
        error = (d[:, 3] + d[:, 7]).abs()   # Δvy1 + Δvy2
        assert error.max().item() < 1e-5


# ── properties ────────────────────────────────────────────────────────────────

class TestProperties:
    def test_sir_population_conservation_type(self):
        p = sir_population_conservation(1e-2)
        assert isinstance(p, Conservation)

    def test_sir_coefficients(self):
        p = sir_population_conservation(1e-2)
        np.testing.assert_array_equal(p.coefficients, [1.0, 1.0, 1.0])

    def test_sir_epsilon(self):
        p = sir_population_conservation(1e-3)
        assert p.epsilon == 1e-3

    def test_particle_px_shape(self):
        p = particle_x_momentum_conservation(0.1)
        assert p.coefficients.shape == (8,)
        assert p.coefficients[2] == 1.0   # vx1
        assert p.coefficients[6] == 1.0   # vx2
        assert p.coefficients.sum() == 2.0

    def test_particle_py_shape(self):
        p = particle_y_momentum_conservation(0.1)
        assert p.coefficients[3] == 1.0   # vy1
        assert p.coefficients[7] == 1.0   # vy2
        assert p.coefficients.sum() == 2.0

    def test_name_labels(self):
        assert sir_population_conservation(0.1).name == "sir_population_conservation"
        assert particle_x_momentum_conservation(0.1).name == "particle_px_conservation"
        assert particle_y_momentum_conservation(0.1).name == "particle_py_conservation"


# ── checkpoint loading ────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (CKPT_DIR / "sir_mlp.pt").exists(),
    reason="SIR checkpoints not available",
)
class TestCheckpointLoading:
    def test_sir_mlp_loads(self):
        import torch
        from verifier import ModelSpec
        # Save the DeltaModel's full state dict (includes base.* and neg_id.*)
        delta = DeltaModel(SirMLP(), n_dims=3)
        torch.save(delta.state_dict(), "/tmp/sir_test.pt")
        fresh = DeltaModel(SirMLP(), n_dims=3)
        spec = ModelSpec.from_checkpoint(
            fresh, "/tmp/sir_test.pt",
            input_lb=[0.05, 0.01, 0.01],
            input_ub=[0.95, 0.60, 0.90],
            name="sir_mlp_test",
        )
        assert spec.checkpoint_hash is not None
        assert spec.model is not None

    def test_sir_geometric_loads(self):
        import torch
        from verifier import ModelSpec
        base = SirGeometricMLP()
        state = torch.load(str(CKPT_DIR / "sir_geometric.pt"), map_location="cpu", weights_only=True)
        base.load_state_dict(state)
        base.eval()
        # Spot-check conservation still holds after loading
        delta = DeltaModel(base, n_dims=3)
        x = torch.rand(100, 3) * torch.tensor([0.9, 0.59, 0.89]) + torch.tensor([0.05, 0.01, 0.01])
        d = delta(x)
        error = (d[:, 0] + d[:, 1] + d[:, 2]).abs().max().item()
        assert error < 1e-5, f"Conservation error after loading: {error:.2e}"
