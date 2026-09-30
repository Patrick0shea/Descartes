"""
Tests for Step 7: PGD falsification baseline.

Covers:
- FalsificationResult JSON round-trip
- falsify() API with tiny models
- Model B (geometric) produces zero violation (algebraic identity)
- Model A (MLP) produces non-zero violation
- Calorimeter non-negativity: PGD finds no violation (ReLU exact)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from verifier import Conservation, FalsificationResult, ModelSpec, NonNegativity, falsify


# ── fixtures ──────────────────────────────────────────────────────────────────

class _TinyMLP(nn.Module):
    """Simple 2-in, 2-out MLP with no conservation structure."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


class _ConservationMLP(nn.Module):
    """2-in, 2-out: output = [f(x), -f(x)], so sum(out) = 0 always."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        v = self.f(x)
        return torch.cat([v, -v], dim=-1)


class _NonNegMLP(nn.Module):
    """2-in, 2-out ending in ReLU → outputs always >= 0."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2), nn.ReLU())

    def forward(self, x):
        return self.net(x)


def _make_spec(model: nn.Module, lb=None, ub=None) -> ModelSpec:
    if lb is None:
        lb = [-1.0, -1.0]
    if ub is None:
        ub = [1.0, 1.0]
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    torch.save(model.state_dict(), tmp)
    fresh = model.__class__()
    spec = ModelSpec.from_checkpoint(fresh, tmp, input_lb=lb, input_ub=ub, name="test_model")
    Path(tmp).unlink(missing_ok=True)
    return spec


# ── FalsificationResult API ───────────────────────────────────────────────────

class TestFalsificationResultAPI:
    def test_json_roundtrip_counterexample(self):
        r = FalsificationResult(
            status="COUNTEREXAMPLE_FOUND",
            property_name="conservation",
            counterexample=[0.1, 0.2],
            violation_magnitude=0.5,
            n_restarts=10,
            runtime_s=1.0,
            model_name="test",
        )
        r2 = FalsificationResult.from_json(r.to_json())
        assert r2.status == "COUNTEREXAMPLE_FOUND"
        assert r2.counterexample == [0.1, 0.2]
        assert r2.violation_magnitude == pytest.approx(0.5)

    def test_json_roundtrip_no_counterexample(self):
        r = FalsificationResult(
            status="NO_COUNTEREXAMPLE_FOUND",
            property_name="non_negativity",
            counterexample=None,
            violation_magnitude=None,
            n_restarts=100,
            runtime_s=2.5,
            model_name="test",
        )
        r2 = FalsificationResult.from_json(r.to_json())
        assert r2.status == "NO_COUNTEREXAMPLE_FOUND"
        assert r2.counterexample is None
        assert r2.violation_magnitude is None

    def test_to_json_is_string(self):
        r = FalsificationResult(
            status="NO_COUNTEREXAMPLE_FOUND",
            property_name="p",
            counterexample=None,
            violation_magnitude=None,
            n_restarts=5,
            runtime_s=0.1,
            model_name="m",
        )
        assert isinstance(r.to_json(), str)


# ── falsify() API ─────────────────────────────────────────────────────────────

class TestFalsifyAPI:
    def test_returns_falsification_result(self):
        model = _TinyMLP()
        spec = _make_spec(model)
        prop = NonNegativity()
        result = falsify(spec, prop, n_restarts=5, n_steps=20, seed=0)
        assert isinstance(result, FalsificationResult)
        assert result.status in {"COUNTEREXAMPLE_FOUND", "NO_COUNTEREXAMPLE_FOUND"}

    def test_non_negativity_property(self):
        model = _NonNegMLP()
        spec = _make_spec(model)
        prop = NonNegativity()
        result = falsify(spec, prop, n_restarts=20, n_steps=50, seed=42)
        # Final ReLU guarantees no negative output → PGD should find nothing
        assert result.status == "NO_COUNTEREXAMPLE_FOUND"
        assert result.violation_magnitude is None

    def test_conservation_property(self):
        model = _TinyMLP()
        spec = _make_spec(model)
        prop = Conservation(
            coefficients=np.array([1.0, 1.0], dtype=np.float32),
            target=0.0,
            epsilon=1e-9,
        )
        result = falsify(spec, prop, n_restarts=20, n_steps=100, seed=42)
        assert isinstance(result, FalsificationResult)

    def test_seed_reproducibility(self):
        model = _TinyMLP()
        spec = _make_spec(model)
        prop = Conservation(
            coefficients=np.array([1.0, 1.0], dtype=np.float32),
            target=0.0,
            epsilon=1e-9,
        )
        r1 = falsify(spec, prop, n_restarts=10, n_steps=50, seed=99)
        r2 = falsify(spec, prop, n_restarts=10, n_steps=50, seed=99)
        # Same seed → same violation magnitude
        assert r1.violation_magnitude == r2.violation_magnitude


# ── algebraic conservation (Model B behaviour) ────────────────────────────────

class TestAlgebraicConservation:
    def test_exact_conservation_zero_violation(self):
        """Model B: sum(output) = 0 algebraically → PGD finds no violation."""
        model = _ConservationMLP()
        spec = _make_spec(model)
        prop = Conservation(
            coefficients=np.array([1.0, 1.0], dtype=np.float32),
            target=0.0,
            epsilon=1e-9,
        )
        result = falsify(spec, prop, n_restarts=50, n_steps=200, seed=42)
        # violation_magnitude should be 0 (or very close to it)
        if result.violation_magnitude is not None:
            assert result.violation_magnitude < 1e-5, (
                f"Expected near-zero violation for exact conservation model, "
                f"got {result.violation_magnitude}"
            )

    def test_mlp_nonzero_violation(self):
        """Model A: MLP without conservation → PGD should find a violation."""
        model = _TinyMLP()
        spec = _make_spec(model)
        prop = Conservation(
            coefficients=np.array([1.0, 1.0], dtype=np.float32),
            target=0.0,
            epsilon=1e-9,
        )
        result = falsify(spec, prop, n_restarts=50, n_steps=200, seed=42)
        # MLP output sum is unconstrained → should find a violation
        assert result.status == "COUNTEREXAMPLE_FOUND"
        assert result.violation_magnitude is not None
        assert result.violation_magnitude > 0


# ── SIR/Particle checkpoint loading ──────────────────────────────────────────

class TestSirParticleFalsification:
    """Integration tests that load real checkpoints."""

    @pytest.fixture(scope="class")
    def ckpt_dir(self):
        base = Path(__file__).parent.parent / "verified-scientific-ml" / "models" / "checkpoints"
        if not base.exists():
            pytest.skip("verified-scientific-ml checkpoints not found")
        return base

    def test_sir_geometric_near_zero_violation(self, ckpt_dir):
        """SIR Model B (geometric): conservation is exact — PGD should find ~0 violation."""
        from case_studies.sir_spring.delta_model import DeltaModel
        from case_studies.sir_spring.models.sir import SirGeometricMLP
        from case_studies.sir_spring.properties import sir_population_conservation

        base = SirGeometricMLP(hidden_dim=16)
        state = torch.load(str(ckpt_dir / "sir_geometric.pt"), map_location="cpu", weights_only=True)
        base.load_state_dict(state)
        base.eval()

        delta = DeltaModel(base, n_dims=3)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        torch.save(delta.state_dict(), tmp)
        fresh = DeltaModel(SirGeometricMLP(hidden_dim=16), n_dims=3)
        spec = ModelSpec.from_checkpoint(
            fresh, tmp,
            input_lb=[0.05, 0.01, 0.01],
            input_ub=[0.95, 0.60, 0.90],
            name="sir_geometric_delta",
        )
        Path(tmp).unlink(missing_ok=True)

        prop = sir_population_conservation(epsilon=1e-9)
        result = falsify(spec, prop, n_restarts=30, n_steps=100, seed=42)

        if result.violation_magnitude is not None:
            assert result.violation_magnitude < 1e-4, (
                f"SIR Model B should have near-zero violation; got {result.violation_magnitude}"
            )

    def test_sir_mlp_finds_violation(self, ckpt_dir):
        """SIR Model A (MLP): no conservation structure — PGD should find a violation."""
        from case_studies.sir_spring.delta_model import DeltaModel
        from case_studies.sir_spring.models.sir import SirMLP
        from case_studies.sir_spring.properties import sir_population_conservation

        base = SirMLP(hidden_dim=16)
        state = torch.load(str(ckpt_dir / "sir_mlp.pt"), map_location="cpu", weights_only=True)
        base.load_state_dict(state)
        base.eval()

        delta = DeltaModel(base, n_dims=3)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        torch.save(delta.state_dict(), tmp)
        fresh = DeltaModel(SirMLP(hidden_dim=16), n_dims=3)
        spec = ModelSpec.from_checkpoint(
            fresh, tmp,
            input_lb=[0.05, 0.01, 0.01],
            input_ub=[0.95, 0.60, 0.90],
            name="sir_mlp_delta",
        )
        Path(tmp).unlink(missing_ok=True)

        prop = sir_population_conservation(epsilon=1e-9)
        result = falsify(spec, prop, n_restarts=30, n_steps=100, seed=42)
        assert result.status == "COUNTEREXAMPLE_FOUND"
        assert result.violation_magnitude is not None
        assert result.violation_magnitude > 0.01
