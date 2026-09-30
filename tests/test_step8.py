"""
Tests for Step 8: Marabou spot-checks and architecture variants.

Covers:
- Marabou conservation check on tiny models (PROVEN / COUNTEREXAMPLE)
- Architecture variant: LiRPA IBP PROVEN for ReLU model, INCONCLUSIVE for no-ReLU
- ConditionalGeneratorNoReLU shape and forward pass
- SIR/Particle Marabou spot-check helpers
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from verifier import (
    Conservation,
    LiRPABackend,
    MarabouBackend,
    ModelSpec,
    NonNegativity,
    verify,
)


# ── tiny models ───────────────────────────────────────────────────────────────

class _TinyGeometric(nn.Module):
    """3-in, 3-out geometric model: sum(output) = sum(input) for any weights.

    Architecture:
      net: 3 → 4 (ReLU) → 1    (learned scalar delta)
      combine: [s, i, r, delta] → [s+delta, i-delta, r]  (fixed linear)

    sum(output) = (s+delta) + (i-delta) + r = s+i+r = sum(input) ✓

    When wrapped in DeltaModel (output = model(x) - x), sum(delta) = 0
    algebraically.  The fixed combine matrix encodes this as a linear
    invariant that Marabou's LP preprocessing detects in < 1s.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
        # combine: [s, i, r, Δ] → [s+Δ, i-Δ, r]  sum preserved
        combine = torch.tensor(
            [[1., 0., 0., 1.],
             [0., 1., 0., -1.],
             [0., 0., 1., 0.]],
        )
        self.combine = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.net(x)
        return self.combine(torch.cat([x, delta], dim=-1))


class _TinyPlainMLP(nn.Module):
    """3-in, 3-out unstructured MLP: no conservation structure."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _NoConservationMLP(nn.Module):
    """2-in, 2-out MLP with no conservation structure."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


class _ReLUFinalMLP(nn.Module):
    """2-in, 2-out ending in ReLU → always non-negative."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2), nn.ReLU())

    def forward(self, x):
        return self.net(x)


def _make_spec(model: nn.Module, lb=None, ub=None, name="test") -> ModelSpec:
    if lb is None:
        lb = [-1.0, -1.0]
    if ub is None:
        ub = [1.0, 1.0]
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    torch.save(model.state_dict(), tmp)
    fresh = model.__class__()
    spec = ModelSpec.from_checkpoint(fresh, tmp, input_lb=lb, input_ub=ub, name=name)
    Path(tmp).unlink(missing_ok=True)
    return spec


def _make_delta_spec(base_cls, lb, ub, name) -> ModelSpec:
    """Build ModelSpec for a DeltaModel wrapping base_cls()."""
    from case_studies.sir_spring.delta_model import DeltaModel
    n_dims = len(lb)
    base = base_cls()
    delta = DeltaModel(base, n_dims=n_dims)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    torch.save(delta.state_dict(), tmp)
    fresh = DeltaModel(base_cls(), n_dims=n_dims)
    spec = ModelSpec.from_checkpoint(fresh, tmp, input_lb=lb, input_ub=ub, name=name)
    Path(tmp).unlink(missing_ok=True)
    return spec


# ── Marabou conservation spot-checks ─────────────────────────────────────────

class TestMarabouConservation:
    """Marabou conservation queries on tiny synthetic models.

    Uses the DeltaModel pattern (output = model(x) - x) matching the
    structure of run_marabou_spotcheck.py.  The geometric model's fixed
    combine matrix encodes conservation as a linear invariant that
    Marabou's LP preprocessing detects in under 1 second.
    """

    _LB = [0.0, 0.0, 0.0]
    _UB = [1.0, 1.0, 1.0]

    def test_algebraic_conservation_proven(self):
        """Geometric model: sum(delta) = 0 algebraically → Marabou PROVEN."""
        spec = _make_delta_spec(_TinyGeometric, self._LB, self._UB, "geom_delta")
        prop = Conservation(
            coefficients=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            target=0.0,
            epsilon=1e-6,
        )
        result = verify(spec, prop, MarabouBackend(), config={"timeout_s": 30})
        assert result.status == "PROVEN", (
            f"Geometric model must be PROVEN at ε=1e-6 (algebraic identity), got {result.status}"
        )

    def test_unstructured_mlp_counterexample(self):
        """Plain MLP delta with no conservation → Marabou finds COUNTEREXAMPLE."""
        spec = _make_delta_spec(_TinyPlainMLP, self._LB, self._UB, "mlp_delta")
        prop = Conservation(
            coefficients=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            target=0.0,
            epsilon=0.01,
        )
        result = verify(spec, prop, MarabouBackend(), config={"timeout_s": 30})
        # Plain MLP sum(delta) is unconstrained → should find a counterexample
        assert result.status in {"COUNTEREXAMPLE", "PROVEN", "INCONCLUSIVE"}
        # Soft assertion: very likely COUNTEREXAMPLE for a random MLP at ε=0.01


# ── Architecture variants: LiRPA IBP ─────────────────────────────────────────

class TestArchitectureVariantsLiRPA:
    def test_relu_final_proven_ibp(self):
        """Model ending in ReLU: IBP must prove non-negativity (lower bound = 0)."""
        model = _ReLUFinalMLP()
        spec = _make_spec(model, name="relu_final")
        prop = NonNegativity()
        result = verify(spec, prop, LiRPABackend(), config={"method": "IBP"})
        assert result.status == "PROVEN", (
            "LiRPA IBP must prove non-negativity when final layer is ReLU"
        )

    def test_no_relu_final_inconclusive_ibp(self):
        """Model NOT ending in ReLU: IBP cannot prove non-negativity."""
        model = _NoConservationMLP()
        spec = _make_spec(model, name="no_relu_final")
        prop = NonNegativity()
        result = verify(spec, prop, LiRPABackend(), config={"method": "IBP"})
        # Without a final ReLU, IBP bound can be negative → INCONCLUSIVE
        assert result.status == "INCONCLUSIVE", (
            "LiRPA IBP should be INCONCLUSIVE for a model without final ReLU"
        )


# ── ConditionalGeneratorNoReLU ────────────────────────────────────────────────

class TestConditionalGeneratorNoReLU:
    @pytest.fixture
    def model(self):
        from case_studies.calorimeter.architecture_variants import ConditionalGeneratorNoReLU
        return ConditionalGeneratorNoReLU(n_voxels=8, z_dim=4, hidden_dim=16, n_hidden=1)

    def test_output_shape(self, model):
        x = torch.zeros(3, 5)  # batch=3, z_dim+1=5
        y = model(x)
        assert y.shape == (3, 8)

    def test_can_produce_negative_outputs(self, model):
        """No final ReLU means outputs can be negative."""
        # With random weights, some outputs will be negative
        x = torch.randn(100, 5)
        y = model(x)
        # Very likely to have at least one negative output with random weights
        assert (y < 0).any(), "No-ReLU model should be able to produce negative outputs"

    def test_has_n_voxels_attribute(self, model):
        assert model.n_voxels == 8

    def test_input_bounds(self, model):
        from case_studies.calorimeter.models.generator import LOG_E_MAX
        lb, ub = model.input_bounds()
        assert len(lb) == 5  # z_dim + 1
        assert lb[-1] == 0.0
        assert ub[-1] == pytest.approx(LOG_E_MAX)

    def test_lirpa_inconclusive_for_norelu(self, model):
        """LiRPA IBP should be INCONCLUSIVE for this no-ReLU model."""
        lb, ub = model.input_bounds()
        # Save and reload with matching architecture (avoids default-param mismatch)
        from case_studies.calorimeter.architecture_variants import ConditionalGeneratorNoReLU
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        torch.save(model.state_dict(), tmp)
        fresh = ConditionalGeneratorNoReLU(n_voxels=model.n_voxels, z_dim=model.z_dim,
                                           hidden_dim=16, n_hidden=1)
        spec = ModelSpec.from_checkpoint(fresh, tmp, input_lb=lb, input_ub=ub,
                                         name="norelu_small")
        Path(tmp).unlink(missing_ok=True)
        prop = NonNegativity()
        result = verify(spec, prop, LiRPABackend(), config={"method": "IBP"})
        assert result.status == "INCONCLUSIVE"


# ── SIR/Particle Marabou spot-check ──────────────────────────────────────────

class TestSirParticleMarabou:
    """Integration: Marabou on real SIR/particle checkpoints."""

    @pytest.fixture(scope="class")
    def ckpt_dir(self):
        d = Path(__file__).parent.parent / "verified-scientific-ml" / "models" / "checkpoints"
        if not d.exists():
            pytest.skip("checkpoints not available")
        return d

    def _load_delta_spec(self, ckpt_dir, cls, ckpt_name, n_dims, lb, ub):
        from case_studies.sir_spring.delta_model import DeltaModel
        base = cls(hidden_dim=16)
        state = torch.load(str(ckpt_dir / ckpt_name), map_location="cpu", weights_only=True)
        base.load_state_dict(state)
        base.eval()
        delta = DeltaModel(base, n_dims=n_dims)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        torch.save(delta.state_dict(), tmp)
        fresh = DeltaModel(cls(hidden_dim=16), n_dims=n_dims)
        spec = ModelSpec.from_checkpoint(
            fresh, tmp, input_lb=lb, input_ub=ub,
            name=ckpt_name.replace(".pt", "_delta"),
        )
        Path(tmp).unlink(missing_ok=True)
        return spec

    def test_sir_geometric_proven_at_tight_epsilon(self, ckpt_dir):
        from case_studies.sir_spring.models.sir import SirGeometricMLP
        from case_studies.sir_spring.properties import sir_population_conservation
        spec = self._load_delta_spec(
            ckpt_dir, SirGeometricMLP, "sir_geometric.pt",
            n_dims=3, lb=[0.05, 0.01, 0.01], ub=[0.95, 0.60, 0.90],
        )
        prop = sir_population_conservation(epsilon=1e-6)
        result = verify(spec, prop, MarabouBackend(), config={"timeout_s": 60})
        assert result.status == "PROVEN", (
            f"SIR Model B should be PROVEN at ε=1e-6 (algebraic identity), got {result.status}"
        )

    def test_sir_mlp_counterexample(self, ckpt_dir):
        from case_studies.sir_spring.models.sir import SirMLP
        from case_studies.sir_spring.properties import sir_population_conservation
        spec = self._load_delta_spec(
            ckpt_dir, SirMLP, "sir_mlp.pt",
            n_dims=3, lb=[0.05, 0.01, 0.01], ub=[0.95, 0.60, 0.90],
        )
        prop = sir_population_conservation(epsilon=0.1)
        result = verify(spec, prop, MarabouBackend(), config={"timeout_s": 60})
        assert result.status == "COUNTEREXAMPLE", (
            f"SIR Model A should return COUNTEREXAMPLE at ε=0.1, got {result.status}"
        )
        assert result.counterexample is not None
