"""
Tests for the Step 4 particle formal verification pipeline: ONNX export
equivalence, momentum-property linear-constraint construction, and the
random-search baseline.

The end-to-end Marabou solve tests use a small SYNTHETIC linear network
(not the trained particle_mlp checkpoint) so they run in well under a
second and check the encoding logic against a network whose momentum
behavior is known exactly, rather than depending on how large the real
trained model is (see verification/particle_properties.py and the
README's "verification scalability" note: the real 8-dim network is
expensive to verify exactly, which is itself a relevant finding).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from verification.export_particle_onnx import (
    STATE_DIM,
    check_equivalence,
    load_full_particle_surrogate,
)
from verification.particle_properties import DOMAIN_LOWER, DOMAIN_UPPER, VX1, VX2
from verification.random_search_particle import run_random_test
from verification.verify_particle import _solve_one_sided
import verification.particle_properties as pp


def test_fused_particle_model_matches_original_inference_pipeline():
    model = load_full_particle_surrogate()
    max_diff = check_equivalence(model, n_samples=200, seed=1)
    assert max_diff < 1e-4


def test_domain_matches_data_generation_bounds():
    """The verified domain must be exactly the data-generation sampling box, not a separate choice."""
    from data.generate_particle_data import STATE_BOUNDS

    for (lo, hi), d_lo, d_hi in zip(STATE_BOUNDS, DOMAIN_LOWER, DOMAIN_UPPER):
        assert lo == d_lo
        assert hi == d_hi


def _build_synthetic_momentum_onnx(path: str, shift: float) -> None:
    """
    A tiny linear network computing next_state = state + shift * e_2
    (perturbing only the vx1 output by a constant `shift`, identity
    elsewhere). Its momentum behavior is known exactly:
    Px_next - Px = shift for every input, so |Px_next - Px| <= eps holds
    iff eps >= |shift|. This isolates the constraint-encoding logic from
    the trained model's actual (harder to predict exactly) behavior.
    """
    class Synthetic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(STATE_DIM, STATE_DIM)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(STATE_DIM))
                bias = torch.zeros(STATE_DIM)
                bias[2] = shift  # index 2 = vx1
                self.linear.bias.copy_(bias)

        def forward(self, x):
            return self.linear(x)

    model = Synthetic()
    dummy = torch.zeros((1, STATE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path, input_names=["state"], output_names=["next_state"],
        opset_version=13, dynamo=False,
    )


def test_momentum_property_unsat_when_shift_within_epsilon(tmp_path):
    """shift=0.01, epsilon=0.05: |Px_next-Px|=0.01 <= 0.05 always -> UNSAT both directions."""
    onnx_path = str(tmp_path / "synthetic.onnx")
    _build_synthetic_momentum_onnx(onnx_path, shift=0.01)

    upper = _solve_one_sided(onnx_path, (VX1, VX2), epsilon=0.05, direction="upper")
    lower = _solve_one_sided(onnx_path, (VX1, VX2), epsilon=0.05, direction="lower")
    assert upper["result"] == "UNSAT"
    assert lower["result"] == "UNSAT"


def test_momentum_property_sat_when_shift_exceeds_epsilon(tmp_path):
    """shift=0.1, epsilon=0.05: |Px_next-Px|=0.1 > 0.05 always -> SAT on the upper-violation query."""
    onnx_path = str(tmp_path / "synthetic.onnx")
    _build_synthetic_momentum_onnx(onnx_path, shift=0.1)

    upper = _solve_one_sided(onnx_path, (VX1, VX2), epsilon=0.05, direction="upper")
    assert upper["result"] == "SAT"
    ce = upper["counterexample"]
    assert np.isclose(ce["momentum_error"], 0.1, atol=1e-4)


def test_random_search_particle_runs_and_returns_expected_shape():
    model = load_full_particle_surrogate()
    results = run_random_test(model, pp.MOMENTUM_PROPERTIES, n_samples=2000, seed=42)
    assert len(results) == len(pp.MOMENTUM_PROPERTIES)
    for record, prop in zip(results, pp.MOMENTUM_PROPERTIES):
        assert record["name"] == prop.name
        assert record["n_samples"] == 2000
        assert record["result"] in ("VIOLATION_FOUND", "NO_VIOLATION_FOUND")
