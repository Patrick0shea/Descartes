"""Tests for the Step 3 formal verification pipeline (ONNX export + Marabou)."""

import os

import numpy as np
import pytest
import torch

from verification.export_onnx import check_equivalence, export, load_full_surrogate, ONNX_PATH
from verification.properties import Property
from verification.random_search import run_random_test
from verification.verify import verify_property


@pytest.fixture(scope="module")
def onnx_path():
    export()
    assert os.path.exists(ONNX_PATH)
    return ONNX_PATH


def test_fused_model_matches_original_inference_pipeline():
    model = load_full_surrogate()
    max_diff = check_equivalence(model, n_samples=200, seed=1)
    assert max_diff < 1e-4


def test_loose_bound_property_is_unsat(onnx_path):
    """A generous bound consistent with the physics should have no counterexample."""
    prop = Property(
        name="x_next_upper_loose_test",
        description="x_next <= 3.0 for all (x, v) in the domain",
        output_index=0, bound_type="ub", bound_value=3.0, expect="UNSAT",
    )
    result = verify_property(onnx_path, prop)
    assert result["result"] == "UNSAT"
    assert result["counterexample"] is None
    assert result["verification_time_seconds"] >= 0


def test_violated_property_is_sat_with_valid_counterexample(onnx_path):
    """A deliberately too-tight bound should be violated, with a real counterexample."""
    prop = Property(
        name="x_next_upper_violated_test",
        description="x_next <= 1.0 for all (x, v) in the domain (deliberately false)",
        output_index=0, bound_type="ub", bound_value=1.0, expect="SAT",
    )
    result = verify_property(onnx_path, prop)
    assert result["result"] == "SAT"
    assert result["counterexample"] is not None

    ce = result["counterexample"]
    assert -2.0 - 1e-6 <= ce["x"] <= 2.0 + 1e-6
    assert -2.0 - 1e-6 <= ce["v"] <= 2.0 + 1e-6

    # Independently confirm the counterexample by re-running the real model
    model = load_full_surrogate()
    with torch.no_grad():
        out = model(torch.tensor([[ce["x"], ce["v"]]], dtype=torch.float32)).numpy()[0]
    assert np.isclose(out[0], ce["x_next"], atol=1e-3)
    assert out[0] >= 1.0 - 1e-3  # actually violates the claimed bound


def test_random_search_finds_violation_for_violated_property():
    model = load_full_surrogate()
    prop = Property(
        name="x_next_upper_violated_test",
        description="x_next <= 1.0 for all (x, v) in the domain (deliberately false)",
        output_index=0, bound_type="ub", bound_value=1.0, expect="SAT",
    )
    results = run_random_test(model, [prop], n_samples=5000, seed=42)
    assert results[0]["result"] == "VIOLATION_FOUND"
    assert results[0]["n_violations_found"] > 0


def test_random_search_returns_no_violation_for_loose_property():
    model = load_full_surrogate()
    prop = Property(
        name="x_next_upper_loose_test",
        description="x_next <= 3.0 for all (x, v) in the domain",
        output_index=0, bound_type="ub", bound_value=3.0, expect="UNSAT",
    )
    results = run_random_test(model, [prop], n_samples=5000, seed=42)
    assert results[0]["result"] == "NO_VIOLATION_FOUND"
    assert results[0]["n_violations_found"] == 0
