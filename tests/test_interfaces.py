"""Tests for ModelSpec, Property, and VerificationResult interfaces."""

import json
import numpy as np
import pytest
import torch.nn as nn

from verifier import (
    ModelSpec,
    NonNegativity,
    Conservation,
    RangeBound,
    VerificationResult,
    FalsificationResult,
)


# ── ModelSpec ─────────────────────────────────────────────────────────────────

class TestModelSpec:
    def test_basic_construction(self):
        model = nn.Linear(2, 2)
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        spec = ModelSpec(model=model, input_lb=lb, input_ub=ub, name="test")
        assert spec.name == "test"
        np.testing.assert_array_equal(spec.input_lb, lb)
        np.testing.assert_array_equal(spec.input_ub, ub)
        assert spec.checkpoint_path is None
        assert spec.checkpoint_hash is None

    def test_coerces_to_float32(self):
        model = nn.Linear(2, 2)
        spec = ModelSpec(
            model=model,
            input_lb=np.array([0, 0], dtype=np.float64),
            input_ub=np.array([1, 1], dtype=np.float64),
            name="t",
        )
        assert spec.input_lb.dtype == np.float32

    def test_invalid_bounds_shape(self):
        with pytest.raises(ValueError, match="same shape"):
            ModelSpec(
                model=nn.Linear(2, 2),
                input_lb=np.array([0.0]),
                input_ub=np.array([1.0, 1.0]),
                name="t",
            )

    def test_invalid_bounds_order(self):
        with pytest.raises(ValueError, match="<="):
            ModelSpec(
                model=nn.Linear(2, 2),
                input_lb=np.array([1.0, 0.0]),
                input_ub=np.array([0.0, 1.0]),
                name="t",
            )


# ── Properties ────────────────────────────────────────────────────────────────

class TestNonNegativity:
    def test_name(self):
        p = NonNegativity()
        assert p.name == "non_negativity"

    def test_custom_label(self):
        p = NonNegativity(label="my_nn")
        assert p.name == "my_nn"

    def test_description(self):
        p = NonNegativity()
        assert "0" in p.description()


class TestConservation:
    def test_basic(self):
        p = Conservation(
            coefficients=np.array([1.0, 1.0, 1.0]),
            target=1.0,
            epsilon=0.01,
            label="pop",
        )
        assert p.name == "pop"
        assert p.epsilon == 0.01
        assert p.target == 1.0

    def test_coerces_coefficients(self):
        p = Conservation(coefficients=[1, 1, 1], target=1.0, epsilon=0.01)
        assert p.coefficients.dtype == np.float32

    def test_negative_epsilon_rejected(self):
        with pytest.raises(ValueError):
            Conservation(coefficients=[1.0], target=1.0, epsilon=-0.1)


class TestRangeBound:
    def test_basic(self):
        p = RangeBound(
            coefficients=np.array([1.0] * 4),
            lower=4.0,
            upper=20.0,
            label="sum_bound",
        )
        assert p.name == "sum_bound"

    def test_one_sided_lower(self):
        p = RangeBound(coefficients=np.array([1.0]), lower=0.0, upper=None)
        assert p.lower == 0.0
        assert p.upper is None

    def test_both_none_rejected(self):
        with pytest.raises(ValueError):
            RangeBound(coefficients=np.array([1.0]), lower=None, upper=None)


# ── VerificationResult ────────────────────────────────────────────────────────

class TestVerificationResult:
    def _make(self, status="PROVEN"):
        return VerificationResult(
            status=status,
            property_name="non_negativity",
            backend="lirpa",
            bound_proven=0.1,
            counterexample=None,
            input_lb=[0.0, 0.0],
            input_ub=[1.0, 1.0],
            runtime_s=0.05,
            model_name="test_model",
        )

    def test_json_roundtrip(self):
        r = self._make()
        r2 = VerificationResult.from_json(r.to_json())
        assert r2.status == r.status
        assert r2.bound_proven == r.bound_proven
        assert r2.backend == r.backend

    def test_json_is_valid(self):
        r = self._make()
        parsed = json.loads(r.to_json())
        assert parsed["status"] == "PROVEN"
        assert "timestamp" in parsed
        assert "config" in parsed

    def test_counterexample_roundtrip(self):
        r = self._make(status="COUNTEREXAMPLE")
        r.counterexample = [0.5, 0.3]
        r2 = VerificationResult.from_json(r.to_json())
        assert r2.counterexample == [0.5, 0.3]
