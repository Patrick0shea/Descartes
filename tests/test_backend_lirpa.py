"""
Tests for the LiRPA verification backend.

These tests use tiny networks with deterministic weights so the expected
result is clear.  They are skipped if auto_LiRPA is not installed.
"""

import numpy as np
import pytest

try:
    from auto_LiRPA import BoundedModule  # noqa: F401
    HAS_LIRPA = True
except ImportError:
    HAS_LIRPA = False

pytestmark = pytest.mark.skipif(not HAS_LIRPA, reason="auto_LiRPA not installed")

from verifier import Conservation, LiRPABackend, NonNegativity, RangeBound


CROWN_CONFIG = {"method": "CROWN"}
IBP_CONFIG = {"method": "IBP"}


class TestLiRPANonNegativity:
    def test_proven_with_final_relu(self, spec_nonneg):
        """
        Network with final ReLU: IBP propagates max(lb_pre, 0) >= 0 exactly.
        IBP always proves non-negativity for any network with a final ReLU,
        regardless of hidden-layer weights.  (CROWN uses a global linear
        relaxation that can give negative lower bounds even with a final ReLU;
        IBP is the right tool for this specific structural guarantee.)
        """
        backend = LiRPABackend()
        result = backend.verify(spec_nonneg, NonNegativity(), IBP_CONFIG)
        assert result.status == "PROVEN", result.notes
        assert result.bound_proven is not None
        assert result.bound_proven >= 0.0

    def test_inconclusive_with_negative_bias(self, spec_negative):
        """
        Network with large negative biases and no final ReLU.
        Outputs are provably negative; LiRPA must return INCONCLUSIVE
        (relaxation correctly shows lb < 0, but LiRPA never returns
        COUNTEREXAMPLE — that's Marabou's job).
        """
        backend = LiRPABackend()
        result = backend.verify(spec_negative, NonNegativity(), CROWN_CONFIG)
        assert result.status == "INCONCLUSIVE", result.notes
        assert result.bound_proven is None


class TestLiRPAConservation:
    def test_proven_constant_sum(self, spec_constant_sum):
        """
        Network always outputs [0.5, 0.5].  Sum = 1.0 exactly.
        Conservation(coeff=[1,1], target=1, epsilon=1e-4) must be PROVEN.
        For a linear model (no ReLU) CROWN gives exact bounds.
        """
        backend = LiRPABackend()
        prop = Conservation(
            coefficients=np.array([1.0, 1.0]),
            target=1.0,
            epsilon=1e-4,
            label="sum_one",
        )
        result = backend.verify(spec_constant_sum, prop, CROWN_CONFIG)
        assert result.status == "PROVEN", result.notes
        assert result.bound_proven is not None
        assert result.bound_proven <= 1e-4

    def test_inconclusive_violated_conservation(self, spec_violated):
        """
        Network output = input, sum can range from -2 to 2.
        Conservation(sum=1, epsilon=0.1) cannot be proven; INCONCLUSIVE.
        """
        backend = LiRPABackend()
        prop = Conservation(
            coefficients=np.array([1.0, 1.0]),
            target=1.0,
            epsilon=0.1,
            label="sum_one",
        )
        result = backend.verify(spec_violated, prop, CROWN_CONFIG)
        assert result.status == "INCONCLUSIVE", result.notes


class TestLiRPARangeBound:
    def test_proven_constant_output(self, spec_constant_sum):
        """
        Constant [0.5, 0.5] output: sum = 1.0.  Range [0.9, 1.1] must be PROVEN.
        """
        backend = LiRPABackend()
        prop = RangeBound(
            coefficients=np.array([1.0, 1.0]),
            lower=0.9,
            upper=1.1,
            label="sum_range",
        )
        result = backend.verify(spec_constant_sum, prop, CROWN_CONFIG)
        assert result.status == "PROVEN", result.notes


class TestLiRPAResultFields:
    def test_result_has_required_fields(self, spec_nonneg):
        backend = LiRPABackend()
        result = backend.verify(spec_nonneg, NonNegativity(), IBP_CONFIG)
        assert result.backend == "lirpa"
        assert result.property_name == "non_negativity"
        assert result.model_name == "nonneg_model"
        assert isinstance(result.input_lb, list)
        assert isinstance(result.input_ub, list)
        assert result.runtime_s > 0.0

    def test_unsupported_property_raises(self, spec_nonneg):
        from verifier.property import Property
        class UnknownProp(Property):
            @property
            def name(self): return "unknown"
            def description(self): return "unknown"
        backend = LiRPABackend()
        with pytest.raises(ValueError, match="not support"):
            backend.verify(spec_nonneg, UnknownProp(), CROWN_CONFIG)
