"""
Tests for the Marabou verification backend.

These tests use tiny networks where the answer is obvious and Marabou
should solve quickly.  They are skipped if maraboupy is not installed.
"""

import numpy as np
import pytest

try:
    import maraboupy  # noqa: F401
    HAS_MARABOU = True
except ImportError:
    HAS_MARABOU = False

pytestmark = pytest.mark.skipif(not HAS_MARABOU, reason="maraboupy not installed")

from verifier import Conservation, MarabouBackend, NonNegativity


CONFIG = {"timeout_s": 30}


class TestMarabouNonNegativity:
    def test_proven_constant_nonneg(self, constant_sum_model, bounds_unit_square):
        """
        Constant [0.5, 0.5] output: both outputs are always positive.
        Marabou should return PROVEN (UNSAT for both "can output_i < 0?" queries).
        """
        import numpy as np
        from verifier import ModelSpec
        lb, ub = bounds_unit_square
        spec = ModelSpec(
            model=constant_sum_model, input_lb=lb, input_ub=ub,
            name="constant_sum"
        )
        backend = MarabouBackend()
        result = backend.verify(spec, NonNegativity(), CONFIG)
        assert result.status == "PROVEN", result.notes

    def test_counterexample_negative_bias(self, negative_bias_model, bounds_unit_square):
        """
        Output is always [-3, -3]: Marabou should find a counterexample immediately.
        """
        from verifier import ModelSpec
        lb, ub = bounds_unit_square
        spec = ModelSpec(
            model=negative_bias_model, input_lb=lb, input_ub=ub,
            name="negative_model"
        )
        backend = MarabouBackend()
        result = backend.verify(spec, NonNegativity(), CONFIG)
        assert result.status == "COUNTEREXAMPLE", result.notes
        assert result.counterexample is not None


class TestMarabouConservation:
    def test_proven_constant_sum(self, constant_sum_model, bounds_unit_square):
        """
        Constant [0.5, 0.5] output: sum = 1.0 exactly.
        Conservation(sum=1, epsilon=0.01) must be PROVEN.
        """
        from verifier import ModelSpec
        lb, ub = bounds_unit_square
        spec = ModelSpec(
            model=constant_sum_model, input_lb=lb, input_ub=ub,
            name="constant_sum"
        )
        backend = MarabouBackend()
        prop = Conservation(
            coefficients=np.array([1.0, 1.0]),
            target=1.0,
            epsilon=0.01,
            label="sum_one",
        )
        result = backend.verify(spec, prop, CONFIG)
        assert result.status == "PROVEN", result.notes

    def test_counterexample_violated(self, violated_conservation_model, bounds_unit_square):
        """
        Output = input, sum can range from -2 to 2.
        Conservation(sum=1, epsilon=0.1) is violated; expect COUNTEREXAMPLE.
        """
        from verifier import ModelSpec
        lb, ub = bounds_unit_square
        spec = ModelSpec(
            model=violated_conservation_model, input_lb=lb, input_ub=ub,
            name="violated"
        )
        backend = MarabouBackend()
        prop = Conservation(
            coefficients=np.array([1.0, 1.0]),
            target=1.0,
            epsilon=0.1,
            label="sum_one",
        )
        result = backend.verify(spec, prop, CONFIG)
        assert result.status == "COUNTEREXAMPLE", result.notes
        assert result.counterexample is not None
