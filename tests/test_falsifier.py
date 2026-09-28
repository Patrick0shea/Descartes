"""Tests for the PGD falsification baseline."""

import numpy as np
import pytest

from verifier import Conservation, FalsificationResult, NonNegativity, falsify


class TestFalsifyNonNegativity:
    def test_finds_violation_negative_model(self, spec_negative):
        """Constant [-3,-3] output: PGD should find a violation immediately."""
        result = falsify(spec_negative, NonNegativity(), n_restarts=5, n_steps=10)
        assert result.status == "COUNTEREXAMPLE_FOUND"
        assert result.counterexample is not None
        assert result.violation_magnitude is not None
        assert result.violation_magnitude > 0.0

    def test_no_violation_nonneg_model(self, spec_nonneg):
        """
        Model with final ReLU: PGD should not find a violation.
        (Not a proof — just confirms PGD can't break it.)
        """
        result = falsify(spec_nonneg, NonNegativity(), n_restarts=20, n_steps=50)
        assert result.status == "NO_COUNTEREXAMPLE_FOUND"


class TestFalsifyConservation:
    def test_finds_violation_violated_model(self, spec_violated):
        """Output = input, sum varies: PGD should find a conservation violation."""
        prop = Conservation(
            coefficients=np.array([1.0, 1.0]),
            target=1.0,
            epsilon=0.1,
        )
        result = falsify(spec_violated, prop, n_restarts=20, n_steps=100)
        assert result.status == "COUNTEREXAMPLE_FOUND"
        assert result.violation_magnitude is not None

    def test_no_violation_constant_sum(self, spec_constant_sum):
        """Constant [0.5,0.5] sum=1: PGD cannot find a violation."""
        prop = Conservation(
            coefficients=np.array([1.0, 1.0]),
            target=1.0,
            epsilon=1e-3,
        )
        result = falsify(spec_constant_sum, prop, n_restarts=20, n_steps=100)
        assert result.status == "NO_COUNTEREXAMPLE_FOUND"


class TestFalsifierResultFields:
    def test_result_is_self_describing(self, spec_negative):
        result = falsify(
            spec_negative,
            NonNegativity(),
            n_restarts=3,
            n_steps=5,
            seed=123,
        )
        assert result.model_name == "negative_model"
        assert result.n_restarts == 3
        assert result.seed == 123
        assert result.timestamp != ""
        assert result.runtime_s > 0.0

    def test_json_roundtrip(self, spec_negative):
        result = falsify(spec_negative, NonNegativity(), n_restarts=3, n_steps=5)
        r2 = FalsificationResult.from_json(result.to_json())
        assert r2.status == result.status
        assert r2.model_name == result.model_name
