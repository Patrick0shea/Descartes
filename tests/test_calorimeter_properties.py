"""
Tests for calorimeter property factory functions.

These tests check:
- Correct property types are returned
- coefficients shape matches n_voxels
- target / epsilon / bounds values are passed through
- Basic contracts from the Property base class
"""

from __future__ import annotations

import numpy as np
import pytest

from case_studies.calorimeter.properties import (
    energy_conservation,
    total_energy_upper_bound,
    voxel_non_negativity,
)
from verifier.property import Conservation, NonNegativity, RangeBound


class TestVoxelNonNegativity:
    def test_returns_nonneg(self):
        p = voxel_non_negativity()
        assert isinstance(p, NonNegativity)

    def test_name(self):
        p = voxel_non_negativity()
        assert p.name == "voxel_non_negativity"

    def test_description_not_empty(self):
        assert len(voxel_non_negativity().description()) > 0


class TestTotalEnergyUpperBound:
    def test_returns_range_bound(self):
        p = total_energy_upper_bound(368, 5000.0)
        assert isinstance(p, RangeBound)

    def test_coefficients_shape(self):
        p = total_energy_upper_bound(368, 5000.0)
        assert p.coefficients.shape == (368,)

    def test_coefficients_all_ones(self):
        p = total_energy_upper_bound(368, 5000.0)
        np.testing.assert_array_equal(p.coefficients, np.ones(368, dtype=np.float32))

    def test_upper_set(self):
        p = total_energy_upper_bound(368, 5000.0)
        assert p.upper == 5000.0
        assert p.lower is None

    def test_pion_coefficients(self):
        p = total_energy_upper_bound(533, 8000.0)
        assert p.coefficients.shape == (533,)

    def test_name(self):
        p = total_energy_upper_bound(368, 5000.0)
        assert p.name == "total_energy_upper_bound"


class TestEnergyConservation:
    def test_returns_conservation(self):
        p = energy_conservation(368, 1000.0, 500.0)
        assert isinstance(p, Conservation)

    def test_coefficients_shape(self):
        p = energy_conservation(368, 1000.0, 500.0)
        assert p.coefficients.shape == (368,)

    def test_coefficients_all_ones(self):
        p = energy_conservation(368, 1000.0, 500.0)
        np.testing.assert_array_equal(
            p.coefficients, np.ones(368, dtype=np.float32)
        )

    def test_target(self):
        p = energy_conservation(368, 1000.0, 500.0)
        assert p.target == 1000.0

    def test_epsilon(self):
        p = energy_conservation(368, 1000.0, 500.0)
        assert p.epsilon == 500.0

    def test_name(self):
        p = energy_conservation(368, 1000.0, 500.0)
        assert p.name == "energy_conservation"

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            energy_conservation(368, 1000.0, -1.0)

    def test_pion_voxels(self):
        p = energy_conservation(533, 2000.0, 1000.0)
        assert p.coefficients.shape == (533,)
