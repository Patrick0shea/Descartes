"""Tests for the runner (verify + save_result)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from auto_LiRPA import BoundedModule  # noqa: F401
    HAS_LIRPA = True
except ImportError:
    HAS_LIRPA = False

pytestmark = pytest.mark.skipif(not HAS_LIRPA, reason="auto_LiRPA not installed")

from verifier import LiRPABackend, NonNegativity, save_result, verify


class TestVerifyRunner:
    def test_adds_traceability_fields(self, spec_nonneg):
        result = verify(
            spec_nonneg,
            NonNegativity(),
            LiRPABackend(),
            config={"method": "CROWN"},
            seed=42,
        )
        assert result.git_commit != ""
        assert result.seed == 42
        assert result.timestamp != ""
        assert result.config == {"method": "CROWN"}

    def test_save_result_creates_file(self, spec_nonneg):
        result = verify(
            spec_nonneg,
            NonNegativity(),
            LiRPABackend(),
            config={"method": "CROWN"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_result(result, output_dir=tmpdir, run_id="test_run")
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["status"] in ("PROVEN", "INCONCLUSIVE", "COUNTEREXAMPLE")
            assert "timestamp" in data
            assert "git_commit" in data

    def test_save_result_creates_directory(self, spec_nonneg):
        result = verify(
            spec_nonneg,
            NonNegativity(),
            LiRPABackend(),
            config={"method": "CROWN"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "results" / "test_domain"
            path = save_result(result, output_dir=nested)
            assert path.exists()
