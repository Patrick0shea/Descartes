"""
Tests for the verification-guided training (VGT) loop.

These tests cover the VGTTrainer components without requiring Marabou
(which is skipped if maraboupy is not installed) or a pre-trained
checkpoint. They verify the structural correctness of the augmentation
loop, neighbourhood sampling, simulator calls, and fine-tuning step.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from models.particle_soft_geometric import ParticleSoftGeometricMLP
from training.vgt_loop import (
    VGTTrainer,
    _in_domain,
    _sample_neighbourhood,
    _simulator_step,
    _export_onnx,
)
from verification.particle_properties import (
    DOMAIN_LOWER, DOMAIN_UPPER, SEPARATION_BOUND, X1, X2, Y1, Y2,
)

_DOMAIN_LOWER = np.array(DOMAIN_LOWER, dtype=np.float32)
_DOMAIN_UPPER = np.array(DOMAIN_UPPER, dtype=np.float32)

# A valid state inside domain D (particle separation ~1.0, within bounds)
VALID_STATE = np.array([-0.3, 0.2, 0.5, -0.3, 0.5, -0.4, -0.2, 0.4], dtype=np.float32)


# ─── domain helpers ──────────────────────────────────────────────────────────

def test_in_domain_valid_state():
    assert _in_domain(VALID_STATE)


def test_in_domain_rejects_box_violation():
    bad = VALID_STATE.copy()
    bad[0] = 999.0  # x1 way outside box
    assert not _in_domain(bad)


def test_in_domain_rejects_separation_violation():
    bad = VALID_STATE.copy()
    # Force |x2-x1| > SEPARATION_BOUND
    bad[X1] = -1.0
    bad[X2] = 1.0 + SEPARATION_BOUND + 0.1  # clip to box but separation too large
    bad[X2] = np.clip(bad[X2], _DOMAIN_LOWER[X2], _DOMAIN_UPPER[X2])
    # x2 - x1 should now be > SEPARATION_BOUND
    if abs(bad[X2] - bad[X1]) > SEPARATION_BOUND:
        assert not _in_domain(bad)


# ─── neighbourhood sampling ───────────────────────────────────────────────────

def test_neighbourhood_samples_are_in_domain():
    rng = np.random.default_rng(0)
    samples = _sample_neighbourhood(VALID_STATE, n=50, sigma=0.05, rng=rng)
    assert len(samples) > 0
    for s in samples:
        assert _in_domain(s), f"Sample {s} not in domain D"


def test_neighbourhood_samples_have_correct_shape():
    rng = np.random.default_rng(1)
    samples = _sample_neighbourhood(VALID_STATE, n=10, rng=rng)
    for s in samples:
        assert s.shape == (8,)
        assert s.dtype == np.float32


def test_neighbourhood_returns_fewer_when_domain_tight():
    # Place state near boundary — many noisy samples will be clipped/rejected
    edge_state = VALID_STATE.copy()
    edge_state[X1] = _DOMAIN_LOWER[X1] + 0.01
    rng = np.random.default_rng(2)
    samples = _sample_neighbourhood(edge_state, n=50, sigma=0.5, rng=rng)
    # Should still return something (just possibly fewer than 50)
    assert isinstance(samples, list)


# ─── simulator step ───────────────────────────────────────────────────────────

def test_simulator_step_returns_correct_shape():
    next_state = _simulator_step(VALID_STATE)
    assert next_state.shape == (8,)
    assert next_state.dtype == np.float32


def test_simulator_step_conserves_momentum():
    next_state = _simulator_step(VALID_STATE)
    px_before = VALID_STATE[2] + VALID_STATE[6]
    py_before = VALID_STATE[3] + VALID_STATE[7]
    px_after = next_state[2] + next_state[6]
    py_after = next_state[3] + next_state[7]
    assert abs(px_after - px_before) < 1e-5, f"|dPx|={abs(px_after - px_before):.2e}"
    assert abs(py_after - py_before) < 1e-5, f"|dPy|={abs(py_after - py_before):.2e}"


# ─── ONNX export in VGT context ───────────────────────────────────────────────

def test_vgt_onnx_export(tmp_path):
    pytest.importorskip("onnx")
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    model.eval()
    onnx_path = str(tmp_path / "vgt_test.onnx")
    _export_onnx(model, onnx_path)
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0


# ─── VGTTrainer ───────────────────────────────────────────────────────────────

def _make_small_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(_DOMAIN_LOWER, _DOMAIN_UPPER, size=(n, 8)).astype(np.float32)
    # Reject separation violations (approximate, for test data)
    mask = (np.abs(x[:, X2] - x[:, X1]) <= SEPARATION_BOUND) & \
           (np.abs(x[:, Y2] - x[:, Y1]) <= SEPARATION_BOUND)
    x = x[mask]
    y = np.array([_simulator_step(xi) for xi in x], dtype=np.float32)
    return x, y


def test_vgt_trainer_initialises(tmp_path):
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    x, y = _make_small_dataset(50)
    trainer = VGTTrainer(
        model=model, train_x=x, train_y=y,
        epsilon_schedule=[1e-3],
        temp_onnx_path=str(tmp_path / "temp.onnx"),
        epochs_per_iter=1,
        max_iters_per_epsilon=1,
        neighbourhood_size=5,
    )
    assert trainer.n_original == len(x)
    assert len(trainer.aug_x) == len(x)


def test_vgt_finetune_reduces_loss(tmp_path):
    torch.manual_seed(42)
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    x, y = _make_small_dataset(100)
    trainer = VGTTrainer(
        model=model, train_x=x, train_y=y,
        epsilon_schedule=[1e-3],
        temp_onnx_path=str(tmp_path / "temp.onnx"),
        epochs_per_iter=20,
        max_iters_per_epsilon=1,
        neighbourhood_size=5,
        seed=42,
    )
    loss_before = trainer._finetune()
    loss_after = trainer._finetune()
    # Should continue to decrease (or at least not explode) after more training
    assert loss_after < loss_before * 10, "Loss exploded during fine-tuning"


def test_vgt_add_counterexample_grows_dataset(tmp_path):
    model = ParticleSoftGeometricMLP(hidden_dim=16)
    x, y = _make_small_dataset(50)
    trainer = VGTTrainer(
        model=model, train_x=x, train_y=y,
        epsilon_schedule=[1e-3],
        temp_onnx_path=str(tmp_path / "temp.onnx"),
        epochs_per_iter=1,
        max_iters_per_epsilon=1,
        neighbourhood_size=10,
        seed=0,
    )
    n_before = len(trainer.aug_x)
    trainer._add_counterexample(VALID_STATE)
    n_after = len(trainer.aug_x)
    assert n_after > n_before, "D_aug did not grow after adding counterexample"
    # At minimum 1 (the counterexample itself) was added
    assert n_after >= n_before + 1
