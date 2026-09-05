"""Tests for the MLP surrogate model and normalization helpers."""

import numpy as np
import torch

from models.normalization import Normalizer
from models.surrogate import MLPSurrogate


def test_mlp_surrogate_output_shape():
    model = MLPSurrogate(hidden_dim=16)
    x = torch.randn(5, 2)
    y = model(x)
    assert y.shape == (5, 2)


def test_normalizer_round_trip():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=[1.0, -2.0], scale=[3.0, 0.5], size=(1000, 2))
    normalizer = Normalizer.fit(data)
    normalized = normalizer.transform(data)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-8)
    recovered = normalizer.inverse_transform(normalized)
    assert np.allclose(recovered, data, atol=1e-8)


def test_normalizer_save_load(tmp_path):
    data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    normalizer = Normalizer.fit(data)
    path = tmp_path / "normalizer.npz"
    normalizer.save(str(path))
    loaded = Normalizer.load(str(path))
    assert np.allclose(loaded.mean, normalizer.mean)
    assert np.allclose(loaded.std, normalizer.std)
