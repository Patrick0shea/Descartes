"""Standardization (zero mean, unit variance) for [x, v] state vectors."""

from __future__ import annotations

import numpy as np


class Normalizer:
    """Per-column (x, v) standardization: (value - mean) / std."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, data: np.ndarray) -> "Normalizer":
        """Compute per-column mean/std from data of shape (N, 2)."""
        return cls(mean=data.mean(axis=0), std=data.std(axis=0))

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        d = np.load(path)
        return cls(mean=d["mean"], std=d["std"])
