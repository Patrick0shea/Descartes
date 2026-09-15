"""Synthetic calorimeter-shower-like data: a 2D Gaussian bump plus noise,
clipped to be non-negative, scaled so the total sum falls in a target range.

Not real shower physics -- just something with the right qualitative shape
(non-negative, spatially concentrated, roughly conserved total) to train
the toy generator on.
"""

from __future__ import annotations

import numpy as np

SUM_TARGET_MIN = 8.0
SUM_TARGET_MAX = 12.0


def make_grid(grid_size: int, rng: np.random.Generator) -> np.ndarray:
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(xs, ys)

    cx, cy = rng.uniform(-0.4, 0.4, size=2)
    sigma = rng.uniform(0.2, 0.45)
    amplitude = rng.uniform(0.8, 1.2)

    bump = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    noise = rng.normal(0, 0.05, size=bump.shape)
    grid = np.clip(bump + noise, 0.0, None)

    total = grid.sum()
    if total > 1e-8:
        target = rng.uniform(SUM_TARGET_MIN, SUM_TARGET_MAX)
        grid = grid * (target / total)
    return grid.astype(np.float32)


def make_dataset(n_samples: int, grid_size: int, latent_dim: int, seed: int = 0):
    """Returns (z, y): random latents and matched synthetic shower grids.

    z is not "inverted" from y (there is no real generative link) -- this
    is only meant to give the toy generator plausible-looking output
    targets to train towards, not a physically faithful simulator.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1, size=(n_samples, latent_dim)).astype(np.float32)
    y = np.stack([make_grid(grid_size, rng) for _ in range(n_samples)])
    y = y.reshape(n_samples, grid_size * grid_size)
    return z, y
