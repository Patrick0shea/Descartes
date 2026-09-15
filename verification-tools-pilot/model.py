"""Toy calorimeter-shower-like generator: latent vector -> 2D energy grid.

A small feedforward net, 4 linear layers with ReLU between them. The final
layer has NO non-negativity-enforcing activation (no ReLU/softplus on the
output) on purpose: non-negativity of the output grid is a genuine bound-
propagation question for the verifiers to answer, not a tautology baked
into the architecture.

Parameterized by latent_dim and grid_size so the pilot can sweep grid
sizes, and so a real calorimeter architecture can later be swapped in.

Not built here, but worth noting as a follow-up: a variant with
nn.Softplus() (or nn.ReLU()) as the last layer of Generator.net would make
non-negativity architecturally provable rather than merely plausible --
see REPORT.md's closing caveat for why this pilot's SAT (violated)
results on every tool are an expected consequence of not doing that.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def hidden_width(grid_size: int, latent_dim: int, max_params: int = 50_000) -> int:
    """Pick a hidden width so the model stays under max_params for a given grid_size."""
    n_out = grid_size * grid_size
    for h in (256, 192, 160, 128, 96, 64, 48, 32, 24, 16):
        params = (
            (latent_dim * h + h)
            + (h * h + h)
            + (h * h + h)
            + (h * n_out + n_out)
        )
        if params <= max_params:
            return h
    return 8


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 8, grid_size: int = 8, hidden: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        n_out = grid_size * grid_size
        h = hidden if hidden is not None else hidden_width(grid_size, latent_dim)
        self.hidden = h
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, n_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GeneratorWithSum(nn.Module):
    """Wraps a Generator with a fixed, non-trainable all-ones Linear(N, 1) head.

    This is the standard auto_LiRPA/CROWN trick for expressing a linear
    combination (here, the total sum) of output neurons as a single scalar
    output, so its bounds can be computed directly by bound propagation.
    """

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator
        n_out = generator.grid_size * generator.grid_size
        self.sum_head = nn.Linear(n_out, 1, bias=False)
        with torch.no_grad():
            self.sum_head.weight.fill_(1.0)
        for p in self.sum_head.parameters():
            p.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.sum_head(self.generator(z))
