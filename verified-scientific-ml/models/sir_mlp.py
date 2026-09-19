"""Small fully-connected MLP surrogate for the SIR epidemiological model's one-step map."""

from __future__ import annotations

import torch
from torch import nn

# State layout: [s, i, r]
S, I, R = 0, 1, 2


class SirMLP(nn.Module):
    """
    Predicts the next 3-dim state [s, i, r] from the current one.

    A small 3 -> hidden -> hidden -> 3 fully-connected network with ReLU
    activations. Standard (non-geometric) baseline: nothing in the
    architecture encodes population conservation (s+i+r=1) or any other
    physical symmetry -- that is deliberate. This is Model A for the SIR
    domain, analogous to ParticleMLP (models/particle_mlp.py) for the
    two-particle system.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def _get_conservation_error(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return delta_population = (s_next+i_next+r_next) - (s+i+r) for each
        state in the batch. Shape: (N,).

        For the plain MLP this is generally non-zero (no conservation
        structure is built in). Compare with SirGeometricMLP, where this
        is identically zero by construction.
        """
        with torch.no_grad():
            next_state = self.forward(state)
        pop_before = state[:, S] + state[:, I] + state[:, R]
        pop_after = next_state[:, S] + next_state[:, I] + next_state[:, R]
        return pop_after - pop_before
