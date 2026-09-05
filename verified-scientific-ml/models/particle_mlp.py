"""Small fully-connected MLP surrogate for the two-particle spring system's one-step map."""

from __future__ import annotations

import torch
from torch import nn


class ParticleMLP(nn.Module):
    """
    Predicts the next 8-dim state [x1,y1,vx1,vy1,x2,y2,vx2,vy2] from the
    current one.

    A small 8 -> hidden -> hidden -> 8 fully-connected network with ReLU
    activations. Standard (non-geometric) baseline: nothing in the
    architecture encodes momentum conservation or any other physical
    symmetry -- that is deliberate, this is Step 4's baseline to compare
    a later geometric/equivariant model against.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
