"""Small fully-connected MLP surrogate for the harmonic oscillator one-step map."""

from __future__ import annotations

import torch
from torch import nn


class MLPSurrogate(nn.Module):
    """
    Predicts [x_(t+1), v_(t+1)] from [x_t, v_t].

    A small 2 -> hidden -> hidden -> 2 fully-connected network with ReLU
    activations. Deliberately simple: no attention, no recurrence, no
    convolutions, no physics built in.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
