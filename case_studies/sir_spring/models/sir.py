"""
SIR epidemiological model surrogates: Models A, B, C.

Adapted from verified-scientific-ml/models/sir_{mlp,geometric,soft_geometric}.py.
The data-module imports have been replaced with a local constant so these
classes can be used standalone (state_dict loading overwrites the fixed-layer
weights anyway, so the DT value only matters when constructing fresh models
for testing without a checkpoint).

Models
------
SirMLP             : Model A — plain 3→16→16→3 MLP, no physical structure.
SirGeometricMLP    : Model B — population conservation by construction.
SirSoftGeometricMLP: Model C — redundancy-aware input, no conservation guarantee.
"""

from __future__ import annotations

import torch
from torch import nn

# One-day time step used during simulation and training
DT: float = 1.0

# State indices
S, I, R = 0, 1, 2


# ── Model A ───────────────────────────────────────────────────────────────────

class SirMLP(nn.Module):
    """Plain MLP baseline (Model A).  No physical structure encoded."""

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


# ── Model B ───────────────────────────────────────────────────────────────────

class SirGeometricMLP(nn.Module):
    """
    Population-conserving-by-construction surrogate (Model B).

    Conservation identity: s_next + i_next + r_next = s + i + r
    holds for ANY weight setting of delta_net (algebraic guarantee).
    """

    def __init__(self, hidden_dim: int = 16, dt: float = DT) -> None:
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.dt = dt

        # Fixed extract: state (3) → [s, i] (2)
        extract = torch.zeros(2, 3)
        extract[0, S] = 1.0
        extract[1, I] = 1.0
        self.extract = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.extract.weight.copy_(extract)
        self.extract.weight.requires_grad_(False)

        # Fixed combine: [s, i, r, Δs, Δi] (5) → next_state (3)
        #   s_next = s + Δs,  i_next = i + Δi,  r_next = r - Δs - Δi
        combine = torch.zeros(3, 5)
        combine[S, 0] = 1.0;  combine[S, 3] = 1.0
        combine[I, 1] = 1.0;  combine[I, 4] = 1.0
        combine[R, 2] = 1.0;  combine[R, 3] = -1.0;  combine[R, 4] = -1.0
        self.combine = nn.Linear(5, 3, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        si = self.extract(state)
        delta = self.delta_net(si)
        combined = torch.cat([state, delta], dim=-1)
        return self.combine(combined)


# ── Model C ───────────────────────────────────────────────────────────────────

class SirSoftGeometricMLP(nn.Module):
    """
    Soft-geometric surrogate (Model C).

    Redundancy-aware input (only [s,i] fed to learned net), but population
    conservation is not guaranteed by construction — only approximately
    enforced through training.
    """

    def __init__(self, hidden_dim: int = 16, dt: float = DT) -> None:
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.dt = dt

        # Fixed extract: state (3) → [s, i] (2)  (identical to Model B)
        extract = torch.zeros(2, 3)
        extract[0, S] = 1.0
        extract[1, I] = 1.0
        self.extract = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.extract.weight.copy_(extract)
        self.extract.weight.requires_grad_(False)

        # Fixed combine: [s, i, r, Δs, Δi, Δr] (6) → next_state (3)
        #   state_next = state + delta  (no algebraic constraint)
        DS, DI, DR = 3, 4, 5
        combine = torch.zeros(3, 6)
        for k in range(3):
            combine[k, k] = 1.0
        combine[S, DS] = 1.0
        combine[I, DI] = 1.0
        combine[R, DR] = 1.0
        self.combine = nn.Linear(6, 3, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        si = self.extract(state)
        delta = self.delta_net(si)
        combined = torch.cat([state, delta], dim=-1)
        return self.combine(combined)
