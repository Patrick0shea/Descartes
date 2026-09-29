"""
Two-particle spring system surrogates: Models A, B, C.

Adapted from verified-scientific-ml/models/particle_{mlp,geometric,soft_geometric}.py.
The data-module imports have been replaced with a local constant.

State layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]

Models
------
ParticleMLP              : Model A — plain 8→128→128→8 MLP, no physical structure.
ParticleGeometricMLP     : Model B — Newton's third law (momentum conservation) by construction.
ParticleSoftGeometricMLP : Model C — translation-invariant input, no momentum guarantee.
"""

from __future__ import annotations

import torch
from torch import nn

# Time step used during simulation and training
DT: float = 0.05

# State indices
X1, Y1, VX1, VY1, X2, Y2, VX2, VY2 = range(8)


# ── Model A ───────────────────────────────────────────────────────────────────

class ParticleMLP(nn.Module):
    """Plain MLP baseline (Model A).  No physical structure encoded."""

    def __init__(self, hidden_dim: int = 16) -> None:  # checkpoint uses 16
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


# ── Model B ───────────────────────────────────────────────────────────────────

class ParticleGeometricMLP(nn.Module):
    """
    Momentum-conserving-by-construction surrogate (Model B).

    Newton's third law identity:
      (vx1_next + vx2_next) - (vx1 + vx2) = 0 for ALL weight settings.
    """

    def __init__(self, hidden_dim: int = 16, dt: float = DT) -> None:
        super().__init__()
        self.force_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.dt = dt

        # Fixed extract: state (8) → [rx, ry, dvx, dvy] (4)
        extract = torch.zeros(4, 8)
        extract[0, X2], extract[0, X1] = 1.0, -1.0
        extract[1, Y2], extract[1, Y1] = 1.0, -1.0
        extract[2, VX2], extract[2, VX1] = 1.0, -1.0
        extract[3, VY2], extract[3, VY1] = 1.0, -1.0
        self.extract_rel = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            self.extract_rel.weight.copy_(extract)
        self.extract_rel.weight.requires_grad_(False)

        # Fixed combine: [state (8), force (2)] (10) → next_state (8)
        FX, FY = 8, 9
        combine = torch.zeros(8, 10)
        for i in range(8):
            combine[i, i] = 1.0
        combine[VX1, FX] += 1.0;   combine[VY1, FY] += 1.0
        combine[VX2, FX] += -1.0;  combine[VY2, FY] += -1.0
        combine[X1, VX1] += dt;    combine[Y1, VY1] += dt
        combine[X2, VX2] += dt;    combine[Y2, VY2] += dt
        combine[X1, FX] += 0.5 * dt;   combine[Y1, FY] += 0.5 * dt
        combine[X2, FX] += -0.5 * dt;  combine[Y2, FY] += -0.5 * dt
        self.combine = nn.Linear(10, 8, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        rel = self.extract_rel(state)
        force = self.force_net(rel)
        combined = torch.cat([state, force], dim=-1)
        return self.combine(combined)


# ── Model C ───────────────────────────────────────────────────────────────────

class ParticleSoftGeometricMLP(nn.Module):
    """
    Soft-geometric surrogate (Model C).

    Translation-invariant by construction (relative-state input); momentum
    conservation only approximately enforced through training.
    """

    def __init__(self, hidden_dim: int = 16, dt: float = DT) -> None:
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.dt = dt

        # Fixed extract: identical to Model B
        extract = torch.zeros(4, 8)
        extract[0, X2], extract[0, X1] = 1.0, -1.0
        extract[1, Y2], extract[1, Y1] = 1.0, -1.0
        extract[2, VX2], extract[2, VX1] = 1.0, -1.0
        extract[3, VY2], extract[3, VY1] = 1.0, -1.0
        self.extract_rel = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            self.extract_rel.weight.copy_(extract)
        self.extract_rel.weight.requires_grad_(False)

        # Fixed combine: [state (8), delta (4)] (12) → next_state (8)
        DVX1, DVY1, DVX2, DVY2 = 8, 9, 10, 11
        combine = torch.zeros(8, 12)
        for i in range(8):
            combine[i, i] = 1.0
        combine[X1, VX1] += dt;    combine[Y1, VY1] += dt
        combine[X2, VX2] += dt;    combine[Y2, VY2] += dt
        combine[VX1, DVX1] += 1.0; combine[VY1, DVY1] += 1.0
        combine[VX2, DVX2] += 1.0; combine[VY2, DVY2] += 1.0
        self.combine = nn.Linear(12, 8, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        rel = self.extract_rel(state)
        delta = self.delta_net(rel)
        combined = torch.cat([state, delta], dim=-1)
        return self.combine(combined)
