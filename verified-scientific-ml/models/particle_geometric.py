"""
Geometric/structurally-conserving surrogate for the two-particle spring
system's one-step map (Step 5). This is the counterpart to the plain
baseline in models/particle_mlp.py: same problem, same training data,
but with physical structure built into the architecture instead of left
for training to approximate.

Two inductive biases are encoded directly in the forward pass, not
learned:

1. Translation invariance of the interaction: the only learned part of
   the model, `force_net`, sees just the RELATIVE position
   (rx, ry) = (x2-x1, y2-y1) and RELATIVE velocity
   (dvx, dvy) = (vx2-vx1, vy2-vy1) -- never absolute coordinates. So
   shifting both particles by the same amount cannot change the
   predicted interaction (exactly mirrors the true physics: the spring
   force in simulator/two_particle_system.py depends only on r = x2-x1).

2. Newton's third law / momentum conservation: `force_net` outputs a
   single shared vector F = (Fx, Fy), applied as +F to particle 1's
   velocity and -F to particle 2's. For equal masses this makes

       Px_next - Px = (vx1+Fx) + (vx2-Fx) - (vx1+vx2) = 0

   an EXACT ALGEBRAIC IDENTITY of the architecture -- true for every
   possible weight setting of force_net, not something training merely
   has to approximate. This is what Step 5's verification compares
   against Step 4's plain MLP, which had no such guarantee built in.

Positions are updated using the exact kinematic relation dx/dt = v (a
symplectic-Euler-style average of the before/after velocity over the
fixed, known timestep DT used to generate the training data) rather than
learned -- there is nothing to learn there, since that part of the ODE is
already exact.

Implementation note: the "extract relative state" and "combine state +
force into next state" steps are both LINEAR functions of their inputs,
so they are implemented as fixed (non-learned, requires_grad=False)
nn.Linear layers rather than tensor indexing/slicing. This is not a
stylistic choice -- Marabou's ONNX parser has no support for the
Gather op that tensor indexing (e.g. state[..., 0]) traces to, only
Gemm/MatMul/Add/Sub/Concat/Relu, so the whole forward pass is written
using just those.
"""

from __future__ import annotations

import torch
from torch import nn

from data.generate_particle_data import DT

# State layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
X1, Y1, VX1, VY1, X2, Y2, VX2, VY2 = range(8)


class ParticleGeometricMLP(nn.Module):
    """
    Momentum-conserving-by-construction surrogate. Only `force_net` (a
    small 4 -> hidden -> hidden -> 2 MLP) is learned; the two `extract_*`
    / `combine` layers are fixed linear maps encoding the structural
    physics described in the module docstring.
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

        # Fixed linear map: state (8) -> [rx, ry, dvx, dvy] (4).
        extract = torch.zeros(4, 8)
        extract[0, X2], extract[0, X1] = 1.0, -1.0  # rx = x2 - x1
        extract[1, Y2], extract[1, Y1] = 1.0, -1.0  # ry = y2 - y1
        extract[2, VX2], extract[2, VX1] = 1.0, -1.0  # dvx = vx2 - vx1
        extract[3, VY2], extract[3, VY1] = 1.0, -1.0  # dvy = vy2 - vy1
        self.extract_rel = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            self.extract_rel.weight.copy_(extract)
        self.extract_rel.weight.requires_grad_(False)

        # Fixed linear map: [state (8), force (2)] (12) -> next_state (8).
        #   vx1_next = vx1 + Fx        vy1_next = vy1 + Fy
        #   vx2_next = vx2 - Fx        vy2_next = vy2 - Fy
        #   x1_next  = x1 + dt*vx1 + 0.5*dt*Fx   (avg-velocity kinematics)
        #   y1_next  = y1 + dt*vy1 + 0.5*dt*Fy
        #   x2_next  = x2 + dt*vx2 - 0.5*dt*Fx
        #   y2_next  = y2 + dt*vy2 - 0.5*dt*Fy
        combine = torch.zeros(8, 10)
        FX, FY = 8, 9  # indices of Fx, Fy within the 10-dim concatenated input
        for i in range(8):
            combine[i, i] = 1.0  # every output dim starts from the matching input dim
        combine[VX1, FX] += 1.0
        combine[VY1, FY] += 1.0
        combine[VX2, FX] += -1.0
        combine[VY2, FY] += -1.0
        combine[X1, VX1] += dt
        combine[Y1, VY1] += dt
        combine[X2, VX2] += dt
        combine[Y2, VY2] += dt
        combine[X1, FX] += 0.5 * dt
        combine[Y1, FY] += 0.5 * dt
        combine[X2, FX] += -0.5 * dt
        combine[Y2, FY] += -0.5 * dt
        self.combine = nn.Linear(10, 8, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        rel = self.extract_rel(state)
        force = self.force_net(rel)
        combined = torch.cat([state, force], dim=-1)
        return self.combine(combined)
