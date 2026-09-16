"""
Soft-geometric surrogate for the two-particle spring system's one-step
map (Model C). This is the intermediate design between the plain MLP
baseline (models/particle_mlp.py, Model A) and the hard momentum-
conserving geometric model (models/particle_geometric.py, Model B).

Two structural choices are encoded directly in the forward pass, not
learned:

1. Translation invariance of the interaction: the only learned part of
   the model, `delta_net`, sees just the RELATIVE position
   (rx, ry) = (x2-x1, y2-y1) and RELATIVE velocity
   (dvx, dvy) = (vx2-vx1, vy2-vy1) -- never absolute coordinates. So
   shifting both particles by the same amount cannot change the
   predicted velocity updates (exactly mirrors the true physics: the
   spring force in simulator/two_particle_system.py depends only on
   r = x2-x1). The extract_rel layer is a fixed copy of the one in
   models/particle_geometric.py.

2. Exact kinematic position update: positions are advanced using the
   exact relation x_next = x + dt*v (a forward-Euler/symplectic step
   over the fixed, known timestep DT), rather than learned. There is
   nothing to learn there since that relation is already exact.

What Model C does NOT enforce:

   Newton's third law / momentum conservation: unlike Model B, which
   forces vx1 += +Fx, vx2 += -Fx (so the velocity changes cancel),
   here `delta_net` predicts FOUR independent velocity updates
   [dvx1, dvy1, dvx2, dvy2]. These are free parameters -- the model
   CAN learn to approximately conserve momentum (if the physics signal
   is strong enough), but there is no algebraic guarantee. Formal
   verification therefore finds SAT (counterexample) at tight epsilon,
   unlike Model B which is UNSAT at any epsilon above float32 noise.

Architecture:
    extract_rel   fixed Linear(8->4, no bias)   extracts relative state
    delta_net     learned Linear(4->16)->ReLU->Linear(16->16)->ReLU->
                  Linear(16->4)                 predicts [dvx1,dvy1,dvx2,dvy2]
    combine       fixed Linear(12->8, no bias)  applies kinematics + updates

The combine layer encodes:
    x1_next  = x1 + dt*vx1            (exact kinematics)
    y1_next  = y1 + dt*vy1
    vx1_next = vx1 + dvx1             (learned update, unconstrained)
    vy1_next = vy1 + dvy1
    x2_next  = x2 + dt*vx2
    y2_next  = y2 + dt*vy2
    vx2_next = vx2 + dvx2
    vy2_next = vy2 + dvy2

Implementation note: all steps are implemented as fixed or learned
nn.Linear layers (no tensor indexing, no Gather ops), so the full
forward pass traces to only Gemm/MatMul/Add/Relu in ONNX -- the only
operations Marabou's ONNX parser supports. See the analogous note in
models/particle_geometric.py for details.
"""

from __future__ import annotations

import torch
from torch import nn

from data.generate_particle_data import DT

# State layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
X1, Y1, VX1, VY1, X2, Y2, VX2, VY2 = range(8)


class ParticleSoftGeometricMLP(nn.Module):
    """
    Soft-geometric surrogate (Model C). Translation-invariant by
    construction; momentum-conserving only approximately (if learned).
    Only `delta_net` (a 4 -> hidden -> hidden -> 4 MLP) is learned;
    the `extract_rel` and `combine` layers are fixed linear maps
    encoding the structural physics described in the module docstring.
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

        # Fixed linear map: state (8) -> [rx, ry, dvx, dvy] (4).
        # Identical to the extract_rel layer in ParticleGeometricMLP.
        extract = torch.zeros(4, 8)
        extract[0, X2], extract[0, X1] = 1.0, -1.0   # rx  = x2 - x1
        extract[1, Y2], extract[1, Y1] = 1.0, -1.0   # ry  = y2 - y1
        extract[2, VX2], extract[2, VX1] = 1.0, -1.0  # dvx = vx2 - vx1
        extract[3, VY2], extract[3, VY1] = 1.0, -1.0  # dvy = vy2 - vy1
        self.extract_rel = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            self.extract_rel.weight.copy_(extract)
        self.extract_rel.weight.requires_grad_(False)

        # Fixed linear map: [state (8), delta (4)] (12) -> next_state (8).
        # Input layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        #                dvx1, dvy1, dvx2, dvy2]
        # Indices for the delta components in the 12-dim concatenated input:
        DVX1, DVY1, DVX2, DVY2 = 8, 9, 10, 11

        combine = torch.zeros(8, 12)
        for i in range(8):
            combine[i, i] = 1.0   # pass-through: output starts from input
        # Exact kinematic position update: x_next = x + dt*v
        combine[X1, VX1] += dt
        combine[Y1, VY1] += dt
        combine[X2, VX2] += dt
        combine[Y2, VY2] += dt
        # Learned (unconstrained) velocity updates: v_next = v + dv
        combine[VX1, DVX1] += 1.0
        combine[VY1, DVY1] += 1.0
        combine[VX2, DVX2] += 1.0
        combine[VY2, DVY2] += 1.0
        self.combine = nn.Linear(12, 8, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        rel = self.extract_rel(state)
        delta = self.delta_net(rel)
        combined = torch.cat([state, delta], dim=-1)
        return self.combine(combined)

    def _get_momentum_conservation_error(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (delta_px, delta_py) for each state in the batch.

        delta_px = (vx1_next + vx2_next) - (vx1 + vx2)
        delta_py = (vy1_next + vy2_next) - (vy1 + vy2)

        For Model B (ParticleGeometricMLP) these are always 0 by
        construction. For Model C they are generally non-zero.
        """
        with torch.no_grad():
            next_state = self.forward(state)
        px_before = state[:, VX1] + state[:, VX2]
        px_after = next_state[:, VX1] + next_state[:, VX2]
        py_before = state[:, VY1] + state[:, VY2]
        py_after = next_state[:, VY1] + next_state[:, VY2]
        return px_after - px_before, py_after - py_before

    def _get_translation_invariance_error(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Verify that shifting both particles by the same vector does not
        change the predicted velocity updates.

        Returns (max_vx_err, max_vy_err): maximum absolute difference
        in vx and vy outputs between the original and shifted states.

        For Model C this should be zero (up to float32 rounding) because
        the only input to delta_net is the relative state, which is
        unchanged by a common translation. Positions in the output shift
        by the same amount as the input, which is also correct.
        """
        shift = torch.zeros(state.shape[-1], dtype=state.dtype, device=state.device)
        shift[X1] = shift[X2] = 1.0
        shift[Y1] = shift[Y2] = 1.0

        with torch.no_grad():
            out = self.forward(state)
            out_shifted = self.forward(state + shift)

        # Velocity outputs must be identical
        vx_err = torch.max(torch.abs(out[:, VX1] - out_shifted[:, VX1]))
        vy_err = torch.max(torch.abs(out[:, VY1] - out_shifted[:, VY1]))
        return vx_err, vy_err
