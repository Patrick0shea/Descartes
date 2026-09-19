"""
Geometric/structurally-conserving surrogate for the SIR epidemiological
model's one-step map (Model B). This is the counterpart to the plain
baseline in models/sir_mlp.py: same problem, same training data, but with
physical structure built into the architecture instead of left for training
to approximate.

The SIR conservation law s+i+r = 1 is encoded as an EXACT ALGEBRAIC
IDENTITY of the architecture:

1. Redundant state elimination: r = 1-s-i is fully determined by s and i,
   so only [s, i] need to be passed to the learned part. The `extract`
   layer is a fixed Linear(3->2, no bias) that picks out [s, i].

2. Exact conservation-preserving combination: `delta_net` predicts [Δs, Δi]
   (changes to the two free variables). The `combine` layer then maps
   [s, i, r, Δs, Δi] to the next state as:

       s_next = s + Δs
       i_next = i + Δi
       r_next = r - Δs - Δi

   which gives (s_next+i_next+r_next) = (s+Δs)+(i+Δi)+(r-Δs-Δi) = s+i+r.
   This is an algebraic identity: true for ANY weight setting of delta_net,
   not just a trained one. Formal verification at any epsilon must find UNSAT.

Implementation note: all steps are implemented as fixed or learned nn.Linear
layers (no tensor indexing, no Gather ops), so the full forward pass traces
to only Gemm/MatMul/Add/Relu in ONNX -- the only operations Marabou's ONNX
parser supports. See models/particle_geometric.py for the analogous pattern.
"""

from __future__ import annotations

import torch
from torch import nn

from data.generate_sir_data import DT

# State layout: [s, i, r]
S, I, R = 0, 1, 2


class SirGeometricMLP(nn.Module):
    """
    Population-conserving-by-construction surrogate for the SIR one-step
    map (Model B). Only `delta_net` (a 2->hidden->hidden->2 MLP) is
    learned; the `extract` and `combine` layers are fixed linear maps
    encoding the structural physics described in the module docstring.
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

        # Fixed linear map: state (3) -> [s, i] (2).
        # r is redundant (r = 1-s-i) so we only pass [s, i] to delta_net.
        #   row 0: s_out = 1*s + 0*i + 0*r
        #   row 1: i_out = 0*s + 1*i + 0*r
        extract = torch.zeros(2, 3)
        extract[0, S] = 1.0   # extract s
        extract[1, I] = 1.0   # extract i
        self.extract = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.extract.weight.copy_(extract)
        self.extract.weight.requires_grad_(False)

        # Fixed linear map: [s, i, r, Δs, Δi] (5) -> next_state (3).
        # Input layout: [s(0), i(1), r(2), Δs(3), Δi(4)]
        # Encoding:
        #   s_next = s + Δs:         [1, 0, 0,  1,  0]
        #   i_next = i + Δi:         [0, 1, 0,  0,  1]
        #   r_next = r - Δs - Δi:   [0, 0, 1, -1, -1]
        # Sum: (s+Δs)+(i+Δi)+(r-Δs-Δi) = s+i+r  (exact algebraic identity)
        combine = torch.zeros(3, 5)
        combine[S, 0] = 1.0    # s pass-through
        combine[S, 3] = 1.0    # + Δs
        combine[I, 1] = 1.0    # i pass-through
        combine[I, 4] = 1.0    # + Δi
        combine[R, 2] = 1.0    # r pass-through
        combine[R, 3] = -1.0   # - Δs
        combine[R, 4] = -1.0   # - Δi
        self.combine = nn.Linear(5, 3, bias=False)
        with torch.no_grad():
            self.combine.weight.copy_(combine)
        self.combine.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        si = self.extract(state)
        delta = self.delta_net(si)
        combined = torch.cat([state, delta], dim=-1)
        return self.combine(combined)

    def _get_conservation_error(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return delta_population = (s_next+i_next+r_next) - (s+i+r) for each
        state in the batch. Shape: (N,).

        For SirGeometricMLP this should be identically zero (up to float32
        rounding of an algebraic identity) for any input and any weight
        setting -- the conservation guarantee does not depend on training.
        """
        with torch.no_grad():
            next_state = self.forward(state)
        pop_before = state[:, S] + state[:, I] + state[:, R]
        pop_after = next_state[:, S] + next_state[:, I] + next_state[:, R]
        return pop_after - pop_before

    def _get_subspace_invariance_error(self, state: torch.Tensor) -> torch.Tensor:
        """
        Verify that the extract layer correctly implements the r=1-s-i
        redundancy: the extracted [s, i] must match the first two components
        of the input state exactly.

        Returns max absolute difference between extract(state) and state[:,:2].
        Should be zero (up to float32 rounding).
        """
        with torch.no_grad():
            si_extracted = self.extract(state)
        si_direct = state[:, :2]
        return torch.max(torch.abs(si_extracted - si_direct))
