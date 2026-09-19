"""
Soft-geometric surrogate for the SIR epidemiological model's one-step map
(Model C). This is the intermediate design between the plain MLP baseline
(models/sir_mlp.py, Model A) and the hard conservation-preserving geometric
model (models/sir_geometric.py, Model B).

One structural choice is encoded directly in the forward pass, not learned:

1. Redundant state elimination: r = 1-s-i is fully determined by s and i,
   so only [s, i] are passed to the learned part. The `extract` layer is a
   fixed Linear(3->2, no bias), identical to the one in SirGeometricMLP.
   This prevents the network from ignoring the conservation manifold
   structure entirely.

What Model C does NOT enforce:

   Population conservation: unlike Model B, which forces r_next = r-Δs-Δi
   (so the three changes sum to zero), here `delta_net` predicts THREE
   independent updates [Δs, Δi, Δr]. These are free parameters -- the model
   CAN learn to approximately conserve population (if the physics signal is
   strong enough), but there is no algebraic guarantee. Formal verification
   therefore finds SAT (counterexample) at tight epsilon, unlike Model B
   which is UNSAT at any epsilon above float32 noise.

Architecture:
    extract    fixed Linear(3->2, no bias)        extracts [s, i]
    delta_net  learned Linear(2->16)->ReLU->
               Linear(16->16)->ReLU->Linear(16->3) predicts [Δs, Δi, Δr]
    combine    fixed Linear(6->3, no bias)         applies updates

The combine layer encodes:
    s_next = s + Δs    (learned update, unconstrained)
    i_next = i + Δi
    r_next = r + Δr

Implementation note: all steps are implemented as fixed or learned nn.Linear
layers (no tensor indexing, no Gather ops), so the full forward pass traces
to only Gemm/MatMul/Add/Relu in ONNX -- the only operations Marabou's ONNX
parser supports. See models/particle_soft_geometric.py for the analogous
pattern.
"""

from __future__ import annotations

import torch
from torch import nn

from data.generate_sir_data import DT

# State layout: [s, i, r]
S, I, R = 0, 1, 2


class SirSoftGeometricMLP(nn.Module):
    """
    Soft-geometric surrogate (Model C). Uses only [s, i] as input to the
    learned part (r=1-s-i redundancy encoded in the fixed extract layer),
    but population conservation is only approximately enforced (if learned),
    not guaranteed by construction. Only `delta_net` (a 2->hidden->hidden->3
    MLP) is learned; `extract` and `combine` are fixed linear maps.
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

        # Fixed linear map: state (3) -> [s, i] (2).
        # Identical to the extract layer in SirGeometricMLP.
        extract = torch.zeros(2, 3)
        extract[0, S] = 1.0   # extract s
        extract[1, I] = 1.0   # extract i
        self.extract = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.extract.weight.copy_(extract)
        self.extract.weight.requires_grad_(False)

        # Fixed linear map: [s, i, r, Δs, Δi, Δr] (6) -> next_state (3).
        # Input layout: [s(0), i(1), r(2), Δs(3), Δi(4), Δr(5)]
        # Learned (unconstrained) updates: state_next = state + delta
        #   s_next = s + Δs:  [1, 0, 0, 1, 0, 0]
        #   i_next = i + Δi:  [0, 1, 0, 0, 1, 0]
        #   r_next = r + Δr:  [0, 0, 1, 0, 0, 1]
        DS, DI, DR = 3, 4, 5  # indices of Δs, Δi, Δr in the 6-dim input
        combine = torch.zeros(3, 6)
        for k in range(3):
            combine[k, k] = 1.0   # pass-through: output starts from input
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

    def _get_conservation_error(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return delta_population = (s_next+i_next+r_next) - (s+i+r) for each
        state in the batch. Shape: (N,).

        For SirSoftGeometricMLP this is generally non-zero (no algebraic
        conservation identity is built in). Compare with SirGeometricMLP,
        where this is identically zero by construction.
        """
        with torch.no_grad():
            next_state = self.forward(state)
        pop_before = state[:, S] + state[:, I] + state[:, R]
        pop_after = next_state[:, S] + next_state[:, I] + next_state[:, R]
        return pop_after - pop_before
