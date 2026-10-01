"""
Conditional feedforward generator for calorimeter shower simulation.

Architecture
------------
Input  : [z_1, …, z_{z_dim}, log_energy]
           z    — latent noise vector, z_i ∈ [-1, 1]
           log_energy — log(E / E_min) where E is the incident particle
                        energy in MeV and E_min = 256 MeV
                        → log_energy ∈ [0, log(E_max / E_min)]
Output : voxel energy depositions (non-negative floats)

Design constraints (all chosen for auto_LiRPA compatibility)
-----------------------------------------------------------
* Only Linear and ReLU layers — the bound-propagation graph is simple.
* Final ReLU enforces non-negativity architecturally, making the
  positivity property PROVEN by IBP regardless of weights.
* No batch-norm, residual connections, or non-standard activations.
* Output dimension matches the CaloChallenge particle geometry
  (368 voxels for photons, 533 for pions).

Input region for verification
------------------------------
z   ∈ [-1, 1]^{z_dim}
log_energy ∈ [0, LOG_E_MAX]  where LOG_E_MAX = log(4096/256) ≈ 2.77

These are the values used in ModelSpec.input_lb / input_ub when this
model is passed to the verifier.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

# ── constants ────────────────────────────────────────────────────────────────

E_MIN_MEV: float = 256.0           # minimum incident energy in the dataset
E_MAX_MEV: float = 4_194_304.0    # maximum: 2^14 × E_MIN_MEV = 4 TeV (real data range)
LOG_E_MAX: float = math.log(E_MAX_MEV / E_MIN_MEV)   # ≈ 9.704

Z_DIM: int = 8                     # latent dimension
INPUT_DIM: int = Z_DIM + 1        # z_dim + log_energy scalar


# ── model ────────────────────────────────────────────────────────────────────

class ConditionalGenerator(nn.Module):
    """
    Linear+ReLU feedforward generator conditioned on incident energy.

    Parameters
    ----------
    n_voxels : int
        Output dimension (368 for photons, 533 for pions).
    z_dim : int
        Latent dimension.  Default 8.
    hidden_dim : int
        Width of each hidden layer.
    n_hidden : int
        Number of hidden Linear+ReLU blocks.
    """

    def __init__(
        self,
        n_voxels: int = 368,
        z_dim: int = Z_DIM,
        hidden_dim: int = 64,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.n_voxels = n_voxels
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        in_dim = z_dim + 1  # z + log_energy
        layers: list[nn.Module] = []
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, n_voxels), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, z_dim + 1)
            Columns 0:z_dim are the latent vector z.
            Column z_dim is log(E / E_min).

        Returns
        -------
        Tensor, shape (batch, n_voxels)
            Predicted voxel energy depositions (>= 0 by final ReLU).
        """
        return self.net(x)

    def input_bounds(self) -> Tuple[list, list]:
        """
        Return (lower_bounds, upper_bounds) for the input region.

        z ∈ [-1, 1]^z_dim,  log_energy ∈ [0, LOG_E_MAX].
        """
        lb = [-1.0] * self.z_dim + [0.0]
        ub = [1.0] * self.z_dim + [LOG_E_MAX]
        return lb, ub


# ── input encoding helpers ────────────────────────────────────────────────────

def encode_input(
    z: torch.Tensor,
    incident_energy_mev: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate z and log-normalised energy into a single input tensor.

    Parameters
    ----------
    z : Tensor, shape (batch, z_dim)
    incident_energy_mev : Tensor, shape (batch,)

    Returns
    -------
    Tensor, shape (batch, z_dim + 1)
    """
    log_e = torch.log(incident_energy_mev / E_MIN_MEV).unsqueeze(1)
    return torch.cat([z, log_e], dim=1)


def decode_energy(log_e_normalised: float) -> float:
    """Convert log_energy (as used in input) back to MeV."""
    return E_MIN_MEV * math.exp(log_e_normalised)
