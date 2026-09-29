"""
DeltaModel: wraps any nn.Module to output f(x) - x (the state change).

This lets conservation properties be expressed as:
    |coeff @ delta(x)| <= epsilon
rather than requiring access to both input and output variables simultaneously
(which Marabou supports natively but auto_LiRPA's framework interface does not).

The subtraction is implemented via a fixed negative-identity linear layer (not
a raw Sub op) so the result is exportable to ONNX as plain Gemm/Add nodes.

Example — SIR population conservation:
    delta_model = DeltaModel(sir_model, n_dims=3)
    prop = Conservation([1.0, 1.0, 1.0], target=0.0, epsilon=1e-2)
    # checks |Δs + Δi + Δr| <= 1e-2 for all states in the input box

Example — x-momentum conservation:
    delta_model = DeltaModel(particle_model, n_dims=8)
    prop = Conservation([0,0,1,0,0,0,1,0], target=0.0, epsilon=1e-2)
    # checks |Δvx1 + Δvx2| <= 1e-2 for all states in the input box
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeltaModel(nn.Module):
    """
    Wraps *base* to compute f(x) - x via a fixed negative-identity layer.

    Parameters
    ----------
    base : nn.Module
        The original surrogate model.  Must map R^n_dims → R^n_dims.
    n_dims : int
        Input/output dimensionality (3 for SIR, 8 for particle).
    """

    def __init__(self, base: nn.Module, n_dims: int) -> None:
        super().__init__()
        self.base = base
        # Fixed Linear implementing -I: output = -1 * x
        neg_id = nn.Linear(n_dims, n_dims, bias=False)
        with torch.no_grad():
            neg_id.weight.copy_(-torch.eye(n_dims))
        for p in neg_id.parameters():
            p.requires_grad_(False)
        self.neg_id = neg_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.neg_id(x)   # f(x) + (-I)(x) = f(x) - x
