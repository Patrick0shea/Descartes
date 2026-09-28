"""
Shared fixtures for the verifier test suite.

All networks here have explicit, deterministic weights so the expected
verification result is obvious and tests are not sensitive to random init.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from verifier import ModelSpec


# ── input regions ─────────────────────────────────────────────────────────────

@pytest.fixture
def bounds_unit_square():
    """2D input box [-1,1]^2."""
    return np.array([-1.0, -1.0]), np.array([1.0, 1.0])


@pytest.fixture
def bounds_positive_square():
    """2D input box [0,1]^2."""
    return np.array([0.0, 0.0]), np.array([1.0, 1.0])


# ── tiny networks ─────────────────────────────────────────────────────────────

@pytest.fixture
def nonneg_model():
    """
    2 → 4 → 2 network with a final ReLU.

    The final ReLU guarantees all outputs >= 0 regardless of earlier
    weights.  CROWN should prove NonNegativity.
    """
    net = nn.Sequential(
        nn.Linear(2, 4, bias=True),
        nn.ReLU(),
        nn.Linear(4, 2, bias=True),
        nn.ReLU(),
    )
    torch.manual_seed(0)
    for p in net.parameters():
        nn.init.normal_(p, mean=0.0, std=0.5)
    net.eval()
    return net


@pytest.fixture
def negative_bias_model():
    """
    2 → 2 linear model with large negative biases (no final ReLU).

    Output is always near [-3, -3] in the centre of the input box.
    CROWN cannot prove NonNegativity; LiRPA should return INCONCLUSIVE.
    Marabou should return COUNTEREXAMPLE.
    """
    net = nn.Linear(2, 2, bias=True)
    net.weight.data.fill_(0.0)
    net.bias.data = torch.tensor([-3.0, -3.0])
    net.eval()
    return net


@pytest.fixture
def constant_sum_model():
    """
    2 → 2 linear model that always outputs [0.5, 0.5].

    Weights are zero; bias is [0.5, 0.5].  Sum is exactly 1.0 for any
    input.  Conservation(coeff=[1,1], target=1, epsilon=1e-4) must be
    PROVEN by both backends.
    """
    net = nn.Linear(2, 2, bias=True)
    net.weight.data.fill_(0.0)
    net.bias.data = torch.tensor([0.5, 0.5])
    net.eval()
    return net


@pytest.fixture
def violated_conservation_model():
    """
    2 → 2 linear model whose sum can vary widely.

    Weights are [[1,0],[0,1]], bias=[0,0].  Output = input, so sum can
    range from -2 to 2 over [-1,1]^2.  Conservation(sum=1, epsilon=0.1)
    is violated.
    """
    net = nn.Linear(2, 2, bias=True)
    net.weight.data = torch.eye(2)
    net.bias.data.fill_(0.0)
    net.eval()
    return net


# ── ModelSpec helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def spec_nonneg(nonneg_model, bounds_unit_square):
    lb, ub = bounds_unit_square
    return ModelSpec(model=nonneg_model, input_lb=lb, input_ub=ub, name="nonneg_model")


@pytest.fixture
def spec_negative(negative_bias_model, bounds_unit_square):
    lb, ub = bounds_unit_square
    return ModelSpec(model=negative_bias_model, input_lb=lb, input_ub=ub, name="negative_model")


@pytest.fixture
def spec_constant_sum(constant_sum_model, bounds_unit_square):
    lb, ub = bounds_unit_square
    return ModelSpec(model=constant_sum_model, input_lb=lb, input_ub=ub, name="constant_sum")


@pytest.fixture
def spec_violated(violated_conservation_model, bounds_unit_square):
    lb, ub = bounds_unit_square
    return ModelSpec(
        model=violated_conservation_model, input_lb=lb, input_ub=ub,
        name="violated_conservation"
    )
