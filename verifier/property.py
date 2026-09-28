"""
Property definitions for formal verification.

Scope
-----
This module covers properties expressible as bounds on linear combinations
of a network's outputs:

* NonNegativity  — every output >= 0
* RangeBound     — lower <= coeff @ output <= upper
* Conservation   — |coeff @ output - target| <= epsilon

Properties based on derivatives (PDE residuals, divergence-free fields),
diffusion/iterative samplers, or layer types not supported by auto_LiRPA
are out of scope and are not modelled here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class Property(ABC):
    """Abstract base class for all verifiable properties."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short snake_case identifier used in result JSON keys."""

    @abstractmethod
    def description(self) -> str:
        """Human-readable statement of the property."""


@dataclass
class NonNegativity(Property):
    """
    Every output of the model is >= 0 over the entire input region.

    Expressed as: for all x in [lb, ub], min_i(output_i(x)) >= 0.
    """

    label: str = "non_negativity"

    @property
    def name(self) -> str:
        return self.label

    def description(self) -> str:
        return "All output values >= 0 over the entire input region"


@dataclass
class RangeBound(Property):
    """
    A linear combination of outputs stays within [lower, upper].

    Expressed as: for all x in [lb, ub],
        lower <= sum_i(coefficients[i] * output_i(x)) <= upper.

    Either *lower* or *upper* may be None to express a one-sided bound.
    """

    coefficients: np.ndarray  # shape (n_outputs,)
    lower: Optional[float]
    upper: Optional[float]
    label: str = "range_bound"

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.float32)
        if self.lower is None and self.upper is None:
            raise ValueError("At least one of lower or upper must be set")

    @property
    def name(self) -> str:
        return self.label

    def description(self) -> str:
        parts = []
        if self.lower is not None:
            parts.append(f"{self.lower} <=")
        parts.append(f"coeff @ output")
        if self.upper is not None:
            parts.append(f"<= {self.upper}")
        return " ".join(parts)


@dataclass
class Conservation(Property):
    """
    Conservation-law property: the model approximately conserves a quantity.

    Expressed as: for all x in [lb, ub],
        |sum_i(coefficients[i] * output_i(x)) - target| <= epsilon.

    Examples
    --------
    * SIR population: coefficients=[1,1,1], target=1.0
    * Linear momentum: coefficients=[1,1,-1,-1], target=0.0
    * Energy consistency: coefficients=[1]*n_voxels, target=f(incident_energy)

    The *epsilon* is what we attempt to prove; PROVEN means the bound holds
    everywhere in the input region.  INCONCLUSIVE means the verifier could
    not establish it, not that it is false.
    """

    coefficients: np.ndarray  # shape (n_outputs,)
    target: float
    epsilon: float
    label: str = "conservation"

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.float32)
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")

    @property
    def name(self) -> str:
        return self.label

    def description(self) -> str:
        return (
            f"|coeff @ output - {self.target}| <= {self.epsilon} "
            f"over the entire input region"
        )
