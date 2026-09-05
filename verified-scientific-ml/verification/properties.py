"""
Definition of the bounded input domain and the output-bound properties
verified in Step 3.

Domain
------
    x in [-2, 2]
    v in [-2, 2]

This matches the domain the Step 1 dataset was sampled from
(data/generate_data.py: X_RANGE = V_RANGE = (-2, 2)), so it is exactly the
region the surrogate was trained on -- not an arbitrary choice.

Properties
----------
Each Property below states a claim of the form:

    "ub": for all (x, v) in the domain, output <= bound_value
    "lb": for all (x, v) in the domain, output >= bound_value

for one of the two outputs (x_next, v_next). verification/verify.py asks
Marabou to find a counterexample to the NEGATION of the claim:

    ub claim  ->  search for an input with output >= bound_value
    lb claim  ->  search for an input with output <= bound_value

UNSAT on the negation means no such counterexample exists anywhere in the
domain: the property is formally verified for that exact domain and no
other. SAT means Marabou found a concrete input in the domain that
violates the claim; it is returned as a counterexample.

`expect` records what we predict the result will be (from the physics and
the empirical error analysis in Step 2's evaluation_report.txt) purely for
documentation and as a sanity check on the pipeline; it is NOT given to
Marabou and has no effect on the solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

X_MIN, X_MAX = -2.0, 2.0
V_MIN, V_MAX = -2.0, 2.0

OUTPUT_NAMES = {0: "x_next", 1: "v_next"}


@dataclass(frozen=True)
class Property:
    name: str
    description: str
    output_index: int  # 0 = x_next, 1 = v_next
    bound_type: Literal["ub", "lb"]
    bound_value: float
    expect: Literal["UNSAT", "SAT"]


PROPERTIES = [
    # --- Loose bounds that the physics + Step 2's measured max error
    # (<= ~4e-3) say should hold everywhere in the domain: energy
    # conservation bounds the true amplitude to sqrt(2*E_max) = sqrt(8)
    # ~= 2.83, so a bound of 3.0 leaves comfortable margin. Expected: UNSAT
    # (formally verified). ---
    Property(
        name="x_next_upper_loose",
        description="x_next <= 3.0 for all (x, v) in the domain",
        output_index=0, bound_type="ub", bound_value=3.0, expect="UNSAT",
    ),
    Property(
        name="x_next_lower_loose",
        description="x_next >= -3.0 for all (x, v) in the domain",
        output_index=0, bound_type="lb", bound_value=-3.0, expect="UNSAT",
    ),
    Property(
        name="v_next_upper_loose",
        description="v_next <= 3.0 for all (x, v) in the domain",
        output_index=1, bound_type="ub", bound_value=3.0, expect="UNSAT",
    ),
    Property(
        name="v_next_lower_loose",
        description="v_next >= -3.0 for all (x, v) in the domain",
        output_index=1, bound_type="lb", bound_value=-3.0, expect="UNSAT",
    ),
    # --- Deliberately too-tight bounds, included as a sanity check: these
    # are physically false (e.g. x=2, v=0 alone already pushes x_next close
    # to 2), so Marabou should find a counterexample. Expected: SAT. ---
    Property(
        name="x_next_upper_violated",
        description="x_next <= 1.0 for all (x, v) in the domain (deliberately false)",
        output_index=0, bound_type="ub", bound_value=1.0, expect="SAT",
    ),
    Property(
        name="v_next_lower_violated",
        description="v_next >= -1.0 for all (x, v) in the domain (deliberately false)",
        output_index=1, bound_type="lb", bound_value=-1.0, expect="SAT",
    ),
]
