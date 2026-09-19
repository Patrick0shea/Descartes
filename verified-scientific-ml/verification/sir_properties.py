"""
Definition of the bounded input domain and the population-conservation
properties verified for the SIR epidemiological model surrogate.

Domain
------
D = box ∩ population constraint, where:

    box = [s_lo, s_hi] x [i_lo, i_hi] x [r_lo, r_hi]

    population constraint: |s + i + r - 1| <= SEPARATION_BOUND

`box` is imported directly from data/generate_sir_data.py's STATE_BOUNDS --
the exact per-dimension box that script samples training data from. The
population constraint tightens the plain box to the conservation manifold
s+i+r=1, encoded as two Marabou linear inequalities:

    s + i + r <= 1 + SEPARATION_BOUND
    -(s + i + r) <= -(1 - SEPARATION_BOUND)

Both are linear in the network's input variables and are added directly to
every Marabou query (analogous to the separation constraints in
verification/particle_properties.py).

Property
--------
State layout (both input and output): [s, i, r].
Total population: P = s + i + r (should be 1.0 everywhere on the manifold).

For a given tolerance epsilon, the CLAIM is:

    for all state in D:  |P_next - P| <= epsilon

where P is computed from the input state and P_next from the network's
output state for that same input. Because this claim is a linear function
of network input/output variables (a sum of the three compartment
fractions), it is checked as TWO one-sided linear conditions:

    upper violation:  P_next - P >= epsilon
    lower violation:  P - P_next >= epsilon

UNSAT on both means no state in D violates the tolerance in either
direction: the property is formally verified for D. SAT on either returns
a counterexample state.

`expect` records what we predict the result will be, purely for
documentation; it has no effect on the solve.

Model A (SirMLP): no conservation structure, so UNSAT only at very loose
epsilon (expected to fail at tight tolerances).

Model B (SirGeometricMLP): conservation is exact by construction -- the
combine layer enforces r_next = r - Δs - Δi so that Δ(s+i+r) = 0
algebraically. Any epsilon above float32 noise (~1e-6) should be UNSAT.

Model C (SirSoftGeometricMLP): intermediate -- redundancy-aware input but
no algebraic conservation guarantee. UNSAT frontier lies between A and B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from data.generate_sir_data import SEPARATION_BOUND, STATE_BOUNDS, STATE_NAMES

DOMAIN_LOWER = [b[0] for b in STATE_BOUNDS]
DOMAIN_UPPER = [b[1] for b in STATE_BOUNDS]

# Indices into the 3-dim state (input and output share this layout).
S, I, R = 0, 1, 2


@dataclass(frozen=True)
class PopulationProperty:
    name: str
    description: str
    epsilon: float
    expect: Literal["UNSAT", "SAT"]


# Model A properties: plain MLP, no conservation structure.
# Epsilon sweep from loose to tight; Marabou reveals the true frontier.
_A_EPSILONS = [1.0, 0.1, 1e-2, 1e-3, 1e-4]

MLP_POPULATION_PROPERTIES: list[PopulationProperty] = []
for _eps in _A_EPSILONS:
    _eps_str = f"{_eps:.0e}" if _eps < 0.1 else str(_eps)
    MLP_POPULATION_PROPERTIES.append(
        PopulationProperty(
            name=f"mlp_population_eps_{_eps_str}",
            description=f"|P_next - P| <= {_eps_str} for all states in D (Model A, MLP)",
            epsilon=_eps,
            expect="UNSAT",
        )
    )

# Model B properties: geometric surrogate, exact conservation by construction.
# Conservation holds algebraically for ANY weight setting, so UNSAT is expected
# down to float32 noise. Sweep includes epsilons far tighter than Model A can
# ever achieve, to demonstrate the structural guarantee.
_B_EPSILONS = [1.0, 1e-2, 1e-4, 1e-5, 1e-6]

GEOMETRIC_POPULATION_PROPERTIES: list[PopulationProperty] = []
for _eps in _B_EPSILONS:
    _eps_str = f"{_eps:.0e}" if _eps < 0.1 else str(_eps)
    GEOMETRIC_POPULATION_PROPERTIES.append(
        PopulationProperty(
            name=f"geometric_population_eps_{_eps_str}",
            description=f"|P_next - P| <= {_eps_str} for all states in D (Model B, geometric)",
            epsilon=_eps,
            expect="UNSAT",
        )
    )

# Model C properties: soft-geometric surrogate -- redundancy-aware input but
# NOT conservation-preserving by construction. Epsilon sweep over
# [1.0, 0.1, 1e-2, 1e-3, 1e-4]; all set to expect="UNSAT" as the hypothesis
# (what we hope training has achieved); Marabou reveals the true frontier.
# Comparing this frontier to Model A's and Model B's quantifies Model C's
# intermediate position.
_C_EPSILONS = [1.0, 0.1, 1e-2, 1e-3, 1e-4]

SOFT_GEOMETRIC_POPULATION_PROPERTIES: list[PopulationProperty] = []
for _eps in _C_EPSILONS:
    _eps_str = f"{_eps:.0e}" if _eps < 0.1 else str(_eps)
    SOFT_GEOMETRIC_POPULATION_PROPERTIES.append(
        PopulationProperty(
            name=f"soft_geometric_population_eps_{_eps_str}",
            description=f"|P_next - P| <= {_eps_str} for all states in D (Model C, soft-geometric)",
            epsilon=_eps,
            expect="UNSAT",
        )
    )
