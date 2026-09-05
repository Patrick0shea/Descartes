"""
Definition of the bounded input domain and the momentum-conservation
properties verified for the two-particle spring system surrogate.

Domain
------
D = box ∩ separation constraints, where:

    box = [x1_lo,x1_hi] x [y1_lo,y1_hi] x [vx1_lo,vx1_hi] x [vy1_lo,vy1_hi]
          x [x2_lo,x2_hi] x [y2_lo,y2_hi] x [vx2_lo,vx2_hi] x [vy2_lo,vy2_hi]

    separation constraints:  |x2 - x1| <= SEPARATION_BOUND
                              |y2 - y1| <= SEPARATION_BOUND

`box` is imported directly from data/generate_particle_data.py's
STATE_BOUNDS -- the exact per-dimension box that script samples training
data from. But data/generate_particle_data.py does NOT sample x2, y2
independently of x1, y1: it places particle 2 at a random separation
d0 in SEPARATION_RANGE (<= 1.4) and angle around particle 1 (see that
module's docstring). An axis-aligned box around x1,y1,x2,y2 independently
is therefore a strict OVERAPPROXIMATION of the actual sampled region: box
corners such as x1=-1,y1=-1,x2=2.4,y2=2.4 (separation ~4.8) are inside
the box but were never within ~3x the training data's separation range,
so the network is extrapolating wildly there and its behavior (including
gross momentum violations) says nothing about the trained regime.

The two extra linear separation constraints above remove most of that
extrapolation region while staying entirely linear (still directly
encodable as Marabou inequalities, no approximation of a nonlinear
distance needed) and remain a (tighter) overapproximation of the true
disk-shaped sampling region -- e.g. the box+diamond corner where both
|x2-x1| and |y2-y1| equal SEPARATION_BOUND has true separation
SEPARATION_BOUND*sqrt(2), still somewhat beyond the training disk's max
radius. This residual gap is a known limitation (see the README): D is
still a superset of the training distribution, just a much tighter one
than the plain box.

Property
--------
State layout (both input and output): [x1,y1,vx1,vy1,x2,y2,vx2,vy2].
Total momentum: Px = vx1 + vx2, Py = vy1 + vy2 (m1 = m2 = 1).

For a given tolerance epsilon, the CLAIM is:

    for all state in D:  |Px_next - Px| <= epsilon
                     and  |Py_next - Py| <= epsilon

where Px, Py are computed from the input state and Px_next, Py_next from
the network's output state for that same input. Because this claim is a
linear function of network input/output variables (a sum of velocity
components), it is checked with exactly the same negate-and-search
approach as verification/properties.py, split into two one-sided linear
conditions per axis:

    upper violation:  Px_next - Px >= epsilon
    lower violation:  Px - Px_next >= epsilon

(and symmetrically for Py). UNSAT on both means no state in D violates
the tolerance in either direction: the property is formally verified for
D. SAT on either returns a counterexample state.

`expect` records what we predict the result will be, purely for
documentation; it has no effect on the solve.

Choosing epsilon: an exploratory sweep (not part of the committed
pipeline) found that most of D produces momentum error consistent with
Step 4's test-set analysis (max ~5.7e-3 over 15,000 test points), but the
residual extrapolation region described above (the box+diamond corners
where true separation still exceeds the training disk's radius) produces
much larger errors -- up to ~0.5 in magnitude for this trained checkpoint.
epsilon=1.0 was the smallest round number found to be safely UNSAT (in
seconds) on both one-sided queries for both axes; epsilon=1e-4 is
deliberately far below the model's actual accuracy specifically to
demonstrate Marabou's SAT/counterexample path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from data.generate_particle_data import SEPARATION_RANGE, STATE_BOUNDS, STATE_NAMES

DOMAIN_LOWER = [b[0] for b in STATE_BOUNDS]
DOMAIN_UPPER = [b[1] for b in STATE_BOUNDS]

# Extra linear constraints tightening the box to the actual sampled
# (disk-shaped) region -- see module docstring.
SEPARATION_BOUND = SEPARATION_RANGE[1]

# Indices into the 8-dim state (input and output share this layout).
X1, Y1, VX1, VY1, X2, Y2, VX2, VY2 = range(8)


@dataclass(frozen=True)
class MomentumProperty:
    name: str
    description: str
    axis: Literal["x", "y"]  # which momentum component
    epsilon: float
    expect: Literal["UNSAT", "SAT"]


MOMENTUM_PROPERTIES = [
    # Loose tolerance: safely UNSAT (see "Choosing epsilon" above) even
    # accounting for the residual extrapolation region at D's corners.
    MomentumProperty(
        name="momentum_x_conserved_loose",
        description="|Px_next - Px| <= 1.0 for all states in D",
        axis="x", epsilon=1.0, expect="UNSAT",
    ),
    MomentumProperty(
        name="momentum_y_conserved_loose",
        description="|Py_next - Py| <= 1.0 for all states in D",
        axis="y", epsilon=1.0, expect="UNSAT",
    ),
    # Tight tolerance: Step 4's analysis found only a small fraction of
    # test examples within 1e-4, so most of the domain is expected to
    # violate this tolerance. Included specifically to demonstrate
    # Marabou's SAT/counterexample path. Expected: SAT.
    MomentumProperty(
        name="momentum_x_conserved_tight",
        description="|Px_next - Px| <= 1e-4 for all states in D (deliberately too tight)",
        axis="x", epsilon=1e-4, expect="SAT",
    ),
    MomentumProperty(
        name="momentum_y_conserved_tight",
        description="|Py_next - Py| <= 1e-4 for all states in D (deliberately too tight)",
        axis="y", epsilon=1e-4, expect="SAT",
    ),
]
