"""
Conservation properties for SIR and two-particle spring domains.

All properties are expressed as Conservation(coefficients, target=0, epsilon)
applied to a DeltaModel wrapper (see delta_model.py).

SIR domain
----------
    Population conservation: |Δs + Δi + Δr| <= epsilon
    Coefficients = [1, 1, 1], target = 0

Particle domain
---------------
    x-momentum conservation: |Δvx1 + Δvx2| <= epsilon
    Coefficients = [0, 0, 1, 0, 0, 0, 1, 0], target = 0

    y-momentum conservation: |Δvy1 + Δvy2| <= epsilon
    Coefficients = [0, 0, 0, 1, 0, 0, 0, 1], target = 0
"""

from __future__ import annotations

import numpy as np

from verifier.property import Conservation

# ── SIR ───────────────────────────────────────────────────────────────────────

def sir_population_conservation(epsilon: float) -> Conservation:
    """
    |Δs + Δi + Δr| <= epsilon over the SIR input box.

    Verified on DeltaModel(sir_model), so this checks the CHANGE in
    population sum, not the absolute sum.
    """
    return Conservation(
        coefficients=np.ones(3, dtype=np.float32),
        target=0.0,
        epsilon=float(epsilon),
        label="sir_population_conservation",
    )


# ── Particle ──────────────────────────────────────────────────────────────────

def particle_x_momentum_conservation(epsilon: float) -> Conservation:
    """
    |Δvx1 + Δvx2| <= epsilon over the particle input box.

    State layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2] → indices 2 and 6.
    """
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[2] = 1.0   # vx1
    coeffs[6] = 1.0   # vx2
    return Conservation(
        coefficients=coeffs,
        target=0.0,
        epsilon=float(epsilon),
        label="particle_px_conservation",
    )


def particle_y_momentum_conservation(epsilon: float) -> Conservation:
    """
    |Δvy1 + Δvy2| <= epsilon over the particle input box.

    State layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2] → indices 3 and 7.
    """
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[3] = 1.0   # vy1
    coeffs[7] = 1.0   # vy2
    return Conservation(
        coefficients=coeffs,
        target=0.0,
        epsilon=float(epsilon),
        label="particle_py_conservation",
    )
