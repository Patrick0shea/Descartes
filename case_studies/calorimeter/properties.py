"""
Calorimeter-specific property instances.

Three properties are defined for the ConditionalGenerator:

1. voxel_non_negativity()
   Every output voxel energy >= 0.  Architecturally guaranteed by the
   final ReLU; IBP proves it formally over the full input box.

2. total_energy_upper_bound(n_voxels, upper_mev)
   The sum of all voxel energies <= upper_mev.  A physical sanity check:
   the deposited energy cannot exceed some finite bound.

3. energy_conservation(n_voxels, target_mev, epsilon_mev)
   |sum(voxels) - target_mev| <= epsilon_mev.  The tightest provable
   epsilon quantifies how well the verifier can certify that the generator
   stays close to a nominal total energy over the whole input box.
"""

from __future__ import annotations

import numpy as np

from verifier.property import Conservation, NonNegativity, RangeBound


def voxel_non_negativity() -> NonNegativity:
    """
    Every output voxel energy is >= 0 over the entire input region.

    This is architecturally guaranteed by the final ReLU in
    ConditionalGenerator; IBP propagates the guarantee formally.
    """
    return NonNegativity(label="voxel_non_negativity")


def total_energy_upper_bound(n_voxels: int, upper_mev: float) -> RangeBound:
    """
    The total deposited energy (sum of all voxels) is bounded above.

    RangeBound: sum(voxels) <= upper_mev.
    """
    return RangeBound(
        coefficients=np.ones(n_voxels, dtype=np.float32),
        lower=None,
        upper=float(upper_mev),
        label="total_energy_upper_bound",
    )


def energy_conservation(
    n_voxels: int,
    target_mev: float,
    epsilon_mev: float,
) -> Conservation:
    """
    Energy conservation property: |sum(voxels) - target_mev| <= epsilon_mev.

    Parameters
    ----------
    n_voxels : int
        Number of output voxels (368 for photon, 533 for pion).
    target_mev : float
        Nominal total deposited energy in MeV.
    epsilon_mev : float
        Maximum allowed deviation from target over the full input box.
    """
    return Conservation(
        coefficients=np.ones(n_voxels, dtype=np.float32),
        target=float(target_mev),
        epsilon=float(epsilon_mev),
        label="energy_conservation",
    )
