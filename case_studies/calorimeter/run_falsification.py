"""
Step 7: PGD falsification baseline for the calorimeter generator.

Loads the trained ConditionalGenerator checkpoint from Step 5 and runs PGD
to try to violate three properties:

  1. voxel_non_negativity    → expected NO_COUNTEREXAMPLE_FOUND
     The final ReLU is an exact architectural constraint; the violation
     gradient is zero everywhere → PGD finds nothing.

  2. total_energy_upper_bound (3989 MeV, CROWN-proven in Step 5)
     → expected NO_COUNTEREXAMPLE_FOUND
     PGD cannot exceed the proven CROWN upper bound.

  3. energy_conservation at a tight target
     → expected COUNTEREXAMPLE_FOUND
     The model is not trained to conserve total energy at a fixed target;
     PGD can easily find inputs whose output sum deviates significantly.

The contrast between (1) and (3) is the key thesis result: an architectural
non-negativity guarantee (final ReLU) is formally verifiable; an unstructured
energy-conservation property is not.

Usage
-----
    python -m case_studies.calorimeter.run_falsification [--save-json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from case_studies.calorimeter.models.generator import (
    LOG_E_MAX,
    Z_DIM,
    ConditionalGenerator,
)
from case_studies.calorimeter.properties import (
    energy_conservation,
    total_energy_upper_bound,
    voxel_non_negativity,
)
from verifier import FalsificationResult, ModelSpec, falsify

# ── paths & model params ──────────────────────────────────────────────────────

CKPT_PATH = (
    Path(__file__).parent / "models" / "checkpoints" / "calo_generator_synthetic.pt"
)
N_VOXELS = 368
HIDDEN_DIM = 64
N_HIDDEN = 2

# CROWN-proven bounds from Step 5 (from step5 result files)
CROWN_UPPER_MEV = 3989.3   # total energy upper bound proven by CROWN
# Midpoint and half-width of the CROWN sum interval [393.5, 3989.3]
CROWN_MIDPOINT_MEV = (393.5 + 3989.3) / 2.0   # ≈ 2191 MeV
CONSERVATION_EPSILON_MEV = 100.0  # tight: PGD should find violations here


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_spec(n_voxels: int = N_VOXELS) -> ModelSpec:
    """Load checkpoint and build ModelSpec."""
    lb, ub = ConditionalGenerator(n_voxels=n_voxels, z_dim=Z_DIM).input_bounds()
    fresh = ConditionalGenerator(
        n_voxels=n_voxels, z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN
    )
    return ModelSpec.from_checkpoint(
        fresh, str(CKPT_PATH),
        input_lb=lb, input_ub=ub,
        name=f"calo_generator_synthetic_n{n_voxels}",
    )


def _save_falsification(result: FalsificationResult, output_dir: Path, run_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}.json"
    path.write_text(result.to_json())
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main(save_json: bool = False) -> None:
    print("=" * 60)
    print("Step 7: Calorimeter PGD falsification baseline")
    print(f"  n_voxels={N_VOXELS}, checkpoint={CKPT_PATH.name}")
    print("=" * 60)

    spec = _build_spec()
    print(f"  checkpoint hash : {spec.checkpoint_hash}")

    out_dir = Path(__file__).parent.parent.parent / "results" / "calorimeter"
    results_list = []

    # ── 1. Voxel non-negativity ────────────────────────────────────────────
    print("\n[1] voxel_non_negativity (PGD)")
    prop1 = voxel_non_negativity()
    r1 = falsify(spec, prop1, n_restarts=100, n_steps=300, step_size=0.01, seed=42)
    results_list.append(("step7_calo_nonneg", r1))
    print(f"  status          : {r1.status}")
    print(f"  violation_mag   : {r1.violation_magnitude}")
    print(f"  notes           : {r1.notes[:80]}")

    # ── 2. Total energy upper bound ────────────────────────────────────────
    print(f"\n[2] total_energy_upper_bound ({CROWN_UPPER_MEV:.1f} MeV, CROWN-proven)")
    prop2 = total_energy_upper_bound(N_VOXELS, upper_mev=CROWN_UPPER_MEV)
    r2 = falsify(spec, prop2, n_restarts=100, n_steps=300, step_size=0.01, seed=42)
    results_list.append(("step7_calo_energy_upper", r2))
    print(f"  status          : {r2.status}")
    print(f"  violation_mag   : {r2.violation_magnitude}")

    # ── 3. Energy conservation at a tight target ───────────────────────────
    print(f"\n[3] energy_conservation target={CROWN_MIDPOINT_MEV:.1f} MeV, "
          f"epsilon={CONSERVATION_EPSILON_MEV:.1f} MeV")
    prop3 = energy_conservation(
        N_VOXELS,
        target_mev=CROWN_MIDPOINT_MEV,
        epsilon_mev=CONSERVATION_EPSILON_MEV,
    )
    r3 = falsify(spec, prop3, n_restarts=100, n_steps=300, step_size=0.01, seed=42)
    results_list.append(("step7_calo_conservation", r3))
    print(f"  status          : {r3.status}")
    viol = r3.violation_magnitude
    if viol is not None:
        print(f"  violation_mag   : {viol:.2f} MeV  (property violated by this much beyond epsilon)")
    else:
        print(f"  violation_mag   : None")

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print(f"  {'Property':<40} {'Status'}")
    print("-" * 60)
    for _, r in results_list:
        print(f"  {r.property_name:<40} {r.status}")
    print("=" * 60)
    print()
    print("  Key result:")
    if r1.status == "NO_COUNTEREXAMPLE_FOUND":
        print("  - NonNegativity: PGD finds no violation  ← architecturally guaranteed (final ReLU)")
    if r3.status == "COUNTEREXAMPLE_FOUND":
        print("  - EnergyConservation: PGD finds violation ← not architecturally guaranteed")

    if save_json:
        for run_id, r in results_list:
            p = _save_falsification(r, out_dir, run_id)
            print(f"  Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    main(save_json=args.save_json)
