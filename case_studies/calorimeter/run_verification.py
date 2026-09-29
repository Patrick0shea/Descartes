"""
Step 5: Calorimeter verification pipeline.

Trains a ConditionalGenerator on synthetic data, saves a checkpoint,
wraps in ModelSpec, then runs three properties through the framework:

  1. Voxel non-negativity      (IBP)   → expected PROVEN
  2. Total energy upper bound  (IBP)   → expected PROVEN (tight bound reported)
  3. Energy conservation sweep (CROWN) → epsilon sweep; reports tightest proven

Results are printed and optionally saved to results/calorimeter/.

Usage
-----
    python -m case_studies.calorimeter.run_verification [--save-json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
from case_studies.calorimeter.run_spike import make_synthetic_batch, train
from verifier import (
    LiRPABackend,
    ModelSpec,
    save_result,
    verify,
)

# ── checkpoint location ───────────────────────────────────────────────────────

CKPT_DIR = Path(__file__).parent / "models" / "checkpoints"
CKPT_PATH = CKPT_DIR / "calo_generator_synthetic.pt"

# ── helpers ───────────────────────────────────────────────────────────────────


def _probe_sum_bounds(
    spec: ModelSpec,
    method: str = "IBP",
    n_voxels: int = 368,
) -> tuple[float, float]:
    """
    Directly compute IBP/CROWN bounds on sum(outputs) using BoundedModule.

    Returns (sum_lb, sum_ub).  Used to calibrate the target and epsilon for
    the Conservation epsilon sweep.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    from verifier.backend.lirpa import _LinearHead

    coeffs = np.ones(n_voxels, dtype=np.float32)
    augmented = _LinearHead(spec.model, coeffs)
    augmented.eval()

    lb_t = torch.tensor(spec.input_lb).unsqueeze(0)
    ub_t = torch.tensor(spec.input_ub).unsqueeze(0)
    dummy = lb_t.clone()

    bounded = BoundedModule(augmented, dummy)
    ptb = PerturbationLpNorm(norm=np.inf, x_L=lb_t, x_U=ub_t)
    x_bnd = BoundedTensor(dummy, ptb)
    lb_out, ub_out = bounded.compute_bounds(x=(x_bnd,), method=method)
    return float(lb_out.item()), float(ub_out.item())


# ── main pipeline ─────────────────────────────────────────────────────────────


def main(n_epochs: int = 20, save_json: bool = False, n_voxels: int = 368) -> None:
    print("=" * 60)
    print("Step 5: Calorimeter Verification Pipeline")
    print(f"  n_voxels={n_voxels}, n_epochs={n_epochs}")
    print("=" * 60)

    # ── 1. Train ──────────────────────────────────────────────────────────
    print("\n[1/5] Training ConditionalGenerator on synthetic data …")
    model = ConditionalGenerator(n_voxels=n_voxels, z_dim=Z_DIM, hidden_dim=64, n_hidden=2)
    losses = train(model, n_epochs=n_epochs)
    print(f"  final loss = {losses[-1]:.4f}")

    # ── 2. Save checkpoint ────────────────────────────────────────────────
    print("\n[2/5] Saving checkpoint …")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"  saved to {CKPT_PATH}")

    # ── 3. Build ModelSpec ────────────────────────────────────────────────
    print("\n[3/5] Building ModelSpec …")
    lb_list, ub_list = model.input_bounds()
    fresh = ConditionalGenerator(n_voxels=n_voxels, z_dim=Z_DIM, hidden_dim=64, n_hidden=2)
    spec = ModelSpec.from_checkpoint(
        fresh, str(CKPT_PATH),
        input_lb=lb_list, input_ub=ub_list,
        name=f"calo_generator_synthetic_n{n_voxels}",
    )
    print(f"  name       : {spec.name}")
    print(f"  ckpt hash  : {spec.checkpoint_hash}")
    print(f"  input dim  : {len(lb_list)}")
    print(f"  input region: z∈[-1,1]^{Z_DIM}, log_e∈[0,{LOG_E_MAX:.4f}]")

    backend = LiRPABackend()
    results = []

    # ── 4. Property 1: Voxel non-negativity (IBP) ─────────────────────────
    print("\n[4/5] Running verification …")
    print("\n  Property 1: voxel_non_negativity (IBP)")
    prop1 = voxel_non_negativity()
    r1 = verify(spec, prop1, backend, config={"method": "IBP"})
    results.append(r1)
    print(f"    status       : {r1.status}")
    print(f"    bound_proven : {r1.bound_proven}")
    print(f"    notes        : {r1.notes}")

    # ── 4b. Property 2: Total energy upper bound ──────────────────────────
    print("\n  Property 2: total_energy_upper_bound")

    # Get IBP bounds on sum(outputs) to find a provable upper limit
    ibp_sum_lb, ibp_sum_ub = _probe_sum_bounds(spec, method="IBP", n_voxels=n_voxels)
    crown_sum_lb, crown_sum_ub = _probe_sum_bounds(spec, method="CROWN", n_voxels=n_voxels)
    print(f"    IBP  sum bounds : [{ibp_sum_lb:.2f}, {ibp_sum_ub:.2f}] MeV")
    print(f"    CROWN sum bounds: [{crown_sum_lb:.2f}, {crown_sum_ub:.2f}] MeV")

    # Verify using IBP-proven upper bound (loose but formally true)
    prop2_ibp = total_energy_upper_bound(n_voxels, upper_mev=ibp_sum_ub + 1.0)
    r2_ibp = verify(spec, prop2_ibp, backend, config={"method": "IBP"})
    results.append(r2_ibp)
    print(f"\n    IBP upper={ibp_sum_ub + 1.0:.2f} MeV")
    print(f"      status  : {r2_ibp.status}  (bound_proven={r2_ibp.bound_proven})")

    # Verify using CROWN-tighter upper bound
    prop2_crown = total_energy_upper_bound(n_voxels, upper_mev=crown_sum_ub + 1.0)
    r2_crown = verify(spec, prop2_crown, backend, config={"method": "CROWN"})
    results.append(r2_crown)
    print(f"\n    CROWN upper={crown_sum_ub + 1.0:.2f} MeV")
    print(f"      status  : {r2_crown.status}  (bound_proven={r2_crown.bound_proven})")

    # ── 4c. Property 3: Energy conservation epsilon sweep (CROWN) ─────────
    print("\n  Property 3: energy_conservation epsilon sweep (CROWN)")
    target = (crown_sum_lb + crown_sum_ub) / 2.0
    print(f"    target = {target:.2f} MeV  (CROWN sum midpoint)")
    print(f"    CROWN window: [{crown_sum_lb:.2f}, {crown_sum_ub:.2f}] MeV")

    # Candidate epsilons: sweep from wide (should be PROVEN) to tight (likely INCONCLUSIVE)
    half_width = (crown_sum_ub - crown_sum_lb) / 2.0
    epsilon_candidates = [
        round(half_width * 1.01, 2),   # just outside the CROWN window → PROVEN
        round(half_width * 0.75, 2),
        round(half_width * 0.50, 2),
        round(half_width * 0.25, 2),
        round(half_width * 0.10, 2),
        round(half_width * 0.01, 2),
    ]

    tightest_proven_eps = None
    sweep_results = []
    for eps in epsilon_candidates:
        prop_e = energy_conservation(n_voxels, target_mev=target, epsilon_mev=eps)
        r_e = verify(spec, prop_e, backend, config={"method": "CROWN"})
        results.append(r_e)
        sweep_results.append({"epsilon": eps, "status": r_e.status})
        marker = "<-- tightest" if (r_e.status == "PROVEN" and tightest_proven_eps is None
                                    and all(x["status"] == "INCONCLUSIVE"
                                            for x in sweep_results[1:])) else ""
        if r_e.status == "PROVEN" and tightest_proven_eps is None:
            tightest_proven_eps = eps
        print(f"    epsilon={eps:>10.2f} MeV → {r_e.status}")

    if tightest_proven_eps is not None:
        print(f"\n    Tightest PROVEN epsilon: {tightest_proven_eps:.2f} MeV")
        print(f"    (over full input box: z∈[-1,1]^{Z_DIM}, log_e∈[0,{LOG_E_MAX:.4f}])")
    else:
        print("\n    All Conservation queries INCONCLUSIVE")

    # ── 5. Summary ────────────────────────────────────────────────────────
    print("\n[5/5] Summary")
    print("-" * 50)
    print(f"  {'Property':<35} {'Status':<15} {'Bound'}")
    print("-" * 50)
    for r in results:
        bnd = f"{r.bound_proven:.4f}" if r.bound_proven is not None else "—"
        print(f"  {r.property_name:<35} {r.status:<15} {bnd}")

    # ── Save results ──────────────────────────────────────────────────────
    if save_json:
        out_dir = Path(__file__).parent.parent.parent / "results" / "calorimeter"
        saved = []
        for r in results:
            p = save_result(r, out_dir, run_id=f"step5_{r.property_name}_{r.config.get('method','')}")
            saved.append(str(p))
        print(f"\n  Saved {len(saved)} result files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--n-voxels", type=int, default=368, choices=[368, 533])
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    main(n_epochs=args.n_epochs, save_json=args.save_json, n_voxels=args.n_voxels)
