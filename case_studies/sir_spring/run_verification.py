"""
Step 6: Generality test — SIR and particle models through the framework.

Loads existing checkpoints from verified-scientific-ml/, wraps each model in
DeltaModel (to express conservation as |coeff @ delta| <= epsilon), builds
ModelSpec, and runs Conservation epsilon sweeps via auto_LiRPA.

Comparison with earlier Marabou results (from README, verified-scientific-ml/):
    Model A (MLP)   SIR: ε=0.1    Particle: ε=1.0
    Model C (soft)  SIR: ε=1e-2   Particle: ε=1e-2
    Model B (hard)  SIR: ε≤1e-6   Particle: ε=1e-5

auto_LiRPA cannot prove the algebraic identity in Model B (IBP/CROWN don't
cancel dual-branch bounds) so it will be INCONCLUSIVE at tight epsilon there.
This is honest and expected; the comparison shows where each tool shines.

Usage
-----
    python -m case_studies.sir_spring.run_verification [--save-json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from case_studies.sir_spring.delta_model import DeltaModel
from case_studies.sir_spring.models.particle import (
    ParticleGeometricMLP,
    ParticleMLP,
    ParticleSoftGeometricMLP,
)
from case_studies.sir_spring.models.sir import (
    SirGeometricMLP,
    SirMLP,
    SirSoftGeometricMLP,
)
from case_studies.sir_spring.properties import (
    particle_x_momentum_conservation,
    particle_y_momentum_conservation,
    sir_population_conservation,
)
from verifier import LiRPABackend, ModelSpec, save_result, verify

# ── checkpoint paths ──────────────────────────────────────────────────────────

CKPT_DIR = Path(__file__).parent.parent.parent / "verified-scientific-ml" / "models" / "checkpoints"

# ── input bounds ──────────────────────────────────────────────────────────────

SIR_LB = [0.05, 0.01, 0.01]   # [s_min, i_min, r_min]
SIR_UB = [0.95, 0.60, 0.90]   # [s_max, i_max, r_max]

# State layout: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
PARTICLE_LB = [-1.0, -1.0, -1.0, -1.0, -2.4, -2.4, -1.0, -1.0]
PARTICLE_UB = [ 1.0,  1.0,  1.0,  1.0,  2.4,  2.4,  1.0,  1.0]

# ── epsilon sweeps ────────────────────────────────────────────────────────────
# From loose (provable) to tight (may be INCONCLUSIVE).
# These bracket the known Marabou results.

SIR_EPSILONS     = [1.0, 0.5, 0.1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]
PARTICLE_EPSILONS = [2.0, 1.0, 0.5, 0.1, 5e-2, 1e-2, 5e-3, 1e-3]

# ── model table ───────────────────────────────────────────────────────────────

MODELS = [
    {
        "label": "SIR / Model A (MLP)",
        "domain": "sir",
        "cls": SirMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "sir_mlp.pt",
        "n_dims": 3,
        "lb": SIR_LB,
        "ub": SIR_UB,
        "epsilons": SIR_EPSILONS,
        "prop_fn": sir_population_conservation,
        "marabou_result": 0.1,   # tightest PROVEN from earlier experiment
    },
    {
        "label": "SIR / Model B (geometric)",
        "domain": "sir",
        "cls": SirGeometricMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "sir_geometric.pt",
        "n_dims": 3,
        "lb": SIR_LB,
        "ub": SIR_UB,
        "epsilons": SIR_EPSILONS,
        "prop_fn": sir_population_conservation,
        "marabou_result": 1e-6,
    },
    {
        "label": "SIR / Model C (soft-geometric)",
        "domain": "sir",
        "cls": SirSoftGeometricMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "sir_soft_geometric.pt",
        "n_dims": 3,
        "lb": SIR_LB,
        "ub": SIR_UB,
        "epsilons": SIR_EPSILONS,
        "prop_fn": sir_population_conservation,
        "marabou_result": 1e-2,
    },
    {
        "label": "Particle / Model A (MLP)",
        "domain": "particle",
        "cls": ParticleMLP,
        "kwargs": {"hidden_dim": 16},   # checkpoint was trained with 16, not the class default of 128
        "ckpt": "particle_mlp.pt",
        "n_dims": 8,
        "lb": PARTICLE_LB,
        "ub": PARTICLE_UB,
        "epsilons": PARTICLE_EPSILONS,
        "prop_fn": particle_x_momentum_conservation,
        "marabou_result": 1.0,
    },
    {
        "label": "Particle / Model B (geometric)",
        "domain": "particle",
        "cls": ParticleGeometricMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "particle_geometric.pt",
        "n_dims": 8,
        "lb": PARTICLE_LB,
        "ub": PARTICLE_UB,
        "epsilons": PARTICLE_EPSILONS,
        "prop_fn": particle_x_momentum_conservation,
        "marabou_result": 1e-5,
    },
    {
        "label": "Particle / Model C (soft-geometric)",
        "domain": "particle",
        "cls": ParticleSoftGeometricMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "particle_soft_geometric.pt",
        "n_dims": 8,
        "lb": PARTICLE_LB,
        "ub": PARTICLE_UB,
        "epsilons": PARTICLE_EPSILONS,
        "prop_fn": particle_x_momentum_conservation,
        "marabou_result": 1e-2,
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_spec(entry: dict) -> ModelSpec:
    """Instantiate model, load checkpoint, wrap in DeltaModel, build ModelSpec."""
    base = entry["cls"](**entry["kwargs"])
    ckpt_path = str(CKPT_DIR / entry["ckpt"])

    # Load the base model weights
    import torch
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(state)
    base.eval()

    # Wrap in DeltaModel and save the delta state_dict to a temp checkpoint
    import tempfile
    delta = DeltaModel(base, n_dims=entry["n_dims"])
    delta.eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(delta.state_dict(), tmp_path)

    # Build ModelSpec from the delta checkpoint
    fresh_delta = DeltaModel(entry["cls"](**entry["kwargs"]), n_dims=entry["n_dims"])
    spec = ModelSpec.from_checkpoint(
        fresh_delta, tmp_path,
        input_lb=entry["lb"],
        input_ub=entry["ub"],
        name=entry["ckpt"].replace(".pt", "_delta"),
    )
    Path(tmp_path).unlink(missing_ok=True)
    return spec


def _epsilon_sweep(spec: ModelSpec, entry: dict, backend: LiRPABackend) -> dict:
    """
    Run Conservation epsilon sweep and return {epsilon → status} dict.
    Returns also the tightest proven epsilon (or None).
    """
    results = {}
    tightest = None
    for eps in entry["epsilons"]:
        prop = entry["prop_fn"](eps)
        r = verify(spec, prop, backend, config={"method": "CROWN"})
        results[eps] = r.status
        if r.status == "PROVEN" and tightest is None:
            tightest = eps
    return results, tightest


# ── main ─────────────────────────────────────────────────────────────────────

def main(save_json: bool = False) -> None:
    print("=" * 68)
    print("Step 6: SIR + Particle conservation verification (LiRPA vs Marabou)")
    print("=" * 68)

    backend = LiRPABackend()
    summary_rows = []

    out_dir = Path(__file__).parent.parent.parent / "results"

    for entry in MODELS:
        print(f"\n{'─'*60}")
        print(f"  {entry['label']}")
        print(f"{'─'*60}")

        spec = _load_spec(entry)
        print(f"  checkpoint hash : {spec.checkpoint_hash}")

        sweep, tightest = _epsilon_sweep(spec, entry, backend)

        for eps, status in sweep.items():
            marker = " ← tightest PROVEN" if eps == tightest else ""
            print(f"    ε={eps:<10g}  {status}{marker}")

        marabou_ref = entry["marabou_result"]
        lirpa_str = f"{tightest:g}" if tightest is not None else "none"
        comparison = (
            f"LiRPA={lirpa_str}  Marabou(ref)={marabou_ref:g}"
        )
        print(f"\n  {comparison}")
        if tightest is not None and tightest > marabou_ref:
            print(f"  → LiRPA bound is {tightest / marabou_ref:.0f}× looser (expected for CROWN vs SMT)")
        elif tightest is not None and abs(tightest - marabou_ref) < 1e-12:
            print(f"  → LiRPA matches Marabou")
        elif tightest is None:
            print(f"  → LiRPA INCONCLUSIVE at all tested ε")

        summary_rows.append({
            "model": entry["label"],
            "lirpa_tightest": tightest,
            "marabou_ref": marabou_ref,
            "sweep": {str(k): v for k, v in sweep.items()},
        })

        if save_json:
            domain = entry["domain"]
            domain_dir = out_dir / domain
            run_id = f"step6_{entry['ckpt'].replace('.pt','')}"
            # Build a VerificationResult for the tightest proven point
            if tightest is not None:
                prop = entry["prop_fn"](tightest)
                r = verify(spec, prop, backend, config={"method": "CROWN"})
                p = save_result(r, domain_dir, run_id=run_id)
                print(f"  Saved → {p}")

    # ── print summary table ───────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  {'Model':<42} {'LiRPA ε':<12} {'Marabou ε (ref)'}")
    print(f"{'─'*68}")
    for row in summary_rows:
        lirpa = f"{row['lirpa_tightest']:g}" if row["lirpa_tightest"] is not None else "—"
        ref = f"{row['marabou_ref']:g}"
        print(f"  {row['model']:<42} {lirpa:<12} {ref}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    main(save_json=args.save_json)
