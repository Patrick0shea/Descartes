"""
Step 7: PGD falsification baseline for SIR and particle models.

For each model (A/B/C), run PGD on the DeltaModel to find the empirical
worst-case conservation violation over the full input box.  Compare against
the LiRPA-proven upper bounds from Step 6.

PGD is a heuristic: a found violation is real evidence the property fails at
that epsilon; no violation found does NOT prove the property holds everywhere.

Key expected results
--------------------
  Model B: violation ≈ 0  (algebraic identity → gradient of loss is always 0)
  Model C: small violation (soft constraint; real but bounded by training)
  Model A: large violation (no physical structure; MLP can freely deviate)

Usage
-----
    python -m case_studies.sir_spring.run_falsification [--save-json]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

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
    sir_population_conservation,
)
from verifier import ModelSpec, falsify

# ── checkpoint paths ──────────────────────────────────────────────────────────

CKPT_DIR = (
    Path(__file__).parent.parent.parent
    / "verified-scientific-ml"
    / "models"
    / "checkpoints"
)

# ── input bounds (identical to run_verification.py) ───────────────────────────

SIR_LB = [0.05, 0.01, 0.01]
SIR_UB = [0.95, 0.60, 0.90]

PARTICLE_LB = [-1.0, -1.0, -1.0, -1.0, -2.4, -2.4, -1.0, -1.0]
PARTICLE_UB = [1.0, 1.0, 1.0, 1.0, 2.4, 2.4, 1.0, 1.0]

# ── model table ───────────────────────────────────────────────────────────────
# lirpa_bound: CROWN worst-case |sum(delta)| upper bound from Step 6 result
# files.  None = LiRPA was INCONCLUSIVE at all tested epsilon.

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
        "prop_fn": sir_population_conservation,
        "lirpa_bound": None,      # INCONCLUSIVE at all ε
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
        "prop_fn": sir_population_conservation,
        "lirpa_bound": 0.0,       # CROWN bound = 0.0 (exact algebraic identity)
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
        "prop_fn": sir_population_conservation,
        "lirpa_bound": 0.017950,  # from step6_sir_soft_geometric.json
    },
    {
        "label": "Particle / Model A (MLP)",
        "domain": "particle",
        "cls": ParticleMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "particle_mlp.pt",
        "n_dims": 8,
        "lb": PARTICLE_LB,
        "ub": PARTICLE_UB,
        "prop_fn": particle_x_momentum_conservation,
        "lirpa_bound": None,      # INCONCLUSIVE at all ε
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
        "prop_fn": particle_x_momentum_conservation,
        "lirpa_bound": 0.0,       # CROWN bound = 0.0 (exact algebraic identity)
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
        "prop_fn": particle_x_momentum_conservation,
        "lirpa_bound": 1.064443,  # from step6_particle_soft_geometric.json
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_spec(entry: dict) -> ModelSpec:
    """Load checkpoint, wrap in DeltaModel, build ModelSpec."""
    base = entry["cls"](**entry["kwargs"])
    state = torch.load(str(CKPT_DIR / entry["ckpt"]), map_location="cpu", weights_only=True)
    base.load_state_dict(state)
    base.eval()

    delta = DeltaModel(base, n_dims=entry["n_dims"])
    delta.eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(delta.state_dict(), tmp_path)

    fresh_delta = DeltaModel(entry["cls"](**entry["kwargs"]), n_dims=entry["n_dims"])
    spec = ModelSpec.from_checkpoint(
        fresh_delta, tmp_path,
        input_lb=entry["lb"],
        input_ub=entry["ub"],
        name=entry["ckpt"].replace(".pt", "_delta"),
    )
    Path(tmp_path).unlink(missing_ok=True)
    return spec


def _save_falsification(result, output_dir: Path, run_id: str) -> Path:
    """Write FalsificationResult JSON to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}.json"
    path.write_text(result.to_json())
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main(save_json: bool = False) -> None:
    print("=" * 68)
    print("Step 7: SIR + Particle PGD falsification baseline")
    print("  epsilon=1e-9 → reports empirical worst-case |sum(delta)|")
    print("=" * 68)

    out_dir = Path(__file__).parent.parent.parent / "results"
    summary_rows = []

    for entry in MODELS:
        print(f"\n{'─'*60}")
        print(f"  {entry['label']}")
        print(f"{'─'*60}")

        spec = _load_spec(entry)

        # epsilon=1e-9 makes the violation function ≈ |sum(coeff @ delta)|,
        # so violation_magnitude ≈ empirical worst-case conservation error.
        prop = entry["prop_fn"](epsilon=1e-9)
        result = falsify(
            spec, prop,
            n_restarts=200,
            n_steps=500,
            step_size=0.005,
            seed=42,
        )

        viol = result.violation_magnitude
        viol_str = f"{viol:.6f}" if viol is not None else "0 (exact)"
        print(f"  status          : {result.status}")
        print(f"  violation_mag   : {viol_str}")
        lirpa = entry["lirpa_bound"]
        if lirpa is not None:
            print(f"  LiRPA bound     : {lirpa:.6f}")
            if viol is not None and viol > 0 and lirpa > 0:
                print(f"  LiRPA/PGD ratio : {lirpa / viol:.2f}×  (CROWN is this much looser than PGD empirical worst-case)")
            elif lirpa == 0.0:
                print(f"  LiRPA bound     : 0 (algebraic identity)")
        else:
            print(f"  LiRPA bound     : INCONCLUSIVE")
        print(f"  runtime_s       : {result.runtime_s:.2f}s")

        summary_rows.append({
            "model": entry["label"],
            "pgd_status": result.status,
            "pgd_violation": viol,
            "lirpa_bound": lirpa,
        })

        if save_json:
            domain_dir = out_dir / entry["domain"]
            run_id = f"step7_{entry['ckpt'].replace('.pt','')}"
            p = _save_falsification(result, domain_dir, run_id)
            print(f"  Saved → {p}")

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  Step 7 summary: PGD worst-case violation vs LiRPA bound")
    print(f"{'─'*68}")
    print(f"  {'Model':<42} {'PGD violation':<16} {'LiRPA bound'}")
    print(f"{'─'*68}")
    for row in summary_rows:
        viol_s = f"{row['pgd_violation']:.6f}" if row["pgd_violation"] else "0 (exact)"
        bound_s = f"{row['lirpa_bound']:.6f}" if row["lirpa_bound"] is not None else "INCONCLUSIVE"
        print(f"  {row['model']:<42} {viol_s:<16} {bound_s}")
    print(f"{'='*68}")
    print()
    print("  Interpretation:")
    print("  - Model B: PGD violation = 0 → algebraic identity, CROWN bound = 0 agree")
    print("  - Model C: PGD violation > 0 → real (non-exact) constraint; CROWN ≥ PGD")
    print("  - Model A: PGD violation > 0 → no conservation structure; LiRPA INCONCLUSIVE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    main(save_json=args.save_json)
