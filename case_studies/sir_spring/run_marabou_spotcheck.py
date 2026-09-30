"""
Step 8: Marabou spot-checks on SIR and particle models.

Cross-validates the LiRPA results from Step 6 using the Marabou SMT solver.
Marabou is sound and complete (no false positives/negatives), so these spot-
checks confirm what LiRPA proved and find tighter counterexamples where LiRPA
was only INCONCLUSIVE.

Spot-checks performed
---------------------
SIR   Model B — Conservation at ε=1e-6  → expect PROVEN   (algebraic identity)
SIR   Model A — Conservation at ε=0.1   → expect COUNTEREXAMPLE (no structure)
SIR   Model C — Conservation at ε=0.018 → expect PROVEN   (within CROWN bound)

Particle Model B — Conservation at ε=1e-5 → expect PROVEN
Particle Model A — Conservation at ε=0.1  → expect COUNTEREXAMPLE

These are deliberately targeted spot-checks, not a full sweep.  Each query
exercises one data point on the verification/falsification diagram.

Usage
-----
    python -m case_studies.sir_spring.run_marabou_spotcheck [--save-json]
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
from verifier import MarabouBackend, ModelSpec, save_result, verify

# ── checkpoint paths ──────────────────────────────────────────────────────────

CKPT_DIR = (
    Path(__file__).parent.parent.parent
    / "verified-scientific-ml"
    / "models"
    / "checkpoints"
)

SIR_LB = [0.05, 0.01, 0.01]
SIR_UB = [0.95, 0.60, 0.90]
PARTICLE_LB = [-1.0, -1.0, -1.0, -1.0, -2.4, -2.4, -1.0, -1.0]
PARTICLE_UB = [1.0, 1.0, 1.0, 1.0, 2.4, 2.4, 1.0, 1.0]

# ── spot-check table ──────────────────────────────────────────────────────────
# Each entry specifies one targeted Marabou query.

SPOTCHECKS = [
    {
        "label": "SIR / Model B — ε=1e-6",
        "domain": "sir",
        "cls": SirGeometricMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "sir_geometric.pt",
        "n_dims": 3,
        "lb": SIR_LB,
        "ub": SIR_UB,
        "epsilon": 1e-6,
        "prop_fn": sir_population_conservation,
        "expect": "PROVEN",
        "rationale": "algebraic identity → UNSAT at any ε > 0",
    },
    {
        "label": "SIR / Model A — ε=0.1",
        "domain": "sir",
        "cls": SirMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "sir_mlp.pt",
        "n_dims": 3,
        "lb": SIR_LB,
        "ub": SIR_UB,
        "epsilon": 0.1,
        "prop_fn": sir_population_conservation,
        "expect": "COUNTEREXAMPLE",
        "rationale": "no conservation structure → SAT (finds violating input)",
    },
    {
        "label": "SIR / Model C — ε=0.018",
        "domain": "sir",
        "cls": SirSoftGeometricMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "sir_soft_geometric.pt",
        "n_dims": 3,
        "lb": SIR_LB,
        "ub": SIR_UB,
        "epsilon": 0.018,
        "prop_fn": sir_population_conservation,
        "expect": "PROVEN",
        "rationale": "within CROWN proven bound (0.017950) so should be PROVEN",
    },
    {
        "label": "Particle / Model B — ε=1e-5",
        "domain": "particle",
        "cls": ParticleGeometricMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "particle_geometric.pt",
        "n_dims": 8,
        "lb": PARTICLE_LB,
        "ub": PARTICLE_UB,
        "epsilon": 1e-5,
        "prop_fn": particle_x_momentum_conservation,
        "expect": "PROVEN",
        "rationale": "algebraic identity → UNSAT at any ε > 0",
    },
    {
        "label": "Particle / Model A — ε=0.1",
        "domain": "particle",
        "cls": ParticleMLP,
        "kwargs": {"hidden_dim": 16},
        "ckpt": "particle_mlp.pt",
        "n_dims": 8,
        "lb": PARTICLE_LB,
        "ub": PARTICLE_UB,
        "epsilon": 0.1,
        "prop_fn": particle_x_momentum_conservation,
        "expect": "COUNTEREXAMPLE",
        "rationale": "no conservation structure → SAT",
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_spec(entry: dict) -> ModelSpec:
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


# ── main ─────────────────────────────────────────────────────────────────────

def main(save_json: bool = False) -> None:
    print("=" * 68)
    print("Step 8: Marabou spot-checks — SIR + Particle")
    print("=" * 68)

    backend = MarabouBackend()
    config = {"timeout_s": 120}
    out_dir = Path(__file__).parent.parent.parent / "results"
    summary_rows = []

    for entry in SPOTCHECKS:
        print(f"\n{'─'*60}")
        print(f"  {entry['label']}")
        print(f"  Rationale : {entry['rationale']}")
        print(f"{'─'*60}")

        spec = _load_spec(entry)
        prop = entry["prop_fn"](epsilon=entry["epsilon"])
        result = verify(spec, prop, backend, config=config, seed=42)

        match = "✓" if result.status == entry["expect"] else "✗ UNEXPECTED"
        print(f"  status   : {result.status}  {match}")
        print(f"  expected : {entry['expect']}")
        print(f"  runtime  : {result.runtime_s:.2f}s")
        if result.counterexample is not None:
            ce = [f"{v:.4f}" for v in result.counterexample]
            print(f"  cex      : [{', '.join(ce)}]")
        print(f"  notes    : {result.notes[:80]}")

        summary_rows.append({
            "label": entry["label"],
            "status": result.status,
            "expected": entry["expect"],
            "ok": result.status == entry["expect"],
        })

        if save_json:
            domain_dir = out_dir / entry["domain"]
            run_id = f"step8_{entry['ckpt'].replace('.pt','')}_eps{entry['epsilon']:g}"
            p = save_result(result, domain_dir, run_id=run_id)
            print(f"  Saved → {p}")

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  Marabou spot-check summary")
    print(f"{'─'*68}")
    all_ok = True
    for row in summary_rows:
        tick = "PASS" if row["ok"] else "FAIL"
        print(f"  [{tick}]  {row['label']:<40} {row['status']}")
        if not row["ok"]:
            all_ok = False
    print(f"{'='*68}")
    if all_ok:
        print("  All spot-checks matched expected results.")
    else:
        print("  Some spot-checks did not match expected results — investigate above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    main(save_json=args.save_json)
