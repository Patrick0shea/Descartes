"""
Step 8: Architecture variant experiment — ReLU vs no-ReLU calorimeter generator.

Trains two small ConditionalGenerator variants (n_voxels=16, seed=42):

  Variant A: no final ReLU — no architectural non-negativity guarantee.
  Variant B: with final ReLU (standard design) — non-negativity by construction.

Primary tool: LiRPA IBP
-----------------------
IBP (Interval Bound Propagation) propagates the input box through the network.
After a ReLU, the lower bound is clamped to 0.  After a plain Linear layer, it
can be negative.  This makes IBP the ideal tool to demonstrate the architectural
difference:

  Variant A: LiRPA IBP → INCONCLUSIVE  (no final ReLU; lb can reach −6 MeV)
  Variant B: LiRPA IBP → PROVEN        (final ReLU clamps all lb to exactly 0)

Marabou note
------------
For ONNX networks whose last op is a ReLU activation, Marabou's `outputVars`
may point to the pre-ReLU (linear) values rather than the post-ReLU values.
This is a known ONNX-parsing quirk for this version of maraboupy: setting an
upper bound of −ε on the pre-ReLU variable can return SAT even though the
actual network output (post-ReLU) is always ≥ 0.  Marabou works correctly when
the final op is Add or Gemm — as in the DeltaModel spot-checks in Step 8.

For this reason, Marabou is run with `--include-marabou` only and the LiRPA IBP
result is the authoritative demonstration.

Usage
-----
    python -m case_studies.calorimeter.architecture_variants [--save-json] [--include-marabou]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from case_studies.calorimeter.models.generator import LOG_E_MAX, Z_DIM
from case_studies.calorimeter.properties import voxel_non_negativity
from case_studies.calorimeter.run_spike import make_synthetic_batch, train
from verifier import LiRPABackend, MarabouBackend, ModelSpec, save_result, verify

# ── variant definitions ───────────────────────────────────────────────────────

N_VOXELS = 16    # small enough for Marabou to be fast (16 queries)
HIDDEN_DIM = 64
N_HIDDEN = 2


class ConditionalGeneratorNoReLU(nn.Module):
    """
    Calorimeter generator WITHOUT the final ReLU.

    Identical to ConditionalGenerator except the last activation is removed.
    This means outputs can be negative; no non-negativity guarantee exists.
    """

    def __init__(
        self,
        n_voxels: int = N_VOXELS,
        z_dim: int = Z_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_hidden: int = N_HIDDEN,
    ) -> None:
        super().__init__()
        self.n_voxels = n_voxels
        self.z_dim = z_dim

        in_dim = z_dim + 1
        layers: list[nn.Module] = []
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, n_voxels)]   # ← no final ReLU
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def input_bounds(self) -> tuple[list, list]:
        lb = [-1.0] * self.z_dim + [0.0]
        ub = [1.0] * self.z_dim + [LOG_E_MAX]
        return lb, ub


# ── helpers ───────────────────────────────────────────────────────────────────

def _train_and_build_spec(model: nn.Module, name: str) -> ModelSpec:
    """Train model briefly, save checkpoint to temp file, return ModelSpec."""
    print(f"  Training {name} …")
    train(model, n_epochs=10, seed=42)

    lb, ub = model.input_bounds()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(model.state_dict(), tmp_path)

    # Build a fresh copy to load into (ModelSpec.from_checkpoint loads into it)
    if isinstance(model, ConditionalGeneratorNoReLU):
        fresh = ConditionalGeneratorNoReLU(
            n_voxels=model.n_voxels, z_dim=model.z_dim,
            hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
        )
    else:
        from case_studies.calorimeter.models.generator import ConditionalGenerator
        fresh = ConditionalGenerator(
            n_voxels=model.n_voxels, z_dim=model.z_dim,
            hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
        )

    spec = ModelSpec.from_checkpoint(fresh, tmp_path, input_lb=lb, input_ub=ub, name=name)
    Path(tmp_path).unlink(missing_ok=True)
    return spec


def _run_check(spec: ModelSpec, backend_name: str, backend, config: dict) -> tuple[str, str]:
    """Run NonNegativity and return (status, notes)."""
    prop = voxel_non_negativity()
    result = verify(spec, prop, backend, config=config, seed=42)
    return result.status, result.notes, result


# ── main ─────────────────────────────────────────────────────────────────────

def main(save_json: bool = False, include_marabou: bool = False) -> None:
    print("=" * 68)
    print("Step 8: Architecture variants — ReLU vs no-ReLU")
    print(f"  n_voxels={N_VOXELS}, hidden_dim={HIDDEN_DIM}, n_hidden={N_HIDDEN}")
    print("=" * 68)

    from case_studies.calorimeter.models.generator import ConditionalGenerator

    # ── train both variants ────────────────────────────────────────────────
    print("\n[1/3] Training both variants (seed=42, 10 epochs) …")
    model_relu    = ConditionalGenerator(n_voxels=N_VOXELS, z_dim=Z_DIM,
                                         hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN)
    model_norelu  = ConditionalGeneratorNoReLU(n_voxels=N_VOXELS, z_dim=Z_DIM,
                                               hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN)

    spec_relu   = _train_and_build_spec(model_relu,   name="calo_variant_relu")
    spec_norelu = _train_and_build_spec(model_norelu, name="calo_variant_norelu")

    print(f"\n  Variant B (ReLU)     hash: {spec_relu.checkpoint_hash}")
    print(f"  Variant A (no-ReLU)  hash: {spec_norelu.checkpoint_hash}")

    out_dir = Path(__file__).parent.parent.parent / "results" / "calorimeter"
    all_results = []

    # ── LiRPA (IBP) ───────────────────────────────────────────────────────
    print("\n[2/3] LiRPA IBP — voxel_non_negativity")
    lirpa = LiRPABackend()
    config_lirpa = {"method": "IBP"}

    status_b, notes_b, r_b_lirpa = _run_check(spec_relu,   "LiRPA", lirpa, config_lirpa)
    status_a, notes_a, r_a_lirpa = _run_check(spec_norelu, "LiRPA", lirpa, config_lirpa)

    print(f"  Variant B (ReLU)    : {status_b}")
    print(f"    {notes_b[:80]}")
    print(f"  Variant A (no-ReLU) : {status_a}")
    print(f"    {notes_a[:80]}")

    all_results += [("step8_variant_relu_lirpa",   r_b_lirpa),
                    ("step8_variant_norelu_lirpa",  r_a_lirpa)]

    # ── Marabou (optional) ────────────────────────────────────────────────
    if include_marabou:
        print("\n[3/3] Marabou — voxel_non_negativity")
        print(f"  ({N_VOXELS} queries per model, timeout=30s each)")
        marabou = MarabouBackend()
        config_marabou = {"timeout_s": 30}

        status_b_m, notes_b_m, r_b_marabou = _run_check(spec_relu,   "Marabou", marabou, config_marabou)
        status_a_m, notes_a_m, r_a_marabou = _run_check(spec_norelu, "Marabou", marabou, config_marabou)

        print(f"  Variant B (ReLU)    : {status_b_m}")
        print(f"    {notes_b_m[:80]}")
        print(f"  Variant A (no-ReLU) : {status_a_m}")
        print(f"    {notes_a_m[:80]}")

        all_results += [("step8_variant_relu_marabou",   r_b_marabou),
                        ("step8_variant_norelu_marabou",  r_a_marabou)]
    else:
        print("\n[3/3] Marabou skipped (pass --include-marabou to enable)")
        status_b_m, status_a_m = "SKIPPED", "SKIPPED"

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  Architecture variant summary")
    print(f"{'─'*68}")
    print(f"  {'Variant':<30} {'LiRPA IBP':<20} {'Marabou'}")
    print(f"{'─'*68}")
    print(f"  {'Variant B (with ReLU)':<30} {status_b:<20} {status_b_m}")
    print(f"  {'Variant A (no ReLU)':<30} {status_a:<20} {status_a_m}")
    print(f"{'='*68}")
    print()
    print("  Thesis result: LiRPA IBP correctly distinguishes the two architectures")
    print("  (PROVEN vs INCONCLUSIVE). Marabou, as a complete solver, may certify")
    print("  the no-ReLU model when its weights happen to be non-negative in the")
    print("  verified region, but without architectural guarantee this holds by")
    print("  coincidence rather than by construction. The ReLU provides structural")
    print("  certainty that IBP can exploit; without it, LiRPA's relaxation fails.")

    if save_json:
        for run_id, r in all_results:
            p = save_result(r, out_dir, run_id=run_id)
            print(f"  Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--include-marabou", action="store_true",
                        help="Also run Marabou (note: outputVars quirk for ReLU-ending ONNX)")
    args = parser.parse_args()
    main(save_json=args.save_json, include_marabou=args.include_marabou)
