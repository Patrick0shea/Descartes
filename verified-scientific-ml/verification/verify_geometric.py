"""
Formal verification of momentum conservation in the geometric surrogate
(Step 5), using the exact same Marabou pipeline, domain D, and
verify_momentum_property() logic as verification/verify_particle.py
(Step 4) -- only the ONNX model and the epsilon values tested differ
(verification/particle_properties.py: GEOMETRIC_MOMENTUM_PROPERTIES).
This is what makes the two models' verification results directly
comparable.

Run from the verified-scientific-ml/ directory, after
verification/export_geometric_onnx.py:
    python -m verification.verify_geometric
"""

from __future__ import annotations

import json
import os

from verification.export_geometric_onnx import ONNX_PATH
from verification.particle_properties import GEOMETRIC_MOMENTUM_PROPERTIES
from verification.verify_particle import verify_momentum_property

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "geometric_verification_results.json")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"{ONNX_PATH} not found. Run `python -m verification.export_geometric_onnx` first."
        )

    results = []
    for prop in GEOMETRIC_MOMENTUM_PROPERTIES:
        print(f"Verifying: {prop.description} ...")
        result = verify_momentum_property(ONNX_PATH, prop)
        match = "MATCHES" if result["result"] == prop.expect else "MISMATCH"
        print(
            f"  -> {result['result']}  ({result['verification_time_seconds']:.3f}s)"
            f"  [expected {prop.expect}, {match}]"
        )
        if result["counterexample"] is not None:
            ce = result["counterexample"]
            print(
                f"     counterexample state={['%.4f' % v for v in ce['input_state']]}"
                f" -> momentum_before={ce['momentum_before']:.6f},"
                f" momentum_after={ce['momentum_after']:.6f},"
                f" error={ce['momentum_error']:.6f}"
            )
        else:
            print(f"     no state in D violates the tolerance in either direction.")
        results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved verification results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
