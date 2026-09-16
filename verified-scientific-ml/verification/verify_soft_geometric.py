"""
Formal verification of momentum conservation in the soft-geometric
surrogate (Model C), using Marabou over an epsilon sweep on
SOFT_GEOMETRIC_MOMENTUM_PROPERTIES.

This is the key scientific comparison for Model C. Unlike Model B
(verify_geometric.py), which is UNSAT at any epsilon above float32
noise because momentum conservation is an exact algebraic identity of
the architecture, Model C has no such structural guarantee -- so Marabou
will find SAT (a counterexample) once epsilon is tight enough. The
epsilon at which the transition from UNSAT to SAT occurs characterises
how well Model C's training signal has pushed the learned velocity
updates toward momentum conservation, compared to Model A's (plain MLP)
transition point.

SOFT_GEOMETRIC_MOMENTUM_PROPERTIES (defined in particle_properties.py)
contains one property per (axis, epsilon) pair sweeping
[1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5] -- six epsilons for each of the
two axes (Px, Py), twelve properties total. All are set to expect="UNSAT"
as the hypothesis; Marabou's actual answer (SAT or UNSAT) reveals the
true verification frontier.

Run from the verified-scientific-ml/ directory, after
verification/export_soft_geometric_onnx.py:
    python -m verification.verify_soft_geometric
"""

from __future__ import annotations

import json
import os

from verification.export_soft_geometric_onnx import ONNX_PATH
from verification.particle_properties import SOFT_GEOMETRIC_MOMENTUM_PROPERTIES
from verification.verify_particle import verify_momentum_property

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "soft_geometric_verification_results.json")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"{ONNX_PATH} not found. Run `python -m verification.export_soft_geometric_onnx` first."
        )

    results = []
    for prop in SOFT_GEOMETRIC_MOMENTUM_PROPERTIES:
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
