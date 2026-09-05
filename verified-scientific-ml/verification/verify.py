"""
Formal verification of the MLP surrogate's input/output bounds using
Marabou.

For every property in verification/properties.py, this searches -- over
the full continuous domain x in [-2,2], v in [-2,2] -- for a
counterexample to the NEGATED property (see properties.py for exactly
what that means). This is a real search over the domain, not a sample of
points: UNSAT is a proof that no counterexample exists anywhere in the
domain; SAT gives one concrete counterexample.

Run from the verified-scientific-ml/ directory, after
verification/export_onnx.py:
    python -m verification.verify
"""

from __future__ import annotations

import json
import os
import time

from maraboupy import Marabou

from verification.export_onnx import ONNX_PATH
from verification.properties import (
    OUTPUT_NAMES,
    PROPERTIES,
    V_MAX,
    V_MIN,
    X_MAX,
    X_MIN,
    Property,
)

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "verification_results.json")


def verify_property(onnx_path: str, prop: Property) -> dict:
    """Run Marabou on a single property; return a result record (see module docstring)."""
    network = Marabou.read_onnx(onnx_path)
    in_vars = network.inputVars[0].flatten()
    out_vars = network.outputVars[0].flatten()

    network.setLowerBound(in_vars[0], X_MIN)
    network.setUpperBound(in_vars[0], X_MAX)
    network.setLowerBound(in_vars[1], V_MIN)
    network.setUpperBound(in_vars[1], V_MAX)

    out_var = out_vars[prop.output_index]
    if prop.bound_type == "ub":
        # claim: output <= bound_value  -->  negate: output >= bound_value
        network.setLowerBound(out_var, prop.bound_value)
    else:
        # claim: output >= bound_value  -->  negate: output <= bound_value
        network.setUpperBound(out_var, prop.bound_value)

    start = time.time()
    exit_code, values, stats = network.solve(options=Marabou.createOptions(verbosity=0))
    wall_time = time.time() - start

    result = {
        "name": prop.name,
        "description": prop.description,
        "input_domain": {"x": [X_MIN, X_MAX], "v": [V_MIN, V_MAX]},
        "output": OUTPUT_NAMES[prop.output_index],
        "bound_type": prop.bound_type,
        "bound_value": prop.bound_value,
        "expected": prop.expect,
        "result": exit_code.upper(),
        "verification_time_seconds": wall_time,
        "marabou_time_seconds": stats.getTotalTimeInMicro() / 1e6,
        "counterexample": None,
    }

    if exit_code == "sat":
        result["counterexample"] = {
            "x": values[in_vars[0]],
            "v": values[in_vars[1]],
            "x_next": values[out_vars[0]],
            "v_next": values[out_vars[1]],
        }

    return result


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"{ONNX_PATH} not found. Run `python -m verification.export_onnx` first."
        )

    results = []
    for prop in PROPERTIES:
        print(f"Verifying: {prop.description} ...")
        result = verify_property(ONNX_PATH, prop)
        match = "MATCHES" if result["result"] == prop.expect else "MISMATCH"
        print(
            f"  -> {result['result']}  ({result['verification_time_seconds']:.3f}s)"
            f"  [expected {prop.expect}, {match}]"
        )
        if result["counterexample"] is not None:
            ce = result["counterexample"]
            print(
                f"     counterexample: x={ce['x']:.6f}, v={ce['v']:.6f} -> "
                f"x_next={ce['x_next']:.6f}, v_next={ce['v_next']:.6f}"
            )
        results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved verification results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
