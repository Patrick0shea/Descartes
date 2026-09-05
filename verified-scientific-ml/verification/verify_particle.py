"""
Formal verification of momentum conservation in the particle MLP
surrogate, using Marabou.

For each MomentumProperty in verification/particle_properties.py, this
searches -- over the full continuous domain D (see particle_properties.py)
-- for a counterexample to the claim |P_next - P| <= epsilon, where P is
Px or Py depending on the property's axis. The claim is checked as TWO
separate one-sided linear queries (see module docstring in
particle_properties.py):

    upper violation: P_next - P >= epsilon
    lower violation: P - P_next >= epsilon

Both are linear in the network's input/output variables (P and P_next
are each a sum of two velocity components), so each is encoded directly
as a single Marabou linear inequality over those four variables -- no
approximation of the momentum expression is needed.

If BOTH one-sided queries are UNSAT, no state in D violates the
tolerance in either direction: the property is formally verified for D.
If EITHER is SAT, that query's counterexample is reported and the
overall property result is SAT.

Run from the verified-scientific-ml/ directory, after
verification/export_particle_onnx.py:
    python -m verification.verify_particle
"""

from __future__ import annotations

import json
import os
import time

from maraboupy import Marabou

from verification.export_particle_onnx import ONNX_PATH
from verification.particle_properties import (
    DOMAIN_LOWER,
    DOMAIN_UPPER,
    MOMENTUM_PROPERTIES,
    SEPARATION_BOUND,
    VX1,
    VX2,
    VY1,
    VY2,
    X1,
    X2,
    Y1,
    Y2,
    MomentumProperty,
)

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "particle_verification_results.json")


def _solve_one_sided(onnx_path: str, axis_vars, epsilon: float, direction: str):
    """
    Solve one of the two one-sided violation queries for a momentum
    axis. axis_vars = (i, j) are the indices (into the shared 8-dim
    state layout) of the two velocity components that sum to P / P_next
    -- e.g. (VX1, VX2) for Px. Since input and output share the same
    layout, these indices are used for both in_vars and out_vars.

    direction = "upper": search for P_next - P >= epsilon
    direction = "lower": search for P - P_next >= epsilon
    """
    network = Marabou.read_onnx(onnx_path)
    in_vars = network.inputVars[0].flatten()
    out_vars = network.outputVars[0].flatten()

    for i, (lo, hi) in enumerate(zip(DOMAIN_LOWER, DOMAIN_UPPER)):
        network.setLowerBound(in_vars[i], lo)
        network.setUpperBound(in_vars[i], hi)

    # Tighten the box to the actual sampled (disk-shaped) region: bound
    # the particle separation components (see particle_properties.py).
    #   x2 - x1 <= SEPARATION_BOUND
    network.addInequality([in_vars[X2], in_vars[X1]], [1.0, -1.0], SEPARATION_BOUND)
    #   x1 - x2 <= SEPARATION_BOUND  (i.e. x2 - x1 >= -SEPARATION_BOUND)
    network.addInequality([in_vars[X1], in_vars[X2]], [1.0, -1.0], SEPARATION_BOUND)
    #   y2 - y1 <= SEPARATION_BOUND
    network.addInequality([in_vars[Y2], in_vars[Y1]], [1.0, -1.0], SEPARATION_BOUND)
    #   y1 - y2 <= SEPARATION_BOUND
    network.addInequality([in_vars[Y1], in_vars[Y2]], [1.0, -1.0], SEPARATION_BOUND)

    v1_in, v2_in = in_vars[axis_vars[0]], in_vars[axis_vars[1]]
    v1_out, v2_out = out_vars[axis_vars[0]], out_vars[axis_vars[1]]

    if direction == "upper":
        # P_next - P >= epsilon  <=>  -v1_out - v2_out + v1_in + v2_in <= -epsilon
        network.addInequality([v1_out, v2_out, v1_in, v2_in], [-1.0, -1.0, 1.0, 1.0], -epsilon)
    else:
        # P - P_next >= epsilon  <=>  -v1_in - v2_in + v1_out + v2_out <= -epsilon
        network.addInequality([v1_in, v2_in, v1_out, v2_out], [-1.0, -1.0, 1.0, 1.0], -epsilon)

    start = time.time()
    exit_code, values, stats = network.solve(options=Marabou.createOptions(verbosity=0))
    wall_time = time.time() - start

    record = {
        "direction": direction,
        "result": exit_code.upper(),
        "verification_time_seconds": wall_time,
        "marabou_time_seconds": stats.getTotalTimeInMicro() / 1e6,
        "counterexample": None,
    }
    if exit_code == "sat":
        state_in = [values[v] for v in in_vars]
        state_out = [values[v] for v in out_vars]
        p_before = values[v1_in] + values[v2_in]
        p_after = values[v1_out] + values[v2_out]
        record["counterexample"] = {
            "input_state": state_in,
            "predicted_next_state": state_out,
            "momentum_before": p_before,
            "momentum_after": p_after,
            "momentum_error": p_after - p_before,
        }
    return record


def verify_momentum_property(onnx_path: str, prop: MomentumProperty) -> dict:
    """Run both one-sided queries for a MomentumProperty; return a combined result record."""
    axis_vars = (VX1, VX2) if prop.axis == "x" else (VY1, VY2)

    upper = _solve_one_sided(onnx_path, axis_vars, prop.epsilon, "upper")
    lower = _solve_one_sided(onnx_path, axis_vars, prop.epsilon, "lower")

    overall_result = "SAT" if (upper["result"] == "SAT" or lower["result"] == "SAT") else "UNSAT"
    total_time = upper["verification_time_seconds"] + lower["verification_time_seconds"]

    counterexample = upper["counterexample"] or lower["counterexample"]

    return {
        "name": prop.name,
        "description": prop.description,
        "axis": prop.axis,
        "epsilon": prop.epsilon,
        "domain_lower": DOMAIN_LOWER,
        "domain_upper": DOMAIN_UPPER,
        "separation_bound": SEPARATION_BOUND,
        "expected": prop.expect,
        "result": overall_result,
        "verification_time_seconds": total_time,
        "upper_violation_query": upper,
        "lower_violation_query": lower,
        "counterexample": counterexample,
    }


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"{ONNX_PATH} not found. Run `python -m verification.export_particle_onnx` first."
        )

    results = []
    for prop in MOMENTUM_PROPERTIES:
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
