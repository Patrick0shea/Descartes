"""
Formal verification of population conservation in the SIR surrogate
models (A, B, C), using Marabou.

For each PopulationProperty in verification/sir_properties.py, this
searches -- over the full continuous domain D (see sir_properties.py)
-- for a counterexample to the claim |P_next - P| <= epsilon, where
P = s+i+r is the total population fraction. The claim is checked as TWO
separate one-sided linear queries (see module docstring in
sir_properties.py):

    upper violation: P_next - P >= epsilon
    lower violation: P - P_next >= epsilon

Both are linear in the network's input/output variables (P and P_next
are each a sum of three compartment fractions), so each is encoded
directly as a single Marabou linear inequality over those six variables
-- no approximation of the population expression is needed.

If BOTH one-sided queries are UNSAT, no state in D violates the
tolerance in either direction: the property is formally verified for D.
If EITHER is SAT, that query's counterexample is reported and the
overall property result is SAT.

Run from the verified-scientific-ml/ directory, after
verification/export_sir_onnx.py:
    python -m verification.verify_sir
"""

from __future__ import annotations

import json
import os
import time

from maraboupy import Marabou

from verification.sir_properties import (
    DOMAIN_LOWER,
    DOMAIN_UPPER,
    GEOMETRIC_POPULATION_PROPERTIES,
    MLP_POPULATION_PROPERTIES,
    SEPARATION_BOUND,
    SOFT_GEOMETRIC_POPULATION_PROPERTIES,
    S,
    I,
    R,
    PopulationProperty,
)

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "sir_verification_results.json")

ONNX_MLP = os.path.join(ARTIFACTS_DIR, "sir_mlp.onnx")
ONNX_GEOMETRIC = os.path.join(ARTIFACTS_DIR, "sir_geometric.onnx")
ONNX_SOFT_GEOMETRIC = os.path.join(ARTIFACTS_DIR, "sir_soft_geometric.onnx")


def _solve_sir_one_sided(onnx_path: str, epsilon: float, direction: str) -> dict:
    """
    Solve one of the two one-sided violation queries for population
    conservation.

    in_vars[0,1,2]  = [s, i, r] input
    out_vars[0,1,2] = [s_next, i_next, r_next] output

    direction = "upper": search for P_next - P >= epsilon
        i.e. add inequality: -out_s-out_i-out_r+in_s+in_i+in_r <= -epsilon
    direction = "lower": search for P - P_next >= epsilon
        i.e. add inequality: -in_s-in_i-in_r+out_s+out_i+out_r <= -epsilon

    Domain constraints are always added:
      - box bounds (DOMAIN_LOWER, DOMAIN_UPPER) on all three input vars
      - population: in_s+in_i+in_r <= 1 + SEPARATION_BOUND
      - population: -(in_s+in_i+in_r) <= -(1 - SEPARATION_BOUND)
    """
    network = Marabou.read_onnx(onnx_path)
    in_vars = network.inputVars[0].flatten()
    out_vars = network.outputVars[0].flatten()

    # Box bounds on input state
    for idx, (lo, hi) in enumerate(zip(DOMAIN_LOWER, DOMAIN_UPPER)):
        network.setLowerBound(in_vars[idx], lo)
        network.setUpperBound(in_vars[idx], hi)

    # Population constraint: |s+i+r - 1| <= SEPARATION_BOUND
    #   upper: s+i+r <= 1+SEPARATION_BOUND
    network.addInequality(
        [in_vars[S], in_vars[I], in_vars[R]],
        [1.0, 1.0, 1.0],
        1.0 + SEPARATION_BOUND,
    )
    #   lower: -(s+i+r) <= -(1-SEPARATION_BOUND)
    network.addInequality(
        [in_vars[S], in_vars[I], in_vars[R]],
        [-1.0, -1.0, -1.0],
        -(1.0 - SEPARATION_BOUND),
    )

    # Violation query
    if direction == "upper":
        # P_next - P >= epsilon  <=>  -out_s-out_i-out_r+in_s+in_i+in_r <= -epsilon
        network.addInequality(
            [out_vars[S], out_vars[I], out_vars[R], in_vars[S], in_vars[I], in_vars[R]],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            -epsilon,
        )
    else:
        # P - P_next >= epsilon  <=>  -in_s-in_i-in_r+out_s+out_i+out_r <= -epsilon
        network.addInequality(
            [in_vars[S], in_vars[I], in_vars[R], out_vars[S], out_vars[I], out_vars[R]],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            -epsilon,
        )

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
        p_before = values[in_vars[S]] + values[in_vars[I]] + values[in_vars[R]]
        p_after = values[out_vars[S]] + values[out_vars[I]] + values[out_vars[R]]
        record["counterexample"] = {
            "input_state": state_in,
            "predicted_next_state": state_out,
            "population_before": p_before,
            "population_after": p_after,
            "population_error": p_after - p_before,
        }
    return record


def verify_sir_property(onnx_path: str, prop: PopulationProperty) -> dict:
    """Run both one-sided queries for a PopulationProperty; return a combined result record."""
    upper = _solve_sir_one_sided(onnx_path, prop.epsilon, "upper")
    lower = _solve_sir_one_sided(onnx_path, prop.epsilon, "lower")

    overall_result = "SAT" if (upper["result"] == "SAT" or lower["result"] == "SAT") else "UNSAT"
    total_time = upper["verification_time_seconds"] + lower["verification_time_seconds"]

    counterexample = upper["counterexample"] or lower["counterexample"]

    return {
        "name": prop.name,
        "description": prop.description,
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


def _run_model_properties(
    onnx_path: str,
    model_label: str,
    properties: list[PopulationProperty],
) -> list[dict]:
    """Verify all properties for one model, print progress, return result list."""
    results = []
    if not os.path.exists(onnx_path):
        print(f"  ONNX not found at {onnx_path} -- skipping {model_label}")
        return results

    print(f"\n=== {model_label} ===")
    for prop in properties:
        print(f"Verifying: {prop.description} ...")
        result = verify_sir_property(onnx_path, prop)
        match = "MATCHES" if result["result"] == prop.expect else "MISMATCH"
        print(
            f"  -> {result['result']}  ({result['verification_time_seconds']:.3f}s)"
            f"  [expected {prop.expect}, {match}]"
        )
        if result["counterexample"] is not None:
            ce = result["counterexample"]
            print(
                f"     counterexample state={['%.4f' % v for v in ce['input_state']]}"
                f" -> pop_before={ce['population_before']:.6f},"
                f" pop_after={ce['population_after']:.6f},"
                f" error={ce['population_error']:.6f}"
            )
        else:
            print(f"     no state in D violates the tolerance in either direction.")
        results.append(result)
    return results


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    all_results = {
        "model_a": _run_model_properties(ONNX_MLP, "Model A (SirMLP)", MLP_POPULATION_PROPERTIES),
        "model_b": _run_model_properties(
            ONNX_GEOMETRIC, "Model B (SirGeometricMLP)", GEOMETRIC_POPULATION_PROPERTIES
        ),
        "model_c": _run_model_properties(
            ONNX_SOFT_GEOMETRIC,
            "Model C (SirSoftGeometricMLP)",
            SOFT_GEOMETRIC_POPULATION_PROPERTIES,
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved verification results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
