"""Verify the toy generator with Marabou (complete SMT-based search over
an ONNX network), for a given grid size.

Marabou has no first-class "bound every output neuron in one shot" call
the way CROWN-style bound propagation does. Two consequences handled
here:

- Non-negativity ("every output cell >= 0") is a per-neuron property, so
  it needs one Marabou query per cell checked. A first attempt at
  exhaustive 8x8 checking (64 queries) showed individual queries taking
  ~3-47s each (average ~40s) even with Split-and-Conquer search enabled
  (see MARABOU_OPTIONS below) -- exhaustive enumeration does not fit this
  pilot's time budget at any grid size, not just 32x32. So every grid
  size here checks a fixed-size representative sample (corners, edges,
  center, plus random cells) instead of every cell -- this is reported
  explicitly as a sample, not an exhaustive per-cell proof (see
  SAMPLE_SIZE_8X8/SAMPLE_SIZE_LARGE below for the exact sizes and why
  they differ).
- Bounded sum ("conserved total") IS expressible directly, via
  network.addInequality() over all output variables with unit
  coefficients (see verified-scientific-ml/verification/verify_particle.py
  for the same addInequality pattern used on a 2-variable linear
  combination). This is a genuine linear equation over the network's
  variables, not an approximation.

Each query gets a hard timeout via Marabou's own timeoutInSeconds option,
so a hard instance can only ever TIMEOUT, never hang the pilot.

Usage: python verify_marabou.py --grid-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from maraboupy import Marabou

ONNX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

Z_MIN, Z_MAX = -1.0, 1.0
TIMEOUT_SECONDS = 30
SUM_MIN, SUM_MAX = 4.0, 20.0  # generous band around the training target (8-12)

# Plain single-threaded search on this 3-hidden-layer, width~128 ReLU net
# was observed to time out even at 60s per query (see REPORT.md). Split-
# and-Conquer mode with multiple workers resolves the same queries in a
# few seconds, so it is used as the default solver configuration here --
# itself a pilot finding about what it takes to make Marabou practical
# at this network size.
MARABOU_OPTIONS = dict(snc=True, numWorkers=4, initialSplits=2, verbosity=0)

# Exhaustive per-cell checking was tried first at 8x8 (64 cells): with
# Split-and-Conquer search (see MARABOU_OPTIONS) individual queries still
# took anywhere from ~3s to a full timeout, averaging ~40s/cell -- so
# exhaustive enumeration, even at the smallest grid, does not fit this
# pilot's time budget (it would take tens of minutes per grid size). All
# grid sizes therefore use a fixed-size representative sample (corners,
# edges, center, plus random cells) instead. The 8x8 sample was run at
# 24 cells (~9 minutes wall clock); at 16x16/32x32 the sample is reduced
# to 10 cells to keep the pilot's total run time reasonable, since the
# 8x8 run already established the qualitative timing pattern (a mix of
# few-second SATs and 30s+ timeouts) that a larger sample would mostly
# just repeat.
FULL_ENUMERATION_MAX_CELLS = 0
SAMPLE_SIZE_8X8 = 24
SAMPLE_SIZE_LARGE = 10


def _cells_to_check(grid_size: int) -> tuple[list[int], bool]:
    n = grid_size * grid_size
    sample_size = SAMPLE_SIZE_8X8 if grid_size == 8 else SAMPLE_SIZE_LARGE
    if n <= FULL_ENUMERATION_MAX_CELLS:
        return list(range(n)), True

    mid = grid_size // 2
    # priority order: the four corners and the center cell always make it
    # into the sample even when sample_size is small; edge cells and
    # random cells fill in the rest, in that order, up to sample_size.
    priority = [0, grid_size - 1, n - grid_size, n - 1, mid * grid_size + mid]
    edges = [c for c in range(grid_size)] + [(grid_size - 1) * grid_size + c for c in range(grid_size)]
    edges += [r * grid_size for r in range(grid_size)] + [r * grid_size + grid_size - 1 for r in range(grid_size)]
    rng = random.Random(0)
    randoms = list(range(n))
    rng.shuffle(randoms)

    idx = []
    seen = set()
    for c in priority + edges + randoms:
        if c not in seen:
            seen.add(c)
            idx.append(c)
        if len(idx) == sample_size:
            break
    return sorted(idx), False


def _new_network(onnx_path: str):
    network = Marabou.read_onnx(onnx_path)
    in_vars = network.inputVars[0].flatten()
    for v in in_vars:
        network.setLowerBound(v, Z_MIN)
        network.setUpperBound(v, Z_MAX)
    return network, in_vars


def check_nonnegativity_cell(onnx_path: str, cell: int) -> dict:
    network, in_vars = _new_network(onnx_path)
    out_vars = network.outputVars[0].flatten()
    out_var = out_vars[cell]
    # claim: out_var >= 0  -->  negate: out_var <= -epsilon, i.e. search out_var < 0
    network.setUpperBound(out_var, -1e-6)

    start = time.time()
    exit_code, values, stats = network.solve(
        options=Marabou.createOptions(timeoutInSeconds=TIMEOUT_SECONDS, **MARABOU_OPTIONS)
    )
    wall_time = time.time() - start

    result = "TIMEOUT" if exit_code == "TIMEOUT" else exit_code.upper()
    record = {
        "cell": cell,
        "result": result,
        "verification_time_seconds": wall_time,
        "counterexample_value": values[out_var] if exit_code == "sat" else None,
    }
    return record


def check_bounded_sum(onnx_path: str, grid_size: int) -> dict:
    n = grid_size * grid_size

    def one_sided(direction: str) -> dict:
        network, in_vars = _new_network(onnx_path)
        out_vars = list(network.outputVars[0].flatten())
        if direction == "above_max":
            # claim: sum <= SUM_MAX --> negate: sum >= SUM_MAX --> -sum <= -SUM_MAX
            network.addInequality(out_vars, [-1.0] * n, -SUM_MAX)
        else:
            # claim: sum >= SUM_MIN --> negate: sum <= SUM_MIN
            network.addInequality(out_vars, [1.0] * n, SUM_MIN)

        start = time.time()
        exit_code, values, stats = network.solve(
            options=Marabou.createOptions(timeoutInSeconds=TIMEOUT_SECONDS, **MARABOU_OPTIONS)
        )
        wall_time = time.time() - start
        result = "TIMEOUT" if exit_code == "TIMEOUT" else exit_code.upper()
        record = {
            "direction": direction,
            "result": result,
            "verification_time_seconds": wall_time,
        }
        if exit_code == "sat":
            record["counterexample_sum"] = sum(values[v] for v in out_vars)
        return record

    above = one_sided("above_max")
    below = one_sided("below_min")
    results = [above, below]
    if any(r["result"] == "TIMEOUT" for r in results):
        overall = "TIMEOUT"
    elif any(r["result"] == "SAT" for r in results):
        overall = "SAT"
    else:
        overall = "UNSAT"
    return {
        "property": "bounded_sum",
        "sum_bounds": [SUM_MIN, SUM_MAX],
        "result": overall,
        "verification_time_seconds": above["verification_time_seconds"] + below["verification_time_seconds"],
        "above_max_query": above,
        "below_min_query": below,
        "api_used": "network.addInequality over all output vars (unit coefficients)",
    }


def run(grid_size: int) -> dict:
    onnx_path = os.path.join(ONNX_DIR, f"generator_{grid_size}x{grid_size}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"{onnx_path} not found. Run export_onnx.py --grid-size {grid_size} first.")

    cells, exhaustive = _cells_to_check(grid_size)
    print(f"[marabou] grid={grid_size}x{grid_size}: non-negativity over {len(cells)} cell(s) "
          f"({'exhaustive' if exhaustive else 'sample'})")

    nonneg_results = []
    t0 = time.time()
    for c in cells:
        rec = check_nonnegativity_cell(onnx_path, c)
        nonneg_results.append(rec)
        print(f"    cell {c:4d}: {rec['result']:8s} ({rec['verification_time_seconds']:.3f}s)")
    nonneg_total_time = time.time() - t0

    if any(r["result"] == "TIMEOUT" for r in nonneg_results):
        nonneg_overall = "TIMEOUT"
    elif any(r["result"] == "SAT" for r in nonneg_results):
        nonneg_overall = "SAT"
    else:
        nonneg_overall = "UNSAT"

    print(f"[marabou] grid={grid_size}x{grid_size}: bounded sum")
    sum_result = check_bounded_sum(onnx_path, grid_size)
    print(f"    sum: {sum_result['result']} ({sum_result['verification_time_seconds']:.3f}s)")

    return {
        "tool": "marabou",
        "grid_size": grid_size,
        "n_cells": grid_size * grid_size,
        "nonnegativity": {
            "cells_checked": cells,
            "exhaustive": exhaustive,
            "overall_result": nonneg_overall,
            "total_verification_time_seconds": nonneg_total_time,
            "per_cell": nonneg_results,
        },
        "bounded_sum": sum_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()

    result = run(args.grid_size)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"marabou_{args.grid_size}x{args.grid_size}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
