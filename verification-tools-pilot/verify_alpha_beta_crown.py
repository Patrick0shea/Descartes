"""Verify the toy generator with alpha-beta-CROWN's high-level Python API
(ABCrownSolver.compute_bounds), for a given grid size.

alpha-beta-CROWN is built on top of auto_LiRPA (same CROWN-style bound
propagation, same "bound all output neurons in one pass" advantage over
Marabou) plus branch-and-bound refinement for tighter/complete results.
Here we use compute_bounds() the same way verify_auto_lirpa.py uses
auto_LiRPA directly: non-negativity is one call bounding all N output
neurons, and bounded sum uses the same fixed all-ones Linear(N,1) head
trick as model.GeneratorWithSum.

Requires the alpha-beta-CROWN repo installed editable in this venv (see
REPORT.md / README.md for the exact install steps and what they needed).

Usage: python verify_alpha_beta_crown.py --grid-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

import torch

from export_onnx import load_generator
from model import GeneratorWithSum

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

Z_MIN, Z_MAX = -1.0, 1.0
TIMEOUT_SECONDS = 60
SUM_MIN, SUM_MAX = 4.0, 20.0


class Timeout(Exception):
    pass


def _with_timeout(fn, seconds):
    def handler(signum, frame):
        raise Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        start = time.time()
        result = fn()
        return result, time.time() - start, None
    except Timeout:
        return None, time.time() - start, "TIMEOUT"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _make_solver(model: torch.nn.Module, latent_dim: int, n_out: int):
    from abcrown import ABCrownSolver, ConfigBuilder, input_vars, output_vars

    x = input_vars((latent_dim,))
    y = output_vars(n_out)
    config = (
        ConfigBuilder.from_defaults()
        .set("general/device", "cpu")
        .set("attack/pgd_order", "skip")
        .set("solver/bound_prop_method", "crown")
        .set("general/complete_verifier", "skip")  # incomplete alpha-CROWN bounds only, no branch-and-bound
    )
    solver = ABCrownSolver(model, x, y, config=config)
    return solver, x, y


def check_nonnegativity(grid_size: int) -> dict:
    from abcrown import IOConstraints

    model = load_generator(grid_size)
    n_out = grid_size * grid_size
    solver, x, y = _make_solver(model, model.latent_dim, n_out)

    lower = [Z_MIN] * model.latent_dim
    upper = [Z_MAX] * model.latent_dim
    objective = [y[i] for i in range(n_out)]

    def run():
        return solver.compute_bounds(
            constraints=IOConstraints(input_vars=x, input_constraint=(x >= lower) & (x <= upper)),
            objective=objective,
        )

    res, wall_time, err = _with_timeout(run, TIMEOUT_SECONDS)
    if err:
        return {"property": "nonnegativity", "result": err, "verification_time_seconds": wall_time}

    lb = res.lower.flatten().tolist()
    min_lb = min(lb)
    result = "UNSAT" if min_lb >= 0 else "SAT"
    return {
        "property": "nonnegativity",
        "result": result,
        "verification_time_seconds": wall_time,
        "n_cells": len(lb),
        "min_lower_bound": min_lb,
        "n_cells_with_negative_lower_bound": sum(1 for v in lb if v < 0),
        "success_flag": bool(res.success),
        "method": "ABCrownSolver.compute_bounds, all output neurons in one call",
    }


def check_bounded_sum(grid_size: int) -> dict:
    from abcrown import IOConstraints

    model = load_generator(grid_size)
    sum_model = GeneratorWithSum(model)
    sum_model.eval()
    solver, x, y = _make_solver(sum_model, model.latent_dim, 1)

    lower = [Z_MIN] * model.latent_dim
    upper = [Z_MAX] * model.latent_dim

    def run():
        return solver.compute_bounds(
            constraints=IOConstraints(input_vars=x, input_constraint=(x >= lower) & (x <= upper)),
            objective=[y[0]],
        )

    res, wall_time, err = _with_timeout(run, TIMEOUT_SECONDS)
    if err:
        return {"property": "bounded_sum", "result": err, "verification_time_seconds": wall_time}

    lb_val = res.lower.flatten().tolist()[0]
    ub_val = res.upper.flatten().tolist()[0]
    holds = (lb_val >= SUM_MIN) and (ub_val <= SUM_MAX)
    return {
        "property": "bounded_sum",
        "sum_bounds": [SUM_MIN, SUM_MAX],
        "result": "UNSAT" if holds else "SAT",
        "verification_time_seconds": wall_time,
        "computed_lower_bound": lb_val,
        "computed_upper_bound": ub_val,
        "method": "fixed all-ones Linear(N,1) head, single compute_bounds call",
    }


def run(grid_size: int) -> dict:
    print(f"[abcrown] grid={grid_size}x{grid_size}: non-negativity")
    nonneg = check_nonnegativity(grid_size)
    print(f"    {nonneg['result']} ({nonneg['verification_time_seconds']:.4f}s)")

    print(f"[abcrown] grid={grid_size}x{grid_size}: bounded sum")
    bsum = check_bounded_sum(grid_size)
    print(f"    {bsum['result']} ({bsum['verification_time_seconds']:.4f}s)")

    return {"tool": "alpha_beta_crown", "grid_size": grid_size, "nonnegativity": nonneg, "bounded_sum": bsum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()

    result = run(args.grid_size)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"alpha_beta_crown_{args.grid_size}x{args.grid_size}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
