"""Verify the toy generator with auto_LiRPA (incomplete bound propagation,
CROWN method), for a given grid size.

Unlike Marabou, auto_LiRPA computes bounds for ALL output neurons in a
single backward pass: non-negativity for every cell in the grid is one
call to compute_bounds(), not one query per cell. Bounded sum uses the
standard trick of appending a fixed, non-trainable all-ones
Linear(N, 1) layer (see model.GeneratorWithSum) so the quantity of
interest becomes a single output neuron with its own CROWN bound.

Bound propagation itself is expected to be near-instant even at 32x32;
a wall-clock timeout is still applied via a SIGALRM-based guard so a
pathological case cannot hang the pilot.

Usage: python verify_auto_lirpa.py --grid-size 8
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


def check_nonnegativity(grid_size: int) -> dict:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    model = load_generator(grid_size)
    dummy = torch.zeros(1, model.latent_dim)
    bounded_model = BoundedModule(model, dummy)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=torch.full((1, model.latent_dim), Z_MIN),
                              x_U=torch.full((1, model.latent_dim), Z_MAX))
    bounded_input = BoundedTensor(dummy, ptb)

    def run():
        return bounded_model.compute_bounds(x=(bounded_input,), method="CROWN")

    (lb, ub), wall_time, err = _with_timeout(run, TIMEOUT_SECONDS)
    if err:
        return {"property": "nonnegativity", "result": err, "verification_time_seconds": wall_time}

    lb_flat = lb.flatten().tolist()
    min_lb = min(lb_flat)
    n_violated = sum(1 for v in lb_flat if v < 0)
    result = "UNSAT" if min_lb >= 0 else "SAT"  # UNSAT = no counterexample = property holds
    return {
        "property": "nonnegativity",
        "result": result,
        "verification_time_seconds": wall_time,
        "n_cells": len(lb_flat),
        "min_lower_bound": min_lb,
        "n_cells_with_negative_lower_bound": n_violated,
        "method": "CROWN, all output neurons bounded in one compute_bounds() call",
    }


def check_bounded_sum(grid_size: int) -> dict:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    model = load_generator(grid_size)
    sum_model = GeneratorWithSum(model)
    sum_model.eval()
    dummy = torch.zeros(1, model.latent_dim)
    bounded_model = BoundedModule(sum_model, dummy)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=torch.full((1, model.latent_dim), Z_MIN),
                              x_U=torch.full((1, model.latent_dim), Z_MAX))
    bounded_input = BoundedTensor(dummy, ptb)

    def run():
        return bounded_model.compute_bounds(x=(bounded_input,), method="CROWN")

    (lb, ub), wall_time, err = _with_timeout(run, TIMEOUT_SECONDS)
    if err:
        return {"property": "bounded_sum", "result": err, "verification_time_seconds": wall_time}

    lb_val = lb.item()
    ub_val = ub.item()
    holds = (lb_val >= SUM_MIN) and (ub_val <= SUM_MAX)
    return {
        "property": "bounded_sum",
        "sum_bounds": [SUM_MIN, SUM_MAX],
        "result": "UNSAT" if holds else "SAT",
        "verification_time_seconds": wall_time,
        "computed_lower_bound": lb_val,
        "computed_upper_bound": ub_val,
        "method": "fixed all-ones Linear(N,1) head appended to model, single CROWN bound",
    }


def run(grid_size: int) -> dict:
    print(f"[auto_lirpa] grid={grid_size}x{grid_size}: non-negativity")
    nonneg = check_nonnegativity(grid_size)
    print(f"    {nonneg['result']} ({nonneg['verification_time_seconds']:.4f}s)")

    print(f"[auto_lirpa] grid={grid_size}x{grid_size}: bounded sum")
    bsum = check_bounded_sum(grid_size)
    print(f"    {bsum['result']} ({bsum['verification_time_seconds']:.4f}s)")

    return {"tool": "auto_lirpa", "grid_size": grid_size, "nonnegativity": nonneg, "bounded_sum": bsum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()

    result = run(args.grid_size)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"auto_lirpa_{args.grid_size}x{args.grid_size}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
