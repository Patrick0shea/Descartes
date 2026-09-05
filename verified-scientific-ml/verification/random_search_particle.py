"""
Random testing baseline over the same input domain and momentum
properties as verification/verify_particle.py, for direct comparison
against Marabou's formal verification results.

Exactly like verification/random_search.py (Step 3), this can only ever
report "no violation found among the sampled points" -- never a proof
that a property holds everywhere in the continuous domain D.

Run from the verified-scientific-ml/ directory, after
verification/export_particle_onnx.py:
    python -m verification.random_search_particle
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from verification.export_particle_onnx import FullParticleSurrogate, load_full_particle_surrogate
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

SEED = 42
N_SAMPLES = 200_000
STATE_DIM = 8

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "particle_random_testing_results.json")


def run_random_test(
    model: FullParticleSurrogate,
    properties: list[MomentumProperty],
    n_samples: int,
    seed: int,
) -> list[dict]:
    """
    Sample n_samples random states in domain D = box ∩ separation
    constraints (see particle_properties.py) and check each momentum
    property. Uses rejection sampling on the box to match Marabou's exact
    domain, so the comparison is over the identical region.
    """
    rng = np.random.default_rng(seed)
    lower = np.array(DOMAIN_LOWER)
    upper = np.array(DOMAIN_UPPER)

    accepted = []
    while sum(len(a) for a in accepted) < n_samples:
        batch = rng.uniform(lower, upper, size=(n_samples, STATE_DIM))
        dx = np.abs(batch[:, X2] - batch[:, X1])
        dy = np.abs(batch[:, Y2] - batch[:, Y1])
        mask = (dx <= SEPARATION_BOUND) & (dy <= SEPARATION_BOUND)
        accepted.append(batch[mask])
    samples = np.concatenate(accepted, axis=0)[:n_samples].astype(np.float32)

    with torch.no_grad():
        outputs = model(torch.from_numpy(samples)).numpy()

    results = []
    for prop in properties:
        i, j = (VX1, VX2) if prop.axis == "x" else (VY1, VY2)
        p_before = samples[:, i] + samples[:, j]
        p_after = outputs[:, i] + outputs[:, j]
        error = p_after - p_before
        violation_mask = np.abs(error) > prop.epsilon

        n_violations = int(violation_mask.sum())
        record = {
            "name": prop.name,
            "description": prop.description,
            "n_samples": n_samples,
            "n_violations_found": n_violations,
            "result": "VIOLATION_FOUND" if n_violations > 0 else "NO_VIOLATION_FOUND",
        }
        if n_violations > 0:
            idx = int(np.argmax(violation_mask))
            record["example_violation"] = {
                "input_state": samples[idx].tolist(),
                "momentum_before": float(p_before[idx]),
                "momentum_after": float(p_after[idx]),
                "momentum_error": float(error[idx]),
            }
        results.append(record)

    return results


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model = load_full_particle_surrogate()

    start = time.time()
    results = run_random_test(model, MOMENTUM_PROPERTIES, N_SAMPLES, SEED)
    elapsed = time.time() - start

    for record in results:
        print(
            f"{record['name']}: {record['result']} "
            f"({record['n_violations_found']}/{record['n_samples']} samples violated)"
        )

    print(f"\nEvaluated {N_SAMPLES} random samples in {elapsed:.2f}s")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved random-testing results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
