"""
Random testing baseline over the same input domain and properties as
verification/verify.py, for direct comparison against Marabou's formal
verification results.

Unlike Marabou, random sampling can only ever report "no violation found
among the sampled points" -- it can NEVER prove a property holds
everywhere in the (continuous, infinite) domain, no matter how many
points are sampled. That is the central distinction this project is
demonstrating: testing vs. verification.

Run from the verified-scientific-ml/ directory, after
verification/export_onnx.py:
    python -m verification.random_search
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from verification.export_onnx import FullSurrogate, load_full_surrogate
from verification.properties import PROPERTIES, V_MAX, V_MIN, X_MAX, X_MIN, Property

SEED = 42
N_SAMPLES = 200_000

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "random_testing_results.json")


def run_random_test(
    model: FullSurrogate,
    properties: list[Property],
    n_samples: int,
    seed: int,
) -> list[dict]:
    """Sample n_samples random (x, v) points in the domain and check each property."""
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, 2), dtype=np.float32)
    samples[:, 0] = rng.uniform(X_MIN, X_MAX, size=n_samples)
    samples[:, 1] = rng.uniform(V_MIN, V_MAX, size=n_samples)

    with torch.no_grad():
        outputs = model(torch.from_numpy(samples)).numpy()

    results = []
    for prop in properties:
        out_col = outputs[:, prop.output_index]
        if prop.bound_type == "ub":
            violation_mask = out_col >= prop.bound_value
        else:
            violation_mask = out_col <= prop.bound_value

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
                "x": float(samples[idx, 0]),
                "v": float(samples[idx, 1]),
                "x_next": float(outputs[idx, 0]),
                "v_next": float(outputs[idx, 1]),
            }
        results.append(record)

    return results


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model = load_full_surrogate()

    start = time.time()
    results = run_random_test(model, PROPERTIES, N_SAMPLES, SEED)
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
