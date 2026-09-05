"""
Random testing baseline over the same domain D and GEOMETRIC_MOMENTUM_PROPERTIES
as verification/verify_geometric.py, for direct comparison against
Marabou's formal verification results on the geometric model. Reuses
verification/random_search_particle.py's run_random_test(), which only
needs a model callable and a property list -- identical sampling logic
to Step 4's baseline comparison.

Run from the verified-scientific-ml/ directory, after
verification/export_geometric_onnx.py:
    python -m verification.random_search_geometric
"""

from __future__ import annotations

import json
import os
import time

from verification.export_geometric_onnx import load_geometric_surrogate
from verification.particle_properties import GEOMETRIC_MOMENTUM_PROPERTIES
from verification.random_search_particle import run_random_test

SEED = 42
N_SAMPLES = 200_000

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "geometric_random_testing_results.json")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model = load_geometric_surrogate()

    start = time.time()
    results = run_random_test(model, GEOMETRIC_MOMENTUM_PROPERTIES, N_SAMPLES, SEED)
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
