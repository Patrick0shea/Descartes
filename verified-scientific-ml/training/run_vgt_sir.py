"""
Run the verification-guided training (VGT) loop on the SIR Model C
(SirSoftGeometricMLP) to show that the VGT algorithm generalises across
domains -- from the two-particle spring system (training/run_vgt.py) to
the SIR epidemiological model.

Loads the pre-trained Model C checkpoint (sir_soft_geometric.pt),
which standard training verifies UNSAT at epsilon=1e-2. The VGT loop
then attempts to push the model to UNSAT at tighter epsilons
[1e-3, 1e-4, 1e-5] by iteratively:
    1. Fine-tuning on the augmented dataset D_aug
    2. Exporting to a temporary ONNX file
    3. Running Marabou to find the worst-case population violation
    4. Adding the counterexample + neighbourhood to D_aug with
       simulator ground-truth labels
    5. Repeating until UNSAT or max iterations reached

Key difference from run_vgt.py: the simulator call uses
sir_simulator.simulate_sir (not two_particle_system.simulate), and the
conservation property is total population (s+i+r) not linear momentum.
Neighbourhood sampling clips to [0,1]^3 and renormalises so that
s+i+r=1 exactly, keeping all augmentation samples on the conservation
manifold.

The VGT-trained model is saved to checkpoints/sir_soft_geometric_vgt.pt
and the full iteration history to verification/artifacts/sir_vgt_results.json.

Run from the verified-scientific-ml/ directory:
    python -m training.run_vgt_sir
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.generate_sir_data import SEPARATION_BOUND, STATE_BOUNDS
from models.sir_soft_geometric import SirSoftGeometricMLP
from simulator.sir_simulator import BETA, GAMMA, simulate_sir
from verification.sir_properties import (
    DOMAIN_LOWER,
    DOMAIN_UPPER,
    SOFT_GEOMETRIC_POPULATION_PROPERTIES,
    S,
    I,
    R,
)
from verification.verify_sir import _solve_sir_one_sided

HIDDEN_DIM = 16
SEED = 42

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAINING_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
DATA_PATH = os.path.join(ROOT_DIR, "data", "sir_dataset.npz")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "verification", "artifacts")
TEMP_ONNX_PATH = os.path.join(ARTIFACTS_DIR, "sir_vgt_temp.onnx")
VGT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sir_soft_geometric_vgt.pt")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "sir_vgt_results.json")

# Start where standard training fails -- 1e-2 is already UNSAT, so begin at 1e-3.
EPSILON_SCHEDULE = [1e-3, 1e-4, 1e-5]

# Per-epsilon budget: 15 iterations x 50 fine-tuning epochs = 750 gradient steps
# per epsilon level. Empirically fast (seconds per Marabou query at hidden_dim=16).
MAX_ITERS_PER_EPSILON = 15
EPOCHS_PER_ITER = 50
NEIGHBOURHOOD_SIZE = 100

_DOMAIN_LOWER = np.array(DOMAIN_LOWER, dtype=np.float32)
_DOMAIN_UPPER = np.array(DOMAIN_UPPER, dtype=np.float32)

DT = 1.0  # one-day time step, matches data/generate_sir_data.py


def _simulator_step(state: np.ndarray) -> np.ndarray:
    """Call the SIR simulator for one timestep DT, return next state (shape 3,)."""
    result = simulate_sir(
        state.astype(np.float64),
        beta=BETA,
        gamma=GAMMA,
        t_span=(0.0, DT),
        n_points=2,
    )
    return result["states"][1].astype(np.float32)


def _in_domain(state: np.ndarray) -> bool:
    """Return True if state satisfies the box + population constraints of domain D."""
    if np.any(state < _DOMAIN_LOWER) or np.any(state > _DOMAIN_UPPER):
        return False
    pop = state[S] + state[I] + state[R]
    if abs(pop - 1.0) > SEPARATION_BOUND:
        return False
    return True


def _sample_neighbourhood(
    x_ce: np.ndarray,
    n: int,
    sigma: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """
    Sample up to n states near counterexample x_ce that lie within domain D.

    Adds Gaussian noise (std=sigma) to x_ce, clips all components to [0, 1],
    then renormalises so that s+i+r=1 exactly -- keeping all samples on the
    conservation manifold. Keeps samples that also satisfy the box bounds.
    Returns fewer than n samples if the neighbourhood overlaps the domain
    boundary heavily.
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = []
    max_attempts = n * 20
    for _ in range(max_attempts):
        if len(samples) >= n:
            break
        noise = rng.normal(0.0, sigma, size=3).astype(np.float32)
        candidate = np.clip(x_ce + noise, 0.0, 1.0)
        total = candidate.sum()
        if total < 1e-8:
            continue
        candidate = candidate / total  # renormalise: s+i+r = 1 exactly
        if _in_domain(candidate):
            samples.append(candidate)
    return samples


def _export_onnx(model: SirSoftGeometricMLP, onnx_path: str) -> None:
    """Export SIR model to ONNX (legacy exporter, Marabou-compatible)."""
    model.eval()
    dummy = torch.zeros((1, 3), dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["state"],
        output_names=["next_state"],
        opset_version=13,
        dynamo=False,
    )


def _verify_population(onnx_path: str, epsilon: float) -> dict:
    """
    Run both one-sided Marabou queries for population conservation at the
    given epsilon.

    Returns dict with keys: result ("UNSAT"/"SAT"), time_s, counterexample (or None).
    """
    upper = _solve_sir_one_sided(onnx_path, epsilon, "upper")
    lower = _solve_sir_one_sided(onnx_path, epsilon, "lower")
    overall = "SAT" if (upper["result"] == "SAT" or lower["result"] == "SAT") else "UNSAT"
    counterexample = upper["counterexample"] or lower["counterexample"]
    total_time = upper["verification_time_seconds"] + lower["verification_time_seconds"]
    return {"result": overall, "time_s": total_time, "counterexample": counterexample}


def run_vgt_sir() -> dict:
    """
    Run the full VGT loop for SIR Model C. Returns a history dict with
    per-iteration records and a summary of which epsilons were formally verified.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load pre-trained Model C as starting point
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "sir_soft_geometric.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `python -m models.train_sir` first."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = SirSoftGeometricMLP(hidden_dim=HIDDEN_DIM)
    model.load_state_dict(state_dict)
    print(f"Loaded pre-trained Model C from {checkpoint_path}")
    print(f"Standard training: UNSAT at epsilon=1e-2 (verified in verification step)")
    print(f"VGT will attempt: {[f'{e:.0e}' for e in EPSILON_SCHEDULE]}\n")

    # Load training data
    data = np.load(DATA_PATH)
    train_x = data["train_states_t"].astype(np.float32)
    train_y = data["train_states_tp1"].astype(np.float32)
    print(f"Training data: {train_x.shape[0]} examples (D_train)")

    aug_x = list(train_x)
    aug_y = list(train_y)
    n_original = len(aug_x)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(SEED)

    history: dict = {
        "epsilon_results": [],
        "iterations": [],
        "final_epsilon_unsat": None,
        "n_original_train": n_original,
    }

    def _finetune() -> float:
        """Fine-tune model on current D_aug for EPOCHS_PER_ITER epochs. Returns final epoch loss."""
        x_arr = np.array(aug_x, dtype=np.float32)
        y_arr = np.array(aug_y, dtype=np.float32)
        x_t = torch.from_numpy(x_arr)
        y_t = torch.from_numpy(y_arr)
        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=256,
            shuffle=True,
        )
        model.train()
        final_loss = 0.0
        for _ in range(EPOCHS_PER_ITER):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)
        model.eval()
        return final_loss

    def _add_counterexample(x_ce: np.ndarray) -> int:
        """Add x_ce and its neighbourhood to D_aug. Returns the number of points added."""
        added = 0
        # Marabou counterexamples may have s+i+r slightly off 1.0 (within
        # POPULATION_BOUND tolerance). Normalise to the simplex before calling
        # the simulator, which asserts exact conservation.
        x_ce = np.clip(x_ce, 0.0, 1.0)
        total = x_ce.sum()
        if total > 1e-8:
            x_ce = x_ce / total
        y_ce = _simulator_step(x_ce)
        aug_x.append(x_ce.astype(np.float32))
        aug_y.append(y_ce)
        added += 1
        neighbours = _sample_neighbourhood(x_ce, NEIGHBOURHOOD_SIZE, rng=rng)
        for x_nb in neighbours:
            y_nb = _simulator_step(x_nb)
            aug_x.append(x_nb)
            aug_y.append(y_nb)
            added += 1
        return added

    for epsilon in EPSILON_SCHEDULE:
        print(f"\n--- VGT (SIR): targeting epsilon={epsilon:.1e} ---")
        converged = False

        for iteration in range(MAX_ITERS_PER_EPSILON):
            t_start = time.time()

            # Step 1: fine-tune
            train_loss = _finetune()

            # Step 2: export ONNX
            _export_onnx(model, TEMP_ONNX_PATH)

            # Step 3: verify population conservation
            res = _verify_population(TEMP_ONNX_PATH, epsilon)

            n_added = 0
            overall = res["result"]

            iter_record = {
                "epsilon": epsilon,
                "iteration": iteration,
                "result": overall,
                "verify_time_s": res["time_s"],
                "train_loss": train_loss,
                "n_augmented_total": len(aug_x) - n_original,
                "n_added_this_iter": 0,
                "wall_time_s": time.time() - t_start,
            }

            print(
                f"  iter {iteration:2d}  pop={overall}"
                f"  loss={train_loss:.3e}  D_aug={len(aug_x)}"
                f"  ({res['time_s']:.1f}s verify)"
            )

            if overall == "UNSAT":
                history["final_epsilon_unsat"] = epsilon
                converged = True
                history["iterations"].append(iter_record)
                print(f"  -> UNSAT proven at epsilon={epsilon:.1e} after {iteration+1} iterations")
                break

            # Step 4: add counterexample to D_aug
            if res["counterexample"] is not None:
                x_ce = np.array(res["counterexample"]["input_state"], dtype=np.float32)
                n_added += _add_counterexample(x_ce)

            iter_record["n_added_this_iter"] = n_added
            history["iterations"].append(iter_record)

        history["epsilon_results"].append({
            "epsilon": epsilon,
            "converged": converged,
            "iterations_used": iteration + 1,
            "n_augmented_total": len(aug_x) - n_original,
        })

        if not converged:
            print(
                f"  -> Did not converge at epsilon={epsilon:.1e} "
                f"after {MAX_ITERS_PER_EPSILON} iterations. Stopping."
            )
            break

    if history["final_epsilon_unsat"] is None:
        print("\nVGT did not achieve UNSAT at any epsilon in the schedule.")
    else:
        print(f"\nVGT (SIR) complete. Tightest formally-verified epsilon: {history['final_epsilon_unsat']:.1e}")

    return model, history


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model, history = run_vgt_sir()

    # Save VGT-trained model
    torch.save(model.state_dict(), VGT_CHECKPOINT_PATH)
    print(f"\nSaved VGT-trained model to {VGT_CHECKPOINT_PATH}")

    # Save history (convert numpy types for JSON serialisation)
    def _jsonify(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    with open(RESULTS_PATH, "w") as f:
        json.dump(_jsonify(history), f, indent=2)
    print(f"Saved VGT history to {RESULTS_PATH}")

    # Summary
    print("\n=== VGT (SIR) Summary ===")
    print(f"Standard Model C: UNSAT at epsilon=1e-2")
    if history["final_epsilon_unsat"] is not None:
        eps = history["final_epsilon_unsat"]
        print(f"VGT Model C:      UNSAT at epsilon={eps:.1e}  ({int(round(1e-2 / eps))}x tighter)")
    else:
        print("VGT Model C:      Did not improve beyond epsilon=1e-2")

    print(f"\nAugmentation: {history['epsilon_results'][-1]['n_augmented_total']} counterexample states added to D_train")
    for er in history["epsilon_results"]:
        status = "UNSAT (converged)" if er["converged"] else "SAT (did not converge)"
        print(f"  epsilon={er['epsilon']:.1e}: {status} in {er['iterations_used']} iterations")


if __name__ == "__main__":
    main()
