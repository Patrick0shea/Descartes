"""
Run the verification-guided training (VGT) loop on Model C (soft geometric).

Loads the pre-trained Model C checkpoint (particle_soft_geometric.pt),
which standard training verifies UNSAT at epsilon=1e-2. The VGT loop then
attempts to push the model to UNSAT at tighter epsilons [1e-3, 1e-4, 1e-5]
by iteratively:
    1. Fine-tuning on the augmented dataset D_aug
    2. Running Marabou to find the worst-case momentum violation
    3. Adding the counterexample + neighbourhood to D_aug with simulator labels
    4. Repeating until UNSAT or max iterations reached

The VGT-trained model is saved to checkpoints/particle_soft_geometric_vgt.pt
and the full iteration history to verification/artifacts/vgt_results.json.

Run from the verified-scientific-ml/ directory:
    python -m training.run_vgt
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from models.particle_soft_geometric import ParticleSoftGeometricMLP
from training.vgt_loop import VGTTrainer

HIDDEN_DIM = 16
SEED = 42

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
DATA_PATH = os.path.join(ROOT_DIR, "data", "particle_dataset.npz")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "verification", "artifacts")
TEMP_ONNX_PATH = os.path.join(ARTIFACTS_DIR, "vgt_temp.onnx")
VGT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "particle_soft_geometric_vgt.pt")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "vgt_results.json")

# Start where standard training fails — 1e-2 is already UNSAT, so begin at 1e-3.
EPSILON_SCHEDULE = [1e-3, 1e-4, 1e-5]

# Per-epsilon budget: 15 iterations × 50 fine-tuning epochs = 750 gradient steps
# per epsilon level. Empirically fast (seconds per Marabou query at hidden_dim=16).
MAX_ITERS_PER_EPSILON = 15
EPOCHS_PER_ITER = 50
NEIGHBOURHOOD_SIZE = 100


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load pre-trained Model C as starting point
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "particle_soft_geometric.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `python -m models.train_soft_geometric` first."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = ParticleSoftGeometricMLP(hidden_dim=HIDDEN_DIM)
    model.load_state_dict(state_dict)
    print(f"Loaded pre-trained Model C from {checkpoint_path}")
    print(f"Standard training: UNSAT at epsilon=1e-2 (verified in Step 6)")
    print(f"VGT will attempt: {[f'{e:.0e}' for e in EPSILON_SCHEDULE]}\n")

    # Load training data
    data = np.load(DATA_PATH)
    train_x = data["train_states_t"].astype(np.float32)
    train_y = data["train_states_tp1"].astype(np.float32)
    print(f"Training data: {train_x.shape[0]} examples (D_train)")

    trainer = VGTTrainer(
        model=model,
        train_x=train_x,
        train_y=train_y,
        epsilon_schedule=EPSILON_SCHEDULE,
        temp_onnx_path=TEMP_ONNX_PATH,
        epochs_per_iter=EPOCHS_PER_ITER,
        max_iters_per_epsilon=MAX_ITERS_PER_EPSILON,
        neighbourhood_size=NEIGHBOURHOOD_SIZE,
        lr=1e-3,
        batch_size=256,
        seed=SEED,
    )

    history = trainer.run()

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
    print("\n=== VGT Summary ===")
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
