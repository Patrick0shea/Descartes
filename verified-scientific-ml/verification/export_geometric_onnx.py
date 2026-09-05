"""
Export the trained geometric surrogate (Step 5) to ONNX for formal
verification.

Unlike verification/export_particle_onnx.py (the plain MLP baseline),
there is no normalization to fuse in here: ParticleGeometricMLP already
operates directly in physical units end-to-end (see
models/particle_geometric.py), so the trained model IS the ONNX-ready
graph as-is.

Run from the verified-scientific-ml/ directory:
    python -m verification.export_geometric_onnx
"""

from __future__ import annotations

import os

import numpy as np
import torch

from models.particle_geometric import ParticleGeometricMLP

STATE_DIM = 8

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VERIFICATION_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "particle_geometric.onnx")


def load_geometric_surrogate() -> ParticleGeometricMLP:
    state_dict = torch.load(os.path.join(CHECKPOINT_DIR, "particle_geometric.pt"))
    hidden_dim = state_dict["force_net.0.weight"].shape[0]
    model = ParticleGeometricMLP(hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def check_equivalence(model: ParticleGeometricMLP, n_samples: int = 200, seed: int = 0) -> float:
    """
    Confirm the model's direct forward pass matches
    models.evaluate_geometric.predict() (they should be identical, since
    there is no fused normalization here -- this is a lightweight
    consistency check rather than a meaningful transformation test).
    """
    from models.evaluate_geometric import predict
    from verification.particle_properties import DOMAIN_LOWER, DOMAIN_UPPER

    rng = np.random.default_rng(seed)
    samples = rng.uniform(DOMAIN_LOWER, DOMAIN_UPPER, size=(n_samples, STATE_DIM)).astype(np.float32)

    with torch.no_grad():
        direct_out = model(torch.from_numpy(samples)).numpy()

    manual_out = predict(model, samples)
    return float(np.max(np.abs(direct_out - manual_out)))


def export(onnx_path: str = ONNX_PATH) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model = load_geometric_surrogate()

    max_diff = check_equivalence(model)
    print(f"Max abs difference vs. original inference pipeline: {max_diff:.3e}")
    assert max_diff < 1e-4, "ONNX-ready model diverges from original inference pipeline"

    dummy_input = torch.zeros((1, STATE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["state"],
        output_names=["next_state"],
        opset_version=13,
        dynamo=False,
    )
    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    export()
