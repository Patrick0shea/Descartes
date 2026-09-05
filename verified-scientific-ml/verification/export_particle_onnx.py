"""
Export the trained particle MLP surrogate (Step 4) to ONNX for formal
verification. Same fused-normalization approach as
verification/export_onnx.py (Step 3), generalized to the 8-dimensional
two-particle state; kept as a separate module so Step 3's code and tests
are untouched.

The exported graph fuses input normalization and output de-normalization
into the network, so the ONNX model takes a *physical* 8-dim state as
input and produces a *physical* 8-dim state as output -- exactly matching
models.evaluate_particle.predict().

Run from the verified-scientific-ml/ directory:
    python -m verification.export_particle_onnx
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from models.normalization import Normalizer
from models.particle_mlp import ParticleMLP

STATE_DIM = 8

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VERIFICATION_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "particle_surrogate.onnx")


class FullParticleSurrogate(nn.Module):
    """
    ParticleMLP with normalization fused in: takes a physical 8-dim state
    and returns a physical 8-dim state. Mathematically identical to
    normalizer.inverse_transform(mlp(normalizer.transform(state))).
    """

    def __init__(self, mlp: ParticleMLP, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32)

        self.input_norm = nn.Linear(STATE_DIM, STATE_DIM)
        with torch.no_grad():
            self.input_norm.weight.copy_(torch.diag(1.0 / std_t))
            self.input_norm.bias.copy_(-mean_t / std_t)

        self.mlp = mlp

        self.output_denorm = nn.Linear(STATE_DIM, STATE_DIM)
        with torch.no_grad():
            self.output_denorm.weight.copy_(torch.diag(std_t))
            self.output_denorm.bias.copy_(mean_t)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.output_denorm(self.mlp(self.input_norm(state)))


def load_full_particle_surrogate() -> FullParticleSurrogate:
    """Load the trained Step 4 checkpoint + normalizer into a FullParticleSurrogate."""
    state_dict = torch.load(os.path.join(CHECKPOINT_DIR, "particle_mlp.pt"))
    hidden_dim = state_dict["net.0.weight"].shape[0]

    mlp = ParticleMLP(hidden_dim=hidden_dim)
    mlp.load_state_dict(state_dict)
    mlp.eval()

    normalizer = Normalizer.load(os.path.join(CHECKPOINT_DIR, "particle_normalizer.npz"))

    full_model = FullParticleSurrogate(mlp, normalizer.mean, normalizer.std)
    full_model.eval()
    return full_model


def check_equivalence(full_model: FullParticleSurrogate, n_samples: int = 200, seed: int = 0) -> float:
    """
    Confirm the fused FullParticleSurrogate matches the original
    normalize -> mlp -> denormalize inference pipeline from
    models/evaluate_particle.py. Returns the max absolute difference
    found over n_samples random points in the verification domain.
    """
    from models.evaluate_particle import predict
    from verification.particle_properties import DOMAIN_LOWER, DOMAIN_UPPER

    normalizer = Normalizer.load(os.path.join(CHECKPOINT_DIR, "particle_normalizer.npz"))
    rng = np.random.default_rng(seed)
    samples = rng.uniform(DOMAIN_LOWER, DOMAIN_UPPER, size=(n_samples, STATE_DIM)).astype(np.float32)

    with torch.no_grad():
        fused_out = full_model(torch.from_numpy(samples)).numpy()

    manual_out = predict(full_model.mlp, normalizer, samples)

    return float(np.max(np.abs(fused_out - manual_out)))


def export(onnx_path: str = ONNX_PATH) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    full_model = load_full_particle_surrogate()

    max_diff = check_equivalence(full_model)
    print(f"Max abs difference vs. original inference pipeline: {max_diff:.3e}")
    assert max_diff < 1e-4, "Fused ONNX-ready model diverges from original inference pipeline"

    dummy_input = torch.zeros((1, STATE_DIM), dtype=torch.float32)
    torch.onnx.export(
        full_model,
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
