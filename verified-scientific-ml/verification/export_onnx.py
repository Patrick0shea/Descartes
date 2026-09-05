"""
Export the trained MLP surrogate (Step 2) to ONNX for formal verification.

The exported graph fuses the input normalization and output
de-normalization directly into the network, so the ONNX model takes a
*physical* state [x, v] as input and produces a *physical* state
[x_next, v_next] as output -- exactly matching what models.evaluate.predict()
computes at inference time. This lets verification properties and the
input domain be stated directly in physical units instead of normalized
ones.

Run from the verified-scientific-ml/ directory:
    python -m verification.export_onnx
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from models.normalization import Normalizer
from models.surrogate import MLPSurrogate

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VERIFICATION_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "surrogate.onnx")


class FullSurrogate(nn.Module):
    """
    MLP surrogate with normalization fused in: takes a physical [x, v]
    state and returns a physical [x_next, v_next] state. Mathematically
    identical to:
        normalizer.inverse_transform(mlp(normalizer.transform(state)))
    (the normalization/de-normalization affine maps are implemented here
    as extra Linear layers with diagonal weight matrices).
    """

    def __init__(self, mlp: MLPSurrogate, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32)

        self.input_norm = nn.Linear(2, 2)
        with torch.no_grad():
            self.input_norm.weight.copy_(torch.diag(1.0 / std_t))
            self.input_norm.bias.copy_(-mean_t / std_t)

        self.mlp = mlp

        self.output_denorm = nn.Linear(2, 2)
        with torch.no_grad():
            self.output_denorm.weight.copy_(torch.diag(std_t))
            self.output_denorm.bias.copy_(mean_t)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.output_denorm(self.mlp(self.input_norm(state)))


def load_full_surrogate() -> FullSurrogate:
    """Load the trained Step 2 checkpoint + normalizer into a FullSurrogate."""
    state_dict = torch.load(os.path.join(CHECKPOINT_DIR, "surrogate.pt"))
    hidden_dim = state_dict["net.0.weight"].shape[0]

    mlp = MLPSurrogate(hidden_dim=hidden_dim)
    mlp.load_state_dict(state_dict)
    mlp.eval()

    normalizer = Normalizer.load(os.path.join(CHECKPOINT_DIR, "normalizer.npz"))

    full_model = FullSurrogate(mlp, normalizer.mean, normalizer.std)
    full_model.eval()
    return full_model


def check_equivalence(full_model: FullSurrogate, n_samples: int = 200, seed: int = 0) -> float:
    """
    Confirm the fused FullSurrogate matches the original
    normalize -> mlp -> denormalize inference pipeline from
    models/evaluate.py. Returns the max absolute difference found over
    n_samples random points in the verification domain.
    """
    from models.evaluate import predict

    normalizer = Normalizer.load(os.path.join(CHECKPOINT_DIR, "normalizer.npz"))
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-2.0, 2.0, size=(n_samples, 2)).astype(np.float32)

    with torch.no_grad():
        fused_out = full_model(torch.from_numpy(samples)).numpy()

    manual_out = predict(full_model.mlp, normalizer, samples)

    return float(np.max(np.abs(fused_out - manual_out)))


def export(onnx_path: str = ONNX_PATH) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    full_model = load_full_surrogate()

    max_diff = check_equivalence(full_model)
    print(f"Max abs difference vs. original inference pipeline: {max_diff:.3e}")
    assert max_diff < 1e-4, "Fused ONNX-ready model diverges from original inference pipeline"

    dummy_input = torch.zeros((1, 2), dtype=torch.float32)
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
