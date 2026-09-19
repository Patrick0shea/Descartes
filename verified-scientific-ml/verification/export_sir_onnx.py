"""
Export all three trained SIR surrogate models (A, B, C) to ONNX for
formal verification with Marabou.

Like the particle-system exports (export_geometric_onnx.py,
export_soft_geometric_onnx.py), there is no normalisation to fuse: all
three SIR models operate directly on [s, i, r] fractions in [0, 1], so
the trained model IS the ONNX-ready graph as-is. All forward passes use
only nn.Linear and nn.ReLU, which trace to Gemm/MatMul/Add/Relu in
ONNX -- the only operations Marabou's parser supports.

Run from the verified-scientific-ml/ directory:
    python -m verification.export_sir_onnx
"""

from __future__ import annotations

import os

import torch

from models.sir_mlp import SirMLP
from models.sir_geometric import SirGeometricMLP
from models.sir_soft_geometric import SirSoftGeometricMLP

STATE_DIM = 3

VERIFICATION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VERIFICATION_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
ARTIFACTS_DIR = os.path.join(VERIFICATION_DIR, "artifacts")


def export_model_a() -> None:
    """Load sir_mlp.pt, export sir_mlp.onnx."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "sir_mlp.pt")
    onnx_path = os.path.join(ARTIFACTS_DIR, "sir_mlp.onnx")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `python -m models.train_sir` first."
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = state_dict["net.0.weight"].shape[0]
    model = SirMLP(hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

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
    print(f"Exported Model A (SirMLP) to {onnx_path}")


def export_model_b() -> None:
    """Load sir_geometric.pt, export sir_geometric.onnx."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "sir_geometric.pt")
    onnx_path = os.path.join(ARTIFACTS_DIR, "sir_geometric.onnx")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `python -m models.train_sir` first."
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = state_dict["delta_net.0.weight"].shape[0]
    model = SirGeometricMLP(hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

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
    print(f"Exported Model B (SirGeometricMLP) to {onnx_path}")


def export_model_c() -> None:
    """Load sir_soft_geometric.pt, export sir_soft_geometric.onnx."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "sir_soft_geometric.pt")
    onnx_path = os.path.join(ARTIFACTS_DIR, "sir_soft_geometric.onnx")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `python -m models.train_sir` first."
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = state_dict["delta_net.0.weight"].shape[0]
    model = SirSoftGeometricMLP(hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

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
    print(f"Exported Model C (SirSoftGeometricMLP) to {onnx_path}")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    export_model_a()
    export_model_b()
    export_model_c()


if __name__ == "__main__":
    main()
