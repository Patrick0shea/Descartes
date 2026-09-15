"""Export a trained generator checkpoint to ONNX for Marabou.

Uses the legacy (dynamo=False) torch.onnx exporter, since the default
dynamo exporter requires the onnxscript package which we don't otherwise
need here.

Usage: python export_onnx.py --grid-size 8
"""

from __future__ import annotations

import argparse
import os

import torch

from model import Generator, GeneratorWithSum

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ONNX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_models")


def load_generator(grid_size: int) -> Generator:
    path = os.path.join(CHECKPOINT_DIR, f"generator_{grid_size}x{grid_size}.pt")
    ckpt = torch.load(path, map_location="cpu")
    model = Generator(latent_dim=ckpt["latent_dim"], grid_size=ckpt["grid_size"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def export(grid_size: int) -> tuple[str, str]:
    model = load_generator(grid_size)
    os.makedirs(ONNX_DIR, exist_ok=True)

    dummy = torch.zeros(1, model.latent_dim)

    plain_path = os.path.join(ONNX_DIR, f"generator_{grid_size}x{grid_size}.onnx")
    torch.onnx.export(
        model,
        (dummy,),
        plain_path,
        input_names=["z"],
        output_names=["grid"],
        dynamo=False,
    )

    sum_model = GeneratorWithSum(model)
    sum_model.eval()
    sum_path = os.path.join(ONNX_DIR, f"generator_sum_{grid_size}x{grid_size}.onnx")
    torch.onnx.export(
        sum_model,
        (dummy,),
        sum_path,
        input_names=["z"],
        output_names=["total"],
        dynamo=False,
    )

    print(f"exported {plain_path}")
    print(f"exported {sum_path}")
    return plain_path, sum_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()
    export(args.grid_size)


if __name__ == "__main__":
    main()
