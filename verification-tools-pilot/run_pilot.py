"""Orchestrates the full pilot sweep: for each grid size, train (if no
checkpoint exists), export ONNX, then run every available verification
tool on both properties. Writes results/*.json per (tool, grid size) and
results/summary.json collecting everything.

Usage: python run_pilot.py
"""

from __future__ import annotations

import importlib
import json
import os
import time

import train as train_mod
import export_onnx as export_mod

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GRID_SIZES = [8, 16, 32]
LATENT_DIM = 8


def ensure_trained_and_exported(grid_size: int):
    ckpt_path = os.path.join(train_mod.CHECKPOINT_DIR, f"generator_{grid_size}x{grid_size}.pt")
    if not os.path.exists(ckpt_path):
        model = train_mod.train(grid_size=grid_size, latent_dim=LATENT_DIM, steps=300)
        os.makedirs(train_mod.CHECKPOINT_DIR, exist_ok=True)
        import torch

        torch.save(
            {"state_dict": model.state_dict(), "latent_dim": LATENT_DIM, "grid_size": grid_size, "hidden": model.hidden},
            ckpt_path,
        )
        print(f"trained and saved {ckpt_path}")
    export_mod.export(grid_size)


def run_tool(module_name: str, grid_size: int) -> dict | None:
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        return {"tool": module_name, "grid_size": grid_size, "status": "import_failed", "error": str(e)}

    try:
        t0 = time.time()
        result = mod.run(grid_size)
        result["wall_clock_total_seconds"] = time.time() - t0
        result["status"] = "ok"
        return result
    except Exception as e:
        return {"tool": module_name, "grid_size": grid_size, "status": "error", "error": str(e)}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []

    for grid_size in GRID_SIZES:
        print(f"\n===== grid size {grid_size}x{grid_size} =====")
        ensure_trained_and_exported(grid_size)

        for module_name in ["verify_auto_lirpa", "verify_alpha_beta_crown", "verify_marabou"]:
            print(f"\n--- {module_name} @ {grid_size}x{grid_size} ---")
            result = run_tool(module_name, grid_size)
            summary.append(result)
            fname = f"{module_name.replace('verify_', '')}_{grid_size}x{grid_size}.json"
            with open(os.path.join(RESULTS_DIR, fname), "w") as f:
                json.dump(result, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {os.path.join(RESULTS_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()
