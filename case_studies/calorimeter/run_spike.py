"""
Step 4 spike: train ConditionalGenerator on synthetic data, wrap in
auto_LiRPA BoundedModule, run IBP and CROWN bounds, report bound widths.

This is a diagnostic script only — the goal is to confirm the model traces
through auto_LiRPA cleanly and to see what the output bounds look like.
No real CaloChallenge data is needed.

Usage
-----
    python -m case_studies.calorimeter.run_spike [--n-epochs 20] [--save-json]

Output
------
Prints bound statistics to stdout.
Optionally writes results/calorimeter/spike.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from case_studies.calorimeter.models.generator import (
    ConditionalGenerator,
    LOG_E_MAX,
    Z_DIM,
    encode_input,
)

# ── synthetic data ────────────────────────────────────────────────────────────

def make_synthetic_batch(
    n: int,
    n_voxels: int = 368,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (x, y_target) for n synthetic events.

    x      : (n, Z_DIM+1)  — random z ∈ [-1,1]^8, random log_energy ∈ [0, LOG_E_MAX]
    y_target : (n, n_voxels) — non-negative synthetic shower; uses exponential
               values with ~70% sparsity to mimic the real data distribution.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    z = torch.rand(n, Z_DIM, generator=rng) * 2 - 1          # uniform [-1, 1]
    log_e = torch.rand(n, generator=rng) * LOG_E_MAX           # uniform [0, LOG_E_MAX]
    x = torch.cat([z, log_e.unsqueeze(1)], dim=1)

    # Target: exponential showers, ~70% zero
    y = torch.zeros(n, n_voxels)
    mask = torch.rand(n, n_voxels, generator=rng) > 0.7
    exp_vals = torch.zeros(mask.sum())
    exp_vals.exponential_(lambd=0.1)   # in-place, uses global RNG (seed fixed above)
    y[mask] = exp_vals

    return x.to(device), y.to(device)


# ── training ──────────────────────────────────────────────────────────────────

def train(
    model: ConditionalGenerator,
    n_epochs: int = 20,
    batch_size: int = 256,
    n_batches_per_epoch: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> list[float]:
    """Brief MSE training on synthetic data.  Returns per-epoch loss."""
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_i in range(n_batches_per_epoch):
            x, y = make_synthetic_batch(
                batch_size,
                n_voxels=model.n_voxels,
                seed=seed + epoch * n_batches_per_epoch + batch_i,
                device=device,
            )
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches_per_epoch
        losses.append(epoch_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:>3d}/{n_epochs}  loss={epoch_loss:.4f}")

    model.eval()
    return losses


# ── bound computation ─────────────────────────────────────────────────────────

def run_bounds(model: ConditionalGenerator) -> dict:
    """
    Wrap the model in auto_LiRPA BoundedModule and run IBP + CROWN bounds.

    Returns a dict with statistics about the computed output bounds.
    """
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except ImportError as exc:
        raise ImportError(
            "auto_LiRPA is not installed.  Install with: pip install auto_LiRPA"
        ) from exc

    lb_list, ub_list = model.input_bounds()
    lb_np = np.array(lb_list, dtype=np.float32)
    ub_np = np.array(ub_list, dtype=np.float32)

    # BoundedModule needs a dummy forward pass
    dummy = torch.zeros(1, len(lb_list))
    bounded_model = BoundedModule(model, dummy, device="cpu")
    bounded_model.eval()

    # Define the input perturbation region (the full input box)
    x_nom = torch.tensor((lb_np + ub_np) / 2, dtype=torch.float32).unsqueeze(0)
    ptb = PerturbationLpNorm(
        norm=np.inf,
        eps=None,
        x_L=torch.tensor(lb_np).unsqueeze(0),
        x_U=torch.tensor(ub_np).unsqueeze(0),
    )
    x_bounded = BoundedTensor(x_nom, ptb)

    results = {}

    for method in ("IBP", "CROWN"):
        t0 = time.perf_counter()
        lb_out, ub_out = bounded_model.compute_bounds(
            x=(x_bounded,), method=method
        )
        elapsed = time.perf_counter() - t0

        lb_out_np = lb_out.detach().numpy()[0]   # (n_voxels,)
        ub_out_np = ub_out.detach().numpy()[0]

        width = ub_out_np - lb_out_np
        n_negative_lb = int((lb_out_np < 0).sum())
        n_outputs = len(lb_out_np)

        results[method] = {
            "lb_min": float(lb_out_np.min()),
            "lb_max": float(lb_out_np.max()),
            "lb_mean": float(lb_out_np.mean()),
            "ub_min": float(ub_out_np.min()),
            "ub_max": float(ub_out_np.max()),
            "ub_mean": float(ub_out_np.mean()),
            "width_min": float(width.min()),
            "width_max": float(width.max()),
            "width_mean": float(width.mean()),
            "n_negative_lb": n_negative_lb,
            "n_outputs": n_outputs,
            "nonneg_proven": n_negative_lb == 0,
            "runtime_s": round(elapsed, 4),
        }

        print(f"\n  [{method}]")
        print(f"    output lower bounds: min={lb_out_np.min():.4f}  max={lb_out_np.max():.4f}  mean={lb_out_np.mean():.4f}")
        print(f"    output upper bounds: min={ub_out_np.min():.4f}  max={ub_out_np.max():.4f}  mean={ub_out_np.mean():.4f}")
        print(f"    bound widths:        min={width.min():.4f}  max={width.max():.4f}  mean={width.mean():.4f}")
        print(f"    outputs with lb < 0: {n_negative_lb} / {n_outputs}")
        print(f"    non-negativity proven: {n_negative_lb == 0}")
        print(f"    runtime: {elapsed:.3f}s")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main(n_epochs: int = 20, save_json: bool = False, n_voxels: int = 368) -> None:
    print(f"Step 4 spike — ConditionalGenerator (n_voxels={n_voxels})")
    print(f"Architecture: z_dim=8, hidden_dim=64, n_hidden=2")
    print(f"Input region: z∈[-1,1]^8, log_energy∈[0, {LOG_E_MAX:.4f}]")
    print()

    model = ConditionalGenerator(n_voxels=n_voxels, z_dim=Z_DIM, hidden_dim=64, n_hidden=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    print(f"Training for {n_epochs} epochs on synthetic data …")
    losses = train(model, n_epochs=n_epochs)
    print(f"  Final training loss: {losses[-1]:.4f}")
    print()

    print("Running bound computation via auto_LiRPA …")
    bound_results = run_bounds(model)

    if save_json:
        out_dir = Path(__file__).parent.parent.parent / "results" / "calorimeter"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "spike.json"
        record = {
            "step": 4,
            "description": "ConditionalGenerator spike: synthetic training + auto_LiRPA bounds",
            "model": {
                "n_voxels": n_voxels,
                "z_dim": Z_DIM,
                "hidden_dim": 64,
                "n_hidden": 2,
                "n_params": n_params,
                "log_e_max": LOG_E_MAX,
            },
            "training": {
                "n_epochs": n_epochs,
                "final_loss": losses[-1],
                "losses": losses,
            },
            "bounds": bound_results,
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--n-voxels", type=int, default=368, choices=[368, 533])
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    main(n_epochs=args.n_epochs, save_json=args.save_json, n_voxels=args.n_voxels)
