"""Brief training of the toy generator for a given grid size.

A few hundred gradient steps on synthetic shower-like targets (see
data.py). This is not meant to produce a good generative model -- only
to move the weights off random init so the verification pilot exercises
a "trained-ish" network rather than pure noise. MSE loss is a reasonable
stand-in since there's no real latent<->image correspondence to fit.

Usage: python train.py --grid-size 8 --latent-dim 8
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from data import make_dataset
from model import Generator

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def train(grid_size: int, latent_dim: int, steps: int = 300, seed: int = 0) -> Generator:
    torch.manual_seed(seed)
    z_np, y_np = make_dataset(n_samples=2000, grid_size=grid_size, latent_dim=latent_dim, seed=seed)
    z = torch.from_numpy(z_np)
    y = torch.from_numpy(y_np)

    model = Generator(latent_dim=latent_dim, grid_size=grid_size)
    print(f"grid_size={grid_size} latent_dim={latent_dim} hidden={model.hidden} params={model.num_params()}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 64
    n = z.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        pred = model(z[idx])
        loss = loss_fn(pred, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = train(args.grid_size, args.latent_dim, args.steps, args.seed)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"generator_{args.grid_size}x{args.grid_size}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "grid_size": args.grid_size,
            "hidden": model.hidden,
        },
        path,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
