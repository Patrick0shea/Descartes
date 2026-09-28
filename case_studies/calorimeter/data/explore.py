"""
CaloChallenge Dataset 1 — exploration script.

Loads the photon training file (or pion if specified), prints summary
statistics, and reports:

  1. Dataset shapes and dtypes
  2. Incident energy range and distribution (min, max, percentiles)
  3. Total voxel energy vs incident energy: ratio statistics
  4. Per-layer energy statistics
  5. Fraction of zero-energy voxels (sparsity)
  6. Non-negativity check (all energies >= 0?)

These numbers directly inform the calorimeter property definitions:
  - Energy consistency bound: what epsilon makes |sum(voxels)/incident - 1| <= eps?
  - Per-layer energy bounds
  - Whether to expect non-negativity violations in raw Geant4 data

Run:
    python -m case_studies.calorimeter.data.explore --particle photon
    python -m case_studies.calorimeter.data.explore --particle pion

The data files must be downloaded first (see loader.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .loader import CaloDataset, geometry, layer_slices, load, preprocess


def explore(particle: str, data_dir: Path | None = None, max_events: int | None = None) -> dict:
    """
    Load and explore a CaloChallenge Dataset 1 file.

    Returns a dict of summary statistics (also printed to stdout).
    """
    print(f"\n{'='*60}")
    print(f"CaloChallenge Dataset 1 — {particle} (train)")
    print(f"{'='*60}\n")

    ds = load(particle, split="train", data_dir=data_dir, max_events=max_events)

    # ── 1. Shapes ────────────────────────────────────────────────────────────
    print("1. Shapes and dtypes")
    print(f"   incident_energies : {ds.incident_energies.shape}  dtype={ds.incident_energies.dtype}")
    print(f"   showers           : {ds.showers.shape}            dtype={ds.showers.dtype}")
    print(f"   n_events          : {ds.n_events}")
    print(f"   n_voxels          : {ds.n_voxels}")
    print(f"   n_layers          : {ds.n_layers}")
    print()

    # ── 2. Incident energy ───────────────────────────────────────────────────
    inc = ds.incident_energies
    print("2. Incident energy (MeV)")
    print(f"   min    : {inc.min():.1f}")
    print(f"   max    : {inc.max():.1f}")
    print(f"   mean   : {inc.mean():.1f}")
    print(f"   unique : {len(np.unique(inc))} distinct values")
    print(f"   values : {sorted(np.unique(inc).tolist())[:5]} … (first 5)")
    print()

    # ── 3. Energy ratio ──────────────────────────────────────────────────────
    ratio = ds.energy_ratio()
    print("3. Total voxel energy / incident energy  (energy ratio)")
    print(f"   min    : {ratio.min():.4f}")
    print(f"   max    : {ratio.max():.4f}")
    print(f"   mean   : {ratio.mean():.4f}")
    print(f"   std    : {ratio.std():.4f}")
    pcts = [1, 5, 25, 50, 75, 95, 99]
    vals = np.percentile(ratio, pcts)
    for p, v in zip(pcts, vals):
        print(f"   p{p:02d}    : {v:.4f}")
    print()

    # Suggest epsilon for energy consistency property
    max_abs_deviation = float(np.abs(ratio - 1.0).max())
    p99_deviation = float(np.percentile(np.abs(ratio - 1.0), 99))
    print(f"   |ratio - 1| max     : {max_abs_deviation:.4f}  (worst case)")
    print(f"   |ratio - 1| p99     : {p99_deviation:.4f}  (99th percentile)")
    print(f"   => property epsilon suggestion: >= {max_abs_deviation:.4f} to cover full dataset")
    print()

    # ── 4. Per-layer statistics ───────────────────────────────────────────────
    slices = layer_slices(particle)
    layer_geo = geometry(particle)
    print("4. Per-layer energy statistics  (NOTE: layer geometry is placeholder;")
    print("   cross-check with geometry XML once data is available)")
    for i, (sl, size) in enumerate(zip(slices, layer_geo)):
        layer_data = ds.showers[:, sl]
        total = layer_data.sum(axis=1)
        frac = total / ds.incident_energies
        print(
            f"   Layer {i}: {size:4d} voxels | "
            f"energy/incE mean={frac.mean():.4f}  std={frac.std():.4f}  "
            f"max={frac.max():.4f}"
        )
    print()

    # ── 5. Sparsity ────────────────────────────────────────────────────────────
    zero_frac = float((ds.showers == 0).mean())
    print(f"5. Sparsity (fraction of zero-energy voxels): {zero_frac:.4f}")
    print()

    # ── 6. Non-negativity ──────────────────────────────────────────────────────
    n_negative = int((ds.showers < 0).sum())
    print(f"6. Non-negativity: {n_negative} negative voxel values")
    if n_negative == 0:
        print("   => All voxels >= 0 in raw data. Non-negativity is a valid property.")
    else:
        print(f"   => {n_negative} negative values ({n_negative / ds.showers.size:.2e} fraction).")
        print("   => Inspect whether these are genuine or numerical noise.")
    print()

    # ── 7. Preprocessing preview ───────────────────────────────────────────────
    ds_pp = preprocess(ds, log1p=True, normalise_by_incident=True)
    print("7. After preprocess(log1p=True, normalise_by_incident=True)")
    print(f"   showers range  : [{ds_pp.showers.min():.4f}, {ds_pp.showers.max():.4f}]")
    print(f"   showers mean   : {ds_pp.showers.mean():.4f}")
    print(f"   showers std    : {ds_pp.showers.std():.4f}")
    print()

    stats = {
        "particle": particle,
        "n_events": ds.n_events,
        "n_voxels": ds.n_voxels,
        "n_layers": ds.n_layers,
        "incident_energy_min_MeV": float(inc.min()),
        "incident_energy_max_MeV": float(inc.max()),
        "energy_ratio_mean": float(ratio.mean()),
        "energy_ratio_std": float(ratio.std()),
        "energy_ratio_min": float(ratio.min()),
        "energy_ratio_max": float(ratio.max()),
        "max_abs_deviation_from_1": max_abs_deviation,
        "p99_abs_deviation_from_1": p99_deviation,
        "sparsity_fraction": zero_frac,
        "n_negative_voxels": n_negative,
        "preprocessed_range": [float(ds_pp.showers.min()), float(ds_pp.showers.max())],
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore CaloChallenge Dataset 1.")
    parser.add_argument("--particle", choices=["photon", "pion"], default="photon")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-events", type=int, default=None,
                        help="Truncate to first N events for a quick look.")
    parser.add_argument("--save-json", type=Path, default=None,
                        help="Save summary stats to this JSON file.")
    args = parser.parse_args()

    stats = explore(args.particle, data_dir=args.data_dir, max_events=args.max_events)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {args.save_json}")
