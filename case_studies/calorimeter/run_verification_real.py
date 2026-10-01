"""
Step 5b: Calorimeter verification pipeline — real CaloChallenge data.

Trains ConditionalGenerator on real Dataset 1 photon showers (Zenodo
10.5281/zenodo.8099322), then runs the same three verification properties
as run_verification.py.  Results are saved with a "real_" prefix so they
can be compared directly against the synthetic run.

Key differences from the synthetic pipeline (run_verification.py):
  * Training data: real Geant4 ATLAS showers, not exponential noise
  * Energy range:  [256, 4194304] MeV (14 doublings) vs [256, 4096] MeV
  * Preprocessing: log1p(voxel / incident_energy), targets in [0, ~1.4]
  * Input bounds:  z ∈ [-1,1]^8, log_E ∈ [0, LOG_E_MAX_REAL ≈ 9.70]

Usage
-----
    python -m case_studies.calorimeter.run_verification_real [--save-json]
    python -m case_studies.calorimeter.run_verification_real [--save-json] \\
        [--n-epochs 20] [--max-events 50000]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from case_studies.calorimeter.data.loader import (
    download_data,
    load,
    preprocess,
)
from case_studies.calorimeter.models.generator import (
    E_MIN_MEV,
    Z_DIM,
    ConditionalGenerator,
)
from case_studies.calorimeter.properties import (
    energy_conservation,
    total_energy_upper_bound,
    voxel_non_negativity,
)
from case_studies.calorimeter.run_verification import _probe_sum_bounds
from verifier import (
    LiRPABackend,
    ModelSpec,
    save_result,
    verify,
)

# ── checkpoint location ───────────────────────────────────────────────────────

CKPT_DIR = Path(__file__).parent / "models" / "checkpoints"
CKPT_PATH = CKPT_DIR / "calo_generator_real.pt"

# ── real-data batch sampler ───────────────────────────────────────────────────


def make_real_batch(
    ds_prep,
    batch_size: int = 256,
    rng: np.random.Generator = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a training batch from preprocessed real data.

    Parameters
    ----------
    ds_prep : CaloDataset
        Preprocessed dataset (log1p + normalise_by_incident applied).
    batch_size : int
    rng : np.random.Generator, optional
        If None, a fresh Generator seeded with `seed` is created.
    seed : int
        Used only if rng is None.

    Returns
    -------
    x : (batch, Z_DIM+1) float32 tensor — random z + log_energy
    y : (batch, n_voxels) float32 tensor — preprocessed shower targets
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    N = ds_prep.n_events
    idx = rng.integers(0, N, size=batch_size)

    # Target: preprocessed shower values in [0, ~1.4]
    y = torch.from_numpy(ds_prep.showers[idx])

    # Input: random latents + log-normalised incident energy for the sampled events
    z = rng.uniform(-1.0, 1.0, size=(batch_size, Z_DIM)).astype(np.float32)
    e_mev = ds_prep.incident_energies[idx]
    log_e = np.log(e_mev / E_MIN_MEV).reshape(-1, 1).astype(np.float32)
    x = torch.from_numpy(np.concatenate([z, log_e], axis=1))

    return x, y


# ── training on real data ─────────────────────────────────────────────────────


def train_on_real(
    model: ConditionalGenerator,
    ds_prep,
    n_epochs: int = 20,
    batch_size: int = 256,
    n_batches_per_epoch: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
) -> list[float]:
    """MSE training on real preprocessed data.  Returns per-epoch loss."""
    rng = np.random.default_rng(seed)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for _ in range(n_batches_per_epoch):
            x, y = make_real_batch(ds_prep, batch_size=batch_size, rng=rng)
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


# ── main pipeline ─────────────────────────────────────────────────────────────


def main(
    n_epochs: int = 20,
    save_json: bool = False,
    max_events: int | None = None,
    n_voxels: int = 368,
) -> None:
    print("=" * 60)
    print("Step 5b: Calorimeter Verification Pipeline (real data)")
    print(f"  particle=photon, n_voxels={n_voxels}, n_epochs={n_epochs}")
    print("=" * 60)

    # ── 0. Download data if needed ────────────────────────────────────────
    data_path = Path(__file__).parent / "data" / "dataset_1_photons_1.hdf5"
    if not data_path.exists():
        print("\n[0/5] Downloading dataset_1_photons_1.hdf5 from Zenodo …")
        download_data(particle="photon", split="train")
    else:
        print(f"\n[0/5] Data found: {data_path.name}")

    # ── 1. Load and preprocess real data ─────────────────────────────────
    print("\n[1/5] Loading and preprocessing real photon data …")
    ds_raw = load("photon", "train", max_events=max_events)
    ds_prep = preprocess(ds_raw, log1p=True, normalise_by_incident=True)
    print(f"  events       : {ds_prep.n_events:,}")
    print(f"  energy range : [{ds_raw.incident_energies.min():.0f}, "
          f"{ds_raw.incident_energies.max():.0f}] MeV")
    print(f"  shower range : [{ds_prep.showers.min():.4f}, {ds_prep.showers.max():.4f}]")
    print(f"  sparsity     : {(ds_raw.showers == 0).mean():.1%}")

    # The real data spans 256–4194304 MeV (2^0 to 2^14 × E_MIN_MEV).
    # LOG_E_MAX_REAL is larger than generator.py's LOG_E_MAX (which was
    # calibrated to 256–4096 MeV only).
    log_e_max_real = float(np.log(ds_raw.incident_energies.max() / E_MIN_MEV))
    print(f"  LOG_E_MAX_REAL: {log_e_max_real:.4f}  "
          f"(cf. synthetic LOG_E_MAX ≈ 2.773)")

    # ── 2. Train on real data ─────────────────────────────────────────────
    print("\n[2/5] Training ConditionalGenerator on real data …")
    model = ConditionalGenerator(n_voxels=n_voxels, z_dim=Z_DIM,
                                 hidden_dim=64, n_hidden=2)
    losses = train_on_real(model, ds_prep, n_epochs=n_epochs)
    print(f"  final loss = {losses[-1]:.4f}")

    # ── 3. Save checkpoint ────────────────────────────────────────────────
    print("\n[3/5] Saving checkpoint …")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"  saved to {CKPT_PATH}")

    # ── 4. Build ModelSpec (use real energy range for input bounds) ───────
    print("\n[4/5] Building ModelSpec …")
    lb_list = [-1.0] * Z_DIM + [0.0]
    ub_list = [1.0] * Z_DIM + [log_e_max_real]
    fresh = ConditionalGenerator(n_voxels=n_voxels, z_dim=Z_DIM,
                                 hidden_dim=64, n_hidden=2)
    spec = ModelSpec.from_checkpoint(
        fresh, str(CKPT_PATH),
        input_lb=lb_list, input_ub=ub_list,
        name=f"calo_generator_real_n{n_voxels}",
    )
    print(f"  name       : {spec.name}")
    print(f"  ckpt hash  : {spec.checkpoint_hash}")
    print(f"  input dim  : {len(lb_list)}")
    print(f"  input region: z∈[-1,1]^{Z_DIM}, log_e∈[0,{log_e_max_real:.4f}]")

    backend = LiRPABackend()
    results = []

    # ── 5. Verification ───────────────────────────────────────────────────
    print("\n[5/5] Running verification …")

    # Property 1: Voxel non-negativity (IBP)
    print("\n  Property 1: voxel_non_negativity (IBP)")
    prop1 = voxel_non_negativity()
    r1 = verify(spec, prop1, backend, config={"method": "IBP"})
    results.append(r1)
    print(f"    status       : {r1.status}")
    print(f"    bound_proven : {r1.bound_proven}")
    print(f"    notes        : {r1.notes}")

    # Property 2: Total energy upper bound
    print("\n  Property 2: total_energy_upper_bound")
    ibp_sum_lb, ibp_sum_ub = _probe_sum_bounds(spec, method="IBP",
                                               n_voxels=n_voxels)
    crown_sum_lb, crown_sum_ub = _probe_sum_bounds(spec, method="CROWN",
                                                   n_voxels=n_voxels)
    print(f"    IBP  sum bounds : [{ibp_sum_lb:.2f}, {ibp_sum_ub:.2f}]")
    print(f"    CROWN sum bounds: [{crown_sum_lb:.2f}, {crown_sum_ub:.2f}]")

    prop2_ibp = total_energy_upper_bound(n_voxels, upper_mev=ibp_sum_ub + 1.0)
    r2_ibp = verify(spec, prop2_ibp, backend, config={"method": "IBP"})
    results.append(r2_ibp)
    print(f"\n    IBP  upper={ibp_sum_ub + 1.0:.2f}")
    print(f"      status: {r2_ibp.status}  (bound_proven={r2_ibp.bound_proven})")

    prop2_crown = total_energy_upper_bound(n_voxels, upper_mev=crown_sum_ub + 1.0)
    r2_crown = verify(spec, prop2_crown, backend, config={"method": "CROWN"})
    results.append(r2_crown)
    print(f"\n    CROWN upper={crown_sum_ub + 1.0:.2f}")
    print(f"      status: {r2_crown.status}  (bound_proven={r2_crown.bound_proven})")

    # Property 3: Energy conservation epsilon sweep (CROWN)
    print("\n  Property 3: energy_conservation epsilon sweep (CROWN)")
    target = (crown_sum_lb + crown_sum_ub) / 2.0
    half_width = (crown_sum_ub - crown_sum_lb) / 2.0
    print(f"    target     = {target:.4f}  (CROWN sum midpoint)")
    print(f"    CROWN window: [{crown_sum_lb:.4f}, {crown_sum_ub:.4f}]")

    epsilon_candidates = [
        round(half_width * 1.01, 4),
        round(half_width * 0.75, 4),
        round(half_width * 0.50, 4),
        round(half_width * 0.25, 4),
        round(half_width * 0.10, 4),
        round(half_width * 0.01, 4),
    ]

    tightest_proven_eps = None
    for eps in epsilon_candidates:
        prop_e = energy_conservation(n_voxels, target_mev=target, epsilon_mev=eps)
        r_e = verify(spec, prop_e, backend, config={"method": "CROWN"})
        results.append(r_e)
        if r_e.status == "PROVEN" and tightest_proven_eps is None:
            tightest_proven_eps = eps
        print(f"    epsilon={eps:>10.4f} → {r_e.status}")

    if tightest_proven_eps is not None:
        print(f"\n    Tightest PROVEN epsilon: {tightest_proven_eps:.4f}")
        print(f"    (full input box: z∈[-1,1]^{Z_DIM}, log_e∈[0,{log_e_max_real:.4f}])")
    else:
        print("\n    All conservation queries INCONCLUSIVE")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("-" * 60)
    print(f"  {'Property':<40} {'Status':<15} {'Bound'}")
    print("-" * 60)
    for r in results:
        bnd = f"{r.bound_proven:.4f}" if r.bound_proven is not None else "—"
        print(f"  {r.property_name:<40} {r.status:<15} {bnd}")

    # ── Save results ──────────────────────────────────────────────────────
    if save_json:
        out_dir = Path(__file__).parent.parent.parent / "results" / "calorimeter"
        saved = []
        for r in results:
            run_id = f"real_{r.property_name}_{r.config.get('method', '')}"
            p = save_result(r, out_dir, run_id=run_id)
            saved.append(str(p))
        print(f"\n  Saved {len(saved)} result files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--n-voxels", type=int, default=368)
    parser.add_argument("--max-events", type=int, default=None,
                        help="Truncate dataset (useful for quick tests)")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    main(
        n_epochs=args.n_epochs,
        save_json=args.save_json,
        max_events=args.max_events,
        n_voxels=args.n_voxels,
    )
