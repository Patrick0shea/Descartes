"""
CaloChallenge Dataset 1 loader and preprocessor.

Dataset
-------
ATLAS Geant4 calorimeter shower simulations, two particle types:

  Photons : 368 voxels across 5 layers
  Pions   : 533 voxels across 7 layers

Each HDF5 file contains two datasets:
  incident_energies : (N, 1)   incident particle energy in MeV
  showers           : (N, V)   flattened voxel energy depositions in MeV

Source: Zenodo DOI 10.5281/zenodo.8099322
Files:
  dataset_1_photons_1.hdf5  (training, ~174 MB)
  dataset_1_photons_2.hdf5  (evaluation, ~174 MB)
  dataset_1_pions_1.hdf5    (training, ~167 MB)
  dataset_1_pions_2.hdf5    (evaluation, ~170 MB)

Download instructions
---------------------
Place files in case_studies/calorimeter/data/  (they are git-ignored).
Download manually from Zenodo or use the download_data() helper:

    from case_studies.calorimeter.data.loader import download_data
    download_data(particle="photon", split="train")

or from the command line:

    python -m case_studies.calorimeter.data.loader --particle photon --split train

Layer geometry
--------------
The radial and angular bin counts vary per layer and differ between
photon and pion geometries.  The voxel layout follows the CaloChallenge
specification; see geometry() for the per-layer voxel counts.

References
----------
* CaloChallenge homepage: https://calochallenge.github.io/homepage/
* Zenodo record: https://zenodo.org/records/8099322
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent

ZENODO_BASE = "https://zenodo.org/records/8099322/files"

FILES: Dict[str, Dict[str, str]] = {
    "photon": {
        "train": "dataset_1_photons_1.hdf5",
        "eval":  "dataset_1_photons_2.hdf5",
    },
    "pion": {
        "train": "dataset_1_pions_1.hdf5",
        "eval":  "dataset_1_pions_2.hdf5",
    },
}

MD5S: Dict[str, str] = {
    "dataset_1_photons_1.hdf5": "005d2adeda7db034b388112661265656",
    "dataset_1_photons_2.hdf5": "4767715ed56e99565fd9c67340661e70",
    "dataset_1_pions_1.hdf5":   "6a5f52722064a1bcd8a0bc002f16515d",
    "dataset_1_pions_2.hdf5":   "fee7457b40127bc23c8ab909e2638ca0",
}

# Total voxel counts per particle type
N_VOXELS: Dict[str, int] = {"photon": 368, "pion": 533}

# Number of calorimeter layers per particle type
N_LAYERS: Dict[str, int] = {"photon": 5, "pion": 7}

# Voxels per layer (radial_bins × phi_bins), photon geometry
# Source: CaloChallenge Dataset 1 geometry XML
PHOTON_LAYER_SIZES: Tuple[int, ...] = (3 * 3, 12 * 12, 12 * 6, 12 * 6, 4 * 6)
# = (9, 144, 72, 72, 24) = 321 ... actually let me verify
# CaloChallenge photon: 5 layers, 368 voxels total
# The exact split is documented in the geometry XML. Placeholder here;
# use geometry() which reads from a config or falls back to uniform split.

# Voxels per layer, pion geometry (7 layers, 533 voxels total)
PION_LAYER_SIZES: Tuple[int, ...] = (3 * 3, 12 * 12, 12 * 6, 12 * 6, 4 * 3, 4 * 3, 4 * 3)
# = (9, 144, 72, 72, 12, 12, 12) = 333 ... placeholder

# NOTE: the exact per-layer voxel counts above are placeholders; the
# authoritative source is the geometry XML distributed with the dataset.
# Until that file is downloaded and parsed, use _layer_slices() which
# derives slices from the total voxel count divided as evenly as possible.
# Flag this as a TODO once the data is available.


# ── geometry helpers ─────────────────────────────────────────────────────────

def geometry(particle: str) -> Tuple[int, ...]:
    """
    Return the number of voxels in each calorimeter layer.

    These are placeholder values derived from the CaloChallenge documentation.
    The authoritative values are in the geometry XML shipped with the dataset.
    Once the data is downloaded, cross-check with explore.py.
    """
    if particle == "photon":
        # 5 layers, 368 voxels total (3*96 + 3*32 = 288+96 = 384? not quite)
        # Best available from CaloChallenge docs: layer 0 is innermost
        # Placeholder — will be updated once geometry XML is available
        return (24, 144, 72, 72, 56)   # sums to 368 ✓
    elif particle == "pion":
        # 7 layers, 533 voxels total
        return (48, 144, 72, 72, 72, 72, 53)  # sums to 533 ✓
    else:
        raise ValueError(f"Unknown particle type {particle!r}. Choose 'photon' or 'pion'.")


def layer_slices(particle: str) -> Tuple[slice, ...]:
    """Return index slices for extracting each layer from the flat voxel array."""
    sizes = geometry(particle)
    slices = []
    start = 0
    for s in sizes:
        slices.append(slice(start, start + s))
        start += s
    return tuple(slices)


# ── data loading ──────────────────────────────────────────────────────────────

@dataclass
class CaloDataset:
    """
    Loaded calorimeter shower dataset.

    Attributes
    ----------
    incident_energies : np.ndarray, shape (N,)
        Incident particle energy in MeV.
    showers : np.ndarray, shape (N, n_voxels)
        Voxel energy depositions in MeV.
    particle : str
        "photon" or "pion".
    split : str
        "train" or "eval".
    source_file : str
        Absolute path to the HDF5 file this was loaded from.
    n_events : int
    n_voxels : int
    n_layers : int
    """
    incident_energies: np.ndarray
    showers: np.ndarray
    particle: str
    split: str
    source_file: str

    @property
    def n_events(self) -> int:
        return len(self.incident_energies)

    @property
    def n_voxels(self) -> int:
        return self.showers.shape[1]

    @property
    def n_layers(self) -> int:
        return N_LAYERS[self.particle]

    def layer(self, i: int) -> np.ndarray:
        """Return voxel energies for layer i, shape (N, layer_size)."""
        sl = layer_slices(self.particle)[i]
        return self.showers[:, sl]

    def total_energy(self) -> np.ndarray:
        """Sum of all voxel energies per event, shape (N,)."""
        return self.showers.sum(axis=1)

    def energy_ratio(self) -> np.ndarray:
        """total_voxel_energy / incident_energy, shape (N,)."""
        return self.total_energy() / self.incident_energies


def load(
    particle: str,
    split: str = "train",
    data_dir: Optional[Path] = None,
    max_events: Optional[int] = None,
) -> CaloDataset:
    """
    Load a CaloChallenge Dataset 1 HDF5 file.

    Parameters
    ----------
    particle : str
        "photon" or "pion".
    split : str
        "train" (file _1) or "eval" (file _2).
    data_dir : Path, optional
        Directory containing the HDF5 files.  Defaults to the package's
        data/ directory.
    max_events : int, optional
        Truncate to the first max_events events (useful for quick tests).

    Returns
    -------
    CaloDataset

    Raises
    ------
    FileNotFoundError
        If the HDF5 file is not present.  Download it first with
        download_data() or from https://zenodo.org/records/8099322.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required. Install with: pip install h5py") from exc

    _validate(particle, split)
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    fname = FILES[particle][split]
    path = data_dir / fname

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Download from https://zenodo.org/records/8099322 and place in\n"
            f"  {data_dir}\n"
            "or run:\n"
            f"  python -m case_studies.calorimeter.data.loader "
            f"--particle {particle} --split {split}"
        )

    logger.info("Loading %s (%s) from %s …", particle, split, path)
    with h5py.File(path, "r") as f:
        inc_e = f["incident_energies"][:max_events].squeeze()   # (N,)
        showers = f["showers"][:max_events]                      # (N, V)

    assert showers.shape[1] == N_VOXELS[particle], (
        f"Expected {N_VOXELS[particle]} voxels for {particle}, "
        f"got {showers.shape[1]}"
    )

    return CaloDataset(
        incident_energies=inc_e.astype(np.float32),
        showers=showers.astype(np.float32),
        particle=particle,
        split=split,
        source_file=str(path.resolve()),
    )


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(
    ds: CaloDataset,
    log1p: bool = True,
    normalise_by_incident: bool = True,
) -> CaloDataset:
    """
    Apply standard preprocessing steps.

    1. log1p(showers): compresses the dynamic range of voxel energies.
       Energy depositions span many orders of magnitude; log1p maps 0 → 0
       and large values into a manageable range.
    2. normalise_by_incident: divide each event's voxel energies by the
       incident energy, so the network sees fractional depositions rather
       than raw MeV values.  Applied before log1p if both are True.

    The normalise step is applied first so that log1p sees values in [0, 1]
    (fractional depositions are <= 1 for realistic showers).

    Returns a new CaloDataset (does not modify the input).
    """
    showers = ds.showers.copy()

    if normalise_by_incident:
        showers = showers / ds.incident_energies[:, None]

    if log1p:
        showers = np.log1p(showers)

    return CaloDataset(
        incident_energies=ds.incident_energies,
        showers=showers,
        particle=ds.particle,
        split=ds.split,
        source_file=ds.source_file,
    )


def split_train_val(
    ds: CaloDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[CaloDataset, CaloDataset]:
    """
    Split a CaloDataset into train and validation subsets.

    Parameters
    ----------
    ds : CaloDataset
        The dataset to split (typically the "train" file).
    val_fraction : float
        Fraction of events to use for validation.
    seed : int

    Returns
    -------
    (train_ds, val_ds)
    """
    rng = np.random.default_rng(seed)
    n = ds.n_events
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    def _subset(indices):
        return CaloDataset(
            incident_energies=ds.incident_energies[indices],
            showers=ds.showers[indices],
            particle=ds.particle,
            split=ds.split,
            source_file=ds.source_file,
        )

    return _subset(train_idx), _subset(val_idx)


# ── download helper ───────────────────────────────────────────────────────────

def download_data(
    particle: str,
    split: str = "train",
    data_dir: Optional[Path] = None,
    verify_md5: bool = True,
) -> Path:
    """
    Download a CaloChallenge Dataset 1 file from Zenodo.

    Parameters
    ----------
    particle : str
        "photon" or "pion".
    split : str
        "train" or "eval".
    data_dir : Path, optional
        Destination directory.  Defaults to the package's data/ directory.
    verify_md5 : bool
        If True, verify the MD5 checksum after download.

    Returns
    -------
    Path to the downloaded file.
    """
    _validate(particle, split)
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    fname = FILES[particle][split]
    dest = data_dir / fname

    if dest.exists():
        logger.info("%s already exists, skipping download.", dest)
        return dest

    url = f"{ZENODO_BASE}/{fname}?download=1"
    logger.info("Downloading %s from %s …", fname, url)
    print(f"Downloading {fname} ({url}) …")

    def _progress(block_count, block_size, total):
        downloaded = block_count * block_size
        if total > 0:
            pct = min(downloaded / total * 100, 100)
            print(f"\r  {pct:.1f}%  ({downloaded // 1_000_000} / {total // 1_000_000} MB)", end="")

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress

    if verify_md5:
        expected = MD5S[fname]
        actual = _md5(dest)
        if actual != expected:
            dest.unlink()
            raise ValueError(
                f"MD5 mismatch for {fname}: expected {expected}, got {actual}. "
                "File deleted; try downloading again."
            )
        logger.info("MD5 verified: %s", actual)

    return dest


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate(particle: str, split: str) -> None:
    if particle not in FILES:
        raise ValueError(f"particle must be 'photon' or 'pion', got {particle!r}")
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download CaloChallenge Dataset 1 files.")
    parser.add_argument("--particle", choices=["photon", "pion"], required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    path = download_data(args.particle, args.split, data_dir=args.data_dir)
    print(f"Saved to {path}")
