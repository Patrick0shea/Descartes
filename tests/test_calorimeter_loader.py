"""
Tests for the CaloChallenge data loader.

These tests do NOT require the real HDF5 files.  They create tiny synthetic
HDF5 files in a temp directory, exercise all loader functions, and verify
the contracts the rest of the pipeline depends on.

Tests that require the real data are marked with
    @pytest.mark.skipif(not DATA_AVAILABLE, reason="CaloChallenge data not downloaded")
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

pytestmark = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_fake_hdf5(path: Path, n_events: int, n_voxels: int, seed: int = 0) -> None:
    """Create a minimal HDF5 file matching the CaloChallenge schema."""
    rng = np.random.default_rng(seed)
    with __import__("h5py").File(path, "w") as f:
        # incident_energies: N distinct powers-of-2 energies in MeV
        energies = np.array([256.0 * (2 ** i) for i in range(n_events)], dtype=np.float32)
        f.create_dataset("incident_energies", data=energies.reshape(-1, 1))
        # showers: non-negative values; most voxels zero (realistic sparsity)
        showers = rng.exponential(scale=10.0, size=(n_events, n_voxels)).astype(np.float32)
        showers *= (rng.random((n_events, n_voxels)) > 0.7)  # ~70% zero
        f.create_dataset("showers", data=showers)


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_photon_dir(tmp_path):
    """Temp dir containing a synthetic photon training HDF5."""
    from case_studies.calorimeter.data.loader import FILES, N_VOXELS
    fname = FILES["photon"]["train"]
    _make_fake_hdf5(tmp_path / fname, n_events=50, n_voxels=N_VOXELS["photon"])
    return tmp_path


@pytest.fixture
def fake_pion_dir(tmp_path):
    """Temp dir containing a synthetic pion training HDF5."""
    from case_studies.calorimeter.data.loader import FILES, N_VOXELS
    fname = FILES["pion"]["train"]
    _make_fake_hdf5(tmp_path / fname, n_events=30, n_voxels=N_VOXELS["pion"])
    return tmp_path


# ── test_load ─────────────────────────────────────────────────────────────────

class TestLoad:
    def test_photon_shapes(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import N_VOXELS, load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        assert ds.showers.shape == (50, N_VOXELS["photon"])
        assert ds.incident_energies.shape == (50,)

    def test_pion_shapes(self, fake_pion_dir):
        from case_studies.calorimeter.data.loader import N_VOXELS, load
        ds = load("pion", "train", data_dir=fake_pion_dir)
        assert ds.showers.shape == (30, N_VOXELS["pion"])
        assert ds.incident_energies.shape == (30,)

    def test_max_events(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load
        ds = load("photon", "train", data_dir=fake_photon_dir, max_events=10)
        assert ds.n_events == 10

    def test_dtypes_float32(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        assert ds.showers.dtype == np.float32
        assert ds.incident_energies.dtype == np.float32

    def test_missing_file_raises(self, tmp_path):
        from case_studies.calorimeter.data.loader import load
        with pytest.raises(FileNotFoundError, match="not found"):
            load("photon", "train", data_dir=tmp_path)

    def test_invalid_particle_raises(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load
        with pytest.raises(ValueError, match="particle"):
            load("electron", "train", data_dir=fake_photon_dir)

    def test_invalid_split_raises(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load
        with pytest.raises(ValueError, match="split"):
            load("photon", "test", data_dir=fake_photon_dir)


# ── test_dataset ──────────────────────────────────────────────────────────────

class TestCaloDataset:
    def test_n_voxels(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import N_VOXELS, load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        assert ds.n_voxels == N_VOXELS["photon"]

    def test_n_layers(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import N_LAYERS, load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        assert ds.n_layers == N_LAYERS["photon"]

    def test_total_energy_shape(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        assert ds.total_energy().shape == (ds.n_events,)

    def test_energy_ratio_shape(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        assert ds.energy_ratio().shape == (ds.n_events,)

    def test_layer_method(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import geometry, load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        sizes = geometry("photon")
        for i, size in enumerate(sizes):
            layer = ds.layer(i)
            assert layer.shape == (ds.n_events, size), (
                f"Layer {i}: expected ({ds.n_events}, {size}), got {layer.shape}"
            )

    def test_layer_slices_cover_all_voxels(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import geometry, load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        total_from_layers = sum(ds.layer(i).sum() for i in range(ds.n_layers))
        np.testing.assert_allclose(total_from_layers, ds.showers.sum(), rtol=1e-5)

    def test_source_file_recorded(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load
        ds = load("photon", "train", data_dir=fake_photon_dir)
        assert "photons_1" in ds.source_file
        assert Path(ds.source_file).exists()


# ── test_preprocess ───────────────────────────────────────────────────────────

class TestPreprocess:
    def test_log1p_preserves_shape(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load, preprocess
        ds = load("photon", "train", data_dir=fake_photon_dir)
        ds_pp = preprocess(ds, log1p=True, normalise_by_incident=False)
        assert ds_pp.showers.shape == ds.showers.shape

    def test_log1p_nonneg_input_gives_nonneg_output(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load, preprocess
        ds = load("photon", "train", data_dir=fake_photon_dir)
        ds_pp = preprocess(ds, log1p=True, normalise_by_incident=False)
        assert np.all(ds_pp.showers >= 0.0)

    def test_normalise_by_incident(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load, preprocess
        ds = load("photon", "train", data_dir=fake_photon_dir)
        ds_pp = preprocess(ds, log1p=False, normalise_by_incident=True)
        # After dividing by incident energy, the normalised total should equal
        # original_total / incident_energy for each event.
        expected = ds.total_energy() / ds.incident_energies
        np.testing.assert_allclose(ds_pp.total_energy(), expected, rtol=1e-4)

    def test_does_not_modify_input(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load, preprocess
        ds = load("photon", "train", data_dir=fake_photon_dir)
        original = ds.showers.copy()
        preprocess(ds, log1p=True, normalise_by_incident=True)
        np.testing.assert_array_equal(ds.showers, original)


# ── test_split ────────────────────────────────────────────────────────────────

class TestSplitTrainVal:
    def test_sizes(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load, split_train_val
        ds = load("photon", "train", data_dir=fake_photon_dir)
        train, val = split_train_val(ds, val_fraction=0.2, seed=0)
        assert train.n_events + val.n_events == ds.n_events

    def test_no_overlap(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load, split_train_val
        ds = load("photon", "train", data_dir=fake_photon_dir)
        train, val = split_train_val(ds, val_fraction=0.2, seed=0)
        # Check energies don't overlap (unique per event in our synthetic data)
        train_set = set(train.incident_energies.tolist())
        val_set = set(val.incident_energies.tolist())
        assert train_set.isdisjoint(val_set)

    def test_reproducible(self, fake_photon_dir):
        from case_studies.calorimeter.data.loader import load, split_train_val
        ds = load("photon", "train", data_dir=fake_photon_dir)
        t1, v1 = split_train_val(ds, seed=42)
        t2, v2 = split_train_val(ds, seed=42)
        np.testing.assert_array_equal(t1.showers, t2.showers)


# ── test_geometry ─────────────────────────────────────────────────────────────

class TestGeometry:
    def test_photon_sums_to_n_voxels(self):
        from case_studies.calorimeter.data.loader import N_VOXELS, geometry
        assert sum(geometry("photon")) == N_VOXELS["photon"]

    def test_pion_sums_to_n_voxels(self):
        from case_studies.calorimeter.data.loader import N_VOXELS, geometry
        assert sum(geometry("pion")) == N_VOXELS["pion"]

    def test_layer_slices_non_overlapping_and_complete(self):
        from case_studies.calorimeter.data.loader import N_VOXELS, layer_slices
        for particle in ("photon", "pion"):
            slices = layer_slices(particle)
            covered = set()
            for sl in slices:
                indices = set(range(sl.start, sl.stop))
                assert indices.isdisjoint(covered), "Overlapping layer slices"
                covered |= indices
            assert covered == set(range(N_VOXELS[particle])), "Slices don't cover all voxels"
