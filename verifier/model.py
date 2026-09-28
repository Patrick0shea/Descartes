"""ModelSpec: a PyTorch model plus its input region and checkpoint provenance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelSpec:
    """
    A model together with its input region and provenance metadata.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.  Must be in eval mode before verification.
    input_lb : np.ndarray
        Per-dimension lower bounds on the input (shape: [n_inputs]).
    input_ub : np.ndarray
        Per-dimension upper bounds on the input (shape: [n_inputs]).
    name : str
        Short human-readable identifier used in results.
    checkpoint_path : str, optional
        Absolute path to the .pt file the model was loaded from.
    checkpoint_hash : str, optional
        First 16 hex chars of SHA-256 of the checkpoint file.
    """

    model: nn.Module
    input_lb: np.ndarray
    input_ub: np.ndarray
    name: str
    checkpoint_path: Optional[str] = None
    checkpoint_hash: Optional[str] = None

    def __post_init__(self) -> None:
        self.input_lb = np.asarray(self.input_lb, dtype=np.float32)
        self.input_ub = np.asarray(self.input_ub, dtype=np.float32)
        if self.input_lb.shape != self.input_ub.shape:
            raise ValueError("input_lb and input_ub must have the same shape")
        if np.any(self.input_lb > self.input_ub):
            raise ValueError("input_lb must be <= input_ub element-wise")

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str,
        input_lb: np.ndarray,
        input_ub: np.ndarray,
        name: str,
    ) -> "ModelSpec":
        """
        Load a state dict from *checkpoint_path* into *model*, compute a
        SHA-256 hash of the file for traceability, and return a ModelSpec.
        """
        path = Path(checkpoint_path).resolve()
        raw = path.read_bytes()
        sha256_prefix = hashlib.sha256(raw).hexdigest()[:16]
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return cls(
            model=model,
            input_lb=np.asarray(input_lb, dtype=np.float32),
            input_ub=np.asarray(input_ub, dtype=np.float32),
            name=name,
            checkpoint_path=str(path),
            checkpoint_hash=sha256_prefix,
        )
