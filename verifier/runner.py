"""
Top-level runner: verify() and save_result().

The runner wraps a backend call and fills in the traceability fields
(git commit, seed, timestamp) that backends do not set.  It also handles
saving results to results/<domain>/<run_id>.json.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .backend.base import Backend
from .model import ModelSpec
from .property import Property
from .result import VerificationResult


# ── git helper ────────────────────────────────────────────────────────────────

def _git_commit(repo_root: Optional[Path] = None) -> str:
    try:
        cwd = str(repo_root or Path(__file__).parent)
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


# ── public API ────────────────────────────────────────────────────────────────

def verify(
    model_spec: ModelSpec,
    prop: Property,
    backend: Backend,
    config: dict,
    seed: int = 42,
) -> VerificationResult:
    """
    Run verification and return a fully populated VerificationResult.

    The backend handles the solver call; this function fills in traceability
    fields (git commit, seed, timestamp, config) and returns the result.
    Call save_result() separately if you want it written to disk.
    """
    result = backend.verify(model_spec, prop, config)
    result.git_commit = _git_commit()
    result.seed = seed
    result.timestamp = datetime.now(timezone.utc).isoformat()
    result.config = config
    result.checkpoint_path = model_spec.checkpoint_path
    result.checkpoint_hash = model_spec.checkpoint_hash
    return result


def save_result(
    result: VerificationResult,
    output_dir: str | Path,
    run_id: Optional[str] = None,
) -> Path:
    """
    Write *result* to results/<domain>/<run_id>.json and return the path.

    Parameters
    ----------
    result : VerificationResult
    output_dir : str or Path
        Directory to write into (created if it does not exist).
    run_id : str, optional
        Explicit run identifier.  Defaults to a UUID prefix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
    path = output_dir / f"{run_id}.json"
    path.write_text(result.to_json())
    return path


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)
