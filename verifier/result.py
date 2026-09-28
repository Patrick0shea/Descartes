"""
Result schema for verification and falsification runs.

Every result is self-describing: it carries the config used, the model and
checkpoint identifiers, the git commit, and the exact input region covered,
so a JSON file can be audited without knowing which script produced it.

Status vocabulary
-----------------
Verification results:
  PROVEN          — property holds over the full input region (sound proof).
  COUNTEREXAMPLE  — a concrete violating input was found.
  INCONCLUSIVE    — neither proven nor disproven within the budget.

Falsification results:
  COUNTEREXAMPLE_FOUND    — PGD found a violating input (not a proof).
  NO_COUNTEREXAMPLE_FOUND — PGD failed to find one (does not imply the
                            property holds everywhere).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    # ── core (set by backend) ──────────────────────────────────────────────
    status: str                          # PROVEN | COUNTEREXAMPLE | INCONCLUSIVE
    property_name: str
    backend: str
    bound_proven: Optional[float]        # tightest epsilon proven (Conservation),
                                         # or minimum lb (NonNegativity), or None
    counterexample: Optional[List[float]]
    input_lb: List[float]
    input_ub: List[float]
    runtime_s: float
    model_name: str
    notes: str = ""

    # ── traceability (filled in by runner) ────────────────────────────────
    checkpoint_path: Optional[str] = None
    checkpoint_hash: Optional[str] = None
    git_commit: str = ""
    seed: int = 0
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=_json_default)

    @classmethod
    def from_json(cls, s: str) -> "VerificationResult":
        return cls(**json.loads(s))


@dataclass
class FalsificationResult:
    # ── core (set by falsifier) ───────────────────────────────────────────
    status: str                          # COUNTEREXAMPLE_FOUND | NO_COUNTEREXAMPLE_FOUND
    property_name: str
    counterexample: Optional[List[float]]
    violation_magnitude: Optional[float] # how much the property is violated
    n_restarts: int
    runtime_s: float
    model_name: str
    notes: str = ""

    # ── traceability ──────────────────────────────────────────────────────
    git_commit: str = ""
    seed: int = 0
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=_json_default)

    @classmethod
    def from_json(cls, s: str) -> "FalsificationResult":
        return cls(**json.loads(s))


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for numpy scalars etc."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
