"""
PGD-based gradient falsification baseline.

Falsification tries to find a counterexample to a property using
projected gradient descent (PGD) with random restarts.  It is NOT a
formal proof: NO_COUNTEREXAMPLE_FOUND does not mean the property holds
everywhere.  Use it as a baseline to show what verification adds beyond
sampling-based testing.

Supported properties: NonNegativity, Conservation, RangeBound.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .model import ModelSpec
from .property import Conservation, NonNegativity, Property, RangeBound
from .result import FalsificationResult


def falsify(
    model_spec: ModelSpec,
    prop: Property,
    n_restarts: int = 100,
    n_steps: int = 200,
    step_size: float = 0.01,
    seed: int = 42,
    config: Optional[dict] = None,
) -> FalsificationResult:
    """
    Try to falsify *prop* using PGD with random restarts.

    For each restart: sample a random point in the input box, then run
    projected gradient ascent on the violation magnitude, projecting back
    into the input box after each step.

    Returns
    -------
    FalsificationResult
        status COUNTEREXAMPLE_FOUND if a violation was found, else
        NO_COUNTEREXAMPLE_FOUND.
    """
    if config is None:
        config = {}

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    model = model_spec.model
    model.eval()
    lb = torch.tensor(model_spec.input_lb, dtype=torch.float32)
    ub = torch.tensor(model_spec.input_ub, dtype=torch.float32)
    n_inputs = len(model_spec.input_lb)

    best_violation: Optional[float] = None
    best_x: Optional[np.ndarray] = None

    for _ in range(n_restarts):
        # Random start within the input box
        x0 = rng.uniform(
            model_spec.input_lb, model_spec.input_ub
        ).astype(np.float32)
        x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)

        for _ in range(n_steps):
            if x.grad is not None:
                x.grad.zero_()
            out = model(x.unsqueeze(0)).squeeze(0)
            loss = _violation(out, prop)  # we want to maximise this
            loss.backward()  # compute d_loss/dx for gradient ascent

            with torch.no_grad():
                x_new = x + step_size * x.grad.sign()  # gradient ascent
                x_new = torch.clamp(x_new, lb, ub)
            x = x_new.detach().requires_grad_(True)

        with torch.no_grad():
            out_final = model(x.unsqueeze(0)).squeeze(0)
            viol = float(_violation(out_final, prop).item())

        if viol > 0.0:
            if best_violation is None or viol > best_violation:
                best_violation = viol
                best_x = x.detach().numpy().copy()

    runtime_s = time.perf_counter() - t0

    if best_x is not None and best_violation is not None and best_violation > 0.0:
        return FalsificationResult(
            status="COUNTEREXAMPLE_FOUND",
            property_name=prop.name,
            counterexample=best_x.tolist(),
            violation_magnitude=best_violation,
            n_restarts=n_restarts,
            runtime_s=runtime_s,
            model_name=model_spec.name,
            notes=(
                f"PGD found violation magnitude {best_violation:.6f} "
                f"({n_restarts} restarts, {n_steps} steps each)"
            ),
            seed=seed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=config,
        )
    else:
        return FalsificationResult(
            status="NO_COUNTEREXAMPLE_FOUND",
            property_name=prop.name,
            counterexample=None,
            violation_magnitude=None,
            n_restarts=n_restarts,
            runtime_s=runtime_s,
            model_name=model_spec.name,
            notes=(
                f"PGD found no violation "
                f"({n_restarts} restarts, {n_steps} steps each). "
                "This does not prove the property holds everywhere."
            ),
            seed=seed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=config,
        )


# ── violation functions ───────────────────────────────────────────────────────

def _violation(output: torch.Tensor, prop: Property) -> torch.Tensor:
    """
    Compute a scalar violation magnitude (>0 means property is violated).
    Gradient flows through this.
    """
    if isinstance(prop, NonNegativity):
        # maximise how negative the most negative output is
        return torch.relu(-output.min())

    elif isinstance(prop, Conservation):
        coeff = torch.tensor(prop.coefficients, dtype=torch.float32)
        s = (coeff * output).sum()
        # maximise |s - target| - epsilon
        return torch.relu(torch.abs(s - prop.target) - prop.epsilon)

    elif isinstance(prop, RangeBound):
        coeff = torch.tensor(prop.coefficients, dtype=torch.float32)
        s = (coeff * output).sum()
        viol = torch.zeros(1)
        if prop.lower is not None:
            viol = torch.max(viol, torch.relu(prop.lower - s))
        if prop.upper is not None:
            viol = torch.max(viol, torch.relu(s - prop.upper))
        return viol

    else:
        raise ValueError(f"No violation function for {type(prop).__name__}")
