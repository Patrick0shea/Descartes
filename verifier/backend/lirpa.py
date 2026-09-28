"""
auto_LiRPA verification backend.

This backend uses bound propagation (CROWN / IBP) to compute sound
lower and upper bounds on network outputs over an input box, then
checks whether the bounds satisfy the requested property.

Result statuses
---------------
PROVEN      — bounds are tight enough to establish the property formally.
INCONCLUSIVE — bounds do not fit the required range.  The property may
               still hold; the relaxation is simply too loose to prove it.

This backend never returns COUNTEREXAMPLE.  Use the Marabou backend or
the PGD falsifier for counterexample search.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ..model import ModelSpec
from ..property import Conservation, NonNegativity, Property, RangeBound
from ..result import VerificationResult
from .base import Backend


# ── internal helper ──────────────────────────────────────────────────────────

class _LinearHead(nn.Module):
    """
    Appends a fixed non-trainable linear projection to a base model.

    Used for the sum-head trick: wrapping the base model so that the
    linear combination of interest appears as an extra output neuron,
    letting auto_LiRPA propagate bounds through it in one call.
    """

    def __init__(self, base: nn.Module, coefficients: np.ndarray) -> None:
        super().__init__()
        self.base = base
        n = len(coefficients)
        self.head = nn.Linear(n, 1, bias=False)
        self.head.weight.data = torch.tensor(
            coefficients, dtype=torch.float32
        ).unsqueeze(0)
        for p in self.head.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.base(x))


# ── backend ───────────────────────────────────────────────────────────────────

class LiRPABackend(Backend):
    """
    Verification via auto_LiRPA bound propagation.

    Install: pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git
    Then:    pip install appdirs graphviz tqdm packaging
    """

    name = "lirpa"

    def verify(
        self,
        model_spec: ModelSpec,
        prop: Property,
        config: dict,
    ) -> VerificationResult:
        try:
            from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        except ImportError as exc:
            raise ImportError(
                "auto_LiRPA is not installed. "
                "Install with: pip install --no-deps "
                "git+https://github.com/Verified-Intelligence/auto_LiRPA.git"
            ) from exc

        method = config.get("method", "CROWN")
        t0 = time.perf_counter()

        lb_t = torch.tensor(model_spec.input_lb, dtype=torch.float32).unsqueeze(0)
        ub_t = torch.tensor(model_spec.input_ub, dtype=torch.float32).unsqueeze(0)
        dummy = lb_t.clone()

        model = model_spec.model
        model.eval()

        if isinstance(prop, NonNegativity):
            return self._verify_non_negativity(
                model, dummy, lb_t, ub_t, prop, method, model_spec, config, t0,
                PerturbationLpNorm, BoundedModule, BoundedTensor,
            )
        elif isinstance(prop, Conservation):
            return self._verify_conservation(
                model, dummy, lb_t, ub_t, prop, method, model_spec, config, t0,
                PerturbationLpNorm, BoundedModule, BoundedTensor,
            )
        elif isinstance(prop, RangeBound):
            return self._verify_range_bound(
                model, dummy, lb_t, ub_t, prop, method, model_spec, config, t0,
                PerturbationLpNorm, BoundedModule, BoundedTensor,
            )
        else:
            raise ValueError(
                f"LiRPABackend does not support property type {type(prop).__name__}. "
                "Supported: NonNegativity, Conservation, RangeBound."
            )

    # ── property-specific helpers ─────────────────────────────────────────

    def _verify_non_negativity(
        self, model, dummy, lb_t, ub_t, prop, method,
        model_spec, config, t0, PerturbationLpNorm, BoundedModule, BoundedTensor,
    ) -> VerificationResult:
        bounded = BoundedModule(model, dummy)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=lb_t, x_U=ub_t)
        x_bnd = BoundedTensor(dummy, ptb)
        out_lb, _ = bounded.compute_bounds(x=(x_bnd,), method=method)
        out_lb_np = out_lb.detach().cpu().numpy().flatten()
        runtime_s = time.perf_counter() - t0

        min_lb = float(np.min(out_lb_np))
        if min_lb >= 0.0:
            return self._result(
                status="PROVEN",
                prop=prop,
                bound_proven=min_lb,
                counterexample=None,
                model_spec=model_spec,
                runtime_s=runtime_s,
                config=config,
                notes=f"CROWN min output lower bound = {min_lb:.6f} >= 0",
            )
        else:
            return self._result(
                status="INCONCLUSIVE",
                prop=prop,
                bound_proven=None,
                counterexample=None,
                model_spec=model_spec,
                runtime_s=runtime_s,
                config=config,
                notes=(
                    f"CROWN min output lower bound = {min_lb:.6f} < 0; "
                    "relaxation too loose to prove non-negativity"
                ),
            )

    def _verify_conservation(
        self, model, dummy, lb_t, ub_t, prop, method,
        model_spec, config, t0, PerturbationLpNorm, BoundedModule, BoundedTensor,
    ) -> VerificationResult:
        augmented = _LinearHead(model, prop.coefficients)
        augmented.eval()

        bounded = BoundedModule(augmented, dummy)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=lb_t, x_U=ub_t)
        x_bnd = BoundedTensor(dummy, ptb)
        sum_lb, sum_ub = bounded.compute_bounds(x=(x_bnd,), method=method)
        sum_lb_val = float(sum_lb.detach().cpu().item())
        sum_ub_val = float(sum_ub.detach().cpu().item())
        runtime_s = time.perf_counter() - t0

        lo_req = prop.target - prop.epsilon
        hi_req = prop.target + prop.epsilon

        if sum_lb_val >= lo_req and sum_ub_val <= hi_req:
            # tightest provable epsilon = max deviation from target
            proven_eps = max(prop.target - sum_lb_val, sum_ub_val - prop.target)
            return self._result(
                status="PROVEN",
                prop=prop,
                bound_proven=proven_eps,
                counterexample=None,
                model_spec=model_spec,
                runtime_s=runtime_s,
                config=config,
                notes=(
                    f"CROWN sum bounds [{sum_lb_val:.6f}, {sum_ub_val:.6f}] "
                    f"fit [{lo_req:.6f}, {hi_req:.6f}]"
                ),
            )
        else:
            return self._result(
                status="INCONCLUSIVE",
                prop=prop,
                bound_proven=None,
                counterexample=None,
                model_spec=model_spec,
                runtime_s=runtime_s,
                config=config,
                notes=(
                    f"CROWN sum bounds [{sum_lb_val:.6f}, {sum_ub_val:.6f}] "
                    f"do not fit [{lo_req:.6f}, {hi_req:.6f}]; "
                    "relaxation too loose to prove conservation"
                ),
            )

    def _verify_range_bound(
        self, model, dummy, lb_t, ub_t, prop, method,
        model_spec, config, t0, PerturbationLpNorm, BoundedModule, BoundedTensor,
    ) -> VerificationResult:
        augmented = _LinearHead(model, prop.coefficients)
        augmented.eval()

        bounded = BoundedModule(augmented, dummy)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=lb_t, x_U=ub_t)
        x_bnd = BoundedTensor(dummy, ptb)
        val_lb, val_ub = bounded.compute_bounds(x=(x_bnd,), method=method)
        val_lb_v = float(val_lb.detach().cpu().item())
        val_ub_v = float(val_ub.detach().cpu().item())
        runtime_s = time.perf_counter() - t0

        lower_ok = (prop.lower is None) or (val_lb_v >= prop.lower)
        upper_ok = (prop.upper is None) or (val_ub_v <= prop.upper)

        if lower_ok and upper_ok:
            return self._result(
                status="PROVEN",
                prop=prop,
                bound_proven=val_lb_v if prop.lower is not None else val_ub_v,
                counterexample=None,
                model_spec=model_spec,
                runtime_s=runtime_s,
                config=config,
                notes=f"CROWN bounds [{val_lb_v:.6f}, {val_ub_v:.6f}] fit range",
            )
        else:
            return self._result(
                status="INCONCLUSIVE",
                prop=prop,
                bound_proven=None,
                counterexample=None,
                model_spec=model_spec,
                runtime_s=runtime_s,
                config=config,
                notes=(
                    f"CROWN bounds [{val_lb_v:.6f}, {val_ub_v:.6f}] "
                    f"do not fit [{prop.lower}, {prop.upper}]"
                ),
            )

    # ── result builder ────────────────────────────────────────────────────

    @staticmethod
    def _result(
        status, prop, bound_proven, counterexample,
        model_spec, runtime_s, config, notes,
    ) -> VerificationResult:
        return VerificationResult(
            status=status,
            property_name=prop.name,
            backend=LiRPABackend.name,
            bound_proven=bound_proven,
            counterexample=counterexample,
            input_lb=model_spec.input_lb.tolist(),
            input_ub=model_spec.input_ub.tolist(),
            runtime_s=runtime_s,
            model_name=model_spec.name,
            notes=notes,
            checkpoint_path=model_spec.checkpoint_path,
            checkpoint_hash=model_spec.checkpoint_hash,
        )
