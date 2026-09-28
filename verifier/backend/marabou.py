"""
Marabou verification backend (complete SMT-based search).

Marabou is an ONNX-only verifier.  This backend exports the model to a
temporary ONNX file, builds the network in Marabou, adds the input-box and
property constraints, and runs the solver.

Marabou solve() API (maraboupy 2.x)
-------------------------------------
    exit_code, values, stats = network.solve(options=..., verbose=False)

    exit_code : str  — "sat", "unsat", or other string (timeout / unknown)
    values    : dict — {var_id: float} when SAT, empty dict when UNSAT/timeout
    stats     : MarabouUtils.Statistics

Result statuses
---------------
PROVEN (UNSAT)         — property holds over the full input region.
COUNTEREXAMPLE (SAT)   — a concrete violating input was found.
INCONCLUSIVE (TIMEOUT) — neither within the time budget.

Scope and limitations
---------------------
* Handles feedforward Linear+ReLU stacks.  Batch-norm, attention, and
  other layer types not supported by the ONNX-to-Marabou path will error
  at export time; the error propagates rather than silently returning
  INCONCLUSIVE.
* NonNegativity requires one Marabou query per output neuron.  At large
  output sizes (e.g. 1024 calorimeter voxels) this is impractical; use
  auto_LiRPA for those cases and Marabou only for targeted spot-checks.
* Conservation / RangeBound expressed as one linear equation per direction
  via network.addInequality (see verification-tools-pilot/REPORT.md).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..model import ModelSpec
from ..property import Conservation, NonNegativity, Property, RangeBound
from ..result import VerificationResult
from .base import Backend


class MarabouBackend(Backend):
    """Verification via Marabou complete SMT-based search."""

    name = "marabou"

    def verify(
        self,
        model_spec: ModelSpec,
        prop: Property,
        config: dict,
    ) -> VerificationResult:
        try:
            from maraboupy import Marabou as M
        except ImportError as exc:
            raise ImportError(
                "maraboupy is not installed.  Install with: pip install maraboupy"
            ) from exc

        timeout = config.get("timeout_s", 60)
        snc = config.get("snc", False)
        num_workers = config.get("num_workers", 1)
        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            self._export_onnx(model_spec.model, model_spec.input_lb, onnx_path)

            if isinstance(prop, NonNegativity):
                result = self._verify_non_negativity(
                    onnx_path, model_spec, prop, config, t0, timeout, snc, num_workers, M,
                )
            elif isinstance(prop, Conservation):
                result = self._verify_conservation(
                    onnx_path, model_spec, prop, config, t0, timeout, snc, num_workers, M,
                )
            elif isinstance(prop, RangeBound):
                result = self._verify_range_bound(
                    onnx_path, model_spec, prop, config, t0, timeout, snc, num_workers, M,
                )
            else:
                raise ValueError(
                    f"MarabouBackend does not support property type {type(prop).__name__}. "
                    "Supported: NonNegativity, Conservation, RangeBound."
                )
        finally:
            Path(onnx_path).unlink(missing_ok=True)

        return result

    # ── ONNX export ───────────────────────────────────────────────────────

    @staticmethod
    def _export_onnx(model: nn.Module, input_lb: np.ndarray, path: str) -> None:
        model.eval()
        n = len(input_lb)
        dummy = torch.zeros(1, n, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            path,
            dynamo=False,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _add_input_bounds(network, lb: np.ndarray, ub: np.ndarray) -> None:
        for i, var in enumerate(network.inputVars[0].flatten()):
            network.setLowerBound(var, float(lb[i]))
            network.setUpperBound(var, float(ub[i]))

    @staticmethod
    def _make_options(timeout: int, snc: bool, num_workers: int, M):
        kwargs = {"timeoutInSeconds": timeout, "verbosity": 0}
        if snc:
            kwargs["snc"] = True
            kwargs["numWorkers"] = num_workers
        return M.createOptions(**kwargs)

    @staticmethod
    def _extract_input(values: dict, network) -> List[float]:
        """Pull input variable values from a SAT assignment."""
        return [float(values.get(v, 0.0)) for v in network.inputVars[0].flatten()]

    # ── property implementations ──────────────────────────────────────────

    def _verify_non_negativity(
        self, onnx_path, model_spec, prop, config, t0,
        timeout, snc, num_workers, M,
    ) -> VerificationResult:
        from maraboupy import Marabou as Mb

        opts = self._make_options(timeout, snc, num_workers, Mb)
        # count outputs from a fresh load
        tmp_net = Mb.read_onnx(onnx_path)
        n_outputs = len(tmp_net.outputVars[0].flatten())

        for i in range(n_outputs):
            net_i = Mb.read_onnx(onnx_path)
            self._add_input_bounds(net_i, model_spec.input_lb, model_spec.input_ub)
            out_vars_i = net_i.outputVars[0].flatten()
            # Query: can output_i < 0?  Constrain output_i <= -1e-9.
            net_i.setUpperBound(out_vars_i[i], -1e-9)

            exit_code, values, stats = net_i.solve(options=opts, verbose=False)

            if exit_code == "sat":
                ce = self._extract_input(values, net_i)
                return self._result(
                    status="COUNTEREXAMPLE", prop=prop, bound_proven=None,
                    counterexample=ce, model_spec=model_spec,
                    runtime_s=time.perf_counter() - t0, config=config,
                    notes=f"Output {i} can be negative (SAT)",
                )
            if exit_code != "unsat":
                return self._result(
                    status="INCONCLUSIVE", prop=prop, bound_proven=None,
                    counterexample=None, model_spec=model_spec,
                    runtime_s=time.perf_counter() - t0, config=config,
                    notes=f"Timeout/unknown on output {i} of {n_outputs} (exit_code={exit_code!r})",
                )

        return self._result(
            status="PROVEN", prop=prop, bound_proven=0.0,
            counterexample=None, model_spec=model_spec,
            runtime_s=time.perf_counter() - t0, config=config,
            notes=f"All {n_outputs} outputs proven >= 0 (UNSAT for all queries)",
        )

    def _verify_conservation(
        self, onnx_path, model_spec, prop, config, t0,
        timeout, snc, num_workers, M,
    ) -> VerificationResult:
        from maraboupy import Marabou as Mb

        opts = self._make_options(timeout, snc, num_workers, Mb)
        coeff_list = prop.coefficients.tolist()
        lo_req = prop.target - prop.epsilon
        hi_req = prop.target + prop.epsilon

        # addInequality(vars, coeffs, scalar) adds:  sum(coeff_i * var_i) <= scalar
        # Query 1: can sum > hi_req?
        #   Equivalent to: -sum >= hi_req + 1e-9
        #   Encoded as: sum(-coeff_i * out_i) <= -(hi_req + 1e-9)
        net1 = Mb.read_onnx(onnx_path)
        self._add_input_bounds(net1, model_spec.input_lb, model_spec.input_ub)
        out1 = net1.outputVars[0].flatten().tolist()
        neg_coeff = [-c for c in coeff_list]
        net1.addInequality(out1, neg_coeff, -(hi_req + 1e-9))
        exit1, vals1, _ = net1.solve(options=opts, verbose=False)

        if exit1 == "sat":
            return self._result(
                status="COUNTEREXAMPLE", prop=prop, bound_proven=None,
                counterexample=self._extract_input(vals1, net1),
                model_spec=model_spec, runtime_s=time.perf_counter() - t0,
                config=config, notes=f"Conservation sum can exceed {hi_req} (SAT upper)",
            )
        timeout1 = exit1 != "unsat"

        # Query 2: can sum < lo_req?
        #   Encoded as: sum(coeff_i * out_i) <= lo_req - 1e-9
        net2 = Mb.read_onnx(onnx_path)
        self._add_input_bounds(net2, model_spec.input_lb, model_spec.input_ub)
        out2 = net2.outputVars[0].flatten().tolist()
        net2.addInequality(out2, coeff_list, lo_req - 1e-9)
        exit2, vals2, _ = net2.solve(options=opts, verbose=False)

        runtime_s = time.perf_counter() - t0

        if exit2 == "sat":
            return self._result(
                status="COUNTEREXAMPLE", prop=prop, bound_proven=None,
                counterexample=self._extract_input(vals2, net2),
                model_spec=model_spec, runtime_s=runtime_s, config=config,
                notes=f"Conservation sum can drop below {lo_req} (SAT lower)",
            )
        timeout2 = exit2 != "unsat"

        if timeout1 or timeout2:
            return self._result(
                status="INCONCLUSIVE", prop=prop, bound_proven=None,
                counterexample=None, model_spec=model_spec, runtime_s=runtime_s,
                config=config, notes="Timeout on one or both conservation queries",
            )

        return self._result(
            status="PROVEN", prop=prop, bound_proven=prop.epsilon,
            counterexample=None, model_spec=model_spec, runtime_s=runtime_s,
            config=config,
            notes=f"Both UNSAT: sum provably in [{lo_req}, {hi_req}] everywhere",
        )

    def _verify_range_bound(
        self, onnx_path, model_spec, prop, config, t0,
        timeout, snc, num_workers, M,
    ) -> VerificationResult:
        from maraboupy import Marabou as Mb

        opts = self._make_options(timeout, snc, num_workers, Mb)
        coeff_list = prop.coefficients.tolist()
        timeouts = []

        if prop.upper is not None:
            net = Mb.read_onnx(onnx_path)
            self._add_input_bounds(net, model_spec.input_lb, model_spec.input_ub)
            out_vars = net.outputVars[0].flatten().tolist()
            neg_coeff = [-c for c in coeff_list]
            net.addInequality(out_vars, neg_coeff, -(prop.upper + 1e-9))
            exit_c, values, _ = net.solve(options=opts, verbose=False)
            if exit_c == "sat":
                return self._result(
                    status="COUNTEREXAMPLE", prop=prop, bound_proven=None,
                    counterexample=self._extract_input(values, net),
                    model_spec=model_spec, runtime_s=time.perf_counter() - t0,
                    config=config, notes=f"Combination exceeds upper {prop.upper}",
                )
            timeouts.append(exit_c != "unsat")

        if prop.lower is not None:
            net = Mb.read_onnx(onnx_path)
            self._add_input_bounds(net, model_spec.input_lb, model_spec.input_ub)
            out_vars = net.outputVars[0].flatten().tolist()
            net.addInequality(out_vars, coeff_list, prop.lower - 1e-9)
            exit_c, values, _ = net.solve(options=opts, verbose=False)
            if exit_c == "sat":
                return self._result(
                    status="COUNTEREXAMPLE", prop=prop, bound_proven=None,
                    counterexample=self._extract_input(values, net),
                    model_spec=model_spec, runtime_s=time.perf_counter() - t0,
                    config=config, notes=f"Combination drops below lower {prop.lower}",
                )
            timeouts.append(exit_c != "unsat")

        runtime_s = time.perf_counter() - t0
        if any(timeouts):
            return self._result(
                status="INCONCLUSIVE", prop=prop, bound_proven=None,
                counterexample=None, model_spec=model_spec, runtime_s=runtime_s,
                config=config, notes="Timeout on one or more range bound queries",
            )
        return self._result(
            status="PROVEN", prop=prop, bound_proven=None,
            counterexample=None, model_spec=model_spec, runtime_s=runtime_s,
            config=config, notes="All range bound queries UNSAT: property proven",
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
            backend=MarabouBackend.name,
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
