"""Abstract base class for verification backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model import ModelSpec
from ..property import Property
from ..result import VerificationResult


class Backend(ABC):
    """
    A verification backend wraps a specific solver.

    Implementors must set ``name`` and implement ``verify``.

    Scope
    -----
    Backends in this framework handle feedforward ReLU networks
    (Linear + ReLU stacks) and properties expressible as bounds on
    linear combinations of outputs.

    Out of scope (document as limitations, do not silently fail):
    * Derivative-based properties (PDE residuals, divergence-free fields)
    * Diffusion or other iterative samplers
    * Layer types not supported by the underlying solver
    """

    name: str

    @abstractmethod
    def verify(
        self,
        model_spec: ModelSpec,
        prop: Property,
        config: dict,
    ) -> VerificationResult:
        """
        Verify that *prop* holds for *model_spec* over its input region.

        Parameters
        ----------
        model_spec : ModelSpec
            The model and its input box.
        prop : Property
            The property to verify.
        config : dict
            Backend-specific options (e.g. ``{"method": "CROWN"}``).

        Returns
        -------
        VerificationResult
            status is one of PROVEN, COUNTEREXAMPLE, INCONCLUSIVE.
            Traceability fields (git_commit, seed, timestamp, config) are
            filled in by the runner; the backend populates the core fields.
        """
