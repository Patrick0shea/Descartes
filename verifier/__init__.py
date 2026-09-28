"""
verifier — simulator-agnostic formal verification framework.

Public API
----------
from verifier import ModelSpec, NonNegativity, Conservation, RangeBound
from verifier import LiRPABackend, MarabouBackend
from verifier import verify, save_result, falsify
"""

from .model import ModelSpec
from .property import Conservation, NonNegativity, Property, RangeBound
from .result import FalsificationResult, VerificationResult
from .runner import load_config, save_result, verify
from .falsifier import falsify
from .backend import LiRPABackend, MarabouBackend

__all__ = [
    "ModelSpec",
    "Property",
    "NonNegativity",
    "Conservation",
    "RangeBound",
    "VerificationResult",
    "FalsificationResult",
    "LiRPABackend",
    "MarabouBackend",
    "verify",
    "save_result",
    "load_config",
    "falsify",
]
