"""Representation-mediated Intelligence research utilities."""

from .alignment import ProcrustesMap, fit_procrustes
from .extraction import extract_hf_activations, pool_token_activations

__all__ = [
    "ProcrustesMap",
    "fit_procrustes",
    "extract_hf_activations",
    "pool_token_activations",
]
__version__ = "0.1.0"
