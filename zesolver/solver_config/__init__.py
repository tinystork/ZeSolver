"""Shared solver configuration builders.

This package is intentionally free of GUI and root-entrypoint imports so core
adapters can build legacy-compatible engine configs without loading
``zesolver.py``.
"""

from .blind import build_blind_solve_config, ensure_loaded_4d_manifest
from .compatibility import BlindConfigInputs, build_blind_config_inputs

__all__ = [
    "BlindConfigInputs",
    "build_blind_config_inputs",
    "build_blind_solve_config",
    "ensure_loaded_4d_manifest",
]
