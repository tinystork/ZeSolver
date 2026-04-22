"""
Top-level package for the ZeSolver WCS project.

Only the catalogue reader is functional today. Detection, solver, and CLI wiring will follow.
"""

__version__ = "1.0.0"

from .catalog290 import CatalogDB, CatalogFamilySpec, CatalogTile, StarBlock

__all__ = [
    "__version__",
    "CatalogDB",
    "CatalogFamilySpec",
    "CatalogTile",
    "StarBlock",
]
