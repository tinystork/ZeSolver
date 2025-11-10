"""
Top-level package for the ZeSolver WCS project.

Only the catalogue reader is functional today. Detection, solver, and CLI wiring will follow.
"""

from .catalog290 import CatalogDB, CatalogFamilySpec, CatalogTile, StarBlock

__all__ = [
    "CatalogDB",
    "CatalogFamilySpec",
    "CatalogTile",
    "StarBlock",
]
