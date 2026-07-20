# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : MIT (voir pyproject.toml / repository metadata)               ║
# ║                                                                                   ║
# ║ Remerciements amont :                                                             ║
# ║ - ASTAP, par Han Kleijn                                                           ║
# ║ - Astrometry.net, par Dustin Lang, David W. Hogg, Keir Mierle, et al.            ║
# ║                                                                                   ║
# ║ Description FR :                                                                  ║
# ║ Ce code sert à transformer des nuages de photons en solutions WCS et en images   ║
# ║ astronomiques exploitables. Merci de créditer les auteurs et projets amont lors   ║
# ║ de toute réutilisation.                                                           ║
# ║                                                                                   ║
# ║ EN Description:                                                                    ║
# ║ This code helps turn clouds of photons into usable WCS solutions and astronomical ║
# ║ imagery outputs. Please credit both project authors and upstream references when  ║
# ║ reusing this work.                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
# """

"""Convenience exports for the ZeSolver blind solver stack (lazy).

This module avoids importing heavy submodules at import time to prevent
`python -m zeblindsolver.<submodule>` from triggering runpy warnings on
Windows and to keep import side‑effects minimal. Public names are re‑exported
on first access via module `__getattr__` (PEP 562).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TileMeta",
    "iter_db_tiles",
    "load_tile_stars",
    "build_index_from_astap",
    "build_4d_index_from_astap",
    "Astap4DBuildConfig",
    "AstapTileMaterializationConfig",
    "materialize_astap_tile_for_4d",
    "zebuildindex",
    "solve_blind",
    "zeblindsolve",
    "WcsSolution",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy re‑exports
    if name in {"TileMeta", "iter_db_tiles", "load_tile_stars"}:
        from .astap_db_reader import TileMeta, iter_tiles as iter_db_tiles, load_tile_stars
        mapping = {
            "TileMeta": TileMeta,
            "iter_db_tiles": iter_db_tiles,
            "load_tile_stars": load_tile_stars,
        }
        return mapping[name]
    if name in {"build_index_from_astap", "zebuildindex"}:
        from . import db_convert as _db
        mapping = {
            "build_index_from_astap": _db.build_index_from_astap,
            "zebuildindex": getattr(_db, "zebuildindex", _db.main),
        }
        return mapping[name]
    if name in {"build_4d_index_from_astap", "Astap4DBuildConfig", "AstapTileMaterializationConfig", "materialize_astap_tile_for_4d"}:
        from . import astap_4d_builder as _direct
        mapping = {
            "build_4d_index_from_astap": _direct.build_4d_index_from_astap,
            "Astap4DBuildConfig": _direct.Astap4DBuildConfig,
            "AstapTileMaterializationConfig": _direct.AstapTileMaterializationConfig,
            "materialize_astap_tile_for_4d": _direct.materialize_astap_tile_for_4d,
        }
        return mapping[name]
    if name in {"solve_blind", "zeblindsolve", "WcsSolution"}:
        from . import zeblindsolver as _core
        mapping = {
            "solve_blind": _core.solve_blind,
            "zeblindsolve": _core.zeblindsolve,
            "WcsSolution": _core.WcsSolution,
        }
        return mapping[name]
    raise AttributeError(name)


def __dir__() -> list[str]:  # helps IDEs
    return sorted(set(globals().keys()) | set(__all__))
