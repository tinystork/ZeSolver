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

"""ZeSolver helper package.

The public, stable contract lives under :mod:`zesolver.api.v1`.  This package
``__init__`` keeps its historical re-exports available *lazily* so that
``import zesolver`` and ``import zesolver.api.v1`` stay lightweight: importing
this package does not import Qt, CuPy, :mod:`zesolver.gui_pipeline`, or the
heavy solver engine.  The heavy modules are only imported when the corresponding
historical name is actually accessed.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def _load_package_version(default: str = "0.0.dev") -> str:
    """Return the project version without adding a second source of truth."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str((data.get("project") or {}).get("version") or "").strip()
        if version:
            return version
    except Exception:
        pass
    try:
        in_project = False
        for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line == "[project]":
                in_project = True
                continue
            if in_project and line.startswith("["):
                break
            if in_project and line.startswith("version") and "=" in line:
                value = line.split("=", 1)[1].strip().strip('"\'')
                if value:
                    return value
    except Exception:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version

        try:
            return pkg_version("ZeSolver")
        except PackageNotFoundError:
            return default
    except Exception:
        return default


__version__ = _load_package_version()


# ---------------------------------------------------------------------------
# Lazy historical re-exports
# ---------------------------------------------------------------------------
# Each entry maps an attribute name to the (module, attribute) pair that
# provides it.  The heavy solver engine is imported only on first access of the
# name, keeping ``import zesolver`` lightweight.
_LAZY_REEXPORTS: dict[str, tuple[str, str]] = {
    # zesolver.blindindex
    "BlindIndex": ("zesolver.blindindex", "BlindIndex"),
    "BlindIndexCandidate": ("zesolver.blindindex", "BlindIndexCandidate"),
    "ObservedQuad": ("zesolver.blindindex", "ObservedQuad"),
    # zesolver.settings_store
    "DEFAULT_FOV_DEG": ("zesolver.settings_store", "DEFAULT_FOV_DEG"),
    "DEFAULT_SEARCH_RADIUS_ATTEMPTS": ("zesolver.settings_store", "DEFAULT_SEARCH_RADIUS_ATTEMPTS"),
    "DEFAULT_SEARCH_RADIUS_SCALE": ("zesolver.settings_store", "DEFAULT_SEARCH_RADIUS_SCALE"),
    "PersistentSettings": ("zesolver.settings_store", "PersistentSettings"),
    "SETTINGS_PATH": ("zesolver.settings_store", "SETTINGS_PATH"),
    "load_persistent_settings": ("zesolver.settings_store", "load_persistent_settings"),
    "normalize_ui_theme": ("zesolver.settings_store", "normalize_ui_theme"),
    "save_persistent_settings": ("zesolver.settings_store", "save_persistent_settings"),
    # zesolver.zeblindsolver
    "BlindSolveResult": ("zesolver.zeblindsolver", "BlindSolveResult"),
    "BlindSolverRuntimeError": ("zesolver.zeblindsolver", "BlindSolverRuntimeError"),
    "blind_solve": ("zesolver.zeblindsolver", "blind_solve"),
    "estimate_scale_and_fov": ("zesolver.zeblindsolver", "estimate_scale_and_fov"),
    "has_valid_wcs": ("zesolver.zeblindsolver", "has_valid_wcs"),
    "near_solve": ("zesolver.zeblindsolver", "near_solve"),
    "sanitize_wcs": ("zesolver.zeblindsolver", "sanitize_wcs"),
    "to_luminance_for_solve": ("zesolver.zeblindsolver", "to_luminance_for_solve"),
    # zeblindsolver.metadata_solver
    "NearSolveConfig": ("zeblindsolver.metadata_solver", "NearSolveConfig"),
}


def __getattr__(name: str):
    spec = _LAZY_REEXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value  # cache for subsequent access
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_REEXPORTS))


__all__ = [
    "BlindSolveResult",
    "BlindSolverRuntimeError",
    "blind_solve",
    "near_solve",
    "estimate_scale_and_fov",
    "has_valid_wcs",
    "sanitize_wcs",
    "to_luminance_for_solve",
    "BlindIndex",
    "BlindIndexCandidate",
    "ObservedQuad",
    "NearSolveConfig",
    "PersistentSettings",
    "load_persistent_settings",
    "normalize_ui_theme",
    "save_persistent_settings",
    "SETTINGS_PATH",
    "DEFAULT_FOV_DEG",
    "DEFAULT_SEARCH_RADIUS_SCALE",
    "DEFAULT_SEARCH_RADIUS_ATTEMPTS",
    "__version__",
]
