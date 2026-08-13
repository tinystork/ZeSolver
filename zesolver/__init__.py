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

"""ZeSolver helper package."""

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

from .blindindex import BlindIndex, BlindIndexCandidate, ObservedQuad
from .settings_store import (
    DEFAULT_FOV_DEG,
    DEFAULT_SEARCH_RADIUS_ATTEMPTS,
    DEFAULT_SEARCH_RADIUS_SCALE,
    PersistentSettings,
    SETTINGS_PATH,
    load_persistent_settings,
    normalize_ui_theme,
    save_persistent_settings,
)
from .zeblindsolver import (
    BlindSolveResult,
    BlindSolverRuntimeError,
    blind_solve,
    estimate_scale_and_fov,
    has_valid_wcs,
    near_solve,
    sanitize_wcs,
    to_luminance_for_solve,
)
from zeblindsolver.metadata_solver import NearSolveConfig

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
