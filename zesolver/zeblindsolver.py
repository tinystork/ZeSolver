# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : GPL V3 (voir pyproject.toml / repository metadata)               ║
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

"""Helper utilities around the internal zeblind solver (no ASTAP executable is invoked)."""
from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, TypedDict

import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.io import fits

from zeblindsolver.fits_utils import (
    estimate_scale_and_fov as _core_estimate_scale_and_fov,
    to_luminance_for_solve as _core_to_luminance_for_solve,
)
from zeblindsolver.metadata_solver import NearSolveConfig as InternalNearConfig, solve_near as _internal_solve_near
from zeblindsolver.near_catalog_provider import NearCatalogProvider
from zeblindsolver.zeblindsolver import SolveConfig as InternalBlindConfig, solve_blind as _internal_solve_blind


class BlindSolveResult(TypedDict):
    success: bool
    message: str
    elapsed_sec: float
    tried_dbs: List[str]
    used_db: Optional[str]
    wrote_wcs: bool
    updated_keywords: dict[str, Any]
    output_path: str
    stats: dict[str, Any]


class BlindSolverRuntimeError(RuntimeError):
    pass


class InvalidInputError(BlindSolverRuntimeError):
    pass


def _default_log(message: str) -> None:
    print(message, flush=True)


def _normalize_header_angle(value: object, *, is_ra: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        val = float(value)
        if is_ra and math.isfinite(val) and abs(val) <= 24.0:
            return val * 15.0
        return val
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        try:
            angle = Angle(text, unit=u.hourangle if is_ra else u.deg)
            return float(angle.degree)
        except (ValueError, u.UnitsError):
            return None


def has_valid_wcs(header: fits.Header) -> bool:
    """Return True when header contains a ZeMosaic-compatible celestial WCS."""

    required = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CTYPE1", "CTYPE2")
    if any(key not in header for key in required):
        return False

    ctype1 = str(header.get("CTYPE1", "")).upper()
    ctype2 = str(header.get("CTYPE2", "")).upper()
    if not any(token in ctype1 for token in ("RA", "LON", "GLON")):
        return False
    if not any(token in ctype2 for token in ("DEC", "LAT", "GLAT")):
        return False

    def _f(key: str) -> float | None:
        try:
            return float(header[key])
        except Exception:
            return None

    cd = None
    cd_keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if all(k in header for k in cd_keys):
        vals = [_f(k) for k in cd_keys]
        if any(v is None for v in vals):
            return False
        cd = np.array([[vals[0], vals[1]], [vals[2], vals[3]]], dtype=float)
    elif all(k in header for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2")):
        vals = [_f(k) for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2")]
        if any(v is None for v in vals):
            return False
        pc = np.array([[vals[0], vals[1]], [vals[2], vals[3]]], dtype=float)
        cd = pc @ np.diag([vals[4], vals[5]])
    else:
        return False

    if not np.all(np.isfinite(cd)):
        return False
    try:
        det = float(np.linalg.det(cd))
    except Exception:
        return False
    if (not math.isfinite(det)) or abs(det) < 1e-16:
        return False

    scales_arcsec = np.abs(np.sqrt(np.sum(cd ** 2, axis=0))) * 3600.0
    finite = scales_arcsec[np.isfinite(scales_arcsec)]
    if finite.size == 0:
        return False
    if float(np.nanmin(finite)) < 0.3 or float(np.nanmax(finite)) > 15.0:
        return False

    for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
        try:
            value = float(header[key])
        except Exception:
            return False
        if not math.isfinite(value):
            return False

    return True


def sanitize_wcs(header: fits.Header) -> int:
    removed = 0
    keys = [
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CDELT1",
        "CDELT2",
        "CROTA1",
        "CROTA2",
        "RADESYS",
        "EQUINOX",
        "LONPOLE",
        "LATPOLE",
    ]
    for key in keys:
        if key in header:
            del header[key]
            removed += 1
    return removed


def estimate_scale_and_fov(
    header: fits.Header,
    width: int,
    height: int,
) -> tuple[Optional[float], tuple[Optional[float], Optional[float]]]:
    return _core_estimate_scale_and_fov(header, width, height)


def to_luminance_for_solve(hdu: fits.PrimaryHDU) -> np.ndarray:
    try:
        return _core_to_luminance_for_solve(hdu)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc


def blind_solve(
    fits_path: str,
    index_root: str,
    *,
    config: Optional[InternalBlindConfig] = None,
    log: Optional[Callable[[str], None]] = None,
    skip_if_valid: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
    prep_cache: Optional[dict[str, Any]] = None,
) -> BlindSolveResult:
    logger = log or _default_log
    start = time.perf_counter()
    fits_path = str(Path(fits_path).expanduser())
    index_root = str(Path(index_root).expanduser())
    logger(f"[ZEBLIND] starting (index_root={index_root})")
    if skip_if_valid:
        try:
            with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
                if has_valid_wcs(hdul[0].header):
                    elapsed = time.perf_counter() - start
                    message = "skipped: already valid"
                    logger(f"[ZEBLIND] {message}")
                    return BlindSolveResult(
                        success=True,
                        message=message,
                        elapsed_sec=elapsed,
                        tried_dbs=[index_root],
                        used_db=None,
                        wrote_wcs=False,
                        updated_keywords={},
                        output_path=fits_path,
                        stats={},
                    )
        except Exception as exc:
            raise InvalidInputError(f"Unable to read FITS header: {exc}") from exc
    try:
        if cancel_check and cancel_check():
            raise BlindSolverRuntimeError("cancelled")
        use_external_retry = bool(
            config is not None
            and getattr(config, "blind_astrometry_external_image2xy_enabled", False)
            and getattr(config, "blind_astrometry_external_image2xy_after_internal_fail_enabled", False)
        )
        first_config = config
        if use_external_retry:
            internal_budget_s = max(
                0.0,
                float(getattr(config, "blind_astrometry_external_image2xy_internal_first_budget_s", 0.0) or 0.0),
            )
            if internal_budget_s > 0.0:
                current_budget_s = max(
                    0.0,
                    float(getattr(config, "blind_global_hard_budget_s", 0.0) or 0.0),
                )
                effective_budget_s = (
                    min(current_budget_s, internal_budget_s)
                    if current_budget_s > 0.0
                    else internal_budget_s
                )
                first_config = replace(
                    config,
                    blind_astrometry_external_image2xy_enabled=False,
                    blind_global_hard_budget_s=effective_budget_s,
                )
            else:
                first_config = replace(config, blind_astrometry_external_image2xy_enabled=False)
        solution = _internal_solve_blind(
            fits_path,
            index_root,
            config=first_config,
            cancel_check=cancel_check,
            prep_cache=prep_cache,
        )
        if use_external_retry and not solution.success:
            first_stats = dict(solution.stats or {})
            logger("[ZEBLIND] internal detector failed; retrying with external image2xy")
            if cancel_check and cancel_check():
                raise BlindSolverRuntimeError("cancelled")
            retry_solution = _internal_solve_blind(
                fits_path,
                index_root,
                config=replace(config, blind_astrometry_external_image2xy_enabled=True),
                cancel_check=cancel_check,
                prep_cache={},
            )
            retry_stats = dict(retry_solution.stats or {})
            retry_stats["external_image2xy_after_internal_fail"] = {
                "enabled": True,
                "internal_success": bool(solution.success),
                "internal_message": solution.message,
                "internal_fail_stage": first_stats.get("fail_stage"),
                "internal_best_fail_inliers": first_stats.get("best_fail_inliers"),
                "internal_budget_s": getattr(first_config, "blind_global_hard_budget_s", None),
                "external_success": bool(retry_solution.success),
            }
            retry_solution.stats = retry_stats
            solution = retry_solution
        elif use_external_retry:
            stats = dict(solution.stats or {})
            stats["external_image2xy_after_internal_fail"] = {
                "enabled": True,
                "internal_success": True,
                "external_attempted": False,
            }
            solution.stats = stats
    except InvalidInputError:
        raise
    except Exception as exc:
        raise BlindSolverRuntimeError(f"Internal blind solver failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    message = solution.message or ""
    status = "succeeded" if solution.success else "failed"
    logger(f"[ZEBLIND] {status}: {message}")
    return BlindSolveResult(
        success=solution.success,
        message=message,
        elapsed_sec=elapsed,
        tried_dbs=[index_root],
        used_db=solution.tile_key,
        wrote_wcs=solution.success,
        updated_keywords=solution.header_updates,
        output_path=fits_path,
        stats=dict(solution.stats or {}),
    )


def near_solve(
    fits_path: str,
    index_root: str | None,
    *,
    config: Optional[InternalNearConfig] = None,
    catalog_provider: NearCatalogProvider | None = None,
    log: Optional[Callable[[str], None]] = None,
    skip_if_valid: bool = True,
    fallback_to_blind: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BlindSolveResult:
    logger = log or _default_log
    start = time.perf_counter()
    fits_path = str(Path(fits_path).expanduser())
    index_root_text = str(Path(index_root).expanduser()) if index_root is not None else ""
    strict_flag = bool(getattr(config, "astap_iso_strict", True)) if config is not None else True
    provider_kind = getattr(catalog_provider, "kind", "legacy_index")
    logger(
        "[ZENEAR] starting "
        f"(index_root={index_root_text or '<provider>'}, "
        f"astap_iso_strict={str(strict_flag).lower()}, "
        f"near_catalog_provider={provider_kind})"
    )
    if skip_if_valid:
        try:
            with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
                if has_valid_wcs(hdul[0].header):
                    elapsed = time.perf_counter() - start
                    message = "skipped: already valid"
                    logger(f"[ZENEAR] {message}")
                    return BlindSolveResult(
                        success=True,
                        message=message,
                        elapsed_sec=elapsed,
                        tried_dbs=[index_root_text] if index_root_text else [provider_kind],
                        used_db=None,
                        wrote_wcs=False,
                        updated_keywords={},
                        output_path=fits_path,
                        stats={},
                    )
        except Exception as exc:
            raise InvalidInputError(f"Unable to read FITS header: {exc}") from exc
    try:
        if cancel_check and cancel_check():
            raise BlindSolverRuntimeError("cancelled")
        solution = _internal_solve_near(
            fits_path,
            index_root_text or None,
            config=config,
            catalog_provider=catalog_provider,
            cancel_check=cancel_check,
        )
    except InvalidInputError:
        raise
    except Exception as exc:
        raise BlindSolverRuntimeError(f"Internal near solver failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    status_text = "succeeded" if solution.success else "failed"
    logger(f"[ZENEAR] {status_text}: {solution.message}")
    orchestration_trace = {
        "fallback_to_blind_requested": bool(fallback_to_blind),
        "blind_fallback_attempted": False,
        "blind_fallback_skipped_reason": "",
    }
    result = BlindSolveResult(
        success=solution.success,
        message=solution.message,
        elapsed_sec=elapsed,
        tried_dbs=[index_root_text] if index_root_text else [provider_kind],
        used_db=solution.tile_key,
        wrote_wcs=solution.success,
        updated_keywords=solution.header_updates,
        output_path=fits_path,
        stats=dict(solution.stats or {}),
    )
    result.setdefault("stats", {})
    if isinstance(result["stats"], dict):
        result["stats"]["near_orchestration"] = dict(orchestration_trace)
    if solution.success or not fallback_to_blind:
        if not solution.success and not fallback_to_blind and isinstance(result.get("stats"), dict):
            result["stats"]["near_orchestration"]["blind_fallback_skipped_reason"] = "fallback_disabled"
        return result
    if cancel_check and cancel_check():
        return BlindSolveResult(
            success=False,
            message="cancelled",
            elapsed_sec=elapsed,
            tried_dbs=[index_root_text] if index_root_text else [provider_kind],
            used_db=None,
            wrote_wcs=False,
            updated_keywords={},
            output_path=fits_path,
            stats={},
        )
    logger(f"[ZENEAR] near solve failed ({solution.message}); 4D-only chain required, no historical fallback")
    orchestration_trace["blind_fallback_skipped_reason"] = "BLIND4D_CONFIGURATION_REQUIRED"
    return BlindSolveResult(
        success=False,
        message=f"near failed: {solution.message}; BLIND4D_CONFIGURATION_REQUIRED",
        elapsed_sec=elapsed,
        tried_dbs=[index_root_text] if index_root_text else [provider_kind],
        used_db=None,
        wrote_wcs=False,
        updated_keywords={},
        output_path=fits_path,
        stats={
            "near_orchestration": dict(orchestration_trace),
            "historical_blind_called": False,
            "final_status": "BLIND4D_CONFIGURATION_REQUIRED",
        },
    )
