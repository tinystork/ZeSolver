"""Helper utilities around the internal zeblind solver (no ASTAP executable is invoked)."""
from __future__ import annotations

import math
import time
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
        solution = _internal_solve_blind(
            fits_path,
            index_root,
            config=config,
            cancel_check=cancel_check,
            prep_cache=prep_cache,
        )
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
    index_root: str,
    *,
    config: Optional[InternalNearConfig] = None,
    log: Optional[Callable[[str], None]] = None,
    skip_if_valid: bool = True,
    fallback_to_blind: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BlindSolveResult:
    logger = log or _default_log
    start = time.perf_counter()
    fits_path = str(Path(fits_path).expanduser())
    index_root = str(Path(index_root).expanduser())
    strict_flag = bool(getattr(config, "astap_iso_strict", True)) if config is not None else True
    logger(f"[ZENEAR] starting (index_root={index_root}, astap_iso_strict={str(strict_flag).lower()})")
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
        solution = _internal_solve_near(fits_path, index_root, config=config, cancel_check=cancel_check)
    except InvalidInputError:
        raise
    except Exception as exc:
        raise BlindSolverRuntimeError(f"Internal near solver failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    status_text = "succeeded" if solution.success else "failed"
    logger(f"[ZENEAR] {status_text}: {solution.message}")
    result = BlindSolveResult(
        success=solution.success,
        message=solution.message,
        elapsed_sec=elapsed,
        tried_dbs=[index_root],
        used_db=solution.tile_key,
        wrote_wcs=solution.success,
        updated_keywords=solution.header_updates,
        output_path=fits_path,
        stats=dict(solution.stats or {}),
    )
    if solution.success or not fallback_to_blind:
        return result
    if cancel_check and cancel_check():
        return BlindSolveResult(
            success=False,
            message="cancelled",
            elapsed_sec=elapsed,
            tried_dbs=[index_root],
            used_db=None,
            wrote_wcs=False,
            updated_keywords={},
            output_path=fits_path,
            stats={},
        )
    logger(f"[ZENEAR] near solve failed ({solution.message}); attempting blind fallback…")

    blind_cfg = InternalBlindConfig()
    try:
        with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
            h = hdul[0].header
            data = hdul[0].data
            hgt = int(getattr(data, "shape", [0, 0])[0]) if data is not None else 0
            wdt = int(getattr(data, "shape", [0, 0])[1]) if data is not None and len(getattr(data, "shape", [])) >= 2 else 0
            ra_hint = (
                _normalize_header_angle(h.get("RA"), is_ra=True)
                or _normalize_header_angle(h.get("OBJCTRA"), is_ra=True)
                or _normalize_header_angle(h.get("CRVAL1"), is_ra=True)
            )
            dec_hint = (
                _normalize_header_angle(h.get("DEC"), is_ra=False)
                or _normalize_header_angle(h.get("OBJCTDEC"), is_ra=False)
                or _normalize_header_angle(h.get("CRVAL2"), is_ra=False)
            )
            blind_cfg.ra_hint_deg = ra_hint
            blind_cfg.dec_hint_deg = dec_hint
            if wdt > 0 and hgt > 0:
                scale_arcsec, (fov_x, fov_y) = _core_estimate_scale_and_fov(h, wdt, hgt)
                if scale_arcsec is not None:
                    blind_cfg.pixel_scale_arcsec = float(scale_arcsec)
                fovs = [v for v in (fov_x, fov_y) if v is not None]
                if fovs:
                    blind_cfg.radius_hint_deg = max(0.5, 0.8 * float(max(fovs)))
    except Exception:
        pass

    # Relax inlier threshold for fallback from near path while keeping geometric guards.
    blind_cfg.quality_inliers = max(12, int(getattr(blind_cfg, "quality_inliers", 40) or 40) // 2)
    blind_cfg.quality_rms = min(1.5, float(getattr(blind_cfg, "quality_rms", 1.2) or 1.2))

    blind_result = blind_solve(
        fits_path,
        index_root,
        config=blind_cfg,
        log=log,
        skip_if_valid=False,
        cancel_check=cancel_check,
    )
    prefix = f"near failed: {solution.message}"
    blind_message = blind_result["message"]
    blind_result["message"] = f"{prefix}; blind {blind_message}"
    return blind_result
