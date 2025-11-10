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
        return float(value)
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
    required = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CTYPE1", "CTYPE2")
    if any(key not in header for key in required):
        return False
    radesys = str(header.get("RADESYS", "")).strip()
    if not radesys:
        return False
    ctype1 = str(header.get("CTYPE1", "")).upper()
    ctype2 = str(header.get("CTYPE2", "")).upper()
    if not any(token in ctype1 for token in ("RA", "GLON")):
        return False
    if not any(token in ctype2 for token in ("DEC", "GLAT")):
        return False
    cd_keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if any(key not in header for key in cd_keys):
        return False
    try:
        cd = np.array(
            [
                [float(header["CD1_1"]), float(header["CD1_2"])],
                [float(header["CD2_1"]), float(header["CD2_2"])],
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cd)):
        return False
    if abs(np.linalg.det(cd)) < 1e-12:
        return False
    for key in ("CDELT1", "CDELT2"):
        if key in header:
            try:
                if math.isclose(float(header[key]), 1.0, rel_tol=0, abs_tol=1e-9):
                    return False
            except (TypeError, ValueError):
                return False
    for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
        try:
            value = float(header[key])
        except (TypeError, ValueError):
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
                    )
        except Exception as exc:
            raise InvalidInputError(f"Unable to read FITS header: {exc}") from exc
    try:
        solution = _internal_solve_blind(fits_path, index_root, config=config)
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
    )


def near_solve(
    fits_path: str,
    index_root: str,
    *,
    config: Optional[InternalNearConfig] = None,
    log: Optional[Callable[[str], None]] = None,
    skip_if_valid: bool = True,
    fallback_to_blind: bool = True,
) -> BlindSolveResult:
    logger = log or _default_log
    start = time.perf_counter()
    fits_path = str(Path(fits_path).expanduser())
    index_root = str(Path(index_root).expanduser())
    logger(f"[ZENEAR] starting (index_root={index_root})")
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
                    )
        except Exception as exc:
            raise InvalidInputError(f"Unable to read FITS header: {exc}") from exc
    try:
        solution = _internal_solve_near(fits_path, index_root, config=config)
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
    )
    if solution.success or not fallback_to_blind:
        return result
    logger(f"[ZENEAR] near solve failed ({solution.message}); attempting blind fallback…")
    blind_result = blind_solve(
        fits_path,
        index_root,
        config=None,
        log=log,
        skip_if_valid=False,
    )
    prefix = f"near failed: {solution.message}"
    blind_message = blind_result["message"]
    blind_result["message"] = f"{prefix}; blind {blind_message}"
    return blind_result
