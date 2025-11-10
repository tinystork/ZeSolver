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
    focal_len = header.get("FOCALLEN") or header.get("FOCLEN") or header.get("FOCALLENGTH")
    if focal_len is None:
        return None, (None, None)
    try:
        focal_len = float(focal_len)
    except (TypeError, ValueError):
        return None, (None, None)
    if focal_len <= 0:
        return None, (None, None)
    pix_x = header.get("XPIXSZ") or header.get("PIXSIZE1")
    pix_y = header.get("YPIXSZ") or header.get("PIXSIZE2")
    if pix_x is None or pix_y is None:
        return None, (None, None)
    try:
        pix_um = float((float(pix_x) + float(pix_y)) / 2.0)
    except (TypeError, ValueError):
        return None, (None, None)
    if pix_um <= 0:
        return None, (None, None)
    scale = 206.265 * pix_um / float(focal_len)
    fov_x = scale * width / 3600.0
    fov_y = scale * height / 3600.0
    return float(scale), (float(fov_x), float(fov_y))


def _extract_bayer_green(data: np.ndarray, pattern: str) -> np.ndarray:
    pattern = (pattern or "").strip().upper()
    if len(pattern) != 4 or data.ndim != 2:
        return data
    mask = np.zeros_like(data, dtype=bool)
    mapping = {(0, 0): pattern[0], (0, 1): pattern[1], (1, 0): pattern[2], (1, 1): pattern[3]}
    for (dy, dx), channel in mapping.items():
        if channel == "G":
            mask[dy::2, dx::2] = True
    if not mask.any():
        return data
    result = np.array(data, copy=True)
    green_values = result[mask]
    if green_values.size == 0:
        return data
    mean_green = float(np.mean(green_values))
    if math.isfinite(mean_green):
        result[~mask] = mean_green
    return result


def to_luminance_for_solve(hdu: fits.PrimaryHDU) -> np.ndarray:
    data = hdu.data
    if data is None:
        raise InvalidInputError("FITS HDU has no data to generate luminance")
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=0)
    elif arr.ndim == 2:
        bayer = hdu.header.get("BAYERPAT") or hdu.header.get("BAYERP")
        arr = _extract_bayer_green(np.asarray(arr), bayer)
    else:
        raise InvalidInputError(f"Unsupported FITS data shape for luminance: {arr.shape}")
    arr = np.asarray(arr, dtype=np.float32)
    try:
        min_val = float(np.nanmin(arr))
    except (ValueError, FloatingPointError):
        min_val = 0.0
    if math.isfinite(min_val):
        arr -= min_val
    arr = np.nan_to_num(arr, copy=False)
    high = float(np.nanpercentile(arr, 99.5)) if arr.size else 1.0
    if not math.isfinite(high) or high <= 0:
        high = 1.0
    arr = np.clip(arr / high, 0.0, 1.0)
    return arr.astype(np.float32, copy=False)


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
