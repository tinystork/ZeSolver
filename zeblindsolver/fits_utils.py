from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

try:
    from astropy.coordinates import Angle
    import astropy.units as u
except Exception:  # pragma: no cover - optional dependency guard
    Angle = None  # type: ignore
    u = None  # type: ignore


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
        raise ValueError("FITS HDU has no data to generate luminance")
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=0)
    elif arr.ndim == 2:
        bayer = hdu.header.get("BAYERPAT") or hdu.header.get("BAYERP")
        arr = _extract_bayer_green(np.asarray(arr), bayer)
    else:
        raise ValueError(f"Unsupported FITS data shape for luminance: {arr.shape}")
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


def parse_angle(value: object, *, is_ra: bool) -> Optional[float]:
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
        if Angle is None or u is None:
            return None
        try:
            angle = Angle(text, unit=u.hourangle if is_ra else u.deg)
            return float(angle.degree)
        except (ValueError, u.UnitsError):
            return None
