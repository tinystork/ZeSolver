from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def project_tan(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_ra: float,
    center_dec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project spherical coordinates to a tangent plane centered on (center_ra, center_dec)."""
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    ra0_rad = math.radians(center_ra)
    dec0_rad = math.radians(center_dec)
    dra = (ra_rad - ra0_rad + math.pi) % (2 * math.pi) - math.pi
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec0 = math.sin(dec0_rad)
    cos_dec0 = math.cos(dec0_rad)
    cos_dra = np.cos(dra)
    sin_dra = np.sin(dra)
    cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    safe = cosc > 1e-8
    x = np.empty_like(ra_rad)
    y = np.empty_like(dec_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        x[~safe] = np.nan
        y[~safe] = np.nan
        x[safe] = (cos_dec[safe] * sin_dra[safe]) / cosc[safe]
        y[safe] = (cos_dec0 * sin_dec[safe] - sin_dec0 * cos_dec[safe] * cos_dra[safe]) / cosc[safe]
    return np.degrees(x), np.degrees(y)
