from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import Sip, WCS
from astropy.wcs.utils import fit_wcs_from_points

from .matcher import SimilarityTransform


def _apply_similarity(transform: "SimilarityTransform", xy: tuple[float, float]) -> tuple[float, float]:
    rot_scale = transform.scale * np.exp(1j * transform.rotation)
    comp = complex(xy[0], xy[1])
    result = rot_scale * comp + complex(*transform.translation)
    return (float(result.real), float(result.imag))


def tan_from_similarity(
    transform: "SimilarityTransform",
    image_shape: tuple[int, int],
    *,
    center_pixel: tuple[float, float] | None = None,
    tile_center: tuple[float, float] | None = None,
) -> WCS:
    if center_pixel is None:
        center_pixel = (image_shape[1] / 2.0, image_shape[0] / 2.0)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crpix = [center_pixel[0], center_pixel[1]]
    theta = transform.rotation
    scale = transform.scale
    # Build the linear map from pixels to tangent-plane degrees
    # A = scale * [[cos, -sin], [sin, cos]]
    a11 = scale * np.cos(theta)
    a12 = -scale * np.sin(theta)
    a21 = scale * np.sin(theta)
    a22 = scale * np.cos(theta)
    A = np.array([[a11, a12], [a21, a22]])
    if tile_center is not None:
        wcs.wcs.crval = [float(tile_center[0]), float(tile_center[1])]
        # Incorporate the similarity translation into CRPIX so that
        # A * (pixel - CRPIX_0) equals the plane offsets from the transform
        t = np.array([float(transform.translation[0]), float(transform.translation[1])])
        try:
            crpix0 = -np.linalg.inv(A) @ t
        except np.linalg.LinAlgError:
            crpix0 = np.array([0.0, 0.0])
        # FITS CRPIX is 1-based; astropy with origin=0 uses (pixel - (CRPIX-1))
        wcs.wcs.crpix = [float(crpix0[0] + 1.0), float(crpix0[1] + 1.0)]
    else:
        # Fallback: keep previous behavior anchored at provided center_pixel
        wcs.wcs.crpix = [center_pixel[0], center_pixel[1]]
        zero_pixel = (center_pixel[0] - 1.0, center_pixel[1] - 1.0)
        ra_dec = _apply_similarity(transform, zero_pixel)
        wcs.wcs.crval = list(ra_dec)
    wcs.wcs.cd = A
    wcs.wcs.radesys = "ICRS"
    return wcs


def _stats_from_wcs(wcs: WCS, matches: np.ndarray) -> dict[str, float | int]:
    world = wcs.wcs_pix2world(matches[:, :2], 0)
    residuals = np.linalg.norm(world - matches[:, 2:], axis=1)
    cd = getattr(wcs.wcs, "cd", None)
    if cd is not None and np.linalg.det(cd) != 0:
        scale = abs(np.linalg.det(cd)) ** 0.5
    else:
        scale = 1.0
    rms_deg = float(np.sqrt(np.mean(residuals ** 2)))
    rms_px = rms_deg / max(scale, 1e-8)
    return {"rms_px": rms_px, "inliers": len(matches)}


def fit_wcs_tan(matches: np.ndarray, *, robust: bool = True) -> tuple[WCS, dict[str, Any]]:
    if matches.size == 0:
        raise ValueError("no matches supplied")
    pixels = matches[:, :2]
    ras = matches[:, 2]
    decs = matches[:, 3]
    center_x = float(np.mean(pixels[:, 0]))
    center_y = float(np.mean(pixels[:, 1]))
    dx = pixels[:, 0] - center_x
    dy = pixels[:, 1] - center_y
    design = np.column_stack((dx, dy, np.ones_like(dx)))
    ra_params, *_ = np.linalg.lstsq(design, ras, rcond=None)
    dec_params, *_ = np.linalg.lstsq(design, decs, rcond=None)
    cd = np.array([[ra_params[0], ra_params[1]], [dec_params[0], dec_params[1]]], dtype=float)
    crval = [float(ra_params[2]), float(dec_params[2])]
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crpix = [center_x + 1.0, center_y + 1.0]
    wcs.wcs.crval = crval
    wcs.wcs.cd = cd
    wcs.wcs.radesys = "ICRS"
    stats = _stats_from_wcs(wcs, matches)
    return wcs, stats


def fit_wcs_sip(matches: np.ndarray, *, robust: bool = True, order: int = 2) -> tuple[WCS, dict[str, Any]]:
    if matches.size == 0:
        raise ValueError("no matches supplied")
    pixels = matches[:, :2]
    world = matches[:, 2:]
    coords = SkyCoord(ra=world[:, 0] * u.deg, dec=world[:, 1] * u.deg, frame="icrs")
    wcs = fit_wcs_from_points((pixels[:, 0], pixels[:, 1]), coords, sip_degree=order, projection="TAN")
    stats = _stats_from_wcs(wcs, matches)
    return wcs, stats


def needs_sip(wcs: WCS, stats: dict[str, Any], fov_deg: float) -> bool:
    rms_px = stats.get("rms_px", float("inf"))
    inliers = stats.get("inliers", 0)
    return fov_deg >= 2.0 and rms_px > 0.9 and inliers >= 50
