from __future__ import annotations

from typing import Mapping

import numpy as np
from astropy.wcs import InvalidTransformError, WCS


def _extract_cd_matrix(wcs: WCS) -> np.ndarray | None:
    try:
        cd_attr = getattr(wcs.wcs, "cd", None)
    except Exception:
        cd_attr = None
    if cd_attr is not None:
        cd = np.asarray(cd_attr, dtype=np.float64)
        if cd.ndim == 2 and cd.shape[0] >= 2 and cd.shape[1] >= 2:
            return cd[:2, :2]
    try:
        pc_attr = getattr(wcs.wcs, "pc", None)
        cdelt_attr = getattr(wcs.wcs, "cdelt", None)
    except Exception:
        pc_attr = None
        cdelt_attr = None
    if pc_attr is not None and cdelt_attr is not None:
        pc = np.asarray(pc_attr, dtype=np.float64)
        cdelt = np.asarray(cdelt_attr, dtype=np.float64)
        if pc.ndim == 2 and pc.shape[0] >= 2 and pc.shape[1] >= 2 and cdelt.size >= 2:
            return pc[:2, :2] @ np.diag(cdelt[:2])
    return None


def _mean_pixel_scale_deg(wcs: WCS) -> float | None:
    cd = _extract_cd_matrix(wcs)
    if cd is None or not np.all(np.isfinite(cd)):
        return None
    try:
        det = float(np.linalg.det(cd))
    except Exception:
        return None
    if not np.isfinite(det) or abs(det) < 1e-16:
        return None
    scales = np.sqrt(np.sum(cd[:2, :2] ** 2, axis=0))
    finite = scales[np.isfinite(scales)]
    if finite.size == 0:
        return None
    return float(np.nanmean(np.abs(finite)))


def validate_solution(
    wcs: WCS,
    matches: np.ndarray,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, float | int | str | bool]:
    if thresholds is None:
        thresholds = {"rms_px": 1.0, "inliers": 60}
    if matches.size == 0:
        return {"quality": "FAIL", "success": False, "reason": "no matches"}
    try:
        world = wcs.wcs_pix2world(matches[:, :2], 0)
    except InvalidTransformError:
        return {"quality": "FAIL", "success": False, "reason": "invalid transform", "rms_px": float("inf"), "inliers": 0}
    except Exception:
        # Catch any wcslib/astropy error (e.g. singular PC/CD matrices) and
        # report a clean validation failure rather than propagating an exception
        return {"quality": "FAIL", "success": False, "reason": "wcs failure", "rms_px": float("inf"), "inliers": 0}
    residuals = np.linalg.norm(world - matches[:, 2:], axis=1)
    scale_deg = _mean_pixel_scale_deg(wcs)
    if scale_deg is None:
        return {
            "quality": "FAIL",
            "success": False,
            "reason": "invalid_pixel_scale",
            "rms_px": float("inf"),
            "inliers": int(matches.shape[0]),
        }

    scale_arcsec = float(scale_deg * 3600.0)
    if scale_arcsec < 0.3 or scale_arcsec > 15.0:
        return {
            "quality": "FAIL",
            "success": False,
            "reason": f"pixel_scale_out_of_range[{scale_arcsec:.3f}]",
            "rms_px": float("inf"),
            "inliers": int(matches.shape[0]),
        }

    residuals_px = residuals / max(scale_deg, 1e-12)
    finite = np.isfinite(residuals_px)
    if not np.any(finite):
        return {
            "quality": "FAIL",
            "success": False,
            "reason": "nonfinite_residuals",
            "rms_px": float("inf"),
            "inliers": 0,
            "pix_scale_arcsec": scale_arcsec,
        }

    rp = residuals_px[finite]
    med = float(np.median(rp)) if rp.size else float("inf")
    mad = float(np.median(np.abs(rp - med))) if rp.size else 0.0
    # Robust inlier gate: keep a strict floor tied to user threshold and clip out heavy tails.
    base_tol = max(1.0, 2.5 * float(thresholds.get("rms_px", 1.0)))
    robust_tol = max(base_tol, med + max(3.5 * mad, 1.5))
    inlier_mask = finite & (residuals_px <= robust_tol)
    n = int(np.count_nonzero(inlier_mask))
    if n <= 0:
        return {
            "quality": "FAIL",
            "success": False,
            "reason": "no_residual_inliers",
            "rms_px": float("inf"),
            "inliers": 0,
            "pix_scale_arcsec": scale_arcsec,
        }

    rms_px = float(np.sqrt(np.mean((residuals_px[inlier_mask]) ** 2)))
    success = rms_px <= float(thresholds.get("rms_px", 1.0)) and n >= int(thresholds.get("inliers", 60))
    return {
        "quality": "GOOD" if success else "FAIL",
        "success": success,
        "rms_px": rms_px,
        "inliers": n,
        "inliers_raw": int(matches.shape[0]),
        "pix_scale_arcsec": scale_arcsec,
    }
