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
    parity_metrics_only = bool(thresholds.get("astrometry_parity_mode", False))
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
    try:
        scale_min = float(thresholds.get("scale_min_arcsec", 0.3))
    except Exception:
        scale_min = 0.3
    try:
        scale_max = float(thresholds.get("scale_max_arcsec", 15.0))
    except Exception:
        scale_max = 15.0
    scale_min = max(0.05, scale_min)
    scale_max = max(scale_min + 1e-6, scale_max)
    scale_in_range = bool(scale_min <= scale_arcsec <= scale_max)
    if (not parity_metrics_only) and (not scale_in_range):
        return {
            "quality": "FAIL",
            "success": False,
            "reason": (
                f"pixel_scale_out_of_range[scale={scale_arcsec:.3f},"
                f"min={scale_min:.3f},max={scale_max:.3f}]"
            ),
            "rms_px": float("inf"),
            "inliers": int(matches.shape[0]),
            "pix_scale_arcsec": scale_arcsec,
            "scale_min_arcsec": scale_min,
            "scale_max_arcsec": scale_max,
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
    rms_thr = float(thresholds.get("rms_px", 1.0))
    inlier_thr = int(thresholds.get("inliers", 60))
    rms_ok = bool(rms_px <= rms_thr)
    inliers_ok = bool(n >= inlier_thr)
    success = bool(rms_ok and inliers_ok)
    metrics_only_progression = bool(parity_metrics_only and (not success))
    reason = None
    if not success:
        reason_tag = "validation_metrics_only" if metrics_only_progression else "validation_failed"
        reason = (
            f"{reason_tag}[rms_ok={int(rms_ok)},inliers_ok={int(inliers_ok)},"
            f"scale_ok={int(scale_in_range)},rms={rms_px:.3f},rms_thr={rms_thr:.3f},"
            f"inliers={n},inliers_thr={inlier_thr}]"
        )
    return {
        "quality": "GOOD" if success else "FAIL",
        "success": bool(success),
        "reason": reason,
        "rms_px": rms_px,
        "inliers": n,
        "inliers_raw": int(matches.shape[0]),
        "pix_scale_arcsec": scale_arcsec,
        "rms_threshold_px": rms_thr,
        "inliers_threshold": inlier_thr,
        "gate_scale_ok": scale_in_range,
        "gate_rms_ok": rms_ok,
        "gate_inliers_ok": inliers_ok,
        "astrometry_parity_mode": bool(parity_metrics_only),
        "validation_metrics_only": bool(metrics_only_progression),
        "validation_progress_eligible": bool(metrics_only_progression),
        "robust_tol_px": float(robust_tol),
        "median_residual_px": float(med),
        "mad_residual_px": float(mad),
    }
