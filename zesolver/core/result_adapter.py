from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales

from .models import EngineSolveResult, SolveRequest, SolveResult, SolveStatus


def pixel_scale_arcsec_from_wcs(wcs_obj) -> float | None:
    if wcs_obj is None or not bool(getattr(wcs_obj, "has_celestial", getattr(wcs_obj, "is_celestial", False))):
        return None
    try:
        scales = np.asarray(proj_plane_pixel_scales(wcs_obj), dtype=float) * 3600.0
    except Exception:
        return None
    finite = scales[np.isfinite(scales)]
    if finite.size < 2:
        return None
    return float(np.mean(np.abs(finite[:2])))


def orientation_deg_from_wcs(wcs_obj) -> float | None:
    if wcs_obj is None:
        return None
    try:
        cd = getattr(wcs_obj.wcs, "cd", None)
        if cd is None:
            pc = getattr(wcs_obj.wcs, "pc", None)
            cdelt = getattr(wcs_obj.wcs, "cdelt", None)
            if pc is None or cdelt is None:
                return None
            cd = np.asarray(pc, dtype=float) @ np.diag(np.asarray(cdelt, dtype=float))
        cd = np.asarray(cd, dtype=float)[:2, :2]
        if not np.all(np.isfinite(cd)) or abs(float(np.linalg.det(cd))) < 1e-16:
            return None
        return float(math.degrees(math.atan2(cd[1, 0], cd[0, 0])))
    except Exception:
        return None


def parity_from_wcs(wcs_obj) -> str | None:
    if wcs_obj is None:
        return None
    try:
        cd = getattr(wcs_obj.wcs, "cd", None)
        if cd is None:
            pc = getattr(wcs_obj.wcs, "pc", None)
            cdelt = getattr(wcs_obj.wcs, "cdelt", None)
            if pc is None or cdelt is None:
                return None
            cd = np.asarray(pc, dtype=float) @ np.diag(np.asarray(cdelt, dtype=float))
        det = float(np.linalg.det(np.asarray(cd, dtype=float)[:2, :2]))
    except Exception:
        return None
    if not math.isfinite(det) or abs(det) < 1e-16:
        return None
    return "negative" if det < 0 else "positive"


def result_from_engine(
    request: SolveRequest,
    engine: EngineSolveResult,
    *,
    profile_ids: Mapping[str, str],
    catalog_status: str | None,
    warnings: tuple[str, ...] = (),
    output_path=None,
) -> SolveResult:
    wcs_obj = engine.wcs
    return SolveResult(
        request_id=request.request_id,
        input_path=request.input_path,
        output_path=output_path if output_path is not None else request.output_path,
        status=engine.status,
        backend=engine.backend,
        wcs_written=engine.wcs_written,
        center_ra_deg=engine.center_ra_deg,
        center_dec_deg=engine.center_dec_deg,
        pixel_scale_arcsec=engine.pixel_scale_arcsec if engine.pixel_scale_arcsec is not None else pixel_scale_arcsec_from_wcs(wcs_obj),
        orientation_deg=engine.orientation_deg if engine.orientation_deg is not None else orientation_deg_from_wcs(wcs_obj),
        parity=engine.parity if engine.parity is not None else parity_from_wcs(wcs_obj),
        inliers=engine.inliers,
        rms_px=engine.rms_px,
        profile_ids=dict(profile_ids),
        catalog_status=catalog_status,
        warnings=tuple(dict.fromkeys((*warnings, *engine.warnings))),
        error=engine.error,
    )


def failure_result(
    request: SolveRequest,
    *,
    status: SolveStatus,
    profile_ids: Mapping[str, str],
    catalog_status: str | None = None,
    warnings: tuple[str, ...] = (),
    error: str | None = None,
) -> SolveResult:
    return SolveResult(
        request_id=request.request_id,
        input_path=request.input_path,
        output_path=request.output_path,
        status=status,
        backend=None,
        wcs_written=False,
        center_ra_deg=None,
        center_dec_deg=None,
        pixel_scale_arcsec=None,
        orientation_deg=None,
        parity=None,
        inliers=None,
        rms_px=None,
        profile_ids=dict(profile_ids),
        catalog_status=catalog_status,
        warnings=warnings,
        error=error,
    )
