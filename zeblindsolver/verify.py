from __future__ import annotations

from typing import Mapping

import numpy as np
from astropy.wcs import InvalidTransformError, WCS


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
    residuals = np.linalg.norm(world - matches[:, 2:], axis=1)
    cd = getattr(wcs.wcs, "cd", None)
    if cd is not None and np.linalg.det(cd) != 0:
        scale = abs(np.linalg.det(cd)) ** 0.5
    else:
        scale = 1.0
    rms_deg = float(np.sqrt(np.mean(residuals ** 2)))
    rms_px = rms_deg / max(scale, 1e-8)
    n = matches.shape[0]
    success = rms_px <= thresholds.get("rms_px", 1.0) and n >= thresholds.get("inliers", 60)
    return {
        "quality": "GOOD" if success else "FAIL",
        "success": success,
        "rms_px": rms_px,
        "inliers": n,
    }
