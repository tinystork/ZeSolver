from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label


def detect_stars(
    img: np.ndarray,
    *,
    min_fwhm_px: float = 1.5,
    max_fwhm_px: float = 8.0,
    k_sigma: float = 3.0,
    min_area: int = 5,
) -> np.ndarray:
    data = np.asarray(img, dtype=np.float32)
    blurred = gaussian_filter(data, sigma=min_fwhm_px)
    mean = float(np.nanmean(blurred))
    std = float(np.nanstd(blurred))
    threshold = mean + k_sigma * std
    mask = blurred > threshold
    if not mask.any():
        return np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    structure = np.ones((3, 3), dtype=bool)
    labeled, count = label(mask, structure=structure)
    stars = []
    for label_idx in range(1, count + 1):
        region = labeled == label_idx
        area = int(region.sum())
        if area < min_area:
            continue
        flux = float((data * region).sum())
        cy, cx = center_of_mass(data, labels=labeled, index=label_idx)
        if not np.isfinite(cx) or not np.isfinite(cy):
            continue
        fwhm = min(max_fwhm_px, max(min_fwhm_px, np.sqrt(area)))
        stars.append((cx, cy, flux, fwhm))
    if not stars:
        return np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    array = np.array(stars, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    order = np.argsort(array["flux"])[::-1]
    return array[order]
