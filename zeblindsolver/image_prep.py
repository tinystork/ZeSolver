from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, zoom


def read_fits_as_luma(path: Path | str) -> np.ndarray:
    """Read the primary HDU and collapse any extra layers into a luma array."""
    data = fits.getdata(path, allow_pickle=False)
    if data is None:
        raise ValueError(f"{path} contains no image data")
    img = np.asarray(data, dtype=np.float32)
    if img.ndim == 2:
        return img
    if img.ndim >= 3:
        return np.mean(img.reshape(-1, img.shape[-2], img.shape[-1]), axis=0)
    raise ValueError("unsupported image dimensionality")


def build_pyramid(img: np.ndarray, levels: list[int] | tuple[int, ...] = (4, 2, 1)) -> list[np.ndarray]:
    """Generate downsampled copies of *img* for multi-scale processing."""
    normalized = [int(level) for level in levels if level >= 1]
    pyramid: list[np.ndarray] = []
    seen = set()
    for level in sorted(normalized, reverse=True):
        if level in seen:
            continue
        seen.add(level)
        if level <= 1:
            pyramid.append(img)
            continue
        factor = 1.0 / level
        pyramid.append(zoom(img, factor, order=1))
    if 1 not in seen:
        pyramid.append(img)
    return pyramid


def remove_background(img: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Subtract a median-filter background to flatten gradients."""
    size = min(max(3, kernel_size), min(img.shape))
    background = median_filter(img, size=size)
    return img - background


def downsample_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample *img* by an integer *factor* using bilinear interpolation."""
    factor = max(1, int(factor))
    if factor == 1:
        return img
    scale = 1.0 / float(factor)
    return zoom(img, scale, order=1)
