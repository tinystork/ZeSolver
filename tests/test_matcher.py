from __future__ import annotations

import math

import numpy as np

from zeblindsolver.matcher import estimate_similarity_RANSAC


def _reflect_points(points: np.ndarray, scale: float, rotation_deg: float, translation: tuple[float, float]) -> np.ndarray:
    complex_points = points[:, 0] + 1j * points[:, 1]
    complex_points = np.conj(complex_points)
    rot_scale = scale * np.exp(1j * math.radians(rotation_deg))
    translation_c = complex(*translation)
    mapped = rot_scale * complex_points + translation_c
    return np.column_stack((mapped.real, mapped.imag)).astype(np.float64)


def test_estimate_similarity_ransac_handles_reflection():
    rng = np.random.default_rng(42)
    image_points = rng.uniform(0, 200, size=(20, 2))
    catalog_points = _reflect_points(image_points, scale=0.002, rotation_deg=23.5, translation=(0.1, -0.05))
    result = estimate_similarity_RANSAC(
        image_points,
        catalog_points,
        allow_reflection=True,
        tol_px=0.5,
        min_inliers=5,
    )
    assert result is not None
    transform, stats = result
    assert transform.parity == -1
    assert stats.inliers >= 5
    src = image_points[:, 0] + 1j * image_points[:, 1]
    src = np.conj(src)
    rot_scale = transform.scale * np.exp(1j * transform.rotation)
    translation = complex(*transform.translation)
    pred = rot_scale * src + translation
    dst = catalog_points[:, 0] + 1j * catalog_points[:, 1]
    err = np.max(np.abs(pred - dst))
    assert err < 1e-6
