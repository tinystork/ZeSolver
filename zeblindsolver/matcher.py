from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: float
    translation: tuple[float, float]
    parity: int = 1


@dataclass(frozen=True)
class SimilarityStats:
    rms_px: float
    inliers: int


def _complexify(points: np.ndarray) -> np.ndarray:
    return points[:, 0] + 1j * points[:, 1]


def _derive_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    reflected: bool = False,
) -> tuple[np.complex128, np.complex128] | None:
    src_c = _complexify(src)
    if reflected:
        src_c = np.conj(src_c)
    dst_c = _complexify(dst)
    src_mean = np.mean(src_c)
    dst_mean = np.mean(dst_c)
    src_zero = src_c - src_mean
    dst_zero = dst_c - dst_mean
    denom = np.sum(np.abs(src_zero) ** 2)
    if denom < 1e-12:
        return None
    rot_scale = np.sum(dst_zero * np.conj(src_zero)) / denom
    translation = dst_mean - rot_scale * src_mean
    return rot_scale, translation


def estimate_similarity_RANSAC(
    image_points: np.ndarray,
    catalog_points: np.ndarray,
    *,
    trials: int = 2000,
    tol_px: float = 3.0,
    min_inliers: int = 3,
    allow_reflection: bool = False,
) -> tuple[SimilarityTransform, SimilarityStats] | None:
    if len(image_points) < 2 or len(catalog_points) < 2:
        return None
    rng = np.random.default_rng()
    best_mask: np.ndarray | None = None
    best_transform: tuple[np.complex128, np.complex128] | None = None
    best_parity = 1
    best_score = 0
    best_rms = float("inf")
    combos = list(range(len(image_points)))
    parity_modes = [1]
    if allow_reflection:
        parity_modes.append(-1)
    for _ in range(trials):
        sample = rng.choice(combos, size=2, replace=False)
        src = image_points[sample]
        dst = catalog_points[sample]
        for parity in parity_modes:
            derived = _derive_similarity(src, dst, reflected=(parity < 0))
            if derived is None:
                continue
            rot_scale, translation = derived
            src_c = _complexify(image_points)
            if parity < 0:
                src_c = np.conj(src_c)
            dst_c = _complexify(catalog_points)
            predictions = rot_scale * src_c + translation
            err_deg = np.abs(predictions - dst_c)
            scale = abs(rot_scale)
            err_px = err_deg / max(scale, 1e-8)
            mask = err_px <= tol_px
            score = int(mask.sum())
            if score < min_inliers:
                continue
            rms = float(np.sqrt(np.mean(err_px[mask] ** 2))) if score else float("inf")
            if score > best_score or (score == best_score and rms < best_rms):
                best_score = score
                best_transform = (rot_scale, translation)
                best_mask = mask
                best_rms = rms
                best_parity = parity
    if best_transform is None or best_mask is None:
        return None
    rot_scale, translation = best_transform
    src_c = _complexify(image_points)
    if best_parity < 0:
        src_c = np.conj(src_c)
    dst_c = _complexify(catalog_points)
    predictions = rot_scale * src_c + translation
    err_deg = np.abs(predictions - dst_c)
    scale = abs(rot_scale)
    err_px = err_deg / max(scale, 1e-8)
    mask = err_px <= tol_px
    # Refit using inliers to stabilize the similarity parameters
    if mask.any():
        in_src = image_points[mask]
        in_dst = catalog_points[mask]
        refined = _derive_similarity(in_src, in_dst, reflected=(best_parity < 0))
        if refined is not None:
            rot_scale, translation = refined
            scale = abs(rot_scale)
            src_c = _complexify(image_points)
            if best_parity < 0:
                src_c = np.conj(src_c)
            dst_c = _complexify(catalog_points)
            predictions = rot_scale * src_c + translation
            err_deg = np.abs(predictions - dst_c)
            err_px = err_deg / max(scale, 1e-8)
            mask = err_px <= tol_px
    rms_px = float(np.sqrt(np.mean(err_px ** 2)))
    transform = SimilarityTransform(
        scale=float(max(scale, 1e-12)),
        rotation=float(np.angle(rot_scale)),
        translation=(float(translation.real), float(translation.imag)),
        parity=int(best_parity),
    )
    stats = SimilarityStats(rms_px=rms_px, inliers=int(mask.sum()))
    return transform, stats
