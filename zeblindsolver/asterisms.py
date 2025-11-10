from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .levels import QuadLevelSpec

MIN_RATIO = 0.25
MAX_RATIO = 4.0
MIN_AREA = 1e-7


@dataclass(frozen=True, slots=True)
class QuadHash:
    indices: np.ndarray  # shape (n, 4)
    hashes: np.ndarray  # dtype=uint64


def _quad_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _quantize_ratio(value: float) -> int:
    clipped = min(max(value, MIN_RATIO), MAX_RATIO)
    norm = (clipped - MIN_RATIO) / (MAX_RATIO - MIN_RATIO)
    return int(round(norm * 65535))


def _quad_diameter(points: np.ndarray) -> float:
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return float(np.nanmax(dists))


def _hash_from_indexes(
    order: np.ndarray,
    positions: np.ndarray,
    *,
    spec: QuadLevelSpec | None = None,
) -> tuple[int, np.ndarray] | None:
    points = positions[order]
    if points.shape != (4, 2):
        return None
    area = _quad_area(points)
    if area < MIN_AREA:
        return None
    if spec:
        if area < spec.min_area or area > spec.max_area:
            return None
        diameter = _quad_diameter(points)
        if spec.min_diameter is not None and diameter < spec.min_diameter:
            return None
        if spec.max_diameter is not None and diameter > spec.max_diameter:
            return None
    a, b, c, d = points
    def dist(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.hypot(*(u - v)))
    d12 = dist(a, b)
    d34 = dist(c, d)
    d13 = dist(a, c)
    d24 = dist(b, d)
    d14 = dist(a, d)
    d23 = dist(b, c)
    eps = 1e-8
    r12 = d12 / (d34 + eps)
    r13 = d13 / (d24 + eps)
    r14 = d14 / (d23 + eps)
    q1 = _quantize_ratio(r12)
    q2 = _quantize_ratio(r13)
    q3 = _quantize_ratio(r14)
    parity = 1 if np.cross(b - a, c - a) >= 0 else 0
    hash_value = (q1 << 48) | (q2 << 32) | (q3 << 16) | parity
    return hash_value, order


def sample_quads(stars: np.ndarray, max_quads: int, strategy: str = "biased_brightness") -> np.ndarray:
    """Return up to *max_quads* index-tuples biased toward bright stars."""
    if max_quads <= 0:
        return np.zeros((0, 4), dtype=np.uint16)
    if stars.shape[0] < 4:
        return np.zeros((0, 4), dtype=np.uint16)
    mags = stars["mag"]
    order = np.argsort(mags)
    if strategy == "biased_brightness":
        limit = min(len(order), max(16, int(max_quads ** 0.5) * 3, 64))
        pool = order[:limit]
    else:
        pool = order
    combos = []
    for combo in itertools.combinations(pool, 4):
        combos.append(combo)
        if len(combos) >= max_quads:
            break
    if not combos:
        return np.zeros((0, 4), dtype=np.uint16)
    return np.array(combos, dtype=np.uint16)

def hash_quads(quads: np.ndarray, positions: np.ndarray, *, spec: QuadLevelSpec | None = None) -> QuadHash:
    """Hash the provided quads using the 3-ratio encoding and parity bit."""
    valid = []
    hashes = []
    for combo in quads:
        if len(set(combo)) < 4:
            continue
        order = np.argsort(np.linalg.norm(positions[combo] - positions[combo].mean(axis=0), axis=1))
        result = _hash_from_indexes(combo[order], positions, spec=spec)
        if result is None:
            continue
        hash_value, ordered = result
        valid.append(combo[order])
        hashes.append(hash_value)
    if not hashes:
        return QuadHash(np.zeros((0, 4), dtype=np.uint16), np.zeros(0, dtype=np.uint64))
    return QuadHash(np.stack(valid), np.array(hashes, dtype=np.uint64))
