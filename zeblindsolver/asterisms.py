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


def _select_neighbors_multiscale(
    positions: np.ndarray,
    seed: int,
    *,
    scale_bins: int = 6,
    neighbors_per_bin: int = 4,
) -> list[int]:
    """Pick neighbors for *seed* across logarithmically spaced distance bins."""
    dx = positions[:, 0] - positions[seed, 0]
    dy = positions[:, 1] - positions[seed, 1]
    dist2 = dx * dx + dy * dy
    finite_mask = np.isfinite(dist2)
    finite_mask[seed] = False
    if not finite_mask.any():
        return []
    positive = dist2[finite_mask]
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return []
    dmin = float(np.nanmin(positive))
    dmax = float(np.nanmax(positive))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= 0.0:
        return []
    bin_count = max(1, min(scale_bins, int(positive.size)))
    if dmax <= dmin * 1.05:
        edges = np.linspace(dmin, dmax + 1e-12, num=bin_count + 1)
    else:
        edges = np.geomspace(max(dmin, 1e-12), dmax, num=bin_count + 1)
    edges[-1] = max(edges[-1], dmax)
    order = np.argsort(np.where(np.isfinite(dist2), dist2, np.inf))
    valid_order = [idx for idx in order if finite_mask[idx] and dist2[idx] > 0.0]
    if not valid_order:
        return []
    dist_sorted = dist2[valid_order]
    bin_lists: list[list[int]] = []
    for i in range(edges.shape[0] - 1):
        start = edges[i]
        end = edges[i + 1]
        mask = (dist_sorted >= start) & (dist_sorted < end if i < edges.shape[0] - 2 else dist_sorted <= end)
        if not mask.any():
            continue
        candidates = np.asarray(valid_order, dtype=np.int64)[mask][:neighbors_per_bin]
        if candidates.size:
            bin_lists.append(candidates.tolist())
    if not bin_lists:
        return []
    prioritized: list[int] = []
    seen: set[int] = set()
    for depth in range(neighbors_per_bin):
        for group in bin_lists:
            if depth < len(group):
                idx = int(group[depth])
                if idx not in seen:
                    prioritized.append(idx)
                    seen.add(idx)
    return prioritized


def _legacy_biased_brightness(order: np.ndarray, max_quads: int) -> np.ndarray:
    limit = min(len(order), max(16, int(max_quads ** 0.5) * 3, 64))
    pool = order[:limit]
    combos = []
    for combo in itertools.combinations(pool, 4):
        combos.append(combo)
        if len(combos) >= max_quads:
            break
    if not combos:
        return np.zeros((0, 4), dtype=np.uint16)
    return np.array(combos, dtype=np.uint16)


def _legacy_local_brightness(order: np.ndarray, stars: np.ndarray, max_quads: int) -> np.ndarray:
    K = 16
    per_seed = (K * (K - 1) * (K - 2)) // 6
    min_seeds = 4
    seeds_needed = max(min_seeds, int(np.ceil(max_quads / max(1, per_seed))))
    seeds = order[:min(len(order), seeds_needed * 2)]
    combos: list[tuple[int, int, int, int]] = []
    xy = np.column_stack((stars["x"], stars["y"]))
    for seed in seeds:
        dx = xy[:, 0] - xy[seed, 0]
        dy = xy[:, 1] - xy[seed, 1]
        dist2 = dx * dx + dy * dy
        order_nn = np.argsort(dist2)
        nn = [idx for idx in order_nn if idx != seed][:K]
        if len(nn) < 3:
            continue
        for a, b, c in itertools.combinations(nn, 3):
            combos.append((seed, a, b, c))
            if len(combos) >= max_quads:
                break
        if len(combos) >= max_quads:
            break
    if not combos:
        return np.zeros((0, 4), dtype=np.uint16)
    return np.array(combos, dtype=np.uint16)


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


def sample_quads(stars: np.ndarray, max_quads: int, strategy: str = "log_spaced") -> np.ndarray:
    """Return up to *max_quads* quads with multi-scale distance coverage.

    Default (any non-legacy strategy):
      - iterate over all finite stars in spatial order
      - pick neighbors for each seed across logarithmic distance bins
      - interleave bins to mix small/medium/large quads and stop once *max_quads* is reached

    Legacy fallbacks (for backward compatibility):
      - "legacy_brightness": previous brightest-pool sampling
      - "legacy_local": previous bright-seed + nearest-neighbor sampling
    """
    if max_quads <= 0 or stars.shape[0] < 4:
        return np.zeros((0, 4), dtype=np.uint16)
    method = (strategy or "log_spaced").lower()
    mags = stars["mag"]
    order = np.argsort(mags)
    if method == "legacy_brightness":
        return _legacy_biased_brightness(order, max_quads)
    if method == "legacy_local":
        return _legacy_local_brightness(order, stars, max_quads)

    positions = np.column_stack((stars["x"].astype(np.float64), stars["y"].astype(np.float64)))
    finite_mask = np.isfinite(positions).all(axis=1)
    valid_indices = np.nonzero(finite_mask)[0]
    if valid_indices.size < 4:
        return np.zeros((0, 4), dtype=np.uint16)
    spatial_order = np.lexsort((positions[valid_indices, 1], positions[valid_indices, 0]))
    seeds = valid_indices[spatial_order]
    combos: list[tuple[int, int, int, int]] = []
    seen_quads: set[tuple[int, int, int, int]] = set()
    per_seed_cap = min(max_quads, 512)
    for seed in seeds:
        if len(combos) >= max_quads:
            break
        neighbors = _select_neighbors_multiscale(positions, seed)
        if len(neighbors) < 3:
            continue
        per_seed_limit = min(per_seed_cap, max_quads - len(combos))
        added = 0
        for a, b, c in itertools.combinations(neighbors, 3):
            quad = (seed, a, b, c)
            key = tuple(sorted(quad))
            if key in seen_quads:
                continue
            seen_quads.add(key)
            combos.append(quad)
            added += 1
            if added >= per_seed_limit or len(combos) >= max_quads:
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
