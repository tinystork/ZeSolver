from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _select_neighbors_multiscale(
    positions: np.ndarray,
    seed: int,
    *,
    scale_bins: int = 6,
    neighbors_per_bin: int = 4,
) -> list[int]:
    """Return nearby indices for *seed* spread across logarithmic distance bins."""
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


def _pair_candidate_pool(
    positions: np.ndarray,
    pair: Tuple[int, int],
    *,
    pool_limit: int = 24,
    u_margin: float = 0.4,
    v_limit: float = 2.0,
) -> list[Tuple[int, int]]:
    idx_a, idx_b = pair
    base = positions[idx_b] - positions[idx_a]
    base_len = float(np.hypot(base[0], base[1]))
    if not np.isfinite(base_len) or base_len < 1e-6:
        return []
    axis_x = base / base_len
    axis_y = np.array([-axis_x[1], axis_x[0]])
    lo = -u_margin
    hi = 1.0 + u_margin
    entries: list[Tuple[float, int, int]] = []
    for idx in range(positions.shape[0]):
        if idx == idx_a or idx == idx_b:
            continue
        delta = positions[idx] - positions[idx_a]
        u = float(np.dot(delta, axis_x) / base_len)
        v = float(np.dot(delta, axis_y) / base_len)
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        if u < lo or u > hi or abs(v) > v_limit:
            continue
        score = abs(v) + 0.25 * abs(u - 0.5)
        side = 1 if v >= 0 else -1
        entries.append((score, idx, side))
    if not entries:
        return []
    entries.sort(key=lambda item: item[0])
    limited = entries[: max(1, pool_limit)]
    return [(int(idx), int(side)) for _score, idx, side in limited]


def _cross_side_pairs(pool: Sequence[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
    pos = [idx for idx, side in pool if side >= 0]
    neg = [idx for idx, side in pool if side < 0]
    for idx_c in pos:
        for idx_d in neg:
            if idx_c != idx_d:
                yield idx_c, idx_d


def generate_pairwise_quads(
    positions: np.ndarray,
    *,
    seed_order: Sequence[int] | None,
    max_quads: int,
    scale_bins: int = 6,
    neighbors_per_bin: int = 4,
    base_neighbor_limit: int = 6,
    candidate_pool: int = 24,
    per_pair_quads: int = 32,
) -> np.ndarray:
    """Return quad index tuples sampled via multi-scale base pairs."""
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    count = pts.shape[0]
    if max_quads <= 0 or count < 4:
        return np.zeros((0, 4), dtype=np.uint16)
    if seed_order is None:
        seeds = np.arange(count, dtype=np.int32)
    else:
        raw = np.asarray(seed_order, dtype=np.int32)
        seeds = raw[(raw >= 0) & (raw < count)]
        if seeds.size == 0:
            seeds = np.arange(count, dtype=np.int32)
    combos: list[Tuple[int, int, int, int]] = []
    seen_pairs: set[Tuple[int, int]] = set()
    seen_quads: set[Tuple[int, int, int, int]] = set()
    per_pair_cap = max(4, min(int(per_pair_quads), max_quads))
    base_limit = max(1, int(base_neighbor_limit))
    for seed in seeds:
        if len(combos) >= max_quads:
            break
        neighbors = _select_neighbors_multiscale(
            pts,
            int(seed),
            scale_bins=scale_bins,
            neighbors_per_bin=neighbors_per_bin,
        )
        if not neighbors:
            continue
        for neighbor in neighbors[:base_limit]:
            if neighbor == seed:
                continue
            pair = tuple(sorted((int(seed), int(neighbor))))
            if pair in seen_pairs:
                continue
            candidates = _pair_candidate_pool(
                pts,
                pair,
                pool_limit=candidate_pool,
            )
            if len(candidates) < 2:
                continue
            seen_pairs.add(pair)
            per_pair_added = 0
            for idx_c, idx_d in _cross_side_pairs(candidates):
                ordered = tuple(sorted((pair[0], pair[1], int(idx_c), int(idx_d))))
                if ordered in seen_quads:
                    continue
                seen_quads.add(ordered)
                combos.append((pair[0], pair[1], int(idx_c), int(idx_d)))
                per_pair_added += 1
                if len(combos) >= max_quads or per_pair_added >= per_pair_cap:
                    break
            if len(combos) >= max_quads:
                break
            if per_pair_added < per_pair_cap:
                base_candidates = [idx for idx, _ in candidates]
                for idx_c, idx_d in itertools.combinations(base_candidates, 2):
                    ordered = tuple(sorted((pair[0], pair[1], int(idx_c), int(idx_d))))
                    if ordered in seen_quads:
                        continue
                    seen_quads.add(ordered)
                    combos.append((pair[0], pair[1], int(idx_c), int(idx_d)))
                    per_pair_added += 1
                    if len(combos) >= max_quads or per_pair_added >= per_pair_cap:
                        break
            if len(combos) >= max_quads:
                break
    if not combos:
        return np.zeros((0, 4), dtype=np.uint16)
    return np.array(combos, dtype=np.uint16)
