# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : GPL V3 (voir pyproject.toml / repository metadata)               ║
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
    delta = positions - positions[idx_a]
    u = (delta @ axis_x) / base_len
    v = (delta @ axis_y) / base_len
    mask = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= lo)
        & (u <= hi)
        & (np.abs(v) <= v_limit)
    )
    mask[idx_a] = False
    mask[idx_b] = False
    if not bool(mask.any()):
        return []
    selected = np.nonzero(mask)[0]
    scores = np.abs(v[selected]) + 0.25 * np.abs(u[selected] - 0.5)
    order = np.argsort(scores, kind="stable")[: max(1, int(pool_limit))]
    limited = selected[order]
    return [(int(idx), 1 if float(v[idx]) >= 0.0 else -1) for idx in limited]


def _cross_side_pairs(pool: Sequence[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
    pos = [idx for idx, side in pool if side >= 0]
    neg = [idx for idx, side in pool if side < 0]
    for idx_c in pos:
        for idx_d in neg:
            if idx_c != idx_d:
                yield idx_c, idx_d


def _hybrid_base_neighbor_positions(neighbor_count: int, limit: int) -> list[int]:
    """Return neighbor-list positions biased to close bases, then spread out."""
    if neighbor_count <= 0 or limit <= 0:
        return []
    selected: list[int] = []

    def add(value: int) -> None:
        pos = max(0, min(neighbor_count - 1, int(value)))
        if pos not in selected:
            selected.append(pos)

    # Preserve the local bases that often generate the useful Astrometry-like
    # hypotheses, then add a few wider anchors for scale diversity.
    for value in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 28, 38, 50, 64):
        if len(selected) >= limit:
            break
        if value < neighbor_count:
            add(value)
    if len(selected) < limit and neighbor_count > 1:
        for value in np.linspace(0, neighbor_count - 1, limit * 2, dtype=int):
            if len(selected) >= limit:
                break
            add(int(value))
    return selected[:limit]


def _spread_base_neighbor_positions(neighbor_count: int, limit: int) -> list[int]:
    if neighbor_count <= 0 or limit <= 0:
        return []
    return [
        int(value)
        for value in np.linspace(
            0,
            neighbor_count - 1,
            min(int(limit), int(neighbor_count)),
            dtype=int,
        )
    ]


def _companion_pair_positions(candidate_count: int) -> Iterable[Tuple[int, int]]:
    """Yield companion-list index pairs with local rank coverage first."""
    if candidate_count < 2:
        return
    seen: set[Tuple[int, int]] = set()

    def emit(first: int, second: int) -> Tuple[int, int] | None:
        if first == second:
            return None
        if first < 0 or second < 0 or first >= candidate_count or second >= candidate_count:
            return None
        pair = tuple(sorted((int(first), int(second))))
        if pair in seen:
            return None
        seen.add(pair)
        return pair

    # Adjacent and near-adjacent rank pairs keep mid-list companions reachable
    # even when each base has a small quota.
    anchors = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20)
    for gap in (1, 2):
        for anchor in anchors:
            pair = emit(anchor, anchor + gap)
            if pair is not None:
                yield pair
    for first, second in itertools.combinations(range(candidate_count), 2):
        pair = emit(first, second)
        if pair is not None:
            yield pair


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
    seed_cap = min(
        int(seeds.size),
        max(16, min(count, 64, int(np.sqrt(max(1, int(max_quads)))))),
    )
    for seed in seeds[:seed_cap]:
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


def generate_coverage_first_quads(
    positions: np.ndarray,
    *,
    seed_order: Sequence[int] | None,
    max_quads: int,
    scale_bins: int = 6,
    neighbors_per_bin: int = 5,
    base_neighbor_limit: int = 8,
    candidate_pool: int = 28,
    per_pair_quads: int = 16,
) -> np.ndarray:
    """Return pairwise quads while preserving seed/pair coverage.

    The default pairwise sampler is intentionally brightness-biased and can
    spend most of a fixed catalog budget on early seeds. Catalog tables need a
    different bias: expose at least a few hypotheses from many seed/pair
    anchors before deepening any single pair.
    """
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

    pair_records: list[list[Tuple[int, int, int, int]]] = []
    seen_pairs: set[Tuple[int, int]] = set()
    seen_record_quads: set[Tuple[int, int, int, int]] = set()
    base_limit = max(1, int(base_neighbor_limit))
    per_pair_cap = max(1, min(int(per_pair_quads), int(max_quads)))

    for seed in seeds:
        neighbors = _select_neighbors_multiscale(
            pts,
            int(seed),
            scale_bins=scale_bins,
            neighbors_per_bin=neighbors_per_bin,
        )
        if not neighbors:
            continue
        for neighbor in neighbors[:base_limit]:
            if int(neighbor) == int(seed):
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
            base_candidates = [idx for idx, _side in candidates]
            local: list[Tuple[int, int, int, int]] = []
            for idx_c, idx_d in _cross_side_pairs(candidates):
                ordered = tuple(sorted((pair[0], pair[1], int(idx_c), int(idx_d))))
                if ordered in seen_record_quads:
                    continue
                seen_record_quads.add(ordered)
                local.append((pair[0], pair[1], int(idx_c), int(idx_d)))
                if len(local) >= per_pair_cap:
                    break
            if len(local) < per_pair_cap:
                for idx_c, idx_d in itertools.combinations(base_candidates, 2):
                    ordered = tuple(sorted((pair[0], pair[1], int(idx_c), int(idx_d))))
                    if ordered in seen_record_quads:
                        continue
                    seen_record_quads.add(ordered)
                    local.append((pair[0], pair[1], int(idx_c), int(idx_d)))
                    if len(local) >= per_pair_cap:
                        break
            if local:
                pair_records.append(local)

    if not pair_records:
        return np.zeros((0, 4), dtype=np.uint16)

    combos: list[Tuple[int, int, int, int]] = []
    seen_quads: set[Tuple[int, int, int, int]] = set()
    max_depth = max(len(record) for record in pair_records)
    for depth in range(max_depth):
        for record in pair_records:
            if depth >= len(record):
                continue
            quad = record[depth]
            ordered = tuple(sorted(quad))
            if ordered in seen_quads:
                continue
            seen_quads.add(ordered)
            combos.append(quad)
            if len(combos) >= max_quads:
                return np.array(combos, dtype=np.uint16)

    return np.array(combos, dtype=np.uint16)


def generate_ring_coverage_quads(
    positions: np.ndarray,
    *,
    seed_order: Sequence[int] | None,
    max_quads: int,
    seed_cap: int = 512,
    neighbor_cap: int = 48,
    base_pair_cap: int = 10,
    companion_pool: int = 16,
    per_seed_quads: int = 24,
    per_base_quads: int = 6,
    u_margin: float = 0.5,
    v_limit: float = 2.5,
    base_selection: str = "hybrid",
    companion_order: str = "coverage",
) -> np.ndarray:
    """Return quads from spread base pairs and local companion rings.

    Catalog tables need coverage for mid-rank local geometries, not just the
    earliest bright seeds. This sampler keeps a fixed budget but samples base
    pairs at multiple neighbor depths for each seed, then forms a small number
    of companions around each base segment.
    """
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

    seed_limit = min(int(seeds.size), max(16, int(seed_cap)))
    neighbor_limit = max(4, int(neighbor_cap))
    base_limit = max(1, int(base_pair_cap))
    pool_limit = max(4, int(companion_pool))
    per_seed_cap = max(1, int(per_seed_quads))
    per_base_cap = max(1, int(per_base_quads))
    combos: list[Tuple[int, int, int, int]] = []
    seen_quads: set[Tuple[int, int, int, int]] = set()

    for seed_raw in seeds[:seed_limit]:
        if len(combos) >= max_quads:
            break
        seed = int(seed_raw)
        d2 = np.sum((pts - pts[seed]) ** 2, axis=1)
        d2[seed] = np.inf
        take = min(neighbor_limit, max(0, count - 1))
        if take <= 0:
            continue
        near_idx = np.argpartition(d2, take - 1)[:take]
        near_idx = near_idx[np.argsort(d2[near_idx], kind="stable")]
        nn = [int(i) for i in near_idx if np.isfinite(float(d2[int(i)]))]
        if len(nn) < 3:
            continue

        # Cover close bases first, then add wider anchors. Useful blind
        # hypotheses often sit at ranks 6-12, while pure linspace sampling can
        # skip them and pure nearest-first can miss broader geometry.
        base_neighbors: list[int] = []
        base_positions = (
            _spread_base_neighbor_positions(len(nn), min(base_limit, len(nn)))
            if str(base_selection).lower() == "spread"
            else _hybrid_base_neighbor_positions(len(nn), min(base_limit, len(nn)))
        )
        for pos in base_positions:
            value = int(nn[int(pos)])
            if value not in base_neighbors:
                base_neighbors.append(value)

        local_added = 0
        for neighbor in base_neighbors:
            if local_added >= per_seed_cap or len(combos) >= max_quads:
                break
            base = pts[neighbor] - pts[seed]
            base_len = float(np.hypot(base[0], base[1]))
            if not np.isfinite(base_len) or base_len < 1e-12:
                continue
            axis_x = base / base_len
            axis_y = np.array([-axis_x[1], axis_x[0]])
            delta = pts - pts[seed]
            u = (delta @ axis_x) / base_len
            v = (delta @ axis_y) / base_len
            mask = (
                np.isfinite(u)
                & np.isfinite(v)
                & (u >= -float(u_margin))
                & (u <= 1.0 + float(u_margin))
                & (np.abs(v) <= float(v_limit))
            )
            mask[seed] = False
            mask[neighbor] = False
            candidates = np.nonzero(mask)[0]
            if candidates.size < 2:
                continue
            scores = np.abs(v[candidates]) + 0.15 * np.abs(u[candidates] - 0.5)
            cand = [int(i) for i in candidates[np.argsort(scores, kind="stable")[:pool_limit]]]
            base_added = 0
            if str(companion_order).lower() == "combinations":
                pair_iter = itertools.combinations(range(len(cand)), 2)
            else:
                pair_iter = _companion_pair_positions(len(cand))
            for pos_c, pos_d in pair_iter:
                idx_c = cand[int(pos_c)]
                idx_d = cand[int(pos_d)]
                key = tuple(sorted((seed, int(neighbor), int(idx_c), int(idx_d))))
                if key in seen_quads:
                    continue
                seen_quads.add(key)
                combos.append((seed, int(neighbor), int(idx_c), int(idx_d)))
                local_added += 1
                base_added += 1
                if local_added >= per_seed_cap or len(combos) >= max_quads:
                    break

    if not combos:
        return np.zeros((0, 4), dtype=np.uint16)
    return np.asarray(combos, dtype=np.uint16)
