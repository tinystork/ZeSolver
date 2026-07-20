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
from dataclasses import dataclass

import numpy as np

from .levels import QuadLevelSpec
from .quad_sampling import generate_coverage_first_quads, generate_pairwise_quads, generate_ring_coverage_quads

QUAD_HASH_QUANTIZATION_MAX = 255
MIN_AREA = 1e-7


@dataclass(frozen=True, slots=True)
class QuadHash:
    indices: np.ndarray  # shape (n, 4)
    hashes: np.ndarray  # dtype=uint64
    source_indices: np.ndarray | None = None  # shape (n,), index in input quads


def _quad_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _quantize_ratio(value: float) -> int:
    """Quantize a bounded, permutation-invariant edge ratio to eight bits."""
    clipped = min(max(value, 0.0), 1.0)
    return int(round(clipped * QUAD_HASH_QUANTIZATION_MAX))


def opposite_edge_ratio_code(points: np.ndarray) -> np.ndarray | None:
    """Return the unlabeled three-ratio code used by the v3 quad hash."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return None
    diffs = pts[:, None, :] - pts[None, :, :]
    distances = np.hypot(diffs[..., 0], diffs[..., 1])
    eps = 1e-12
    ratios: list[float] = []
    for (a, b), (c, d) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        first = float(distances[a, b])
        second = float(distances[c, d])
        if first <= eps or second <= eps:
            return None
        ratios.append(min(first, second) / max(first, second))
    return np.sort(np.asarray(ratios, dtype=np.float64))


def _quad_diameter(points: np.ndarray) -> float:
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return float(np.nanmax(dists))


def _canonical_quad_order(combo: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return a non-crossing cyclic ordering for a quad.

    Sorting by distance-to-centroid can create crossing polygons (bow-ties),
    which collapses polygon area and rejects valid quads. The hash itself is
    permutation invariant, so this order is used only for area checks and
    downstream permutation trials; it must not depend on star identifiers.
    """
    idx = np.asarray(combo, dtype=np.int64)
    points = positions[idx]
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles, kind="stable")
    cyclic = idx[order]

    # Select one of the eight cyclic/reflected representations by geometry,
    # not by local star IDs or absolute orientation. This gives corresponding
    # image/catalog quads the same slot order when their shape is asymmetric.
    candidates: list[tuple[tuple[float, ...], np.ndarray]] = []
    for base in (cyclic, cyclic[::-1]):
        for shift in range(4):
            candidate = np.roll(base, -shift)
            p = positions[candidate]
            distances = (
                np.linalg.norm(p[0] - p[1]),
                np.linalg.norm(p[1] - p[2]),
                np.linalg.norm(p[2] - p[3]),
                np.linalg.norm(p[3] - p[0]),
                np.linalg.norm(p[0] - p[2]),
                np.linalg.norm(p[1] - p[3]),
            )
            scale = max(float(max(distances)), 1e-12)
            signature = tuple(round(float(value) / scale, 12) for value in distances)
            candidates.append((signature, candidate))
    return min(candidates, key=lambda item: item[0])[1].astype(np.uint16, copy=False)


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


def _sparse_triple_quads(order: np.ndarray, stars: np.ndarray, max_quads: int) -> np.ndarray:
    """Build quads from sparse triangle anchors + one extra star.

    This helps low-density fields where pairwise sampling may produce too few
    stable hypotheses. We keep generation bounded and deterministic.
    """
    if max_quads <= 0 or stars.shape[0] < 4:
        return np.zeros((0, 4), dtype=np.uint16)
    xy = np.column_stack((stars["x"], stars["y"])).astype(np.float64, copy=False)
    if not np.isfinite(xy).all():
        return np.zeros((0, 4), dtype=np.uint16)
    # Favor bright seeds, but keep enough breadth for sparse fields.
    seed_cap = min(len(order), max(8, int(np.sqrt(max_quads)) * 3))
    seeds = [int(v) for v in order[:seed_cap]]
    if not seeds:
        return np.zeros((0, 4), dtype=np.uint16)
    combos: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    neigh_k = min(max(6, int(np.sqrt(max_quads))), max(6, stars.shape[0] - 1))
    for seed in seeds:
        if len(combos) >= max_quads:
            break
        d2 = np.sum((xy - xy[seed]) ** 2, axis=1)
        nn = [int(i) for i in np.argsort(d2) if int(i) != seed][:neigh_k]
        if len(nn) < 3:
            continue
        for a, b in itertools.combinations(nn, 2):
            tri = (seed, int(a), int(b))
            cxy = (xy[tri[0]] + xy[tri[1]] + xy[tri[2]]) / 3.0
            dtri = np.sum((xy - cxy) ** 2, axis=1)
            # Try a few 4th-star candidates around centroid then farther out.
            near = [int(i) for i in np.argsort(dtri) if int(i) not in tri][:4]
            far = [int(i) for i in np.argsort(dtri)[::-1] if int(i) not in tri][:2]
            for d in near + far:
                q = tuple(sorted((tri[0], tri[1], tri[2], int(d))))
                if q in seen:
                    continue
                seen.add(q)
                combos.append(q)
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
    # The three ways to split four vertices into two opposite edge pairs are
    # intrinsic to the unlabeled quad. Sorting their bounded ratios makes the
    # code invariant to all 24 vertex permutations, reflection, rotation,
    # translation and scale.
    ratio_code = opposite_edge_ratio_code(points)
    if ratio_code is None:
        return None
    q1, q2, q3 = (_quantize_ratio(value) for value in ratio_code)

    # The low 16 bits are reserved for future schema-local flags. Parity is
    # intentionally absent: reflection must produce the same candidate code.
    hash_value = (q1 << 48) | (q2 << 32) | (q3 << 16)
    return hash_value, order




def _dedup_quads_preserve_order(quads: np.ndarray, max_quads: int) -> np.ndarray:
    if quads.size == 0:
        return np.zeros((0, 4), dtype=np.uint16)
    seen: set[tuple[int, int, int, int]] = set()
    kept: list[np.ndarray] = []
    for row in quads:
        key = tuple(int(v) for v in row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(np.asarray(row, dtype=np.uint16))
        if len(kept) >= max_quads:
            break
    if not kept:
        return np.zeros((0, 4), dtype=np.uint16)
    return np.vstack(kept).astype(np.uint16, copy=False)

def _priority_order_indices(stars: np.ndarray) -> np.ndarray:
    names = stars.dtype.names or ()
    if "mag" in names:
        return np.argsort(stars["mag"])
    if "flux" in names:
        return np.argsort(stars["flux"])[::-1]
    return np.arange(stars.shape[0])


def sample_quads(stars: np.ndarray, max_quads: int, strategy: str = "log_spaced") -> np.ndarray:
    """Return up to *max_quads* quads using robust pair-based sampling.

    Default (any non-legacy strategy):
      - iterate over stars ordered by brightness/flux
      - form base pairs (A,B) across logarithmically spaced distance bins
      - gather nearby companions around the A-B segment to pick C/D with stable relative geometry
      - interleave scales until *max_quads* are generated

    Legacy fallbacks (for backward compatibility):
      - "legacy_brightness": previous brightest-pool sampling
      - "legacy_local": previous bright-seed + nearest-neighbor sampling
    """
    if max_quads <= 0 or stars.shape[0] < 4:
        return np.zeros((0, 4), dtype=np.uint16)
    method = (strategy or "log_spaced").lower()
    priority_order = _priority_order_indices(stars)
    if method == "legacy_brightness":
        return _legacy_biased_brightness(priority_order, max_quads)
    if method == "legacy_local":
        return _legacy_local_brightness(priority_order, stars, max_quads)

    catalog_coverage_first = method in {"catalog_coverage_first", "coverage_first"}
    ring_coverage = method in {"catalog_ring_coverage", "ring_coverage"}
    sparse_preferred = method == "sparse_triples"
    sparse_auto = method in {"local_brightness", "log_spaced", "catalog_coverage_first", "coverage_first"} and stars.shape[0] <= 96
    sparse_quads = _sparse_triple_quads(priority_order, stars, max_quads) if (sparse_preferred or sparse_auto) else np.zeros((0, 4), dtype=np.uint16)

    positions = np.column_stack(
        (stars["x"].astype(np.float64), stars["y"].astype(np.float64))
    )
    finite_mask = np.isfinite(positions).all(axis=1)
    if finite_mask.sum() < 4:
        return np.zeros((0, 4), dtype=np.uint16)
    valid_positions = positions[finite_mask]
    index_map = np.nonzero(finite_mask)[0]
    lookup = np.full(stars.shape[0], -1, dtype=np.int32)
    lookup[index_map] = np.arange(index_map.shape[0], dtype=np.int32)
    seed_order = np.array(
        [lookup[idx] for idx in priority_order if lookup[idx] >= 0],
        dtype=np.int32,
    )
    if ring_coverage:
        if int(max_quads) >= 12000:
            spread_budget = max(1, int(max_quads) - 6000)
            hybrid_budget = min(6000, int(max_quads))
        else:
            spread_budget = int(max_quads)
            hybrid_budget = max(1, int(max_quads) // 3)
        spread_quads = generate_ring_coverage_quads(
            valid_positions,
            seed_order=seed_order,
            max_quads=spread_budget,
            base_pair_cap=6,
            per_seed_quads=20,
            per_base_quads=max_quads,
            base_selection="spread",
            companion_order="combinations",
        )
        hybrid_quads = generate_ring_coverage_quads(
            valid_positions,
            seed_order=seed_order,
            max_quads=hybrid_budget,
        )
        if spread_quads.size and hybrid_quads.size:
            quads = _dedup_quads_preserve_order(np.vstack((spread_quads, hybrid_quads)), max_quads)
        elif spread_quads.size:
            quads = spread_quads
        else:
            quads = hybrid_quads
        if quads.shape[0] < max_quads:
            fill = generate_coverage_first_quads(
                valid_positions,
                seed_order=seed_order,
                max_quads=max_quads,
            )
            if fill.size:
                quads = _dedup_quads_preserve_order(np.vstack((quads, fill)), max_quads)
    else:
        generator = generate_coverage_first_quads if catalog_coverage_first else generate_pairwise_quads
        quads = generator(
            valid_positions,
            seed_order=seed_order,
            max_quads=max_quads,
        )
    pairwise = index_map[quads] if quads.size else np.zeros((0, 4), dtype=np.uint16)

    if sparse_quads.size:
        if pairwise.size:
            pairwise = _dedup_quads_preserve_order(np.vstack((sparse_quads, pairwise)), max_quads)
        else:
            pairwise = _dedup_quads_preserve_order(sparse_quads, max_quads)

    # Robustness fallback: when pairwise sampling is too sparse (or empty),
    # blend with legacy local-neighborhood quads to keep enough hypotheses.
    min_target = max(64, min(max_quads, max_quads // 6 if max_quads >= 6 else max_quads))
    if pairwise.shape[0] < min_target:
        legacy = _legacy_local_brightness(priority_order, stars, max_quads)
        if legacy.size:
            if pairwise.size:
                merged = np.vstack((pairwise, legacy))
            else:
                merged = legacy
            return _dedup_quads_preserve_order(merged, max_quads)

    if pairwise.size == 0:
        return np.zeros((0, 4), dtype=np.uint16)
    return _dedup_quads_preserve_order(pairwise, max_quads)


def hash_quads(
    quads: np.ndarray,
    positions: np.ndarray,
    *,
    spec: QuadLevelSpec | None = None,
    return_source_indices: bool = False,
) -> QuadHash:
    """Hash quads with an unlabeled, similarity-invariant 8-bit ratio code."""
    valid = []
    hashes = []
    source_idx: list[int] | None = [] if return_source_indices else None
    for idx, combo in enumerate(quads):
        if len(set(combo)) < 4:
            continue
        ordered = _canonical_quad_order(combo, positions)
        result = _hash_from_indexes(ordered, positions, spec=spec)
        if result is None:
            continue
        hash_value, canonical = result
        valid.append(canonical)
        hashes.append(hash_value)
        if source_idx is not None:
            source_idx.append(int(idx))
    if not hashes:
        return QuadHash(
            np.zeros((0, 4), dtype=np.uint16),
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.int32) if return_source_indices else None,
        )
    return QuadHash(
        np.stack(valid),
        np.array(hashes, dtype=np.uint64),
        (np.array(source_idx, dtype=np.int32) if source_idx is not None else None),
    )
