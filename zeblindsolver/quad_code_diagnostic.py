from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .asterisms import hash_quads, opposite_edge_ratio_code


@dataclass(frozen=True, slots=True)
class AstrometryQuadCode:
    order: tuple[int, int, int, int]
    code: np.ndarray


@dataclass(frozen=True, slots=True)
class QuadCodeRecord:
    source_quad_index: int
    quad_indices: tuple[int, int, int, int]
    ordered_indices: tuple[int, int, int, int]
    code: np.ndarray
    ratio_hash: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryQuadCodeIndex:
    tile_key: str
    codes: np.ndarray
    records: tuple[QuadCodeRecord, ...]
    tree: cKDTree | None

    def query(self, code: np.ndarray, *, code_tol: float) -> list[tuple[int, float]]:
        arr = np.asarray(code, dtype=np.float64)
        if arr.shape != (4,) or self.tree is None:
            return []
        neighbor_ids = self.tree.query_ball_point(arr, r=float(code_tol))
        out: list[tuple[int, float]] = []
        for idx in neighbor_ids:
            dist = float(np.linalg.norm(arr - self.codes[int(idx)]))
            out.append((int(idx), dist))
        out.sort(key=lambda item: item[1])
        return out


def astrometry_ab_code_4d(points: np.ndarray, order: tuple[int, int, int, int] | None = None) -> np.ndarray | None:
    """Return Astrometry-like 4D code for ordered points A, B, C, D.

    The transform maps A to (0, 0) and B to (1, 1), matching the square-diagonal
    convention used by Astrometry.net quad codes. The returned vector is
    (Cx, Cy, Dx, Dy).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return None
    if order is not None:
        order_arr = np.asarray(order, dtype=np.int64)
        if order_arr.shape != (4,) or len(set(int(v) for v in order_arr)) != 4:
            return None
        pts = pts[order_arr]
    a, b, c, d = pts
    ab = b - a
    norm2 = float(np.dot(ab, ab))
    if norm2 <= 1e-24:
        return None
    rot = np.array([-ab[1], ab[0]], dtype=np.float64)
    axis_x = 0.5 * (ab + rot)
    axis_y = 0.5 * (ab - rot)
    denom = float(np.dot(axis_x, axis_x))
    if denom <= 1e-24:
        return None

    def project(point: np.ndarray) -> tuple[float, float]:
        rel = point - a
        return float(np.dot(rel, axis_x) / denom), float(np.dot(rel, axis_y) / denom)

    cx, cy = project(c)
    dx, dy = project(d)
    code = np.asarray([cx, cy, dx, dy], dtype=np.float64)
    if not np.all(np.isfinite(code)):
        return None
    return code


def astrometry_code_flip_parity(code: np.ndarray) -> np.ndarray:
    """Return the Astrometry parity-flipped code: swap x/y for C and D."""
    arr = np.asarray(code, dtype=np.float64)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)):
        raise ValueError("code must be a finite 4-vector")
    return arr[[1, 0, 3, 2]].copy()


def canonicalize_astrometry_abcd(points: np.ndarray, order: tuple[int, int, int, int] | None = None) -> AstrometryQuadCode | None:
    """Canonicalize an ordered AB/C/D quad with Astrometry-like invariants."""
    pts_all = np.asarray(points, dtype=np.float64)
    if pts_all.shape != (4, 2) or not np.all(np.isfinite(pts_all)):
        return None
    if order is None:
        order_arr = np.arange(4, dtype=np.int64)
    else:
        order_arr = np.asarray(order, dtype=np.int64)
        if order_arr.shape != (4,) or len(set(int(v) for v in order_arr)) != 4:
            return None
    local = pts_all[order_arr]
    code = astrometry_ab_code_4d(local)
    if code is None:
        return None
    if float(code[0] + code[2]) > 1.0:
        order_arr = order_arr[[1, 0, 2, 3]]
        local = pts_all[order_arr]
        code = astrometry_ab_code_4d(local)
        if code is None:
            return None
    if (float(code[0]) > float(code[2])) or (np.isclose(code[0], code[2]) and float(code[1]) > float(code[3])):
        order_arr = order_arr[[0, 1, 3, 2]]
        local = pts_all[order_arr]
        code = astrometry_ab_code_4d(local)
        if code is None:
            return None
    return AstrometryQuadCode(tuple(int(v) for v in order_arr), code)


def canonicalize_astrometry_longest_ab(points: np.ndarray) -> AstrometryQuadCode | None:
    """Pick the longest pair as AB, then apply AB/C/D canonicalization."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return None
    best_pair: tuple[int, int] | None = None
    best_dist2 = -1.0
    for a, b in combinations(range(4), 2):
        dist2 = float(np.sum((pts[a] - pts[b]) ** 2))
        if dist2 > best_dist2:
            best_dist2 = dist2
            best_pair = (a, b)
    if best_pair is None or best_dist2 <= 1e-24:
        return None
    rest = tuple(idx for idx in range(4) if idx not in best_pair)
    return canonicalize_astrometry_abcd(pts, (best_pair[0], best_pair[1], rest[0], rest[1]))


def current_opposite_edge_ratio_hash(points: np.ndarray) -> int | None:
    """Return the current ZeBlind v1 hash for a single quad."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return None
    hashed = hash_quads(np.asarray([[0, 1, 2, 3]], dtype=np.uint16), pts)
    if hashed.hashes.size == 0:
        return None
    return int(hashed.hashes[0])


def build_astrometry_quad_records(
    quads: np.ndarray,
    positions: np.ndarray,
) -> tuple[QuadCodeRecord, ...]:
    """Build Astrometry-like AB/C/D records for a set of quads."""
    quad_arr = np.asarray(quads, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.float64)
    if quad_arr.ndim != 2 or quad_arr.shape[1] != 4:
        return tuple()
    records: list[QuadCodeRecord] = []
    for source_idx, quad in enumerate(quad_arr):
        if np.any(quad < 0) or np.any(quad >= pos.shape[0]) or len(set(int(v) for v in quad)) != 4:
            continue
        local_points = pos[quad]
        canonical = canonicalize_astrometry_longest_ab(local_points)
        if canonical is None:
            continue
        ordered = tuple(int(quad[int(local_idx)]) for local_idx in canonical.order)
        records.append(
            QuadCodeRecord(
                source_quad_index=int(source_idx),
                quad_indices=tuple(int(v) for v in quad),
                ordered_indices=ordered,
                code=canonical.code.astype(np.float64, copy=True),
                ratio_hash=current_opposite_edge_ratio_hash(local_points),
            )
        )
    return tuple(records)


def build_memory_quad_code_index(
    quads: np.ndarray,
    positions: np.ndarray,
    *,
    tile_key: str,
) -> MemoryQuadCodeIndex:
    """Build an offline in-memory 4D quad-code index for one tile."""
    records = build_astrometry_quad_records(quads, positions)
    if not records:
        return MemoryQuadCodeIndex(
            tile_key=str(tile_key),
            codes=np.zeros((0, 4), dtype=np.float64),
            records=tuple(),
            tree=None,
        )
    codes = np.vstack([record.code for record in records]).astype(np.float64, copy=False)
    return MemoryQuadCodeIndex(
        tile_key=str(tile_key),
        codes=codes,
        records=records,
        tree=cKDTree(codes),
    )


def compare_quad_codes(points: np.ndarray) -> dict[str, object]:
    """Compare ZeBlind's current hash with the Astrometry-like 4D code."""
    pts = np.asarray(points, dtype=np.float64)
    ratio = opposite_edge_ratio_code(pts)
    ast = canonicalize_astrometry_longest_ab(pts)
    return {
        "opposite_edge_ratio_code": None if ratio is None else [float(v) for v in ratio],
        "opposite_edge_ratio_8bit_v1": current_opposite_edge_ratio_hash(pts),
        "astrometry_order": None if ast is None else list(ast.order),
        "astrometry_ab_code_4d": None if ast is None else [float(v) for v in ast.code],
    }
