"""In-memory quad-based blind index builder and matcher."""
from __future__ import annotations

import itertools
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

if __package__ is None:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from zewcs290 import CatalogDB

logger = logging.getLogger(__name__)

DESCRIPTOR_RANGES: Tuple[Tuple[float, float], ...] = (
    (-0.5, 2.5),   # u coordinate (point 2)
    (-1.5, 1.5),   # v coordinate (point 2)
    (-0.5, 2.5),   # u coordinate (point 3)
    (-1.5, 1.5),   # v coordinate (point 3)
    (0.0, 3.0),    # distance between the last two vertices relative to base
)
DESCRIPTOR_DIM = len(DESCRIPTOR_RANGES)
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
DEFAULT_MAX_TILE_STARS = 40
DEFAULT_MAX_QUADS_PER_TILE = 256
DEFAULT_OBSERVED_MAX_STARS = 35
DEFAULT_OBSERVED_MAX_QUADS = 2500
DEFAULT_QUERY_MIN_SCORE = 3
DEFAULT_QUERY_MAX_TILES = 15


@dataclass(slots=True)
class ObservedQuad:
    """Quad assembled from detected image stars (unit + pixel coordinates)."""

    indices: Tuple[int, int, int, int]
    unit_points: np.ndarray  # shape (4, 2)
    pixel_points: np.ndarray  # shape (4, 2)
    quantized: np.ndarray  # shape (DESCRIPTOR_DIM,), dtype=uint16
    hash_value: int


@dataclass(slots=True)
class BlindMatch:
    quad_index: int
    observed: ObservedQuad


@dataclass(slots=True)
class BlindIndexCandidate:
    tile_key: str
    tile_family: str
    center_ra_deg: float
    center_dec_deg: float
    radius_deg: float
    score: int
    matches: List[BlindMatch]


def _normalize_value(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return (value - lo) / (hi - lo)


def _quantize_descriptor(values: Sequence[float]) -> np.ndarray:
    quants = np.empty(DESCRIPTOR_DIM, dtype=np.uint16)
    for idx, raw in enumerate(values):
        normalized = _normalize_value(raw, DESCRIPTOR_RANGES[idx])
        quants[idx] = int(round(normalized * 65535.0))
    return quants


def _hash_quantized(values: np.ndarray) -> int:
    hash_value = FNV_OFFSET
    for val in values:
        hash_value ^= int(val)
        hash_value = (hash_value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def _pairwise_dist2(points: np.ndarray) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]
    return np.einsum("...i,...i->...", diffs, diffs)


def _order_quad_vertices(points: np.ndarray) -> Optional[np.ndarray]:
    dist2 = _pairwise_dist2(points)
    scores = dist2.sum(axis=1)

    def _key(idx: int) -> Tuple[float, Tuple[float, float, float]]:
        row = np.delete(dist2[idx], idx)
        return (float(scores[idx]), tuple(sorted(float(val) for val in row)))

    order = sorted(range(4), key=_key)
    if len({tuple(points[i]) for i in order}) < 4:
        return None
    return np.array(order, dtype=np.int32)


def _describe_points(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if points.shape != (4, 2):
        raise ValueError("quad descriptor expects a (4,2) array")
    order = _order_quad_vertices(points)
    if order is None:
        return None
    ordered = points[order]
    a, b, c, d = ordered
    base = b - a
    base_len = float(np.hypot(base[0], base[1]))
    if base_len < 1e-6:
        return None
    axis_x = base / base_len
    axis_y = np.array([-axis_x[1], axis_x[0]])

    def _project(vertex: np.ndarray) -> Tuple[float, float]:
        delta = vertex - a
        u = float(np.dot(delta, axis_x)) / base_len
        v = float(np.dot(delta, axis_y)) / base_len
        return u, v

    uc, vc = _project(c)
    ud, vd = _project(d)
    dist_cd = float(np.linalg.norm(c - d) / base_len)
    descriptor = np.array([uc, vc, ud, vd, dist_cd], dtype=np.float32)
    quantized = _quantize_descriptor(descriptor)
    hash_value = _hash_quantized(quantized)
    return quantized, hash_value


def build_observed_quads(
    unit_points: np.ndarray,
    pixel_points: np.ndarray,
    *,
    max_stars: int = DEFAULT_OBSERVED_MAX_STARS,
    max_quads: int = DEFAULT_OBSERVED_MAX_QUADS,
) -> List[ObservedQuad]:
    if unit_points.shape != pixel_points.shape:
        raise ValueError("unit_points and pixel_points must have matching shapes")
    count = min(len(unit_points), len(pixel_points), max_stars)
    if count < 4:
        return []
    indices = list(range(count))
    quads: List[ObservedQuad] = []
    for combo in itertools.combinations(indices, 4):
        sample = unit_points[list(combo)]
        descriptor = _describe_points(sample)
        if descriptor is None:
            continue
        quantized, hash_value = descriptor
        quads.append(
            ObservedQuad(
                indices=combo,
                unit_points=unit_points[list(combo)],
                pixel_points=pixel_points[list(combo)],
                quantized=quantized,
                hash_value=hash_value,
            )
        )
        if len(quads) >= max_quads:
            break
    return quads


def _tile_center(tile) -> Tuple[float, float]:
    segments = tile.bounds.ra_segments
    if tile.bounds.covers_full_ra or not segments:
        ra_center = 0.0
    else:
        total_weight = 0.0
        x_sum = 0.0
        y_sum = 0.0
        for start, end in segments:
            span = (end - start) if end >= start else (end + 360.0 - start)
            mid = (start + 0.5 * span) % 360.0
            rad = math.radians(mid)
            x_sum += math.cos(rad) * span
            y_sum += math.sin(rad) * span
            total_weight += span
        ra_center = math.degrees(math.atan2(y_sum, x_sum)) % 360.0 if total_weight else 0.0
    dec_center = tile.bounds.dec_center
    return ra_center, dec_center


def _tile_radius(tile, center_ra: float, center_dec: float) -> float:
    dec_half = 0.5 * (tile.bounds.dec_max - tile.bounds.dec_min)
    max_ra_span = 0.0
    for start, end in tile.bounds.ra_segments:
        span = (end - start) if end >= start else (end + 360.0 - start)
        max_ra_span = max(max_ra_span, span)
    projected_ra = max_ra_span * math.cos(math.radians(center_dec))
    return 0.5 * math.hypot(projected_ra, 2.0 * dec_half)


def _project_catalog_points(stars: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    dra = (stars["ra_deg"] - ra0 + 540.0) % 360.0 - 180.0
    xi = dra * math.cos(math.radians(dec0))
    eta = stars["dec_deg"] - dec0
    return np.column_stack((xi, eta)).astype(np.float32)


class BlindIndex:
    """Quad-based matcher backed by a pre-computed catalogue index."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        tile_keys: Sequence[str],
        tile_families: Sequence[str],
        tile_centers: np.ndarray,
        tile_radii: np.ndarray,
        quad_tile_ids: np.ndarray,
        quad_hashes: np.ndarray,
        quad_quantized: np.ndarray,
        quad_catalog_coords: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        self.tile_keys = tuple(tile_keys)
        self.tile_families = tuple(tile_families)
        self.tile_centers = np.asarray(tile_centers, dtype=np.float32)
        self.tile_radii = np.asarray(tile_radii, dtype=np.float32)
        self.quad_tile_ids = np.asarray(quad_tile_ids, dtype=np.uint32)
        self.quad_hashes = np.asarray(quad_hashes, dtype=np.uint64)
        self.quad_quantized = np.asarray(quad_quantized, dtype=np.uint16)
        self.quad_catalog_coords = np.asarray(quad_catalog_coords, dtype=np.float32)
        self.metadata = metadata
        self._buckets = self._build_buckets()

    def _build_buckets(self) -> dict[int, List[int]]:
        buckets: dict[int, List[int]] = {}
        for idx, hash_value in enumerate(self.quad_hashes):
            buckets.setdefault(int(hash_value), []).append(idx)
        return buckets

    @classmethod
    def load(cls, path: Path) -> "BlindIndex":
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            tile_keys = payload["tile_keys"]
            tile_families = payload["tile_families"]
            tile_centers = payload["tile_centers"]
            tile_radii = payload["tile_radii"]
            quad_tile_ids = payload["quad_tile_ids"]
            quad_hashes = payload["quad_hashes"]
            quad_quantized = payload["quad_quantized"]
            quad_catalog_coords = payload["quad_catalog_coords"]
            metadata = json.loads(str(payload["metadata"][0]))
        logger.info(
            "Loaded blind index from %s (%d quads across %d tiles)",
            path,
            quad_hashes.shape[0],
            tile_keys.shape[0],
        )
        return cls(
            tile_keys=[str(value) for value in tile_keys],
            tile_families=[str(value) for value in tile_families],
            tile_centers=tile_centers,
            tile_radii=tile_radii,
            quad_tile_ids=quad_tile_ids,
            quad_hashes=quad_hashes,
            quad_quantized=quad_quantized,
            quad_catalog_coords=quad_catalog_coords,
            metadata=metadata,
        )

    @classmethod
    def build_from_catalog(
        cls,
        db_root: Path | str,
        *,
        families: Optional[Sequence[str]],
        output_dir: Path,
        max_tile_stars: int = DEFAULT_MAX_TILE_STARS,
        max_quads_per_tile: int = DEFAULT_MAX_QUADS_PER_TILE,
    ) -> Path:
        db = CatalogDB(db_root, families=families)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "-".join(db.families) if db.families else "all"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{tag}_blind_index_{stamp}.npz"

        builder = _IndexBuilder(
            db=db,
            max_tile_stars=max_tile_stars,
            max_quads_per_tile=max_quads_per_tile,
        )
        payload = builder.build()
        payload.write(output_path)
        logger.info(
            "Blind index (%d quads, %d tiles) ready at %s",
            payload.quad_hashes.shape[0],
            len(payload.tile_keys),
            output_path,
        )
        return output_path

    def query(
        self,
        quads: Sequence[ObservedQuad],
        *,
        min_score: int = DEFAULT_QUERY_MIN_SCORE,
        max_tiles: int = DEFAULT_QUERY_MAX_TILES,
    ) -> List[BlindIndexCandidate]:
        if not quads:
            return []
        hits: dict[int, List[BlindMatch]] = {}
        scores: dict[int, int] = {}
        for obs in quads:
            candidates = self._buckets.get(obs.hash_value)
            if not candidates:
                continue
            for quad_index in candidates:
                if not np.array_equal(obs.quantized, self.quad_quantized[quad_index]):
                    continue
                tile_id = int(self.quad_tile_ids[quad_index])
                hits.setdefault(tile_id, []).append(BlindMatch(quad_index=quad_index, observed=obs))
                scores[tile_id] = scores.get(tile_id, 0) + 1
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results: List[BlindIndexCandidate] = []
        for tile_id, score in ordered:
            if score < min_score:
                continue
            candidate = BlindIndexCandidate(
                tile_key=self.tile_keys[tile_id],
                tile_family=self.tile_families[tile_id],
                center_ra_deg=float(self.tile_centers[tile_id, 0]),
                center_dec_deg=float(self.tile_centers[tile_id, 1]),
                radius_deg=float(self.tile_radii[tile_id]),
                score=score,
                matches=hits[tile_id],
            )
            results.append(candidate)
            if len(results) >= max_tiles:
                break
        return results


class _IndexPayload:
    def __init__(
        self,
        *,
        tile_keys: List[str],
        tile_families: List[str],
        tile_centers: np.ndarray,
        tile_radii: np.ndarray,
        quad_tile_ids: np.ndarray,
        quad_hashes: np.ndarray,
        quad_quantized: np.ndarray,
        quad_catalog_coords: np.ndarray,
        metadata: dict[str, object],
    ):
        self.tile_keys = tile_keys
        self.tile_families = tile_families
        self.tile_centers = tile_centers
        self.tile_radii = tile_radii
        self.quad_tile_ids = quad_tile_ids
        self.quad_hashes = quad_hashes
        self.quad_quantized = quad_quantized
        self.quad_catalog_coords = quad_catalog_coords
        self.metadata = metadata

    def write(self, path: Path) -> None:
        tile_key_dtype = f"<U{max(len(key) for key in self.tile_keys)}" if self.tile_keys else "<U1"
        tile_family_dtype = f"<U{max(len(key) for key in self.tile_families)}" if self.tile_families else "<U1"
        meta_text = json.dumps(self.metadata, sort_keys=True)
        np.savez_compressed(
            path,
            tile_keys=np.array(self.tile_keys, dtype=tile_key_dtype),
            tile_families=np.array(self.tile_families, dtype=tile_family_dtype),
            tile_centers=self.tile_centers.astype(np.float32, copy=False),
            tile_radii=self.tile_radii.astype(np.float32, copy=False),
            quad_tile_ids=self.quad_tile_ids.astype(np.uint32, copy=False),
            quad_hashes=self.quad_hashes.astype(np.uint64, copy=False),
            quad_quantized=self.quad_quantized.astype(np.uint16, copy=False),
            quad_catalog_coords=self.quad_catalog_coords.astype(np.float32, copy=False),
            metadata=np.array([meta_text], dtype=f"<U{len(meta_text)}"),
        )


class _IndexBuilder:
    def __init__(self, *, db: CatalogDB, max_tile_stars: int, max_quads_per_tile: int) -> None:
        self.db = db
        self.max_tile_stars = max(4, max_tile_stars)
        self.max_quads_per_tile = max(1, max_quads_per_tile)

    def build(self) -> _IndexPayload:
        tile_keys: List[str] = []
        tile_families: List[str] = []
        tile_centers: List[Tuple[float, float]] = []
        tile_radii: List[float] = []
        quad_tile_ids: List[int] = []
        quad_hashes: List[int] = []
        quad_quantized: List[np.ndarray] = []
        quad_catalog_coords: List[np.ndarray] = []

        total_quads = 0
        for tile in self.db.tiles:
            block = self.db._load_tile(tile)  # pylint: disable=protected-access
            stars = block.stars
            if stars.size < 4:
                continue
            order = np.argsort(stars["mag"])
            selected = stars[order[: self.max_tile_stars]]
            if selected.size < 4:
                continue
            ra_center, dec_center = _tile_center(tile)
            projected = _project_catalog_points(selected, ra_center, dec_center)
            combos = itertools.combinations(range(selected.shape[0]), 4)
            tile_hashes: List[int] = []
            tile_quantized: List[np.ndarray] = []
            tile_coords: List[np.ndarray] = []
            for combo in combos:
                sample = projected[list(combo)]
                descriptor = _describe_points(sample)
                if descriptor is None:
                    continue
                quantized, hash_value = descriptor
                tile_hashes.append(hash_value)
                tile_quantized.append(quantized)
                tile_coords.append(
                    np.column_stack((selected["ra_deg"][list(combo)], selected["dec_deg"][list(combo)])).astype(
                        np.float32
                    )
                )
                if len(tile_hashes) >= self.max_quads_per_tile:
                    break
            if not tile_hashes:
                continue
            tile_id = len(tile_keys)
            quad_tile_ids.extend([tile_id] * len(tile_hashes))
            quad_hashes.extend(tile_hashes)
            quad_quantized.extend(tile_quantized)
            quad_catalog_coords.extend(tile_coords)
            total_quads += len(tile_hashes)
            tile_keys.append(tile.key)
            tile_families.append(tile.spec.key)
            tile_centers.append((ra_center, dec_center))
            tile_radii.append(_tile_radius(tile, ra_center, dec_center))

        metadata = {
            "version": BlindIndex.FORMAT_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "db_root": str(self.db.root),
            "families": list(self.db.families),
            "max_tile_stars": self.max_tile_stars,
            "max_quads_per_tile": self.max_quads_per_tile,
            "quad_count": total_quads,
            "tile_count": len(tile_keys),
        }

        return _IndexPayload(
            tile_keys=tile_keys,
            tile_families=tile_families,
            tile_centers=np.array(tile_centers, dtype=np.float32),
            tile_radii=np.array(tile_radii, dtype=np.float32),
            quad_tile_ids=np.array(quad_tile_ids, dtype=np.uint32),
            quad_hashes=np.array(quad_hashes, dtype=np.uint64),
            quad_quantized=np.vstack(quad_quantized) if quad_quantized else np.empty((0, DESCRIPTOR_DIM), dtype=np.uint16),
            quad_catalog_coords=np.array(quad_catalog_coords, dtype=np.float32),
            metadata=metadata,
        )
