"""In-memory quad-based blind index builder and matcher."""
from __future__ import annotations

import itertools
import json
import logging
import math
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

if __package__ is None:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from zewcs290 import CatalogDB
from zeblindsolver.quad_sampling import generate_pairwise_quads

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
HASH_PREFIX_BITS = 12
DEFAULT_BUCKET_CACHE_SIZE = 64
INDEX_MANIFEST_NAME = "manifest.npz"
BUCKET_DIR_NAME = "buckets"
QUAD_TILE_IDS_NAME = "quad_tile_ids.npy"
QUAD_QUANTIZED_NAME = "quad_quantized.npy"
QUAD_CATALOG_COORDS_NAME = "quad_catalog_coords.npy"
BUCKET_FILE_TEMPLATE = "bucket_{:04x}.npy"
BUCKET_RECORD_DTYPE = np.dtype([("hash", np.uint64), ("quad_index", np.uint32)])


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

    FORMAT_VERSION = 2

    def __init__(
        self,
        *,
        index_root: Path,
        tile_keys: Sequence[str],
        tile_families: Sequence[str],
        tile_centers: np.ndarray,
        tile_radii: np.ndarray,
        metadata: dict[str, object],
        bucket_bits: int,
        quad_tile_ids_path: Path,
        quad_quantized_path: Path,
        quad_catalog_coords_path: Path,
    ) -> None:
        self.index_root = Path(index_root)
        self.tile_keys = tuple(tile_keys)
        self.tile_families = tuple(tile_families)
        self.tile_centers = np.asarray(tile_centers, dtype=np.float32)
        self.tile_radii = np.asarray(tile_radii, dtype=np.float32)
        self.metadata = metadata
        self.bucket_bits = int(bucket_bits)
        if self.bucket_bits <= 0 or self.bucket_bits >= 32:
            raise ValueError(f"invalid bucket bit width: {self.bucket_bits}")
        self._bucket_shift = 64 - self.bucket_bits
        self.bucket_dir = self.index_root / BUCKET_DIR_NAME
        self._bucket_cache: OrderedDict[int, np.memmap] = OrderedDict()
        self._bucket_cache_size = DEFAULT_BUCKET_CACHE_SIZE
        self._quad_tile_ids_path = Path(quad_tile_ids_path)
        self._quad_quantized_path = Path(quad_quantized_path)
        self._quad_catalog_coords_path = Path(quad_catalog_coords_path)
        self._quad_tile_ids: Optional[np.memmap] = None
        self._quad_quantized: Optional[np.memmap] = None
        self._quad_catalog_coords: Optional[np.memmap] = None

    def _bucket_id(self, hash_value: int) -> int:
        return int((hash_value >> self._bucket_shift) & ((1 << self.bucket_bits) - 1))

    @staticmethod
    def _close_memmap(memmap_obj: Optional[np.memmap]) -> None:
        if memmap_obj is None:
            return
        mm = getattr(memmap_obj, "_mmap", None)
        if mm is not None:
            mm.close()

    def _get_bucket(self, bucket_id: int) -> Optional[np.memmap]:
        bucket = self._bucket_cache.get(bucket_id)
        if bucket is not None:
            self._bucket_cache.move_to_end(bucket_id)
            return bucket
        path = self.bucket_dir / BUCKET_FILE_TEMPLATE.format(bucket_id)
        if not path.exists():
            return None
        bucket = np.load(path, mmap_mode="r")
        self._bucket_cache[bucket_id] = bucket
        if len(self._bucket_cache) > self._bucket_cache_size:
            _, evicted = self._bucket_cache.popitem(last=False)
            self._close_memmap(evicted)
        return bucket

    def _load_quad_tile_ids(self) -> np.memmap:
        if self._quad_tile_ids is None:
            self._quad_tile_ids = np.load(self._quad_tile_ids_path, mmap_mode="r")
        return self._quad_tile_ids

    def _load_quad_quantized(self) -> np.memmap:
        if self._quad_quantized is None:
            self._quad_quantized = np.load(self._quad_quantized_path, mmap_mode="r")
        return self._quad_quantized

    def _load_quad_catalog_coords(self) -> np.memmap:
        if self._quad_catalog_coords is None:
            self._quad_catalog_coords = np.load(self._quad_catalog_coords_path, mmap_mode="r")
        return self._quad_catalog_coords

    @classmethod
    def load(cls, path: Path) -> "BlindIndex":
        root = Path(path)
        manifest_path = root / INDEX_MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"blind index manifest not found at {manifest_path}")
        with np.load(manifest_path, allow_pickle=False) as payload:
            tile_keys = payload["tile_keys"]
            tile_families = payload["tile_families"]
            tile_centers = payload["tile_centers"]
            tile_radii = payload["tile_radii"]
            metadata = json.loads(str(payload["metadata"][0]))
            bucket_bits = int(payload["bucket_bits"][0]) if "bucket_bits" in payload else HASH_PREFIX_BITS
            quad_tile_ids_rel = str(payload["quad_tile_ids_path"][0])
            quad_quantized_rel = str(payload["quad_quantized_path"][0])
            quad_catalog_rel = str(payload["quad_catalog_coords_path"][0])
            bucket_dir_name = str(payload["bucket_dir"][0]) if "bucket_dir" in payload else BUCKET_DIR_NAME
        bucket_dir = root / bucket_dir_name
        if not bucket_dir.exists():
            raise FileNotFoundError(f"bucket directory missing: {bucket_dir}")
        quad_count = int(metadata.get("quad_count", 0))
        logger.info(
            "Blind index ready at %s (%d quads across %d tiles)",
            root,
            quad_count,
            tile_keys.shape[0],
        )
        return cls(
            index_root=root,
            tile_keys=[str(value) for value in tile_keys],
            tile_families=[str(value) for value in tile_families],
            tile_centers=tile_centers,
            tile_radii=tile_radii,
            metadata=metadata,
            bucket_bits=bucket_bits,
            quad_tile_ids_path=root / quad_tile_ids_rel,
            quad_quantized_path=root / quad_quantized_rel,
            quad_catalog_coords_path=root / quad_catalog_rel,
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
        output_path = output_dir / f"{tag}_blind_index_{stamp}"

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
        quad_tile_ids = self._load_quad_tile_ids()
        quad_quantized = self._load_quad_quantized()
        for obs in quads:
            bucket_id = self._bucket_id(obs.hash_value)
            bucket = self._get_bucket(bucket_id)
            if bucket is None or bucket.size == 0:
                continue
            hashes = bucket["hash"]
            left = int(np.searchsorted(hashes, obs.hash_value, side="left"))
            right = int(np.searchsorted(hashes, obs.hash_value, side="right"))
            if left == right:
                continue
            candidate_indexes = np.asarray(bucket["quad_index"][left:right], dtype=np.uint32)
            if candidate_indexes.size == 0:
                continue
            quantized_candidates = np.asarray(quad_quantized[candidate_indexes])
            matches_mask = np.all(quantized_candidates == obs.quantized, axis=1)
            if not np.any(matches_mask):
                continue
            matched_indexes = candidate_indexes[matches_mask]
            if matched_indexes.size == 0:
                continue
            for quad_index in matched_indexes:
                tile_id = int(quad_tile_ids[int(quad_index)])
                hits.setdefault(tile_id, []).append(BlindMatch(quad_index=int(quad_index), observed=obs))
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
        bucket_bits: int = HASH_PREFIX_BITS,
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
        self.bucket_bits = int(bucket_bits)

    def write(self, path: Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        bucket_dir = destination / BUCKET_DIR_NAME
        bucket_dir.mkdir(parents=True, exist_ok=True)
        tile_key_dtype = f"<U{max(len(key) for key in self.tile_keys)}" if self.tile_keys else "<U1"
        tile_family_dtype = f"<U{max(len(key) for key in self.tile_families)}" if self.tile_families else "<U1"
        manifest_meta = dict(self.metadata)
        manifest_meta["hash_prefix_bits"] = self.bucket_bits
        manifest_meta["bucket_count"] = 1 << self.bucket_bits
        meta_text = json.dumps(manifest_meta, sort_keys=True)
        np.savez_compressed(
            destination / INDEX_MANIFEST_NAME,
            tile_keys=np.array(self.tile_keys, dtype=tile_key_dtype),
            tile_families=np.array(self.tile_families, dtype=tile_family_dtype),
            tile_centers=self.tile_centers.astype(np.float32, copy=False),
            tile_radii=self.tile_radii.astype(np.float32, copy=False),
            metadata=np.array([meta_text], dtype=f"<U{len(meta_text)}"),
            bucket_bits=np.array([self.bucket_bits], dtype=np.uint16),
            quad_tile_ids_path=np.array([QUAD_TILE_IDS_NAME], dtype=f"<U{len(QUAD_TILE_IDS_NAME)}"),
            quad_quantized_path=np.array([QUAD_QUANTIZED_NAME], dtype=f"<U{len(QUAD_QUANTIZED_NAME)}"),
            quad_catalog_coords_path=np.array([QUAD_CATALOG_COORDS_NAME], dtype=f"<U{len(QUAD_CATALOG_COORDS_NAME)}"),
            bucket_dir=np.array([BUCKET_DIR_NAME], dtype=f"<U{len(BUCKET_DIR_NAME)}"),
        )
        np.save(destination / QUAD_TILE_IDS_NAME, self.quad_tile_ids.astype(np.uint32, copy=False))
        np.save(destination / QUAD_QUANTIZED_NAME, self.quad_quantized.astype(np.uint16, copy=False))
        np.save(destination / QUAD_CATALOG_COORDS_NAME, self.quad_catalog_coords.astype(np.float32, copy=False))
        self._write_buckets(bucket_dir)

    def _write_buckets(self, bucket_dir: Path) -> None:
        bucket_dir = Path(bucket_dir)
        bucket_count = 1 << self.bucket_bits
        assignments: List[List[int]] = [[] for _ in range(bucket_count)]
        hashes = self.quad_hashes.astype(np.uint64, copy=False)
        shift = 64 - self.bucket_bits
        for idx, hash_value in enumerate(hashes):
            bucket_index = int((hash_value >> shift) & (bucket_count - 1))
            assignments[bucket_index].append(idx)
        for bucket_index, indexes in enumerate(assignments):
            if not indexes:
                continue
            idx_array = np.array(indexes, dtype=np.uint32)
            bucket_data = np.empty(idx_array.size, dtype=BUCKET_RECORD_DTYPE)
            bucket_data["hash"] = hashes[idx_array]
            bucket_data["quad_index"] = idx_array
            order = np.argsort(bucket_data["hash"], kind="mergesort")
            np.save(bucket_dir / BUCKET_FILE_TEMPLATE.format(bucket_index), bucket_data[order])


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
            finite_mask = np.isfinite(projected).all(axis=1)
            if finite_mask.sum() < 4:
                continue
            projected_valid = projected[finite_mask]
            selected_valid = selected[finite_mask]
            seed_order = np.arange(projected_valid.shape[0], dtype=np.int32)
            combos = generate_pairwise_quads(
                projected_valid,
                seed_order=seed_order,
                max_quads=self.max_quads_per_tile,
            )
            if combos.size == 0:
                continue
            tile_hashes: List[int] = []
            tile_quantized: List[np.ndarray] = []
            tile_coords: List[np.ndarray] = []
            for combo in combos:
                sample = projected_valid[list(combo)]
                descriptor = _describe_points(sample)
                if descriptor is None:
                    continue
                quantized, hash_value = descriptor
                tile_hashes.append(hash_value)
                tile_quantized.append(quantized)
                tile_coords.append(
                    np.column_stack(
                        (
                            selected_valid["ra_deg"][list(combo)],
                            selected_valid["dec_deg"][list(combo)],
                        )
                    ).astype(
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
            "hash_prefix_bits": HASH_PREFIX_BITS,
            "bucket_count": 1 << HASH_PREFIX_BITS,
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
            bucket_bits=HASH_PREFIX_BITS,
        )
