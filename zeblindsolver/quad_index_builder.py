from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import math
import os
import concurrent.futures
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import threading

from .asterisms import hash_quads, sample_quads
from .levels import LEVEL_MAP

logger = logging.getLogger(__name__)
HASH_DIR = "hash_tables"
MANIFEST_FILENAME = "manifest.json"
_INDEX_CACHE: dict[tuple[Path, str], "QuadIndex"] = {}
# Prevent redundant concurrent loads of the same quad table
_INDEX_LOCK = threading.Lock()
# Cache manifest contents with a simple stat signature to avoid re-reading JSON repeatedly
_MANIFEST_CACHE: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
_MANIFEST_LOCK = threading.Lock()


def _normalize_ra(value: float) -> float:
    result = float(value) % 360.0
    return result + 360.0 if result < 0.0 else result


def _segments_for_interval(ra_min: float, ra_max: float) -> Tuple[Tuple[float, float], ...]:
    span = float(ra_max) - float(ra_min)
    if abs(span) >= 360.0:
        return ((0.0, 360.0),)
    start = _normalize_ra(ra_min)
    end = _normalize_ra(ra_max)
    wrapped = (end - start) % 360.0
    if math.isclose(wrapped, 0.0) and not math.isclose(span, 0.0):
        return ((0.0, 360.0),)
    if end >= start:
        return ((start, end),)
    return ((start, 360.0), (0.0, end))


def _segments_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return (a0 <= b1) and (b0 <= a1)


def _normalize_bounds_segments(raw_segments: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float], ...]:
    normalized: List[Tuple[float, float]] = []
    for segment in raw_segments:
        if len(segment) != 2:
            continue
        try:
            start = float(segment[0])
            end = float(segment[1])
        except (TypeError, ValueError):
            continue
        if math.isclose(start, 0.0, abs_tol=1e-6) and math.isclose(end, 360.0, abs_tol=1e-6):
            return ((0.0, 360.0),)
        normalized.extend(_segments_for_interval(start, end))
    return tuple(normalized)


def _cone_search_window(ra_deg: float, dec_deg: float, radius_deg: float) -> tuple[Tuple[Tuple[float, float], ...], float, float]:
    radius = min(max(float(radius_deg), 0.05), 180.0)
    dec_min = max(-90.0, float(dec_deg) - radius)
    dec_max = min(90.0, float(dec_deg) + radius)
    if abs(dec_deg) + radius >= 89.5 or radius >= 90.0:
        segments = ((0.0, 360.0),)
    else:
        cos_dec = max(math.cos(math.radians(dec_deg)), 1e-6)
        span = radius / cos_dec
        span = max(radius, min(span * 1.2, 360.0))
        ra_min = ra_deg - span
        ra_max = ra_deg + span
        segments = _segments_for_interval(ra_min, ra_max)
    return segments, dec_min, dec_max


def _entry_intersects_search_box(
    entry: dict[str, Any],
    ra_segments: Tuple[Tuple[float, float], ...],
    dec_min: float,
    dec_max: float,
) -> bool:
    bounds = entry.get("bounds") or {}
    try:
        tile_dec_min = float(bounds.get("dec_min", -90.0))
        tile_dec_max = float(bounds.get("dec_max", 90.0))
    except (TypeError, ValueError):
        tile_dec_min, tile_dec_max = -90.0, 90.0
    if tile_dec_max < dec_min or tile_dec_min > dec_max:
        return False
    raw_segments = bounds.get("ra_segments")
    if not raw_segments:
        return True
    segments = _normalize_bounds_segments(raw_segments)
    if not segments:
        return True
    if len(segments) == 1 and math.isclose(segments[0][0], 0.0, abs_tol=1e-6) and math.isclose(segments[0][1], 360.0, abs_tol=1e-6):
        return True
    for t_start, t_end in segments:
        for q_start, q_end in ra_segments:
            if _segments_overlap(t_start, t_end, q_start, q_end):
                return True
    return False


def select_tiles_in_cone(
    manifest: dict[str, Any],
    ra_deg: float | None,
    dec_deg: float | None,
    radius_deg: float | None,
) -> list[int]:
    """Return tile indices from *manifest* whose bounds intersect the given cone."""
    tiles = manifest.get("tiles", []) or []
    if ra_deg is None or dec_deg is None or radius_deg is None:
        return list(range(len(tiles)))
    try:
        ra = float(ra_deg)
        dec = float(dec_deg)
        radius = float(radius_deg)
    except (TypeError, ValueError):
        return list(range(len(tiles)))
    if not math.isfinite(ra) or not math.isfinite(dec) or radius <= 0.0:
        return list(range(len(tiles)))
    ra_segments, dec_min, dec_max = _cone_search_window(ra, dec, radius)
    selected: list[int] = []
    for idx, entry in enumerate(tiles):
        if _entry_intersects_search_box(entry, ra_segments, dec_min, dec_max):
            selected.append(idx)
    return selected


def _load_manifest(index_root: Path) -> dict[str, Any]:
    manifest_path = index_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    try:
        st = manifest_path.stat()
        sig = (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        sig = (0, 0)
    with _MANIFEST_LOCK:
        cached = _MANIFEST_CACHE.get(manifest_path)
        if cached and cached[0] == sig:
            return cached[1]
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        _MANIFEST_CACHE[manifest_path] = (sig, data)
        return data


def load_manifest(index_root: Path | str) -> dict[str, Any]:
    return _load_manifest(Path(index_root).expanduser().resolve())


def _tile_file_path(index_root: Path, entry: dict[str, Any]) -> Path:
    """Resolve a tile file path from the manifest entry.

    The manifest stores relative paths. Normalize any Windows-style backslashes
    to forward slashes so the index remains portable across platforms.
    """
    raw = entry.get("tile_file", "")
    # Normalize separators for cross-platform compatibility
    rel = str(raw).replace("\\", "/")
    candidate = index_root / rel
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def validate_index(index_root: Path | str, *, db_root: Path | str | None = None) -> Dict[str, Any]:
    """Validate index integrity.

    Returns a dict with keys:
      - manifest_ok: bool
      - present_quads: list[str]
      - missing_quads: list[str]
      - levels: list[str]
      - manifest_tile_count: int
      - db_root_mismatch: bool
      - db_tile_count: int | None
      - tile_key_mismatch: bool | None
      - empty_tiles_total: int
      - empty_ratio_overall: float
      - ring_empty_counts: dict[int, int]
      - ring_total_counts: dict[int, int]
      - bad_empty_rings: list[int]  # rings with >=80% empty tiles
    """
    root = Path(index_root).expanduser().resolve()
    result: Dict[str, Any] = {
        "manifest_ok": False,
        "present_quads": [],
        "missing_quads": [],
        "levels": [],
        "manifest_tile_count": 0,
        "db_root_mismatch": False,
        "db_tile_count": None,
        "tile_key_mismatch": None,
        "empty_tiles_total": 0,
        "empty_ratio_overall": 0.0,
        "ring_empty_counts": {},
        "ring_total_counts": {},
        "bad_empty_rings": [],
    }
    try:
        manifest = _load_manifest(root)
        result["manifest_ok"] = True
    except Exception:
        return result
    levels = [lvl.get("name") for lvl in manifest.get("levels", []) if isinstance(lvl, dict) and lvl.get("name")]
    result["levels"] = levels
    ht = root / HASH_DIR
    present: list[str] = []
    missing: list[str] = []
    for name in levels:
        path = ht / f"quads_{name}.npz"
        if path.exists():
            present.append(name)
        else:
            missing.append(name)
    result["present_quads"] = present
    result["missing_quads"] = missing
    tiles = manifest.get("tiles", [])
    result["manifest_tile_count"] = int(len(tiles))
    # Compute empty-tile stats per ring using manifest data when available
    empty_total = 0
    ring_empty: Dict[int, int] = {}
    ring_total: Dict[int, int] = {}
    for entry in tiles:
        try:
            # Prefer explicit tile_code if present; otherwise parse from tile_key suffix
            code = str(entry.get("tile_code") or str(entry.get("tile_key", "")).split("_", 1)[-1])
            ring_idx = int(code[:2]) if len(code) >= 2 and code[:2].isdigit() else -1
        except Exception:
            ring_idx = -1
        stars = entry.get("stars")
        is_empty = False
        if isinstance(stars, (int, np.integer)):
            is_empty = int(stars) == 0
        else:
            # Fallback: inspect tile file if manifest lacks a star count
            try:
                tile_path = _tile_file_path(root, entry)
                with np.load(tile_path) as data:
                    size = int(np.shape(data.get("ra_deg", ())) and data["ra_deg"].size)
                is_empty = size == 0
            except Exception:
                # Missing/unreadable tile counts as empty for the purpose of this health check
                is_empty = True
        empty_total += 1 if is_empty else 0
        if ring_idx not in ring_total:
            ring_total[ring_idx] = 0
            ring_empty[ring_idx] = 0
        ring_total[ring_idx] += 1
        if is_empty:
            ring_empty[ring_idx] += 1
    result["empty_tiles_total"] = empty_total
    total_tiles = max(1, int(len(tiles)))
    result["empty_ratio_overall"] = float(empty_total) / float(total_tiles)
    # Store per-ring stats, excluding unknown ring -1
    if -1 in ring_total:
        ring_total.pop(-1, None)
        ring_empty.pop(-1, None)
    result["ring_empty_counts"] = {int(k): int(v) for k, v in ring_empty.items()}
    result["ring_total_counts"] = {int(k): int(v) for k, v in ring_total.items()}
    # Identify rings with a large fraction of empty tiles (>=80%)
    bad_rings: list[int] = []
    for k, tot in ring_total.items():
        if tot <= 0:
            continue
        if float(ring_empty.get(k, 0)) / float(tot) >= 0.80:
            bad_rings.append(int(k))
    result["bad_empty_rings"] = sorted(bad_rings)
    # Compare DB content if requested
    if db_root is not None:
        try:
            from zewcs290.catalog290 import CatalogDB
            db = CatalogDB(db_root)
            db_keys = {tile.key for tile in db.tiles}
            result["db_tile_count"] = len(db_keys)
            manifest_keys = {t.get("tile_key") for t in tiles if isinstance(t, dict)}
            result["tile_key_mismatch"] = db_keys != manifest_keys
            # Detect root mismatch
            manifest_root = Path(str(manifest.get("db_root", "")).strip() or "").expanduser()
            try:
                result["db_root_mismatch"] = manifest_root.resolve() != Path(db_root).expanduser().resolve()
            except Exception:
                result["db_root_mismatch"] = True
        except Exception:
            # If DB cannot be read, leave comparison fields as-is
            pass
    return result


@dataclass
class QuadIndex:
    level: str
    hashes: np.ndarray
    tile_indices: np.ndarray
    quad_indices: np.ndarray
    bucket_hashes: np.ndarray
    bucket_offsets: np.ndarray
    bucket_cap: int

    @classmethod
    def load(cls, index_root: Path, level: str) -> "QuadIndex":
        key = (index_root.resolve(), level)
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        with _INDEX_LOCK:
            cached = _INDEX_CACHE.get(key)
            if cached is not None:
                return cached
            table_path = index_root / HASH_DIR / f"quads_{level}.npz"
            if not table_path.exists():
                raise FileNotFoundError(table_path)
            payload = np.load(table_path)
            spec = LEVEL_MAP.get(level)
            bucket_cap = spec.bucket_cap if spec else 0
            index = cls(
                level=level,
                hashes=payload["hashes"],
                tile_indices=payload["tile_indices"],
                quad_indices=payload["quad_indices"],
                bucket_hashes=payload["bucket_hashes"],
                bucket_offsets=payload["bucket_offsets"],
                bucket_cap=bucket_cap,
            )
            _INDEX_CACHE[key] = index
            return index


def _process_tile_for_level(
    tile_path: str,
    tile_key: str,
    tile_index: int,
    level_name: str,
    max_quads_per_tile: int,
) -> tuple[int, np.ndarray, np.ndarray] | None:
    try:
        with np.load(tile_path) as data:
            if "x_deg" not in data or "mag" not in data:
                return None
            coords = np.column_stack((data["x_deg"], data["y_deg"]))
            stars = np.zeros(data["mag"].shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
            stars["x"] = data["x_deg"].astype(np.float32)
            stars["y"] = data["y_deg"].astype(np.float32)
            stars["mag"] = data["mag"].astype(np.float32)
        from .levels import LEVEL_MAP  # local import is safe in subproc
        from .asterisms import sample_quads, hash_quads
        spec = LEVEL_MAP.get(level_name)
        strategy = "biased_brightness" if level_name == "L" else "local_brightness"
        quads = sample_quads(stars, max_quads_per_tile, strategy=strategy)
        if quads.size == 0:
            return None
        quad_hash = hash_quads(quads, coords, spec=spec)
        if quad_hash.hashes.size == 0:
            return None
        return tile_index, quad_hash.hashes, quad_hash.indices.astype(np.uint16)
    except Exception:
        return None


def build_quad_index(
    index_root: Path | str,
    level: str,
    *,
    max_quads_per_tile: int = 20000,
    on_progress: Callable[[int, int, str], None] | None = None,
    workers: int | None = None,
) -> Path:
    index_root = Path(index_root).expanduser().resolve()
    spec = LEVEL_MAP.get(level)
    if spec is None:
        raise ValueError(f"unknown level {level!r} (supported {list(LEVEL_MAP)})")
    manifest = _load_manifest(index_root)
    levels = {entry["name"]: entry for entry in manifest.get("levels", [])}
    if level not in levels:
        raise ValueError(f"unknown level {level!r}, manifest has {list(levels)}")
    hash_dir = index_root / HASH_DIR
    hash_dir.mkdir(parents=True, exist_ok=True)
    tile_entries = manifest.get("tiles", [])
    if not tile_entries:
        raise RuntimeError("manifest contains no tiles")
    hashes: list[np.ndarray] = []
    tile_indices: list[np.ndarray] = []
    quad_indices: list[np.ndarray] = []
    total = len(tile_entries)
    # Progress helpers
    done = 0
    def _record_progress(label: str) -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress(done, total, label)

    # Build task list (pre-resolve paths to avoid pickling manifest entries)
    tasks: list[tuple[str, str, int]] = []
    for tile_index, entry in enumerate(tile_entries):
        try:
            path = _tile_file_path(index_root, entry)
        except FileNotFoundError as exc:
            logger.warning("missing tile file %s: %s", entry.get("tile_file"), exc)
            _record_progress(str(entry.get("tile_key", "")))
            continue
        tasks.append((str(path), str(entry.get("tile_key", "")), tile_index))

    # Choose workers (default: half CPUs, min 1) if not provided
    if workers is None:
        try:
            workers = max(1, (os.cpu_count() or 1) // 2)
        except Exception:
            workers = 1

    if workers <= 1:
        for tile_path, tile_key, tile_index in tasks:
            result = _process_tile_for_level(tile_path, tile_key, tile_index, level, max_quads_per_tile)
            if result is None:
                _record_progress(tile_key)
                continue
            idx, h, q = result
            hashes.append(h)
            tile_indices.append(np.full(h.shape, idx, dtype=np.uint32))
            quad_indices.append(q)
            logger.debug("level %s: %s -> %d hashed quads", level, tile_key, h.size)
            _record_progress(tile_key)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as pool:
            future_map = {
                pool.submit(
                    _process_tile_for_level,
                    tile_path,
                    tile_key,
                    tile_index,
                    level,
                    max_quads_per_tile,
                ): (tile_key, tile_index)
                for (tile_path, tile_key, tile_index) in tasks
            }
            for fut in concurrent.futures.as_completed(future_map):
                tile_key, _idx = future_map[fut]
                try:
                    result = fut.result()
                except Exception:
                    result = None
                if result is None:
                    _record_progress(tile_key)
                    continue
                idx, h, q = result
                hashes.append(h)
                tile_indices.append(np.full(h.shape, idx, dtype=np.uint32))
                quad_indices.append(q)
                _record_progress(tile_key)
    if not hashes:
        raise RuntimeError("no quads were hashed for level %s" % level)
    all_hashes = np.concatenate(hashes)
    all_tiles = np.concatenate(tile_indices)
    all_quads = np.vstack(quad_indices)
    order = np.argsort(all_hashes)
    sorted_hashes = all_hashes[order]
    sorted_tiles = all_tiles[order]
    sorted_quads = all_quads[order]
    bucket_hashes, start_idx = np.unique(sorted_hashes, return_index=True)
    bucket_offsets = np.empty(len(bucket_hashes) + 1, dtype=np.uint32)
    bucket_offsets[:-1] = start_idx.astype(np.uint32)
    bucket_offsets[-1] = sorted_hashes.shape[0]
    out_path = hash_dir / f"quads_{level}.npz"
    np.savez_compressed(
        out_path,
        hashes=sorted_hashes,
        tile_indices=sorted_tiles,
        quad_indices=sorted_quads,
        bucket_hashes=bucket_hashes,
        bucket_offsets=bucket_offsets,
    )
    logger.info("built quad index %s with %d entries", out_path, sorted_hashes.shape[0])
    return out_path


def lookup_hashes(index_root: Path | str, level: str, keys: np.ndarray) -> list[slice]:
    index = QuadIndex.load(Path(index_root), level)
    buckets = index.bucket_hashes
    offsets = index.bucket_offsets
    positions = np.searchsorted(buckets, keys)
    result: list[slice] = []
    for key, pos in zip(keys, positions):
        if pos < buckets.shape[0] and buckets[pos] == key:
            start = int(offsets[pos])
            end = int(offsets[pos + 1])
            result.append(slice(start, end))
        else:
            result.append(slice(0, 0))
    return result
