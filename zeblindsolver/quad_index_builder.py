from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Callable

import numpy as np

from .asterisms import hash_quads, sample_quads
from .levels import LEVEL_MAP

logger = logging.getLogger(__name__)
HASH_DIR = "hash_tables"
MANIFEST_FILENAME = "manifest.json"
_INDEX_CACHE: dict[tuple[Path, str], "QuadIndex"] = {}


def _load_manifest(index_root: Path) -> dict[str, Any]:
    manifest_path = index_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        if key in _INDEX_CACHE:
            return _INDEX_CACHE[key]
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


def build_quad_index(
    index_root: Path | str,
    level: str,
    *,
    max_quads_per_tile: int = 20000,
    on_progress: Callable[[int, int, str], None] | None = None,
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
    for tile_index, entry in enumerate(tile_entries):
        try:
            tile_path = _tile_file_path(index_root, entry)
        except FileNotFoundError as exc:
            logger.warning("missing tile file %s: %s", entry.get("tile_file"), exc)
            if on_progress:
                on_progress(tile_index + 1, total, str(entry.get("tile_key", "")))
            continue
        with np.load(tile_path) as data:
            if "x_deg" not in data or "mag" not in data:
                logger.warning("tile %s missing x_deg/mag arrays", tile_path)
                if on_progress:
                    on_progress(tile_index + 1, total, str(entry.get("tile_key", "")))
                continue
            coords = np.column_stack((data["x_deg"], data["y_deg"]))
            stars = np.zeros(data["mag"].shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
            stars["x"] = data["x_deg"].astype(np.float32)
            stars["y"] = data["y_deg"].astype(np.float32)
            stars["mag"] = data["mag"].astype(np.float32)
        # Use locality-aware sampling for smaller-diameter levels to avoid overly large quads
        strategy = "biased_brightness" if level == "L" else "local_brightness"
        quads = sample_quads(stars, max_quads_per_tile, strategy=strategy)
        if quads.size == 0:
            if on_progress:
                on_progress(tile_index + 1, total, str(entry.get("tile_key", "")))
            continue
        quad_hash = hash_quads(quads, coords, spec=spec)
        if quad_hash.hashes.size == 0:
            if on_progress:
                on_progress(tile_index + 1, total, str(entry.get("tile_key", "")))
            continue
        hashes.append(quad_hash.hashes)
        tile_indices.append(np.full(quad_hash.hashes.shape, tile_index, dtype=np.uint32))
        quad_indices.append(quad_hash.indices.astype(np.uint16))
        logger.debug("level %s: %s -> %d hashed quads", level, entry.get("tile_key"), quad_hash.hashes.size)
        if on_progress:
            on_progress(tile_index + 1, total, str(entry.get("tile_key", "")))
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
