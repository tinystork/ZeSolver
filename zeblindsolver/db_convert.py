from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ is None:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from zeblindsolver.astap_db_reader import TileMeta, iter_tiles, load_tile_stars
from zeblindsolver.levels import LEVEL_SPECS
from zeblindsolver.projections import project_tan
from zeblindsolver.quad_index_builder import build_quad_index

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional CLI progress
    tqdm = None

logger = logging.getLogger(__name__)
MANIFEST_FILENAME = "manifest.json"
DEFAULT_MAG_CAP = 15.5
DEFAULT_MAX_STARS = 2000
DEFAULT_MAX_QUADS_PER_TILE = 20000
_BOUND_MARGIN_DEG = 0.1


def _cartesian_center(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[float, float]:
    if ra_deg.size == 0:
        return 0.0, 0.0
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    cos_dec = np.cos(dec_rad)
    x = np.sum(cos_dec * np.cos(ra_rad))
    y = np.sum(cos_dec * np.sin(ra_rad))
    z = np.sum(np.sin(dec_rad))
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return float(np.mean(ra_deg)), float(np.mean(dec_deg))
    vx = x / norm
    vy = y / norm
    vz = z / norm
    ra = math.degrees(math.atan2(vy, vx)) % 360.0
    vz = min(1.0, max(-1.0, vz))
    dec = math.degrees(math.asin(vz))
    return ra, dec


def _serialize_bounds(bounds) -> dict[str, list[list[float]] | float]:
    return {
        "dec_min": float(bounds.dec_min),
        "dec_max": float(bounds.dec_max),
        "ra_segments": [[float(a), float(b)] for a, b in bounds.ra_segments],
    }


def _bounds_from_points(ra_deg: np.ndarray, dec_deg: np.ndarray, center_ra: float) -> dict[str, object]:
    if ra_deg.size == 0:
        return {
            "dec_min": 0.0,
            "dec_max": 0.0,
            "ra_segments": [[0.0, 360.0]],
        }
    dec_min = float(np.min(dec_deg)) - _BOUND_MARGIN_DEG
    dec_max = float(np.max(dec_deg)) + _BOUND_MARGIN_DEG
    offsets = ((ra_deg - center_ra + 540.0) % 360.0) - 180.0
    lo = float(np.min(offsets)) - _BOUND_MARGIN_DEG
    hi = float(np.max(offsets)) + _BOUND_MARGIN_DEG
    span = hi - lo
    if span >= 359.0:
        segments = [(0.0, 360.0)]
    else:
        ra_min = (center_ra + lo) % 360.0
        ra_max = (center_ra + hi) % 360.0
        if ra_min <= ra_max:
            segments = [(ra_min, ra_max)]
        else:
            segments = [(ra_min, 360.0), (0.0, ra_max)]
    return {
        "dec_min": max(-90.0, dec_min),
        "dec_max": min(90.0, dec_max),
        "ra_segments": [[float(a), float(b)] for a, b in segments],
    }


def _progress_iter(items: Iterable, *, desc: str) -> Iterable:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, leave=False)


def build_index_from_astap(
    db_root: Path | str,
    index_root: Path | str,
    *,
    mag_cap: float = DEFAULT_MAG_CAP,
    max_stars: int = DEFAULT_MAX_STARS,
    max_quads_per_tile: int = DEFAULT_MAX_QUADS_PER_TILE,
    skip_quads: bool = False,
    quads_only: bool = False,
    quad_storage: str = "npz",
    tile_compression: str = "compressed",
    workers: int | None = None,
) -> Path:
    db_root = Path(db_root).expanduser().resolve()
    index_root = Path(index_root).expanduser().resolve()
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    if skip_quads and quads_only:
        raise ValueError("skip_quads and quads_only cannot be combined")
    compression = (tile_compression or "compressed").lower()
    if compression not in {"compressed", "uncompressed"}:
        raise ValueError("tile_compression must be 'compressed' or 'uncompressed'")
    quad_fmt = (quad_storage or "npz").lower()
    if quad_fmt not in {"npz", "npz_uncompressed", "npy"}:
        raise ValueError("quad_storage must be 'npz', 'npz_uncompressed', or 'npy'")
    save_tile = np.savez_compressed if compression == "compressed" else np.savez
    manifest_path = index_root / MANIFEST_FILENAME
    if quads_only:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing at {manifest_path} (required for quads-only rebuild)")
        if not tiles_dir.is_dir():
            raise FileNotFoundError(f"{tiles_dir} missing; run a full zebuildindex first")
        logger.info("Skipping tile conversion; rebuilding quad tables only in %s", index_root)
    else:
        tile_entries = []
        tiles = list(iter_tiles(db_root))
        logger.info("Starting conversion of %d tiles from %s", len(tiles), db_root)
        for tile_index, tile_meta in enumerate(_progress_iter(tiles, desc="convert")):
            stars = load_tile_stars(db_root, tile_meta)
            total = len(stars)
            if total:
                ra = stars["ra_deg"].astype(np.float64, copy=False)
                dec = stars["dec_deg"].astype(np.float64, copy=False)
                center_ra_deg, center_dec_deg = _cartesian_center(ra, dec)
                x_deg, y_deg = project_tan(ra, dec, center_ra_deg, center_dec_deg)
                valid = np.isfinite(x_deg) & np.isfinite(y_deg)
                if not valid.any():
                    logger.warning(
                        "tile %s: cartesian center %.2f/%.2f yielded no TAN projection, falling back to layout center",
                        tile_meta.key,
                        center_ra_deg,
                        center_dec_deg,
                    )
                    center_ra_deg = tile_meta.center_ra_deg
                    center_dec_deg = tile_meta.center_dec_deg
                    x_deg, y_deg = project_tan(ra, dec, center_ra_deg, center_dec_deg)
                    valid = np.isfinite(x_deg) & np.isfinite(y_deg)
                stars = stars[valid]
                ra = ra[valid]
                dec = dec[valid]
                x_deg = x_deg[valid]
                y_deg = y_deg[valid]
            else:
                center_ra_deg = tile_meta.center_ra_deg
                center_dec_deg = tile_meta.center_dec_deg
                ra = np.empty(0, dtype=np.float64)
                dec = np.empty(0, dtype=np.float64)
                x_deg = np.empty(0, dtype=np.float32)
                y_deg = np.empty(0, dtype=np.float32)
            if mag_cap is not None:
                mask_mag = stars["mag"] <= mag_cap
                stars = stars[mask_mag]
                x_deg = x_deg[mask_mag]
                y_deg = y_deg[mask_mag]
                ra = ra[mask_mag]
                dec = dec[mask_mag]
            if stars.size > max_stars:
                order = np.argsort(stars["mag"])
                order = order[:max_stars]
                stars = stars[order]
                x_deg = x_deg[order]
                y_deg = y_deg[order]
                ra = ra[order]
                dec = dec[order]
            tile_path = tiles_dir / f"{tile_meta.key}.npz"
            save_tile(
                tile_path,
                ra_deg=stars["ra_deg"].astype(np.float64, copy=False),
                dec_deg=stars["dec_deg"].astype(np.float64, copy=False),
                mag=stars["mag"].astype(np.float32, copy=False),
                x_deg=x_deg.astype(np.float32, copy=False),
                y_deg=y_deg.astype(np.float32, copy=False),
            )
            if stars.size:
                bounds = _bounds_from_points(ra, dec, center_ra_deg)
            else:
                bounds = _serialize_bounds(tile_meta.bounds)
            tile_entries.append(
                {
                    "tile_index": tile_index,
                    "tile_key": tile_meta.key,
                    "family": tile_meta.family,
                    "tile_code": tile_meta.tile_code,
                    "center_ra_deg": float(center_ra_deg),
                    "center_dec_deg": float(center_dec_deg),
                    "bounds": bounds,
                    "stars": int(stars.size),
                    # Store relative path using POSIX separators for cross-platform portability
                    "tile_file": tile_path.relative_to(index_root).as_posix(),
                    "usable_ratio": float(stars.size / total) if total else 0.0,
                }
            )
            logger.debug(
                "converted %s -> %s (%d stars of %d total)",
                tile_meta.key,
                tile_path,
                stars.size,
                total,
            )
        manifest = {
            "version": 1,
            "mag_cap": mag_cap,
            "max_stars": max_stars,
            "levels": [level.to_manifest() for level in LEVEL_SPECS],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "db_root": str(db_root),
            "tile_count": len(tile_entries),
            "tiles": tile_entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Index manifest written to %s", manifest_path)
    if skip_quads:
        return manifest_path
    for level in LEVEL_SPECS:
        build_quad_index(
            index_root,
            level.name,
            max_quads_per_tile=max_quads_per_tile,
            workers=workers,
            storage_format=quad_fmt,
        )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert ASTAP tiles to a blind solver index")
    parser.add_argument("--db-root", required=True, help="Path to ASTAP catalog root")
    parser.add_argument("--index-root", required=True, help="Path where the index will be stored")
    parser.add_argument("--mag-cap", type=float, default=DEFAULT_MAG_CAP, help="Maximum magnitude to keep per tile")
    parser.add_argument(
        "--max-stars",
        type=int,
        default=DEFAULT_MAX_STARS,
        help="Maximum stars to keep per tile after filtering",
    )
    parser.add_argument(
        "--max-quads-per-tile",
        type=int,
        default=DEFAULT_MAX_QUADS_PER_TILE,
        help="Maximum quads hashed per tile",
    )
    parser.add_argument("--skip-quads", action="store_true", help="Only write manifest/tiles and skip quad hashing")
    parser.add_argument(
        "--quads-only",
        action="store_true",
        help="Reuse existing tiles/manifest and rebuild only quad hash tables",
    )
    parser.add_argument(
        "--quad-storage",
        choices=("npz", "npz_uncompressed", "npy"),
        default="npz",
        help="Storage format for quad hash tables",
    )
    parser.add_argument(
        "--tile-compression",
        choices=("compressed", "uncompressed"),
        default="compressed",
        help="NPZ compression mode for tiles/*.npz",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for quad hashing (default: half of CPUs)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    try:
        if args.skip_quads and args.quads_only:
            parser.error("--quads-only cannot be combined with --skip-quads")
        build_index_from_astap(
            args.db_root,
            args.index_root,
            mag_cap=args.mag_cap,
            max_stars=args.max_stars,
            max_quads_per_tile=args.max_quads_per_tile,
            skip_quads=bool(args.skip_quads),
            quads_only=bool(args.quads_only),
            quad_storage=args.quad_storage,
            tile_compression=args.tile_compression,
            workers=args.workers,
        )
        return 0
    except Exception as exc:  # pragma: no cover - bubble error
        logger.exception("Index build failed: %s", exc)
        return 1


zebuildindex = main
