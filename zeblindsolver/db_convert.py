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


def _project_tan(ra_deg: np.ndarray, dec_deg: np.ndarray, center_ra: float, center_dec: float) -> tuple[np.ndarray, np.ndarray]:
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    ra0_rad = np.deg2rad(center_ra)
    dec0_rad = np.deg2rad(center_dec)
    dra = (ra_rad - ra0_rad + math.pi) % (2 * math.pi) - math.pi
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec0 = math.sin(dec0_rad)
    cos_dec0 = math.cos(dec0_rad)
    cos_dra = np.cos(dra)
    sin_dra = np.sin(dra)
    cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    safe = cosc > 1e-8
    x = np.empty_like(ra_rad)
    y = np.empty_like(dec_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        x[~safe] = np.nan
        y[~safe] = np.nan
        x[safe] = (cos_dec[safe] * sin_dra[safe]) / cosc[safe]
        y[safe] = (cos_dec0 * sin_dec[safe] - sin_dec0 * cos_dec[safe] * cos_dra[safe]) / cosc[safe]
    return np.degrees(x), np.degrees(y)


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
) -> Path:
    db_root = Path(db_root).expanduser().resolve()
    index_root = Path(index_root).expanduser().resolve()
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_root / MANIFEST_FILENAME
    tile_entries = []
    tiles = list(iter_tiles(db_root))
    logger.info("Starting conversion of %d tiles from %s", len(tiles), db_root)
    for tile_index, tile_meta in enumerate(_progress_iter(tiles, desc="convert")):
        stars = load_tile_stars(db_root, tile_meta)
        total = len(stars)
        ra = stars["ra_deg"].astype(np.float64, copy=False)
        dec = stars["dec_deg"].astype(np.float64, copy=False)
        x_deg, y_deg = _project_tan(ra, dec, tile_meta.center_ra_deg, tile_meta.center_dec_deg)
        valid = np.isfinite(x_deg) & np.isfinite(y_deg)
        stars = stars[valid]
        x_deg = x_deg[valid]
        y_deg = y_deg[valid]
        if mag_cap is not None:
            mask_mag = stars["mag"] <= mag_cap
            stars = stars[mask_mag]
            x_deg = x_deg[mask_mag]
            y_deg = y_deg[mask_mag]
        if stars.size > max_stars:
            order = np.argsort(stars["mag"])
            order = order[:max_stars]
            stars = stars[order]
            x_deg = x_deg[order]
            y_deg = y_deg[order]
        tile_path = tiles_dir / f"{tile_meta.key}.npz"
        np.savez_compressed(
            tile_path,
            ra_deg=stars["ra_deg"].astype(np.float64, copy=False),
            dec_deg=stars["dec_deg"].astype(np.float64, copy=False),
            mag=stars["mag"].astype(np.float32, copy=False),
            x_deg=x_deg.astype(np.float32, copy=False),
            y_deg=y_deg.astype(np.float32, copy=False),
        )
        tile_entries.append(
            {
                "tile_index": tile_index,
                "tile_key": tile_meta.key,
                "family": tile_meta.family,
                "tile_code": tile_meta.tile_code,
                "center_ra_deg": tile_meta.center_ra_deg,
                "center_dec_deg": tile_meta.center_dec_deg,
                "bounds": {
                    "dec_min": tile_meta.bounds.dec_min,
                    "dec_max": tile_meta.bounds.dec_max,
                    "ra_segments": [[float(a), float(b)] for a, b in tile_meta.bounds.ra_segments],
                },
                "stars": int(stars.size),
                "tile_file": str(tile_path.relative_to(index_root)),
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
    for level in LEVEL_SPECS:
        build_quad_index(index_root, level.name, max_quads_per_tile=max_quads_per_tile)
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
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    try:
        build_index_from_astap(
            args.db_root,
            args.index_root,
            mag_cap=args.mag_cap,
            max_stars=args.max_stars,
            max_quads_per_tile=args.max_quads_per_tile,
        )
        return 0
    except Exception as exc:  # pragma: no cover - bubble error
        logger.exception("Index build failed: %s", exc)
        return 1


zebuildindex = main
