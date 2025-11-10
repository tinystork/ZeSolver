from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from zewcs290.catalog290 import CatalogDB, CatalogTile, SkyBox

logger = logging.getLogger(__name__)

_TileCache = {}

@dataclass(frozen=True, slots=True)
class TileMeta:
    """Metadata for one ASTAP tile after conversion to TAN coordinates."""

    key: str
    family: str
    tile_code: str
    path: Path
    center_ra_deg: float
    center_dec_deg: float
    bounds: SkyBox
    ring_index: int
    tile_index: int


def _catalog_for_root(db_root: Path | str) -> CatalogDB:
    root = Path(db_root).expanduser().resolve()
    if root in _TileCache:
        return _TileCache[root]
    catalog = CatalogDB(root)
    _TileCache[root] = catalog
    logger.debug("ASTAP catalog loaded: %s (%d tiles)", root, len(catalog.tiles))
    return catalog


def _ra_center_from_segments(bounds: SkyBox) -> float:
    if bounds.covers_full_ra:
        return 0.0
    segments = bounds.ra_segments
    if not segments:
        return 0.0
    total_weight = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for start, end in segments:
        span = (end - start) if end >= start else (end + 360.0 - start)
        if span <= 0:
            continue
        mid = (start + 0.5 * span) % 360.0
        rad = math.radians(mid)
        x_sum += math.cos(rad) * span
        y_sum += math.sin(rad) * span
        total_weight += span
    if total_weight == 0:
        return 0.0
    angle = math.degrees(math.atan2(y_sum, x_sum))
    return angle % 360.0


def iter_tiles(db_root: Path | str) -> Iterator[TileMeta]:
    """Yield metadata for every ASTAP tile found under *db_root*."""
    catalog = _catalog_for_root(db_root)
    for tile in catalog.tiles:
        center_ra = _ra_center_from_segments(tile.bounds)
        center_dec = tile.bounds.dec_center
        yield TileMeta(
            key=tile.key,
            family=tile.spec.key,
            tile_code=tile.tile_code,
            path=tile.path,
            center_ra_deg=center_ra,
            center_dec_deg=center_dec,
            bounds=tile.bounds,
            ring_index=tile.ring_index,
            tile_index=tile.tile_index,
        )


def load_tile_stars(db_root: Path | str, tile_meta: TileMeta) -> np.ndarray:
    """Return the field-array of (ra_deg, dec_deg, mag) stored in the ASTAP tile.

    The result has the dtype defined by :data:`zewcs290.catalog290.STAR_DTYPE`,
    the rows are sorted as they appear on disk, and ``mag`` is Gaia BP/Johnson-V.

    """
    catalog = _catalog_for_root(db_root)
    target = str(tile_meta.path)
    for tile in catalog.tiles:
        if str(tile.path) == target:
            block = catalog._load_tile(tile)
            return block.stars
    raise FileNotFoundError(f"tile {tile_meta.key} ({tile_meta.path}) not in {db_root}")
