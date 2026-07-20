"""Coverage helpers for catalogue library status calculation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import CatalogCoverage, CoverageStatus


def coverage_from_payload(payload: object, *, provenance: str | None = None) -> CatalogCoverage:
    if not isinstance(payload, dict):
        return CatalogCoverage(status=CoverageStatus.UNKNOWN, provenance=provenance)
    status = _coverage_status(str(payload.get("status") or "UNKNOWN"))
    tile_keys = tuple(str(v) for v in (payload.get("tile_keys") or ()) if str(v))
    families = tuple(str(v).lower() for v in (payload.get("families") or ()) if str(v))
    ra_segments = tuple(
        (float(seg[0]), float(seg[1]))
        for seg in (payload.get("ra_segments_deg") or ())
        if isinstance(seg, (list, tuple)) and len(seg) == 2
    )
    return CatalogCoverage(
        status=status,
        all_sky=bool(payload.get("all_sky", False)),
        families=families,
        tile_keys=tile_keys,
        dec_min_deg=_float_or_none(payload.get("dec_min_deg")),
        dec_max_deg=_float_or_none(payload.get("dec_max_deg")),
        ra_segments_deg=ra_segments,
        covered_tiles=int(payload.get("covered_tiles") or len(set(tile_keys))),
        total_tiles=_int_or_none(payload.get("total_tiles")),
        fraction=_float_or_none(payload.get("fraction")),
        provenance=provenance,
        notes=str(payload.get("notes") or ""),
    )


def merge_coverages(coverages: Iterable[CatalogCoverage]) -> CatalogCoverage:
    items = tuple(coverages)
    if not items:
        return CatalogCoverage(status=CoverageStatus.MISSING)
    statuses = {item.status for item in items}
    all_sky = all(item.all_sky for item in items) and bool(items)
    families = tuple(sorted({family for item in items for family in item.families}))
    tile_keys = tuple(sorted({tile for item in items for tile in item.tile_keys}))
    dec_values_min = [item.dec_min_deg for item in items if item.dec_min_deg is not None]
    dec_values_max = [item.dec_max_deg for item in items if item.dec_max_deg is not None]
    if CoverageStatus.CORRUPT in statuses:
        status = CoverageStatus.CORRUPT
    elif CoverageStatus.INCOMPATIBLE in statuses:
        status = CoverageStatus.INCOMPATIBLE
    elif all_sky and statuses <= {CoverageStatus.FULL, CoverageStatus.PARTIAL}:
        status = CoverageStatus.FULL
    elif tile_keys or any(item.status is CoverageStatus.PARTIAL for item in items):
        status = CoverageStatus.PARTIAL
    elif any(item.status is CoverageStatus.UNKNOWN for item in items):
        status = CoverageStatus.UNKNOWN
    else:
        status = CoverageStatus.MISSING
    by_family: dict[str, int] = defaultdict(int)
    total_by_family: dict[str, int] = {}
    for item in items:
        if item.families and item.covered_tiles:
            for family in item.families:
                by_family[family] += item.covered_tiles
        if item.families and item.total_tiles is not None:
            for family in item.families:
                total_by_family[family] = max(total_by_family.get(family, 0), item.total_tiles)
    total_tiles = sum(total_by_family.values()) or None
    covered_tiles = len(tile_keys) if tile_keys else sum(by_family.values())
    fraction = (covered_tiles / total_tiles) if total_tiles else None
    return CatalogCoverage(
        status=status,
        all_sky=all_sky,
        families=families,
        tile_keys=tile_keys,
        dec_min_deg=min(dec_values_min) if dec_values_min else None,
        dec_max_deg=max(dec_values_max) if dec_values_max else None,
        covered_tiles=covered_tiles,
        total_tiles=total_tiles,
        fraction=fraction,
        provenance="merged",
    )


def coverage_for_tiles(
    *,
    family: str,
    tile_keys: Iterable[str],
    total_tiles: int | None,
    dec_min_deg: float | None = None,
    dec_max_deg: float | None = None,
    provenance: str | None = None,
) -> CatalogCoverage:
    tiles = tuple(sorted(set(str(v) for v in tile_keys if str(v))))
    covered = len(tiles)
    all_sky = bool(total_tiles and covered >= total_tiles)
    status = CoverageStatus.FULL if all_sky else CoverageStatus.PARTIAL if covered else CoverageStatus.MISSING
    return CatalogCoverage(
        status=status,
        all_sky=all_sky,
        families=(family.lower(),),
        tile_keys=tiles,
        dec_min_deg=dec_min_deg,
        dec_max_deg=dec_max_deg,
        covered_tiles=covered,
        total_tiles=total_tiles,
        fraction=(covered / total_tiles) if total_tiles else None,
        provenance=provenance,
    )


def _coverage_status(value: str) -> CoverageStatus:
    try:
        return CoverageStatus(value)
    except ValueError:
        return CoverageStatus.UNKNOWN


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return None
