"""Near-solver catalog tile providers.

The provider boundary supplies only tile geometry and raw catalogue stars to
ZeNear.  It deliberately knows nothing about FITS inputs, RANSAC, WCS fitting,
GUI state, Blind 4D, or product settings.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .astap_db_reader import TileMeta, iter_tiles as iter_astap_tiles, load_tile_stars as load_astap_tile_stars
from .quad_index_builder import load_manifest


class NearCatalogProviderError(RuntimeError):
    """Raised when a Near catalog provider cannot serve its explicit source."""


@dataclass(frozen=True, slots=True)
class NearTileBounds:
    ra_segments: tuple[tuple[float, float], ...]
    dec_min: float
    dec_max: float

    @property
    def covers_full_ra(self) -> bool:
        return (
            len(self.ra_segments) == 1
            and math.isclose(self.ra_segments[0][0], 0.0, abs_tol=1e-6)
            and math.isclose(self.ra_segments[0][1], 360.0, abs_tol=1e-6)
        )


@dataclass(frozen=True, slots=True)
class NearCatalogTile:
    family: str
    tile_code: str
    center_ra_deg: float
    center_dec_deg: float
    bounds: NearTileBounds
    tile_key: str
    tile_file: str | None = None
    source: object | None = None

    def to_manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "tile_key": self.tile_key,
            "family": self.family,
            "tile_code": self.tile_code,
            "center_ra_deg": float(self.center_ra_deg),
            "center_dec_deg": float(self.center_dec_deg),
            "bounds": {
                "dec_min": float(self.bounds.dec_min),
                "dec_max": float(self.bounds.dec_max),
                "ra_segments": [list(segment) for segment in self.bounds.ra_segments],
            },
        }
        if self.tile_file:
            entry["tile_file"] = self.tile_file
        return entry

    def intersects_cone(self, ra_deg: float, dec_deg: float, radius_deg: float) -> tuple[bool, float]:
        return _tile_intersects(self, ra_deg, dec_deg, radius_deg)


@dataclass(frozen=True, slots=True)
class NearCatalogStars:
    """Raw catalog stars in degrees.

    Arrays are one-dimensional, same length, finite-filtering is left to the
    solver, RA is degrees in [0, 360), DEC is degrees, and magnitude follows the
    source catalogue band/order.
    """

    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray

    @property
    def size(self) -> int:
        return int(min(self.ra_deg.size, self.dec_deg.size, self.mag.size))


class NearCatalogProvider(Protocol):
    kind: str

    @property
    def families(self) -> tuple[str, ...]:
        ...

    def select_tiles(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        limit: int,
        *,
        families: Sequence[str] | None = None,
    ) -> tuple[NearCatalogTile, ...]:
        ...

    def load_stars(self, tile: NearCatalogTile) -> NearCatalogStars:
        ...

    def telemetry(self) -> dict[str, object]:
        ...


def _normalize_families(families: Sequence[str] | None) -> tuple[str, ...]:
    if not families:
        return ()
    return tuple(dict.fromkeys(str(fam).strip().lower() for fam in families if str(fam).strip()))


def _angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad = math.radians(float(ra1))
    ra2_rad = math.radians(float(ra2))
    dec1_rad = math.radians(float(dec1))
    dec2_rad = math.radians(float(dec2))
    cos_c = (
        math.sin(dec1_rad) * math.sin(dec2_rad)
        + math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad)
    )
    return math.degrees(math.acos(min(1.0, max(-1.0, cos_c))))


def _wrap_ra(delta: float) -> float:
    return (float(delta) + 540.0) % 360.0 - 180.0


def _ra_segments_for_interval(ra_min: float, ra_max: float) -> tuple[tuple[float, float], ...]:
    span = float(ra_max) - float(ra_min)
    if span >= 360.0:
        return ((0.0, 360.0),)
    start = float(ra_min) % 360.0
    end = float(ra_max) % 360.0
    if start <= end:
        return ((start, end),)
    return ((start, 360.0), (0.0, end))


def _segments_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(float(a0), float(b0)) <= min(float(a1), float(b1))


def _tile_extent(tile: NearCatalogTile) -> float:
    dec_span = abs(float(tile.bounds.dec_max) - float(tile.bounds.dec_min))
    ra_span = 0.0
    for start, end in tile.bounds.ra_segments:
        ra_span = max(ra_span, abs(_wrap_ra(float(end) - float(start))))
    cos_dec = max(math.cos(math.radians(float(tile.center_dec_deg))), 1e-3)
    return 0.5 * max(dec_span, ra_span * cos_dec)


def _tile_intersects(tile: NearCatalogTile, ra_deg: float, dec_deg: float, radius_deg: float) -> tuple[bool, float]:
    distance = _angular_distance(tile.center_ra_deg, tile.center_dec_deg, ra_deg, dec_deg)
    radius = float(radius_deg)
    if (float(dec_deg) + radius) < float(tile.bounds.dec_min) or (float(dec_deg) - radius) > float(tile.bounds.dec_max):
        return False, distance
    if tile.bounds.ra_segments:
        cosd = max(1e-3, math.cos(math.radians(float(dec_deg))))
        ra_span = radius / cosd
        query_segments = _ra_segments_for_interval(float(ra_deg) - ra_span, float(ra_deg) + ra_span)
        for start, end in tile.bounds.ra_segments:
            parts = ((start, end),) if start <= end else ((start, 360.0), (0.0, end))
            for p0, p1 in parts:
                if any(_segments_overlap(p0, p1, q0, q1) for q0, q1 in query_segments):
                    return True, distance
        return False, distance
    return distance <= radius + max(_tile_extent(tile), 0.25), distance


def _bounds_from_manifest(entry: dict[str, Any]) -> NearTileBounds:
    raw = entry.get("bounds") or {}
    raw_segments = raw.get("ra_segments") or ()
    segments: list[tuple[float, float]] = []
    for segment in raw_segments:
        if isinstance(segment, Sequence) and len(segment) >= 2:
            segments.append((float(segment[0]), float(segment[1])))
    return NearTileBounds(
        ra_segments=tuple(segments),
        dec_min=float(raw.get("dec_min", entry.get("center_dec_deg", 0.0))),
        dec_max=float(raw.get("dec_max", entry.get("center_dec_deg", 0.0))),
    )


def _bounds_from_astap(meta: TileMeta) -> NearTileBounds:
    return NearTileBounds(
        ra_segments=tuple((float(start), float(end)) for start, end in meta.bounds.ra_segments),
        dec_min=float(meta.bounds.dec_min),
        dec_max=float(meta.bounds.dec_max),
    )


class _MemoryStarCache:
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[str, NearCatalogStars] = OrderedDict()

    def get(self, key: str) -> NearCatalogStars | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.pop(key)
        self._items[key] = value
        return NearCatalogStars(value.ra_deg.copy(), value.dec_deg.copy(), value.mag.copy())

    def put(self, key: str, value: NearCatalogStars) -> NearCatalogStars:
        stored = NearCatalogStars(
            np.asarray(value.ra_deg, dtype=np.float64).copy(),
            np.asarray(value.dec_deg, dtype=np.float64).copy(),
            np.asarray(value.mag, dtype=np.float32).copy(),
        )
        self._items[key] = stored
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
        return NearCatalogStars(stored.ra_deg.copy(), stored.dec_deg.copy(), stored.mag.copy())


class LegacyIndexNearCatalogProvider:
    kind = "legacy_index"

    def __init__(self, index_root: Path | str, *, cache_size: int = 128) -> None:
        self.index_root = Path(index_root).expanduser().resolve()
        self.manifest = load_manifest(self.index_root)
        self.db_root = Path(str(self.manifest.get("db_root", "")).strip()).expanduser().resolve() if self.manifest.get("db_root") else None
        self._tiles = tuple(self._tile_from_entry(entry) for entry in (self.manifest.get("tiles") or ()))
        if not self._tiles:
            raise NearCatalogProviderError("legacy index manifest has no tiles")
        self._cache = _MemoryStarCache(cache_size)
        self._fallback_used = False
        self._fallback_reason: str | None = None

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({tile.family for tile in self._tiles if tile.family}))

    def _tile_from_entry(self, entry: dict[str, Any]) -> NearCatalogTile:
        family = str(entry.get("family") or "").strip().lower()
        tile_code = str(entry.get("tile_code") or "")
        tile_key = str(entry.get("tile_key") or (f"{family}_{tile_code}" if family and tile_code else tile_code))
        return NearCatalogTile(
            family=family,
            tile_code=tile_code,
            center_ra_deg=float(entry.get("center_ra_deg", 0.0)),
            center_dec_deg=float(entry.get("center_dec_deg", 0.0)),
            bounds=_bounds_from_manifest(entry),
            tile_key=tile_key,
            tile_file=str(entry.get("tile_file") or "") or None,
            source=dict(entry),
        )

    def select_tiles(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        limit: int,
        *,
        families: Sequence[str] | None = None,
    ) -> tuple[NearCatalogTile, ...]:
        family_filter = set(_normalize_families(families))
        selected: list[tuple[NearCatalogTile, float]] = []
        for tile in self._tiles:
            if family_filter and tile.family not in family_filter:
                continue
            intersects, distance = tile.intersects_cone(ra_deg, dec_deg, radius_deg)
            if intersects:
                selected.append((tile, distance))
        selected.sort(key=lambda item: item[1])
        cap = max(1, int(limit)) if int(limit or 0) > 0 else len(selected)
        return tuple(tile for tile, _distance in selected[:cap])

    def load_stars(self, tile: NearCatalogTile) -> NearCatalogStars:
        cache_key = f"legacy:{tile.tile_key}:{tile.tile_file or ''}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tile_path = (self.index_root / str(tile.tile_file or "").replace("\\", "/")).resolve() if tile.tile_file else None
        try:
            if tile_path is not None:
                with np.load(tile_path, allow_pickle=False) as data:
                    stars = NearCatalogStars(
                        np.asarray(data["ra_deg"], dtype=np.float64),
                        np.asarray(data["dec_deg"], dtype=np.float64),
                        np.asarray(data["mag"], dtype=np.float32),
                    )
            else:
                raise FileNotFoundError(str(tile.tile_key))
        except Exception as exc:
            if self.db_root is None:
                raise NearCatalogProviderError(f"failed to read legacy tile {tile.tile_key}: {exc}") from exc
            try:
                self._fallback_used = True
                self._fallback_reason = f"legacy_tile_load_failed:{type(exc).__name__}"
                stars = self._load_astap_fallback(tile)
            except Exception as fallback_exc:
                raise NearCatalogProviderError(
                    f"failed to read legacy tile {tile.tile_key}: {exc}; ASTAP fallback failed: {fallback_exc}"
                ) from fallback_exc
        return self._cache.put(cache_key, stars)

    def _load_astap_fallback(self, tile: NearCatalogTile) -> NearCatalogStars:
        assert self.db_root is not None
        for meta in iter_astap_tiles(self.db_root):
            if str(meta.family).strip().lower() != tile.family:
                continue
            if str(meta.tile_code) != str(tile.tile_code):
                continue
            raw = load_astap_tile_stars(self.db_root, meta)
            return NearCatalogStars(
                np.asarray(raw["ra_deg"], dtype=np.float64),
                np.asarray(raw["dec_deg"], dtype=np.float64),
                np.asarray(raw["mag"], dtype=np.float32),
            )
        raise FileNotFoundError(f"ASTAP fallback tile not found: {tile.family}_{tile.tile_code}")

    def telemetry(self) -> dict[str, object]:
        return {
            "near_catalog_provider": self.kind,
            "near_catalog_family": list(self.families),
            "near_catalog_fallback_used": bool(self._fallback_used),
            "near_catalog_fallback_reason": self._fallback_reason,
        }


class AstapNearCatalogProvider:
    kind = "astap_native"

    def __init__(self, db_root: Path | str, *, families: Sequence[str] | None = None, cache_size: int = 128) -> None:
        self.db_root = Path(db_root).expanduser().resolve()
        requested = set(_normalize_families(families))
        all_tiles: list[NearCatalogTile] = []
        for meta in iter_astap_tiles(self.db_root):
            family = str(meta.family).strip().lower()
            if requested and family not in requested:
                continue
            all_tiles.append(
                NearCatalogTile(
                    family=family,
                    tile_code=str(meta.tile_code),
                    center_ra_deg=float(meta.center_ra_deg),
                    center_dec_deg=float(meta.center_dec_deg),
                    bounds=_bounds_from_astap(meta),
                    tile_key=str(meta.key),
                    tile_file=None,
                    source=meta,
                )
            )
        if not all_tiles:
            detail = ",".join(sorted(requested)) if requested else "all"
            raise NearCatalogProviderError(f"ASTAP catalog has no usable Near tiles for families={detail}")
        available = {tile.family for tile in all_tiles}
        missing = requested - available
        if missing:
            raise NearCatalogProviderError(f"ASTAP catalog missing requested family/families: {', '.join(sorted(missing))}")
        self._tiles = tuple(all_tiles)
        self._cache = _MemoryStarCache(cache_size)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({tile.family for tile in self._tiles}))

    def select_tiles(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        limit: int,
        *,
        families: Sequence[str] | None = None,
    ) -> tuple[NearCatalogTile, ...]:
        family_filter = set(_normalize_families(families))
        selected: list[tuple[NearCatalogTile, float]] = []
        for tile in self._tiles:
            if family_filter and tile.family not in family_filter:
                continue
            intersects, distance = tile.intersects_cone(ra_deg, dec_deg, radius_deg)
            if intersects:
                selected.append((tile, distance))
        selected.sort(key=lambda item: (item[1], item[0].family, item[0].tile_code))
        cap = max(1, int(limit)) if int(limit or 0) > 0 else len(selected)
        return tuple(tile for tile, _distance in selected[:cap])

    def load_stars(self, tile: NearCatalogTile) -> NearCatalogStars:
        if not isinstance(tile.source, TileMeta):
            raise NearCatalogProviderError(f"ASTAP tile has no TileMeta source: {tile.tile_key}")
        cache_key = f"astap:{tile.tile_key}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            raw = load_astap_tile_stars(self.db_root, tile.source)
        except Exception as exc:
            raise NearCatalogProviderError(f"failed to read ASTAP tile {tile.tile_key}: {exc}") from exc
        stars = NearCatalogStars(
            np.asarray(raw["ra_deg"], dtype=np.float64),
            np.asarray(raw["dec_deg"], dtype=np.float64),
            np.asarray(raw["mag"], dtype=np.float32),
        )
        return self._cache.put(cache_key, stars)

    def telemetry(self) -> dict[str, object]:
        return {
            "near_catalog_provider": self.kind,
            "near_catalog_family": list(self.families),
            "near_catalog_fallback_used": False,
        }
