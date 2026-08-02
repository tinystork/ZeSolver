"""Near-solver catalog tile providers.

The provider boundary supplies only tile geometry and raw catalogue stars to
ZeNear.  It deliberately knows nothing about FITS inputs, RANSAC, WCS fitting,
GUI state, Blind 4D, or product settings.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .astap_db_reader import TileMeta, _ra_center_from_segments, iter_tiles as iter_astap_tiles, load_tile_stars as load_astap_tile_stars
from .quad_index_builder import load_manifest
from zewcs290.catalog290 import CatalogDB, CatalogTile


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


MetricCallback = Callable[[str, int], None]


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
    def __init__(self, max_entries: int = 128, *, metrics_callback: MetricCallback | None = None) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[str, NearCatalogStars] = OrderedDict()
        self._inflight: dict[str, _InflightLoad] = {}
        self._lock = threading.Lock()
        self._metrics_callback = metrics_callback

    def _metric(self, key: str, amount: int = 1) -> None:
        if self._metrics_callback is not None:
            self._metrics_callback(key, amount)

    def get(self, key: str) -> NearCatalogStars | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.pop(key)
            self._items[key] = value
            self._metric("near_catalog_payload_cache_hits")
            return value

    def put(self, key: str, value: NearCatalogStars) -> NearCatalogStars:
        stored = _immutable_stars(value)
        with self._lock:
            self._store_locked(key, stored)
        return stored

    def get_or_load(self, key: str, loader: Callable[[], NearCatalogStars]) -> NearCatalogStars:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.pop(key)
                self._items[key] = value
                self._metric("near_catalog_payload_cache_hits")
                return value
            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = _InflightLoad()
                self._inflight[key] = inflight
                owner = True
                self._metric("near_catalog_payload_cache_misses")
            else:
                owner = False
                self._metric("near_catalog_payload_duplicate_loads")
                self._metric("near_catalog_payload_singleflight_waiters")

        if owner:
            try:
                self._metric("near_catalog_payload_physical_loads")
                stored = _immutable_stars(loader())
            except BaseException as exc:  # pragma: no cover - propagated to all waiters
                with self._lock:
                    inflight.error = exc
                    self._inflight.pop(key, None)
                    inflight.event.set()
                raise
            with self._lock:
                inflight.value = stored
                self._store_locked(key, stored)
                self._inflight.pop(key, None)
                inflight.event.set()
            return stored

        inflight.event.wait()
        if inflight.error is not None:
            raise inflight.error
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.pop(key)
                self._items[key] = value
                self._metric("near_catalog_payload_cache_hits")
                return value
        if inflight.value is None:  # pragma: no cover - defensive
            raise NearCatalogProviderError(f"cache load failed without an exception for {key}")
        return inflight.value

    def _store_locked(self, key: str, value: NearCatalogStars) -> None:
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
            self._metric("near_catalog_payload_cache_evictions")

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._inflight.clear()

    def telemetry(self) -> dict[str, object]:
        with self._lock:
            return {
                "near_catalog_payload_cache_id": id(self),
                "near_catalog_payload_cache_size": len(self._items),
                "near_catalog_payload_cache_capacity": self.max_entries,
                "near_catalog_payload_cache_inflight": len(self._inflight),
            }


@dataclass(slots=True)
class _InflightLoad:
    event: threading.Event = field(default_factory=threading.Event)
    value: NearCatalogStars | None = None
    error: BaseException | None = None


def _readonly_array(values: np.ndarray, dtype: object) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.flags.writeable = False
    return array


def _immutable_stars(value: NearCatalogStars) -> NearCatalogStars:
    return NearCatalogStars(
        _readonly_array(value.ra_deg, np.float64),
        _readonly_array(value.dec_deg, np.float64),
        _readonly_array(value.mag, np.float32),
    )


class LegacyIndexNearCatalogProvider:
    kind = "legacy_index"

    def __init__(
        self,
        index_root: Path | str,
        *,
        cache_size: int = 128,
        metrics_callback: MetricCallback | None = None,
    ) -> None:
        self.index_root = Path(index_root).expanduser().resolve()
        self._metrics_callback = metrics_callback
        if self._metrics_callback is not None:
            self._metrics_callback("near_catalog_provider_created", 1)
        self.manifest = load_manifest(self.index_root)
        self.db_root = Path(str(self.manifest.get("db_root", "")).strip()).expanduser().resolve() if self.manifest.get("db_root") else None
        self._tiles = tuple(self._tile_from_entry(entry) for entry in (self.manifest.get("tiles") or ()))
        if not self._tiles:
            raise NearCatalogProviderError("legacy index manifest has no tiles")
        if self._metrics_callback is not None:
            self._metrics_callback("near_catalog_inventory_load_count", 1)
        self._cache = _MemoryStarCache(cache_size, metrics_callback=self._metrics_callback)
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
        return self._cache.get_or_load(cache_key, lambda: self._load_stars_uncached(tile))

    def _load_stars_uncached(self, tile: NearCatalogTile) -> NearCatalogStars:
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
        return stars

    def _load_astap_fallback(self, tile: NearCatalogTile) -> NearCatalogStars:
        assert self.db_root is not None
        for meta in iter_astap_tiles(self.db_root, metrics_callback=self._metrics_callback):
            if str(meta.family).strip().lower() != tile.family:
                continue
            if str(meta.tile_code) != str(tile.tile_code):
                continue
            raw = load_astap_tile_stars(self.db_root, meta, metrics_callback=self._metrics_callback)
            return NearCatalogStars(
                np.asarray(raw["ra_deg"], dtype=np.float64),
                np.asarray(raw["dec_deg"], dtype=np.float64),
                np.asarray(raw["mag"], dtype=np.float32),
            )
        raise FileNotFoundError(f"ASTAP fallback tile not found: {tile.family}_{tile.tile_code}")

    def telemetry(self) -> dict[str, object]:
        data = {
            "near_catalog_provider": self.kind,
            "near_catalog_family": list(self.families),
            "near_catalog_fallback_used": bool(self._fallback_used),
            "near_catalog_fallback_reason": self._fallback_reason,
            "near_catalog_provider_id": id(self),
            "near_catalog_inventory_id": id(self._tiles),
            "near_catalog_tile_count": len(self._tiles),
        }
        data.update(self._cache.telemetry())
        return data

    def close(self) -> None:
        self._cache.clear()


class AstapNearCatalogProvider:
    kind = "astap_native"

    def __init__(
        self,
        db_root: Path | str,
        *,
        families: Sequence[str] | None = None,
        cache_size: int = 128,
        metrics_callback: MetricCallback | None = None,
    ) -> None:
        self.db_root = Path(db_root).expanduser().resolve()
        self._metrics_callback = metrics_callback
        if self._metrics_callback is not None:
            self._metrics_callback("near_catalog_provider_created", 1)
        requested = set(_normalize_families(families))
        all_tiles: list[NearCatalogTile] = []
        self._catalog_tiles_by_key: dict[str, CatalogTile] = {}
        try:
            self._catalog_db = CatalogDB(self.db_root, families=tuple(sorted(requested)) or None, cache_size=1)
            if self._metrics_callback is not None:
                self._metrics_callback("near_catalog_db_created", 1)
            for catalog_tile in self._catalog_db.tiles:
                meta = TileMeta(
                    key=catalog_tile.key,
                    family=catalog_tile.spec.key,
                    tile_code=catalog_tile.tile_code,
                    path=catalog_tile.path,
                    center_ra_deg=_ra_center_from_segments(catalog_tile.bounds),
                    center_dec_deg=catalog_tile.bounds.dec_center,
                    bounds=catalog_tile.bounds,
                    ring_index=catalog_tile.ring_index,
                    tile_index=catalog_tile.tile_index,
                )
                family = str(meta.family).strip().lower()
                if requested and family not in requested:
                    continue
                self._catalog_tiles_by_key[str(meta.key)] = catalog_tile
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
        except Exception as exc:
            detail = ",".join(sorted(requested)) if requested else "all"
            raise NearCatalogProviderError(f"ASTAP catalog has no usable Near tiles for families={detail}: {exc}") from exc
        if not all_tiles:
            detail = ",".join(sorted(requested)) if requested else "all"
            raise NearCatalogProviderError(f"ASTAP catalog has no usable Near tiles for families={detail}")
        available = {tile.family for tile in all_tiles}
        missing = requested - available
        if missing:
            raise NearCatalogProviderError(f"ASTAP catalog missing requested family/families: {', '.join(sorted(missing))}")
        self._tiles = tuple(all_tiles)
        if self._metrics_callback is not None:
            self._metrics_callback("near_catalog_inventory_load_count", 1)
        self._cache = _MemoryStarCache(cache_size, metrics_callback=self._metrics_callback)

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
        return self._cache.get_or_load(cache_key, lambda: self._load_stars_uncached(tile))

    def _load_stars_uncached(self, tile: NearCatalogTile) -> NearCatalogStars:
        assert isinstance(tile.source, TileMeta)
        try:
            catalog_tile = self._catalog_tiles_by_key[str(tile.tile_key)]
            raw = self._catalog_db._load_tile(catalog_tile).stars
        except Exception as exc:
            raise NearCatalogProviderError(f"failed to read ASTAP tile {tile.tile_key}: {exc}") from exc
        return NearCatalogStars(
            np.asarray(raw["ra_deg"], dtype=np.float64),
            np.asarray(raw["dec_deg"], dtype=np.float64),
            np.asarray(raw["mag"], dtype=np.float32),
        )

    def telemetry(self) -> dict[str, object]:
        data = {
            "near_catalog_provider": self.kind,
            "near_catalog_family": list(self.families),
            "near_catalog_fallback_used": False,
            "near_catalog_provider_id": id(self),
            "near_catalog_inventory_id": id(self._tiles),
            "near_catalog_db_id": id(self._catalog_db),
            "near_catalog_tile_count": len(self._tiles),
        }
        data.update(self._cache.telemetry())
        return data

    def close(self) -> None:
        self._cache.clear()
        clear_cache = getattr(self._catalog_db, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
