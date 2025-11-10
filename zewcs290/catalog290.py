"\"\"\"High level access to ASTAP/HNSKY ``.1476`` and ``.290`` catalogues.\"\"\""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import logging
import math

import numpy as np

from .layouts import get_layout

logger = logging.getLogger(__name__)

STAR_DTYPE = np.dtype(
    [
        ("ra_deg", "<f8"),
        ("dec_deg", "<f8"),
        ("mag", "<f4"),
        ("bp_rp", "<f4"),
        ("flags", "<u4"),
    ]
)

RA_SCALE = 360.0 / ((1 << 24) - 1)  # 360 / 16,777,215
DEC_SCALE = 90.0 / ((1 << 23) - 1)  # 90 / 8,388,607


class FormatId:
    HNSKY_1476_5 = 1
    HNSKY_1476_6 = 2
    HNSKY_290_5 = 3


@dataclass(frozen=True)
class CatalogFamilySpec:
    key: str
    title: str
    prefix: str
    extension: str
    record_size: int
    format_name: str
    format_id: int
    magnitude_band: str
    has_color: bool
    family_id: int

    def glob_pattern(self) -> str:
        return f"{self.prefix}_*.{self.extension}"


def _build_family_specs() -> Dict[str, CatalogFamilySpec]:
    definitions = [
        ("d05", "D05 Gaia DR3 (≤500 stars/deg²)", "1476", 5, FormatId.HNSKY_1476_5, "Gaia BP", False),
        ("d20", "D20 Gaia DR3 (≤2000 stars/deg²)", "1476", 5, FormatId.HNSKY_1476_5, "Gaia BP", False),
        ("d50", "D50 Gaia DR3 (≤5000 stars/deg²)", "1476", 5, FormatId.HNSKY_1476_5, "Gaia BP", False),
        ("d80", "D80 Gaia DR3 (≤8000 stars/deg²)", "1476", 5, FormatId.HNSKY_1476_5, "Gaia BP", False),
        ("v50", "V50 Gaia Johnson-V + color", "1476", 6, FormatId.HNSKY_1476_6, "Johnson-V", True),
        ("g05", "G05 Gaia DR3 wide-field", "290", 5, FormatId.HNSKY_290_5, "Gaia BP", False),
    ]
    specs: Dict[str, CatalogFamilySpec] = {}
    for family_id, (key, title, layout_tag, record_size, fmt_id, mag_band, has_color) in enumerate(definitions, start=1):
        extension = layout_tag
        specs[key] = CatalogFamilySpec(
            key=key,
            title=title,
            prefix=key,
            extension=extension,
            record_size=record_size,
            format_name="1476-6" if fmt_id == FormatId.HNSKY_1476_6 else "1476-5" if layout_tag == "1476" else "290-5",
            format_id=fmt_id,
            magnitude_band=mag_band,
            has_color=has_color,
            family_id=family_id,
        )
    return specs


FAMILY_SPECS: Dict[str, CatalogFamilySpec] = _build_family_specs()


@dataclass(frozen=True)
class SkyBox:
    ra_segments: Tuple[Tuple[float, float], ...]
    dec_min: float
    dec_max: float

    @property
    def covers_full_ra(self) -> bool:
        return len(self.ra_segments) == 1 and math.isclose(self.ra_segments[0][0], 0.0) and math.isclose(
            self.ra_segments[0][1], 360.0
        )

    @property
    def dec_center(self) -> float:
        return 0.5 * (self.dec_min + self.dec_max)


@dataclass(frozen=True)
class CatalogTile:
    spec: CatalogFamilySpec
    tile_code: str  # e.g. "0501"
    path: Path
    ring_index: int
    tile_index: int
    bounds: SkyBox

    @property
    def key(self) -> str:
        return f"{self.spec.key}_{self.tile_code}"


@dataclass(frozen=True)
class RingGeometry:
    dec_min: float
    dec_max: float
    ra_cells: int


@dataclass
class StarBlock:
    stars: np.ndarray
    header_records: int
    record_size: int
    description: str
    payload_bytes: int

    @property
    def star_count(self) -> int:
        return int(self.stars.shape[0])

    @property
    def mag_range(self) -> Tuple[float, float]:
        if not self.star_count:
            return (float("nan"), float("nan"))
        return (float(np.min(self.stars["mag"])), float(np.max(self.stars["mag"])))


class _TileCache:
    def __init__(self, max_entries: int):
        from collections import OrderedDict

        if max_entries < 1:
            raise ValueError("cache size must be >= 1")
        self._store: "OrderedDict[str, StarBlock]" = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: str, loader) -> StarBlock:
        store = self._store
        if key in store:
            value = store.pop(key)
            store[key] = value
            return value
        value = loader()
        store[key] = value
        if len(store) > self._max_entries:
            store.popitem(last=False)
        return value


def _normalize_ra(value):
    if isinstance(value, np.ndarray):
        result = np.mod(value, 360.0)
        result[result < 0] += 360.0
        return result
    result = value % 360.0
    return result + 360.0 if result < 0 else result


def _segments_for_interval(ra_min: float, ra_max: float) -> Tuple[Tuple[float, float], ...]:
    span = ra_max - ra_min
    if abs(span) >= 360.0:
        return ((0.0, 360.0),)
    start = _normalize_ra(ra_min)
    end = _normalize_ra(ra_max)
    if math.isclose(((end - start) % 360.0), 0.0) and not math.isclose(span, 0.0):
        return ((0.0, 360.0),)
    if end >= start:
        return ((start, end),)
    return ((start, 360.0), (0.0, end))


def _segments_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return (a0 <= b1) and (b0 <= a1)


def _point_in_segments(values: np.ndarray, segments: Tuple[Tuple[float, float], ...]) -> np.ndarray:
    vals = _normalize_ra(values)
    if len(segments) == 1 and math.isclose(segments[0][0], 0.0) and math.isclose(segments[0][1], 360.0):
        return np.ones_like(vals, dtype=bool)
    mask = np.zeros_like(vals, dtype=bool)
    for start, end in segments:
        mask |= (vals >= start) & (vals <= end)
    return mask


def _tile_bounds(geom: RingGeometry, tile_index: int) -> SkyBox:
    if tile_index < 1 or tile_index > geom.ra_cells:
        raise ValueError(f"tile index {tile_index} outside [1,{geom.ra_cells}]")
    ra_step = 360.0 / geom.ra_cells
    ra_min = (tile_index - 1) * ra_step
    ra_max = tile_index * ra_step
    segments = _segments_for_interval(ra_min, ra_max)
    return SkyBox(ra_segments=segments, dec_min=geom.dec_min, dec_max=geom.dec_max)


def _angular_separation(ra0_deg: float, dec0_deg: float, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cos_sep = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_sep))


class CatalogDB:
    """Access to a directory full of ASTAP/HNSKY catalogue shards."""

    def __init__(
        self,
        db_root: Path | str,
        families: Optional[Sequence[str]] = None,
        cache_size: int = 8,
    ) -> None:
        self.root = Path(db_root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self._cache = _TileCache(cache_size)
        self._prefetched: Dict[str, Tuple[np.ndarray, int, int, str]] = {}
        if families:
            requested = [fam.lower() for fam in families]
        else:
            requested = list(FAMILY_SPECS.keys())

        self._tiles: List[CatalogTile] = []
        for key in requested:
            spec = FAMILY_SPECS.get(key)
            if spec is None:
                raise KeyError(f"unknown family {key!r}")
            pattern = spec.glob_pattern()
            ring_files: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
            for path in sorted(self.root.glob(pattern)):
                tile_code = path.stem.split("_", 1)[1]
                ring_index = int(tile_code[:2])
                tile_index = int(tile_code[2:])
                ring_files[ring_index].append((tile_index, path))

            if not ring_files:
                logger.info("family %s: no tiles matched %s under %s", key, pattern, self.root)
                continue

            layout_name = "hnsky_1476" if spec.extension == "1476" else "hnsky_290"
            layout = get_layout(layout_name)

            before_count = len(self._tiles)
            for ring_index, entries in ring_files.items():
                entries.sort()
                try:
                    ring_def = layout.ring_for_index(ring_index)
                except KeyError:
                    logger.warning(
                        "family %s: ring %02d missing from layout %s (skipping %d tile(s))",
                        key,
                        ring_index,
                        layout_name,
                        len(entries),
                    )
                    continue
                geom = RingGeometry(
                    dec_min=ring_def.dec_min_deg,
                    dec_max=ring_def.dec_max_deg,
                    ra_cells=ring_def.ra_cells,
                )
                sample_path = entries[0][1]
                sample_key = str(sample_path)
                if sample_key not in self._prefetched:
                    try:
                        self._prefetched[sample_key] = _parse_catalog_file(sample_path, spec)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning("prefetch failed for %s: %s", sample_path, exc)
                for tile_index, path in entries:
                    try:
                        bounds = _tile_bounds(geom, tile_index)
                    except ValueError as exc:
                        logger.warning("skip %s: %s", path, exc)
                        continue
                    tile = CatalogTile(
                        spec=spec,
                        tile_code=path.stem.split("_", 1)[1],
                        path=path,
                        ring_index=ring_index,
                        tile_index=tile_index,
                        bounds=bounds,
                    )
                    self._tiles.append(tile)

            added = len(self._tiles) - before_count
            ring_count = len({tile.ring_index for tile in self._tiles[before_count:]})
            logger.info(
                "family %s: loaded %d tile(s) across %d ring(s) using layout %s",
                key,
                added,
                ring_count,
                layout_name,
            )

        if not self._tiles:
            raise RuntimeError(f"No supported catalogue tiles found in {self.root}")
        family_list = sorted({tile.spec.key for tile in self._tiles})
        total_rings = len({tile.ring_index for tile in self._tiles})
        logger.info(
            "catalog ready: %d tile(s) across %d family(ies) (%s) and %d ring(s) from %s",
            len(self._tiles),
            len(family_list),
            ", ".join(family_list),
            total_rings,
            self.root,
        )

    @property
    def tiles(self) -> Sequence[CatalogTile]:
        return tuple(self._tiles)

    @property
    def families(self) -> Tuple[str, ...]:
        return tuple(sorted({tile.spec.key for tile in self._tiles}))

    def _load_tile(self, tile: CatalogTile) -> StarBlock:
        key = str(tile.path)

        def loader() -> StarBlock:
            parsed = self._prefetched.pop(key, None)
            if parsed is None:
                parsed = _parse_catalog_file(tile.path, tile.spec)
            return _build_star_block(tile, parsed)

        return self._cache.get(key, loader)

    def describe(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for tile in self._tiles:
            summary[tile.spec.key] = summary.get(tile.spec.key, 0) + 1
        return summary

    def _iter_candidate_tiles(
        self,
        ra_segments: Tuple[Tuple[float, float], ...],
        dec_min: float,
        dec_max: float,
        families: Optional[Sequence[str]] = None,
    ) -> Iterator[CatalogTile]:
        fam_filter = {fam.lower() for fam in families} if families else None
        for tile in self._tiles:
            if fam_filter and tile.spec.key not in fam_filter:
                continue
            if tile.bounds.dec_max < dec_min or tile.bounds.dec_min > dec_max:
                continue
            if tile.bounds.covers_full_ra:
                yield tile
                continue
            for t_start, t_end in tile.bounds.ra_segments:
                if any(_segments_overlap(t_start, t_end, q_start, q_end) for q_start, q_end in ra_segments):
                    yield tile
                    break

    def query_box(
        self,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        *,
        families: Optional[Sequence[str]] = None,
        mag_limit: Optional[float] = None,
        max_stars: Optional[int] = None,
    ) -> np.ndarray:
        dec_min = max(-90.0, min(dec_min_deg, dec_max_deg))
        dec_max = min(90.0, max(dec_min_deg, dec_max_deg))
        ra_segments = _segments_for_interval(ra_min_deg, ra_max_deg)

        chunks: List[np.ndarray] = []
        for tile in self._iter_candidate_tiles(ra_segments, dec_min, dec_max, families):
            block = self._load_tile(tile)
            stars = block.stars
            if stars.size == 0:
                continue
            mask = (
                (stars["dec_deg"] >= dec_min)
                & (stars["dec_deg"] <= dec_max)
                & _point_in_segments(stars["ra_deg"], ra_segments)
            )
            if mag_limit is not None:
                mask &= stars["mag"] <= mag_limit
            subset = stars[mask]
            if subset.size:
                chunks.append(subset)
        if not chunks:
            return np.empty(0, dtype=STAR_DTYPE)
        combined = np.concatenate(chunks, axis=0)
        if max_stars and combined.size > max_stars:
            order = np.argsort(combined["mag"])
            combined = combined[order[:max_stars]]
        return combined

    def query_cone(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        *,
        families: Optional[Sequence[str]] = None,
        mag_limit: Optional[float] = None,
        max_stars: Optional[int] = None,
    ) -> np.ndarray:
        dec_min = max(-90.0, dec_deg - radius_deg)
        dec_max = min(90.0, dec_deg + radius_deg)
        if abs(dec_deg) > 89.0:
            ra_min = 0.0
            ra_max = 360.0
        else:
            span = radius_deg / max(math.cos(math.radians(dec_deg)), 1e-6)
            span = min(span * 1.2, 360.0)
            ra_min = ra_deg - span
            ra_max = ra_deg + span
        candidates = self.query_box(
            ra_min,
            ra_max,
            dec_min,
            dec_max,
            families=families,
            mag_limit=mag_limit,
        )
        if candidates.size == 0:
            return candidates
        sep = _angular_separation(ra_deg, dec_deg, candidates["ra_deg"], candidates["dec_deg"])
        mask = sep <= radius_deg
        subset = candidates[mask]
        if subset.size and max_stars and subset.size > max_stars:
            order = np.argsort(subset["mag"])
            subset = subset[order[:max_stars]]
        return subset


def _parse_catalog_file(path: Path, spec: CatalogFamilySpec) -> Tuple[np.ndarray, int, int, str]:
    header_len = 110
    raw_bytes = path.stat().st_size
    if raw_bytes < header_len:
        raise ValueError(f"{path} is too small to be a catalogue tile")
    with path.open("rb") as handle:
        header = handle.read(header_len)
        payload = np.frombuffer(handle.read(), dtype=np.uint8)

    description = header[:-1].decode("ascii", errors="ignore").rstrip("\x00 \n\r")
    record_size = header[-1]
    if record_size != spec.record_size:
        logger.warning(
            "record size mismatch for %s (expected %s, saw %s)",
            path.name,
            spec.record_size,
            record_size,
        )
    if payload.size == 0 or record_size == 0:
        return (
            np.empty(0, dtype=STAR_DTYPE),
            0,
            record_size,
            description,
        )
    usable = (payload.size // record_size) * record_size
    raw = payload[:usable].reshape(-1, record_size)
    is_header = np.all(raw[:, :3] == 0xFF, axis=1)
    if not is_header.any():
        raise ValueError(f"{path} does not appear to contain header records")
    header_indices = np.cumsum(is_header).astype(int) - 1
    star_rows = raw[~is_header]
    if not star_rows.size:
        return (
            np.empty(0, dtype=STAR_DTYPE),
            int(is_header.sum()),
            record_size,
            description,
        )
    owner = header_indices[~is_header]
    if (owner < 0).any():
        raise ValueError(f"{path} has star records before first header")
    header_rows = raw[is_header]
    dec9_bytes = header_rows[:, 3]
    mag_bytes = header_rows[:, 4].astype(np.int16)
    magnitudes = (mag_bytes - 16) / 10.0
    dec9 = dec9_bytes[owner]
    mags = magnitudes[owner].astype(np.float32)

    ra = star_rows[:, 0].astype(np.uint32) | (star_rows[:, 1].astype(np.uint32) << 8) | (
        star_rows[:, 2].astype(np.uint32) << 16
    )
    ra_deg = ra.astype(np.float64) * RA_SCALE

    dec = (
        star_rows[:, 3].astype(np.int32)
        | (star_rows[:, 4].astype(np.int32) << 8)
        | (dec9.astype(np.int32) << 16)
    )
    negative = dec9 >= 128
    if negative.any():
        dec = dec.astype(np.int64)
        dec[negative] -= 1 << 24
    dec_deg = dec.astype(np.float64) * DEC_SCALE

    bp_rp = np.full(star_rows.shape[0], np.nan, dtype=np.float32)
    if spec.has_color and record_size >= 6:
        color_raw = star_rows[:, 5].astype(np.int16)
        color_raw[color_raw >= 128] -= 256
        bp_rp = color_raw.astype(np.float32) / 10.0

    stars = np.zeros(star_rows.shape[0], dtype=STAR_DTYPE)
    stars["ra_deg"] = ra_deg
    stars["dec_deg"] = dec_deg
    stars["mag"] = mags
    stars["bp_rp"] = bp_rp

    return stars, int(is_header.sum()), record_size, description


def _build_star_block(tile: CatalogTile, parsed) -> StarBlock:
    stars, header_records, record_size, description = parsed
    if stars.size == 0:
        return StarBlock(
            stars=stars,
            header_records=header_records,
            record_size=record_size,
            description=description,
            payload_bytes=0,
        )
    flags = (
        (tile.spec.family_id << 24)
        | (tile.ring_index << 16)
        | (tile.tile_index << 8)
        | (tile.spec.format_id & 0xFF)
    )
    star_flags = np.full(stars.shape[0], flags, dtype=np.uint32)
    stars = stars.copy()
    stars["flags"] = star_flags
    return StarBlock(
        stars=stars,
        header_records=header_records,
        record_size=record_size,
        description=description,
        payload_bytes=int(stars.shape[0] * record_size),
    )


def _decode_tile(tile: CatalogTile) -> StarBlock:
    parsed = _parse_catalog_file(tile.path, tile.spec)
    return _build_star_block(tile, parsed)
