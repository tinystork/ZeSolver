from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
import threading

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .asterisms import hash_quads, sample_quads
from .candidate_search import tally_candidates
from .image_io import load_raster_image
from .image_prep import build_pyramid, downsample_image, read_fits_as_luma, remove_background
from .fits_utils import parse_angle
from .matcher import SimilarityStats, SimilarityTransform, estimate_similarity_RANSAC
from .matcher import _derive_similarity  # quad-based hypothesis helper
from .quad_index_builder import QuadIndex, load_manifest, lookup_hashes, select_tiles_in_cone
from .levels import LEVEL_MAP, QuadLevelSpec, set_bucket_cap_overrides
from .star_detect import detect_stars
from .verify import validate_solution
from .wcs_fit import fit_wcs_sip, fit_wcs_tan, needs_sip, tan_from_similarity

try:
    from importlib.metadata import PackageNotFoundError, version as pkg_version
except ImportError:  # pragma: no cover - older Python
    PackageNotFoundError = Exception  # type: ignore[misc]
    def pkg_version(_: str) -> str:  # type: ignore[override]
        return "0.0.dev"

try:
    __version__ = pkg_version("zewcs290")
except PackageNotFoundError:
    __version__ = "0.0.dev"

ZEBLIND_VERSION = __version__
logger = logging.getLogger(__name__)
FITS_EXTENSIONS = {".fits", ".fit", ".fts"}


def _env_tile_cache_size(default: int = 128) -> int:
    raw = os.environ.get("ZE_TILE_CACHE_SIZE")
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


_TILE_CACHE_DEFAULT_CAPACITY = _env_tile_cache_size()


@dataclass
class SolveConfig:
    max_candidates: int = 10
    max_stars: int = 500
    max_quads: int = 8000
    detect_k_sigma: float = 3.0
    detect_min_area: int = 5
    bucket_cap_S: int = 0
    bucket_cap_M: int = 0
    bucket_cap_L: int = 0
    sip_order: int = 2
    quality_rms: float = 1.2
    quality_inliers: int = 40
    pixel_tolerance: float = 2.5
    log_level: str = "INFO"
    verbose: bool = False
    try_parity_flip: bool = True
    fast_mode: bool = True
    downsample: int = 1
    ra_hint_deg: float | None = None
    dec_hint_deg: float | None = None
    radius_hint_deg: float | None = None
    focal_length_mm: float | None = None
    pixel_size_um: float | None = None
    pixel_scale_arcsec: float | None = None
    pixel_scale_min_arcsec: float | None = None
    pixel_scale_max_arcsec: float | None = None
    tile_cache_size: int = _TILE_CACHE_DEFAULT_CAPACITY
    bucket_limit_override: int | None = None
    vote_percentile: int = 40


@dataclass
class WcsSolution:
    success: bool
    message: str
    wcs: WCS | None
    stats: dict[str, Any]
    tile_key: str | None
    header_updates: dict[str, Any]


def _image_positions(stars: np.ndarray) -> np.ndarray:
    return np.column_stack((stars["x"], stars["y"]))


@dataclass(frozen=True)
class _TileCacheEntry:
    signature: tuple[int, int]
    xy: np.ndarray
    world: np.ndarray


class _TileCache:
    def __init__(self, capacity: int) -> None:
        self._capacity = max(0, int(capacity))
        self._entries: OrderedDict[Path, _TileCacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _trim(self) -> None:
        while self._capacity >= 0 and len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def _stat_signature(self, path: Path) -> tuple[int, int]:
        stat = path.stat()
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
        return int(mtime_ns), int(stat.st_size)

    def configure(self, capacity: int) -> None:
        with self._lock:
            self._capacity = max(0, int(capacity))
            self._trim()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> tuple[int, int]:
        with self._lock:
            return self._hits, self._misses

    def capacity(self) -> int:
        with self._lock:
            return self._capacity

    def load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        real_path = path.resolve()
        try:
            signature = self._stat_signature(real_path)
        except FileNotFoundError:
            with self._lock:
                self._entries.pop(real_path, None)
            raise
        with self._lock:
            entry = self._entries.get(real_path)
            if entry and entry.signature == signature:
                self._entries.move_to_end(real_path)
                self._hits += 1
                return entry.xy, entry.world
            if entry:
                self._entries.pop(real_path, None)
            self._misses += 1
        xy, world = _read_tile_payload(real_path)
        xy.setflags(write=False)
        world.setflags(write=False)
        if self._capacity <= 0:
            return xy, world
        with self._lock:
            existing = self._entries.get(real_path)
            if existing and existing.signature == signature:
                self._entries.move_to_end(real_path)
                return existing.xy, existing.world
            self._entries[real_path] = _TileCacheEntry(signature, xy, world)
            self._entries.move_to_end(real_path)
            self._trim()
        return xy, world


def _read_tile_payload(tile_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(tile_path) as data:
        xy = np.column_stack((data["x_deg"], data["y_deg"]))
        world = np.column_stack((data["ra_deg"], data["dec_deg"]))
    return xy.astype(np.float32), world.astype(np.float32)


_TILE_CACHE = _TileCache(_TILE_CACHE_DEFAULT_CAPACITY)


def _configure_tile_cache(capacity: int | None = None) -> None:
    if capacity is None:
        return
    _TILE_CACHE.configure(capacity)


def _tile_cache_clear() -> None:
    _TILE_CACHE.clear()


def _tile_cache_stats() -> tuple[int, int]:
    return _TILE_CACHE.stats()


def _tile_cache_capacity() -> int:
    return _TILE_CACHE.capacity()


def _load_tile_positions(index_root: Path, entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # Normalize path separators to support manifests written on Windows
    rel = str(entry["tile_file"]).replace("\\", "/")
    tile_path = (index_root / rel).expanduser()
    return _TILE_CACHE.load(tile_path)


def _deduplicate_hashes(
    hashes: np.ndarray,
    quads: np.ndarray,
    *,
    label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hashes.size == 0:
        return hashes, quads[:0], np.zeros(0, dtype=np.uint32)
    unique_hashes, indices, counts = np.unique(hashes, return_index=True, return_counts=True)
    unique_quads = quads[indices] if indices.size else quads[:0]
    if logger.isEnabledFor(logging.DEBUG):
        ratio = float(hashes.size) / float(unique_hashes.size or 1)
        suffix = f" ({label})" if label else ""
        logger.debug(
            "observed hash dedup%s: %d -> %d unique (%.2fx)",
            suffix,
            hashes.size,
            unique_hashes.size,
            ratio,
        )
    return unique_hashes, unique_quads, counts.astype(np.uint32, copy=False)


def _angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1 = math.radians(float(ra1))
    r2 = math.radians(float(ra2))
    d1 = math.radians(float(dec1))
    d2 = math.radians(float(dec2))
    cos_sep = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def _resolve_scale_arcsec(config: SolveConfig, header_scale_arcsec: float | None) -> float | None:
    candidates: list[float] = []
    if config.pixel_scale_arcsec:
        candidates.append(float(config.pixel_scale_arcsec))
    if config.focal_length_mm and config.pixel_size_um:
        try:
            focal = float(config.focal_length_mm)
            pixel = float(config.pixel_size_um)
            if focal > 0.0 and pixel > 0.0:
                candidates.append(206.265 * pixel / focal)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    range_hint: float | None = None
    if config.pixel_scale_min_arcsec and config.pixel_scale_max_arcsec:
        try:
            lo = float(config.pixel_scale_min_arcsec)
            hi = float(config.pixel_scale_max_arcsec)
            if lo > 0.0 and hi > 0.0:
                range_hint = 0.5 * (lo + hi)
        except (TypeError, ValueError):
            range_hint = None
    elif config.pixel_scale_min_arcsec:
        try:
            lo = float(config.pixel_scale_min_arcsec)
            if lo > 0.0:
                range_hint = lo
        except (TypeError, ValueError):
            range_hint = None
    elif config.pixel_scale_max_arcsec:
        try:
            hi = float(config.pixel_scale_max_arcsec)
            if hi > 0.0:
                range_hint = hi
        except (TypeError, ValueError):
            range_hint = None
    if range_hint:
        candidates.append(range_hint)
    if header_scale_arcsec:
        try:
            hdr_scale = float(header_scale_arcsec)
            if hdr_scale > 0.0:
                candidates.append(hdr_scale)
        except (TypeError, ValueError):
            pass
    for value in candidates:
        if value and value > 0.0 and math.isfinite(value):
            return float(value)
    return None


def _radius_from_hints(
    config: SolveConfig,
    approx_scale_deg: float | None,
    width: int,
    height: int,
) -> float | None:
    if config.radius_hint_deg and config.radius_hint_deg > 0.0:
        try:
            return max(0.05, float(config.radius_hint_deg))
        except (TypeError, ValueError):
            pass
    if approx_scale_deg and approx_scale_deg > 0.0:
        diag_px = math.hypot(float(width), float(height))
        radius = 0.5 * diag_px * approx_scale_deg * 1.1
        return max(0.05, min(radius, 20.0))
    return None


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def _write_wcs_sidecar(output: Path, wcs: WCS, header_updates: dict[str, Any]) -> None:
    header = wcs.to_header(relax=True)
    for key, value in header_updates.items():
        header[key] = value
    payload = {key: _serialize_value(value) for key, value in header.items()}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _nearest_tile_key(tile_entries: Iterable[dict[str, Any]], ra_deg: float, dec_deg: float) -> str | None:
    best_key: str | None = None
    best_dist = float("inf")
    for entry in tile_entries:
        try:
            tra = float(entry.get("center_ra_deg", 0.0))
            tdec = float(entry.get("center_dec_deg", 0.0))
        except (TypeError, ValueError):
            continue
        dist = _angular_separation(ra_deg, dec_deg, tra, tdec)
        if dist < best_dist:
            best_dist = dist
            best_key = str(entry.get("tile_key"))
    return best_key


def _collect_tile_matches(
    index_root: Path,
    levels: Iterable[str],
    tile_index: int,
    observed_hashes: np.ndarray,
    observed_quads: np.ndarray,
    image_positions: np.ndarray,
    tile_positions: np.ndarray,
    tile_world: np.ndarray,
    bucket_limit: int,
    vote_percentile: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Accumulate votes for (image_star, tile_star) pairs across matching quads,
    # then keep only pairs with sufficient support to reduce spurious matches.
    from collections import Counter
    votes: Counter[tuple[int, int]] = Counter()
    for level in levels:
        try:
            index = QuadIndex.load(index_root, level)
        except FileNotFoundError:
            continue
        slices = lookup_hashes(index_root, level, observed_hashes)
        for idx, slc in enumerate(slices):
            if slc.start == slc.stop:
                continue
            obs_combo = observed_quads[idx]
            # Do not skip large buckets outright; cap per-bucket iterations to bound work
            per_bucket_cap = bucket_limit
            for off, bucket in enumerate(range(slc.start, slc.stop)):
                if off >= per_bucket_cap:
                    break
                if int(index.tile_indices[bucket]) != tile_index:
                    continue
                tile_combo = index.quad_indices[bucket]
                for obs_star, tile_star in zip(obs_combo, tile_combo):
                    votes[(int(obs_star), int(tile_star))] += 1
    if not votes:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, empty
    # Determine a minimal vote threshold adaptively; allow 1 to seed hypotheses.
    counts = np.array(list(votes.values()), dtype=int)
    # Use a moderate percentile to keep plausible pairs; too high discards signal.
    percentile = min(95, max(5, vote_percentile))
    thr = max(1, int(np.percentile(counts, percentile)))
    pairs = [(i, t, c) for (i, t), c in votes.items() if c >= thr]
    if not pairs:
        # fallback: keep top-N by votes
        top = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[: max(200, int(len(votes) * 0.1))]
        pairs = [(i, t, c) for (i, t), c in top]
    pairs.sort(key=lambda itc: itc[2], reverse=True)
    # Cap pairs to bound runtime but keep enough support for RANSAC
    cap = min(len(pairs), max(800, int(image_positions.shape[0] * 12)))
    chosen = pairs[:cap]
    img_pts = np.array([image_positions[i] for i, _, _ in chosen], dtype=np.float32)
    tile_pts = np.array([tile_positions[t] for _, t, _ in chosen], dtype=np.float32)
    world_pts = np.array([tile_world[t] for _, t, _ in chosen], dtype=np.float32)
    return img_pts, tile_pts, world_pts


def _build_matches_array(image_pts: np.ndarray, sky_pts: np.ndarray) -> np.ndarray:
    if image_pts.size == 0 or sky_pts.size == 0:
        return np.empty((0, 4), dtype=float)
    return np.column_stack((image_pts, sky_pts))


def _log_phase(stage: str, start: float) -> None:
    logger.debug("%s completed in %.2fs", stage, time.time() - start)


def solve_blind(
    input_fits: Path | str,
    index_root: Path | str,
    *,
    config: SolveConfig | None = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> WcsSolution:
    config = config or SolveConfig()
    _configure_tile_cache(getattr(config, "tile_cache_size", _TILE_CACHE_DEFAULT_CAPACITY))
    cap_overrides: dict[str, int] = {}
    for level_name, attr in (("S", "bucket_cap_S"), ("M", "bucket_cap_M"), ("L", "bucket_cap_L")):
        value = int(getattr(config, attr, 0) or 0)
        if value > 0:
            cap_overrides[level_name] = value
    set_bucket_cap_overrides(cap_overrides or None)

    def _finish(result: WcsSolution) -> WcsSolution:
        if logger.isEnabledFor(logging.DEBUG):
            hits, misses = _tile_cache_stats()
            total = hits + misses
            capacity = _tile_cache_capacity()
            if total:
                hit_rate = 100.0 * hits / total
                logger.debug(
                    "tile cache stats: %d hits / %d misses (%.1f%% hit rate, capacity=%d)",
                    hits,
                    misses,
                    hit_rate,
                    capacity,
                )
            else:
                logger.debug("tile cache stats: 0 hits / 0 misses (capacity=%d)", capacity)
        return result
    index_root = Path(index_root).expanduser().resolve()
    downsample_factor = max(1, min(4, int(getattr(config, "downsample", 1) or 1)))
    bucket_override = int(getattr(config, "bucket_limit_override", 0) or 0)
    if bucket_override > 0:
        bucket_limit = max(256, bucket_override)
    else:
        bucket_limit = max(1024, int(4096 / downsample_factor))
    vote_percentile = int(getattr(config, "vote_percentile", 40) or 40)
    vote_percentile = min(95, max(5, vote_percentile))
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
    manifest = load_manifest(index_root)
    levels = [level["name"] for level in manifest.get("levels", [])] or ["L", "M", "S"]
    # Prefer trying smaller-diameter levels first for selectivity
    pref = ("S", "M", "L")
    levels = [lvl for lvl in pref if lvl in levels] + [lvl for lvl in levels if lvl not in pref]
    # Preflight: ensure there is at least one quad hash table present.
    ht_root = index_root / "hash_tables"
    def _has_quad_table(level_name: str) -> bool:
        return (ht_root / f"quads_{level_name}.npz").exists() or (ht_root / f"quads_{level_name}").is_dir()

    present_levels = [lvl for lvl in levels if _has_quad_table(lvl)]
    missing_levels = [lvl for lvl in levels if lvl not in present_levels]
    if not present_levels:
        manifest_path = index_root / "manifest.json"
        details = [
            f"root={index_root}",
            f"manifest={'ok' if manifest_path.exists() else 'missing'}",
        ]
        for lvl in levels:
            details.append(f"{lvl}={'ok' if _has_quad_table(lvl) else 'missing'}")
        msg = (
            "index has no quad hash tables (levels: "
            + ", ".join(levels)
            + ") — details: "
            + ", ".join(details)
            + ". Build the index with zebuildindex (see firstrun.txt)"
        )
        logger.error(msg)
        return _finish(WcsSolution(False, msg, None, {}, None, {}))
    if missing_levels:
        logger.warning(
            "some quad hash tables are missing: %s (continuing with %s)",
            ", ".join(missing_levels),
            ", ".join(present_levels),
        )
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
    tile_entries = manifest.get("tiles", [])
    tile_count = len(tile_entries)
    tile_map = {entry["tile_key"]: idx for idx, entry in enumerate(tile_entries)}
    logging.getLogger().setLevel(config.log_level)
    source_path = Path(input_fits)
    suffix = source_path.suffix.lower()
    is_fits = suffix in FITS_EXTENSIONS
    header = None
    image_meta: dict[str, Any] = {}
    if is_fits:
        try:
            header = fits.getheader(source_path)
        except Exception:
            header = None
        image = read_fits_as_luma(source_path)
    else:
        image, image_meta = load_raster_image(source_path)
        logger.info(
            "loaded %s via %s (shape=%s)",
            suffix or "<unknown>",
            image_meta.get("backend", "unknown"),
            image.shape,
        )
    ra_hint = config.ra_hint_deg
    dec_hint = config.dec_hint_deg
    preferred_tile_hint: str | None = None
    if header is not None:
        ra0 = parse_angle(header.get("RA") or header.get("OBJCTRA") or header.get("OBJRA") or header.get("OBJ_RA") or header.get("CRVAL1"), is_ra=True)
        dec0 = parse_angle(header.get("DEC") or header.get("OBJCTDEC") or header.get("OBJDEC") or header.get("OBJ_DEC") or header.get("CRVAL2"), is_ra=False)
        if ra_hint is None and ra0 is not None:
            ra_hint = float(ra0)
        if dec_hint is None and dec0 is not None:
            dec_hint = float(dec0)
    if tile_entries and ra_hint is not None and dec_hint is not None:
        preferred_tile_hint = _nearest_tile_key(tile_entries, ra_hint, dec_hint)
    stage = time.time()
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
    allowed_tile_indices: set[int] | None = None
    height, width = image.shape
    work_image = image
    if downsample_factor > 1:
        logger.info("downsampling input by factor %d", downsample_factor)
        work_image = downsample_image(work_image, downsample_factor)
    # Estimate pixel scale (deg/px) if header contains optical info
    header_scale_arcsec = None
    if header is None and is_fits:
        try:
            with fits.open(source_path, mode="readonly", memmap=False) as _hdul_tmp:
                header = _hdul_tmp[0].header
        except Exception:
            header = None
    if header is not None:
        from .fits_utils import estimate_scale_and_fov as _est_scale
        scale_arcsec, _ = _est_scale(header, width, height)
        if scale_arcsec:
            header_scale_arcsec = float(scale_arcsec)
    approx_scale_arcsec = _resolve_scale_arcsec(config, header_scale_arcsec)
    approx_scale_deg = approx_scale_arcsec / 3600.0 if approx_scale_arcsec else None
    approx_fov_deg = None
    if approx_scale_deg is not None:
        approx_fov_deg = float(approx_scale_deg) * float(max(width, height))
    radius_for_filter = _radius_from_hints(config, approx_scale_deg, width, height)
    if ra_hint is not None and dec_hint is not None and radius_for_filter is not None:
        selected = select_tiles_in_cone(manifest, ra_hint, dec_hint, radius_for_filter)
        if selected:
            if len(selected) < tile_count:
                allowed_tile_indices = set(selected)
                logger.info(
                    "RA/Dec hint constrained candidate tiles to %d/%d (radius=%.2f°)",
                    len(selected),
                    tile_count,
                    radius_for_filter,
                )
            else:
                logger.debug(
                    "RA/Dec hint covers all %d tiles (radius=%.2f°) — no cone pruning applied",
                    tile_count,
                    radius_for_filter,
                )
        else:
            logger.warning(
                "RA/Dec hint radius %.2f° excluded all manifest tiles; ignoring cone filter",
                radius_for_filter,
            )
    _log_phase("read image", stage)
    stage = time.time()
    # Use a smaller median kernel for speed on typical Seestar frames
    kernel = max(5, int(round(15 / downsample_factor)))
    work_image = remove_background(work_image, kernel_size=kernel)
    pyramid = build_pyramid(work_image)
    _log_phase("preprocess", stage)
    detection = pyramid[-1]
    stage = time.time()
    min_fwhm = max(1.0, 1.5 / downsample_factor)
    max_fwhm = max(2.5, 8.0 / downsample_factor)
    detect_k_sigma = max(0.5, float(getattr(config, "detect_k_sigma", 3.0)))
    detect_min_area = max(1, int(getattr(config, "detect_min_area", 5)))
    stars = detect_stars(
        detection,
        min_fwhm_px=min_fwhm,
        max_fwhm_px=max_fwhm,
        k_sigma=detect_k_sigma,
        min_area=detect_min_area,
    )
    if stars.size == 0:
        return _finish(WcsSolution(False, "no stars found", None, {}, None, {}))
    if config.max_stars and stars.size > config.max_stars:
        stars = stars[: config.max_stars]
    if downsample_factor > 1:
        stars["x"] *= downsample_factor
        stars["y"] *= downsample_factor
    logger.info(
        "detected %d stars (using top %d, downsample=%d)",
        stars.shape[0],
        config.max_stars or stars.shape[0],
        downsample_factor,
    )
    image_positions = _image_positions(stars)
    obs_stars = np.zeros(stars.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    obs_stars["x"] = stars["x"]
    obs_stars["y"] = stars["y"]
    obs_stars["mag"] = -stars["flux"]
    # Prefer local-neighborhood quads to improve geometric stability on TAN plane
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
    quads = sample_quads(obs_stars, config.max_quads, strategy="local_brightness")
    if quads.size == 0:
        return _finish(WcsSolution(False, "no quads sampled", None, {}, None, {}))
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
    obs_hash = hash_quads(quads, image_positions)
    base_hashes, base_quads, base_counts = _deduplicate_hashes(obs_hash.hashes, obs_hash.indices, label="base")
    logger.info(
        "sampled %d quads producing %d hashes (%d unique)",
        quads.shape[0],
        obs_hash.hashes.size,
        base_hashes.size,
    )
    _log_phase("detect/quads", stage)
    thresholds = {"rms_px": config.quality_rms, "inliers": config.quality_inliers}

    def _prioritize_candidates(
        candidates: list[tuple[str, int]],
        preferred: str | None,
    ) -> list[tuple[str, int]]:
        if not preferred:
            return list(candidates)
        prioritized: list[tuple[str, int]] = []
        for key, value in candidates:
            if key == preferred:
                prioritized.append((key, value))
                break
        prioritized.extend((key, value) for key, value in candidates if key not in {preferred})
        return prioritized

    def _spec_pixels(level_name: str) -> QuadLevelSpec | None:
        if approx_scale_deg is None:
            return None
        spec = LEVEL_MAP.get(level_name)
        if spec is None:
            return None
        s = max(approx_scale_deg, 1e-8)
        # Convert deg thresholds to pixel thresholds; be permissive on the lower bound
        # to avoid discarding valid image quads. Keep only upper bounds as guidance.
        min_area_px = 0.0
        max_area_px = spec.max_area / (s * s)
        min_diam_px = None
        max_diam_px = None if spec.max_diameter is None else spec.max_diameter / s
        return QuadLevelSpec(
            name=spec.name,
            min_area=float(min_area_px),
            max_area=float(max_area_px),
            min_diameter=min_diam_px,
            max_diameter=max_diam_px,
            bucket_cap=spec.bucket_cap,
        )

    # Precompute observed hashes per level with pixel-adapted specs when possible
    obs_by_level: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lvl in levels:
        if cancel_check and cancel_check():
            return WcsSolution(False, "cancelled", None, {}, None, {})
        px = _spec_pixels(lvl)
        if px is None:
            obs_by_level[lvl] = (base_hashes, base_quads, base_counts)
        else:
            oh = hash_quads(quads, image_positions, spec=px)
            obs_by_level[lvl] = _deduplicate_hashes(oh.hashes, oh.indices, label=lvl)

    def _attempt_level(
        level_name: str,
        hashes: np.ndarray,
        preferred_tile: str | None,
        parity_label: str,
        *,
        use_px_spec: bool = True,
        use_ra_filter: bool = True,
        agg_levels: Iterable[str] | None = None,
        early_exit_ratio: float | None = None,
    ) -> WcsSolution | None:
        if cancel_check and cancel_check():
            return None
        # Use precomputed observed quads for this level (optionally bypassing px filters)
        if use_px_spec:
            entry = obs_by_level.get(level_name)
            if entry is None:
                entry = (base_hashes, base_quads, base_counts)
        else:
            entry = (base_hashes, base_quads, base_counts)
        level_hashes, level_quads, level_counts = entry
        if cancel_check and cancel_check():
            return None
        candidates = tally_candidates(
            (level_hashes, level_counts),
            index_root,
            levels=[level_name],
            allowed_tiles=allowed_tile_indices,
        )
        if not candidates:
            logger.debug("level %s produced no candidates (parity=%s)", level_name, parity_label)
            return None
        ordered = _prioritize_candidates(candidates, preferred_tile)
        if early_exit_ratio and len(ordered) >= 2:
            top_score = max(1, ordered[0][1])
            runner_up = max(1, ordered[1][1])
            if runner_up == 0 or top_score >= early_exit_ratio * runner_up:
                logger.debug(
                    "early-exit ratio %.1fx satisfied at level %s (parity=%s)",
                    early_exit_ratio,
                    level_name,
                    parity_label,
                )
                ordered = ordered[:1]
        # If we have RA/DEC hint, ignore far tiles to speed up and reduce false tries
        if use_ra_filter and allowed_tile_indices is None and ra_hint is not None and dec_hint is not None:
            # derive a radius from explicit hint when available; otherwise approximate from FOV
            radius_limit = radius_for_filter
            if radius_limit is None:
                fallback_fov = approx_fov_deg or 1.5
                radius_limit = min(8.0, max(5.0, fallback_fov * 3.0))
            filtered: list[tuple[str, int]] = []
            for key, score in ordered:
                if cancel_check and cancel_check():
                    return None
                idx = tile_map.get(key)
                if idx is None:
                    continue
                entry = tile_entries[idx]
                tra = float(entry.get("center_ra_deg", ra_hint))
                tdec = float(entry.get("center_dec_deg", dec_hint))
                if _angular_separation(ra_hint, dec_hint, tra, tdec) <= radius_limit:
                    filtered.append((key, score))
            if filtered:
                ordered = filtered
        logger.info(
            "level %s (parity=%s) candidate search returned %d candidate(s)",
            level_name,
            parity_label,
            len(candidates),
        )
        for candidate_key, score in ordered[: config.max_candidates]:
            if cancel_check and cancel_check():
                return None
            logger.debug(
                "trying tile %s (score=%d) at level %s (parity=%s)",
                candidate_key,
                score,
                level_name,
                parity_label,
            )
            tile_index = tile_map.get(candidate_key)
            if tile_index is None:
                continue
            if allowed_tile_indices is not None and tile_index not in allowed_tile_indices:
                continue
            tile_entry = tile_entries[tile_index]
            try:
                tile_positions, tile_world = _load_tile_positions(index_root, tile_entry)
            except FileNotFoundError:
                continue
            # Build matches by aggregating across available levels to boost support
            img_list: list[np.ndarray] = []
            tile_list: list[np.ndarray] = []
            world_list: list[np.ndarray] = []
            levels_to_use = list(agg_levels) if agg_levels is not None else list(levels)
            for lvl in levels_to_use:
                if cancel_check and cancel_check():
                    return None
                if use_px_spec:
                    lvl_entry = obs_by_level.get(lvl)
                    if lvl_entry is None:
                        lvl_entry = (base_hashes, base_quads, base_counts)
                else:
                    lvl_entry = (base_hashes, base_quads, base_counts)
                ohashes, oquads, _ = lvl_entry
                ip, tp, wp = _collect_tile_matches(
                    index_root,
                    (lvl,),
                    tile_index,
                    ohashes,
                    oquads,
                    image_positions,
                    tile_positions,
                    tile_world,
                    bucket_limit,
                    vote_percentile,
                )
                if ip.size:
                    img_list.append(ip)
                    tile_list.append(tp)
                    world_list.append(wp)
            if img_list:
                img_points = np.vstack(img_list)
                tile_points = np.vstack(tile_list)
                tile_world_matches = np.vstack(world_list)
            else:
                img_points = np.empty((0, 2), dtype=np.float32)
                tile_points = np.empty((0, 2), dtype=np.float32)
                tile_world_matches = np.empty((0, 2), dtype=np.float32)
            if img_points.shape[0] < 4:
                continue
            # Try quad-based hypotheses first (local, stable scale) then fall back to RANSAC
            transform_result: tuple[SimilarityTransform, SimilarityStats] | None = None
            try:
                index = QuadIndex.load(index_root, level_name)
            except FileNotFoundError:
                index = None
            if index is not None:
                slices = lookup_hashes(index_root, level_name, level_hashes)
                best = None
                best_inliers = -1
                tested = 0
                # Cap buckets per candidate tile to keep runtime reasonable
                max_buckets = 800
                src_all_c = (img_points[:, 0] + 1j * img_points[:, 1]).astype(np.complex128)
                dst_all_c = (tile_points[:, 0] + 1j * tile_points[:, 1]).astype(np.complex128)
                for idx2, slc in enumerate(slices):
                    if cancel_check and cancel_check():
                        return None
                    if slc.start == slc.stop:
                        continue
                    if tested >= max_buckets:
                        break
                    obs_combo = level_quads[idx2]
                    for b in range(slc.start, slc.stop):
                        if cancel_check and cancel_check():
                            return None
                        if int(index.tile_indices[b]) != tile_index:
                            continue
                        tested += 1
                        if tested >= max_buckets:
                            break
                        tile_combo = index.quad_indices[b]
                        src4 = image_positions[obs_combo].astype(np.float64)
                        dst4 = tile_positions[tile_combo].astype(np.float64)
                        hyp = _derive_similarity(src4, dst4)
                        if hyp is None:
                            continue
                        rot_scale, translation = hyp
                        scale = abs(rot_scale)
                        # Accept only plausible scales (deg/px)
                        if not (1e-5 <= scale <= 1e-2):
                            continue
                        pred = rot_scale * src_all_c + translation
                        err_deg = np.abs(pred - dst_all_c)
                        tol_deg = max(1e-6, float(config.pixel_tolerance) * scale)
                        inliers_mask = err_deg <= tol_deg
                        inliers = int(np.sum(inliers_mask))
                        if inliers <= best_inliers:
                            continue
                        rms_px = float(np.sqrt(np.mean((err_deg[inliers_mask] / max(scale, 1e-12)) ** 2))) if inliers else float("inf")
                        tr = SimilarityTransform(
                            scale=float(scale),
                            rotation=float(np.angle(rot_scale)),
                            translation=(float(translation.real), float(translation.imag)),
                        )
                        st = SimilarityStats(rms_px=rms_px, inliers=inliers)
                        best = (tr, st)
                        best_inliers = inliers
                        if inliers >= config.quality_inliers and rms_px <= config.quality_rms:
                            break
                    if best_inliers >= config.quality_inliers:
                        break
                transform_result = best
            if transform_result is None:
                if cancel_check and cancel_check():
                    return None
                transform_result = estimate_similarity_RANSAC(
                    img_points,
                    tile_points,
                    trials=1200,
                    tol_px=config.pixel_tolerance,
                    min_inliers=4,
                    allow_reflection=bool(config.try_parity_flip),
                    early_stop_inliers=int(getattr(config, "quality_inliers", 60) or 60),
                )
            if transform_result is None:
                continue
            # Build localized inliers and refit similarity on that subset
            transform, _stats0 = transform_result
            scale = float(transform.scale)
            src_all_c = (img_points[:, 0] + 1j * img_points[:, 1]).astype(np.complex128)
            dst_all_c = (tile_points[:, 0] + 1j * tile_points[:, 1]).astype(np.complex128)
            rot_scale = scale * np.exp(1j * transform.rotation)
            translation = complex(*transform.translation)
            if getattr(transform, "parity", 1) < 0:
                src_all_c = np.conj(src_all_c)
            pred = rot_scale * src_all_c + translation
            err_deg = np.abs(pred - dst_all_c)
            tol_deg = max(1e-6, float(config.pixel_tolerance) * max(scale, 1e-12))
            inliers_mask = err_deg <= tol_deg
            if not inliers_mask.any():
                continue
            img_in = img_points[inliers_mask]
            tile_in = tile_points[inliers_mask]
            world_in = tile_world_matches[inliers_mask]
            # Cluster locally around the median predicted position
            pred_in = pred[inliers_mask]
            cx = float(np.median(pred_in.real))
            cy = float(np.median(pred_in.imag))
            dloc = np.hypot(tile_in[:, 0] - cx, tile_in[:, 1] - cy)
            approx_fov_deg = max(image.shape) * max(scale, 1e-12)
            radius = max(0.2, 0.6 * approx_fov_deg)
            local_mask = dloc <= radius
            if local_mask.sum() >= 6:
                img_in = img_in[local_mask]
                tile_in = tile_in[local_mask]
                world_in = world_in[local_mask]
            # Refit similarity on local inliers
            reflected = getattr(transform, "parity", 1) < 0
            hyp2 = _derive_similarity(
                img_in.astype(np.float64),
                tile_in.astype(np.float64),
                reflected=reflected,
            )
            if hyp2 is not None:
                rot_scale2, translation2 = hyp2
                transform = SimilarityTransform(
                    scale=float(abs(rot_scale2)),
                    rotation=float(np.angle(rot_scale2)),
                    translation=(float(translation2.real), float(translation2.imag)),
                    parity=-1 if reflected else 1,
                )
            crpix = (image.shape[1] / 2.0 + 1.0, image.shape[0] / 2.0 + 1.0)
            wcs = tan_from_similarity(
                transform,
                image.shape,
                center_pixel=crpix,
                tile_center=(float(tile_entry.get("center_ra_deg", 0.0)), float(tile_entry.get("center_dec_deg", 0.0))),
            )
            matches_array = _build_matches_array(img_in, world_in)
            # Adapt inlier requirement to available matches (no hardcoded floor);
            # let the user-configured quality_inliers cap the requirement.
            n_pairs = int(matches_array.shape[0])
            adaptive_inliers = min(
                int(config.quality_inliers),
                max(2, int(0.4 * max(0, n_pairs))),
            )
            th_local = {"rms_px": float(config.quality_rms), "inliers": adaptive_inliers}
            stats = validate_solution(wcs, matches_array, th_local)
            final_wcs = wcs
            final_stats = stats
            sip_used = 0
            # NumPy 2.0 removed ndarray.ptp; use np.ptp(array, axis=...) instead
            fov_deg = float(np.hypot(*np.ptp(tile_positions, axis=0))) if tile_positions.size else 0.0
            if final_stats.get("quality") == "GOOD" and needs_sip(final_wcs, final_stats, fov_deg):
                for order in range(2, config.sip_order + 1):
                    if cancel_check and cancel_check():
                        return None
                    try:
                        candidate_wcs, _ = fit_wcs_sip(matches_array, order=order)
                    except Exception as exc:
                        logger.info("SIP fit failed (order=%d): %s", order, exc)
                        continue
                    candidate_stats = validate_solution(candidate_wcs, matches_array, th_local)
                    if candidate_stats["rms_px"] < final_stats["rms_px"]:
                        final_wcs = candidate_wcs
                        final_stats = candidate_stats
                        sip_used = order
                    if not needs_sip(final_wcs, final_stats, fov_deg):
                        break
            stats = final_stats
            if stats.get("quality") != "GOOD":
                logger.info(
                    "validation failed: tile=%s level=%s parity=%s rms=%.3f inliers=%d pairs=%d",
                    candidate_key,
                    level_name,
                    parity_label,
                    float(stats.get("rms_px", float("inf"))),
                    int(stats.get("inliers", 0)),
                    int(matches_array.shape[0]),
                )
                continue
            header_updates = {
                "SOLVED": 1,
                "DBSET": tile_entry.get("family"),
                "TILE_ID": tile_entry.get("tile_key"),
                "RMSPX": stats["rms_px"],
                "INLIERS": stats["inliers"],
                "SIPORD": sip_used,
                "QUALITY": stats["quality"],
                "USED_DB": tile_entry.get("family"),
                "SOLVER": "ZeSolver",
                "SOLVMODE": "BLIND",
            }
            return WcsSolution(
                True,
                f"solution found (level={level_name}, parity={parity_label})",
                final_wcs,
                stats,
                candidate_key,
                header_updates,
            )
        return None

    def _run_levels(
        hashes: np.ndarray,
        parity_label: str,
        *,
        use_px_spec: bool = True,
        use_ra_filter: bool = True,
        levels_seq: list[str] | None = None,
        early_exit_ratio: float | None = None,
    ) -> WcsSolution | None:
        best: WcsSolution | None = None
        preferred_tile: str | None = preferred_tile_hint

        def _is_better(solution: WcsSolution, current: WcsSolution | None) -> bool:
            if current is None:
                return True
            new_inliers = solution.stats.get("inliers", 0)
            cur_inliers = current.stats.get("inliers", 0)
            if new_inliers != cur_inliers:
                return new_inliers > cur_inliers
            new_rms = solution.stats.get("rms_px", float("inf"))
            cur_rms = current.stats.get("rms_px", float("inf"))
            return new_rms < cur_rms

        cur_levels = levels if levels_seq is None else levels_seq
        for level_name in cur_levels:
            solution = _attempt_level(
                level_name,
                hashes,
                preferred_tile,
                parity_label,
                use_px_spec=use_px_spec,
                use_ra_filter=use_ra_filter,
                agg_levels=cur_levels,
                early_exit_ratio=early_exit_ratio,
            )
            # Do not bail out early if the first level fails; try remaining levels (e.g., M/S)
            if solution is None:
                continue
            if _is_better(solution, best):
                best = solution
            preferred_tile = solution.tile_key
        return best

    stage = time.time()
    variants = [(base_hashes, "nominal")]
    if config.try_parity_flip:
        variants.append((base_hashes ^ 1, "mirror"))
    best_solution: WcsSolution | None = None
    levels_all = list(levels)
    levels_fast = [lvl for lvl in ("S",) if lvl in levels_all]
    levels_scale_focus = [lvl for lvl in ("S", "M") if lvl in levels_all] or levels_all

    def _build_level_sets(base_levels: list[str]) -> list[list[str]]:
        sets: list[list[str]] = []
        if config.fast_mode and levels_fast:
            sets.append(levels_fast)
        if base_levels:
            if not sets or sets[-1] != base_levels:
                sets.append(base_levels)
        return sets or [levels_all]

    ra_available = ra_hint is not None and dec_hint is not None
    scale_available = (approx_scale_deg is not None) or (radius_for_filter is not None)
    phase_specs: list[dict[str, Any]] = []
    if ra_available and scale_available:
        phase_specs.append(
            {
                "name": "hinted",
                "require_ra": True,
                "require_scale": True,
                "level_sets": _build_level_sets(levels_scale_focus),
                "use_ra_filter": True,
                "early_exit": 4.0,
            }
        )
    if scale_available:
        phase_specs.append(
            {
                "name": "scale_only",
                "require_ra": False,
                "require_scale": True,
                "level_sets": _build_level_sets(levels_scale_focus),
                "use_ra_filter": False,
                "early_exit": 2.5,
            }
        )
    phase_specs.append(
        {
            "name": "blind",
            "require_ra": False,
            "require_scale": False,
            "level_sets": _build_level_sets(levels_all),
            "use_ra_filter": False,
            "early_exit": None,
        }
    )
    for phase in phase_specs:
        if phase["require_ra"] and not ra_available:
            continue
        if phase["require_scale"] and not scale_available:
            continue
        phase_start = time.time()
        logger.info(
            "phase %s starting (ra_filter=%s, level_sets=%d)",
            phase["name"],
            phase["use_ra_filter"],
            len(phase["level_sets"]),
        )
        for level_seq in phase["level_sets"]:
            for variant_hashes, parity_label in variants:
                solution = _run_levels(
                    variant_hashes,
                    parity_label,
                    use_px_spec=True,
                    use_ra_filter=phase["use_ra_filter"],
                    levels_seq=level_seq,
                    early_exit_ratio=phase["early_exit"],
                )
                if solution is None:
                    continue
                solution.stats["phase"] = phase["name"]
                solution.message += f" (parity={parity_label}, phase={phase['name']})"
                solution.stats["phase_elapsed_s"] = time.time() - phase_start
                solution.stats["phase_levels"] = list(level_seq)
                solution.stats["phase_level_count"] = len(level_seq)
                best_solution = solution
                break
            if best_solution:
                break
        if best_solution:
            break
    _log_phase("candidate search", stage)
    if not best_solution:
        return _finish(WcsSolution(False, "no valid solution", None, {}, None, {}))
    header_updates = {
        **best_solution.header_updates,
        "BLINDVER": ZEBLIND_VERSION,
    }
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
    if is_fits:
        with fits.open(source_path, mode="update", memmap=False) as hdul:
            header_out = hdul[0].header
            for key, value in best_solution.wcs.to_header(relax=True).items():
                header_out[key] = value
            for key, value in header_updates.items():
                header_out[key] = value
            header_out["ZBLNDVER"] = ZEBLIND_VERSION
            hdul.flush()
        logger.info(
            "blind solve succeeded (tile=%s, rms=%.3f px, inliers=%d)",
            best_solution.tile_key,
            best_solution.stats.get("rms_px", float("nan")),
            best_solution.stats.get("inliers", 0),
        )
    else:
        sidecar = source_path.with_suffix(source_path.suffix + ".wcs.json")
        if best_solution.wcs is not None:
            _write_wcs_sidecar(sidecar, best_solution.wcs, header_updates)
            logger.info(
                "blind solve succeeded (tile=%s, wrote %s)",
                best_solution.tile_key,
                sidecar.name,
            )
        else:
            logger.info("blind solve succeeded (tile=%s)", best_solution.tile_key)
    return _finish(
        WcsSolution(
            True,
            best_solution.message,
            best_solution.wcs,
            best_solution.stats,
            best_solution.tile_key,
            header_updates,
        )
    )


def _load_cli_defaults_from_settings() -> dict:
    """Load basic defaults from ~/.zesolver_settings.json if present.

    Avoids importing the GUI module to keep this CLI lean.
    """
    settings_path = Path.home() / ".zesolver_settings.json"
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZeSolver blind solve pipeline (with optional Astrometry.net web backend)")
    parser.add_argument("input", help="Path to the FITS file to solve")
    parser.add_argument("--index-root", required=False, help="Directory containing the blind index")
    # Local blind options
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-stars", type=int, default=800)
    parser.add_argument("--max-quads", type=int, default=12000)
    parser.add_argument("--detect-k-sigma", type=float, default=3.0)
    parser.add_argument("--detect-min-area", type=int, default=5)
    parser.add_argument("--bucket-cap-s", type=int, default=0)
    parser.add_argument("--bucket-cap-m", type=int, default=0)
    parser.add_argument("--bucket-cap-l", type=int, default=0)
    parser.add_argument("--sip-order", type=int, choices=(2, 3), default=2)
    parser.add_argument("--quality-rms", type=float, default=1.0)
    parser.add_argument("--quality-inliers", type=int, default=60)
    parser.add_argument("--pixel-tolerance", type=float, default=3.0)
    parser.add_argument("--downsample", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--ra-hint", type=float, help="Center RA hint in degrees")
    parser.add_argument("--dec-hint", type=float, help="Center Dec hint in degrees")
    parser.add_argument("--radius-hint", type=float, help="Search radius hint in degrees")
    parser.add_argument("--focal-length", type=float, help="Focal length hint in millimetres")
    parser.add_argument("--pixel-size", type=float, help="Pixel size hint in microns")
    parser.add_argument("--pixel-scale", type=float, help="Pixel scale hint in arcsec/pixel")
    parser.add_argument("--pixel-scale-min", type=float, help="Minimum pixel scale bound (arcsec/pixel)")
    parser.add_argument("--pixel-scale-max", type=float, help="Maximum pixel scale bound (arcsec/pixel)")
    parser.add_argument(
        "--tile-cache-size",
        type=int,
        default=None,
        help="Tile position cache capacity (overrides ZE_TILE_CACHE_SIZE, default=128)",
    )
    parser.add_argument("--log-level", default="INFO")
    # Backend selection
    parser.add_argument("--solver-backend", choices=("local", "astrometry"), default=None)
    parser.add_argument("--astrometry-api-url", default=None)
    parser.add_argument("--astrometry-api-key", default=None)
    parser.add_argument("--astrometry-use-hints", action="store_true")
    parser.add_argument("--astrometry-timeout-s", type=int, default=None)
    parser.add_argument("--astrometry-parallel-jobs", type=int, default=None)
    parser.add_argument("--astrometry-fallback-local", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # Load defaults from settings file if flags not provided
    defaults = _load_cli_defaults_from_settings()
    tile_cache_size = args.tile_cache_size
    if tile_cache_size is None:
        cfg_value = defaults.get("tile_cache_size") if isinstance(defaults, dict) else None
        if isinstance(cfg_value, int):
            tile_cache_size = cfg_value
    if tile_cache_size is None:
        env_value = os.environ.get("ZE_TILE_CACHE_SIZE")
        if env_value:
            try:
                tile_cache_size = int(env_value)
            except ValueError:
                tile_cache_size = None
    if tile_cache_size is None:
        tile_cache_size = _TILE_CACHE_DEFAULT_CAPACITY
    tile_cache_size = max(0, int(tile_cache_size))
    backend = args.solver_backend or (defaults.get("solver_backend") if isinstance(defaults.get("solver_backend"), str) else "local")
    backend = (backend or "local").lower()
    if backend == "astrometry":
        from .astrometry_backend import AstrometryConfig, solve_single  # lazy import

        api_url = args.astrometry_api_url or defaults.get("astrometry_api_url") or "https://nova.astrometry.net/api"
        api_key = args.astrometry_api_key or defaults.get("astrometry_api_key") or os.environ.get("ASTROMETRY_API_KEY")
        if not api_key:
            logger.error("astrometry backend selected but no API key provided")
            return 2
        use_hints = bool(args.astrometry_use_hints or defaults.get("astrometry_use_hints", True))
        timeout_s = int(args.astrometry_timeout_s or defaults.get("astrometry_timeout_s", 600))
        parallel = int(args.astrometry_parallel_jobs or defaults.get("astrometry_parallel_jobs", 2))
        fallback_local = bool(args.astrometry_fallback_local or defaults.get("astrometry_fallback_local", True))
        cfg = AstrometryConfig(
            api_url=api_url,
            api_key=api_key,
            parallel_jobs=parallel,
            timeout_s=timeout_s,
            use_hints=use_hints,
            fallback_local=fallback_local,
            index_root=(args.index_root or defaults.get("index_root")),
        )
        res = solve_single(args.input, cfg)
        if res.success:
            logger.info("astrometry solve succeeded for %s", args.input)
            return 0
        logger.error("astrometry solve failed: %s", res.message)
        return 2

    # Local blind backend (default)
    if not args.index_root:
        logger.error("--index-root is required for local backend")
        return 2
    config = SolveConfig(
        max_candidates=args.max_candidates,
        max_stars=args.max_stars,
        max_quads=args.max_quads,
        detect_k_sigma=max(0.5, float(args.detect_k_sigma)),
        detect_min_area=max(1, int(args.detect_min_area)),
        bucket_cap_S=max(0, int(args.bucket_cap_s or 0)),
        bucket_cap_M=max(0, int(args.bucket_cap_m or 0)),
        bucket_cap_L=max(0, int(args.bucket_cap_l or 0)),
        sip_order=args.sip_order,
        quality_rms=args.quality_rms,
        quality_inliers=args.quality_inliers,
        pixel_tolerance=args.pixel_tolerance,
        log_level=args.log_level.upper(),
        downsample=args.downsample,
        ra_hint_deg=args.ra_hint,
        dec_hint_deg=args.dec_hint,
        radius_hint_deg=args.radius_hint,
        focal_length_mm=args.focal_length,
        pixel_size_um=args.pixel_size,
        pixel_scale_arcsec=args.pixel_scale,
        pixel_scale_min_arcsec=args.pixel_scale_min,
        pixel_scale_max_arcsec=args.pixel_scale_max,
        tile_cache_size=tile_cache_size,
    )
    solution = solve_blind(args.input, args.index_root, config=config)
    if solution.success:
        logger.info("blind solve succeeded for %s", args.input)
        return 0
    logger.error("blind solve failed: %s", solution.message)
    return 2


zeblindsolve = main
