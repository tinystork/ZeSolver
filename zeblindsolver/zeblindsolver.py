# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : MIT (voir pyproject.toml / repository metadata)               ║
# ║                                                                                   ║
# ║ Remerciements amont :                                                             ║
# ║ - ASTAP, par Han Kleijn                                                           ║
# ║ - Astrometry.net, par Dustin Lang, David W. Hogg, Keir Mierle, et al.            ║
# ║                                                                                   ║
# ║ Description FR :                                                                  ║
# ║ Ce code sert à transformer des nuages de photons en solutions WCS et en images   ║
# ║ astronomiques exploitables. Merci de créditer les auteurs et projets amont lors   ║
# ║ de toute réutilisation.                                                           ║
# ║                                                                                   ║
# ║ EN Description:                                                                    ║
# ║ This code helps turn clouds of photons into usable WCS solutions and astronomical ║
# ║ imagery outputs. Please credit both project authors and upstream references when  ║
# ║ reusing this work.                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
# """

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, replace
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
from .wcs_header import apply_wcs_solution_to_header, validate_wcs_for_zemosaic

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
    collect_matches_vectorized_experimental: bool = False
    fail_attempt_budget_s: float = 70.0
    fail_attempt_min_validations: int = 18
    fail_attempt_max_best_inliers: int = 4
    fail_attempt_min_candidates: int = 20
    global_budget_fast_s: float = 8.0
    global_budget_slow_s: float = 18.0
    verify_logodds_enabled: bool = False
    verify_logodds_bail: float = -24.0
    verify_logodds_stoplooking: float = 24.0
    verify_logodds_min_validations: int = 8
    hard_max_candidates_tried: int = 0
    hard_max_validations: int = 0
    depth_ladder_enabled: bool = False
    depth_ladder_caps: tuple[int, ...] = (80, 160, 500)


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


def _blind_geometric_guardrails(img_points: np.ndarray, image_shape: tuple[int, int]) -> tuple[bool, dict[str, float | int | str]]:
    """Lightweight geometric sanity checks before accepting a blind solution.

    Inspired by ASTAP-like robustness: reject highly fragile fits with weak spatial
    support (tiny footprint or near-collinear inliers), while staying permissive on
    very small sparse sets.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    n = int(img_points.shape[0]) if img_points is not None else 0
    if n <= 0 or h <= 0 or w <= 0:
        return False, {"reason": "invalid geometry inputs", "n": n}
    x = np.asarray(img_points[:, 0], dtype=np.float64)
    y = np.asarray(img_points[:, 1], dtype=np.float64)
    bw = float(np.nanmax(x) - np.nanmin(x)) if x.size else 0.0
    bh = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
    cov_x = bw / max(float(w), 1.0)
    cov_y = bh / max(float(h), 1.0)
    cov_area = cov_x * cov_y
    # Keep permissive behavior for tiny sparse sets: only footprint checks.
    if n < 10:
        ok = (max(cov_x, cov_y) >= 0.08) and (cov_area >= 0.003)
        return ok, {
            "reason": "ok" if ok else "insufficient spatial footprint",
            "n": n,
            "cov_x": cov_x,
            "cov_y": cov_y,
            "cov_area": cov_area,
            "cond": float("nan"),
        }
    pts = np.column_stack((x, y))
    c = np.cov(pts.T)
    try:
        evals = np.linalg.eigvalsh(c)
        lam_min = float(max(np.min(evals), 1e-12))
        lam_max = float(max(np.max(evals), 1e-12))
        cond = lam_max / lam_min
    except Exception:
        cond = float("inf")
    ok = (max(cov_x, cov_y) >= 0.10) and (cov_area >= 0.005) and (cond <= 2.0e4)
    reason = "ok" if ok else ("near-collinear inliers" if cond > 2.0e4 else "insufficient spatial footprint")
    return ok, {
        "reason": reason,
        "n": n,
        "cov_x": cov_x,
        "cov_y": cov_y,
        "cov_area": cov_area,
        "cond": cond,
    }


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
    use_vectorized: bool = False,
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
    if use_vectorized:
        # Experimental vectorized path (same semantics, lower Python overhead in pair filtering).
        vote_items = list(votes.items())
        pair_idx = np.asarray([k for k, _ in vote_items], dtype=np.int64)
        counts = np.asarray([int(c) for _, c in vote_items], dtype=np.int64)
        percentile = min(95, max(5, vote_percentile))
        thr = max(1, int(np.percentile(counts, percentile)))
        keep = counts >= thr
        if np.any(keep):
            sel = np.flatnonzero(keep)
            sel = sel[np.argsort(counts[sel])[::-1]]
        else:
            top_n = max(200, int(len(counts) * 0.1))
            top_n = min(top_n, len(counts))
            sel = np.argsort(counts)[-top_n:][::-1]

        pair_idx = pair_idx[sel]
        img_cap = int(image_positions.shape[0])
        tile_cap = int(tile_positions.shape[0])
        if img_cap == 0 or tile_cap == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, empty

        valid = (pair_idx[:, 0] >= 0) & (pair_idx[:, 0] < img_cap) & (pair_idx[:, 1] >= 0) & (pair_idx[:, 1] < tile_cap)
        dropped = int(valid.size - np.count_nonzero(valid))
        if not np.any(valid):
            empty = np.empty((0, 2), dtype=np.float32)
            if dropped:
                logger.warning("discarded %d invalid vote pairs (image=%d, tile=%d)", dropped, img_cap, tile_cap)
            return empty, empty, empty
        if dropped:
            logger.debug("discarded %d vote pairs outside valid ranges (image=%d, tile=%d)", dropped, img_cap, tile_cap)

        pair_idx = pair_idx[valid]
        cap = min(int(pair_idx.shape[0]), max(800, int(img_cap * 12)))
        if cap <= 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, empty
        pair_idx = pair_idx[:cap]
        img_ids = pair_idx[:, 0].astype(np.intp, copy=False)
        tile_ids = pair_idx[:, 1].astype(np.intp, copy=False)
        img_pts = np.asarray(image_positions[img_ids], dtype=np.float32)
        tile_pts = np.asarray(tile_positions[tile_ids], dtype=np.float32)
        world_pts = np.asarray(tile_world[tile_ids], dtype=np.float32)
        return img_pts, tile_pts, world_pts

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
    img_cap = image_positions.shape[0]
    tile_cap = tile_positions.shape[0]
    if img_cap == 0 or tile_cap == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, empty
    filtered: list[tuple[int, int, int]] = []
    dropped = 0
    for i, t, c in pairs:
        if 0 <= i < img_cap and 0 <= t < tile_cap:
            filtered.append((i, t, c))
        else:
            dropped += 1
    if not filtered:
        empty = np.empty((0, 2), dtype=np.float32)
        if dropped:
            logger.warning("discarded %d invalid vote pairs (image=%d, tile=%d)", dropped, img_cap, tile_cap)
        return empty, empty, empty
    if dropped:
        logger.debug("discarded %d vote pairs outside valid ranges (image=%d, tile=%d)", dropped, img_cap, tile_cap)
    # Cap pairs to bound runtime but keep enough support for RANSAC
    cap = min(len(filtered), max(800, int(img_cap * 12)))
    chosen = filtered[:cap]
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


def _coerce_depth_ladder_caps(raw_caps: Any) -> list[int]:
    items: list[Any]
    if isinstance(raw_caps, str):
        items = [part.strip() for part in raw_caps.replace(';', ',').split(',') if part.strip()]
    elif isinstance(raw_caps, Iterable) and not isinstance(raw_caps, (bytes, bytearray)):
        items = list(raw_caps)
    else:
        items = [raw_caps]

    caps: list[int] = []
    for item in items:
        try:
            value = int(item)
        except Exception:
            continue
        if value > 0:
            caps.append(value)

    dedup: list[int] = []
    seen: set[int] = set()
    for value in sorted(caps):
        if value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return dedup


def solve_blind(
    input_fits: Path | str,
    index_root: Path | str,
    *,
    config: SolveConfig | None = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    prep_cache: dict[str, Any] | None = None,
    _depth_ladder_internal: bool = False,
    _depth_ladder_stage: int = 0,
    _depth_ladder_total: int = 0,
    _depth_ladder_caps: tuple[int, ...] | None = None,
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

    if not _depth_ladder_internal and bool(getattr(config, "depth_ladder_enabled", False)):
        raw_caps = _coerce_depth_ladder_caps(getattr(config, "depth_ladder_caps", (80, 160, 500)))
        max_stars_cfg = max(0, int(getattr(config, "max_stars", 0) or 0))
        if max_stars_cfg > 0:
            capped = [v for v in raw_caps if v < max_stars_cfg]
            capped.append(max_stars_cfg)
            raw_caps = capped
        if not raw_caps:
            raw_caps = [max_stars_cfg] if max_stars_cfg > 0 else [80, 160, 500]
        ladder_caps = [int(v) for v in raw_caps if int(v) > 0]
        if len(ladder_caps) > 1:
            logger.info(
                "depth ladder enabled: stages=%s",
                " -> ".join(str(v) for v in ladder_caps),
            )
            ladder_rows: list[dict[str, Any]] = []
            last_fail: WcsSolution | None = None
            total_stages = len(ladder_caps)
            for stage_idx, stage_max_stars in enumerate(ladder_caps, start=1):
                if cancel_check and cancel_check():
                    return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
                stage_cfg = replace(
                    config,
                    max_stars=int(stage_max_stars),
                    depth_ladder_enabled=False,
                )
                if stage_idx == 1 and total_stages > 1:
                    stage_cfg = replace(
                        stage_cfg,
                        hard_max_candidates_tried=(
                            min(int(stage_cfg.hard_max_candidates_tried), 96)
                            if int(stage_cfg.hard_max_candidates_tried or 0) > 0
                            else 96
                        ),
                    )
                stage_started = time.time()
                logger.info(
                    "depth ladder stage %d/%d: max_stars=%d",
                    stage_idx,
                    total_stages,
                    int(stage_max_stars),
                )
                stage_solution = solve_blind(
                    input_fits,
                    index_root,
                    config=stage_cfg,
                    cancel_check=cancel_check,
                    prep_cache=prep_cache,
                    _depth_ladder_internal=True,
                    _depth_ladder_stage=stage_idx,
                    _depth_ladder_total=total_stages,
                    _depth_ladder_caps=tuple(ladder_caps),
                )
                stage_elapsed = float(time.time() - stage_started)
                stage_stats = dict(stage_solution.stats or {})
                ladder_rows.append(
                    {
                        "stage": int(stage_idx),
                        "total_stages": int(total_stages),
                        "max_stars": int(stage_max_stars),
                        "success": bool(stage_solution.success),
                        "elapsed_s": stage_elapsed,
                        "attempt_elapsed_s": float(stage_stats.get("attempt_elapsed_s", stage_elapsed) or stage_elapsed),
                        "total_candidates_tried": int(stage_stats.get("total_candidates_tried", 0) or 0),
                        "best_fail_inliers": int(stage_stats.get("best_fail_inliers", -1) or -1),
                        "fail_validation_count": int(stage_stats.get("fail_validation_count", 0) or 0),
                    }
                )
                if stage_solution.success:
                    merged_stats = dict(stage_stats)
                    merged_stats["depth_ladder_enabled"] = True
                    merged_stats["depth_ladder_used"] = True
                    merged_stats["depth_ladder_stage"] = int(stage_idx)
                    merged_stats["depth_ladder_total"] = int(total_stages)
                    merged_stats["depth_ladder_caps"] = [int(v) for v in ladder_caps]
                    merged_stats["depth_ladder_rows"] = list(ladder_rows)
                    stage_solution.stats = merged_stats
                    if stage_solution.message:
                        stage_solution.message += f" [depth_ladder {stage_idx}/{total_stages}, max_stars={int(stage_max_stars)}]"
                    else:
                        stage_solution.message = f"depth ladder success (stage {stage_idx}/{total_stages}, max_stars={int(stage_max_stars)})"
                    return _finish(stage_solution)
                last_fail = stage_solution
            if last_fail is None:
                return _finish(WcsSolution(False, "no valid solution", None, {}, None, {}))
            fail_stats = dict(last_fail.stats or {})
            fail_stats["depth_ladder_enabled"] = True
            fail_stats["depth_ladder_used"] = True
            fail_stats["depth_ladder_stage"] = int(total_stages)
            fail_stats["depth_ladder_total"] = int(total_stages)
            fail_stats["depth_ladder_caps"] = [int(v) for v in ladder_caps]
            fail_stats["depth_ladder_rows"] = list(ladder_rows)
            return _finish(
                WcsSolution(
                    False,
                    last_fail.message or "no valid solution",
                    last_fail.wcs,
                    fail_stats,
                    last_fail.tile_key,
                    last_fail.header_updates,
                )
            )

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
    ransac_seed_base = zlib.crc32(str(source_path).encode("utf-8", errors="ignore")) & 0xFFFFFFFF
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
    expanded_tile_indices: set[int] | None = None
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

    # Build a widened RA/Dec cone for a second hinted pass before global blind.
    # This keeps ASTAP-like progressive expansion while preserving global fallback safety.
    if ra_hint is not None and dec_hint is not None and tile_count > 0:
        base_radius = radius_for_filter
        if base_radius is None:
            fallback_fov = approx_fov_deg or 1.5
            base_radius = min(8.0, max(0.8, fallback_fov * 1.2))
        base_count = len(allowed_tile_indices) if allowed_tile_indices is not None else tile_count
        expanded_candidate: set[int] | None = None
        expanded_radius: float | None = None
        expansion_radii = [
            min(20.0, max(base_radius * 1.8, base_radius + 0.40)),
            min(20.0, max(base_radius * 3.0, base_radius + 1.20)),
            min(20.0, max(base_radius * 5.0, base_radius + 2.40)),
        ]
        seen_r = set()
        for radius_try in expansion_radii:
            rr = round(float(radius_try), 3)
            if rr in seen_r or rr <= float(base_radius):
                continue
            seen_r.add(rr)
            sel = select_tiles_in_cone(manifest, ra_hint, dec_hint, float(radius_try))
            if not sel:
                continue
            if len(sel) >= tile_count:
                continue
            if len(sel) <= base_count:
                continue
            expanded_candidate = set(sel)
            expanded_radius = float(radius_try)
            # Good enough neighborhood, keep bounded and avoid drifting back to near-global.
            if len(sel) >= max(4, 2 * max(1, base_count)):
                break
        if expanded_candidate is not None:
            expanded_tile_indices = expanded_candidate
            logger.info(
                "RA/Dec widened cone prepared: %d/%d tiles (radius=%.2f°)",
                len(expanded_tile_indices),
                tile_count,
                float(expanded_radius or 0.0),
            )
    detect_k_sigma = max(0.5, float(getattr(config, "detect_k_sigma", 3.0)))
    detect_min_area = max(1, int(getattr(config, "detect_min_area", 5)))
    cache_key: str | None = None
    cache_sig: tuple[int, int] | None = None
    cached_stars: np.ndarray | None = None
    cache_entry: dict[str, Any] | None = None
    if prep_cache is not None:
        try:
            resolved_source = source_path.resolve()
        except Exception:
            resolved_source = source_path
        cache_key = str(resolved_source)
        try:
            stat = resolved_source.stat()
            cache_sig = (int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))), int(stat.st_size))
        except Exception:
            cache_sig = None
        entry = prep_cache.get(cache_key)
        if isinstance(entry, dict):
            try:
                if (
                    entry.get("sig") == cache_sig
                    and int(entry.get("downsample", -1)) == int(downsample_factor)
                    and float(entry.get("detect_k_sigma", -1.0)) == float(detect_k_sigma)
                    and int(entry.get("detect_min_area", -1)) == int(detect_min_area)
                ):
                    cache_entry = entry
                    stars_cached = entry.get("stars")
                    if isinstance(stars_cached, np.ndarray) and stars_cached.size > 0:
                        cached_stars = stars_cached
                        logger.info("reusing blind star detections from prep cache (%d stars)", int(stars_cached.shape[0]))
            except Exception:
                cached_stars = None
                cache_entry = None

    _log_phase("read image", stage)
    if cached_stars is None:
        stage = time.time()
        # Use a smaller median kernel for speed on typical Seestar frames
        kernel = max(5, int(round(15 / downsample_factor)))
        try:
            work_image = remove_background(work_image, kernel_size=kernel)
        except TypeError:
            # Test/mocked call-sites may provide a simplified signature.
            work_image = remove_background(work_image)
        pyramid = build_pyramid(work_image)
        _log_phase("preprocess", stage)
        detection = pyramid[-1]
        stage = time.time()
        min_fwhm = max(1.0, 1.5 / downsample_factor)
        max_fwhm = max(2.5, 8.0 / downsample_factor)
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
    else:
        stage = time.time()
        _log_phase("preprocess", stage)
        stage = time.time()
        stars = cached_stars
        if config.max_stars and stars.size > config.max_stars:
            stars = stars[: config.max_stars]

    if prep_cache is not None and cache_key is not None:
        if cache_entry is None:
            cache_entry = {}
        try:
            cache_entry.update(
                {
                    "sig": cache_sig,
                    "downsample": int(downsample_factor),
                    "detect_k_sigma": float(detect_k_sigma),
                    "detect_min_area": int(detect_min_area),
                    "stars": stars.copy(),
                }
            )
            prep_cache[cache_key] = cache_entry
        except Exception:
            pass

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
    quad_strategy = "sparse_triples" if stars.shape[0] <= 96 else "local_brightness"
    requested_max_quads = max(1, int(config.max_quads))
    target_max_quads = requested_max_quads
    if prep_cache is not None:
        try:
            hinted_target = int(prep_cache.get("__target_max_quads__", requested_max_quads) or requested_max_quads)
            target_max_quads = max(requested_max_quads, hinted_target)
        except Exception:
            target_max_quads = requested_max_quads

    quads_cache: dict[str, Any] | None = None
    quads_full: np.ndarray | None = None
    if isinstance(cache_entry, dict):
        qc = cache_entry.get("quads_cache")
        if isinstance(qc, dict):
            try:
                if (
                    str(qc.get("strategy", "")) == quad_strategy
                    and int(qc.get("star_count", -1)) == int(stars.shape[0])
                    and isinstance(qc.get("quads"), np.ndarray)
                ):
                    quads_cache = qc
            except Exception:
                quads_cache = None

    if quads_cache is not None:
        try:
            q_cached = quads_cache.get("quads")
            q_cached_n = int(quads_cache.get("max_quads", 0) or 0)
            if isinstance(q_cached, np.ndarray) and q_cached_n >= target_max_quads and q_cached.shape[0] >= target_max_quads:
                quads_full = q_cached[:target_max_quads]
                logger.info(
                    "reusing blind quads from prep cache (strategy=%s, cached=%d, using=%d)",
                    quad_strategy,
                    int(q_cached_n),
                    int(target_max_quads),
                )
        except Exception:
            quads_full = None

    if quads_full is None:
        quads_full = sample_quads(obs_stars, target_max_quads, strategy=quad_strategy)
        logger.info(
            "quad sampling strategy=%s produced %d quads (requested=%d)",
            quad_strategy,
            int(quads_full.shape[0]),
            int(target_max_quads),
        )
        if isinstance(cache_entry, dict):
            try:
                cache_entry["quads_cache"] = {
                    "strategy": quad_strategy,
                    "star_count": int(stars.shape[0]),
                    "max_quads": int(quads_full.shape[0]),
                    "quads": quads_full.copy(),
                }
                cache_entry.pop("base_hash_full", None)
                cache_entry["level_hash_full"] = {}
            except Exception:
                pass
    else:
        logger.info(
            "quad sampling strategy=%s reused %d cached quads (requested=%d)",
            quad_strategy,
            int(quads_full.shape[0]),
            int(target_max_quads),
        )

    active_quads_n = min(int(requested_max_quads), int(quads_full.shape[0]))
    quads = quads_full[:active_quads_n]
    if quads.size == 0:
        return _finish(WcsSolution(False, "no quads sampled", None, {}, None, {}))
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))

    def _bundle_subset_and_dedup(
        bundle: dict[str, Any] | None,
        *,
        subset_quads: int,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if not isinstance(bundle, dict):
            return np.zeros(0, dtype=np.uint64), np.zeros((0, 4), dtype=np.uint16), np.zeros(0, dtype=np.uint32), 0
        h_all = bundle.get("hashes")
        q_all = bundle.get("indices")
        s_all = bundle.get("source_indices")
        if not isinstance(h_all, np.ndarray) or not isinstance(q_all, np.ndarray):
            return np.zeros(0, dtype=np.uint64), np.zeros((0, 4), dtype=np.uint16), np.zeros(0, dtype=np.uint32), 0
        if isinstance(s_all, np.ndarray) and s_all.shape[0] == h_all.shape[0]:
            mask = s_all < int(subset_quads)
            raw_hashes = h_all[mask]
            raw_quads = q_all[mask]
            raw_count = int(np.count_nonzero(mask))
        else:
            raw_hashes = h_all
            raw_quads = q_all
            raw_count = int(h_all.shape[0])
        dedup_h, dedup_q, dedup_c = _deduplicate_hashes(raw_hashes, raw_quads, label=label)
        return dedup_h, dedup_q, dedup_c, raw_count

    base_bundle: dict[str, Any] | None = None
    if isinstance(cache_entry, dict):
        bb = cache_entry.get("base_hash_full")
        if isinstance(bb, dict):
            try:
                if (
                    str(bb.get("strategy", "")) == quad_strategy
                    and int(bb.get("star_count", -1)) == int(stars.shape[0])
                    and int(bb.get("max_quads", -1)) == int(quads_full.shape[0])
                ):
                    base_bundle = bb
            except Exception:
                base_bundle = None

    if base_bundle is None:
        obs_hash_full = hash_quads(quads_full, image_positions, return_source_indices=True)
        base_bundle = {
            "strategy": quad_strategy,
            "star_count": int(stars.shape[0]),
            "max_quads": int(quads_full.shape[0]),
            "hashes": obs_hash_full.hashes,
            "indices": obs_hash_full.indices,
            "source_indices": obs_hash_full.source_indices,
        }
        if isinstance(cache_entry, dict):
            try:
                cache_entry["base_hash_full"] = base_bundle
            except Exception:
                pass
    else:
        logger.info(
            "reusing blind base hashes from prep cache (max_quads=%d)",
            int(quads_full.shape[0]),
        )

    base_hashes, base_quads, base_counts, base_raw_hash_count = _bundle_subset_and_dedup(
        base_bundle,
        subset_quads=active_quads_n,
        label="base",
    )
    logger.info(
        "sampled %d quads producing %d hashes (%d unique)",
        int(active_quads_n),
        int(base_raw_hash_count),
        int(base_hashes.size),
    )

    thresholds = {"rms_px": config.quality_rms, "inliers": config.quality_inliers}
    fail_best_inliers_seen = -1
    fail_validation_count = 0
    phases_attempted: list[str] = []

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

    def _spec_signature(spec: QuadLevelSpec | None) -> tuple[Any, ...]:
        if spec is None:
            return ("none",)
        return (
            str(spec.name),
            round(float(spec.min_area), 8),
            round(float(spec.max_area), 8),
            None if spec.min_diameter is None else round(float(spec.min_diameter), 8),
            None if spec.max_diameter is None else round(float(spec.max_diameter), 8),
        )

    # Precompute observed hashes per level with pixel-adapted specs when possible
    level_hash_cache: dict[str, Any]
    if isinstance(cache_entry, dict) and isinstance(cache_entry.get("level_hash_full"), dict):
        level_hash_cache = cache_entry.get("level_hash_full")
    else:
        level_hash_cache = {}
        if isinstance(cache_entry, dict):
            cache_entry["level_hash_full"] = level_hash_cache

    obs_by_level: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lvl in levels:
        if cancel_check and cancel_check():
            return WcsSolution(False, "cancelled", None, {}, None, {})
        px = _spec_pixels(lvl)
        if px is None:
            obs_by_level[lvl] = (base_hashes, base_quads, base_counts)
            continue

        spec_sig = _spec_signature(px)
        lvl_bundle: dict[str, Any] | None = None
        cached_lvl = level_hash_cache.get(lvl)
        if isinstance(cached_lvl, dict):
            try:
                if (
                    str(cached_lvl.get("strategy", "")) == quad_strategy
                    and int(cached_lvl.get("star_count", -1)) == int(stars.shape[0])
                    and int(cached_lvl.get("max_quads", -1)) == int(quads_full.shape[0])
                    and tuple(cached_lvl.get("spec_sig", ())) == spec_sig
                ):
                    lvl_bundle = cached_lvl
            except Exception:
                lvl_bundle = None

        if lvl_bundle is None:
            oh_full = hash_quads(quads_full, image_positions, spec=px, return_source_indices=True)
            lvl_bundle = {
                "strategy": quad_strategy,
                "star_count": int(stars.shape[0]),
                "max_quads": int(quads_full.shape[0]),
                "spec_sig": spec_sig,
                "hashes": oh_full.hashes,
                "indices": oh_full.indices,
                "source_indices": oh_full.source_indices,
            }
            level_hash_cache[lvl] = lvl_bundle
        else:
            logger.info("reusing blind level hashes from prep cache (level=%s, max_quads=%d)", lvl, int(quads_full.shape[0]))

        lvl_hashes, lvl_quads, lvl_counts, _ = _bundle_subset_and_dedup(
            lvl_bundle,
            subset_quads=active_quads_n,
            label=lvl,
        )
        if lvl_hashes.size == 0 and base_hashes.size > 0:
            logger.debug(
                "level %s pixel-adapted filter produced no hashes, falling back to base hash set",
                lvl,
            )
            obs_by_level[lvl] = (base_hashes, base_quads, base_counts)
        else:
            obs_by_level[lvl] = (lvl_hashes, lvl_quads, lvl_counts)

    _log_phase("detect/quads", stage)

    candidate_search_cache: dict[tuple[Any, ...], list[tuple[str, int]]] = {}
    level_lookup_cache: dict[tuple[str, bool], tuple[QuadIndex | None, list[slice] | None]] = {}
    collect_matches_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    allowed_tiles_sig_cache: dict[int, tuple[int, ...]] = {}
    phase_perf: dict[str, dict[str, Any]] = {}
    collect_metrics: dict[str, Any] = {
        "calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "compute_s": 0.0,
    }
    attempt_started = time.time()
    total_candidates_tried = 0
    fail_early_abort = False
    fail_early_abort_reason: str | None = None
    fail_early_abort_phase: str | None = None
    fail_budget_s = max(0.0, float(getattr(config, "fail_attempt_budget_s", 70.0) or 0.0))
    fail_budget_min_validations = max(0, int(getattr(config, "fail_attempt_min_validations", 18) or 0))
    fail_budget_max_inliers = max(0, int(getattr(config, "fail_attempt_max_best_inliers", 4) or 0))
    fail_budget_min_candidates = max(0, int(getattr(config, "fail_attempt_min_candidates", 20) or 0))
    verify_logodds_enabled = bool(getattr(config, "verify_logodds_enabled", False))
    verify_logodds_bail = float(getattr(config, "verify_logodds_bail", -24.0) or -24.0)
    verify_logodds_stoplooking = float(getattr(config, "verify_logodds_stoplooking", 24.0) or 24.0)
    verify_logodds_min_validations = max(0, int(getattr(config, "verify_logodds_min_validations", 8) or 0))
    verify_logodds_cum = 0.0
    verify_logodds_last = 0.0
    hard_max_candidates_tried = max(0, int(getattr(config, "hard_max_candidates_tried", 0) or 0))
    hard_max_validations = max(0, int(getattr(config, "hard_max_validations", 0) or 0))

    def _phase_slot(name: str) -> dict[str, Any]:
        slot = phase_perf.get(name)
        if slot is None:
            slot = {
                "elapsed_s": 0.0,
                "level_calls": 0,
                "candidates_seen": 0,
                "candidates_considered": 0,
                "candidates_tried": 0,
                "solutions": 0,
                "failed_validations": 0,
                "early_abort": False,
                "early_abort_reason": None,
                "verify_logodds_cum": 0.0,
                "verify_logodds_last": 0.0,
            }
            phase_perf[name] = slot
        return slot

    def _maybe_fail_early_abort(phase_name: str) -> bool:
        nonlocal fail_early_abort, fail_early_abort_reason, fail_early_abort_phase, verify_logodds_cum
        if fail_early_abort:
            return True
        if phase_name not in {"scale_only", "blind"}:
            return False

        if (
            verify_logodds_enabled
            and verify_logodds_min_validations > 0
            and fail_validation_count >= verify_logodds_min_validations
            and verify_logodds_cum >= verify_logodds_stoplooking
        ):
            return False

        def _set_abort(reason: str) -> bool:
            nonlocal fail_early_abort, fail_early_abort_reason, fail_early_abort_phase
            fail_early_abort = True
            fail_early_abort_phase = phase_name
            fail_early_abort_reason = reason
            try:
                slot = _phase_slot(phase_name)
                slot["early_abort"] = True
                slot["early_abort_reason"] = str(fail_early_abort_reason)
                slot["verify_logodds_cum"] = float(verify_logodds_cum)
            except Exception:
                pass
            logger.info(
                "blind fail-early abort triggered in phase %s: %s",
                phase_name,
                fail_early_abort_reason,
            )
            return True

        if hard_max_candidates_tried > 0 and total_candidates_tried >= hard_max_candidates_tried:
            return _set_abort(
                f"hard_budget candidates_tried>={hard_max_candidates_tried}"
                f" (validations={fail_validation_count}, best_inliers={fail_best_inliers_seen})"
            )

        if hard_max_validations > 0 and fail_validation_count >= hard_max_validations:
            return _set_abort(
                f"hard_budget validations>={hard_max_validations}"
                f" (candidates={total_candidates_tried}, best_inliers={fail_best_inliers_seen})"
            )

        if (
            verify_logodds_enabled
            and fail_validation_count >= verify_logodds_min_validations
            and fail_best_inliers_seen <= max(fail_budget_max_inliers, 8)
            and verify_logodds_cum <= verify_logodds_bail
        ):
            return _set_abort(
                f"logodds_bail reached ({verify_logodds_cum:.2f}<={verify_logodds_bail:.2f})"
                f" after validations={fail_validation_count}, candidates={total_candidates_tried}, best_inliers={fail_best_inliers_seen}"
            )

        if fail_budget_s <= 0.0:
            return False
        elapsed = time.time() - attempt_started
        if elapsed < fail_budget_s:
            return False
        if fail_validation_count < fail_budget_min_validations:
            return False
        if fail_best_inliers_seen > fail_budget_max_inliers:
            return False
        if total_candidates_tried < fail_budget_min_candidates:
            return False
        return _set_abort(
            f"elapsed>{fail_budget_s:.1f}s and weak_support"
            f" (best_inliers={fail_best_inliers_seen}, validations={fail_validation_count}, candidates={total_candidates_tried})"
        )

    def _update_verify_logodds(*, phase_slot: dict[str, Any], inliers: int, rms_px: float) -> None:
        nonlocal verify_logodds_cum, verify_logodds_last
        if not verify_logodds_enabled:
            return
        safe_inliers = max(0, int(inliers))
        safe_rms = float(rms_px)
        if not np.isfinite(safe_rms) or safe_rms <= 0.0:
            safe_rms = max(float(getattr(config, "quality_rms", 1.2) or 1.2), 1.0) * 4.0
        cand = math.log1p(float(safe_inliers)) - (1.35 * math.log(max(1e-6, safe_rms))) - 1.4
        if safe_inliers <= 2:
            cand -= 0.9
        elif safe_inliers >= max(8, int(0.5 * int(getattr(config, "quality_inliers", 40) or 40))):
            cand += 0.5
        verify_logodds_last = float(cand)
        verify_logodds_cum = float(verify_logodds_cum + cand)
        phase_slot["verify_logodds_last"] = float(verify_logodds_last)
        phase_slot["verify_logodds_cum"] = float(verify_logodds_cum)

    def _allowed_tiles_key(tiles: set[int] | None) -> tuple[Any, ...]:
        if tiles is None:
            return ("all",)
        key_id = id(tiles)
        sig = allowed_tiles_sig_cache.get(key_id)
        if sig is None:
            sig = tuple(sorted(int(v) for v in tiles))
            allowed_tiles_sig_cache[key_id] = sig
        return ("tiles", sig)

    def _collect_cache_key(
        lvl: str,
        tile_idx: int,
        use_px: bool,
        ohashes: np.ndarray,
        oquads: np.ndarray,
    ) -> tuple[Any, ...]:
        return (
            lvl,
            int(tile_idx),
            bool(use_px),
            int(id(ohashes)),
            int(id(oquads)),
            int(bucket_limit),
            int(vote_percentile),
        )

    def _attempt_level(
        level_name: str,
        hashes: np.ndarray,
        preferred_tile: str | None,
        parity_label: str,
        phase_name: str,
        *,
        use_px_spec: bool = True,
        use_ra_filter: bool = True,
        allowed_tiles: set[int] | None = None,
        agg_levels: Iterable[str] | None = None,
        early_exit_ratio: float | None = None,
    ) -> WcsSolution | None:
        nonlocal fail_best_inliers_seen, fail_validation_count, total_candidates_tried
        phase_slot = _phase_slot(phase_name)
        phase_slot["level_calls"] += 1
        if cancel_check and cancel_check():
            return None
        if _maybe_fail_early_abort(phase_name):
            phase_slot["early_abort"] = True
            return None
        active_allowed_tiles = allowed_tiles if use_ra_filter else None
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
        cand_key = (
            level_name,
            bool(use_px_spec),
            _allowed_tiles_key(active_allowed_tiles),
            int(level_hashes.size),
        )
        cached_candidates = candidate_search_cache.get(cand_key)
        if cached_candidates is None:
            candidates = tally_candidates(
                (level_hashes, level_counts),
                index_root,
                levels=[level_name],
                allowed_tiles=active_allowed_tiles,
            )
            candidate_search_cache[cand_key] = list(candidates)
        else:
            candidates = list(cached_candidates)
        if not candidates:
            logger.debug("level %s produced no candidates (parity=%s)", level_name, parity_label)
            return None
        phase_slot["candidates_seen"] += int(len(candidates))
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
        if use_ra_filter and active_allowed_tiles is None and ra_hint is not None and dec_hint is not None:
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
        global_mode = not bool(use_ra_filter)
        candidate_limit = max(1, int(config.max_candidates))
        low_support_streak = 0
        weak_validation_streak = 0
        medium_validation_streak = 0
        best_support = 0
        level_started = time.time()
        global_budget_s = 0.0
        if global_mode:
            fast_global = bool(getattr(config, "fast_mode", True))
            # Keep rescue breadth, tighten only fast/global probing.
            if fast_global:
                candidate_limit = min(candidate_limit, 18)
            top_score = max(1, int(ordered[0][1])) if ordered else 1
            score_floor = max(2, int(top_score * (0.06 if fast_global else 0.04)))
            ordered_pruned = [(k, sc) for (k, sc) in ordered if int(sc) >= score_floor]
            if len(ordered_pruned) >= max(8, candidate_limit):
                ordered = ordered_pruned
            # Additional gentle tail pruning when there is a clear front-runner.
            # Keep this conservative to avoid dropping valid rescue candidates.
            if len(ordered) >= 2:
                runner_up = max(1, int(ordered[1][1]))
                if top_score >= int(1.25 * runner_up):
                    rel_floor = 0.10 if fast_global else 0.08
                    score_floor2 = max(2, int(top_score * rel_floor))
                    ordered_pruned2 = [(k, sc) for (k, sc) in ordered if int(sc) >= score_floor2]
                    if len(ordered_pruned2) >= max(6, min(candidate_limit, 12)):
                        ordered = ordered_pruned2
                dominance = float(top_score) / float(max(1, runner_up))
                if fast_global:
                    if dominance >= 3.0:
                        candidate_limit = min(candidate_limit, 10)
                    elif dominance >= 2.0:
                        candidate_limit = min(candidate_limit, 12)
                    elif dominance >= 1.4:
                        candidate_limit = min(candidate_limit, 14)
            global_budget_s = float(
                getattr(config, "global_budget_fast_s", 8.0)
                if fast_global
                else getattr(config, "global_budget_slow_s", 18.0)
            )
            if global_budget_s > 0.0:
                if level_name == "M":
                    global_budget_s *= 1.35
                elif level_name == "L":
                    global_budget_s *= 1.80

        def _global_abort_now() -> bool:
            if not global_mode:
                return False
            if best_support >= 4:
                return False
            if global_budget_s > 0.0 and (time.time() - level_started) >= global_budget_s:
                phase_slot["early_abort"] = True
                phase_slot["early_abort_reason"] = f"time_budget>{global_budget_s:.1f}s"
                return True
            streak_limit = 12 if bool(getattr(config, "fast_mode", True)) else 28
            if low_support_streak >= streak_limit:
                phase_slot["early_abort"] = True
                phase_slot["early_abort_reason"] = f"low_support_streak>={streak_limit}"
                return True
            return False

        level_index: QuadIndex | None = None
        level_slices: list[slice] | None = None
        lookup_key = (level_name, bool(use_px_spec))
        cached_lookup = level_lookup_cache.get(lookup_key)
        if cached_lookup is not None:
            level_index, level_slices = cached_lookup
        elif level_hashes.size:
            try:
                level_index = QuadIndex.load(index_root, level_name)
                level_slices = lookup_hashes(index_root, level_name, level_hashes)
            except FileNotFoundError:
                level_index = None
                level_slices = None
            level_lookup_cache[lookup_key] = (level_index, level_slices)
        top_candidate_score = max(1, int(ordered[0][1])) if ordered else 1

        logger.info(
            "level %s (parity=%s) candidate search returned %d candidate(s)",
            level_name,
            parity_label,
            len(candidates),
        )
        phase_slot["candidates_considered"] += int(min(len(ordered), candidate_limit))
        for candidate_key, score in ordered[: candidate_limit]:
            if cancel_check and cancel_check():
                return None
            if _maybe_fail_early_abort(phase_name):
                phase_slot["early_abort"] = True
                break
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
            if active_allowed_tiles is not None and tile_index not in active_allowed_tiles:
                continue
            total_candidates_tried += 1
            phase_slot["candidates_tried"] += 1
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
                collect_metrics["calls"] += 1
                ckey = _collect_cache_key(lvl, tile_index, use_px_spec, ohashes, oquads)
                cached_triplet = collect_matches_cache.get(ckey)
                if cached_triplet is None:
                    collect_metrics["cache_misses"] += 1
                    _t0 = time.time()
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
                        use_vectorized=bool(getattr(config, "collect_matches_vectorized_experimental", False)),
                    )
                    collect_metrics["compute_s"] += float(time.time() - _t0)
                    collect_matches_cache[ckey] = (ip, tp, wp)
                else:
                    collect_metrics["cache_hits"] += 1
                    ip, tp, wp = cached_triplet
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
                if global_mode:
                    low_support_streak += 1
                    if _global_abort_now():
                        logger.debug("global early-stop: weak candidate support (tile=%s, level=%s, parity=%s)", candidate_key, level_name, parity_label)
                        break
                continue
            # Try quad-based hypotheses first (local, stable scale) then fall back to RANSAC
            transform_result: tuple[SimilarityTransform, SimilarityStats] | None = None
            if level_index is not None and level_slices is not None:
                best = None
                best_inliers = -1
                tested = 0
                # Cap buckets per candidate tile to keep runtime reasonable.
                # Give more budget to top-ranked candidates and less to tail candidates.
                fast_mode = bool(getattr(config, "fast_mode", True))
                rel_score = float(max(1, int(score))) / float(top_candidate_score)
                max_buckets_f = 420.0 if fast_mode else 650.0
                if level_name == "M":
                    max_buckets_f *= 1.20
                elif level_name == "L":
                    max_buckets_f *= 1.45
                if rel_score >= 0.80:
                    max_buckets_f *= 1.40
                elif rel_score <= 0.30:
                    max_buckets_f *= 0.70
                max_buckets = int(max(160, min(1200, round(max_buckets_f))))
                src_all_c = (img_points[:, 0] + 1j * img_points[:, 1]).astype(np.complex128)
                dst_all_c = (tile_points[:, 0] + 1j * tile_points[:, 1]).astype(np.complex128)
                for idx2, slc in enumerate(level_slices):
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
                        if int(level_index.tile_indices[b]) != tile_index:
                            continue
                        tested += 1
                        if tested >= max_buckets:
                            break
                        tile_combo = level_index.quad_indices[b]
                        if np.any(obs_combo < 0) or np.any(obs_combo >= image_positions.shape[0]):
                            logger.debug("skipping quad with invalid image indices (level=%s)", level_name)
                            continue
                        if np.any(tile_combo < 0) or np.any(tile_combo >= tile_positions.shape[0]):
                            logger.debug(
                                "skipping quad with invalid tile indices (tile=%s, level=%s)",
                                tile_index,
                                level_name,
                            )
                            continue
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
                ransac_seed = (
                    ransac_seed_base
                    ^ ((int(tile_index) + 1) * 0x9E3779B1)
                    ^ ((ord(level_name[0]) if level_name else 0) << 24)
                    ^ (1 if parity_label == "mirror" else 0)
                ) & 0xFFFFFFFF
                transform_result = estimate_similarity_RANSAC(
                    img_points,
                    tile_points,
                    trials=1200,
                    tol_px=config.pixel_tolerance,
                    min_inliers=4,
                    allow_reflection=bool(config.try_parity_flip),
                    early_stop_inliers=int(getattr(config, "quality_inliers", 60) or 60),
                    random_state=ransac_seed,
                )
            if transform_result is None:
                if global_mode:
                    low_support_streak += 1
                    if _global_abort_now():
                        logger.debug("global early-stop: no transform after repeated attempts (level=%s, parity=%s)", level_name, parity_label)
                        break
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
                if global_mode:
                    low_support_streak += 1
                    if _global_abort_now():
                        logger.debug("global early-stop: empty inlier masks (level=%s, parity=%s)", level_name, parity_label)
                        break
                continue
            inlier_count = int(np.count_nonzero(inliers_mask))
            if inlier_count < 4:
                if global_mode:
                    low_support_streak += 1
                    if _global_abort_now():
                        logger.debug("global early-stop: too few preliminary inliers (%d) (level=%s, parity=%s)", inlier_count, level_name, parity_label)
                        break
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
                max(4, int(0.4 * max(0, n_pairs))),
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
            if stats.get("quality") == "GOOD":
                geo_ok, geo = _blind_geometric_guardrails(img_in, image.shape)
                stats["geo_cov_x"] = float(geo.get("cov_x", float("nan")))
                stats["geo_cov_y"] = float(geo.get("cov_y", float("nan")))
                stats["geo_cov_area"] = float(geo.get("cov_area", float("nan")))
                stats["geo_cond"] = float(geo.get("cond", float("nan")))
                if not geo_ok:
                    logger.info(
                        "geometric guard failed: tile=%s level=%s parity=%s reason=%s n=%d cov=(%.3f,%.3f) area=%.4f cond=%s",
                        candidate_key,
                        level_name,
                        parity_label,
                        str(geo.get("reason", "unknown")),
                        int(geo.get("n", 0)),
                        float(geo.get("cov_x", 0.0)),
                        float(geo.get("cov_y", 0.0)),
                        float(geo.get("cov_area", 0.0)),
                        f"{float(geo.get('cond', float('nan'))):.1f}" if np.isfinite(float(geo.get('cond', float('nan')))) else "inf",
                    )
                    stats = {"quality": "FAIL", "success": False, "reason": "geometric guard failed", **stats}
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
                inl = int(stats.get("inliers", 0) or 0)
                fail_best_inliers_seen = max(fail_best_inliers_seen, inl)
                fail_validation_count += 1
                phase_slot["failed_validations"] += 1
                _update_verify_logodds(
                    phase_slot=phase_slot,
                    inliers=inl,
                    rms_px=float(stats.get("rms_px", float("inf"))),
                )
                if global_mode:
                    inl = int(stats.get("inliers", 0) or 0)
                    best_support = max(best_support, inl)
                    low_support_streak = (low_support_streak + 1) if inl <= 2 else 0
                    weak_validation_streak = (weak_validation_streak + 1) if inl <= 4 else 0
                    medium_validation_streak = (medium_validation_streak + 1) if inl <= 8 else 0
                    score_rel = float(max(1, int(score))) / float(max(1, top_candidate_score))
                    weak_break_streak = 8 if bool(getattr(config, "fast_mode", True)) else 16
                    if weak_validation_streak >= weak_break_streak and score_rel <= 0.35:
                        logger.debug(
                            "global early-stop: prolonged weak validations in tail (streak=%d, rel=%.2f, level=%s, parity=%s)",
                            weak_validation_streak,
                            score_rel,
                            level_name,
                            parity_label,
                        )
                        break
                    medium_break_streak = 14 if bool(getattr(config, "fast_mode", True)) else 28
                    elapsed_level = time.time() - level_started
                    medium_elapsed_gate = max(6.0, (0.75 * global_budget_s) if global_budget_s > 0.0 else 10.0)
                    if medium_validation_streak >= medium_break_streak and score_rel <= 0.45 and elapsed_level >= medium_elapsed_gate:
                        logger.debug(
                            "global early-stop: medium-quality validation stall (streak=%d, rel=%.2f, elapsed=%.1fs, level=%s, parity=%s)",
                            medium_validation_streak,
                            score_rel,
                            elapsed_level,
                            level_name,
                            parity_label,
                        )
                        break
                    if _global_abort_now():
                        logger.debug("global early-stop: repeated low-support validations (level=%s, parity=%s)", level_name, parity_label)
                        break
                continue

            zemo_ok, zemo_reason, pix_scale_arcsec = validate_wcs_for_zemosaic(final_wcs)
            if not zemo_ok:
                logger.info(
                    "zemosaic-compat failed: tile=%s level=%s parity=%s reason=%s",
                    candidate_key,
                    level_name,
                    parity_label,
                    zemo_reason,
                )
                inl = int(stats.get("inliers", 0) or 0)
                fail_best_inliers_seen = max(fail_best_inliers_seen, inl)
                fail_validation_count += 1
                phase_slot["failed_validations"] += 1
                _update_verify_logodds(
                    phase_slot=phase_slot,
                    inliers=inl,
                    rms_px=float(stats.get("rms_px", float("inf"))),
                )
                if global_mode:
                    inl = int(stats.get("inliers", 0) or 0)
                    best_support = max(best_support, inl)
                    low_support_streak = (low_support_streak + 1) if inl <= 2 else 0
                    weak_validation_streak = (weak_validation_streak + 1) if inl <= 4 else 0
                    medium_validation_streak = (medium_validation_streak + 1) if inl <= 8 else 0
                    score_rel = float(max(1, int(score))) / float(max(1, top_candidate_score))
                    weak_break_streak = 8 if bool(getattr(config, "fast_mode", True)) else 16
                    if weak_validation_streak >= weak_break_streak and score_rel <= 0.35:
                        logger.debug(
                            "global early-stop: prolonged weak zemosaic rejects in tail (streak=%d, rel=%.2f, level=%s, parity=%s)",
                            weak_validation_streak,
                            score_rel,
                            level_name,
                            parity_label,
                        )
                        break
                    medium_break_streak = 14 if bool(getattr(config, "fast_mode", True)) else 28
                    elapsed_level = time.time() - level_started
                    medium_elapsed_gate = max(6.0, (0.75 * global_budget_s) if global_budget_s > 0.0 else 10.0)
                    if medium_validation_streak >= medium_break_streak and score_rel <= 0.45 and elapsed_level >= medium_elapsed_gate:
                        logger.debug(
                            "global early-stop: medium-quality zemosaic stall (streak=%d, rel=%.2f, elapsed=%.1fs, level=%s, parity=%s)",
                            medium_validation_streak,
                            score_rel,
                            elapsed_level,
                            level_name,
                            parity_label,
                        )
                        break
                    if _global_abort_now():
                        logger.debug("global early-stop: zemosaic-compat rejects at low support (level=%s, parity=%s)", level_name, parity_label)
                        break
                continue

            header_updates = {
                "SOLVED": 1,
                "DBSET": tile_entry.get("family"),
                "TILE_ID": tile_entry.get("tile_key"),
                "RMSPX": stats["rms_px"],
                "INLIERS": stats["inliers"],
                "PIXSCAL": pix_scale_arcsec,
                "SIPORD": sip_used,
                "QUALITY": stats["quality"],
                "USED_DB": tile_entry.get("family"),
                "SOLVER": "ZeSolver",
                "SOLVMODE": "BLIND",
            }
            phase_slot["solutions"] += 1
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
        phase_name: str,
        *,
        use_px_spec: bool = True,
        use_ra_filter: bool = True,
        allowed_tiles: set[int] | None = None,
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
                phase_name,
                use_px_spec=use_px_spec,
                use_ra_filter=use_ra_filter,
                allowed_tiles=allowed_tiles,
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
        base_locked_count = len(allowed_tile_indices) if allowed_tile_indices is not None else 0
        phase_specs.append(
            {
                "name": "hinted",
                "require_ra": True,
                "require_scale": True,
                "level_sets": _build_level_sets(levels_scale_focus),
                "use_ra_filter": True,
                "allowed_tiles": allowed_tile_indices,
                "early_exit": 4.0,
            }
        )
        if (
            expanded_tile_indices is not None
            and len(expanded_tile_indices) > max(1, base_locked_count)
            and len(expanded_tile_indices) <= 96
        ):
            phase_specs.append(
                {
                    "name": "hinted_wide",
                    "require_ra": True,
                    "require_scale": True,
                    "level_sets": _build_level_sets(levels_fast or levels_scale_focus),
                    "use_ra_filter": True,
                    "allowed_tiles": expanded_tile_indices,
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
                "allowed_tiles": None,
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
            "allowed_tiles": None,
            "early_exit": None,
        }
    )
    for phase in phase_specs:
        if phase["require_ra"] and not ra_available:
            continue
        if phase["require_scale"] and not scale_available:
            continue
        phases_attempted.append(str(phase.get("name", "unknown")))
        phase_start = time.time()
        phase_slot = _phase_slot(str(phase["name"]))
        logger.info(
            "phase %s starting (ra_filter=%s, level_sets=%d)",
            phase["name"],
            phase["use_ra_filter"],
            len(phase["level_sets"]),
        )
        phase_allowed = phase.get("allowed_tiles")
        if phase["use_ra_filter"]:
            if phase_allowed is not None:
                logger.info(
                    "phase %s using RA/Dec tile lock (%d/%d tiles)",
                    phase["name"],
                    len(phase_allowed),
                    tile_count,
                )
            elif allowed_tile_indices is not None:
                logger.info(
                    "phase %s using RA/Dec tile lock (%d/%d tiles)",
                    phase["name"],
                    len(allowed_tile_indices),
                    tile_count,
                )
            else:
                logger.info(
                    "phase %s using dynamic RA/Dec cone filtering",
                    phase["name"],
                )
        elif allowed_tile_indices is not None:
            logger.info(
                "phase %s disables RA/Dec tile lock (global candidate pool)",
                phase["name"],
            )
        for level_seq in phase["level_sets"]:
            for variant_hashes, parity_label in variants:
                if _maybe_fail_early_abort(str(phase["name"])):
                    phase_slot["early_abort"] = True
                    break
                solution = _run_levels(
                    variant_hashes,
                    parity_label,
                    phase_name=str(phase["name"]),
                    use_px_spec=True,
                    use_ra_filter=phase["use_ra_filter"],
                    allowed_tiles=phase.get("allowed_tiles"),
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
            if fail_early_abort:
                break
            if best_solution:
                break
        phase_slot["elapsed_s"] = float(time.time() - phase_start)
        if fail_early_abort:
            logger.info(
                "phase %s ended by fail-early abort after %.2fs",
                phase["name"],
                phase_slot["elapsed_s"],
            )
            break
        if best_solution:
            break
    _log_phase("candidate search", stage)
    if not best_solution:
        attempt_elapsed_s = float(time.time() - attempt_started)
        fail_stats = {
            "best_fail_inliers": int(fail_best_inliers_seen),
            "fail_validation_count": int(fail_validation_count),
            "phases_attempted": list(phases_attempted),
            "has_hints": bool(ra_available and scale_available),
            "attempt_elapsed_s": attempt_elapsed_s,
            "fail_early_abort": bool(fail_early_abort),
            "fail_early_abort_phase": fail_early_abort_phase,
            "fail_early_abort_reason": fail_early_abort_reason,
            "fail_budget_s": float(fail_budget_s),
            "verify_logodds_enabled": bool(verify_logodds_enabled),
            "verify_logodds_bail": float(verify_logodds_bail),
            "verify_logodds_stoplooking": float(verify_logodds_stoplooking),
            "verify_logodds_min_validations": int(verify_logodds_min_validations),
            "verify_logodds_cum": float(verify_logodds_cum),
            "verify_logodds_last": float(verify_logodds_last),
            "hard_max_candidates_tried": int(hard_max_candidates_tried),
            "hard_max_validations": int(hard_max_validations),
            "total_candidates_tried": int(total_candidates_tried),
            "phase_perf": {k: dict(v) for k, v in phase_perf.items()},
            "collect_metrics": dict(collect_metrics),
        }
        if _depth_ladder_internal:
            fail_stats["depth_ladder_internal"] = True
            fail_stats["depth_ladder_stage"] = int(_depth_ladder_stage)
            fail_stats["depth_ladder_total"] = int(_depth_ladder_total)
            if _depth_ladder_caps is not None:
                fail_stats["depth_ladder_caps"] = [int(v) for v in _depth_ladder_caps]
        return _finish(WcsSolution(False, "no valid solution", None, fail_stats, None, {}))
    best_solution.stats["attempt_elapsed_s"] = float(time.time() - attempt_started)
    best_solution.stats["total_candidates_tried"] = int(total_candidates_tried)
    best_solution.stats["verify_logodds_enabled"] = bool(verify_logodds_enabled)
    best_solution.stats["verify_logodds_bail"] = float(verify_logodds_bail)
    best_solution.stats["verify_logodds_stoplooking"] = float(verify_logodds_stoplooking)
    best_solution.stats["verify_logodds_min_validations"] = int(verify_logodds_min_validations)
    best_solution.stats["verify_logodds_cum"] = float(verify_logodds_cum)
    best_solution.stats["verify_logodds_last"] = float(verify_logodds_last)
    best_solution.stats["hard_max_candidates_tried"] = int(hard_max_candidates_tried)
    best_solution.stats["hard_max_validations"] = int(hard_max_validations)
    best_solution.stats["phase_perf"] = {k: dict(v) for k, v in phase_perf.items()}
    best_solution.stats["collect_metrics"] = dict(collect_metrics)
    if _depth_ladder_internal:
        best_solution.stats["depth_ladder_internal"] = True
        best_solution.stats["depth_ladder_stage"] = int(_depth_ladder_stage)
        best_solution.stats["depth_ladder_total"] = int(_depth_ladder_total)
        if _depth_ladder_caps is not None:
            best_solution.stats["depth_ladder_caps"] = [int(v) for v in _depth_ladder_caps]
    header_updates = {
        **best_solution.header_updates,
        "BLINDVER": ZEBLIND_VERSION,
    }
    if cancel_check and cancel_check():
        return _finish(WcsSolution(False, "cancelled", None, {}, None, {}))
    if is_fits:
        with fits.open(source_path, mode="update", memmap=False) as hdul:
            header_out = hdul[0].header
            apply_wcs_solution_to_header(
                header_out,
                best_solution.wcs,
                header_updates=header_updates,
                remove_sip_before_write=True,
            )
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
    parser.add_argument("--fail-attempt-budget-s", type=float, default=70.0, help="Fail-fast budget (s) for weak blind attempts; <=0 disables")
    parser.add_argument("--fail-attempt-min-validations", type=int, default=18, help="Minimum failed validations before fail-fast abort")
    parser.add_argument("--fail-attempt-max-best-inliers", type=int, default=4, help="Maximum best inliers considered weak for fail-fast abort")
    parser.add_argument("--fail-attempt-min-candidates", type=int, default=20, help="Minimum tried candidates before fail-fast abort")
    parser.add_argument("--verify-logodds-enabled", type=int, choices=(0, 1), default=0, help="Enable verification log-odds early bail/stoplooking policy")
    parser.add_argument("--verify-logodds-bail", type=float, default=-24.0, help="Cumulative log-odds threshold to bail out weak attempts")
    parser.add_argument("--verify-logodds-stoplooking", type=float, default=24.0, help="Cumulative log-odds threshold that suppresses weak-attempt abort")
    parser.add_argument("--verify-logodds-min-validations", type=int, default=8, help="Minimum failed validations before applying log-odds policy")
    parser.add_argument("--hard-max-candidates-tried", type=int, default=0, help="Hard cap on total candidates tried (0 disables)")
    parser.add_argument("--hard-max-validations", type=int, default=0, help="Hard cap on failed validations (0 disables)")
    parser.add_argument("--depth-ladder-enabled", type=int, choices=(0, 1), default=0, help="Enable depth ladder (progressive max_stars)")
    parser.add_argument("--depth-ladder-caps", default="80,160,500", help="Comma-separated max_stars caps for depth ladder")
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
    depth_ladder_caps = tuple(_coerce_depth_ladder_caps(args.depth_ladder_caps))
    if not depth_ladder_caps:
        depth_ladder_caps = (max(1, int(args.max_stars or 1)),)
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
        fail_attempt_budget_s=max(0.0, float(args.fail_attempt_budget_s or 0.0)),
        fail_attempt_min_validations=max(0, int(args.fail_attempt_min_validations or 0)),
        fail_attempt_max_best_inliers=max(0, int(args.fail_attempt_max_best_inliers or 0)),
        fail_attempt_min_candidates=max(0, int(args.fail_attempt_min_candidates or 0)),
        verify_logodds_enabled=bool(int(args.verify_logodds_enabled)),
        verify_logodds_bail=float(args.verify_logodds_bail),
        verify_logodds_stoplooking=float(args.verify_logodds_stoplooking),
        verify_logodds_min_validations=max(0, int(args.verify_logodds_min_validations or 0)),
        hard_max_candidates_tried=max(0, int(args.hard_max_candidates_tried or 0)),
        hard_max_validations=max(0, int(args.hard_max_validations or 0)),
        depth_ladder_enabled=bool(int(args.depth_ladder_enabled)),
        depth_ladder_caps=tuple(int(v) for v in depth_ladder_caps),
    )
    solution = solve_blind(args.input, args.index_root, config=config)
    if solution.success:
        logger.info("blind solve succeeded for %s", args.input)
        return 0
    logger.error("blind solve failed: %s", solution.message)
    return 2


zeblindsolve = main
