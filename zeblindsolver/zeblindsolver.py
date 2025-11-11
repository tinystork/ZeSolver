from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Callable, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .asterisms import hash_quads, sample_quads
from .candidate_search import tally_candidates
from .image_prep import build_pyramid, read_fits_as_luma, remove_background
from .fits_utils import parse_angle
from .matcher import SimilarityStats, SimilarityTransform, estimate_similarity_RANSAC
from .matcher import _derive_similarity  # quad-based hypothesis helper
from .quad_index_builder import QuadIndex, load_manifest, lookup_hashes
from .levels import LEVEL_MAP, QuadLevelSpec
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


@dataclass
class SolveConfig:
    max_candidates: int = 12
    max_stars: int = 800
    max_quads: int = 12000
    sip_order: int = 2
    quality_rms: float = 1.0
    quality_inliers: int = 60
    pixel_tolerance: float = 3.0
    log_level: str = "INFO"
    verbose: bool = False
    try_parity_flip: bool = True
    fast_mode: bool = False


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


def _load_tile_positions(index_root: Path, entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # Normalize path separators to support manifests written on Windows
    rel = str(entry["tile_file"]).replace("\\", "/")
    tile_path = index_root / rel
    with np.load(tile_path) as data:
        xy = np.column_stack((data["x_deg"], data["y_deg"]))
        world = np.column_stack((data["ra_deg"], data["dec_deg"]))
    return xy.astype(np.float32), world.astype(np.float32)


def _collect_tile_matches(
    index_root: Path,
    levels: Iterable[str],
    tile_index: int,
    observed_hashes: np.ndarray,
    observed_quads: np.ndarray,
    image_positions: np.ndarray,
    tile_positions: np.ndarray,
    tile_world: np.ndarray,
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
            per_bucket_cap = 4096
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
    thr = max(1, int(np.percentile(counts, 40)))
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
    index_root = Path(index_root).expanduser().resolve()
    if cancel_check and cancel_check():
        return WcsSolution(False, "cancelled", None, {}, None, {})
    manifest = load_manifest(index_root)
    levels = [level["name"] for level in manifest.get("levels", [])] or ["L", "M", "S"]
    # Prefer trying smaller-diameter levels first for selectivity
    pref = ("S", "M", "L")
    levels = [lvl for lvl in pref if lvl in levels] + [lvl for lvl in levels if lvl not in pref]
    # Preflight: ensure there is at least one quad hash table present.
    ht_root = index_root / "hash_tables"
    present_levels = [lvl for lvl in levels if (ht_root / f"quads_{lvl}.npz").exists()]
    missing_levels = [lvl for lvl in levels if lvl not in present_levels]
    if not present_levels:
        manifest_path = index_root / "manifest.json"
        details = [
            f"root={index_root}",
            f"manifest={'ok' if manifest_path.exists() else 'missing'}",
        ]
        for lvl in levels:
            details.append(f"{lvl}={'ok' if (ht_root / f'quads_{lvl}.npz').exists() else 'missing'}")
        msg = (
            "index has no quad hash tables (levels: "
            + ", ".join(levels)
            + ") — details: "
            + ", ".join(details)
            + ". Build the index with zebuildindex (see firstrun.txt)"
        )
        logger.error(msg)
        return WcsSolution(False, msg, None, {}, None, {})
    if missing_levels:
        logger.warning(
            "some quad hash tables are missing: %s (continuing with %s)",
            ", ".join(missing_levels),
            ", ".join(present_levels),
        )
    if cancel_check and cancel_check():
        return WcsSolution(False, "cancelled", None, {}, None, {})
    tile_entries = manifest.get("tiles", [])
    tile_map = {entry["tile_key"]: idx for idx, entry in enumerate(tile_entries)}
    logging.getLogger().setLevel(config.log_level)
    # Optional: use RA/DEC metadata (if present) to prioritize nearby tiles
    preferred_tile_from_header: str | None = None
    ra_hint = None
    dec_hint = None
    try:
        hdr = fits.getheader(input_fits)
        ra0 = parse_angle(hdr.get("RA") or hdr.get("OBJCTRA") or hdr.get("OBJRA") or hdr.get("OBJ_RA") or hdr.get("CRVAL1"), is_ra=True)
        dec0 = parse_angle(hdr.get("DEC") or hdr.get("OBJCTDEC") or hdr.get("OBJDEC") or hdr.get("OBJ_DEC") or hdr.get("CRVAL2"), is_ra=False)
        if ra0 is not None and dec0 is not None and tile_entries:
            ra_hint, dec_hint = float(ra0), float(dec0)
            # pick manifest tile whose center is closest to the metadata position
            import math
            def ang_sep(ra1, dec1, ra2, dec2):
                ra1, ra2 = math.radians(ra1), math.radians(ra2)
                d1, d2 = math.radians(dec1), math.radians(dec2)
                return math.degrees(math.acos(max(-1.0, min(1.0, math.sin(d1)*math.sin(d2)+math.cos(d1)*math.cos(d2)*math.cos(ra1-ra2)))))
            best_key = None
            best_dist = 1e9
            for entry in tile_entries:
                tra = float(entry.get("center_ra_deg", 0.0))
                tdec = float(entry.get("center_dec_deg", 0.0))
                d = ang_sep(ra0, dec0, tra, tdec)
                if d < best_dist:
                    best_dist = d
                    best_key = str(entry.get("tile_key"))
            preferred_tile_from_header = best_key
    except Exception:
        preferred_tile_from_header = None
    stage = time.time()
    if cancel_check and cancel_check():
        return WcsSolution(False, "cancelled", None, {}, None, {})
    image = read_fits_as_luma(input_fits)
    height, width = image.shape
    # Estimate pixel scale (deg/px) if header contains optical info
    try:
        with fits.open(input_fits, mode="readonly", memmap=False) as _hdul_tmp:
            _hdr2 = _hdul_tmp[0].header
    except Exception:
        _hdr2 = None
    approx_scale_deg = None
    if _hdr2 is not None:
        from .fits_utils import estimate_scale_and_fov as _est_scale
        scale_arcsec, _ = _est_scale(_hdr2, width, height)
        if scale_arcsec:
            approx_scale_deg = float(scale_arcsec) / 3600.0
    _log_phase("read image", stage)
    stage = time.time()
    # Use a smaller median kernel for speed on typical Seestar frames
    image = remove_background(image, kernel_size=15)
    pyramid = build_pyramid(image)
    _log_phase("preprocess", stage)
    detection = pyramid[-1]
    stage = time.time()
    stars = detect_stars(detection)
    if stars.size == 0:
        return WcsSolution(False, "no stars found", None, {}, None, {})
    if config.max_stars and stars.size > config.max_stars:
        stars = stars[: config.max_stars]
    logger.info("detected %d stars (using top %d)", stars.shape[0], config.max_stars or stars.shape[0])
    image_positions = _image_positions(stars)
    obs_stars = np.zeros(stars.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    obs_stars["x"] = stars["x"]
    obs_stars["y"] = stars["y"]
    obs_stars["mag"] = -stars["flux"]
    # Prefer local-neighborhood quads to improve geometric stability on TAN plane
    if cancel_check and cancel_check():
        return WcsSolution(False, "cancelled", None, {}, None, {})
    quads = sample_quads(obs_stars, config.max_quads, strategy="local_brightness")
    if quads.size == 0:
        return WcsSolution(False, "no quads sampled", None, {}, None, {})
    if cancel_check and cancel_check():
        return WcsSolution(False, "cancelled", None, {}, None, {})
    obs_hash = hash_quads(quads, image_positions)
    logger.info("sampled %d quads producing %d hashes", quads.shape[0], obs_hash.hashes.size)
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
    obs_by_level: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lvl in levels:
        if cancel_check and cancel_check():
            return WcsSolution(False, "cancelled", None, {}, None, {})
        px = _spec_pixels(lvl)
        if px is None:
            # fallback to unfiltered
            obs_by_level[lvl] = (obs_hash.hashes, obs_hash.indices)
        else:
            oh = hash_quads(quads, image_positions, spec=px)
            obs_by_level[lvl] = (oh.hashes, oh.indices)

    def _attempt_level(
        level_name: str,
        hashes: np.ndarray,
        preferred_tile: str | None,
        parity_label: str,
        *,
        use_px_spec: bool = True,
        use_ra_filter: bool = True,
        agg_levels: Iterable[str] | None = None,
    ) -> WcsSolution | None:
        if cancel_check and cancel_check():
            return None
        # Use precomputed observed quads for this level (optionally bypassing px filters)
        if use_px_spec:
            level_hashes, level_quads = obs_by_level.get(level_name, (obs_hash.hashes, obs_hash.indices))
        else:
            level_hashes, level_quads = (obs_hash.hashes, obs_hash.indices)
        if cancel_check and cancel_check():
            return None
        candidates = tally_candidates(level_hashes, index_root, levels=[level_name])
        if not candidates:
            logger.debug("level %s produced no candidates (parity=%s)", level_name, parity_label)
            return None
        ordered = _prioritize_candidates(candidates, preferred_tile)
        # If we have RA/DEC hint, ignore far tiles to speed up and reduce false tries
        if use_ra_filter and ra_hint is not None and dec_hint is not None:
            import math
            def ang_sep(ra1, dec1, ra2, dec2):
                ra1, ra2 = math.radians(ra1), math.radians(ra2)
                d1, d2 = math.radians(dec1), math.radians(dec2)
                return math.degrees(math.acos(max(-1.0, min(1.0, math.sin(d1)*math.sin(d2)+math.cos(d1)*math.cos(d2)*math.cos(ra1-ra2)))))
            # derive a radius from approximate FOV if available; otherwise 5°
            approx_fov = None
            if approx_scale_deg is not None:
                approx_fov = float(approx_scale_deg) * max(image.shape)
            radius_limit = min(8.0, max(5.0, (approx_fov or 1.5) * 3.0))
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
                if ang_sep(ra_hint, dec_hint, tra, tdec) <= radius_limit:
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
                ohashes, oquads = (obs_by_level[lvl] if use_px_spec else (obs_hash.hashes, obs_hash.indices))
                ip, tp, wp = _collect_tile_matches(
                    index_root,
                    (lvl,),
                    tile_index,
                    ohashes,
                    oquads,
                    image_positions,
                    tile_positions,
                    tile_world,
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
    ) -> WcsSolution | None:
        best: WcsSolution | None = None
        preferred_tile: str | None = preferred_tile_from_header

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
            )
            # Do not bail out early if the first level fails; try remaining levels (e.g., M/S)
            if solution is None:
                continue
            if _is_better(solution, best):
                best = solution
            preferred_tile = solution.tile_key
        return best

    stage = time.time()
    variants = [(obs_hash.hashes, "nominal")]
    if config.try_parity_flip:
        variants.append((obs_hash.hashes ^ 1, "mirror"))
    best_solution: WcsSolution | None = None
    levels_all = list(levels)
    levels_fast = [lvl for lvl in ("S",) if lvl in levels_all]
    for variant_hashes, parity_label in variants:
        logger.info("starting candidate search (parity=%s)", parity_label)
        # Fast mode: try S only first (guided then relaxed)
        if config.fast_mode and levels_fast:
            best_solution = _run_levels(
                variant_hashes,
                parity_label,
                use_px_spec=True,
                use_ra_filter=True,
                levels_seq=levels_fast,
            )
            if not best_solution:
                logger.info("no solution in guided mode; retrying relaxed search (parity=%s)", parity_label)
                best_solution = _run_levels(
                    variant_hashes,
                    parity_label,
                    use_px_spec=False,
                    use_ra_filter=False,
                    levels_seq=levels_fast,
                )
            if best_solution:
                best_solution.message += f" (parity={parity_label})"
                break
        # Full run
        best_solution = _run_levels(
            variant_hashes,
            parity_label,
            use_px_spec=True,
            use_ra_filter=True,
            levels_seq=levels_all,
        )
        if not best_solution:
            logger.info("no solution in guided mode; retrying relaxed search (parity=%s)", parity_label)
            best_solution = _run_levels(
                variant_hashes,
                parity_label,
                use_px_spec=False,
                use_ra_filter=False,
                levels_seq=levels_all,
            )
        if best_solution:
            best_solution.message += f" (parity={parity_label})"
            break
    _log_phase("candidate search", stage)
    if not best_solution:
        return WcsSolution(False, "no valid solution", None, {}, None, {})
    header_updates = {
        **best_solution.header_updates,
        "BLINDVER": ZEBLIND_VERSION,
    }
    if cancel_check and cancel_check():
        return WcsSolution(False, "cancelled", None, {}, None, {})
    with fits.open(input_fits, mode="update", memmap=False) as hdul:
        header = hdul[0].header
        for key, value in best_solution.wcs.to_header(relax=True).items():
            header[key] = value
        for key, value in header_updates.items():
            header[key] = value
        header["ZBLNDVER"] = ZEBLIND_VERSION
        hdul.flush()
    logger.info(
        "blind solve succeeded (tile=%s, rms=%.3f px, inliers=%d)",
        best_solution.tile_key,
        best_solution.stats.get("rms_px", float("nan")),
        best_solution.stats.get("inliers", 0),
    )
    return WcsSolution(
        True,
        best_solution.message,
        best_solution.wcs,
        best_solution.stats,
        best_solution.tile_key,
        header_updates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZeSolver blind solve pipeline")
    parser.add_argument("input", help="Path to the FITS file to solve")
    parser.add_argument("--index-root", required=True, help="Directory containing the blind index")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-stars", type=int, default=800)
    parser.add_argument("--max-quads", type=int, default=12000)
    parser.add_argument("--sip-order", type=int, choices=(2, 3), default=2)
    parser.add_argument("--quality-rms", type=float, default=1.0)
    parser.add_argument("--quality-inliers", type=int, default=60)
    parser.add_argument("--pixel-tolerance", type=float, default=3.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    config = SolveConfig(
        max_candidates=args.max_candidates,
        max_stars=args.max_stars,
        max_quads=args.max_quads,
        sip_order=args.sip_order,
        quality_rms=args.quality_rms,
        quality_inliers=args.quality_inliers,
        pixel_tolerance=args.pixel_tolerance,
        log_level=args.log_level.upper(),
    )
    solution = solve_blind(args.input, args.index_root, config=config)
    if solution.success:
        logger.info("blind solve succeeded for %s", args.input)
        return 0
    logger.error("blind solve failed: %s", solution.message)
    return 2


zeblindsolve = main
