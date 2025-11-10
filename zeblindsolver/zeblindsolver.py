from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .asterisms import hash_quads, sample_quads
from .candidate_search import tally_candidates
from .image_prep import build_pyramid, read_fits_as_luma, remove_background
from .matcher import SimilarityStats, SimilarityTransform, estimate_similarity_RANSAC
from .matcher import _derive_similarity  # quad-based hypothesis helper
from .quad_index_builder import QuadIndex, load_manifest, lookup_hashes
from .levels import LEVEL_MAP
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
            # Skip excessively large buckets (too ambiguous)
            if getattr(index, "bucket_cap", 0) and (slc.stop - slc.start) > index.bucket_cap:
                continue
            for bucket in range(slc.start, slc.stop):
                if int(index.tile_indices[bucket]) != tile_index:
                    continue
                tile_combo = index.quad_indices[bucket]
                for obs_star, tile_star in zip(obs_combo, tile_combo):
                    votes[(int(obs_star), int(tile_star))] += 1
    if not votes:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, empty
    # Determine a minimal vote threshold adaptively; at least 2.
    counts = np.array(list(votes.values()), dtype=int)
    # use percentile to drop the long tail of singletons
    thr = max(2, int(np.percentile(counts, 60)))
    pairs = [(i, t, c) for (i, t), c in votes.items() if c >= thr]
    if not pairs:
        # fallback: keep top-N by votes
        top = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[: max(50, int(len(votes) * 0.05))]
        pairs = [(i, t, c) for (i, t), c in top]
    pairs.sort(key=lambda itc: itc[2], reverse=True)
    img_pts = np.array([image_positions[i] for i, _, _ in pairs], dtype=np.float32)
    tile_pts = np.array([tile_positions[t] for _, t, _ in pairs], dtype=np.float32)
    world_pts = np.array([tile_world[t] for _, t, _ in pairs], dtype=np.float32)
    return img_pts, tile_pts, world_pts


def _build_matches_array(image_pts: np.ndarray, sky_pts: np.ndarray) -> np.ndarray:
    if image_pts.size == 0 or sky_pts.size == 0:
        return np.empty((0, 4), dtype=float)
    return np.column_stack((image_pts, sky_pts))


def _log_phase(stage: str, start: float) -> None:
    logger.debug("%s completed in %.2fs", stage, time.time() - start)


def solve_blind(input_fits: Path | str, index_root: Path | str, *, config: SolveConfig | None = None) -> WcsSolution:
    config = config or SolveConfig()
    index_root = Path(index_root).expanduser().resolve()
    manifest = load_manifest(index_root)
    levels = [level["name"] for level in manifest.get("levels", [])] or ["L", "M", "S"]
    # Preflight: ensure there is at least one quad hash table present.
    present_levels = [lvl for lvl in levels if (index_root / "hash_tables" / f"quads_{lvl}.npz").exists()]
    missing_levels = [lvl for lvl in levels if lvl not in present_levels]
    if not present_levels:
        msg = (
            "index has no quad hash tables (levels: "
            + ", ".join(levels)
            + "); build the index with zebuildindex (see firstrun.txt)"
        )
        logger.error(msg)
        return WcsSolution(False, msg, None, {}, None, {})
    if missing_levels:
        logger.warning(
            "some quad hash tables are missing: %s (continuing with %s)",
            ", ".join(missing_levels),
            ", ".join(present_levels),
        )
    tile_entries = manifest.get("tiles", [])
    tile_map = {entry["tile_key"]: idx for idx, entry in enumerate(tile_entries)}
    logging.getLogger().setLevel(config.log_level)
    stage = time.time()
    image = read_fits_as_luma(input_fits)
    _log_phase("read image", stage)
    stage = time.time()
    image = remove_background(image)
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
    quads = sample_quads(obs_stars, config.max_quads, strategy="local_brightness")
    if quads.size == 0:
        return WcsSolution(False, "no quads sampled", None, {}, None, {})
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

    def _attempt_level(
        level_name: str,
        hashes: np.ndarray,
        preferred_tile: str | None,
        parity_label: str,
    ) -> WcsSolution | None:
        # Re-hash observed quads with level-specific spec to reduce collisions
        level_spec = LEVEL_MAP.get(level_name)
        if level_spec is not None:
            obs_level = hash_quads(quads, image_positions, spec=level_spec)
            level_hashes = obs_level.hashes
            level_quads = obs_level.indices
        else:
            level_hashes = hashes
            level_quads = obs_hash.indices
        candidates = tally_candidates(level_hashes, index_root, levels=[level_name])
        if not candidates:
            logger.debug("level %s produced no candidates (parity=%s)", level_name, parity_label)
            return None
        ordered = _prioritize_candidates(candidates, preferred_tile)
        logger.info(
            "level %s (parity=%s) candidate search returned %d candidate(s)",
            level_name,
            parity_label,
            len(candidates),
        )
        for candidate_key, score in ordered[: config.max_candidates]:
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
            img_points, tile_points, tile_world_matches = _collect_tile_matches(
                index_root,
                (level_name,),
                tile_index,
                level_hashes,
                level_quads,
                image_positions,
                tile_positions,
                tile_world,
            )
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
                max_buckets = 2000
                src_all_c = (img_points[:, 0] + 1j * img_points[:, 1]).astype(np.complex128)
                dst_all_c = (tile_points[:, 0] + 1j * tile_points[:, 1]).astype(np.complex128)
                for idx2, slc in enumerate(slices):
                    if slc.start == slc.stop:
                        continue
                    if tested >= max_buckets:
                        break
                    obs_combo = level_quads[idx2]
                    for b in range(slc.start, slc.stop):
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
                transform_result = estimate_similarity_RANSAC(
                    img_points,
                    tile_points,
                    trials=2000,
                    tol_px=config.pixel_tolerance,
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
            hyp2 = _derive_similarity(img_in.astype(np.float64), tile_in.astype(np.float64))
            if hyp2 is not None:
                rot_scale2, translation2 = hyp2
                transform = SimilarityTransform(
                    scale=float(abs(rot_scale2)),
                    rotation=float(np.angle(rot_scale2)),
                    translation=(float(translation2.real), float(translation2.imag)),
                )
            crpix = (image.shape[1] / 2.0 + 1.0, image.shape[0] / 2.0 + 1.0)
            wcs = tan_from_similarity(
                transform,
                image.shape,
                center_pixel=crpix,
                tile_center=(float(tile_entry.get("center_ra_deg", 0.0)), float(tile_entry.get("center_dec_deg", 0.0))),
            )
            matches_array = _build_matches_array(img_in, world_in)
            stats = validate_solution(wcs, matches_array, thresholds)
            final_wcs = wcs
            final_stats = stats
            sip_used = 0
            fov_deg = float(np.hypot(*tile_positions.ptp(axis=0))) if tile_positions.size else 0.0
            if final_stats.get("quality") == "GOOD" and needs_sip(final_wcs, final_stats, fov_deg):
                for order in range(2, config.sip_order + 1):
                    candidate_wcs, _ = fit_wcs_sip(matches_array, order=order)
                    candidate_stats = validate_solution(candidate_wcs, matches_array, thresholds)
                    if candidate_stats["rms_px"] < final_stats["rms_px"]:
                        final_wcs = candidate_wcs
                        final_stats = candidate_stats
                        sip_used = order
                    if not needs_sip(final_wcs, final_stats, fov_deg):
                        break
            stats = final_stats
            if stats.get("quality") != "GOOD":
                logger.debug(
                    "validation failed for tile %s (level=%s, parity=%s): %s",
                    candidate_key,
                    level_name,
                    parity_label,
                    stats,
                )
                continue
            header_updates = {
                "SOLVED": 1,
                "DBSET": tile_entry.get("family"),
                "TILE_ID": tile_entry.get("tile_key"),
                "RMS_PX": stats["rms_px"],
                "N_INLIERS": stats["inliers"],
                "SIP_ORDER": sip_used,
                "QUALITY": stats["quality"],
                "USED_DB": tile_entry.get("family"),
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

    def _run_levels(hashes: np.ndarray, parity_label: str) -> WcsSolution | None:
        best: WcsSolution | None = None
        preferred_tile: str | None = None

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

        for level_name in levels:
            solution = _attempt_level(level_name, hashes, preferred_tile, parity_label)
            if solution is None:
                if best is None:
                    return None
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
    for variant_hashes, parity_label in variants:
        logger.info("starting candidate search (parity=%s)", parity_label)
        best_solution = _run_levels(variant_hashes, parity_label)
        if best_solution:
            best_solution.message += f" (parity={parity_label})"
            break
    _log_phase("candidate search", stage)
    if not best_solution:
        return WcsSolution(False, "no valid solution", None, {}, None, {})
    header_updates = {
        **best_solution.header_updates,
        "BLIND_VER": ZEBLIND_VERSION,
    }
    with fits.open(input_fits, mode="update", memmap=False) as hdul:
        header = hdul[0].header
        for key, value in best_solution.wcs.to_header(relax=True).items():
            header[key] = value
        for key, value in header_updates.items():
            header[key] = value
        header["ZEBLINDVER"] = ZEBLIND_VERSION
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
