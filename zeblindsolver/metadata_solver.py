from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from astropy.io import fits

from .fits_utils import estimate_scale_and_fov, parse_angle, to_luminance_for_solve
from .matcher import SimilarityTransform, estimate_similarity_RANSAC
from .projections import project_tan
from .quad_index_builder import load_manifest
from zewcs290.catalog290 import CatalogDB
from .star_detect import detect_stars
from .verify import validate_solution
from .wcs_fit import fit_wcs_sip, fit_wcs_tan, needs_sip, tan_from_similarity
from .zeblindsolver import WcsSolution

try:
    from importlib.metadata import PackageNotFoundError, version as pkg_version
except ImportError:  # pragma: no cover - stdlib fallback
    PackageNotFoundError = Exception  # type: ignore[misc]

try:
    __version__ = pkg_version("zewcs290")
except PackageNotFoundError:
    __version__ = "0.0.dev"

NEAR_SOLVER_VERSION = __version__
logger = logging.getLogger(__name__)
_MIN_SEARCH_RADIUS = 0.7
_MAX_SEARCH_RADIUS = 5.0
_MAX_TILE_CANDIDATES = 48
_MAX_NEIGHBORS = 6
_RANK_TOLERANCE = 0.45


@dataclass
class NearSolveConfig:
    max_img_stars: int = 800
    max_cat_stars: int = 2000
    search_margin: float = 1.2
    pixel_tolerance: float = 3.0
    sip_order: int = 2
    quality_rms: float = 1.0
    quality_inliers: int = 60
    try_parity_flip: bool = True
    log_level: str = "INFO"


def _failure(message: str) -> WcsSolution:
    return WcsSolution(False, message, None, {}, None, {})


def _wrap_ra(delta: float) -> float:
    return (delta + 540.0) % 360.0 - 180.0


def _angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad = math.radians(ra1)
    ra2_rad = math.radians(ra2)
    dec1_rad = math.radians(dec1)
    dec2_rad = math.radians(dec2)
    sin_d1 = math.sin(dec1_rad)
    cos_d1 = math.cos(dec1_rad)
    sin_d2 = math.sin(dec2_rad)
    cos_d2 = math.cos(dec2_rad)
    delta_ra = ra1_rad - ra2_rad
    cos_c = sin_d1 * sin_d2 + cos_d1 * cos_d2 * math.cos(delta_ra)
    cos_c = min(1.0, max(-1.0, cos_c))
    return math.degrees(math.acos(cos_c))


def _tile_extent(entry: dict) -> float:
    bounds = entry.get("bounds") or {}
    dec_min = float(bounds.get("dec_min", entry.get("center_dec_deg", 0.0)))
    dec_max = float(bounds.get("dec_max", entry.get("center_dec_deg", 0.0)))
    dec_span = abs(dec_max - dec_min)
    ra_span = 0.0
    for segment in bounds.get("ra_segments", []):
        if not isinstance(segment, Iterable):
            continue
        start, end = segment
        width = abs(_wrap_ra(float(end) - float(start)))
        ra_span = max(ra_span, width)
    cos_dec = math.cos(math.radians(entry.get("center_dec_deg", 0.0)))
    cos_dec = max(cos_dec, 1e-3)
    return 0.5 * max(dec_span, ra_span * cos_dec)


def _tile_intersects(entry: dict, ra0: float, dec0: float, radius: float) -> tuple[bool, float]:
    center_ra = float(entry.get("center_ra_deg", ra0))
    center_dec = float(entry.get("center_dec_deg", dec0))
    distance = _angular_distance(center_ra, center_dec, ra0, dec0)
    extent = max(_tile_extent(entry), 0.25)
    return distance <= radius + extent, distance


def _select_tiles(manifest: dict, ra0: float, dec0: float, radius: float) -> list[dict]:
    tiles = manifest.get("tiles", [])
    selected: list[tuple[dict, float]] = []
    for entry in tiles:
        intersects, distance = _tile_intersects(entry, ra0, dec0, radius)
        if not intersects:
            continue
        selected.append((entry, distance))
    selected.sort(key=lambda item: item[1])
    return [entry for entry, _ in selected[: _MAX_TILE_CANDIDATES]]


def _extract_angle(header: fits.Header, keys: Iterable[str], *, is_ra: bool) -> Optional[float]:
    for key in keys:
        if key not in header:
            continue
        value = parse_angle(header.get(key), is_ra=is_ra)
        if value is not None:
            return value
    return None


def _load_tile_catalog(
    index_root: Path,
    entry: dict,
    ra0: float,
    dec0: float,
    *,
    db_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = str(entry.get("tile_file", "")).replace("\\", "/")
    tile_path = index_root / rel
    ra = dec = mag = None
    if tile_path.exists():
        try:
            with np.load(tile_path) as data:
                ra = data.get("ra_deg")
                dec = data.get("dec_deg")
                mag = data.get("mag")
                if ra is not None:
                    ra = ra.astype(np.float64, copy=False)
                if dec is not None:
                    dec = dec.astype(np.float64, copy=False)
                if mag is not None:
                    mag = mag.astype(np.float32, copy=False)
        except Exception:
            ra = dec = mag = None
    # Fallback: read directly from the ASTAP DB if the tile blob is missing or empty
    if (ra is None or dec is None or mag is None or ra.size == 0 or dec.size == 0) and db_root is not None:
        try:
            fam = str(entry.get("family") or "").strip().lower() or None
            db = CatalogDB(db_root, families=[fam] if fam else None)
            target_code = str(entry.get("tile_code") or "")
            ra = dec = mag = None
            for tile in db.tiles:
                if tile.tile_code == target_code and (fam is None or tile.spec.key == fam):
                    block = db._load_tile(tile)
                    stars = block.stars
                    ra = stars["ra_deg"].astype(np.float64, copy=False)
                    dec = stars["dec_deg"].astype(np.float64, copy=False)
                    mag = stars["mag"].astype(np.float32, copy=False)
                    break
        except Exception:
            ra = dec = mag = None
    if ra is None or dec is None or mag is None:
        raise FileNotFoundError(tile_path)
    x_deg, y_deg = project_tan(ra, dec, ra0, dec0)
    mask = np.isfinite(x_deg) & np.isfinite(y_deg)
    if not mask.any():
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)
    positions = np.column_stack((x_deg[mask], y_deg[mask])).astype(np.float32, copy=False)
    world = np.column_stack((ra[mask], dec[mask])).astype(np.float64, copy=False)
    mags = mag[mask]
    return positions, world, mags


def _compute_ranks(values: np.ndarray, *, descending: bool = False) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    if descending:
        order = np.argsort(values)[::-1]
    else:
        order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=False)
    return ranks


def _build_candidate_pairs(
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    catalog_world: np.ndarray,
    img_ranks: np.ndarray,
    cat_ranks: np.ndarray,
    center_xy: tuple[float, float],
    approx_scale_deg: float,
    pixel_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image_positions.size == 0 or catalog_positions.size == 0:
        return (np.empty((0, 2), dtype=np.float32),) * 3
    cx, cy = center_xy
    img_delta = image_positions - np.array([cx, cy], dtype=np.float32)
    img_radius_px = np.hypot(img_delta[:, 0], img_delta[:, 1])
    cat_radius_deg = np.hypot(catalog_positions[:, 0], catalog_positions[:, 1])
    order = np.argsort(cat_radius_deg)
    radius_sorted = cat_radius_deg[order]
    idx_sorted = order
    base_window = max(0.01, approx_scale_deg * max(pixel_tolerance * 2.5, 0.5))
    votes: list[tuple[int, int, float]] = []
    for img_idx, radius_px in enumerate(img_radius_px):
        target = radius_px * approx_scale_deg
        window = max(base_window, target * 0.25)
        left = np.searchsorted(radius_sorted, target - window, side="left")
        right = np.searchsorted(radius_sorted, target + window, side="right")
        if right <= left:
            continue
        candidates = idx_sorted[left:right]
        for cat_idx in candidates[:_MAX_NEIGHBORS]:
            rank_gap = abs(float(img_ranks[img_idx]) - float(cat_ranks[cat_idx]))
            if rank_gap > _RANK_TOLERANCE:
                continue
            diff = abs(cat_radius_deg[cat_idx] - target)
            score = (1.0 - min(0.9, rank_gap)) / (diff + 1e-6)
            votes.append((img_idx, int(cat_idx), float(score)))
    if not votes:
        return (np.empty((0, 2), dtype=np.float32),) * 3
    votes.sort(key=lambda item: item[2], reverse=True)
    max_pairs = min(len(votes), max(200, image_positions.shape[0] * 6))
    chosen = votes[:max_pairs]
    img_points = np.array([image_positions[i] for i, _, _ in chosen], dtype=np.float32)
    cat_points = np.array([catalog_positions[j] for _, j, _ in chosen], dtype=np.float32)
    cat_world_points = np.array([catalog_world[j] for _, j, _ in chosen], dtype=np.float64)
    return img_points, cat_points, cat_world_points


def _compute_inliers(
    transform: SimilarityTransform,
    img_points: np.ndarray,
    catalog_points: np.ndarray,
    pixel_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if img_points.size == 0 or catalog_points.size == 0:
        return np.zeros(0, dtype=bool), np.empty(0, dtype=np.float32)
    src = img_points[:, 0] + 1j * img_points[:, 1]
    if getattr(transform, "parity", 1) < 0:
        src = np.conj(src)
    dst = catalog_points[:, 0] + 1j * catalog_points[:, 1]
    rot_scale = transform.scale * np.exp(1j * transform.rotation)
    translation = complex(*transform.translation)
    prediction = rot_scale * src + translation
    err_deg = np.abs(prediction - dst)
    tol_deg = max(1e-6, float(pixel_tolerance) * max(transform.scale, 1e-12))
    mask = err_deg <= tol_deg
    err_px = err_deg / max(transform.scale, 1e-8)
    return mask, err_px.astype(np.float32, copy=False)


def _pix_scale_arcsec(wcs) -> Optional[float]:
    cd = getattr(wcs.wcs, "cd", None)
    if cd is None:
        return None
    det = float(np.linalg.det(cd))
    if not math.isfinite(det) or det == 0.0:
        return None
    return math.sqrt(abs(det)) * 3600.0


def solve_near(
    input_fits: Path | str,
    index_root: Path | str,
    *,
    config: NearSolveConfig | None = None,
) -> WcsSolution:
    cfg = config or NearSolveConfig()
    logger.setLevel(cfg.log_level.upper())
    start = time.perf_counter()
    fits_path = Path(input_fits).expanduser().resolve()
    index_path = Path(index_root).expanduser().resolve()
    try:
        manifest = load_manifest(index_path)
    except FileNotFoundError as exc:
        return _failure(f"index manifest missing: {exc}")
    tiles = manifest.get("tiles") or []
    if not tiles:
        return _failure("index manifest has no tiles")
    try:
        with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
            primary = hdul[0]
            if primary.data is None:
                return _failure("FITS HDU has no image data")
            image = to_luminance_for_solve(primary)
            header = primary.header
    except Exception as exc:
        return _failure(f"failed to read FITS data: {exc}")
    height, width = image.shape
    ra_keys = ("RA", "OBJCTRA", "OBJRA", "OBJ_RA", "CRVAL1")
    dec_keys = ("DEC", "OBJCTDEC", "OBJDEC", "OBJ_DEC", "CRVAL2")
    ra0 = _extract_angle(header, ra_keys, is_ra=True)
    dec0 = _extract_angle(header, dec_keys, is_ra=False)
    if ra0 is None or dec0 is None:
        return _failure("metadata RA/DEC missing for near solve")
    scale_arcsec, (fov_x, fov_y) = estimate_scale_and_fov(header, width, height)
    fov_candidates = [value for value in (fov_x, fov_y) if value is not None]
    approx_fov = max(fov_candidates) if fov_candidates else None
    approx_scale_deg = scale_arcsec / 3600.0 if scale_arcsec else None
    if approx_scale_deg is None:
        approx_fov = approx_fov or 1.5
        approx_scale_deg = approx_fov / max(width, height)
    approx_scale_deg = max(approx_scale_deg, 1e-5)
    fov_for_radius = approx_fov or (approx_scale_deg * max(width, height))
    radius = max(_MIN_SEARCH_RADIUS, 0.5 * fov_for_radius * max(cfg.search_margin, 1.0))
    radius = min(radius, _MAX_SEARCH_RADIUS)
    logger.info(
        "near solve start for %s (radius=%.2f°, approx_scale=%.3g°/px)",
        fits_path.name,
        radius,
        approx_scale_deg,
    )
    candidates = _select_tiles(manifest, ra0, dec0, radius)
    if not candidates:
        return _failure("manifest present but no tile intersects the metadata cone")
    catalog_positions: list[np.ndarray] = []
    catalog_world: list[np.ndarray] = []
    catalog_mags: list[np.ndarray] = []
    missing_tiles: list[str] = []
    # Resolve DB root for optional fallback when tile blobs are empty/missing
    try:
        db_root_text = str(manifest.get("db_root", "")).strip()
        db_root_path: Path | None = Path(db_root_text).expanduser().resolve() if db_root_text else None
    except Exception:
        db_root_path = None
    for entry in candidates:
        try:
            positions, world, mags = _load_tile_catalog(index_path, entry, ra0, dec0, db_root=db_root_path)
        except FileNotFoundError:
            missing_tiles.append(str(entry.get("tile_file")))
            continue
        if positions.size == 0:
            continue
        catalog_positions.append(positions)
        catalog_world.append(world)
        catalog_mags.append(mags)
    if not catalog_positions:
        if missing_tiles:
            return _failure(f"tile files missing: {missing_tiles[0]}")
        return _failure("candidate tiles found but none yielded catalog stars")
    cat_positions = np.vstack(catalog_positions)
    cat_world = np.vstack(catalog_world)
    cat_mags = np.concatenate(catalog_mags)
    if cfg.max_cat_stars and cat_positions.shape[0] > cfg.max_cat_stars:
        order = np.argsort(cat_mags)
        keep = order[: cfg.max_cat_stars]
        cat_positions = cat_positions[keep]
        cat_world = cat_world[keep]
        cat_mags = cat_mags[keep]
    stars = detect_stars(image)
    if stars.size == 0:
        return _failure("no stars detected in the frame")
    if cfg.max_img_stars and stars.size > cfg.max_img_stars:
        stars = stars[: cfg.max_img_stars]
    image_positions = np.column_stack((stars["x"], stars["y"])).astype(np.float32, copy=False)
    img_ranks = _compute_ranks(stars["flux"], descending=True)
    cat_ranks = _compute_ranks(cat_mags, descending=False)
    center_xy = (width / 2.0, height / 2.0)
    img_pairs, cat_pairs, cat_world_pairs = _build_candidate_pairs(
        image_positions,
        cat_positions,
        cat_world,
        img_ranks,
        cat_ranks,
        center_xy,
        approx_scale_deg,
        cfg.pixel_tolerance,
    )
    if img_pairs.size == 0:
        return _failure("unable to build candidate matches from metadata")
    hypothesis = estimate_similarity_RANSAC(
        img_pairs,
        cat_pairs,
        trials=2000,
        tol_px=cfg.pixel_tolerance,
        min_inliers=4,
        allow_reflection=cfg.try_parity_flip,
    )
    if hypothesis is None:
        return _failure("near solver could not estimate a similarity transform")
    transform, _ = hypothesis
    inlier_mask, _ = _compute_inliers(transform, img_pairs, cat_pairs, cfg.pixel_tolerance)
    if not inlier_mask.any():
        return _failure("no geometric consensus found for metadata solve")
    img_in = img_pairs[inlier_mask]
    cat_world_in = cat_world_pairs[inlier_mask]
    matches = np.column_stack((img_in, cat_world_in)).astype(np.float64, copy=False)
    tile_center = (float(ra0), float(dec0))
    wcs = tan_from_similarity(transform, image.shape, tile_center=tile_center)
    stats = validate_solution(
        wcs,
        matches,
        thresholds={"rms_px": cfg.quality_rms, "inliers": cfg.quality_inliers},
    )
    final_wcs = wcs
    final_stats = stats
    try:
        ls_wcs, _ = fit_wcs_tan(matches)
    except Exception:
        ls_wcs = None
    else:
        ls_stats = validate_solution(
            ls_wcs,
            matches,
            thresholds={"rms_px": cfg.quality_rms, "inliers": cfg.quality_inliers},
        )
        if ls_stats.get("rms_px", float("inf")) < final_stats.get("rms_px", float("inf")):
            final_wcs = ls_wcs
            final_stats = ls_stats
    fov_est = approx_fov or (2.0 * np.max(np.hypot(cat_positions[:, 0], cat_positions[:, 1])))
    if final_stats.get("quality") == "GOOD" and needs_sip(final_wcs, final_stats, fov_est):
        for order in range(2, cfg.sip_order + 1):
            candidate_wcs, _ = fit_wcs_sip(matches, order=order)
            candidate_stats = validate_solution(
                candidate_wcs,
                matches,
                thresholds={"rms_px": cfg.quality_rms, "inliers": cfg.quality_inliers},
            )
            if candidate_stats["rms_px"] < final_stats["rms_px"]:
                final_wcs = candidate_wcs
                final_stats = candidate_stats
            if not needs_sip(final_wcs, final_stats, fov_est):
                break
    if final_stats.get("quality") != "GOOD":
        return _failure(f"near solution failed validation ({final_stats})")
    header_updates = {
        "SOLVED": 1,
        "QUALITY": final_stats.get("quality", "GOOD"),
        "NEAR_VER": NEAR_SOLVER_VERSION,
        "RMSPX": final_stats.get("rms_px"),
        "INLIERS": final_stats.get("inliers"),
        "TILE_ID": candidates[0].get("tile_key") if candidates else None,
        "SOLVMODE": "NEAR",
        "SOLVER": "ZeSolver",
    }
    pix_scale_arcsec = _pix_scale_arcsec(final_wcs)
    if pix_scale_arcsec is not None:
        header_updates["PIXSCAL"] = pix_scale_arcsec
    elapsed = time.perf_counter() - start
    header_updates["NEARTIME"] = f"{elapsed:.2f}s"
    try:
        with fits.open(fits_path, mode="update", memmap=False) as hdul:
            header = hdul[0].header
            for key, value in final_wcs.to_header(relax=True).items():
                header[key] = value
            for key, value in header_updates.items():
                if value is not None:
                    header[key] = value
            hdul.flush()
    except Exception as exc:
        return _failure(f"unable to write WCS to FITS: {exc}")
    logger.info(
        "near solve succeeded for %s (rms=%.3f px, inliers=%d, %.1fs)",
        fits_path.name,
        final_stats.get("rms_px", float("nan")),
        final_stats.get("inliers", 0),
        elapsed,
    )
    return WcsSolution(
        True,
        "near solution found",
        final_wcs,
        final_stats,
        candidates[0].get("tile_key") if candidates else None,
        header_updates,
    )
