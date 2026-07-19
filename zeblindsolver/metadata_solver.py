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

import logging
import json
import hashlib
import math
import re
import time
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from collections import OrderedDict
import threading
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Callable

import numpy as np
from astropy.io import fits

from .fits_utils import estimate_scale_and_fov, parse_angle, to_luminance_for_solve
from .image_prep import remove_background
from .matcher import SimilarityTransform, estimate_similarity_RANSAC
from .projections import project_tan
from .quad_index_builder import load_manifest
from .astap_db_reader import iter_tiles as iter_astap_tiles, load_tile_stars as load_astap_tile_stars
from .near_catalog_provider import NearCatalogProvider, NearCatalogProviderError, NearCatalogTile
from zewcs290.catalog290 import CatalogDB
from .star_detect import detect_stars
from .verify import validate_solution
from .wcs_fit import fit_wcs_sip, fit_wcs_tan, needs_sip, tan_from_similarity
from .wcs_header import apply_wcs_solution_to_header, validate_wcs_for_zemosaic
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
_QUAD_RATIO_TOL = 0.035
_MAX_QUAD_MATCHES_PER_IMG = 3


@dataclass
class NearSolveConfig:
    max_img_stars: int = 800
    max_cat_stars: int = 2000
    search_margin: float = 1.8
    pixel_tolerance: float = 3.0
    sip_order: int = 2
    quality_rms: float = 1.0
    quality_inliers: int = 60
    try_parity_flip: bool = True
    log_level: str = "INFO"
    # Optional restrictions/overrides
    # - family: restrict candidate tiles to a single catalog family (e.g. "d50")
    # - fov_override_deg: override approximate FOV when estimating search radius; 0/None → auto
    family: str | None = None
    fov_override_deg: float | None = None
    # Performance tuning
    # - max_tile_candidates: cap how many intersecting tiles to consider per image
    # - tile_cache_size: LRU size for cached tile blobs (RA/DEC/MAG arrays)
    max_tile_candidates: int = _MAX_TILE_CANDIDATES
    tile_cache_size: int = 128
    # Star detection compute backend
    detect_backend: str = "auto"  # "auto" | "cpu" | "cuda" | "astap"
    detect_device: int | None = None
    # Star detection tuning (for throughput/quality trade-off)
    detect_k_sigma: float = 4.0
    detect_min_area: int = 8
    detect_max_labels: int = 2500
    # ASTAP extract tuning
    # - astap_extract_bin_factor: optional pre-binning factor before ASTAP -extract
    # - astap_extract_bin_strict_only: apply only in strict ASTAP-ISO mode
    # - astap_extract_bin_min_stars: retry full-res extract if binned result is too sparse
    astap_extract_bin_factor: int = 1
    astap_extract_bin_strict_only: bool = True
    astap_extract_bin_min_stars: int = 12
    # Max concurrent GPU detection slots across workers (hybrid pipeline guard).
    # 1 = serialize detect on GPU to reduce CPU<->GPU contention; >1 allows more overlap.
    detect_gpu_slots: int = 1
    # RANSAC settings
    ransac_trials: int = 1200
    # Optional deterministic seed. When None, derived from FITS path for reproducibility.
    ransac_seed: int | None = None
    # Optional warm-start seed (from previous near solution)
    seed_scale_deg: float | None = None
    seed_rotation: float | None = None  # radians
    seed_parity: int = 1
    # ASTAP-like hint dynamics (throughput-first near mode, mainly for ZeMosaic usage)
    # - fastpath=True: prefer strict hinted-local search first (smaller candidate cone/window)
    # - hint_radius_deg: radius used for hinted-local search (like ASTAP -r)
    # - second_pass_refine_in_fastpath: keep disabled by default for speed
    astap_hint_fastpath: bool = False
    astap_hint_radius_deg: float = 3.0
    second_pass_refine_in_fastpath: bool = False
    # Strict ASTAP-ISO execution path (diagnostic/parity mode).
    # When enabled, solve_near follows the ASTAP-ISO core path and bypasses
    # non-ISO branches/gates guarded by `strict_astap_iso` checks.
    astap_iso_strict: bool = True
    # Mirror ASTAP quad matching tolerance in strict mode.
    astap_iso_quad_tolerance: float = 0.007
    # Strict auto-FOV retry (used only when FOV comes from scale inference).
    # Keeps explicit FOV hints (override/header) authoritative.
    strict_auto_fov_retry: bool = True
    strict_auto_fov_retry_scales: tuple[float, ...] = (1.25, 0.82, 1.6, 0.65, 2.4, 4.0)
    # 0 means unlimited attempts (bounded by scales list).
    strict_auto_fov_retry_max_attempts: int = 0
    # Break retries when repeated attempts keep returning zero refs.
    strict_auto_fov_retry_zero_ref_patience: int = 3
    # Non-strict ASTAP-ISO scale gate and adaptive retry.
    # Useful when hinted focal/pixel metadata is imperfect (binning/crop mismatch).
    astap_iso_scale_ratio_min: float = 0.55
    astap_iso_scale_ratio_max: float = 1.80
    astap_iso_auto_scale_retry: bool = True
    astap_iso_auto_scale_factors: tuple[float, ...] = (0.50, 2.00, 0.75, 1.33, 1.60)
    astap_iso_auto_scale_last_chance_no_gate: bool = True
    # Reliability-first conformance gates (used for final near acceptance)
    conformance_scale_min_ratio: float = 0.60
    conformance_scale_max_ratio: float = 1.80
    conformance_center_extra_deg: float = 0.6
    conformance_center_fov_mult: float = 1.2
    conformance_center_max_deg: float = 0.90
    # Strict ASTAP-ISO acceptance gate.  ZN3.5 moves strict mode from a mirror
    # path to a product candidate path: a candidate must be accepted before WCS
    # writing.  Use "diagnostic" to compute metrics without changing outcome.
    strict_acceptance_mode: str = "diagnostic"  # "off" | "diagnostic" | "enforce"
    strict_acceptance_min_unique_matches: int = 12
    strict_acceptance_min_holdout_matches: int = 8
    strict_acceptance_max_rms_px: float = 3.0
    strict_acceptance_max_cd_cond: float = 1.0e4
    strict_acceptance_min_span_fraction: float = 0.12
    # Diagnostic-only dumps. Disabled by default; used by ZN probes to compare
    # ZeNear's exact selected lists against external solvers without changing
    # the solving path.
    diagnostic_dump_dir: str | None = None
    diagnostic_dump_label: str | None = None
    diagnostic_image_stars_csv: str | None = None
    diagnostic_catalog_stars_csv: str | None = None
    diagnostic_iso_trace: bool = False


def _failure(message: str) -> WcsSolution:
    return WcsSolution(False, message, None, {}, None, {})


def _near_reports_dir() -> Path:
    # ZeSolver/zeblindsolver/metadata_solver.py -> ZeSolver/reports
    return Path(__file__).resolve().parents[1] / "reports"


def _emit_near_debug_record(record: dict) -> None:
    try:
        out_dir = _near_reports_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "near_debug.jsonl"
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _dump_near_diagnostic_lists(
    *,
    dump_dir: str | Path | None,
    label: str,
    stars: np.ndarray,
    cat_positions: np.ndarray,
    cat_world: np.ndarray,
    cat_mags: np.ndarray,
) -> None:
    if not dump_dir:
        return
    try:
        out_dir = Path(dump_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label or "zenear")).strip("_") or "zenear"

        img_path = out_dir / f"{safe_label}_zenear_image_stars.csv"
        with img_path.open("w", encoding="utf-8") as f:
            f.write("rank,x,y,flux\n")
            n = int(stars.size)
            for i in range(n):
                f.write(
                    f"{i + 1},"
                    f"{float(stars['x'][i]):.6f},"
                    f"{float(stars['y'][i]):.6f},"
                    f"{float(stars['flux'][i]):.6f}\n"
                )

        cat_path = out_dir / f"{safe_label}_zenear_catalog_stars.csv"
        with cat_path.open("w", encoding="utf-8") as f:
            f.write("rank,ra_deg,dec_deg,x_tan_deg,y_tan_deg,mag\n")
            n = int(min(cat_positions.shape[0], cat_world.shape[0], cat_mags.shape[0]))
            for i in range(n):
                f.write(
                    f"{i + 1},"
                    f"{float(cat_world[i, 0]):.10f},"
                    f"{float(cat_world[i, 1]):.10f},"
                    f"{float(cat_positions[i, 0]):.10f},"
                    f"{float(cat_positions[i, 1]):.10f},"
                    f"{float(cat_mags[i]):.6f}\n"
                )
    except Exception:
        pass


def _astap_iso_collect_quad_hits(
    quads_ref: np.ndarray,
    quads_img: np.ndarray,
    *,
    minimum_count: int,
    quad_tolerance: float,
) -> dict:
    """Diagnostic mirror of ASTAP-ISO quad matching.

    This intentionally does not feed the solver decision.  It exposes the
    raw signature hits and the scale-ratio retained subset used by the fit
    functions so probes can distinguish lookup from transform failures.
    """
    n1 = int(quads_ref.shape[1]) if quads_ref.ndim == 2 else 0
    n2 = int(quads_img.shape[1]) if quads_img.ndim == 2 else 0
    if n1 < int(minimum_count) or n2 < int(minimum_count):
        return {"path_used": None, "matches_raw": 0, "matches_kept": 0, "median_ratio": None, "hits": []}

    use_bruteforce = n1 < 180
    matches: list[tuple[int, int, float]] = []
    tol = max(1e-12, float(quad_tolerance))
    if use_bruteforce:
        for i in range(n1):
            r1 = quads_ref[1:6, i]
            for j in range(n2):
                r2 = quads_img[1:6, j]
                d = float(np.max(np.abs(r1 - r2)))
                if d <= tol:
                    matches.append((i, j, d))
    else:
        hash_bins = max(1, 2 * max(n1, n2))
        h1 = [[] for _ in range(hash_bins)]
        h2 = [[] for _ in range(hash_bins)]
        for i in range(n1):
            b = int(quads_ref[1, i] / tol) % hash_bins
            h1[b].append(i)
        for j in range(n2):
            b = int(quads_img[1, j] / tol) % hash_bins
            h2[b].append(j)
        for b in range(hash_bins):
            if not h1[b]:
                continue
            for db in (-1, 0, 1):
                bb = (b + db) % hash_bins
                if not h2[bb]:
                    continue
                for i in h1[b]:
                    r1 = quads_ref[1:6, i]
                    for j in h2[bb]:
                        r2 = quads_img[1:6, j]
                        d = float(np.max(np.abs(r1 - r2)))
                        if d <= tol:
                            matches.append((i, j, d))

    if not matches:
        return {"path_used": "find_fit" if use_bruteforce else "find_fit_using_hash", "matches_raw": 0, "matches_kept": 0, "median_ratio": None, "hits": []}

    ratios = np.array([quads_ref[0, i] / max(quads_img[0, j], 1e-12) for i, j, _ in matches], dtype=np.float64)
    median_ratio = float(np.median(ratios)) if ratios.size else float("nan")
    hits: list[dict] = []
    kept = 0
    for rank, ((i, j, d), ratio) in enumerate(zip(matches, ratios), start=1):
        retained = bool(np.isfinite(median_ratio) and median_ratio > 0 and abs(median_ratio - float(ratio)) <= tol * median_ratio)
        if retained:
            kept += 1
        hits.append({
            "lookup_rank": int(rank),
            "catalog_quad_rank": int(i + 1),
            "image_quad_rank": int(j + 1),
            "code_distance": float(d),
            "scale_ratio": float(ratio) if np.isfinite(ratio) else None,
            "retained": retained,
            "rejection_reason": None if retained else "SCALE_RATIO_MEDIAN_FILTER",
        })
    return {
        "path_used": "find_fit" if use_bruteforce else "find_fit_using_hash",
        "matches_raw": int(len(matches)),
        "matches_kept": int(kept),
        "median_ratio": float(median_ratio) if np.isfinite(median_ratio) else None,
        "hits": hits,
    }


def _dump_astap_iso_trace(
    *,
    dump_dir: str | Path | None,
    label: str,
    stage: str,
    trace: dict | None,
    diag: dict | None,
    minimum_count: int,
    quad_tolerance: float,
) -> None:
    if not dump_dir or not trace:
        return
    try:
        out_dir = Path(dump_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label or "zenear")).strip("_") or "zenear"
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage or "initial")).strip("_") or "initial"
        prefix = out_dir / f"{safe_label}_iso_{safe_stage}"

        img = np.asarray(trace.get("image_positions", np.empty((0, 2))), dtype=np.float64)
        img_src = np.asarray(trace.get("image_source_indices", np.arange(img.shape[0])), dtype=np.int64)
        img_flux = trace.get("image_fluxes")
        img_flux_arr = np.asarray(img_flux, dtype=np.float64) if img_flux is not None else None
        with prefix.with_name(prefix.name + "_image_final_for_quads.csv").open("w", encoding="utf-8") as f:
            f.write("rank,source_index,x,y,flux\n")
            for i in range(int(img.shape[0])):
                flux_text = ""
                if img_flux_arr is not None and i < int(img_flux_arr.shape[0]) and np.isfinite(img_flux_arr[i]):
                    flux_text = f"{float(img_flux_arr[i]):.10f}"
                f.write(f"{i + 1},{int(img_src[i])},{float(img[i,0]):.10f},{float(img[i,1]):.10f},{flux_text}\n")

        cat = np.asarray(trace.get("catalog_positions", np.empty((0, 2))), dtype=np.float64)
        cat_src = np.asarray(trace.get("catalog_source_indices", np.arange(cat.shape[0])), dtype=np.int64)
        cat_world = trace.get("catalog_world")
        cat_world_arr = np.asarray(cat_world, dtype=np.float64) if cat_world is not None else None
        cat_mags = trace.get("catalog_mags")
        cat_mag_arr = np.asarray(cat_mags, dtype=np.float64) if cat_mags is not None else None
        with prefix.with_name(prefix.name + "_catalog_final_for_quads.csv").open("w", encoding="utf-8") as f:
            f.write("rank,source_index,x_arcsec,y_arcsec,ra_deg,dec_deg,magnitude,identity_kind,decoded_identity\n")
            for i in range(int(cat.shape[0])):
                ra_text = dec_text = mag_text = ""
                identity = ""
                if cat_world_arr is not None and i < int(cat_world_arr.shape[0]):
                    ra = float(cat_world_arr[i, 0])
                    dec = float(cat_world_arr[i, 1])
                    if np.isfinite(ra) and np.isfinite(dec):
                        ra_text = f"{ra:.10f}"
                        dec_text = f"{dec:.10f}"
                if cat_mag_arr is not None and i < int(cat_mag_arr.shape[0]) and np.isfinite(cat_mag_arr[i]):
                    mag_text = f"{float(cat_mag_arr[i]):.6f}"
                if ra_text and dec_text and mag_text:
                    identity = f"radec_mag:{float(ra_text):.8f}:{float(dec_text):.8f}:{float(mag_text):.3f}"
                f.write(
                    f"{i + 1},{int(cat_src[i])},{float(cat[i,0]):.10f},{float(cat[i,1]):.10f},"
                    f"{ra_text},{dec_text},{mag_text},decoded_coordinate_surrogate,{identity}\n"
                )

        def _write_quads(name: str, q: np.ndarray) -> None:
            q = np.asarray(q, dtype=np.float64)
            with prefix.with_name(prefix.name + f"_{name}_quads.csv").open("w", encoding="utf-8") as f:
                f.write("quad_rank,longest,signature1,signature2,signature3,signature4,signature5,center_x,center_y\n")
                n = int(q.shape[1]) if q.ndim == 2 else 0
                for i in range(n):
                    f.write(
                        f"{i + 1},"
                        f"{float(q[0,i]):.10f},{float(q[1,i]):.10f},{float(q[2,i]):.10f},"
                        f"{float(q[3,i]):.10f},{float(q[4,i]):.10f},{float(q[5,i]):.10f},"
                        f"{float(q[6,i]):.10f},{float(q[7,i]):.10f}\n"
                    )

        q_img = np.asarray(trace.get("image_quads", np.empty((8, 0))), dtype=np.float64)
        q_cat = np.asarray(trace.get("catalog_quads", np.empty((8, 0))), dtype=np.float64)
        _write_quads("image", q_img)
        _write_quads("catalog", q_cat)

        hits = _astap_iso_collect_quad_hits(
            q_cat,
            q_img,
            minimum_count=int(minimum_count),
            quad_tolerance=float(quad_tolerance),
        )
        with prefix.with_name(prefix.name + "_hits.csv").open("w", encoding="utf-8") as f:
            f.write("lookup_rank,image_quad_rank,catalog_quad_rank,code_distance,scale_ratio,retained,rejection_reason\n")
            for row in hits.get("hits", []):
                ratio_text = "" if row.get("scale_ratio") is None else f"{float(row['scale_ratio']):.10f}"
                f.write(
                    f"{int(row['lookup_rank'])},"
                    f"{int(row['image_quad_rank'])},"
                    f"{int(row['catalog_quad_rank'])},"
                    f"{float(row['code_distance']):.10f},"
                    f"{ratio_text},"
                    f"{str(bool(row['retained'])).lower()},"
                    f"{row.get('rejection_reason') or ''}\n"
                )
        summary = {
            "stage": safe_stage,
            "image_final_for_quads": int(img.shape[0]),
            "catalog_final_for_quads": int(cat.shape[0]),
            "image_quads": int(q_img.shape[1]) if q_img.ndim == 2 else 0,
            "catalog_quads": int(q_cat.shape[1]) if q_cat.ndim == 2 else 0,
            "hits": {k: v for k, v in hits.items() if k != "hits"},
            "diag": diag or {},
            "stage_meta": trace.get("stage_meta") if isinstance(trace.get("stage_meta"), dict) else {},
        }
        prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _load_near_diagnostic_image_stars_csv(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    try:
        csv_path = Path(path).expanduser().resolve()
        if not csv_path.exists():
            return None
        arr = np.genfromtxt(
            csv_path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            invalid_raise=False,
        )
        if arr is None or getattr(arr, "size", 0) == 0:
            return np.zeros(0, dtype=_STAR_DETECT_DTYPE)
        arr = np.atleast_1d(arr)
        names = set(arr.dtype.names or ())
        if not {"x", "y", "flux"}.issubset(names):
            return None
        out = np.zeros(int(arr.shape[0]), dtype=_STAR_DETECT_DTYPE)
        out["x"] = np.asarray(arr["x"], dtype=np.float32)
        out["y"] = np.asarray(arr["y"], dtype=np.float32)
        out["flux"] = np.maximum(np.asarray(arr["flux"], dtype=np.float32), np.float32(1.0))
        finite = np.isfinite(out["x"]) & np.isfinite(out["y"]) & np.isfinite(out["flux"])
        out = out[finite]
        if out.size > 1:
            order = np.argsort(out["flux"], kind="stable")[::-1]
            out = out[order]
        return out
    except Exception:
        return None


def _load_near_diagnostic_catalog_stars_csv(path: str | Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path:
        return None
    try:
        csv_path = Path(path).expanduser().resolve()
        if not csv_path.exists():
            return None
        arr = np.genfromtxt(
            csv_path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            invalid_raise=False,
        )
        if arr is None or getattr(arr, "size", 0) == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                np.empty(0, dtype=np.float32),
            )
        arr = np.atleast_1d(arr)
        names = set(arr.dtype.names or ())
        x_name = "x_tan_deg" if "x_tan_deg" in names else ("x_projected" if "x_projected" in names else "x")
        y_name = "y_tan_deg" if "y_tan_deg" in names else ("y_projected" if "y_projected" in names else "y")
        if x_name not in names or y_name not in names:
            return None
        x = np.asarray(arr[x_name], dtype=np.float64)
        y = np.asarray(arr[y_name], dtype=np.float64)
        # ASTAP internal dumps are in standard coordinates/arcsec. ZeNear
        # diagnostic dumps use tangent degrees. Normalize to degrees here.
        if "x_projected" in names or (np.nanmax(np.abs(x)) > 20.0 if x.size else False):
            x_deg = x / 3600.0
            y_deg = y / 3600.0
        else:
            x_deg = x
            y_deg = y
        if {"ra_deg", "dec_deg"}.issubset(names):
            world = np.column_stack((np.asarray(arr["ra_deg"], dtype=np.float64), np.asarray(arr["dec_deg"], dtype=np.float64)))
        else:
            world = np.full((int(x_deg.size), 2), np.nan, dtype=np.float64)
        mag_name = "mag" if "mag" in names else ("magnitude" if "magnitude" in names else None)
        mags = np.asarray(arr[mag_name], dtype=np.float32) if mag_name else np.arange(int(x_deg.size), dtype=np.float32)
        finite = np.isfinite(x_deg) & np.isfinite(y_deg)
        return (
            np.column_stack((x_deg[finite], y_deg[finite])).astype(np.float64, copy=False),
            world[finite].astype(np.float64, copy=False),
            mags[finite].astype(np.float32, copy=False),
        )
    except Exception:
        return None


def _stable_seed_for_path(path: Path) -> int:
    payload = str(path).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0xFFFFFFFF


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


def _ra_segments_for_interval(ra_min: float, ra_max: float) -> list[tuple[float, float]]:
    span = float(ra_max - ra_min)
    if span >= 360.0:
        return [(0.0, 360.0)]
    a = ra_min % 360.0
    b = ra_max % 360.0
    if a <= b:
        return [(a, b)]
    return [(a, 360.0), (0.0, b)]


def _segments_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(float(a0), float(b0)) <= min(float(a1), float(b1))


def _tile_intersects(entry: dict, ra0: float, dec0: float, radius: float) -> tuple[bool, float]:
    center_ra = float(entry.get("center_ra_deg", ra0))
    center_dec = float(entry.get("center_dec_deg", dec0))
    distance = _angular_distance(center_ra, center_dec, ra0, dec0)

    bounds = entry.get("bounds") or {}
    try:
        dec_min = float(bounds.get("dec_min", center_dec))
        dec_max = float(bounds.get("dec_max", center_dec))
    except Exception:
        dec_min = dec_max = center_dec

    # First gate on declination overlap of the hinted cone.
    if (dec0 + radius) < dec_min or (dec0 - radius) > dec_max:
        return False, distance

    ra_segments = bounds.get("ra_segments") or []
    if isinstance(ra_segments, list) and len(ra_segments) > 0:
        cosd = max(1e-3, math.cos(math.radians(dec0)))
        ra_span = float(radius) / cosd
        query_segments = _ra_segments_for_interval(float(ra0) - ra_span, float(ra0) + ra_span)
        for seg in ra_segments:
            if not isinstance(seg, Iterable) or len(seg) < 2:
                continue
            s0 = float(seg[0]) % 360.0
            s1 = float(seg[1]) % 360.0
            tile_parts = [(s0, s1)] if s0 <= s1 else [(s0, 360.0), (0.0, s1)]
            for t0, t1 in tile_parts:
                if any(_segments_overlap(t0, t1, q0, q1) for q0, q1 in query_segments):
                    return True, distance
        return False, distance

    # Fallback when bounds are missing: conservative center-distance test.
    extent = max(_tile_extent(entry), 0.25)
    return distance <= radius + extent, distance


def _select_tiles(manifest: dict, ra0: float, dec0: float, radius: float, limit: int) -> list[dict]:
    tiles = manifest.get("tiles", [])
    selected: list[tuple[dict, float]] = []
    for entry in tiles:
        intersects, distance = _tile_intersects(entry, ra0, dec0, radius)
        if not intersects:
            continue
        selected.append((entry, distance))
    selected.sort(key=lambda item: item[1])
    cap = max(1, int(limit)) if isinstance(limit, int) else _MAX_TILE_CANDIDATES
    return [entry for entry, _ in selected[:cap]]

# In-process LRU cache for tile RA/DEC/MAG arrays across images.
_TILE_RAW_CACHE: "OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
_TILE_RAW_CACHE_LOCK = threading.Lock()
_TILE_RAW_CACHE_CAP = 128

# Global GPU detect slot controls (shared across images/workers)
_GPU_DETECT_LOCK = threading.Lock()
_GPU_DETECT_SEMAPHORES: dict[int, threading.Semaphore] = {}
_CUDA_READY_CACHE: Optional[tuple[bool, str]] = None

# Shared catalog objects for fallback tile extraction (avoid re-parsing catalogs per image).
_CATALOG_DB_CACHE: dict[tuple[str, str], CatalogDB] = {}
_CATALOG_DB_CACHE_LOCK = threading.Lock()

# Shared ASTAP tile metadata lookup by (db_root, family).
_RAW_TILE_LOOKUP_CACHE: dict[tuple[str, str], dict[tuple[str, str], object]] = {}
_RAW_TILE_LOOKUP_CACHE_LOCK = threading.Lock()

# Common dtype for star detection tuples.
_STAR_DETECT_DTYPE = np.dtype([('x', 'f4'), ('y', 'f4'), ('flux', 'f4')])


def astap_iso_image_for_solve(hdu: fits.PrimaryHDU) -> np.ndarray:
    """Return the native-ADU mono image consumed by ASTAP-ISO strict detection."""
    data = hdu.data
    if data is None:
        raise ValueError("FITS HDU has no data to generate strict ASTAP-ISO image")
    raw_image = np.asarray(data)
    if raw_image.ndim == 3:
        image = np.mean(raw_image, axis=0, dtype=np.float32)
    elif raw_image.ndim == 2:
        image = np.asarray(raw_image, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported FITS data shape for strict ASTAP-ISO image: {raw_image.shape}")
    return np.nan_to_num(image, copy=False)


def _get_cached_catalog_db(db_root: Path, family: str | None) -> CatalogDB:
    root = str(Path(db_root).expanduser().resolve())
    fam_key = (family or "").strip().lower()
    key = (root, fam_key)
    with _CATALOG_DB_CACHE_LOCK:
        cached = _CATALOG_DB_CACHE.get(key)
        if cached is not None:
            return cached
    db = CatalogDB(Path(root), families=[fam_key] if fam_key else None)
    with _CATALOG_DB_CACHE_LOCK:
        _CATALOG_DB_CACHE[key] = db
    return db


def _get_raw_tile_lookup(db_root: Path, family: str | None) -> dict[tuple[str, str], object]:
    root = str(Path(db_root).expanduser().resolve())
    fam_key = (family or "").strip().lower()
    key = (root, fam_key)
    with _RAW_TILE_LOOKUP_CACHE_LOCK:
        cached = _RAW_TILE_LOOKUP_CACHE.get(key)
        if cached is not None:
            return cached

    lookup: dict[tuple[str, str], object] = {}
    for tm in iter_astap_tiles(Path(root)):
        fam = str(tm.family).strip().lower()
        if fam_key and fam != fam_key:
            continue
        lookup[(fam, str(tm.tile_code))] = tm

    with _RAW_TILE_LOOKUP_CACHE_LOCK:
        _RAW_TILE_LOOKUP_CACHE[key] = lookup
    return lookup


def _cuda_runtime_ready() -> tuple[bool, str]:
    global _CUDA_READY_CACHE
    cached = _CUDA_READY_CACHE
    if cached is not None:
        return cached
    try:
        import cupy  # type: ignore
        from cupy.cuda import runtime as _rt  # type: ignore
        from cupy.cuda import nvrtc as _nvrtc  # type: ignore
        n = int(_rt.getDeviceCount())
        if n <= 0:
            _CUDA_READY_CACHE = (False, "no CUDA device detected")
        else:
            # Probes NVRTC availability; this is where missing libnvrtc usually surfaces.
            _nvrtc.getVersion()
            _CUDA_READY_CACHE = (True, f"{n} CUDA device(s)")
    except Exception as exc:
        _CUDA_READY_CACHE = (False, str(exc))
    return _CUDA_READY_CACHE


def _gpu_detect_semaphore(slots: int) -> threading.Semaphore:
    slots = max(1, int(slots))
    with _GPU_DETECT_LOCK:
        sem = _GPU_DETECT_SEMAPHORES.get(slots)
        if sem is None:
            sem = threading.Semaphore(slots)
            _GPU_DETECT_SEMAPHORES[slots] = sem
        return sem


def _detect_stars_astap_cli(fits_path: Path, *, snr_min: int = 10, timeout_s: int = 120) -> np.ndarray:
    """Use ASTAP CLI -extract to get star list (x,y,flux)."""
    cmd = ["astap", "-f", str(fits_path), "-extract", str(int(max(1, snr_min)))]
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )
    except Exception as exc:
        logger.warning("near detect ASTAP extract failed to run: %s", exc)
        return np.zeros(0, dtype=_STAR_DETECT_DTYPE)

    csv_path = fits_path.with_suffix('.csv')
    if cp.returncode != 0 or not csv_path.exists():
        logger.warning("near detect ASTAP extract returned rc=%s (csv missing=%s)", cp.returncode, (not csv_path.exists()))
        return np.zeros(0, dtype=_STAR_DETECT_DTYPE)

    arr: np.ndarray | None = None
    try:
        try:
            if csv_path.stat().st_size <= 8:
                return np.zeros(0, dtype=_STAR_DETECT_DTYPE)
        except Exception:
            pass
        # Vectorized parse is significantly faster than line-by-line Python loops.
        # Columns: x, y, ..., ..., flux
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            arr = np.genfromtxt(
                csv_path,
                delimiter=",",
                skip_header=1,
                usecols=(0, 1, 4),
                dtype=np.float32,
                invalid_raise=False,
            )
    except Exception as exc:
        logger.warning("near detect ASTAP parse failed: %s", exc)
        arr = None
    finally:
        try:
            csv_path.unlink()
        except Exception:
            pass

    if arr is None:
        return np.zeros(0, dtype=_STAR_DETECT_DTYPE)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=_STAR_DETECT_DTYPE)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            return np.zeros(0, dtype=_STAR_DETECT_DTYPE)
        arr = arr.reshape(1, 3)

    finite_mask = np.isfinite(arr).all(axis=1)
    if not finite_mask.all():
        arr = arr[finite_mask]
    if arr.size == 0:
        return np.zeros(0, dtype=_STAR_DETECT_DTYPE)

    out = np.zeros(int(arr.shape[0]), dtype=_STAR_DETECT_DTYPE)
    out['x'] = arr[:, 0]
    out['y'] = arr[:, 1]
    out['flux'] = np.maximum(arr[:, 2], np.float32(1.0))
    return out


def astap_binned_to_full_coords(
    x_binned: np.ndarray | float,
    y_binned: np.ndarray | float,
    factor: int,
    *,
    crop: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Convert ASTAP binned-image coordinates back to full-resolution pixels."""
    factor_f = float(max(1, int(factor)))
    crop_f = float(crop)
    shift_x = (float(width) * (1.0 - crop_f) / 2.0) if width is not None else 0.0
    shift_y = (float(height) * (1.0 - crop_f) / 2.0) if height is not None else 0.0
    offset = (factor_f - 1.0) * 0.5
    return offset + factor_f * x_binned + shift_x, offset + factor_f * y_binned + shift_y


def astap_full_to_binned_coords(
    x_full: np.ndarray | float,
    y_full: np.ndarray | float,
    factor: int,
    *,
    crop: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Inverse of :func:`astap_binned_to_full_coords` for diagnostics/tests."""
    factor_f = float(max(1, int(factor)))
    crop_f = float(crop)
    shift_x = (float(width) * (1.0 - crop_f) / 2.0) if width is not None else 0.0
    shift_y = (float(height) * (1.0 - crop_f) / 2.0) if height is not None else 0.0
    offset = (factor_f - 1.0) * 0.5
    return (x_full - shift_x - offset) / factor_f, (y_full - shift_y - offset) / factor_f


def astap_compatible_mean_bin_image(image: np.ndarray, factor: int, *, crop: float = 1.0) -> tuple[np.ndarray, int]:
    """ASTAP-style mono block-average binning with truncating dimensions."""
    factor = max(1, int(factor))
    crop = min(1.0, max(0.01, float(crop)))
    data = np.asarray(image, dtype=np.float32)
    if data.ndim == 3:
        # Accept either channel-first or channel-last arrays. FITS solve input is
        # normally already mono, but tests/probes exercise color conversion.
        if data.shape[0] in (3, 4) and data.shape[0] < min(data.shape[1], data.shape[2]):
            data = np.mean(data[:3], axis=0, dtype=np.float32)
        else:
            data = np.mean(data[..., :3], axis=-1, dtype=np.float32)
    elif data.ndim != 2:
        data = np.squeeze(data).astype(np.float32, copy=False)
    if data.ndim != 2:
        raise ValueError("ASTAP-compatible binning expects a 2-D mono image or color image")
    if factor <= 1:
        return data.astype(np.float32, copy=False), 1
    h0, w0 = int(data.shape[0]), int(data.shape[1])
    shift_x = int(round(float(w0) * (1.0 - crop) / 2.0))
    shift_y = int(round(float(h0) * (1.0 - crop) / 2.0))
    hb = int(math.trunc(float(crop) * float(h0) / float(factor)))
    wb = int(math.trunc(float(crop) * float(w0) / float(factor)))
    if hb < 1 or wb < 1:
        return data.astype(np.float32, copy=False), 1
    cropped = data[shift_y : shift_y + hb * factor, shift_x : shift_x + wb * factor]
    binned = (
        cropped.reshape(hb, factor, wb, factor)
        .mean(axis=(1, 3), dtype=np.float32)
        .astype(np.float32, copy=False)
    )
    return binned, factor


def _mean_bin_image(image: np.ndarray, factor: int) -> tuple[np.ndarray, float]:
    binned, used = astap_compatible_mean_bin_image(image, factor)
    return binned, float(used)


def choose_astap_compatible_bin_factor(
    *,
    width: int,
    height: int,
    fov_deg: float | None = None,
    scale_arcsec: float | None = None,
    requested: int | None = None,
) -> int:
    """Explicit binning policy for strict ASTAP-ISO diagnostics.

    A user/requested factor wins. Otherwise this mirrors the observed M31
    `-z 2` regime for tall 1080x1920 frames while keeping smaller/S30-style
    control images at native scale.
    """
    if requested is not None and int(requested) > 0:
        return max(1, int(requested))
    major = max(int(width), int(height))
    if major >= 1600:
        return 2
    if scale_arcsec is not None and math.isfinite(float(scale_arcsec)) and float(scale_arcsec) < 2.0 and major >= 1200:
        return 2
    if fov_deg is not None and math.isfinite(float(fov_deg)) and float(fov_deg) >= 1.0 and major >= 1400:
        return 2
    return 1


def astap_background_noise_stats(image: np.ndarray) -> dict[str, float]:
    data = np.asarray(image, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"background": math.nan, "noise": math.nan, "threshold_7sigma": math.nan, "threshold_30sigma": math.nan}
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    noise = float(1.4826 * mad) if mad > 0 else float(np.std(finite))
    return {
        "background": med,
        "noise": noise,
        "threshold_7sigma": med + 7.0 * noise,
        "threshold_30sigma": med + 30.0 * noise,
    }


def _astap_histogram(image: np.ndarray, *, upper: int = 65000) -> tuple[np.ndarray, int]:
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 2:
        data = np.squeeze(data).astype(np.float32, copy=False)
    height, width = int(data.shape[0]), int(data.shape[1])
    offset_w = int(math.trunc(float(width) * 0.042))
    offset_h = int(math.trunc(float(height) * 0.015))
    core = data[offset_h : height - offset_h, offset_w : width - offset_w]
    values = np.rint(core).astype(np.int64, copy=False).ravel()
    mask = (values >= 1) & (values < int(upper))
    values = values[mask]
    hist = np.bincount(values, minlength=int(upper) + 1)
    hist_mean = int(round(float(np.sum(values)) / float(values.size + 1))) if values.size else 0
    return hist.astype(np.int64, copy=False), hist_mean


def estimate_astap_global_background(image: np.ndarray, *, max_stars: int = 500) -> dict[str, float]:
    """Mirror ASTAP CLI ``get_background`` for solver star detection."""
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 2:
        data = np.squeeze(data).astype(np.float32, copy=False)
    if data.ndim != 2 or data.size == 0:
        return {"background": math.nan, "noise": math.nan, "star_level": math.nan, "star_level2": math.nan}
    hist, hist_mean = _astap_histogram(data)
    background = float(data[0, 0])
    if hist_mean <= 0:
        background = 0.0
    else:
        peak_slice = hist[1 : hist_mean + 1]
        if peak_slice.size:
            background = float(int(np.argmax(peak_slice)) + 1)
        if float(hist_mean) > 1.5 * background:
            background = float(hist_mean)

    height, width = int(data.shape[0]), int(data.shape[1])
    step_size = int(round(float(height) / 71.0))
    if step_size % 2 == 0:
        step_size += 1
    sd = 99999.0
    iterations = 0
    while True:
        sd_old = float(sd)
        total = 0.0
        counter = 0
        for fits_x in range(15, max(15, width - 15), step_size):
            for fits_y in range(15, max(15, height - 15), step_size):
                value = float(data[fits_y, fits_x])
                if value < background * 2.0 and value != 0.0:
                    if iterations == 0 or abs(value - background) <= 3.0 * sd_old:
                        total += (value - background) * (value - background)
                        counter += 1
        sd = math.sqrt(total / float(counter)) if counter else 0.0
        iterations += 1
        if (sd_old - sd) < 0.05 * sd or iterations >= 7:
            break

    factor = 6 * int(max_stars)
    factor2 = 24 * int(max_stars)
    above = 0
    star_level = 0
    star_level2 = 0
    i = 65001
    while star_level == 0 and i > background + 1.0:
        i -= 1
        if i < hist.size:
            above += int(hist[i])
        if above >= factor:
            star_level = i
    while star_level2 == 0 and i > background + 1.0:
        i -= 1
        if i < hist.size:
            above += int(hist[i])
        if above >= factor2:
            star_level2 = i
    star_level_f = max(max(3.5 * sd, 1.0), float(star_level) - background - 1.0)
    star_level2_f = max(max(3.5 * sd, 1.0), float(star_level2) - background - 1.0)
    return {
        "background": float(background),
        "noise": float(sd),
        "star_level": float(star_level_f),
        "star_level2": float(star_level2_f),
        "iterations": int(iterations),
        "hist_mean": int(hist_mean),
    }


def _astap_sigma_clipped_mean_from_histogram(
    image: np.ndarray,
    start_x: int,
    end_x: int,
    start_y: int,
    end_y: int,
    upper_limit: int,
    *,
    max_iterations: int = 6,
    convergence_threshold: float = 0.1,
) -> tuple[float, float, int]:
    data = np.asarray(image, dtype=np.float32)
    upper = max(1, int(upper_limit))
    sub = data[int(start_y) : int(end_y) + 1, int(start_x) : int(end_x) + 1]
    values = np.rint(sub).astype(np.int64, copy=False).ravel()
    values = values[(values >= 1) & (values < upper)]
    if values.size == 0:
        return 0.0, 0.0, 0
    hist = np.bincount(values, minlength=upper + 1)
    lower = 0
    current_upper = upper
    mean_v = 0.0
    stdev = 0.0
    iterations = 0
    for iteration in range(max(1, int(max_iterations))):
        previous_mean = mean_v
        previous_stdev = stdev
        bins = np.arange(lower, current_upper + 1, dtype=np.float64)
        counts = hist[lower : current_upper + 1].astype(np.float64, copy=False)
        total_count = float(np.sum(counts))
        if total_count <= 0.0:
            return 0.0, 0.0, int(iterations)
        total = float(np.sum(bins * counts))
        total_sq = float(np.sum(bins * bins * counts))
        mean_v = total / total_count
        if total_count > 1.0:
            variance = (total_sq - (total * total) / total_count) / (total_count - 1.0)
            stdev = math.sqrt(variance) if variance > 0.0 else 0.0
        else:
            stdev = 0.0
        iterations = iteration + 1
        if stdev > 0.0:
            lower = 0
            current_upper = min(upper, int(round(mean_v + 2.0 * stdev)))
        if (
            iteration > 0
            and abs(mean_v - previous_mean) < float(convergence_threshold)
            and abs(stdev - previous_stdev) < float(convergence_threshold)
        ):
            break
    return float(mean_v), float(stdev), int(iterations)


def astap_section_grid(width: int, height: int, *, rastersteps: int = 12) -> list[tuple[int, int, int, int, int]]:
    if int(height) < int(width):
        steps_x = int(rastersteps)
        steps_y = int(round(float(rastersteps) * float(height) / max(1.0, float(width))))
    else:
        steps_y = int(rastersteps)
        steps_x = int(round(float(rastersteps) * float(width) / max(1.0, float(height))))
    sections: list[tuple[int, int, int, int, int]] = []
    section_id = 0
    for yy in range(steps_y + 1):
        for xx in range(steps_x + 1):
            start_x = 1 + int(round(float(width) * float(xx) / float(steps_x + 1)))
            end_x = min(int(width) - 2, int(round(float(width) * float(xx + 1) / float(steps_x + 1))))
            start_y = 1 + int(round(float(height) * float(yy) / float(steps_y + 1)))
            end_y = min(int(height) - 2, int(round(float(height) * float(yy + 1) / float(steps_y + 1))))
            sections.append((section_id, start_x, end_x, start_y, end_y))
            section_id += 1
    return sections


def _astap_bilinear(image: np.ndarray, x: float, y: float) -> float:
    width = int(image.shape[1])
    height = int(image.shape[0])
    x_trunc = int(math.trunc(float(x)))
    y_trunc = int(math.trunc(float(y)))
    if x_trunc <= 0 or x_trunc >= width - 2 or y_trunc <= 0 or y_trunc >= height - 2:
        return 0.0
    x_frac = float(x) - float(x_trunc)
    y_frac = float(y) - float(y_trunc)
    return (
        float(image[y_trunc, x_trunc]) * (1.0 - x_frac) * (1.0 - y_frac)
        + float(image[y_trunc, x_trunc + 1]) * x_frac * (1.0 - y_frac)
        + float(image[y_trunc + 1, x_trunc]) * (1.0 - x_frac) * y_frac
        + float(image[y_trunc + 1, x_trunc + 1]) * x_frac * y_frac
    )


def _astap_hfd_measure(image: np.ndarray, x1: int, y1: int, *, rs: int = 14) -> dict[str, float] | None:
    data = np.asarray(image, dtype=np.float32)
    width = int(data.shape[1])
    height = int(data.shape[0])
    rs_i = int(rs)
    r2 = rs_i + 1
    if x1 - r2 <= 0 or x1 + r2 >= width - 1 or y1 - r2 <= 0 or y1 + r2 >= height - 1:
        return None
    bg_values: list[float] = []
    for i in range(-r2, r2 + 1):
        for j in range(-r2, r2 + 1):
            distance = i * i + j * j
            if distance > rs_i * rs_i and distance <= r2 * r2:
                bg_values.append(float(data[y1 + j, x1 + i]))
    if not bg_values:
        return None
    bg_arr = np.asarray(bg_values, dtype=np.float64)
    star_bg = float(np.median(bg_arr))
    mad_bg = float(np.median(np.abs(bg_arr - star_bg)))
    sd_bg = max(mad_bg * 1.4826, 1.0)

    cur_rs = rs_i
    while True:
        sum_val = 0.0
        sum_val_x = 0.0
        sum_val_y = 0.0
        signal_counter = 0
        for i in range(-cur_rs, cur_rs + 1):
            for j in range(-cur_rs, cur_rs + 1):
                val = float(data[y1 + j, x1 + i]) - star_bg
                if val > 3.0 * sd_bg:
                    sum_val += val
                    sum_val_x += val * float(i)
                    sum_val_y += val * float(j)
                    signal_counter += 1
        if sum_val <= 12.0 * sd_bg or signal_counter <= 1:
            return None
        xc = float(x1) + sum_val_x / sum_val
        yc = float(y1) + sum_val_y / sum_val
        if xc - cur_rs < 0 or xc + cur_rs > width - 1 or yc - cur_rs < 0 or yc + cur_rs > height - 1:
            return None
        boxed = float(signal_counter) >= (2.0 / 9.0) * float((cur_rs + cur_rs + 1) ** 2)
        if not boxed:
            cur_rs -= 2 if cur_rs > 4 else 1
        if boxed or cur_rs <= 1:
            break

    cur_rs += 2
    distance_histogram = [0] * (cur_rs + 1)
    valmax = 0.0
    for i in range(-cur_rs, cur_rs + 1):
        for j in range(-cur_rs, cur_rs + 1):
            distance = int(round(math.sqrt(float(i * i + j * j))))
            if distance <= cur_rs:
                val = _astap_bilinear(data, xc + float(i), yc + float(j)) - star_bg
                if val > 3.0 * sd_bg:
                    distance_histogram[distance] += 1
                    if val > valmax:
                        valmax = val
    r_aperture = -1
    distance_top_value = 0
    hist_start = False
    illuminated_pixels = 0
    while True:
        r_aperture += 1
        illuminated_pixels += distance_histogram[r_aperture]
        if distance_histogram[r_aperture] > 0:
            hist_start = True
        if distance_top_value < distance_histogram[r_aperture]:
            distance_top_value = distance_histogram[r_aperture]
        if r_aperture >= cur_rs or (hist_start and distance_histogram[r_aperture] <= 0.1 * float(distance_top_value)):
            break
    if r_aperture >= cur_rs:
        return None
    if r_aperture > 2 and illuminated_pixels < 0.35 * float((r_aperture + r_aperture - 2) ** 2):
        return None

    sum_val = 0.0
    sum_val_r = 0.0
    pixel_counter = 0
    for i in range(-r_aperture, r_aperture + 1):
        for j in range(-r_aperture, r_aperture + 1):
            val = _astap_bilinear(data, xc + float(i), yc + float(j)) - star_bg
            radius = math.sqrt(float(i * i + j * j))
            sum_val += val
            sum_val_r += val * radius
            if val >= valmax * 0.5:
                pixel_counter += 1
    flux = max(sum_val, 0.00001)
    hfd1 = max(0.7, 2.0 * sum_val_r / flux)
    fwhm = 2.0 * math.sqrt(float(pixel_counter) / math.pi)
    snr = flux / math.sqrt(flux + float(r_aperture * r_aperture) * math.pi * sd_bg * sd_bg)
    return {
        "x": float(xc),
        "y": float(yc),
        "flux": float(flux),
        "snr": float(snr),
        "hfd": float(hfd1),
        "fwhm": float(fwhm),
        "background": float(star_bg),
        "noise": float(sd_bg),
        "aperture": float(r_aperture),
    }


def _astap_find_stars_routine(
    image: np.ndarray,
    *,
    background: float,
    noise: float,
    detection_level: float,
    retry_marker: int,
    start_x: int,
    end_x: int,
    start_y: int,
    end_y: int,
    hfd_min: float,
    img_sa: np.ndarray,
) -> tuple[list[dict[str, float]], float]:
    data = np.asarray(image, dtype=np.float32)
    height, width = int(data.shape[0]), int(data.shape[1])
    stars: list[dict[str, float]] = []
    highest_snr = 0.0
    for fits_y in range(int(start_y), int(end_y) + 1):
        for fits_x in range(int(start_x), int(end_x) + 1):
            if img_sa[fits_y, fits_x] == int(retry_marker):
                continue
            if float(data[fits_y, fits_x]) - float(background) <= float(detection_level):
                continue
            starpixels = 0
            if float(data[fits_y, fits_x - 1]) - float(background) > 4.0 * float(noise):
                starpixels += 1
            if float(data[fits_y, fits_x + 1]) - float(background) > 4.0 * float(noise):
                starpixels += 1
            if float(data[fits_y - 1, fits_x]) - float(background) > 4.0 * float(noise):
                starpixels += 1
            if float(data[fits_y + 1, fits_x]) - float(background) > 4.0 * float(noise):
                starpixels += 1
            if starpixels < 2:
                continue
            measured = _astap_hfd_measure(data, fits_x, fits_y, rs=14)
            if measured is None:
                continue
            hfd1 = float(measured["hfd"])
            snr = float(measured["snr"])
            xc = float(measured["x"])
            yc = float(measured["y"])
            xci = int(round(xc))
            yci = int(round(yc))
            if not (0 <= xci < width and 0 <= yci < height):
                continue
            if hfd1 <= 30.0 and snr > 10.0 and hfd1 > float(hfd_min) and img_sa[yci, xci] != int(retry_marker):
                radius = int(round(3.0 * hfd1))
                sqr_radius = radius * radius
                for n in range(-radius, radius + 1):
                    yy = yci + n
                    if yy < 0 or yy >= height:
                        continue
                    for m in range(-radius, radius + 1):
                        xx = xci + m
                        if xx >= 0 and xx < width and (m * m + n * n) <= sqr_radius:
                            img_sa[yy, xx] = int(retry_marker)
                measured["scan_x"] = float(fits_x)
                measured["scan_y"] = float(fits_y)
                stars.append(measured)
                if snr > highest_snr:
                    highest_snr = snr
    return stars, float(highest_snr)


def _astap_select_brightest_stars(stars: list[dict[str, float]], max_stars: int, highest_snr: float) -> list[dict[str, float]]:
    if int(max_stars) <= 0 or len(stars) <= int(max_stars):
        return list(stars)
    snr_values = np.asarray([float(s.get("snr", 0.0)) for s in stars], dtype=np.float64)
    mask = astap_sqrt_snr_selection_mask(snr_values, int(max_stars))
    return [star for star, keep in zip(stars, mask) if bool(keep)]


def astap_adaptive_image_detection(
    image: np.ndarray,
    *,
    bin_factor: int = 2,
    max_stars: int = 500,
    hfd_min: float = 0.8,
    crop: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reproduce ASTAP CLI solver star detection in strict diagnostics."""
    binned, used = astap_compatible_mean_bin_image(image, bin_factor, crop=crop)
    data = np.asarray(binned, dtype=np.float32)
    height, width = int(data.shape[0]), int(data.shape[1])
    global_stats = estimate_astap_global_background(data, max_stars=int(max_stars))
    background = float(global_stats["background"])
    noise = float(global_stats["noise"])
    passes: list[dict[str, object]] = []
    final_stars: list[dict[str, float]] = []
    highest_snr = 0.0
    final_retry = 0

    retries = 4
    pass_id = 0
    while True:
        highest_snr = 0.0
        stars: list[dict[str, float]] = []
        img_sa = np.zeros((height, width), dtype=np.int16)
        if retries == 4:
            pass_id += 1
            if float(global_stats["star_level"]) > 30.0 * noise:
                detection_level = float(global_stats["star_level"])
                stars, highest_snr = _astap_find_stars_routine(
                    data,
                    background=background,
                    noise=noise,
                    detection_level=detection_level,
                    retry_marker=4,
                    start_x=1,
                    end_x=width - 2,
                    start_y=1,
                    end_y=height - 2,
                    hfd_min=float(hfd_min),
                    img_sa=img_sa,
                )
                passes.append({"pass_id": pass_id, "retry_index": 4, "threshold": detection_level, "background": background, "noise": noise, "candidate_count": len(stars), "reason": "star_level"})
            else:
                passes.append({"pass_id": pass_id, "retry_index": 4, "threshold": 0.0, "background": background, "noise": noise, "candidate_count": 0, "reason": "skipped_star_level"})
                retries = 3
        if retries == 3:
            pass_id += 1
            if float(global_stats["star_level2"]) > 30.0 * noise:
                detection_level = float(global_stats["star_level2"])
                stars, highest_snr = _astap_find_stars_routine(
                    data,
                    background=background,
                    noise=noise,
                    detection_level=detection_level,
                    retry_marker=3,
                    start_x=1,
                    end_x=width - 2,
                    start_y=1,
                    end_y=height - 2,
                    hfd_min=float(hfd_min),
                    img_sa=img_sa,
                )
                passes.append({"pass_id": pass_id, "retry_index": 3, "threshold": detection_level, "background": background, "noise": noise, "candidate_count": len(stars), "reason": "star_level2"})
            else:
                passes.append({"pass_id": pass_id, "retry_index": 3, "threshold": 0.0, "background": background, "noise": noise, "candidate_count": 0, "reason": "skipped_star_level2"})
                retries = 2
        if retries == 2:
            pass_id += 1
            detection_level = 30.0 * noise
            stars, highest_snr = _astap_find_stars_routine(
                data,
                background=background,
                noise=noise,
                detection_level=detection_level,
                retry_marker=2,
                start_x=1,
                end_x=width - 2,
                start_y=1,
                end_y=height - 2,
                hfd_min=float(hfd_min),
                img_sa=img_sa,
            )
            passes.append({"pass_id": pass_id, "retry_index": 2, "threshold": detection_level, "background": background, "noise": noise, "candidate_count": len(stars), "reason": "thirty_sigma"})
        if retries == 1:
            pass_id += 1
            img_sa = np.zeros((height, width), dtype=np.int16)
            stars = []
            highest_snr = 0.0
            sections: list[dict[str, object]] = []
            upper_limit = max(65500, int(math.trunc(background * 2.0)))
            for section_id, start_x, end_x, start_y, end_y in astap_section_grid(width, height):
                local_bg, local_noise, iterations = _astap_sigma_clipped_mean_from_histogram(
                    data,
                    start_x,
                    end_x,
                    start_y,
                    end_y,
                    upper_limit,
                    max_iterations=6,
                    convergence_threshold=0.1,
                )
                detection_level = 7.0 * local_noise
                before = len(stars)
                found, hi = _astap_find_stars_routine(
                    data,
                    background=local_bg,
                    noise=local_noise,
                    detection_level=detection_level,
                    retry_marker=1,
                    start_x=start_x,
                    end_x=end_x,
                    start_y=start_y,
                    end_y=end_y,
                    hfd_min=float(hfd_min),
                    img_sa=img_sa,
                )
                stars.extend(found)
                if hi > highest_snr:
                    highest_snr = hi
                sections.append(
                    {
                        "section_id": int(section_id),
                        "x0": int(start_x),
                        "x1": int(end_x),
                        "y0": int(start_y),
                        "y1": int(end_y),
                        "background": float(local_bg),
                        "noise": float(local_noise),
                        "threshold_delta": float(detection_level),
                        "candidate_count": int(len(stars) - before),
                        "iterations": int(iterations),
                    }
                )
            passes.append({"pass_id": pass_id, "retry_index": 1, "threshold": 7.0 * local_noise, "background": float(local_bg), "noise": float(local_noise), "candidate_count": len(stars), "reason": "section_sigma_clip", "sections": sections})
        final_stars = stars
        final_retry = retries
        retries -= 1
        if len(final_stars) >= int(max_stars) or retries <= 0:
            break

    selected = _astap_select_brightest_stars(final_stars, int(max_stars), highest_snr)
    out = np.zeros(len(selected), dtype=_STAR_DETECT_DTYPE)
    for idx, star in enumerate(selected):
        x_full, y_full = astap_binned_to_full_coords(float(star["x"]), float(star["y"]), int(used), crop=crop)
        out["x"][idx] = np.float32(x_full)
        out["y"][idx] = np.float32(y_full)
        out["flux"][idx] = np.float32(max(1.0e-5, float(star.get("flux", 1.0))))
    diag: dict[str, object] = {
        "bin_factor": int(used),
        "binned_shape": [int(height), int(width)],
        "global_background": global_stats,
        "passes": passes,
        "final_retry": int(final_retry),
        "raw_candidates": int(len(final_stars)),
        "selected_count": int(out.size),
        "max_stars": int(max_stars),
        "hfd_min": float(hfd_min),
    }
    return out, diag


def astap_sqrt_snr_selection_mask(scores: np.ndarray, nr_stars_required: int) -> np.ndarray:
    """ASTAP get_brightest_stars-style threshold mask preserving input order."""
    values = np.asarray(scores, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    required = max(0, int(nr_stars_required))
    if required <= 0 or n <= required:
        return np.ones(n, dtype=bool)
    highest = float(np.nanmax(values))
    if not math.isfinite(highest) or highest <= 0:
        order = np.argsort(values, kind="stable")[::-1][:required]
        mask = np.zeros(n, dtype=bool)
        mask[order] = True
        return mask
    range_hi = 199
    sqrt_range = math.sqrt(highest)
    scaled = np.trunc(np.sqrt(np.maximum(values, 0.0)) * range_hi / max(sqrt_range, 1e-12)).astype(np.int64)
    scaled = np.clip(scaled, 0, range_hi)
    hist = np.bincount(scaled, minlength=range_hi + 1)
    count = 0
    i = range_hi
    while True:
        i -= 1
        count += int(hist[i])
        if i <= 0 or count >= required:
            break
    overshoot = ((count - required) / float(hist[i])) if int(hist[i]) != 0 else 0.0
    snr_required = (sqrt_range * (float(i) + overshoot) / float(range_hi)) ** 2
    return values >= snr_required


def astap_compatible_image_detection(
    image: np.ndarray,
    *,
    bin_factor: int = 2,
    max_stars: int = 500,
    k_sigma: float = 3.0,
    min_area: int = 4,
) -> tuple[np.ndarray, dict[str, object]]:
    binned, used = astap_compatible_mean_bin_image(image, bin_factor)
    stats = astap_background_noise_stats(binned)
    stars = detect_stars(
        binned,
        backend="cpu",
        mode="global",
        k_sigma=float(k_sigma),
        min_area=int(min_area),
        max_labels=max(5000, int(max_stars) * 4),
    )
    if stars.size > 0 and used > 1:
        stars = stars.copy()
        x_full, y_full = astap_binned_to_full_coords(stars["x"], stars["y"], int(used))
        stars["x"] = np.asarray(x_full, dtype=np.float32)
        stars["y"] = np.asarray(y_full, dtype=np.float32)
    if stars.size > 0 and int(max_stars) > 0:
        mask = astap_sqrt_snr_selection_mask(np.asarray(stars["flux"], dtype=np.float64), int(max_stars))
        stars = stars[mask]
    diag = {
        "bin_factor": int(used),
        "binned_shape": [int(binned.shape[0]), int(binned.shape[1])],
        "background": stats,
        "detected_count": int(stars.size),
        "k_sigma": float(k_sigma),
        "min_area": int(min_area),
    }
    out = np.zeros(int(stars.size), dtype=_STAR_DETECT_DTYPE)
    if stars.size > 0:
        out["x"] = np.asarray(stars["x"], dtype=np.float32)
        out["y"] = np.asarray(stars["y"], dtype=np.float32)
        out["flux"] = np.asarray(stars["flux"], dtype=np.float32)
    return out, diag


def _detect_stars_astap_cli_binned(
    image: np.ndarray,
    source_fits_path: Path,
    *,
    bin_factor: int,
    snr_min: int = 10,
    timeout_s: int = 120,
) -> tuple[np.ndarray, float]:
    binned, scale = _mean_bin_image(image, int(bin_factor))
    if scale <= 1.0:
        return _detect_stars_astap_cli(source_fits_path, snr_min=snr_min, timeout_s=timeout_s), 1.0

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".zenear_astap_bin{int(scale)}_",
            suffix=".fit",
            dir=str(source_fits_path.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        fits.writeto(tmp_path, binned, overwrite=True)
        stars = _detect_stars_astap_cli(tmp_path, snr_min=snr_min, timeout_s=timeout_s)
    except Exception as exc:
        logger.info("near detect ASTAP binned extract unavailable (%s), fallback full-res", exc)
        return _detect_stars_astap_cli(source_fits_path, snr_min=snr_min, timeout_s=timeout_s), 1.0
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass
            try:
                tmp_path.with_suffix('.csv').unlink()
            except Exception:
                pass

    if stars.size > 0:
        stars = stars.copy()
        x_full, y_full = astap_binned_to_full_coords(stars['x'], stars['y'], int(scale))
        stars['x'] = np.asarray(x_full, dtype=np.float32)
        stars['y'] = np.asarray(y_full, dtype=np.float32)
    return stars, scale


def _tile_path_from_entry(index_root: Path, entry: dict) -> Path:
    rel = str(entry.get("tile_file", "")).replace("\\", "/")
    return index_root / rel


def _get_tile_raw_arrays(
    index_root: Path,
    entry: dict,
    *,
    db_root: Path | None,
    cache_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _TILE_RAW_CACHE_CAP
    if cache_cap and cache_cap != _TILE_RAW_CACHE_CAP:
        _TILE_RAW_CACHE_CAP = max(1, int(cache_cap))
    tile_path = _tile_path_from_entry(index_root, entry)
    key = str(tile_path.resolve())
    with _TILE_RAW_CACHE_LOCK:
        cached = _TILE_RAW_CACHE.get(key)
        if cached is not None:
            _TILE_RAW_CACHE.pop(key)
            _TILE_RAW_CACHE[key] = cached
            return cached
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
    if (ra is None or dec is None or mag is None or ra.size == 0 or dec.size == 0) and db_root is not None:
        try:
            fam = str(entry.get("family") or "").strip().lower() or None
            db = _get_cached_catalog_db(Path(db_root), fam)
            target_code = str(entry.get("tile_code") or "")
            for tile in db.tiles:
                if tile.tile_code == target_code and (fam is None or tile.spec.key == fam):
                    block = db._load_tile(tile)
                    stars = block.stars
                    ra = stars["ra_deg"].astype(np.float64, copy=False)
                    dec = stars["dec_deg"].astype(np.float64, copy=False)
                    mag = stars["mag"].astype(np.float32, copy=False)
                    break
            # Persist fallback extract to NPZ so next runs hit the fast path.
            if ra is not None and dec is not None and mag is not None and ra.size > 0 and dec.size > 0:
                try:
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    if not tile_path.exists():
                        np.savez_compressed(tile_path, ra_deg=ra, dec_deg=dec, mag=mag)
                except Exception:
                    pass
        except Exception:
            ra = dec = mag = None
    if ra is None or dec is None or mag is None:
        raise FileNotFoundError(tile_path)
    with _TILE_RAW_CACHE_LOCK:
        _TILE_RAW_CACHE[key] = (ra, dec, mag)
        while len(_TILE_RAW_CACHE) > _TILE_RAW_CACHE_CAP:
            _TILE_RAW_CACHE.popitem(last=False)
        return _TILE_RAW_CACHE[key]


def _extract_angle(header: fits.Header, keys: Iterable[str], *, is_ra: bool) -> Optional[float]:
    for key in keys:
        if key not in header:
            continue
        value = parse_angle(header.get(key), is_ra=is_ra)
        if value is not None:
            return value
    return None


def _extract_near_center_angle(header: fits.Header, keys: Iterable[str], *, is_ra: bool, strict_astap_iso: bool) -> Optional[float]:
    """Extract Near solve center with strict ASTAP-ISO FITS-key semantics.

    `parse_angle(..., is_ra=True)` treats any numeric RA <= 24 as hours.  The
    Seestar runtime copies used by the ASTAP parity probes store the plain `RA`
    keyword as decimal degrees, while ASTAP CLI receives that value converted to
    hours by the caller.  In strict ASTAP-ISO mode, keep numeric `RA` as degrees
    and leave textual hour-angle forms (OBJCTRA, "hh:mm:ss") to parse_angle.
    """
    for key in keys:
        if key not in header:
            continue
        raw = header.get(key)
        if strict_astap_iso and is_ra and str(key).strip().upper() == "RA":
            try:
                value = float(raw)
            except Exception:
                value = parse_angle(raw, is_ra=True)
        else:
            value = parse_angle(raw, is_ra=is_ra)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def _extract_float(header: fits.Header, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key not in header:
            continue
        try:
            value = float(header.get(key))
        except Exception:
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _load_tile_catalog(
    index_root: Path,
    entry: dict,
    ra0: float,
    dec0: float,
    *,
    db_root: Path | None = None,
    cache_cap: int = _TILE_RAW_CACHE_CAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra, dec, mag = _get_tile_raw_arrays(index_root, entry, db_root=db_root, cache_cap=cache_cap)
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


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    # points: (4,2)
    d = []
    for i in range(4):
        for j in range(i + 1, 4):
            d.append(float(np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])))
    return np.asarray(d, dtype=np.float64)


def _quad_signature(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    if points.shape != (4, 2):
        return None
    d = _pairwise_distances(points)
    dmax = float(np.max(d)) if d.size else 0.0
    if not np.isfinite(dmax) or dmax <= 1e-9:
        return None
    ratios = np.sort((d / dmax).astype(np.float64))
    # ratios has 6 elems including 1.0 at end; keep first 5 as invariant signature.
    ratios = ratios[:5]
    c = np.mean(points, axis=0).astype(np.float64)
    r = np.hypot(points[:, 0] - c[0], points[:, 1] - c[1])
    radial_order = np.argsort(r)
    return ratios, radial_order, dmax, c


def _build_quads(points: np.ndarray, ranked_indices: np.ndarray, *, neighbors: int = 5) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    # Returns list of (signature[5], quad_indices[4], radial_order[4], dmax, center_xy[2]).
    if points.size == 0 or ranked_indices.size < 8:
        return []
    ranked = np.asarray(ranked_indices, dtype=np.int32)
    subset = points[ranked]
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []
    for local_anchor, anchor_global in enumerate(ranked):
        p0 = points[anchor_global]
        delta = subset - p0[None, :]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        nn = np.argsort(dist)
        # skip self at nn[0]
        nn = [int(i) for i in nn[1 : 1 + max(3, neighbors)]]
        if len(nn) < 3:
            continue
        for c in combinations(nn, 3):
            quad_local = np.array([local_anchor, c[0], c[1], c[2]], dtype=np.int32)
            quad_global = ranked[quad_local]
            qpts = points[quad_global]
            sig = _quad_signature(qpts)
            if sig is None:
                continue
            ratios, radial, dmax, center = sig
            out.append((ratios, quad_global, radial.astype(np.int32), float(dmax), center.astype(np.float64)))
    return out


def _derive_similarity_local(src: np.ndarray, dst: np.ndarray, *, reflected: bool = False) -> tuple[complex, complex] | None:
    if src.shape[0] < 2 or dst.shape[0] < 2:
        return None
    src_c = src[:, 0] + 1j * src[:, 1]
    if reflected:
        src_c = np.conj(src_c)
    dst_c = dst[:, 0] + 1j * dst[:, 1]
    src_mean = np.mean(src_c)
    dst_mean = np.mean(dst_c)
    src_zero = src_c - src_mean
    dst_zero = dst_c - dst_mean
    denom = np.sum(np.abs(src_zero) ** 2)
    if denom < 1e-12:
        return None
    rot_scale = np.sum(dst_zero * np.conj(src_zero)) / denom
    translation = dst_mean - rot_scale * src_mean
    return rot_scale, translation


def _similarity_from_affine(matrix: np.ndarray, offset: np.ndarray) -> SimilarityTransform | None:
    """Project a 2x2 affine matrix to the closest similarity transform."""
    if matrix.shape != (2, 2) or offset.shape != (2,):
        return None
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
        return None
    try:
        u, s, vt = np.linalg.svd(matrix)
    except Exception:
        return None
    if s.size < 2:
        return None
    r = u @ vt
    parity = 1
    if np.linalg.det(r) < 0:
        parity = -1
        # enforce proper rotation for angle extraction
        u[:, -1] *= -1
        s[-1] *= -1
        r = u @ vt
    scale = float(np.mean(np.abs(s[:2])))
    if not np.isfinite(scale) or scale <= 0:
        return None
    rotation = float(np.arctan2(r[1, 0], r[0, 0]))
    return SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=(float(offset[0]), float(offset[1])),
        parity=int(parity),
    )


def _match_quads_by_hash(
    img_quads: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    cat_quads: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    *,
    tol: float,
    neighbor_bins: int = 1,
) -> list[tuple[int, int, float]]:
    """ASTAP-like quad matching using hash bins on first ratio component."""
    n1 = len(img_quads)
    n2 = len(cat_quads)
    if n1 == 0 or n2 == 0:
        return []
    hash_bins = max(1, 2 * max(n1, n2))

    def _bin(v: float) -> int:
        return int(v / max(tol, 1e-6)) % hash_bins

    bins_img: list[list[int]] = [[] for _ in range(hash_bins)]
    bins_cat: list[list[int]] = [[] for _ in range(hash_bins)]

    for i, (sig, *_rest) in enumerate(img_quads):
        bins_img[_bin(float(sig[0]))].append(i)
    for j, (sig, *_rest) in enumerate(cat_quads):
        bins_cat[_bin(float(sig[0]))].append(j)

    matches: list[tuple[int, int, float]] = []
    for b in range(hash_bins):
        if not bins_img[b]:
            continue
        for db in range(-neighbor_bins, neighbor_bins + 1):
            bb = (b + db) % hash_bins
            if not bins_cat[bb]:
                continue
            for ii in bins_img[b]:
                sig_i = img_quads[ii][0]
                for jj in bins_cat[bb]:
                    sig_c = cat_quads[jj][0]
                    d = np.abs(sig_i - sig_c)
                    if np.max(d) <= tol:
                        matches.append((ii, jj, float(np.max(d))))

    # Best matches first
    matches.sort(key=lambda t: t[2])
    return matches


def _find_quad_hypothesis(
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    img_ranks: np.ndarray,
    cat_ranks: np.ndarray,
    img_pairs: np.ndarray,
    cat_pairs: np.ndarray,
    *,
    min_scale: float | None,
    max_scale: float | None,
    pixel_tolerance: float,
    allow_reflection: bool,
    cancel_check: Callable[[], bool] | None = None,
) -> SimilarityTransform | None:
    if image_positions.shape[0] < 10 or catalog_positions.shape[0] < 20:
        return None

    # Keep more stars for ASTAP-like quad matching.
    img_order = np.argsort(img_ranks)[: min(260, image_positions.shape[0])]
    cat_order = np.argsort(cat_ranks)[: min(900, catalog_positions.shape[0])]

    neighbors = 6 if img_order.size < 30 else 5
    img_quads = _build_quads(image_positions, img_order, neighbors=neighbors)
    cat_quads = _build_quads(catalog_positions, cat_order, neighbors=neighbors)
    if not img_quads or not cat_quads:
        return None

    best_transform: SimilarityTransform | None = None
    best_inliers = 0
    best_rms = float('inf')

    # ASTAP-style tolerance escalation.
    for tol in (0.02, 0.03, 0.04, 0.06):
        if cancel_check and cancel_check():
            return None

        quad_matches = _match_quads_by_hash(img_quads, cat_quads, tol=tol, neighbor_bins=1)
        if len(quad_matches) < 6:
            continue

        center_matches_src: list[np.ndarray] = []
        center_matches_dst: list[np.ndarray] = []
        ratio_scales: list[float] = []

        for ii, jj, _dist in quad_matches[:3000]:
            _sig_i, _img_idx4, _img_radial, img_dmax, img_center = img_quads[int(ii)]
            _sig_c, _cat_idx4, _cat_radial, cat_dmax, cat_center = cat_quads[int(jj)]
            if img_dmax <= 1e-9:
                continue
            ratio = float(cat_dmax / img_dmax)
            if not np.isfinite(ratio) or ratio <= 0:
                continue
            center_matches_src.append(np.asarray(img_center, dtype=np.float64))
            center_matches_dst.append(np.asarray(cat_center, dtype=np.float64))
            ratio_scales.append(ratio)

        if len(center_matches_src) < 6:
            continue

        ratios = np.asarray(ratio_scales, dtype=np.float64)
        median_ratio = float(np.median(ratios))
        if not np.isfinite(median_ratio) or median_ratio <= 0:
            continue

        keep = np.abs(ratios - median_ratio) <= (tol * median_ratio)
        if int(np.count_nonzero(keep)) < 4:
            continue

        src_centers = np.stack(center_matches_src, axis=0)[keep]
        dst_centers = np.stack(center_matches_dst, axis=0)[keep]
        if src_centers.shape[0] < 4:
            continue

        design = np.column_stack((src_centers[:, 0], src_centers[:, 1], np.ones(src_centers.shape[0], dtype=np.float64)))
        try:
            px, *_ = np.linalg.lstsq(design, dst_centers[:, 0], rcond=None)
            py, *_ = np.linalg.lstsq(design, dst_centers[:, 1], rcond=None)
        except Exception:
            continue

        matrix = np.array([[float(px[0]), float(px[1])], [float(py[0]), float(py[1])]], dtype=np.float64)
        offset = np.array([float(px[2]), float(py[2])], dtype=np.float64)

        transform = _similarity_from_affine(matrix, offset)
        if transform is None:
            continue

        # Honor reflection policy and scale bounds.
        if (not allow_reflection) and int(getattr(transform, 'parity', 1)) < 0:
            continue
        scale = float(transform.scale)
        if min_scale is not None and scale < float(min_scale):
            continue
        if max_scale is not None and scale > float(max_scale):
            continue

        _matches_tmp, inliers, rms = _build_matches_from_transform(
            transform,
            image_positions,
            catalog_positions,
            catalog_positions.astype(np.float64, copy=False),
            pixel_tolerance=pixel_tolerance,
            max_matches=500,
        )
        if inliers <= 0:
            continue
        if inliers > best_inliers or (inliers == best_inliers and rms < best_rms):
            best_inliers = inliers
            best_rms = rms
            best_transform = transform

        # Early stop if we already have strong consensus.
        if best_inliers >= 24 and best_rms <= 1.5:
            break

    return best_transform


def _build_quad_votes(
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    img_ranks: np.ndarray,
    cat_ranks: np.ndarray,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> list[tuple[int, int, float]]:
    if image_positions.shape[0] < 10 or catalog_positions.shape[0] < 20:
        return []

    img_order = np.argsort(img_ranks)[: min(120, image_positions.shape[0])]
    cat_order = np.argsort(cat_ranks)[: min(220, catalog_positions.shape[0])]

    img_quads = _build_quads(image_positions, img_order, neighbors=5)
    cat_quads = _build_quads(catalog_positions, cat_order, neighbors=5)
    if not img_quads or not cat_quads:
        return []

    votes: dict[tuple[int, int], float] = {}

    quad_matches = _match_quads_by_hash(img_quads, cat_quads, tol=_QUAD_RATIO_TOL, neighbor_bins=1)
    if not quad_matches:
        return []

    for qi, qj, d in quad_matches:
        if cancel_check and cancel_check():
            return []
        sig_i, img_idx4, img_radial, _img_dmax, _img_center = img_quads[int(qi)]
        _sig_c, cat_idx4, cat_radial, _cat_dmax, _cat_center = cat_quads[int(qj)]
        score = 2.0 / (float(d) + 1e-4)
        # Pair vertices by radial rank from quad centroid (rotation/reflection agnostic)
        for k in range(4):
            ii = int(img_idx4[int(img_radial[k])])
            jj = int(cat_idx4[int(cat_radial[k])])
            key = (ii, jj)
            votes[key] = votes.get(key, 0.0) + score

    ranked_votes = sorted(((i, j, float(sc)) for (i, j), sc in votes.items()), key=lambda t: t[2], reverse=True)
    return ranked_votes




def _astap_iso_find_many_quads(points: np.ndarray, group_size: int) -> np.ndarray:
    """Portage plus fidèle ASTAP find_many_quads (modes 5 et 6)."""
    n = int(points.shape[0])
    if group_size not in (5, 6):
        return np.empty((8, 0), dtype=np.float64)
    if n < group_size:
        return np.empty((8, 0), dtype=np.float64)

    num_closest = int(group_size)
    # ASTAP command-line source mapping (2026): 15 cases, with last duplicated by design.
    combos6 = (
        # ASTAP command-line 2026 behavior: q-cases define 4 indices but only
        # the first 3 are effectively used with anchor i.
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 3),
        (0, 1, 3),
        (0, 2, 3),
        (0, 2, 3),
        (0, 2, 4),
        (0, 3, 4),
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
        (2, 3, 4),
        (2, 3, 4),
    )

    quads: list[list[float]] = []

    for i in range(n):
        x1i = float(points[i, 0])
        y1i = float(points[i, 1])

        # Find closest stars (insertion into sorted nearest list)
        closest_idx = [-1] * num_closest
        closest_d2 = [1.0e99] * num_closest
        for j in range(n):
            if j == i:
                continue
            dx = float(points[j, 0]) - x1i
            dy = float(points[j, 1]) - y1i
            d2 = dx * dx + dy * dy
            if d2 <= 1.0:
                continue
            for k in range(num_closest - 1, -1, -1):
                if d2 < closest_d2[k]:
                    if k < (num_closest - 1):
                        closest_d2[k + 1] = closest_d2[k]
                        closest_idx[k + 1] = closest_idx[k]
                    closest_d2[k] = d2
                    closest_idx[k] = j
                else:
                    break

        if closest_idx[-1] < 0:
            continue

        if group_size == 5:
            # ASTAP mode 5: 5 quads by rotating excluded star.
            variants: list[tuple[int, int, int, int]] = [
                (i,              closest_idx[0], closest_idx[1], closest_idx[2]),
                (closest_idx[3], closest_idx[0], closest_idx[1], closest_idx[2]),
                (closest_idx[3], i,              closest_idx[1], closest_idx[2]),
                (closest_idx[3], i,              closest_idx[0], closest_idx[2]),
                (closest_idx[3], i,              closest_idx[0], closest_idx[1]),
            ]
        else:
            # ASTAP mode 6: 15 quads from 6 nearest stars around anchor i.
            variants = []
            for a, b, c in combos6:
                variants.append((i, closest_idx[a], closest_idx[b], closest_idx[c]))

        for q in variants:
            if len({int(q[0]), int(q[1]), int(q[2]), int(q[3])}) < 4:
                continue
            x1 = float(points[q[0], 0]); y1 = float(points[q[0], 1])
            x2 = float(points[q[1], 0]); y2 = float(points[q[1], 1])
            x3 = float(points[q[2], 0]); y3 = float(points[q[2], 1])
            x4 = float(points[q[3], 0]); y4 = float(points[q[3], 1])

            xt = (x1 + x2 + x3 + x4) * 0.25
            yt = (y1 + y2 + y3 + y4) * 0.25

            identical = False
            for qq in quads:
                if abs(xt - qq[6]) < 1.0 and abs(yt - qq[7]) < 1.0:
                    identical = True
                    break
            if identical:
                continue

            d1 = float(np.hypot(x1 - x2, y1 - y2))
            d2 = float(np.hypot(x1 - x3, y1 - y3))
            d3 = float(np.hypot(x1 - x4, y1 - y4))
            d4 = float(np.hypot(x2 - x3, y2 - y3))
            d5 = float(np.hypot(x2 - x4, y2 - y4))
            d6 = float(np.hypot(x3 - x4, y3 - y4))
            d = sorted([d1, d2, d3, d4, d5, d6], reverse=True)
            if d[0] <= 1.0e-12:
                continue
            quads.append([d[0], d[1] / d[0], d[2] / d[0], d[3] / d[0], d[4] / d[0], d[5] / d[0], xt, yt])

    if not quads:
        return np.empty((8, 0), dtype=np.float64)
    return np.asarray(quads, dtype=np.float64).T

def _astap_iso_find_quads(points: np.ndarray, nrstars_image: int) -> np.ndarray:
    """Portage ISO de find_quads (ASTAP 2026): sortie [8, n_quads]."""
    nrstars = int(points.shape[0])
    if nrstars_image < 30:
        return _astap_iso_find_many_quads(points, 6)
    if nrstars_image < 60:
        return _astap_iso_find_many_quads(points, 5)
    if nrstars < 4:
        return np.empty((8, 0), dtype=np.float64)

    # quickSort_starlist(starlist, X) + bandw
    if nrstars >= 150:
        order = np.argsort(points[:, 0], kind='stable')
        pts = points[order]
        bandw = int(round(2.0 * np.sqrt(nrstars)))
    else:
        pts = points
        bandw = nrstars

    grid_inv = 0.2
    bucket_capacity = 10
    hash_table_len = max(1, nrstars * 2)
    hash_table: list[list[int]] = [[] for _ in range(hash_table_len)]

    quads: list[list[float]] = []

    for i in range(nrstars):
        distance1 = 1e99
        distance2 = 1e99
        distance3 = 1e99
        j_index1 = 0
        j_index2 = 0
        j_index3 = 0

        sstart = max(0, i - bandw)
        send = min(nrstars - 1, i + bandw)

        x1 = float(pts[i, 0])
        y1 = float(pts[i, 1])

        # before i
        for j in range(sstart, i):
            disty = (float(pts[j, 1]) - y1) ** 2
            if disty < distance3:
                distance = (float(pts[j, 0]) - x1) ** 2 + disty
                if distance > 1.0:
                    if distance < distance1:
                        distance3, j_index3 = distance2, j_index2
                        distance2, j_index2 = distance1, j_index1
                        distance1, j_index1 = distance, j
                    elif distance < distance2:
                        distance3, j_index3 = distance2, j_index2
                        distance2, j_index2 = distance, j
                    elif distance < distance3:
                        distance3, j_index3 = distance, j

        # after i
        for j in range(i + 1, send + 1):
            disty = (float(pts[j, 1]) - y1) ** 2
            if disty < distance3:
                distance = (float(pts[j, 0]) - x1) ** 2 + disty
                if distance > 1.0:
                    if distance < distance1:
                        distance3, j_index3 = distance2, j_index2
                        distance2, j_index2 = distance1, j_index1
                        distance1, j_index1 = distance, j
                    elif distance < distance2:
                        distance3, j_index3 = distance2, j_index2
                        distance2, j_index2 = distance, j
                    elif distance < distance3:
                        distance3, j_index3 = distance, j

        if distance3 >= 1e99:
            continue

        x2 = float(pts[j_index1, 0]); y2 = float(pts[j_index1, 1])
        x3 = float(pts[j_index2, 0]); y3 = float(pts[j_index2, 1])
        x4 = float(pts[j_index3, 0]); y4 = float(pts[j_index3, 1])

        xt = (x1 + x2 + x3 + x4) * 0.25
        yt = (y1 + y2 + y3 + y4) * 0.25

        hash_x = int(np.trunc(xt * grid_inv))
        hash_y = int(np.trunc(yt * grid_inv))
        idx = abs(hash_x * 31 + hash_y) % hash_table_len

        identical = False
        for qidx in hash_table[idx]:
            qx = quads[qidx][6]
            qy = quads[qidx][7]
            if abs(xt - qx) < 1.0 and abs(yt - qy) < 1.0:
                identical = True
                break
        if identical:
            continue

        dist1 = np.sqrt(distance1)
        dist2 = np.sqrt(distance2)
        dist3 = np.sqrt(distance3)
        dist4 = np.hypot(x2 - x3, y2 - y3)
        dist5 = np.hypot(x2 - x4, y2 - y4)
        dist6 = np.hypot(x3 - x4, y3 - y4)

        d = [dist1, dist2, dist3, dist4, dist5, dist6]
        d.sort(reverse=True)
        if d[0] <= 1e-12:
            continue

        q = [d[0], d[1] / d[0], d[2] / d[0], d[3] / d[0], d[4] / d[0], d[5] / d[0], xt, yt]
        quads.append(q)
        hash_table[idx].append(len(quads) - 1)
        if len(hash_table[idx]) > bucket_capacity:
            # ASTAP redimensionne dynamiquement, ici Python list gère nativement.
            pass

    if not quads:
        return np.empty((8, 0), dtype=np.float64)
    return np.asarray(quads, dtype=np.float64).T




def _astap_iso_find_fit(
    quads_ref: np.ndarray,
    quads_img: np.ndarray,
    *,
    minimum_count: int,
    quad_tolerance: float,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, int, int]:
    """ASTAP-like brute-force find_fit for small quad sets."""
    n1 = int(quads_ref.shape[1]) if quads_ref.ndim == 2 else 0
    n2 = int(quads_img.shape[1]) if quads_img.ndim == 2 else 0
    if n1 < minimum_count or n2 < minimum_count:
        return False, None, None, 0, 0

    matches: list[tuple[int, int]] = []
    for i in range(n1):
        q1 = quads_ref[1:6, i]
        for j in range(n2):
            q2 = quads_img[1:6, j]
            if np.max(np.abs(q1 - q2)) <= quad_tolerance:
                matches.append((i, j))

    nr2 = len(matches)
    if nr2 < minimum_count:
        return False, None, None, 0, nr2

    ratios = np.array([quads_ref[0, i] / max(quads_img[0, j], 1e-12) for i, j in matches], dtype=np.float64)
    median_ratio = float(np.median(ratios)) if ratios.size else 0.0
    if not np.isfinite(median_ratio) or median_ratio <= 0:
        return False, None, None, 0, nr2

    keep: list[tuple[int, int]] = []
    for (i, j), r in zip(matches, ratios):
        if abs(median_ratio - float(r)) <= quad_tolerance * median_ratio:
            keep.append((i, j))

    nr = len(keep)
    if nr < 3:
        return False, None, None, nr, nr2

    x = np.array([quads_img[6, j] for (_, j) in keep], dtype=np.float64)
    y = np.array([quads_img[7, j] for (_, j) in keep], dtype=np.float64)
    xr = np.array([quads_ref[6, i] for (i, _) in keep], dtype=np.float64)
    yr = np.array([quads_ref[7, i] for (i, _) in keep], dtype=np.float64)

    D = np.column_stack((x, y, np.ones_like(x)))
    try:
        px, *_ = np.linalg.lstsq(D, xr, rcond=None)
        py, *_ = np.linalg.lstsq(D, yr, rcond=None)
    except Exception:
        return False, None, None, nr, nr2

    M = np.array([[float(px[0]), float(px[1])], [float(py[0]), float(py[1])]], dtype=np.float64)
    t = np.array([float(px[2]), float(py[2])], dtype=np.float64)
    # ASTAP-like final guard: x/y scales should be nearly isotropic.
    sx2 = float(M[0, 0] * M[0, 0] + M[0, 1] * M[0, 1])
    sy2 = float(M[1, 0] * M[1, 0] + M[1, 1] * M[1, 1])
    xy_sqr_ratio = sx2 / max(1e-8, sy2)
    if not (0.9 <= xy_sqr_ratio <= 1.1):
        return False, None, None, nr, nr2
    return True, M, t, nr, nr2



def _astap_iso_find_fit_using_hash(
    quads_ref: np.ndarray,
    quads_img: np.ndarray,
    *,
    minimum_count: int,
    quad_tolerance: float,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, int, int]:
    """ASTAP-like find_fit_using_hash, returns affine map img->ref center coordinates."""
    n1 = int(quads_ref.shape[1]) if quads_ref.ndim == 2 else 0
    n2 = int(quads_img.shape[1]) if quads_img.ndim == 2 else 0
    if n1 < minimum_count or n2 < minimum_count:
        return False, None, None, 0, 0

    hash_bins = max(1, 2 * max(n1, n2))
    max_quads_per_bin = 15

    h1 = [[] for _ in range(hash_bins)]
    h2 = [[] for _ in range(hash_bins)]

    for i in range(n1):
        b = int(quads_ref[1, i] / max(quad_tolerance, 1e-9)) % hash_bins
        h1[b].append(i)
    for j in range(n2):
        b = int(quads_img[1, j] / max(quad_tolerance, 1e-9)) % hash_bins
        h2[b].append(j)

    matches=[]
    for b in range(hash_bins):
        if not h1[b]:
            continue
        for db in (-1,0,1):
            bb = (b + db) % hash_bins
            if not h2[bb]:
                continue
            for i in h1[b]:
                r1 = quads_ref[1:6, i]
                for j in h2[bb]:
                    r2 = quads_img[1:6, j]
                    if np.max(np.abs(r1-r2)) <= quad_tolerance:
                        matches.append((i,j))

    nr2 = len(matches)
    if nr2 < minimum_count:
        return False, None, None, 0, nr2

    ratios = np.array([quads_ref[0, i] / max(quads_img[0, j], 1e-12) for i,j in matches], dtype=np.float64)
    med = float(np.median(ratios)) if ratios.size else 0.0
    if not np.isfinite(med) or med <= 0:
        return False, None, None, 0, nr2

    keep=[]
    for (i,j),r in zip(matches, ratios):
        if abs(med - r) <= quad_tolerance * med:
            keep.append((i,j))
    nr = len(keep)
    if nr < 3:
        return False, None, None, nr, nr2

    # Solve affine on quad centers: ref = A * [x_img,y_img,1]
    x = np.array([quads_img[6, j] for (_,j) in keep], dtype=np.float64)
    y = np.array([quads_img[7, j] for (_,j) in keep], dtype=np.float64)
    xr = np.array([quads_ref[6, i] for (i,_) in keep], dtype=np.float64)
    yr = np.array([quads_ref[7, i] for (i,_) in keep], dtype=np.float64)

    D = np.column_stack((x, y, np.ones_like(x)))
    try:
        px, *_ = np.linalg.lstsq(D, xr, rcond=None)
        py, *_ = np.linalg.lstsq(D, yr, rcond=None)
    except Exception:
        return False, None, None, nr, nr2

    M = np.array([[float(px[0]), float(px[1])], [float(py[0]), float(py[1])]], dtype=np.float64)
    t = np.array([float(px[2]), float(py[2])], dtype=np.float64)
    # ASTAP-like final guard: x/y scales should be nearly isotropic.
    sx2 = float(M[0, 0] * M[0, 0] + M[0, 1] * M[0, 1])
    sy2 = float(M[1, 0] * M[1, 0] + M[1, 1] * M[1, 1])
    xy_sqr_ratio = sx2 / max(1e-8, sy2)
    if not (0.9 <= xy_sqr_ratio <= 1.1):
        return False, None, None, nr, nr2
    return True, M, t, nr, nr2


def _astap_iso_hypothesis(
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    *,
    img_ranks: np.ndarray | None = None,
    cat_ranks: np.ndarray | None = None,
    expected_scale_arcsec: float | None = None,
    expected_scale_ratio_min: float = 0.55,
    expected_scale_ratio_max: float = 1.80,
    minimum_count: int = 3,
    strict_astap_iso: bool = False,
    quad_tolerance: float = 0.007,
    diag: dict | None = None,
    trace: dict | None = None,
) -> tuple[SimilarityTransform | None, np.ndarray | None, np.ndarray | None, int]:
    if image_positions.shape[0] < 8 or catalog_positions.shape[0] < 20:
        return None, None, None, 0

    if strict_astap_iso:
        img_n = int(image_positions.shape[0])
        cat_n = int(catalog_positions.shape[0])
    else:
        img_n = min(image_positions.shape[0], 500)
        cat_n = min(catalog_positions.shape[0], 1200)

    if img_ranks is not None and img_ranks.size == image_positions.shape[0]:
        img_idx = np.argsort(img_ranks, kind='stable')[:img_n]
        img = image_positions[img_idx].astype(np.float64, copy=False)
    else:
        img_idx = np.arange(int(image_positions.shape[0]), dtype=np.int64)[:img_n]
        img = image_positions[:img_n].astype(np.float64, copy=False)

    if cat_ranks is not None and cat_ranks.size == catalog_positions.shape[0]:
        cat_idx = np.argsort(cat_ranks, kind='stable')[:cat_n]
        cat = catalog_positions[cat_idx].astype(np.float64, copy=False)
    else:
        cat_idx = np.arange(int(catalog_positions.shape[0]), dtype=np.int64)[:cat_n]
        cat = catalog_positions[:cat_n].astype(np.float64, copy=False)

    q_img = _astap_iso_find_quads(img, int(img.shape[0]))
    q_cat = _astap_iso_find_quads(cat, int(cat.shape[0]))

    if trace is not None:
        trace.clear()
        trace.update({
            "image_positions": img.copy(),
            "catalog_positions": cat.copy(),
            "image_source_indices": np.asarray(img_idx, dtype=np.int64).copy(),
            "catalog_source_indices": np.asarray(cat_idx, dtype=np.int64).copy(),
            "image_quads": q_img.copy(),
            "catalog_quads": q_cat.copy(),
        })

    if diag is not None:
        diag.clear()
        diag.update({
            "strict_mode": bool(strict_astap_iso),
            "quad_tolerance": float(quad_tolerance),
            "stars_img": int(img.shape[0]),
            "stars_cat": int(cat.shape[0]),
            "quads_img": int(q_img.shape[1]) if q_img.ndim == 2 else 0,
            "quads_cat": int(q_cat.shape[1]) if q_cat.ndim == 2 else 0,
            "minimum_count": int(minimum_count),
            "scale_ratio_gate": {
                "min": float(expected_scale_ratio_min),
                "max": float(expected_scale_ratio_max),
            },
            "tolerances": [],
        })

    if q_img.shape[1] == 0 or q_cat.shape[1] == 0:
        if diag is not None:
            diag["path_used"] = None
            diag["reason"] = "no_quads"
        return None, None, None, 0

    # ASTAP-like branch selection: brute force for small quad sets, hash otherwise.
    use_bruteforce = int(q_cat.shape[1]) < 180
    fit_fn = _astap_iso_find_fit if use_bruteforce else _astap_iso_find_fit_using_hash
    if diag is not None:
        diag["path_used"] = "find_fit" if use_bruteforce else "find_fit_using_hash"

    if strict_astap_iso:
        tol = max(1e-6, float(quad_tolerance))
        ok, M, t, nr, nr_raw = fit_fn(q_cat, q_img, minimum_count=minimum_count, quad_tolerance=tol)
        if diag is not None:
            diag["tolerances"].append({
                "tol": float(tol),
                "matches_raw": int(nr_raw),
                "matches_kept": int(nr),
                "ok": bool(ok),
            })
        if ok and M is not None and t is not None:
            tr = _similarity_from_affine(M, t)
            if tr is not None:
                if diag is not None:
                    diag["selected"] = {
                        "best_refs": int(nr),
                        "best_quick_inliers": None,
                        "best_quick_med": None,
                    }
                return tr, M, t, int(nr)
        if diag is not None:
            diag["reason"] = "no_valid_hypothesis"
            diag["selected"] = {
                "best_refs": int(nr),
                "best_quick_inliers": None,
                "best_quick_med": None,
            }
        return None, None, None, int(nr)

    # Non-strict path: keep current ZeNear tolerance sweep/scoring behavior.
    best_tr: SimilarityTransform | None = None
    best_M: np.ndarray | None = None
    best_t: np.ndarray | None = None
    best_nr = 0
    best_quick_inliers = -1
    best_quick_med = float('inf')
    best_scale_score = float('inf')
    ratio_min = float(expected_scale_ratio_min)
    ratio_max = float(expected_scale_ratio_max)
    if not math.isfinite(ratio_min):
        ratio_min = 0.55
    if not math.isfinite(ratio_max):
        ratio_max = 1.80
    ratio_min = max(0.05, ratio_min)
    ratio_max = max(ratio_min + 1.0e-6, ratio_max)
    for tol in (0.007, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060):
        ok, M, t, nr, nr_raw = fit_fn(q_cat, q_img, minimum_count=minimum_count, quad_tolerance=tol)
        if ok and M is not None and t is not None:
            tr = _similarity_from_affine(M, t)
            if tr is None:
                logger.info("near astap-iso tol=%.4f ok but similarity_from_affine failed (nr=%d)", float(tol), int(nr))
                continue
            scale_px = float(getattr(tr, 'scale', 0.0))
            ratio = float('nan')
            if expected_scale_arcsec is not None and expected_scale_arcsec > 0:
                ratio = scale_px / float(expected_scale_arcsec)
                if not (ratio_min <= ratio <= ratio_max):
                    logger.info(
                        "near astap-iso tol=%.4f rejected on scale ratio=%.3f not in [%.3f, %.3f] (nr=%d)",
                        float(tol),
                        float(ratio),
                        float(ratio_min),
                        float(ratio_max),
                        int(nr),
                    )
                    if diag is not None:
                        diag["tolerances"].append({
                            "tol": float(tol),
                            "matches_raw": int(nr_raw),
                            "matches_kept": int(nr),
                            "ok": False,
                            "reason": "scale_ratio_rejected",
                            "scale_ratio": float(ratio) if math.isfinite(ratio) else None,
                            "scale_ratio_min": float(ratio_min),
                            "scale_ratio_max": float(ratio_max),
                        })
                    continue
                scale_score = abs(math.log(max(scale_px, 1e-9) / float(expected_scale_arcsec)))
            else:
                scale_score = abs(math.log(max(scale_px, 1e-9)))
            quick_inliers = 0
            quick_med = float('inf')
            quick_tol = max(8.0, 5.0 * max(scale_px, 1e-6))
            try:
                pred = _apply_affine_points(M, t, img)
                d2 = np.sum((pred[:, None, :] - cat[None, :, :]) ** 2, axis=2)
                jbest = np.argmin(d2, axis=1)
                dist = np.sqrt(d2[np.arange(d2.shape[0]), jbest])
                valid = np.isfinite(dist)
                if np.any(valid):
                    quick_med = float(np.median(dist[valid]))
                    quick_inliers = int(np.count_nonzero(valid & (dist <= quick_tol)))
            except Exception:
                quick_inliers = 0
                quick_med = float('inf')
            logger.info(
                "near astap-iso[%s] tol=%.4f candidate nr=%d scale=%.3f arcsec/px ratio=%.3f quick_inliers=%d quick_med=%.3f quick_tol=%.3f",
                "find_fit" if use_bruteforce else "find_fit_using_hash",
                float(tol),
                int(nr),
                float(scale_px),
                float(ratio),
                int(quick_inliers),
                float(quick_med) if math.isfinite(quick_med) else float('nan'),
                float(quick_tol),
            )
            if diag is not None:
                diag["tolerances"].append({
                    "tol": float(tol),
                    "matches_raw": int(nr_raw),
                    "matches_kept": int(nr),
                    "ok": True,
                    "scale_ratio": float(ratio) if math.isfinite(ratio) else None,
                    "quick_inliers": int(quick_inliers),
                    "quick_med": float(quick_med) if math.isfinite(quick_med) else None,
                })
            if (
                (quick_inliers > best_quick_inliers)
                or (
                    quick_inliers == best_quick_inliers
                    and (
                        (quick_med < best_quick_med)
                        or (
                            abs(float(quick_med) - float(best_quick_med)) <= 1e-9
                            and ((nr > best_nr) or (nr == best_nr and scale_score < best_scale_score))
                        )
                    )
                )
            ):
                best_tr = tr
                best_M = M
                best_t = t
                best_nr = int(nr)
                best_quick_inliers = int(quick_inliers)
                best_quick_med = float(quick_med)
                best_scale_score = float(scale_score)
        else:
            logger.info(
                "near astap-iso[%s] tol=%.4f no fit (nr=%d)",
                "find_fit" if use_bruteforce else "find_fit_using_hash",
                float(tol),
                int(nr),
            )
            if diag is not None:
                diag["tolerances"].append({
                    "tol": float(tol),
                    "matches_raw": int(nr_raw),
                    "matches_kept": int(nr),
                    "ok": False,
                })
            if nr > best_nr:
                best_nr = int(nr)

    if best_tr is not None:
        s = float(getattr(best_tr, 'scale', 0.0))
        if 0.2 <= s <= 20.0:
            if diag is not None:
                diag["selected"] = {
                    "best_refs": int(best_nr),
                    "best_quick_inliers": int(best_quick_inliers),
                    "best_quick_med": float(best_quick_med) if math.isfinite(best_quick_med) else None,
                }
            return best_tr, best_M, best_t, best_nr
    if diag is not None:
        diag["reason"] = "no_valid_hypothesis"
        diag["selected"] = {
            "best_refs": int(best_nr),
            "best_quick_inliers": int(best_quick_inliers),
            "best_quick_med": float(best_quick_med) if math.isfinite(best_quick_med) else None,
        }
    return None, None, None, best_nr
def _standard_equatorial_astap(ra0_deg: float, dec0_deg: float, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ASTAP-equivalent standard->equatorial transform (cdelt=1 arcsec)."""
    ra0 = np.deg2rad(float(ra0_deg))
    dec0 = np.deg2rad(float(dec0_deg))
    sin_dec0 = math.sin(dec0)
    cos_dec0 = math.cos(dec0)
    fac = math.pi / (180.0 * 3600.0)
    x = np.asarray(x_arcsec, dtype=np.float64) * fac
    y = np.asarray(y_arcsec, dtype=np.float64) * fac
    ra = ra0 + np.arctan2(-x, cos_dec0 - y * sin_dec0)
    ra = np.mod(ra, 2.0 * math.pi)
    dec = np.arcsin((sin_dec0 + y * cos_dec0) / np.sqrt(1.0 + x * x + y * y))
    return np.rad2deg(ra), np.rad2deg(dec)


def _build_matches_from_affine_standard(
    matrix: np.ndarray,
    offset: np.ndarray,
    image_positions: np.ndarray,
    *,
    ra_center_deg: float,
    dec_center_deg: float,
    max_matches: int = 700,
) -> np.ndarray:
    """Build direct image->(RA,DEC) correspondences from ASTAP affine standard coords."""
    if image_positions.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    n = min(int(max_matches), int(image_positions.shape[0]))
    pts = image_positions[:n].astype(np.float64, copy=False)
    std = _apply_affine_points(matrix, offset, pts)
    ra, dec = _standard_equatorial_astap(float(ra_center_deg), float(dec_center_deg), std[:, 0], std[:, 1])
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        mask = np.isfinite(ra) & np.isfinite(dec)
        pts = pts[mask]
        ra = ra[mask]
        dec = dec[mask]
    if pts.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    return np.column_stack((pts, np.column_stack((ra, dec)))).astype(np.float64, copy=False)


def _apply_transform_points(transform: SimilarityTransform, points_xy: np.ndarray) -> np.ndarray:
    src = points_xy[:, 0] + 1j * points_xy[:, 1]
    if getattr(transform, "parity", 1) < 0:
        src = np.conj(src)
    rot_scale = float(transform.scale) * np.exp(1j * float(transform.rotation))
    translation = complex(*transform.translation)
    dst = rot_scale * src + translation
    return np.column_stack((np.real(dst), np.imag(dst))).astype(np.float64, copy=False)


def _apply_affine_points(matrix: np.ndarray, offset: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    return (points_xy @ matrix.T) + offset[None, :]


def _refine_affine_from_nn(
    matrix: np.ndarray,
    offset: np.ndarray,
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    *,
    tol_arcsec_seq: tuple[float, ...] = (60.0, 35.0, 20.0, 12.0),
    min_pairs: int = 6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Iteratively refine affine image->catalog mapping from NN correspondences."""
    M = np.asarray(matrix, dtype=np.float64)
    t = np.asarray(offset, dtype=np.float64)
    best_pairs = 0

    if image_positions.size == 0 or catalog_positions.size == 0:
        return M, t, best_pairs

    img = image_positions.astype(np.float64, copy=False)
    cat = catalog_positions.astype(np.float64, copy=False)

    for tol in tol_arcsec_seq:
        pred = _apply_affine_points(M, t, img)
        d2 = np.sum((pred[:, None, :] - cat[None, :, :]) ** 2, axis=2)
        jbest = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(d2.shape[0]), jbest])
        valid = np.where(np.isfinite(dist) & (dist <= float(tol)))[0]
        if valid.size < int(min_pairs):
            continue

        order = valid[np.argsort(dist[valid])]
        used_cat: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for i in order:
            j = int(jbest[i])
            if j in used_cat:
                continue
            used_cat.add(j)
            pairs.append((int(i), j))
        if len(pairs) < int(min_pairs):
            continue

        img_idx = np.array([p[0] for p in pairs], dtype=np.int32)
        cat_idx = np.array([p[1] for p in pairs], dtype=np.int32)
        X = np.column_stack((img[img_idx, 0], img[img_idx, 1], np.ones(img_idx.shape[0], dtype=np.float64)))
        Yx = cat[cat_idx, 0]
        Yy = cat[cat_idx, 1]
        try:
            px, *_ = np.linalg.lstsq(X, Yx, rcond=None)
            py, *_ = np.linalg.lstsq(X, Yy, rcond=None)
        except Exception:
            continue

        M_new = np.array([[float(px[0]), float(px[1])], [float(py[0]), float(py[1])]], dtype=np.float64)
        t_new = np.array([float(px[2]), float(py[2])], dtype=np.float64)

        sx2 = float(M_new[0, 0] * M_new[0, 0] + M_new[0, 1] * M_new[0, 1])
        sy2 = float(M_new[1, 0] * M_new[1, 0] + M_new[1, 1] * M_new[1, 1])
        xy_sqr_ratio = sx2 / max(1e-8, sy2)
        if not (0.8 <= xy_sqr_ratio <= 1.25):
            continue

        M, t = M_new, t_new
        best_pairs = max(best_pairs, int(len(pairs)))

    return M, t, best_pairs


def _build_matches_from_affine(
    matrix: np.ndarray,
    offset: np.ndarray,
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    catalog_world: np.ndarray,
    *,
    pixel_tolerance: float,
    max_matches: int = 400,
) -> tuple[np.ndarray, int, float]:
    if image_positions.size == 0 or catalog_positions.size == 0:
        return np.empty((0, 4), dtype=np.float64), 0, float('inf')
    pred = _apply_affine_points(matrix, offset, image_positions.astype(np.float64, copy=False))
    d2 = np.sum((pred[:, None, :] - catalog_positions[None, :, :]) ** 2, axis=2)
    jbest = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(d2.shape[0]), jbest])

    # ASTAP-like: tolerance derived from affine scale (not from loose residual quantiles).
    try:
        scale_aff = float(np.sqrt(abs(np.linalg.det(matrix))))
    except Exception:
        scale_aff = 0.0
    if not np.isfinite(scale_aff) or scale_aff <= 0:
        scale_aff = float(np.mean(np.sqrt(np.sum(matrix * matrix, axis=1)))) if matrix.size else 1e-4
    floor = max(1e-7, 0.25 * float(pixel_tolerance) * max(scale_aff, 1e-9))
    tol_deg = max(floor, 1.5 * float(pixel_tolerance) * max(scale_aff, 1e-9))

    # Mutual nearest-neighbour gating to reduce false correspondences.
    ibest = np.argmin(d2, axis=0)
    mutual = np.zeros(d2.shape[0], dtype=bool)
    idx = np.arange(d2.shape[0])
    mutual = (ibest[jbest] == idx)

    valid = np.where(np.isfinite(dist) & mutual & (dist <= max(tol_deg, 1e-9)))[0]
    if valid.size == 0:
        # fallback to non-mutual when scene is sparse
        valid = np.where(np.isfinite(dist) & (dist <= max(tol_deg, 1e-9)))[0]
        if valid.size == 0:
            return np.empty((0, 4), dtype=np.float64), 0, float('inf')

    # Robust residual clipping
    dsel = dist[valid]
    med = float(np.median(dsel)) if dsel.size else float('inf')
    mad = float(np.median(np.abs(dsel - med))) if dsel.size else 0.0
    robust_cut = med + max(3.5 * mad, 2.5 * max(med, floor))
    robust_cut = max(robust_cut, tol_deg)
    valid = valid[dist[valid] <= robust_cut]
    if valid.size == 0:
        return np.empty((0, 4), dtype=np.float64), 0, float('inf')

    order = valid[np.argsort(dist[valid])]
    used_cat: set[int] = set()
    pairs: list[tuple[int,int,float]] = []
    for i in order:
        j = int(jbest[i])
        if j in used_cat:
            continue
        used_cat.add(j)
        pairs.append((int(i), j, float(dist[i])))
        if len(pairs) >= int(max_matches):
            break

    if not pairs:
        return np.empty((0, 4), dtype=np.float64), 0, float('inf')

    img_idx = np.array([p[0] for p in pairs], dtype=np.int32)
    cat_idx = np.array([p[1] for p in pairs], dtype=np.int32)
    errs = np.array([p[2] for p in pairs], dtype=np.float64)
    matches = np.column_stack((
        image_positions[img_idx].astype(np.float64, copy=False),
        catalog_world[cat_idx].astype(np.float64, copy=False),
    )).astype(np.float64, copy=False)
    inliers = int(matches.shape[0])
    rms_px = float(np.sqrt(np.mean((errs / max(floor, 1e-9)) ** 2))) if inliers > 0 else float('inf')
    return matches, inliers, rms_px


def _build_matches_from_transform(
    transform: SimilarityTransform,
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    catalog_world: np.ndarray,
    *,
    pixel_tolerance: float,
    max_matches: int = 400,
) -> tuple[np.ndarray, int, float]:
    """Build unique image->catalog matches from a transform using NN assignment."""
    if image_positions.size == 0 or catalog_positions.size == 0:
        return np.empty((0, 4), dtype=np.float64), 0, float("inf")

    pred = _apply_transform_points(transform, image_positions)
    # distance matrix in tangent-plane degrees
    d2 = np.sum((pred[:, None, :] - catalog_positions[None, :, :]) ** 2, axis=2)
    jbest = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(d2.shape[0]), jbest])

    tol_deg = max(1e-9, float(pixel_tolerance) * max(float(transform.scale), 1e-12))
    valid = np.where(np.isfinite(dist) & (dist <= tol_deg))[0]
    if valid.size == 0:
        return np.empty((0, 4), dtype=np.float64), 0, float("inf")

    # enforce unique catalog assignment by increasing residual
    order = valid[np.argsort(dist[valid])]
    used_cat: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for i in order:
        j = int(jbest[i])
        if j in used_cat:
            continue
        used_cat.add(j)
        pairs.append((int(i), j, float(dist[i])))
        if len(pairs) >= int(max_matches):
            break

    if not pairs:
        return np.empty((0, 4), dtype=np.float64), 0, float("inf")

    img_idx = np.array([p[0] for p in pairs], dtype=np.int32)
    cat_idx = np.array([p[1] for p in pairs], dtype=np.int32)
    errs = np.array([p[2] for p in pairs], dtype=np.float64)

    matches = np.column_stack((
        image_positions[img_idx].astype(np.float64, copy=False),
        catalog_world[cat_idx].astype(np.float64, copy=False),
    )).astype(np.float64, copy=False)

    inliers = int(matches.shape[0])
    rms_px = float(np.sqrt(np.mean((errs / max(float(transform.scale), 1e-12)) ** 2))) if inliers > 0 else float("inf")
    return matches, inliers, rms_px


def _build_candidate_pairs(
    image_positions: np.ndarray,
    catalog_positions: np.ndarray,
    catalog_world: np.ndarray,
    img_ranks: np.ndarray,
    cat_ranks: np.ndarray,
    center_xy: tuple[float, float],
    approx_scale_deg: float,
    pixel_tolerance: float,
    *,
    max_neighbors: int = _MAX_NEIGHBORS,
    rank_tolerance: float = _RANK_TOLERANCE,
    cancel_check: Callable[[], bool] | None = None,
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
        if cancel_check and cancel_check():
            return (np.empty((0, 2), dtype=np.float32),) * 3
        target = radius_px * approx_scale_deg
        window = max(base_window, target * 0.25)
        left = np.searchsorted(radius_sorted, target - window, side="left")
        right = np.searchsorted(radius_sorted, target + window, side="right")
        if right <= left:
            continue
        candidates = idx_sorted[left:right]
        for cat_idx in candidates[:max(1, int(max_neighbors))]:
            rank_gap = abs(float(img_ranks[img_idx]) - float(cat_ranks[cat_idx]))
            if rank_gap > float(rank_tolerance):
                continue
            diff = abs(cat_radius_deg[cat_idx] - target)
            score = (1.0 - min(0.9, rank_gap)) / (diff + 1e-6)
            votes.append((img_idx, int(cat_idx), float(score)))
    quad_votes = _build_quad_votes(
        image_positions,
        catalog_positions,
        img_ranks,
        cat_ranks,
        cancel_check=cancel_check,
    )
    if quad_votes:
        votes.extend(quad_votes)

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


def validate_strict_astap_iso_candidate(
    wcs,
    matches: np.ndarray,
    *,
    width: int,
    height: int,
    ra_hint_deg: float | None,
    dec_hint_deg: float | None,
    search_radius_deg: float | None,
    approx_fov_deg: float | None,
    approx_scale_arcsec: float | None,
    pixel_tolerance: float,
    iso_refs: int,
    min_unique_matches: int = 12,
    min_holdout_matches: int = 8,
    max_rms_px: float = 3.0,
    max_cd_cond: float = 1.0e4,
    min_span_fraction: float = 0.12,
) -> dict[str, object]:
    """Runtime-only acceptance gate for strict ASTAP-ISO Near candidates."""
    diag: dict[str, object] = {"decision": "REJECT", "reason": "STRICT_ACCEPTANCE_REJECTED"}

    def reject(reason: str) -> dict[str, object]:
        diag["decision"] = "REJECT"
        diag["reason"] = reason
        return diag

    if wcs is None or not bool(getattr(wcs, "is_celestial", False)):
        return reject("STRICT_ACCEPTANCE_NONFINITE_WCS")
    try:
        cd = getattr(wcs.wcs, "cd", None)
        if cd is None:
            pc = getattr(wcs.wcs, "pc", None)
            cdelt = getattr(wcs.wcs, "cdelt", None)
            if pc is not None and cdelt is not None:
                cd = np.asarray(pc, dtype=np.float64)[:2, :2] @ np.diag(np.asarray(cdelt, dtype=np.float64)[:2])
        cd_arr = np.asarray(cd, dtype=np.float64)[:2, :2]
    except Exception:
        return reject("STRICT_ACCEPTANCE_BAD_CD")
    if cd_arr.shape != (2, 2) or not np.all(np.isfinite(cd_arr)):
        return reject("STRICT_ACCEPTANCE_BAD_CD")
    try:
        cd_det = float(np.linalg.det(cd_arr))
        cd_cond = float(np.linalg.cond(cd_arr))
    except Exception:
        return reject("STRICT_ACCEPTANCE_BAD_CD")
    diag["cd_det"] = cd_det
    diag["cd_cond"] = cd_cond
    if (not np.isfinite(cd_det)) or abs(cd_det) < 1.0e-16 or (not np.isfinite(cd_cond)) or cd_cond > float(max_cd_cond):
        return reject("STRICT_ACCEPTANCE_BAD_CD")

    scales_arcsec = np.sqrt(np.sum(cd_arr * cd_arr, axis=0)) * 3600.0
    finite_scales = scales_arcsec[np.isfinite(scales_arcsec)]
    if finite_scales.size == 0:
        return reject("STRICT_ACCEPTANCE_SCALE")
    scale_arcsec = float(np.mean(np.abs(finite_scales)))
    diag["pix_scale_arcsec"] = scale_arcsec
    if not (0.3 <= scale_arcsec <= 15.0):
        return reject("STRICT_ACCEPTANCE_SCALE")
    if approx_scale_arcsec is not None:
        try:
            expected = float(approx_scale_arcsec)
        except Exception:
            expected = float("nan")
        if np.isfinite(expected) and expected > 0:
            ratio = scale_arcsec / expected
            diag["scale_ratio_to_hint"] = float(ratio)
            if not (0.45 <= ratio <= 2.20):
                return reject("STRICT_ACCEPTANCE_SCALE")

    if ra_hint_deg is not None and dec_hint_deg is not None:
        try:
            center = np.asarray(wcs.wcs_pix2world([[float(width) / 2.0, float(height) / 2.0]], 0), dtype=np.float64)
            center_ra = float(center[0, 0])
            center_dec = float(center[0, 1])
            center_sep = _angular_distance(float(ra_hint_deg), float(dec_hint_deg), center_ra, center_dec)
        except Exception:
            return reject("STRICT_ACCEPTANCE_CENTER")
        diag["center_ra_deg"] = center_ra
        diag["center_dec_deg"] = center_dec
        diag["center_sep_deg"] = float(center_sep)
        try:
            fov = float(approx_fov_deg) if approx_fov_deg is not None else 0.0
        except Exception:
            fov = 0.0
        try:
            radius = float(search_radius_deg) if search_radius_deg is not None else 0.0
        except Exception:
            radius = 0.0
        center_tol = max(0.25, min(6.0, max(radius, 0.6 * fov) + 0.35))
        diag["center_tol_deg"] = float(center_tol)
        if (not np.isfinite(center_sep)) or center_sep > center_tol:
            return reject("STRICT_ACCEPTANCE_CENTER")

    pts = np.asarray(matches, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 4 or pts.shape[0] <= 0:
        return reject("STRICT_ACCEPTANCE_TOO_FEW_UNIQUE_MATCHES")
    pts = pts[np.isfinite(pts[:, :4]).all(axis=1)]
    if pts.shape[0] <= 0:
        return reject("STRICT_ACCEPTANCE_TOO_FEW_UNIQUE_MATCHES")
    try:
        key = np.round(pts[:, :2], decimals=3)
        _, keep = np.unique(key, axis=0, return_index=True)
        pts = pts[np.sort(keep)]
    except Exception:
        pass
    unique_matches = int(pts.shape[0])
    diag["unique_matches"] = unique_matches
    min_unique = max(4, int(min_unique_matches))
    if unique_matches < min_unique:
        return reject("STRICT_ACCEPTANCE_TOO_FEW_UNIQUE_MATCHES")

    try:
        world = np.asarray(wcs.wcs_pix2world(pts[:, :2], 0), dtype=np.float64)
        residual_deg = np.linalg.norm(world[:, :2] - pts[:, 2:4], axis=1)
        residual_px = residual_deg / max(scale_arcsec / 3600.0, 1.0e-12)
        residual_px = residual_px[np.isfinite(residual_px)]
    except Exception:
        return reject("STRICT_ACCEPTANCE_POOR_RESIDUALS")
    if residual_px.size <= 0:
        return reject("STRICT_ACCEPTANCE_POOR_RESIDUALS")
    rms_px = float(np.sqrt(np.mean(residual_px * residual_px)))
    median_px = float(np.median(residual_px))
    p95_px = float(np.percentile(residual_px, 95))
    diag["rms_px"] = rms_px
    diag["median_residual_px"] = median_px
    diag["p95_residual_px"] = p95_px
    robust_limit = max(float(max_rms_px), 1.25 * float(pixel_tolerance))
    diag["rms_limit_px"] = float(robust_limit)
    if (not np.isfinite(rms_px)) or rms_px > robust_limit or p95_px > max(2.5 * robust_limit, robust_limit + 2.0):
        return reject("STRICT_ACCEPTANCE_POOR_RESIDUALS")

    span_x = float(np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])) if pts.shape[0] else 0.0
    span_y = float(np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])) if pts.shape[0] else 0.0
    span_x_frac = span_x / max(1.0, float(width))
    span_y_frac = span_y / max(1.0, float(height))
    diag["span_x_fraction"] = float(span_x_frac)
    diag["span_y_fraction"] = float(span_y_frac)
    min_span = max(0.02, float(min_span_fraction))
    if span_x_frac < min_span or span_y_frac < min_span:
        return reject("STRICT_ACCEPTANCE_POOR_SPATIAL_COVERAGE")
    try:
        qx = pts[:, 0] >= (float(width) / 2.0)
        qy = pts[:, 1] >= (float(height) / 2.0)
        quadrants = {int(x) + 2 * int(y) for x, y in zip(qx, qy, strict=False)}
    except Exception:
        quadrants = set()
    diag["quadrants"] = int(len(quadrants))
    if len(quadrants) < 2:
        return reject("STRICT_ACCEPTANCE_POOR_SPATIAL_COVERAGE")

    holdout = max(0, unique_matches - max(0, int(iso_refs)))
    diag["holdout_matches"] = int(holdout)
    min_holdout = max(0, int(min_holdout_matches))
    if unique_matches >= (min_unique + min_holdout) and holdout < min_holdout:
        return reject("STRICT_ACCEPTANCE_INSUFFICIENT_HOLDOUT")

    diag["decision"] = "ACCEPT"
    diag["reason"] = "STRICT_ACCEPTANCE_ACCEPTED"
    return diag


def _pix_scale_arcsec(wcs) -> Optional[float]:
    cd = getattr(wcs.wcs, "cd", None)
    if cd is None:
        return None
    det = float(np.linalg.det(cd))
    if not math.isfinite(det) or det == 0.0:
        return None
    return math.sqrt(abs(det)) * 3600.0


def _near_conformance_check(
    wcs,
    *,
    width: int,
    height: int,
    ra_hint_deg: float,
    dec_hint_deg: float,
    search_radius_deg: float,
    approx_fov_deg: float | None,
    approx_scale_arcsec: float | None,
    scale_min_ratio: float = 0.60,
    scale_max_ratio: float = 1.80,
    center_extra_deg: float = 0.6,
    center_fov_mult: float = 1.2,
    center_max_deg: float = 0.90,
) -> tuple[bool, str, dict]:
    """Reject near-solves that are mathematically valid but astrophysically implausible."""

    try:
        cx = 0.5 * float(width)
        cy = 0.5 * float(height)
        ra_c, dec_c = wcs.pixel_to_world_values(cx, cy)
        ra_c = float(ra_c)
        dec_c = float(dec_c)
    except Exception as exc:
        return False, f"center_projection_failed:{exc}", {}

    if not (math.isfinite(ra_c) and math.isfinite(dec_c)):
        return False, "nonfinite_center_world", {"center_ra": ra_c, "center_dec": dec_c}

    dist_deg = _angular_distance(ra_c, dec_c, float(ra_hint_deg), float(dec_hint_deg))
    allowed_dist = max(0.3, min(float(search_radius_deg) + float(center_extra_deg), float(center_fov_mult) * float(approx_fov_deg or 1.0) + float(center_extra_deg), float(center_max_deg)))
    if dist_deg > allowed_dist:
        return False, f"center_offset_too_large[{dist_deg:.3f}>{allowed_dist:.3f}]", {"center_ra": ra_c, "center_dec": dec_c, "center_offset_deg": dist_deg, "center_offset_max_deg": allowed_dist}

    try:
        crpix1 = float(getattr(wcs.wcs, "crpix", [np.nan, np.nan])[0])
        crpix2 = float(getattr(wcs.wcs, "crpix", [np.nan, np.nan])[1])
    except Exception:
        crpix1 = crpix2 = float("nan")

    if not (math.isfinite(crpix1) and math.isfinite(crpix2)):
        return False, "nonfinite_crpix", {"crpix1": crpix1, "crpix2": crpix2}

    if not (-0.5 * width <= crpix1 <= 1.5 * width and -0.5 * height <= crpix2 <= 1.5 * height):
        return False, f"crpix_out_of_bounds[{crpix1:.1f},{crpix2:.1f}]", {"crpix1": crpix1, "crpix2": crpix2, "width": width, "height": height}

    solved_scale = _pix_scale_arcsec(wcs)
    if solved_scale is not None and approx_scale_arcsec is not None and approx_scale_arcsec > 0:
        ratio = float(solved_scale) / float(approx_scale_arcsec)
        if not (float(scale_min_ratio) <= ratio <= float(scale_max_ratio)):
            return False, f"pixscale_mismatch_ratio[{ratio:.3f}]", {"solved_scale_arcsec": solved_scale, "approx_scale_arcsec": approx_scale_arcsec, "scale_ratio": ratio}

    return True, "ok", {"center_ra": ra_c, "center_dec": dec_c, "center_offset_deg": dist_deg, "center_offset_max_deg": allowed_dist, "crpix1": crpix1, "crpix2": crpix2, "solved_scale_arcsec": solved_scale, "approx_scale_arcsec": approx_scale_arcsec}


def solve_near(
    input_fits: Path | str,
    index_root: Path | str | None = None,
    *,
    config: NearSolveConfig | None = None,
    catalog_provider: NearCatalogProvider | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> WcsSolution:
    cfg = config or NearSolveConfig()
    logger.setLevel(cfg.log_level.upper())
    start = time.perf_counter()
    fits_path = Path(input_fits).expanduser().resolve()
    index_path = Path(index_root).expanduser().resolve() if index_root is not None else None
    ransac_seed = int(cfg.ransac_seed) if cfg.ransac_seed is not None else _stable_seed_for_path(fits_path)
    if cancel_check and cancel_check():
        return _failure("cancelled")
    provider: NearCatalogProvider | None = catalog_provider
    manifest: dict | None = None
    if provider is None:
        if index_path is None:
            return _failure("near catalog provider absent and index_root not supplied")
        try:
            manifest = load_manifest(index_path)
        except FileNotFoundError as exc:
            return _failure(f"index manifest missing: {exc}")
        tiles = manifest.get("tiles") or []
        # Optional: restrict to a specific catalog family to speed up candidate selection
        if cfg.family:
            fam = str(cfg.family).strip().lower()
            tiles = [entry for entry in tiles if str(entry.get("family", "")).strip().lower() == fam]
        if not tiles:
            return _failure("index manifest has no tiles")
    else:
        try:
            families = tuple(str(fam).strip().lower() for fam in provider.families if str(fam).strip())
        except Exception as exc:
            return _failure(f"near catalog provider invalid: {exc}")
        if cfg.family and str(cfg.family).strip().lower() not in set(families):
            return _failure(f"near catalog provider missing requested family: {cfg.family}")
        if not families:
            return _failure("near catalog provider has no families")
    if cancel_check and cancel_check():
        return _failure("cancelled")
    strict_astap_iso = bool(getattr(cfg, "astap_iso_strict", True))
    try:
        with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
            primary = hdul[0]
            if primary.data is None:
                return _failure("FITS HDU has no image data")
            header = primary.header
            if strict_astap_iso:
                image = astap_iso_image_for_solve(primary)
            else:
                image = to_luminance_for_solve(primary)
    except Exception as exc:
        return _failure(f"failed to read FITS data: {exc}")
    height, width = image.shape
    ra_keys = ("RA", "OBJCTRA", "OBJRA", "OBJ_RA", "CRVAL1")
    dec_keys = ("DEC", "OBJCTDEC", "OBJDEC", "OBJ_DEC", "CRVAL2")
    ra0 = _extract_near_center_angle(header, ra_keys, is_ra=True, strict_astap_iso=strict_astap_iso)
    dec0 = _extract_near_center_angle(header, dec_keys, is_ra=False, strict_astap_iso=strict_astap_iso)
    if ra0 is None or dec0 is None:
        return _failure("metadata RA/DEC missing for near solve")
    scale_arcsec, (fov_x, fov_y) = estimate_scale_and_fov(header, width, height)
    fov_candidates = [value for value in (fov_x, fov_y) if value is not None]
    approx_fov = max(fov_candidates) if fov_candidates else None
    fov_hint_source = "scale"

    # Optional per-frame FOV hint in FITS headers, keeps hint-by-FOV workflows.
    # Supported keys are degrees for full frame height/major axis.
    header_fov = _extract_float(
        header,
        (
            "FOV",
            "FOVDEG",
            "FOVH",
            "FOVY",
            "FIELD",
            "FIELDDEG",
        ),
    )
    if header_fov is not None:
        approx_fov = float(header_fov)
        fov_hint_source = "header"

    if cfg.fov_override_deg and cfg.fov_override_deg > 0:
        approx_fov = float(cfg.fov_override_deg)
        fov_hint_source = "override"

    approx_scale_deg = scale_arcsec / 3600.0 if scale_arcsec else None
    if approx_scale_deg is None:
        approx_fov = approx_fov or 1.5
        approx_scale_deg = approx_fov / max(width, height)
    approx_scale_deg = max(approx_scale_deg, 1e-5)
    scale_min_deg = approx_scale_deg * 0.60
    scale_max_deg = approx_scale_deg * 1.60
    fov_for_radius = approx_fov or (approx_scale_deg * max(width, height))
    radius = max(_MIN_SEARCH_RADIUS, 1.0 * fov_for_radius * max(cfg.search_margin, 1.0))
    radius = min(radius, _MAX_SEARCH_RADIUS)
    hint_fastpath = bool(getattr(cfg, "astap_hint_fastpath", True))
    hint_radius_deg = float(getattr(cfg, "astap_hint_radius_deg", 3.0) or 0.0)
    if hint_fastpath and hint_radius_deg > 0:
        # Throughput-first hinted solve (ASTAP-like -r behavior).
        radius = max(_MIN_SEARCH_RADIUS, min(radius, float(hint_radius_deg)))
    logger.info(
        "near solve start for %s (radius=%.2f°, approx_scale=%.3g°/px, astap_iso_strict=%s)",
        fits_path.name,
        radius,
        approx_scale_deg,
        str(strict_astap_iso).lower(),
    )
    if cancel_check and cancel_check():
        return _failure("cancelled")
    provider_tiles_by_key: dict[str, NearCatalogTile] = {}
    if provider is None:
        candidates = _select_tiles(manifest or {}, ra0, dec0, radius, cfg.max_tile_candidates)
    else:
        try:
            selected_tiles = provider.select_tiles(
                float(ra0),
                float(dec0),
                float(radius),
                int(cfg.max_tile_candidates),
                families=(str(cfg.family).strip().lower(),) if cfg.family else None,
            )
        except NearCatalogProviderError as exc:
            return _failure(f"near catalog provider selection failed: {exc}")
        except Exception as exc:
            return _failure(f"near catalog provider selection failed: {exc}")
        provider_tiles_by_key = {tile.tile_key: tile for tile in selected_tiles}
        candidates = [tile.to_manifest_entry() for tile in selected_tiles]
    logger.info("near candidates selected: %d", len(candidates))
    provider_telemetry = provider.telemetry() if provider is not None else {
        "near_catalog_provider": "legacy_index",
        "near_catalog_fallback_used": False,
    }
    provider_telemetry["near_catalog_candidate_tiles"] = int(len(candidates))
    logger.info(
        "near catalog provider: provider=%s families=%s fallback_used=%s",
        provider_telemetry.get("near_catalog_provider"),
        provider_telemetry.get("near_catalog_family", []),
        str(provider_telemetry.get("near_catalog_fallback_used", False)).lower(),
    )
    catalog_positions: list[np.ndarray] = []
    catalog_world: list[np.ndarray] = []
    catalog_mags: list[np.ndarray] = []
    missing_tiles: list[str] = []
    # Resolve DB root for optional fallback when tile blobs are empty/missing
    try:
        db_root_text = str((manifest or {}).get("db_root", "")).strip()
        db_root_path: Path | None = Path(db_root_text).expanduser().resolve() if db_root_text else None
    except Exception:
        db_root_path = None

    db_for_stepping: CatalogDB | None = None
    db_family: str | None = None
    raw_tile_lookup: dict[tuple[str, str], object] | None = None
    if db_root_path is not None:
        try:
            if cfg.family:
                db_family = str(cfg.family).strip().lower()
            elif candidates:
                db_family = str(candidates[0].get("family") or "").strip().lower() or None
            db_for_stepping = _get_cached_catalog_db(db_root_path, db_family)
        except Exception:
            db_for_stepping = None

        if strict_astap_iso:
            try:
                raw_tile_lookup = _get_raw_tile_lookup(db_root_path, db_family)
            except Exception:
                raw_tile_lookup = None

    if strict_astap_iso and provider is not None:
        try:
            search_deg = max(float(radius), float(approx_fov or 1.0)) * 1.6
            cap = int(cfg.max_tile_candidates) if int(cfg.max_tile_candidates or 0) > 0 else len(candidates)
            selected_tiles = provider.select_tiles(
                float(ra0),
                float(dec0),
                float(search_deg),
                max(1, cap),
                families=(str(cfg.family).strip().lower(),) if cfg.family else None,
            )
            provider_tiles_by_key = {tile.tile_key: tile for tile in selected_tiles}
            candidates = [tile.to_manifest_entry() for tile in selected_tiles]
            logger.info("near strict candidates selected from provider: %d", len(candidates))
            provider_telemetry["near_catalog_candidate_tiles"] = int(len(candidates))
        except Exception as exc:
            return _failure(f"near catalog provider strict selection failed: {exc}")
    elif strict_astap_iso and raw_tile_lookup:
        try:
            search_deg = max(float(radius), float(approx_fov or 1.0)) * 1.6
            dec_min_q = max(-90.0, float(dec0) - search_deg)
            dec_max_q = min(90.0, float(dec0) + search_deg)
            cosd = max(1.0e-3, math.cos(math.radians(float(dec0))))
            ra_span = min(180.0, search_deg / cosd)

            def _norm_ra_local(v: float) -> float:
                r = float(v) % 360.0
                return r + 360.0 if r < 0 else r

            def _segments_local(a: float, b: float) -> tuple[tuple[float, float], ...]:
                span = float(b) - float(a)
                if abs(span) >= 360.0:
                    return ((0.0, 360.0),)
                s0 = _norm_ra_local(a)
                s1 = _norm_ra_local(b)
                if s1 >= s0:
                    return ((s0, s1),)
                return ((s0, 360.0), (0.0, s1))

            def _overlap_local(a0: float, a1: float, b0: float, b1: float) -> bool:
                return (a0 <= b1) and (b0 <= a1)

            qsegs = _segments_local(float(ra0) - ra_span, float(ra0) + ra_span)
            strict_entries: list[dict] = []
            for (fam, code), tm in raw_tile_lookup.items():
                if db_family and fam != db_family:
                    continue
                b = tm.bounds
                if b.dec_max < dec_min_q or b.dec_min > dec_max_q:
                    continue
                if not b.covers_full_ra:
                    hit = False
                    for ts, te in b.ra_segments:
                        if any(_overlap_local(float(ts), float(te), float(qs), float(qe)) for qs, qe in qsegs):
                            hit = True
                            break
                    if not hit:
                        continue
                try:
                    sep_deg = float(_angular_distance(float(tm.center_ra_deg), float(tm.center_dec_deg), float(ra0), float(dec0)))
                except Exception:
                    sep_deg = float("inf")
                strict_entries.append(
                    {
                        "family": fam,
                        "tile_code": str(code),
                        "tile_key": f"{fam}_{code}" if fam and code else str(code),
                        "center_ra_deg": float(tm.center_ra_deg),
                        "center_dec_deg": float(tm.center_dec_deg),
                        "_sep_deg": sep_deg,
                    }
                )

            strict_entries.sort(key=lambda e: float(e.get("_sep_deg", float("inf"))))
            cap = int(cfg.max_tile_candidates) if int(cfg.max_tile_candidates or 0) > 0 else len(strict_entries)
            candidates = [{k: v for k, v in e.items() if k != "_sep_deg"} for e in strict_entries[: max(1, cap)]]
            logger.info("near strict candidates selected from ASTAP DB: %d", len(candidates))
            provider_telemetry["near_catalog_candidate_tiles"] = int(len(candidates))
        except Exception:
            pass

    if not candidates:
        if provider is not None:
            return _failure("near catalog provider returned no tile intersecting the metadata cone")
        return _failure("manifest present but no tile intersects the metadata cone")

    for entry in candidates:
        if cancel_check and cancel_check():
            return _failure("cancelled")

        if provider is not None:
            try:
                tile = provider_tiles_by_key.get(str(entry.get("tile_key") or ""))
                if tile is None:
                    raise NearCatalogProviderError(f"selected tile not found: {entry.get('tile_key')}")
                stars_provider = provider.load_stars(tile)
                if stars_provider.size > 0:
                    ra_deg = stars_provider.ra_deg.astype(np.float64, copy=False)
                    dec_deg = stars_provider.dec_deg.astype(np.float64, copy=False)
                    mags_raw = stars_provider.mag.astype(np.float32, copy=False)
                    x_deg, y_deg = project_tan(ra_deg, dec_deg, float(ra0), float(dec0))
                    valid = np.isfinite(x_deg) & np.isfinite(y_deg)
                    if np.any(valid):
                        positions = np.column_stack((x_deg[valid], y_deg[valid])).astype(np.float32, copy=False)
                        world = np.column_stack((ra_deg[valid], dec_deg[valid])).astype(np.float64, copy=False)
                        mags = mags_raw[valid]
                        catalog_positions.append(positions)
                        catalog_world.append(world)
                        catalog_mags.append(mags)
                        continue
            except Exception as exc:
                return _failure(f"near catalog provider load failed: {exc}")

        if strict_astap_iso and db_root_path is not None and raw_tile_lookup is not None:
            try:
                fam = str(entry.get("family") or "").strip().lower()
                code = str(entry.get("tile_code") or "")
                tm = raw_tile_lookup.get((fam, code))
                if tm is not None:
                    stars_raw = load_astap_tile_stars(db_root_path, tm)
                    if stars_raw.size > 0:
                        ra_deg = stars_raw["ra_deg"].astype(np.float64, copy=False)
                        dec_deg = stars_raw["dec_deg"].astype(np.float64, copy=False)
                        mags_raw = stars_raw["mag"].astype(np.float32, copy=False)
                        x_deg, y_deg = project_tan(ra_deg, dec_deg, float(ra0), float(dec0))
                        valid = np.isfinite(x_deg) & np.isfinite(y_deg)
                        if np.any(valid):
                            positions = np.column_stack((x_deg[valid], y_deg[valid])).astype(np.float32, copy=False)
                            world = np.column_stack((ra_deg[valid], dec_deg[valid])).astype(np.float64, copy=False)
                            mags = mags_raw[valid]
                            catalog_positions.append(positions)
                            catalog_world.append(world)
                            catalog_mags.append(mags)
                            continue
            except Exception:
                pass

        try:
            if index_path is None:
                raise FileNotFoundError(str(entry.get("tile_key") or "provider_tile"))
            positions, world, mags = _load_tile_catalog(
                index_path,
                entry,
                ra0,
                dec0,
                db_root=db_root_path,
                cache_cap=cfg.tile_cache_size,
            )
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
    provider_telemetry["near_catalog_loaded_tiles"] = int(len(catalog_positions))
    provider_telemetry["near_catalog_loaded_stars"] = int(sum(arr.shape[0] for arr in catalog_positions))
    logger.info("near catalog stacks: tiles=%d stars=%d", len(catalog_positions), int(sum(arr.shape[0] for arr in catalog_positions)))
    cat_world = np.vstack(catalog_world)
    cat_mags = np.concatenate(catalog_mags)

    # Keep full stacked catalog for ASTAP-like center stepping.
    cat_positions_all = cat_positions
    cat_world_all = cat_world
    cat_mags_all = cat_mags

    # ASTAP-like local database window around hinted center (square window in tangent plane).
    if strict_astap_iso:
        # Mirror ASTAP default oversize~1 for star-rich fields.
        search_window_deg = max(0.4, float(approx_fov or (approx_scale_deg * max(width, height))))
    else:
        search_window_deg = max(0.4, float((approx_fov or (approx_scale_deg * max(width, height))) * max(cfg.search_margin, 1.0)))
        if hint_fastpath and hint_radius_deg > 0:
            # Keep local window aligned with hinted radius to reduce catalog pressure.
            search_window_deg = min(search_window_deg, max(0.8, 2.0 * float(hint_radius_deg)))
    half_window = 0.5 * search_window_deg
    local_mask = (
        np.isfinite(cat_positions[:, 0]) & np.isfinite(cat_positions[:, 1])
        & (np.abs(cat_positions[:, 0]) <= half_window)
        & (np.abs(cat_positions[:, 1]) <= half_window)
    )
    if np.any(local_mask):
        cat_positions = cat_positions[local_mask]
        cat_world = cat_world[local_mask]
        cat_mags = cat_mags[local_mask]
    logger.info("near catalog local window: %.2f deg square -> stars=%d", search_window_deg, int(cat_positions.shape[0]))

    # ASTAP uses a standard-coordinate reference system in arcseconds for catalog stars.
    # Keep image stars in pixels and map to catalog tangent-plane arcseconds.
    cat_positions_iso = cat_positions.astype(np.float64, copy=False) * 3600.0
    if (not strict_astap_iso) and cfg.max_cat_stars and cat_positions.shape[0] > cfg.max_cat_stars:
        k = int(cfg.max_cat_stars)
        n = int(cat_positions.shape[0])
        # Faster than full argsort on large catalogs while keeping deterministic
        # ordering of the retained subset by magnitude.
        if n > (4 * k):
            keep = np.argpartition(cat_mags, k - 1)[:k]
            keep = keep[np.argsort(cat_mags[keep], kind="stable")]
        else:
            order = np.argsort(cat_mags, kind="stable")
            keep = order[:k]
        cat_positions = cat_positions[keep]
        cat_positions_iso = cat_positions_iso[keep]
        cat_world = cat_world[keep]
        cat_mags = cat_mags[keep]
    if cancel_check and cancel_check():
        return _failure("cancelled")
    strict_db_target_stars: int | None = None
    logger.info("near detect start")
    t_detect0 = time.perf_counter()
    detect_backend = str(getattr(cfg, "detect_backend", "auto") or "auto").lower()
    work_image = image
    if detect_backend != "astap" and not strict_astap_iso:
        try:
            kernel = max(7, int(round(min(height, width) / 120)))
            work_image = remove_background(work_image, kernel_size=kernel)
        except TypeError:
            work_image = remove_background(work_image)
        except Exception:
            work_image = image
    detect_device = getattr(cfg, "detect_device", None)
    detect_k_sigma = float(getattr(cfg, "detect_k_sigma", 4.0) or 4.0)
    detect_min_area = int(getattr(cfg, "detect_min_area", 8) or 8)
    detect_max_labels = int(getattr(cfg, "detect_max_labels", 2500) or 2500)
    if strict_astap_iso and detect_backend != "astap":
        # ASTAP-like detector defaults in strict mirror mode.
        detect_k_sigma = min(detect_k_sigma, 3.0)
        detect_max_labels = min(detect_max_labels, 500)
    detect_trace: dict[str, str] = {}
    detect_gpu_slots = max(1, int(getattr(cfg, "detect_gpu_slots", 1) or 1))

    detect_image = work_image
    detect_coord_scale = 1.0
    if strict_astap_iso and detect_backend == "legacy_strict_binned":
        # ASTAP commonly runs the solver detector on a binned grayscale image.
        # Emulate the dominant x2 case in strict mirror mode.
        try:
            detect_image, detect_coord_scale = _mean_bin_image(work_image, 2)
        except Exception:
            detect_image = work_image
            detect_coord_scale = 1.0

    if detect_backend == "astap":
        astap_bin_factor = max(1, int(getattr(cfg, "astap_extract_bin_factor", 2) or 1))
        astap_bin_strict_only = bool(getattr(cfg, "astap_extract_bin_strict_only", True))
        astap_bin_min_stars = max(0, int(getattr(cfg, "astap_extract_bin_min_stars", 12) or 0))
        astap_use_binned = astap_bin_factor > 1 and (strict_astap_iso or not astap_bin_strict_only)

        if astap_use_binned:
            stars, used_scale = _detect_stars_astap_cli_binned(
                work_image,
                fits_path,
                bin_factor=astap_bin_factor,
                snr_min=10,
                timeout_s=180,
            )
            if used_scale > 1.0:
                detect_trace["used"] = f"astap-bin{int(used_scale)}"
                if stars.size < astap_bin_min_stars:
                    stars_full = _detect_stars_astap_cli(fits_path, snr_min=10, timeout_s=180)
                    if stars_full.size > stars.size:
                        stars = stars_full
                        detect_trace["fallback"] = "astap_fullres_retry"
                        detect_trace["used"] = "astap"
            else:
                detect_trace["used"] = "astap"
        else:
            stars = _detect_stars_astap_cli(fits_path, snr_min=10, timeout_s=180)
            detect_trace["used"] = "astap"
    elif strict_astap_iso:
        stars, astap_detect_diag = astap_adaptive_image_detection(
            image,
            bin_factor=2,
            max_stars=500,
            hfd_min=0.8,
            crop=1.0,
        )
        detect_trace["used"] = "astap_adaptive"
        detect_trace["raw_candidates"] = str(astap_detect_diag.get("raw_candidates", ""))
        detect_trace["selected_count"] = str(astap_detect_diag.get("selected_count", ""))
    else:
        use_gpu_slot_guard = False
        if detect_backend in {"cuda", "auto"}:
            ready, reason = _cuda_runtime_ready()
            if ready:
                use_gpu_slot_guard = True
            elif detect_backend == "cuda":
                logger.warning("near detect CUDA runtime not ready (%s) -> fallback to CPU expected", reason)
        if use_gpu_slot_guard:
            with _gpu_detect_semaphore(detect_gpu_slots):
                stars = detect_stars(
                    detect_image,
                    backend=detect_backend,
                    device=detect_device,
                    mode="global",
                    k_sigma=detect_k_sigma,
                    min_area=detect_min_area,
                    max_labels=detect_max_labels,
                    backend_trace=detect_trace,
                )
        else:
            stars = detect_stars(
                detect_image,
                backend=detect_backend,
                device=detect_device,
                mode="global",
                k_sigma=detect_k_sigma,
                min_area=detect_min_area,
                max_labels=detect_max_labels,
                backend_trace=detect_trace,
            )
    logger.info(
        "near detect backend used: requested=%s used=%s device=%s gpu_slots=%d%s%s",
        detect_backend,
        detect_trace.get("used", "unknown"),
        detect_device,
        detect_gpu_slots,
        " (fallback)" if detect_trace.get("fallback") else "",
        f" reason={detect_trace.get('error')}" if detect_trace.get("error") else "",
    )
    # Faint/edge frames can be over-suppressed by aggressive background removal.
    # Retry on raw image with progressively looser thresholds only when support is low.
    if (not strict_astap_iso) and detect_backend != "astap" and stars.size < 12:
        logger.info("near detect fallback #1 (raw/global k=3.0), stars=%d", int(stars.size))
        detect_trace_fb1: dict[str, str] = {}
        stars_fb = detect_stars(
            image,
            backend=detect_backend,
            device=detect_device,
            mode="global",
            k_sigma=max(2.2, detect_k_sigma - 1.0),
            min_area=5,
            max_labels=2500,
            backend_trace=detect_trace_fb1,
        )
        logger.info(
            "near detect fallback #1 backend used: requested=%s used=%s device=%s gpu_slots=%d%s%s",
            detect_backend,
            detect_trace_fb1.get("used", "unknown"),
            detect_device,
            detect_gpu_slots,
            " (fallback)" if detect_trace_fb1.get("fallback") else "",
            f" reason={detect_trace_fb1.get('error')}" if detect_trace_fb1.get("error") else "",
        )
        if stars_fb.size > stars.size:
            stars = stars_fb
            # Raw-image fallback already returns full-resolution coordinates.
            # Do not re-apply the strict-mode bin2 coordinate scale afterwards.
            detect_coord_scale = 1.0
    if (not strict_astap_iso) and detect_backend != "astap" and stars.size < 8:
        logger.info("near detect fallback #2 (raw/global k=2.5), stars=%d", int(stars.size))
        detect_trace_fb2: dict[str, str] = {}
        stars_fb2 = detect_stars(
            image,
            backend=detect_backend,
            device=detect_device,
            mode="global",
            k_sigma=max(2.0, detect_k_sigma - 1.5),
            min_area=4,
            max_labels=3500,
            backend_trace=detect_trace_fb2,
        )
        logger.info(
            "near detect fallback #2 backend used: requested=%s used=%s device=%s gpu_slots=%d%s%s",
            detect_backend,
            detect_trace_fb2.get("used", "unknown"),
            detect_device,
            detect_gpu_slots,
            " (fallback)" if detect_trace_fb2.get("fallback") else "",
            f" reason={detect_trace_fb2.get('error')}" if detect_trace_fb2.get("error") else "",
        )
        if stars_fb2.size > stars.size:
            stars = stars_fb2
            # Raw-image fallback already returns full-resolution coordinates.
            # Do not re-apply the strict-mode bin2 coordinate scale afterwards.
            detect_coord_scale = 1.0
    if detect_coord_scale != 1.0 and stars.size > 0:
        stars = stars.copy()
        x_full, y_full = astap_binned_to_full_coords(stars["x"], stars["y"], int(detect_coord_scale))
        stars["x"] = np.asarray(x_full, dtype=np.float32)
        stars["y"] = np.asarray(y_full, dtype=np.float32)

    t_detect_s = time.perf_counter() - t_detect0
    if stars.size == 0:
        return _failure("no stars detected in the frame")
    logger.info("near detected stars: %d", int(stars.size))
    # Mirror ASTAP behavior more closely by working on brightest image stars.
    # Strict ASTAP-ISO already receives stars in ASTAP scan/selection order.
    # Keep that order explicit via img_ranks below; do not encode it as fake flux.
    if (not strict_astap_iso) and stars.size > 1:
        try:
            order = np.argsort(stars["flux"], kind="stable")[::-1]
            stars = stars[order]
        except Exception:
            pass
    if (not strict_astap_iso) and cfg.max_img_stars and stars.size > cfg.max_img_stars:
        stars = stars[: cfg.max_img_stars]
    injected_stars = _load_near_diagnostic_image_stars_csv(getattr(cfg, "diagnostic_image_stars_csv", None))
    if injected_stars is not None:
        logger.info("near diagnostic image-star injection: %d -> %d stars", int(stars.size), int(injected_stars.size))
        stars = injected_stars

    injected_catalog = _load_near_diagnostic_catalog_stars_csv(getattr(cfg, "diagnostic_catalog_stars_csv", None))
    if injected_catalog is not None:
        cat_positions, cat_world, cat_mags = injected_catalog
        cat_positions_iso = cat_positions.astype(np.float64, copy=False) * 3600.0
        cat_positions_all = cat_positions
        cat_world_all = cat_world
        cat_mags_all = cat_mags
        logger.info("near diagnostic catalog-star injection: selected=%d", int(cat_positions.shape[0]))

    if strict_astap_iso and cat_positions.shape[0] > 0 and injected_catalog is None:
        # Mirror ASTAP database star request count (nrstars_required2).
        nrstars_image = int(stars.size)
        nrstars_required = max(32, int(round(float(nrstars_image) * (float(height) / max(1.0, float(width))))))
        if nrstars_image < 35:
            oversize = 2.0
        elif nrstars_image > 140:
            oversize = 1.0
        else:
            oversize = 2.0 * math.sqrt(35.0 / max(float(nrstars_image), 1.0))
        nrstars_required2 = max(64, int(round(float(nrstars_required) * oversize * oversize)))
        strict_db_target_stars = int(nrstars_required2)

        # Mirror ASTAP square-search window: window = oversize * fov2, capped by
        # the catalog tile size (5.142857° for .1476, 9.53° for .290).
        try:
            fov2_deg = float(approx_fov or (approx_scale_deg * max(width, height)))
            fov2_deg = max(0.4, fov2_deg)
            max_fov_deg = 5.142857
            if raw_tile_lookup is not None and len(candidates) > 0:
                fam0 = str(candidates[0].get("family") or "").strip().lower()
                code0 = str(candidates[0].get("tile_code") or "")
                tm0 = raw_tile_lookup.get((fam0, code0))
                if tm0 is not None and str(getattr(tm0, "path", "")).lower().endswith(".290"):
                    max_fov_deg = 9.53
            search_window_deg = min(float(max_fov_deg), max(0.4, float(fov2_deg) * float(oversize)))
            half_window = 0.5 * float(search_window_deg)
            local_mask_strict = (
                np.isfinite(cat_positions_all[:, 0]) & np.isfinite(cat_positions_all[:, 1])
                & (np.abs(cat_positions_all[:, 0]) <= half_window)
                & (np.abs(cat_positions_all[:, 1]) <= half_window)
            )
            if np.any(local_mask_strict):
                cat_positions = cat_positions_all[local_mask_strict]
                cat_world = cat_world_all[local_mask_strict]
                cat_mags = cat_mags_all[local_mask_strict]
            else:
                cat_positions = cat_positions_all
                cat_world = cat_world_all
                cat_mags = cat_mags_all
            cat_positions_iso = cat_positions.astype(np.float64, copy=False) * 3600.0
            logger.info(
                "near strict astap-iso window: fov2=%.2f oversize=%.2f window=%.2f stars=%d",
                float(fov2_deg),
                float(oversize),
                float(search_window_deg),
                int(cat_positions.shape[0]),
            )
        except Exception:
            pass

        # ASTAP-like read_stars emulation: read stars from up to four corner areas
        # with cumulative quotas, preserving on-disk order inside each tile.
        if db_root_path is not None and raw_tile_lookup is not None and len(candidates) > 0:
            try:
                ra0d = float(ra0)
                dec0d = float(dec0)
                fov_deg = float(search_window_deg)
                half = 0.5 * fov_deg

                def _norm_ra(v: float) -> float:
                    r = float(v) % 360.0
                    return r + 360.0 if r < 0 else r

                def _delta_ra_deg(a: float, b: float) -> float:
                    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
                    return d

                def _point_in_tile(tm, ra_deg: float, dec_deg: float) -> bool:
                    if dec_deg < float(tm.bounds.dec_min) or dec_deg > float(tm.bounds.dec_max):
                        return False
                    if tm.bounds.covers_full_ra:
                        return True
                    r = _norm_ra(ra_deg)
                    for s0, s1 in tm.bounds.ra_segments:
                        ss0 = float(s0)
                        ss1 = float(s1)
                        if ss0 <= ss1:
                            if ss0 <= r <= ss1:
                                return True
                        else:
                            if r >= ss0 or r <= ss1:
                                return True
                    return False

                def _tile_for_corner(ra_c: float, dec_c: float):
                    for ent in candidates:
                        fam = str(ent.get("family") or "").strip().lower()
                        code = str(ent.get("tile_code") or "")
                        tm = raw_tile_lookup.get((fam, code))
                        if tm is None:
                            continue
                        if _point_in_tile(tm, ra_c, dec_c):
                            return fam, code, tm
                    return None

                dec_n = dec0d + half
                dec_s = dec0d - half
                cos_n = max(1.0e-6, math.cos(math.radians(dec_n)))
                cos_s = max(1.0e-6, math.cos(math.radians(dec_s)))
                ra_en = _norm_ra(ra0d + half / cos_n)
                ra_wn = _norm_ra(ra0d - half / cos_n)
                ra_es = _norm_ra(ra0d + half / cos_s)
                ra_ws = _norm_ra(ra0d - half / cos_s)

                # ASTAP corner order: EN, WN, ES, WS
                area_defs = [
                    (ra_en, dec_n, "W", "S"),
                    (ra_wn, dec_n, "E", "S"),
                    (ra_es, dec_s, "W", "N"),
                    (ra_ws, dec_s, "E", "N"),
                ]

                area_rows: list[tuple[str, str, object, float]] = []
                used_keys: set[tuple[str, str]] = set()

                for ra_c, dec_c, x_dir, y_dir in area_defs:
                    found = _tile_for_corner(ra_c, dec_c)
                    if found is None:
                        continue
                    fam, code, tm = found
                    key = (fam, code)
                    if key in used_keys:
                        continue
                    used_keys.add(key)

                    # Approximate ASTAP area fraction from distances to tile boundaries.
                    b = tm.bounds
                    if tm.bounds.covers_full_ra:
                        space_e = space_w = fov_deg
                    else:
                        # pick containing RA segment
                        r = _norm_ra(ra_c)
                        seg = None
                        for s0, s1 in tm.bounds.ra_segments:
                            ss0 = float(s0); ss1 = float(s1)
                            inside = (ss0 <= r <= ss1) if ss0 <= ss1 else (r >= ss0 or r <= ss1)
                            if inside:
                                seg = (ss0, ss1)
                                break
                        if seg is None:
                            seg = tuple(map(float, tm.bounds.ra_segments[0]))
                        ss0, ss1 = seg
                        if ss0 <= ss1:
                            east = (ss1 - r)
                            west = (r - ss0)
                        else:
                            east = (ss1 - r) % 360.0
                            west = (r - ss0) % 360.0
                        cosc = max(1.0e-6, math.cos(math.radians(dec_c)))
                        space_e = east * cosc
                        space_w = west * cosc

                    space_n = float(b.dec_max) - float(dec_c)
                    space_s = float(dec_c) - float(b.dec_min)

                    sx = space_w if x_dir == "W" else space_e
                    sy = space_s if y_dir == "S" else space_n
                    frac = max(0.0, min(float(sx), fov_deg) * min(float(sy), fov_deg) / max(1.0e-9, fov_deg * fov_deg))
                    if frac >= 0.01:
                        area_rows.append((fam, code, tm, float(frac)))

                # Fallback if corners did not yield enough unique areas.
                if not area_rows:
                    for ent in candidates[:4]:
                        fam = str(ent.get("family") or "").strip().lower()
                        code = str(ent.get("tile_code") or "")
                        tm = raw_tile_lookup.get((fam, code))
                        if tm is not None:
                            area_rows.append((fam, code, tm, 1.0))

                selected_world: list[np.ndarray] = []
                selected_mag: list[np.ndarray] = []
                selected_n = 0
                frac_sum = sum(fr for *_x, fr in area_rows) if area_rows else 0.0
                cum_frac = 0.0
                cos0 = max(1.0e-6, math.cos(math.radians(dec0d)))

                for fam, code, tm, frac in area_rows:
                    cum_frac += (float(frac) / frac_sum) if frac_sum > 0 else (1.0 / max(1, len(area_rows)))
                    target_n = min(int(nrstars_required2), int(math.trunc(float(nrstars_required2) * cum_frac)))
                    need = max(0, target_n - selected_n)
                    if need <= 0:
                        continue

                    stars_raw = load_astap_tile_stars(db_root_path, tm)
                    if stars_raw.size == 0:
                        continue
                    ra_deg = stars_raw["ra_deg"].astype(np.float64, copy=False)
                    dec_deg = stars_raw["dec_deg"].astype(np.float64, copy=False)
                    mag = stars_raw["mag"].astype(np.float32, copy=False)
                    dra = ((ra_deg - ra0d + 180.0) % 360.0) - 180.0
                    mask = (np.abs(dra * cos0) < half) & (np.abs(dec_deg - dec0d) < half)
                    if not np.any(mask):
                        continue
                    idx = np.flatnonzero(mask)
                    take = idx[:need]
                    if take.size == 0:
                        continue
                    selected_world.append(np.column_stack((ra_deg[take], dec_deg[take])).astype(np.float64, copy=False))
                    selected_mag.append(mag[take].astype(np.float32, copy=False))
                    selected_n += int(take.size)
                    if selected_n >= int(nrstars_required2):
                        break

                if selected_world:
                    cat_world = np.vstack(selected_world)
                    cat_mags = np.concatenate(selected_mag)
                    x_deg, y_deg = project_tan(cat_world[:, 0], cat_world[:, 1], float(ra0), float(dec0))
                    m = np.isfinite(x_deg) & np.isfinite(y_deg)
                    cat_world = cat_world[m]
                    cat_mags = cat_mags[m]
                    cat_positions = np.column_stack((x_deg[m], y_deg[m])).astype(np.float32, copy=False)
                    cat_positions_iso = cat_positions.astype(np.float64, copy=False) * 3600.0
            except Exception:
                pass

        if int(cat_positions.shape[0]) > int(nrstars_required2):
            # Safety cap: preserve incoming order (ASTAP-like read order) in strict mode.
            keep = np.arange(int(nrstars_required2), dtype=np.int64)
            cat_positions = cat_positions[keep]
            cat_positions_iso = cat_positions_iso[keep]
            cat_world = cat_world[keep]
            cat_mags = cat_mags[keep]

        logger.info(
            "near strict astap-iso db stars target: requested=%d selected=%d",
            int(nrstars_required2),
            int(cat_positions.shape[0]),
        )

    image_positions = np.column_stack((stars["x"], stars["y"])).astype(np.float32, copy=False)
    if strict_astap_iso:
        img_ranks = np.arange(int(stars.size), dtype=np.float64)
    else:
        img_ranks = _compute_ranks(stars["flux"], descending=True)
    cat_ranks = _compute_ranks(cat_mags, descending=False)
    _dump_near_diagnostic_lists(
        dump_dir=getattr(cfg, "diagnostic_dump_dir", None),
        label=str(getattr(cfg, "diagnostic_dump_label", None) or fits_path.stem),
        stars=stars,
        cat_positions=cat_positions,
        cat_world=cat_world,
        cat_mags=cat_mags,
    )

    # ASTAP ISO hypothesis path first (quad hash on-the-fly), before legacy pair pipeline.
    if strict_astap_iso:
        # ASTAP plate-solve rule: minimum_quads := 3 + nrstars_image div 140
        minimum_quads = max(3, 3 + int(image_positions.shape[0]) // 140)
    else:
        minimum_quads = max(3, min(12, 3 + int(image_positions.shape[0]) // 140))
    iso_diag_initial: dict = {}
    iso_diag_best: dict | None = None
    iso_diag_second: dict | None = None
    iso_diag_autoscale: list[dict] = []
    iso_diag_autofov: list[dict] = []
    iso_ratio_min = float(getattr(cfg, "astap_iso_scale_ratio_min", 0.55) or 0.55)
    iso_ratio_max = float(getattr(cfg, "astap_iso_scale_ratio_max", 1.80) or 1.80)
    if not math.isfinite(iso_ratio_min):
        iso_ratio_min = 0.55
    if not math.isfinite(iso_ratio_max):
        iso_ratio_max = 1.80
    iso_ratio_min = max(0.05, float(iso_ratio_min))
    iso_ratio_max = max(float(iso_ratio_min) + 1.0e-6, float(iso_ratio_max))
    iso_diag_agg: dict = {
        "calls": 0,
        "ok_calls": 0,
        "max_refs": 0,
        "max_quick_inliers": -1,
        "path_counts": {},
    }

    def _acc_iso_diag(d: dict | None) -> None:
        if not d:
            return
        iso_diag_agg["calls"] = int(iso_diag_agg.get("calls", 0)) + 1
        path = d.get("path_used")
        if isinstance(path, str) and path:
            pc = dict(iso_diag_agg.get("path_counts", {}))
            pc[path] = int(pc.get(path, 0)) + 1
            iso_diag_agg["path_counts"] = pc
        sel = d.get("selected") if isinstance(d.get("selected"), dict) else None
        if sel is not None:
            try:
                iso_diag_agg["max_refs"] = max(int(iso_diag_agg.get("max_refs", 0)), int(sel.get("best_refs", 0) or 0))
            except Exception:
                pass
            try:
                qi = int(sel.get("best_quick_inliers", -1) or -1)
                iso_diag_agg["max_quick_inliers"] = max(int(iso_diag_agg.get("max_quick_inliers", -1)), qi)
                if qi >= 0:
                    iso_diag_agg["ok_calls"] = int(iso_diag_agg.get("ok_calls", 0)) + 1
            except Exception:
                pass

    def _new_iso_trace() -> dict | None:
        return {} if bool(getattr(cfg, "diagnostic_iso_trace", False)) else None

    def _dump_iso_trace_stage(
        stage: str,
        trace: dict | None,
        diag: dict | None,
        minimum: int | None = None,
        *,
        catalog_world_for_trace: np.ndarray | None = None,
        catalog_mags_for_trace: np.ndarray | None = None,
        stage_meta: dict | None = None,
    ) -> None:
        if trace is None:
            return
        try:
            img_src = np.asarray(trace.get("image_source_indices", np.empty(0, dtype=np.int64)), dtype=np.int64)
            if img_src.size and stars.size:
                fluxes = np.full(int(img_src.size), np.nan, dtype=np.float64)
                ok = (img_src >= 0) & (img_src < int(stars.size))
                if np.any(ok):
                    fluxes[ok] = np.asarray(stars["flux"], dtype=np.float64)[img_src[ok]]
                trace["image_fluxes"] = fluxes
        except Exception:
            pass
        try:
            cat_src = np.asarray(trace.get("catalog_source_indices", np.empty(0, dtype=np.int64)), dtype=np.int64)
            if catalog_world_for_trace is not None and cat_src.size:
                cw_arr = np.asarray(catalog_world_for_trace, dtype=np.float64)
                world = np.full((int(cat_src.size), 2), np.nan, dtype=np.float64)
                ok = (cat_src >= 0) & (cat_src < int(cw_arr.shape[0]))
                if np.any(ok):
                    world[ok] = cw_arr[cat_src[ok], :2]
                trace["catalog_world"] = world
            if catalog_mags_for_trace is not None and cat_src.size:
                cm_arr = np.asarray(catalog_mags_for_trace, dtype=np.float64)
                mags_trace = np.full(int(cat_src.size), np.nan, dtype=np.float64)
                ok = (cat_src >= 0) & (cat_src < int(cm_arr.shape[0]))
                if np.any(ok):
                    mags_trace[ok] = cm_arr[cat_src[ok]]
                trace["catalog_mags"] = mags_trace
        except Exception:
            pass
        if isinstance(stage_meta, dict):
            trace["stage_meta"] = {
                k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)) else v)
                for k, v in stage_meta.items()
            }
        _dump_astap_iso_trace(
            dump_dir=getattr(cfg, "diagnostic_dump_dir", None),
            label=str(getattr(cfg, "diagnostic_dump_label", None) or fits_path.stem),
            stage=stage,
            trace=trace,
            diag=diag,
            minimum_count=int(minimum if minimum is not None else minimum_quads),
            quad_tolerance=float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
        )

    iso_trace_initial = _new_iso_trace()
    iso_transform, iso_matrix, iso_offset, iso_refs = _astap_iso_hypothesis(
        image_positions,
        cat_positions_iso,
        img_ranks=img_ranks,
        cat_ranks=cat_ranks,
        expected_scale_arcsec=scale_arcsec,
        expected_scale_ratio_min=iso_ratio_min,
        expected_scale_ratio_max=iso_ratio_max,
        minimum_count=int(minimum_quads),
        strict_astap_iso=bool(strict_astap_iso),
        quad_tolerance=float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
        diag=iso_diag_initial,
        trace=iso_trace_initial,
    )
    _dump_iso_trace_stage(
        "initial",
        iso_trace_initial,
        iso_diag_initial,
        catalog_world_for_trace=cat_world,
        catalog_mags_for_trace=cat_mags,
        stage_meta={
            "stage_type": "initial",
            "center_ra_deg": float(ra0),
            "center_dec_deg": float(dec0),
            "tangent_center_ra_deg": float(ra0),
            "tangent_center_dec_deg": float(dec0),
            "fov_deg": float(approx_fov or (approx_scale_deg * max(width, height))),
            "search_window_deg": float(search_window_deg),
            "search_radius_deg": float(radius),
            "strict_db_target_stars": int(strict_db_target_stars) if strict_db_target_stars is not None else None,
        },
    )
    _acc_iso_diag(iso_diag_initial)
    if iso_transform is not None:
        iso_diag_best = {"stage": "initial", **dict(iso_diag_initial)}

    initial_selected = iso_diag_initial.get("selected") if isinstance(iso_diag_initial.get("selected"), dict) else {}
    initial_best_refs = int(initial_selected.get("best_refs", 0) or 0)
    initial_tolerances = iso_diag_initial.get("tolerances") if isinstance(iso_diag_initial.get("tolerances"), list) else []
    initial_max_matches_raw = 0
    for _tr in initial_tolerances:
        if not isinstance(_tr, dict):
            continue
        try:
            initial_max_matches_raw = max(initial_max_matches_raw, int(_tr.get("matches_raw", 0) or 0))
        except Exception:
            continue
    try:
        initial_stars_img = int(iso_diag_initial.get("stars_img", 0) or 0)
    except Exception:
        initial_stars_img = 0
    try:
        initial_quads_img = int(iso_diag_initial.get("quads_img", 0) or 0)
    except Exception:
        initial_quads_img = 0

    # Non-strict auto-scale retry: if hinted scale appears wrong, retry with
    # alternate scale hypotheses before falling back to blind astrometry.
    if (
        (not strict_astap_iso)
        and bool(getattr(cfg, "astap_iso_auto_scale_retry", True))
        and (iso_transform is None or iso_matrix is None or iso_offset is None)
        and (scale_arcsec is not None and float(scale_arcsec) > 0)
        and (initial_stars_img >= 16)
        and (initial_quads_img >= 3)
    ):
        try:
            retry_factors_cfg = getattr(cfg, "astap_iso_auto_scale_factors", (0.50, 2.00, 0.75, 1.33, 1.60))
            retry_factors: list[float] = []
            if isinstance(retry_factors_cfg, str):
                for part in retry_factors_cfg.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        fv = float(part)
                    except Exception:
                        continue
                    if math.isfinite(fv) and fv > 0:
                        retry_factors.append(fv)
            else:
                try:
                    for item in retry_factors_cfg:
                        try:
                            fv = float(item)
                        except Exception:
                            continue
                        if math.isfinite(fv) and fv > 0:
                            retry_factors.append(fv)
                except Exception:
                    retry_factors = []

            if not retry_factors:
                retry_factors = [0.50, 2.00, 0.75, 1.33, 1.60]

            seen_factors: set[int] = set()
            for fac in retry_factors:
                if abs(float(fac) - 1.0) <= 1.0e-6:
                    continue
                key = int(round(float(fac) * 1000.0))
                if key in seen_factors:
                    continue
                seen_factors.add(key)

                expected_retry = float(scale_arcsec) * float(fac)
                if not math.isfinite(expected_retry) or expected_retry <= 0:
                    continue

                logger.info(
                    "near auto-scale retry: trying expected_scale=%.3f\"/px (factor=%.3f)",
                    float(expected_retry),
                    float(fac),
                )
                diag_retry: dict = {}
                trace_retry = _new_iso_trace()
                t_retry, m_retry, o_retry, r_retry = _astap_iso_hypothesis(
                    image_positions,
                    cat_positions_iso,
                    img_ranks=img_ranks,
                    cat_ranks=cat_ranks,
                    expected_scale_arcsec=expected_retry,
                    expected_scale_ratio_min=iso_ratio_min,
                    expected_scale_ratio_max=iso_ratio_max,
                    minimum_count=int(minimum_quads),
                    strict_astap_iso=False,
                    quad_tolerance=float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
                    diag=diag_retry,
                    trace=trace_retry,
                )
                _dump_iso_trace_stage(
                    f"auto_scale_factor_{float(fac):.3f}",
                    trace_retry,
                    diag_retry,
                    catalog_world_for_trace=cat_world,
                    catalog_mags_for_trace=cat_mags,
                    stage_meta={
                        "stage_type": "auto_scale",
                        "center_ra_deg": float(ra0),
                        "center_dec_deg": float(dec0),
                        "tangent_center_ra_deg": float(ra0),
                        "tangent_center_dec_deg": float(dec0),
                        "fov_deg": float(approx_fov or (approx_scale_deg * max(width, height))),
                        "search_window_deg": float(search_window_deg),
                        "search_radius_deg": float(radius),
                        "scale_factor": float(fac),
                    },
                )
                _acc_iso_diag(diag_retry)
                iso_diag_autoscale.append(
                    {
                        "mode": "scaled_expected",
                        "factor": float(fac),
                        "expected_scale_arcsec": float(expected_retry),
                        "refs": int(r_retry),
                        "ok": bool(t_retry is not None and m_retry is not None and o_retry is not None),
                    }
                )
                if t_retry is not None and m_retry is not None and o_retry is not None:
                    iso_transform = t_retry
                    iso_matrix = m_retry
                    iso_offset = o_retry
                    iso_refs = int(r_retry)
                    iso_diag_best = {
                        "stage": "auto_scale_retry",
                        "factor": float(fac),
                        "expected_scale_arcsec": float(expected_retry),
                        **dict(diag_retry),
                    }
                    break

            if (
                (iso_transform is None or iso_matrix is None or iso_offset is None)
                and bool(getattr(cfg, "astap_iso_auto_scale_last_chance_no_gate", True))
            ):
                logger.info("near auto-scale retry: last chance without expected-scale gate")
                diag_retry: dict = {}
                trace_retry = _new_iso_trace()
                t_retry, m_retry, o_retry, r_retry = _astap_iso_hypothesis(
                    image_positions,
                    cat_positions_iso,
                    img_ranks=img_ranks,
                    cat_ranks=cat_ranks,
                    expected_scale_arcsec=None,
                    expected_scale_ratio_min=iso_ratio_min,
                    expected_scale_ratio_max=iso_ratio_max,
                    minimum_count=int(minimum_quads),
                    strict_astap_iso=False,
                    quad_tolerance=float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
                    diag=diag_retry,
                    trace=trace_retry,
                )
                _dump_iso_trace_stage(
                    "auto_scale_no_gate",
                    trace_retry,
                    diag_retry,
                    catalog_world_for_trace=cat_world,
                    catalog_mags_for_trace=cat_mags,
                    stage_meta={
                        "stage_type": "auto_scale",
                        "center_ra_deg": float(ra0),
                        "center_dec_deg": float(dec0),
                        "tangent_center_ra_deg": float(ra0),
                        "tangent_center_dec_deg": float(dec0),
                        "fov_deg": float(approx_fov or (approx_scale_deg * max(width, height))),
                        "search_window_deg": float(search_window_deg),
                        "search_radius_deg": float(radius),
                        "scale_factor": None,
                    },
                )
                _acc_iso_diag(diag_retry)
                iso_diag_autoscale.append(
                    {
                        "mode": "no_scale_gate",
                        "factor": None,
                        "expected_scale_arcsec": None,
                        "refs": int(r_retry),
                        "ok": bool(t_retry is not None and m_retry is not None and o_retry is not None),
                    }
                )
                if t_retry is not None and m_retry is not None and o_retry is not None:
                    iso_transform = t_retry
                    iso_matrix = m_retry
                    iso_offset = o_retry
                    iso_refs = int(r_retry)
                    iso_diag_best = {
                        "stage": "auto_scale_retry_no_gate",
                        **dict(diag_retry),
                    }
        except Exception:
            pass

    # Strict auto-FOV retry (no external oracle): only when no explicit FOV hint
    # was provided and the first strict attempt failed.
    if (
        strict_astap_iso
        and bool(getattr(cfg, "strict_auto_fov_retry", True))
        and fov_hint_source == "scale"
        and (iso_transform is None or iso_matrix is None or iso_offset is None)
        and (initial_stars_img >= 24)
        and (initial_quads_img >= 3)
    ):
        try:
            max_fov_auto = 5.142857
            if raw_tile_lookup is not None and len(candidates) > 0:
                fam0 = str(candidates[0].get("family") or "").strip().lower()
                code0 = str(candidates[0].get("tile_code") or "")
                tm0 = raw_tile_lookup.get((fam0, code0))
                if tm0 is not None and str(getattr(tm0, "path", "")).lower().endswith(".290"):
                    max_fov_auto = 9.53
            base_fov = max(0.4, float(approx_fov or 1.0))
            retry_scales_cfg = getattr(cfg, "strict_auto_fov_retry_scales", (1.25, 0.82, 1.6, 0.65, 2.4, 4.0))
            retry_scales: list[float] = []
            if isinstance(retry_scales_cfg, str):
                for part in retry_scales_cfg.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        v = float(part)
                    except Exception:
                        continue
                    if math.isfinite(v) and v > 0:
                        retry_scales.append(v)
            else:
                try:
                    for item in retry_scales_cfg:
                        try:
                            v = float(item)
                        except Exception:
                            continue
                        if math.isfinite(v) and v > 0:
                            retry_scales.append(v)
                except Exception:
                    retry_scales = []
            if not retry_scales:
                retry_scales = [1.25, 0.82, 1.6, 0.65, 2.4, 4.0]

            # Context-aware ordering/budget:
            # if initial hypothesis had no refs and no raw matches, prefer expanding
            # windows first and cap attempts to keep cost bounded.
            no_initial_clue = (initial_best_refs <= 0 and initial_max_matches_raw <= 0)
            if no_initial_clue:
                retry_scales = [v for v in retry_scales if v >= 1.0] + [v for v in retry_scales if v < 1.0]

            seen_windows = {round(float(search_window_deg), 6)}
            seen_subset_fp: set[tuple[int, int, int, int]] = set()
            cfg_max_attempts = int(getattr(cfg, "strict_auto_fov_retry_max_attempts", 0) or 0)
            context_max_attempts = 3 if no_initial_clue else 0
            max_attempts = cfg_max_attempts if cfg_max_attempts > 0 else context_max_attempts
            zero_ref_patience = int(getattr(cfg, "strict_auto_fov_retry_zero_ref_patience", 3) or 0)
            consecutive_zero_refs = 0
            attempts = 0
            for sc in retry_scales:
                if max_attempts > 0 and attempts >= max_attempts:
                    iso_diag_autofov.append({"window_deg": None, "stars_cat": 0, "ok": False, "skip": "max_attempts_reached"})
                    break
                win = max(0.4, min(float(max_fov_auto), float(base_fov) * float(sc)))
                key = round(float(win), 6)
                if key in seen_windows:
                    continue
                seen_windows.add(key)
                half = 0.5 * float(win)
                local_mask_retry = (
                    np.isfinite(cat_positions_all[:, 0])
                    & np.isfinite(cat_positions_all[:, 1])
                    & (np.abs(cat_positions_all[:, 0]) <= half)
                    & (np.abs(cat_positions_all[:, 1]) <= half)
                )
                if not np.any(local_mask_retry):
                    iso_diag_autofov.append({"window_deg": float(win), "stars_cat": 0, "ok": False, "skip": "empty_window"})
                    continue

                idx = np.flatnonzero(local_mask_retry)
                if strict_db_target_stars is not None and int(idx.size) > int(strict_db_target_stars):
                    idx = idx[: int(strict_db_target_stars)]
                if idx.size == 0:
                    iso_diag_autofov.append({"window_deg": float(win), "stars_cat": 0, "ok": False, "skip": "empty_after_cap"})
                    continue

                if idx.size <= 64:
                    sample = idx.astype(np.int64, copy=False)
                else:
                    sample = np.concatenate((idx[:32], idx[-32:])).astype(np.int64, copy=False)
                fp = (int(idx.size), int(sample.sum()), int(idx[0]), int(idx[-1]))
                if fp in seen_subset_fp:
                    iso_diag_autofov.append({"window_deg": float(win), "stars_cat": int(idx.size), "ok": False, "skip": "duplicate_subset"})
                    continue
                seen_subset_fp.add(fp)

                cp = cat_positions_all[idx]
                cw = cat_world_all[idx]
                cm = cat_mags_all[idx]
                cp_iso = cp.astype(np.float64, copy=False) * 3600.0
                cr = _compute_ranks(cm, descending=False)
                diag_retry: dict = {}
                attempts += 1
                trace_retry = _new_iso_trace()
                t_retry, m_retry, o_retry, r_retry = _astap_iso_hypothesis(
                    image_positions,
                    cp_iso,
                    img_ranks=img_ranks,
                    cat_ranks=cr,
                    expected_scale_arcsec=scale_arcsec,
                    expected_scale_ratio_min=iso_ratio_min,
                    expected_scale_ratio_max=iso_ratio_max,
                    minimum_count=int(minimum_quads),
                    strict_astap_iso=bool(strict_astap_iso),
                    quad_tolerance=float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
                    diag=diag_retry,
                    trace=trace_retry,
                )
                _dump_iso_trace_stage(
                    f"autofov_{attempts}_win_{float(win):.6f}",
                    trace_retry,
                    diag_retry,
                    catalog_world_for_trace=cw,
                    catalog_mags_for_trace=cm,
                    stage_meta={
                        "stage_type": "autofov",
                        "attempt_order": int(attempts),
                        "center_ra_deg": float(ra0),
                        "center_dec_deg": float(dec0),
                        "tangent_center_ra_deg": float(ra0),
                        "tangent_center_dec_deg": float(dec0),
                        "fov_deg": float(approx_fov or (approx_scale_deg * max(width, height))),
                        "search_window_deg": float(win),
                        "search_radius_deg": float(radius),
                        "strict_db_target_stars": int(strict_db_target_stars) if strict_db_target_stars is not None else None,
                    },
                )
                _acc_iso_diag(diag_retry)
                iso_diag_autofov.append(
                    {
                        "window_deg": float(win),
                        "stars_cat": int(cp.shape[0]),
                        "refs": int(r_retry),
                        "ok": bool(t_retry is not None and m_retry is not None and o_retry is not None),
                    }
                )
                if int(r_retry) <= 0:
                    consecutive_zero_refs += 1
                else:
                    consecutive_zero_refs = 0

                if (
                    t_retry is None
                    and m_retry is None
                    and o_retry is None
                    and zero_ref_patience > 0
                    and consecutive_zero_refs >= zero_ref_patience
                ):
                    iso_diag_autofov.append(
                        {
                            "window_deg": None,
                            "stars_cat": int(cp.shape[0]),
                            "ok": False,
                            "skip": "zero_refs_patience",
                            "consecutive_zero_refs": int(consecutive_zero_refs),
                        }
                    )
                    break

                if t_retry is not None and m_retry is not None and o_retry is not None:
                    iso_transform = t_retry
                    iso_matrix = m_retry
                    iso_offset = o_retry
                    iso_refs = int(r_retry)
                    cat_positions = cp
                    cat_world = cw
                    cat_mags = cm
                    cat_positions_iso = cp_iso
                    cat_ranks = cr
                    search_window_deg = float(win)
                    iso_diag_best = {"stage": "auto_fov_retry", "window_deg": float(win), **dict(diag_retry)}
                    break
        except Exception:
            pass
    elif strict_astap_iso and fov_hint_source == "scale" and (iso_transform is None or iso_matrix is None or iso_offset is None):
        iso_diag_autofov.append({
            "window_deg": None,
            "stars_cat": int(cat_positions.shape[0]) if cat_positions is not None else 0,
            "ok": False,
            "skip": "low_support",
            "stars_img": int(initial_stars_img),
            "quads_img": int(initial_quads_img),
        })

    iso_center_ra = float(ra0)
    iso_center_dec = float(dec0)

    # ASTAP-like center stepping around hint/tile centers (only when needed).
    need_center_stepping = True
    if iso_transform is not None and iso_matrix is not None and iso_offset is not None:
        try:
            sc0 = float(getattr(iso_transform, "scale", 0.0))
            if scale_arcsec is not None and scale_arcsec > 0:
                ratio0 = sc0 / float(scale_arcsec)
            else:
                ratio0 = float("inf")
            center_xy0 = np.array([[0.5 * float(width), 0.5 * float(height)]], dtype=np.float64)
            std0 = _apply_affine_points(iso_matrix, iso_offset, center_xy0)
            ra_c0, dec_c0 = _standard_equatorial_astap(float(iso_center_ra), float(iso_center_dec), std0[:, 0], std0[:, 1])
            off0 = float(_angular_distance(float(ra_c0[0]), float(dec_c0[0]), float(ra0), float(dec0)))
            off_max0 = max(float(radius) + 0.8, 1.8 * float(approx_fov or 1.0))
            need_center_stepping = not ((0.35 <= ratio0 <= 2.8) and (off0 <= off_max0) and (int(iso_refs) >= 16))
        except Exception:
            need_center_stepping = True
    if strict_astap_iso:
        # Mirror ASTAP: if a first-center solution already exists, keep it and skip spiral.
        need_center_stepping = not (iso_transform is not None and iso_matrix is not None and iso_offset is not None)
    elif hint_fastpath:
        # Throughput-first mode: skip center stepping entirely.
        # Unsolved frames will be handled by later blind phase in batch workflow.
        need_center_stepping = False

    if need_center_stepping:
        centers: list[tuple[float, float]] = []
        centers.append((float(ra0), float(dec0)))

        if strict_astap_iso:
            # ASTAP-like: search by square spiral around the hinted center.
            step = max(0.2, float(approx_fov or 1.0))
            max_distance = max(1, int(round(float(radius) / max(step, 1e-6))))
            sx = sy = 0
            dx, dy = 0, -1
            max_iter = (2 * max_distance + 1) ** 2
            for _ in range(max_iter):
                if (-max_distance <= sx <= max_distance) and (-max_distance <= sy <= max_distance):
                    cdec = float(dec0) + float(step) * float(sy)
                    if -89.9 <= cdec <= 89.9:
                        cosd = max(0.2, math.cos(math.radians(cdec)))
                        cra = float(ra0) + (float(step) * float(sx)) / cosd
                        try:
                            sep = float(_angular_distance(float(cra), float(cdec), float(ra0), float(dec0)))
                        except Exception:
                            sep = float('inf')
                        if sep <= float(radius) + 0.5 * float(step):
                            centers.append((cra, cdec))
                if (sx == sy) or (sx < 0 and sx == -sy) or (sx > 0 and sx == 1 - sy):
                    dx, dy = -dy, dx
                sx += dx
                sy += dy
        else:
            tile_seed_count = 3 if hint_fastpath else 8
            for ent in candidates[: min(len(candidates), tile_seed_count)]:
                try:
                    centers.append((float(ent.get("center_ra_deg", ra0)), float(ent.get("center_dec_deg", dec0))))
                except Exception:
                    pass
            if hint_fastpath:
                # Small cross pattern around the hint (ASTAP-like local retry),
                # avoids full 3x3 stepping grid for speed.
                step = max(0.12, min(0.45, 0.35 * float(approx_fov or 1.0)))
                cosd = max(0.2, math.cos(math.radians(float(dec0))))
                centers.extend([
                    (float(ra0) - float(step) / cosd, float(dec0)),
                    (float(ra0) + float(step) / cosd, float(dec0)),
                    (float(ra0), float(dec0) - float(step)),
                    (float(ra0), float(dec0) + float(step)),
                ])
            else:
                step = max(0.2, float(approx_fov or 1.0))
                # Local 3x3 stencil around hint
                for dy in (-0.5 * step, 0.0, 0.5 * step):
                    for dx in (-0.5 * step, 0.0, 0.5 * step):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        cosd = max(0.2, math.cos(math.radians(float(dec0))))
                        centers.append((float(ra0) + float(dx) / cosd, float(dec0) + float(dy)))

                # ASTAP-like square-spiral stepping over the hinted search radius.
                max_distance = max(1, int(round(float(radius) / max(step, 1e-6))))
                sx = sy = 0
                dx, dy = 0, -1
                max_iter = (2 * max_distance + 1) ** 2
                for _ in range(max_iter):
                    if (-max_distance <= sx <= max_distance) and (-max_distance <= sy <= max_distance):
                        cdec = float(dec0) + float(step) * float(sy)
                        if -89.9 <= cdec <= 89.9:
                            cosd = max(0.2, math.cos(math.radians(cdec)))
                            cra = float(ra0) + (float(step) * float(sx)) / cosd
                            try:
                                sep = float(_angular_distance(float(cra), float(cdec), float(ra0), float(dec0)))
                            except Exception:
                                sep = float('inf')
                            if sep <= float(radius) + 0.5 * float(step):
                                centers.append((cra, cdec))
                    if (sx == sy) or (sx < 0 and sx == -sy) or (sx > 0 and sx == 1 - sy):
                        dx, dy = -dy, dx
                    sx += dx
                    sy += dy

        seen_centers: set[tuple[int, int]] = set()
        best_tuple: tuple[SimilarityTransform | None, np.ndarray | None, np.ndarray | None, int, float] = (None, None, None, 0, float("inf"))
        best_bad = True
        best_center_off = float("inf")
        best_eval_inliers = 0
        best_eval_rms = float("inf")

        if iso_transform is not None and iso_matrix is not None and iso_offset is not None:
            sc0 = float(getattr(iso_transform, "scale", 0.0))
            if scale_arcsec is not None and scale_arcsec > 0:
                scale_err0 = abs(math.log(max(sc0, 1e-9) / float(scale_arcsec)))
                ratio0 = sc0 / float(scale_arcsec)
            else:
                scale_err0 = abs(math.log(max(sc0, 1e-9)))
                ratio0 = float("inf")
            try:
                center_xy0 = np.array([[0.5 * float(width), 0.5 * float(height)]], dtype=np.float64)
                std0 = _apply_affine_points(iso_matrix, iso_offset, center_xy0)
                ra_c0, dec_c0 = _standard_equatorial_astap(float(iso_center_ra), float(iso_center_dec), std0[:, 0], std0[:, 1])
                off0 = float(_angular_distance(float(ra_c0[0]), float(dec_c0[0]), float(ra0), float(dec0)))
            except Exception:
                off0 = float("inf")
            off_max = max(float(radius) + 0.8, 1.8 * float(approx_fov or 1.0))
            _eval_matches0, eval_inliers0, eval_rms0 = _build_matches_from_affine(
                iso_matrix,
                iso_offset,
                image_positions,
                cat_positions_iso,
                cat_world,
                pixel_tolerance=float(cfg.pixel_tolerance),
                max_matches=300,
            )
            bad0 = not (0.4 <= ratio0 <= 2.5 and off0 <= off_max and int(eval_inliers0) >= 4)
            best_tuple = (iso_transform, iso_matrix, iso_offset, int(iso_refs), float(scale_err0))
            best_bad = bool(bad0)
            best_center_off = float(off0)
            best_eval_inliers = int(eval_inliers0)
            best_eval_rms = float(eval_rms0)

        for cra, cdec in centers:
            key = (int(round(cra * 1e4)), int(round(cdec * 1e4)))
            if key in seen_centers:
                continue
            seen_centers.add(key)

            # Prefer direct DB query per center (ASTAP-like), fallback to stacked local catalog.
            pos_deg = None
            wld = None
            mags = None
            if db_for_stepping is not None:
                try:
                    dec_min = max(-90.0, float(cdec) - half_window)
                    dec_max = min(+90.0, float(cdec) + half_window)
                    cosd = max(1e-3, math.cos(math.radians(float(cdec))))
                    ra_span = half_window / cosd
                    stars_box = db_for_stepping.query_box(
                        float(cra) - ra_span,
                        float(cra) + ra_span,
                        dec_min,
                        dec_max,
                        families=[db_family] if db_family else None,
                        max_stars=int(max(200, min(6000, (int(stars.size) * 8)))) if int(stars.size) > 0 else 800,
                    )
                    if stars_box.size > 0:
                        x_deg, y_deg = project_tan(stars_box["ra_deg"], stars_box["dec_deg"], float(cra), float(cdec))
                        m = np.isfinite(x_deg) & np.isfinite(y_deg)
                        if np.any(m):
                            pos_deg = np.column_stack((x_deg[m], y_deg[m])).astype(np.float32, copy=False)
                            wld = np.column_stack((stars_box["ra_deg"][m], stars_box["dec_deg"][m])).astype(np.float64, copy=False)
                            mags = stars_box["mag"][m].astype(np.float32, copy=False)
                except Exception:
                    pos_deg = None

            if pos_deg is None or wld is None or mags is None or pos_deg.shape[0] == 0:
                x_deg, y_deg = project_tan(cat_world_all[:, 0], cat_world_all[:, 1], float(cra), float(cdec))
                m = np.isfinite(x_deg) & np.isfinite(y_deg)
                if not np.any(m):
                    continue
                pos_deg = np.column_stack((x_deg[m], y_deg[m])).astype(np.float32, copy=False)
                wld = cat_world_all[m].astype(np.float64, copy=False)
                mags = cat_mags_all[m].astype(np.float32, copy=False)

                lm = (
                    np.isfinite(pos_deg[:, 0]) & np.isfinite(pos_deg[:, 1])
                    & (np.abs(pos_deg[:, 0]) <= half_window)
                    & (np.abs(pos_deg[:, 1]) <= half_window)
                )
                if not np.any(lm):
                    continue
                pos_deg = pos_deg[lm]
                wld = wld[lm]
                mags = mags[lm]

            if strict_astap_iso and strict_db_target_stars is not None and pos_deg.shape[0] > strict_db_target_stars:
                k = int(strict_db_target_stars)
                keep = np.argsort(mags, kind="stable")[:k]
                pos_deg = pos_deg[keep]
                wld = wld[keep]
                mags = mags[keep]
            elif (not strict_astap_iso) and cfg.max_cat_stars and pos_deg.shape[0] > cfg.max_cat_stars:
                k = int(cfg.max_cat_stars)
                n = int(pos_deg.shape[0])
                if n > (4 * k):
                    keep = np.argpartition(mags, k - 1)[:k]
                    keep = keep[np.argsort(mags[keep], kind="stable")]
                else:
                    keep = np.argsort(mags, kind="stable")[:k]
                pos_deg = pos_deg[keep]
                wld = wld[keep]
                mags = mags[keep]

            pos_px = pos_deg.astype(np.float64, copy=False) * 3600.0
            cat_r_tmp = _compute_ranks(mags, descending=False)
            iso_diag_tmp: dict = {}
            iso_trace_tmp = _new_iso_trace()
            tr_tmp, M_tmp, t_tmp, refs_tmp = _astap_iso_hypothesis(
                image_positions,
                pos_px,
                img_ranks=img_ranks,
                cat_ranks=cat_r_tmp,
                expected_scale_arcsec=scale_arcsec,
                expected_scale_ratio_min=iso_ratio_min,
                expected_scale_ratio_max=iso_ratio_max,
                minimum_count=int(minimum_quads),
                strict_astap_iso=bool(strict_astap_iso),
                quad_tolerance=float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
                diag=iso_diag_tmp,
                trace=iso_trace_tmp,
            )
            _dump_iso_trace_stage(
                f"spiral_{len(seen_centers)}",
                iso_trace_tmp,
                iso_diag_tmp,
                catalog_world_for_trace=wld,
                catalog_mags_for_trace=mags,
                stage_meta={
                    "stage_type": "spiral",
                    "attempt_order": int(len(seen_centers)),
                    "center_ra_deg": float(cra),
                    "center_dec_deg": float(cdec),
                    "tangent_center_ra_deg": float(cra),
                    "tangent_center_dec_deg": float(cdec),
                    "initial_ra_deg": float(ra0),
                    "initial_dec_deg": float(dec0),
                    "offset_from_initial_deg": float(_angular_distance(float(cra), float(cdec), float(ra0), float(dec0))),
                    "fov_deg": float(approx_fov or (approx_scale_deg * max(width, height))),
                    "search_window_deg": float(search_window_deg),
                    "search_radius_deg": float(radius),
                    "strict_db_target_stars": int(strict_db_target_stars) if strict_db_target_stars is not None else None,
                },
            )
            _acc_iso_diag(iso_diag_tmp)
            if tr_tmp is None or M_tmp is None or t_tmp is None:
                if int(refs_tmp) > int(best_tuple[3]):
                    best_tuple = (best_tuple[0], best_tuple[1], best_tuple[2], int(refs_tmp), best_tuple[4])
                continue

            if strict_astap_iso:
                # Mirror ASTAP spiral search: keep first solved center and stop stepping.
                best_tuple = (tr_tmp, M_tmp, t_tmp, int(refs_tmp), 0.0)
                cat_positions = pos_deg
                cat_positions_iso = pos_px
                cat_world = wld
                cat_mags = mags
                cat_ranks = cat_r_tmp
                iso_center_ra = float(cra)
                iso_center_dec = float(cdec)
                iso_diag_best = {
                    "stage": "center_stepping",
                    "center_ra": float(cra),
                    "center_dec": float(cdec),
                    "eval_inliers": None,
                    "eval_rms": None,
                    **dict(iso_diag_tmp),
                }
                break

            sc = float(getattr(tr_tmp, "scale", 0.0))
            if scale_arcsec is not None and scale_arcsec > 0:
                score = abs(math.log(max(sc, 1e-9) / float(scale_arcsec)))
                ratio = sc / float(scale_arcsec)
            else:
                score = abs(math.log(max(sc, 1e-9)))
                ratio = float("inf")
            try:
                center_xy = np.array([[0.5 * float(width), 0.5 * float(height)]], dtype=np.float64)
                std_xy = _apply_affine_points(M_tmp, t_tmp, center_xy)
                ra_c, dec_c = _standard_equatorial_astap(float(cra), float(cdec), std_xy[:, 0], std_xy[:, 1])
                center_off = float(_angular_distance(float(ra_c[0]), float(dec_c[0]), float(ra0), float(dec0)))
            except Exception:
                center_off = float("inf")
            off_max = max(float(radius) + 0.8, 1.8 * float(approx_fov or 1.0))
            eval_matches, eval_inliers, eval_rms = _build_matches_from_affine(
                M_tmp,
                t_tmp,
                image_positions,
                pos_px,
                wld,
                pixel_tolerance=float(cfg.pixel_tolerance),
                max_matches=300,
            )
            cand_bad = not (0.4 <= ratio <= 2.5 and center_off <= off_max and int(eval_inliers) >= 4)

            better = False
            if best_tuple[0] is None:
                better = True
            elif (best_bad and not cand_bad):
                better = True
            elif (best_bad == cand_bad):
                if int(eval_inliers) > int(best_eval_inliers):
                    better = True
                elif int(eval_inliers) == int(best_eval_inliers):
                    if float(eval_rms) < float(best_eval_rms) - 1e-9:
                        better = True
                    elif abs(float(eval_rms) - float(best_eval_rms)) <= 1e-9:
                        if int(refs_tmp) > int(best_tuple[3]):
                            better = True
                        elif int(refs_tmp) == int(best_tuple[3]):
                            if score < float(best_tuple[4]) - 1e-12:
                                better = True
                            elif abs(score - float(best_tuple[4])) <= 1e-12 and center_off < best_center_off:
                                better = True

            if better:
                best_tuple = (tr_tmp, M_tmp, t_tmp, int(refs_tmp), float(score))
                best_bad = bool(cand_bad)
                best_center_off = float(center_off)
                best_eval_inliers = int(eval_inliers)
                best_eval_rms = float(eval_rms)
                # carry catalog slice used by best center
                cat_positions = pos_deg
                cat_positions_iso = pos_px
                cat_world = wld
                cat_mags = mags
                cat_ranks = cat_r_tmp
                iso_center_ra = float(cra)
                iso_center_dec = float(cdec)
                iso_diag_best = {
                    "stage": "center_stepping",
                    "center_ra": float(cra),
                    "center_dec": float(cdec),
                    "eval_inliers": int(eval_inliers),
                    "eval_rms": float(eval_rms) if np.isfinite(eval_rms) else None,
                    **dict(iso_diag_tmp),
                }

        if best_tuple[0] is not None and best_tuple[1] is not None and best_tuple[2] is not None:
            iso_transform, iso_matrix, iso_offset, iso_refs = best_tuple[0], best_tuple[1], best_tuple[2], int(best_tuple[3])
        else:
            iso_refs = max(int(iso_refs), int(best_tuple[3]))

    # ASTAP-like second-pass refinement: re-center on solved image center and solve again.
    # Skip this pass for already strong/consistent hypotheses to reduce per-image latency.
    run_second_pass = bool(need_center_stepping or int(iso_refs) < 48)
    if strict_astap_iso:
        # ASTAP-style maximum-accuracy second solve (match_nr up to 2).
        run_second_pass = True
    elif hint_fastpath and not bool(getattr(cfg, "second_pass_refine_in_fastpath", False)):
        run_second_pass = False
    if iso_transform is not None and iso_matrix is not None and iso_offset is not None and run_second_pass:
        try:
            center_xy = np.array([[0.5 * float(width), 0.5 * float(height)]], dtype=np.float64)
            std_xy = _apply_affine_points(iso_matrix, iso_offset, center_xy)
            ra_c_arr, dec_c_arr = _standard_equatorial_astap(float(iso_center_ra), float(iso_center_dec), std_xy[:, 0], std_xy[:, 1])
            ra_ref = float(ra_c_arr[0])
            dec_ref = float(dec_c_arr[0])
        except Exception:
            ra_ref = dec_ref = float("nan")

        if math.isfinite(ra_ref) and math.isfinite(dec_ref):
            pos_deg = None
            wld = None
            mags = None
            if db_for_stepping is not None:
                try:
                    dec_min = max(-90.0, float(dec_ref) - half_window)
                    dec_max = min(+90.0, float(dec_ref) + half_window)
                    cosd = max(1e-3, math.cos(math.radians(float(dec_ref))))
                    ra_span = half_window / cosd
                    stars_box = db_for_stepping.query_box(
                        float(ra_ref) - ra_span,
                        float(ra_ref) + ra_span,
                        dec_min,
                        dec_max,
                        families=[db_family] if db_family else None,
                        max_stars=int(max(200, min(6000, (int(stars.size) * 8)))) if int(stars.size) > 0 else 800,
                    )
                    if stars_box.size > 0:
                        x_deg, y_deg = project_tan(stars_box["ra_deg"], stars_box["dec_deg"], float(ra_ref), float(dec_ref))
                        m = np.isfinite(x_deg) & np.isfinite(y_deg)
                        if np.any(m):
                            pos_deg = np.column_stack((x_deg[m], y_deg[m])).astype(np.float32, copy=False)
                            wld = np.column_stack((stars_box["ra_deg"][m], stars_box["dec_deg"][m])).astype(np.float64, copy=False)
                            mags = stars_box["mag"][m].astype(np.float32, copy=False)
                except Exception:
                    pos_deg = None

            if pos_deg is None or wld is None or mags is None or pos_deg.shape[0] == 0:
                x_deg, y_deg = project_tan(cat_world_all[:, 0], cat_world_all[:, 1], float(ra_ref), float(dec_ref))
                m = np.isfinite(x_deg) & np.isfinite(y_deg)
                if np.any(m):
                    pos_deg = np.column_stack((x_deg[m], y_deg[m])).astype(np.float32, copy=False)
                    wld = cat_world_all[m].astype(np.float64, copy=False)
                    mags = cat_mags_all[m].astype(np.float32, copy=False)

            if pos_deg is not None and wld is not None and mags is not None and pos_deg.shape[0] > 0:
                lm = (
                    np.isfinite(pos_deg[:, 0]) & np.isfinite(pos_deg[:, 1])
                    & (np.abs(pos_deg[:, 0]) <= half_window)
                    & (np.abs(pos_deg[:, 1]) <= half_window)
                )
                if np.any(lm):
                    pos_deg = pos_deg[lm]
                    wld = wld[lm]
                    mags = mags[lm]

                    if strict_astap_iso and iso_matrix is not None and iso_offset is not None and pos_deg.shape[0] > 0:
                        # ASTAP-like second-pass behavior: keep only database stars that
                        # project inside current image bounds according to pass-1 solution.
                        try:
                            pos_iso_vis = pos_deg.astype(np.float64, copy=False) * 3600.0
                            invM = np.linalg.inv(np.asarray(iso_matrix, dtype=np.float64))
                            img_xy = (pos_iso_vis - np.asarray(iso_offset, dtype=np.float64)[None, :]) @ invM.T
                            mvis = (
                                np.isfinite(img_xy[:, 0]) & np.isfinite(img_xy[:, 1])
                                & (img_xy[:, 0] >= -2.0) & (img_xy[:, 0] <= float(width) + 2.0)
                                & (img_xy[:, 1] >= -2.0) & (img_xy[:, 1] <= float(height) + 2.0)
                            )
                            if np.any(mvis):
                                pos_deg = pos_deg[mvis]
                                wld = wld[mvis]
                                mags = mags[mvis]
                        except Exception:
                            pass

                    if strict_astap_iso and strict_db_target_stars is not None and pos_deg.shape[0] > strict_db_target_stars:
                        k = int(strict_db_target_stars)
                        keep = np.argsort(mags, kind="stable")[:k]
                        pos_deg = pos_deg[keep]
                        wld = wld[keep]
                        mags = mags[keep]
                    elif (not strict_astap_iso) and cfg.max_cat_stars and pos_deg.shape[0] > cfg.max_cat_stars:
                        k = int(cfg.max_cat_stars)
                        n = int(pos_deg.shape[0])
                        if n > (4 * k):
                            keep = np.argpartition(mags, k - 1)[:k]
                            keep = keep[np.argsort(mags[keep], kind="stable")]
                        else:
                            keep = np.argsort(mags, kind="stable")[:k]
                        pos_deg = pos_deg[keep]
                        wld = wld[keep]
                        mags = mags[keep]

                    pos_iso = pos_deg.astype(np.float64, copy=False) * 3600.0
                    cat_r_tmp = _compute_ranks(mags, descending=False)
                    iso_diag_tmp2: dict = {}
                    iso_trace_tmp2 = _new_iso_trace()
                    tr2, M2, t2, refs2 = _astap_iso_hypothesis(
                        image_positions,
                        pos_iso,
                        img_ranks=img_ranks,
                        cat_ranks=cat_r_tmp,
                        expected_scale_arcsec=scale_arcsec,
                        expected_scale_ratio_min=iso_ratio_min,
                        expected_scale_ratio_max=iso_ratio_max,
                        minimum_count=int(minimum_quads),
                        strict_astap_iso=bool(strict_astap_iso),
                        quad_tolerance=float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
                        diag=iso_diag_tmp2,
                        trace=iso_trace_tmp2,
                    )
                    _dump_iso_trace_stage(
                        "recenter_second_pass",
                        iso_trace_tmp2,
                        iso_diag_tmp2,
                        catalog_world_for_trace=wld,
                        catalog_mags_for_trace=mags,
                        stage_meta={
                            "stage_type": "recenter_second_pass",
                            "center_ra_deg": float(ra_ref),
                            "center_dec_deg": float(dec_ref),
                            "tangent_center_ra_deg": float(ra_ref),
                            "tangent_center_dec_deg": float(dec_ref),
                            "initial_ra_deg": float(ra0),
                            "initial_dec_deg": float(dec0),
                            "fov_deg": float(approx_fov or (approx_scale_deg * max(width, height))),
                            "search_window_deg": float(search_window_deg),
                            "search_radius_deg": float(radius),
                            "strict_db_target_stars": int(strict_db_target_stars) if strict_db_target_stars is not None else None,
                        },
                    )
                    _acc_iso_diag(iso_diag_tmp2)
                    if tr2 is not None and M2 is not None and t2 is not None:
                        cur_scale_err = abs(math.log(max(float(getattr(iso_transform, "scale", 1e-9)), 1e-9) / max(float(scale_arcsec or 1.0), 1e-9)))
                        new_scale_err = abs(math.log(max(float(getattr(tr2, "scale", 1e-9)), 1e-9) / max(float(scale_arcsec or 1.0), 1e-9)))
                        refs2_i = int(refs2)
                        iso_refs_i = int(iso_refs)
                        if strict_astap_iso:
                            # Match ASTAP's second-pass intent: recenter then resolve for
                            # final geometry (maximum accuracy mode).
                            iso_transform, iso_matrix, iso_offset, iso_refs = tr2, M2, t2, refs2_i
                            iso_center_ra, iso_center_dec = float(ra_ref), float(dec_ref)
                            cat_positions = pos_deg
                            cat_positions_iso = pos_iso
                            cat_world = wld
                            cat_mags = mags
                            cat_ranks = cat_r_tmp
                            iso_diag_second = {
                                "stage": "second_pass",
                                "center_ra": float(ra_ref),
                                "center_dec": float(dec_ref),
                                **dict(iso_diag_tmp2),
                            }
                        else:
                            better_refs = refs2_i > (iso_refs_i + 2)
                            comparable_refs = refs2_i >= max(int(minimum_quads), int(0.75 * max(1, iso_refs_i)))
                            better_scale = (new_scale_err + 0.08) < cur_scale_err
                            if better_refs or (comparable_refs and better_scale) or (refs2_i == iso_refs_i and new_scale_err < cur_scale_err):
                                iso_transform, iso_matrix, iso_offset, iso_refs = tr2, M2, t2, refs2_i
                                iso_center_ra, iso_center_dec = float(ra_ref), float(dec_ref)
                                cat_positions = pos_deg
                                cat_positions_iso = pos_iso
                                cat_world = wld
                                cat_mags = mags
                                cat_ranks = cat_r_tmp
                                iso_diag_second = {
                                    "stage": "second_pass",
                                    "center_ra": float(ra_ref),
                                    "center_dec": float(dec_ref),
                                    **dict(iso_diag_tmp2),
                                }

    astap_iso_diag = {
        "initial": iso_diag_initial if iso_diag_initial else None,
        "best_center": iso_diag_best,
        "second_pass": iso_diag_second,
        "scale_ratio_gate": {"min": float(iso_ratio_min), "max": float(iso_ratio_max)},
        "auto_scale_retries": iso_diag_autoscale if iso_diag_autoscale else None,
        "auto_scale_retry_factors": getattr(cfg, "astap_iso_auto_scale_factors", (0.50, 2.00, 0.75, 1.33, 1.60)),
        "auto_fov_retries": iso_diag_autofov if iso_diag_autofov else None,
        "auto_fov_retry_scales": getattr(cfg, "strict_auto_fov_retry_scales", (1.25, 0.82, 1.6, 0.65, 2.4, 4.0)),
        "aggregate": iso_diag_agg,
        "strict_window_deg": float(search_window_deg) if strict_astap_iso else None,
        "strict_db_target_stars": int(strict_db_target_stars) if strict_db_target_stars is not None else None,
        "strict_db_selected_stars": int(cat_positions.shape[0]) if strict_astap_iso else None,
        "detect_coord_scale": float(detect_coord_scale),
        "fov_hint_source": str(fov_hint_source),
    }

    # Strict alignment target: near solve must follow ASTAP-like iso quad/hash core.
    if strict_astap_iso and (iso_transform is None or iso_matrix is None or iso_offset is None):
        _emit_near_debug_record({
            "event": "zenear_no_transform",
            "fits": str(fits_path),
            "reason": "no_astap_iso_hypothesis",
            "radius_deg": float(radius),
            "approx_scale_arcsec": float(scale_arcsec) if scale_arcsec is not None else None,
            "scale_min_deg": float(scale_min_deg),
            "scale_max_deg": float(scale_max_deg),
            "stars_detected": int(stars.size),
            "candidates": int(len(candidates)),
            "iso_refs": int(iso_refs),
            "astap_iso_diag": astap_iso_diag,
        })
        return _failure("near solver could not estimate a similarity transform")
    center_xy = (width / 2.0, height / 2.0)
    logger.info("near pair-build start")
    t_pair0 = time.perf_counter()
    approx_scale_for_pairs = float(scale_arcsec) if scale_arcsec is not None and float(scale_arcsec) > 0 else float(approx_scale_deg) * 3600.0
    img_pairs = np.empty((0, 2), dtype=np.float32)
    cat_pairs = np.empty((0, 2), dtype=np.float32)
    cat_world_pairs = np.empty((0, 2), dtype=np.float64)
    pair_specs = (
        (min(_MAX_NEIGHBORS, 3), 0.20),
        (_MAX_NEIGHBORS, _RANK_TOLERANCE),
    )
    if strict_astap_iso:
        logger.info("near strict astap-iso: skipping candidate-pair builder")
    else:
        for pass_idx, (pass_neighbors, pass_rank_tol) in enumerate(pair_specs, start=1):
            img_try, cat_try, world_try = _build_candidate_pairs(
                image_positions,
                cat_positions_iso,
                cat_world,
                img_ranks,
                cat_ranks,
                center_xy,
                approx_scale_for_pairs,
                cfg.pixel_tolerance,
                max_neighbors=int(pass_neighbors),
                rank_tolerance=float(pass_rank_tol),
                cancel_check=cancel_check,
            )
            if img_try.shape[0] > img_pairs.shape[0]:
                img_pairs, cat_pairs, cat_world_pairs = img_try, cat_try, world_try
            logger.info("near pair-build pass=%d neighbors=%d rank_tol=%.2f -> %d pairs", pass_idx, int(pass_neighbors), float(pass_rank_tol), int(img_try.shape[0]))
            if img_pairs.shape[0] >= 24:
                break
        if img_pairs.shape[0] < 6:
            sparse_neighbors = max(
                int(_MAX_NEIGHBORS),
                min(int(cat_positions_iso.shape[0]), 24),
            )
            img_try, cat_try, world_try = _build_candidate_pairs(
                image_positions,
                cat_positions_iso,
                cat_world,
                img_ranks,
                cat_ranks,
                center_xy,
                approx_scale_for_pairs,
                cfg.pixel_tolerance,
                max_neighbors=int(sparse_neighbors),
                rank_tolerance=1.0,
                cancel_check=cancel_check,
            )
            if img_try.shape[0] > img_pairs.shape[0]:
                img_pairs, cat_pairs, cat_world_pairs = img_try, cat_try, world_try
            logger.info(
                "near pair-build sparse fallback neighbors=%d rank_tol=%.2f -> %d pairs",
                int(sparse_neighbors),
                1.0,
                int(img_try.shape[0]),
            )
    t_pair_s = time.perf_counter() - t_pair0
    logger.info("near pair-build done: %d pairs", int(img_pairs.shape[0]))
    ransac_min_inliers = 4
    quality_target = int(getattr(cfg, "quality_inliers", 60) or 60)
    if img_pairs.shape[0] <= 8 or cat_pairs.shape[0] <= 8 or quality_target <= 3:
        ransac_min_inliers = 3
    ransac_min_scale = float(scale_min_deg) * 3600.0
    ransac_max_scale = float(scale_max_deg) * 3600.0
    if not (math.isfinite(ransac_min_scale) and math.isfinite(ransac_max_scale) and ransac_min_scale > 0 and ransac_max_scale > ransac_min_scale):
        ransac_min_scale = None
        ransac_max_scale = None
    if img_pairs.size == 0 and iso_transform is None:
        return _failure("unable to build candidate matches from metadata")
    if cancel_check and cancel_check():
        return _failure("cancelled")
    # Try ASTAP ISO transform first if available.
    t_ransac0 = time.perf_counter()
    used_transform: SimilarityTransform | None = None
    hypothesis = None
    if iso_transform is not None:
        if strict_astap_iso:
            logger.info(
                "near strict astap-iso hypothesis selected (refs=%d, quad_tolerance=%.4f)",
                int(iso_refs),
                float(getattr(cfg, "astap_iso_quad_tolerance", 0.007) or 0.007),
            )
            hypothesis = (iso_transform, None)
        else:
            iso_scale = float(getattr(iso_transform, "scale", 0.0) or 0.0)
            approx_scale_arcsec = float(scale_arcsec) if scale_arcsec is not None else float(approx_scale_deg) * 3600.0
            scale_ratio = (iso_scale / approx_scale_arcsec) if approx_scale_arcsec > 0 else float("nan")
            iso_refs_min = int(max(8, int(minimum_quads)))

            iso_eval_inliers = 0
            iso_eval_rms = float("inf")
            if iso_matrix is not None and iso_offset is not None:
                try:
                    _m_eval, iso_eval_inliers, iso_eval_rms = _build_matches_from_affine(
                        iso_matrix,
                        iso_offset,
                        image_positions,
                        cat_positions_iso,
                        cat_world,
                        pixel_tolerance=float(cfg.pixel_tolerance),
                        max_matches=300,
                    )
                except Exception:
                    iso_eval_inliers = 0
                    iso_eval_rms = float("inf")

            iso_ok = bool(
                int(iso_refs) >= iso_refs_min
                and math.isfinite(scale_ratio)
                and float(iso_ratio_min) <= scale_ratio <= float(iso_ratio_max)
            )

            if iso_ok:
                logger.info(
                    "near astap-iso hypothesis selected (refs=%d, scale_ratio=%.3f, eval_inliers=%d, eval_rms=%.3f)",
                    int(iso_refs),
                    float(scale_ratio) if math.isfinite(scale_ratio) else float('nan'),
                    int(iso_eval_inliers),
                    float(iso_eval_rms) if math.isfinite(iso_eval_rms) else float('nan'),
                )
                hypothesis = (iso_transform, None)
            else:
                logger.info(
                    "near astap-iso hypothesis rejected (refs=%d<%d or scale_ratio=%.3f not in [%.3f, %.3f])",
                    int(iso_refs),
                    int(iso_refs_min),
                    float(scale_ratio) if math.isfinite(scale_ratio) else float('nan'),
                    float(iso_ratio_min),
                    float(iso_ratio_max),
                )
    if (not strict_astap_iso) and img_pairs.shape[0] >= 6 and cat_pairs.shape[0] >= 6 and cfg.seed_rotation is not None and cfg.seed_scale_deg is not None:
        try:
            # One-shot LS fit with parity hint; validates via inlier thresholds below
            ls = estimate_similarity_RANSAC(
                img_pairs,
                cat_pairs,
                trials=1,
                tol_px=cfg.pixel_tolerance,
                min_inliers=ransac_min_inliers,
                allow_reflection=cfg.try_parity_flip,
                early_stop_inliers=int(getattr(cfg, "quality_inliers", 60) or 60),
                min_scale=ransac_min_scale,
                max_scale=ransac_max_scale,
                random_state=ransac_seed,
            )
            if ls is not None:
                used_transform, _ = ls
                hypothesis = ls
        except Exception:
            hypothesis = None
    if hypothesis is None and (not strict_astap_iso):
        quad_transform = _find_quad_hypothesis(
            image_positions,
            cat_positions_iso,
            img_ranks,
            cat_ranks,
            img_pairs,
            cat_pairs,
            min_scale=ransac_min_scale,
            max_scale=ransac_max_scale,
            pixel_tolerance=cfg.pixel_tolerance,
            allow_reflection=cfg.try_parity_flip,
            cancel_check=cancel_check,
        )
        if quad_transform is not None:
            q_scale = float(getattr(quad_transform, "scale", 0.0) or 0.0)
            if (ransac_min_scale is None or q_scale >= ransac_min_scale) and (ransac_max_scale is None or q_scale <= ransac_max_scale):
                logger.info("near quad-hypothesis accepted before ransac")
                hypothesis = (quad_transform, None)
            else:
                logger.info("near quad-hypothesis rejected (scale=%.6g deg/px outside [%.6g, %.6g])", q_scale, float(ransac_min_scale or float('nan')), float(ransac_max_scale or float('nan')))

    if hypothesis is None and (not strict_astap_iso):
        logger.info("near ransac start (trials=%d)", int(getattr(cfg, "ransac_trials", 1200) or 1200))
        hypothesis = estimate_similarity_RANSAC(
        img_pairs,
        cat_pairs,
        trials=int(getattr(cfg, "ransac_trials", 1200) or 1200),
        tol_px=cfg.pixel_tolerance,
        min_inliers=ransac_min_inliers,
        allow_reflection=cfg.try_parity_flip,
        early_stop_inliers=int(getattr(cfg, "quality_inliers", 60) or 60),
        min_scale=ransac_min_scale,
        max_scale=ransac_max_scale,
        random_state=(ransac_seed ^ 0x9E3779B9),
    )
        logger.info("near ransac done")
    if hypothesis is None:
        _emit_near_debug_record({
            "event": "zenear_no_transform",
            "fits": str(fits_path),
            "n_pairs": int(img_pairs.shape[0]),
            "radius_deg": float(radius),
            "approx_scale_arcsec": float(scale_arcsec) if scale_arcsec is not None else None,
            "scale_min_deg": float(scale_min_deg),
            "scale_max_deg": float(scale_max_deg),
            "stars_detected": int(stars.size),
            "candidates": int(len(candidates)),
            "iso_refs": int(iso_refs),
            "astap_iso_diag": astap_iso_diag,
        })
        return _failure("near solver could not estimate a similarity transform")
    t_ransac_s = time.perf_counter() - t_ransac0

    if strict_astap_iso:
        logger.info("near strict astap-iso: affine NN refinement disabled (mirror mode)")

    transform, _ = hypothesis
    used_transform = used_transform or transform
    tr_scale_raw = float(getattr(transform, "scale", 0.0) or 0.0)
    if tr_scale_raw > 0.05:
        transform_deg = SimilarityTransform(
            scale=tr_scale_raw / 3600.0,
            rotation=float(getattr(transform, "rotation", 0.0) or 0.0),
            translation=(
                float(getattr(transform, "translation", (0.0, 0.0))[0] or 0.0) / 3600.0,
                float(getattr(transform, "translation", (0.0, 0.0))[1] or 0.0) / 3600.0,
            ),
            parity=int(getattr(transform, "parity", 1) or 1),
        )
    else:
        transform_deg = transform
    matches = np.empty((0, 4), dtype=np.float64)
    n_global_inliers = 0
    rms_global_px = float("inf")

    if iso_matrix is not None and iso_offset is not None:
        tol_muls = (1.0,) if strict_astap_iso else (1.0, 1.8, 3.0, 5.0)
        for tol_mul in tol_muls:
            matches_try, inl_try, rms_try = _build_matches_from_affine(
                iso_matrix,
                iso_offset,
                image_positions,
                cat_positions_iso,
                cat_world,
                pixel_tolerance=float(cfg.pixel_tolerance) * float(tol_mul),
                max_matches=600,
            )
            if inl_try > n_global_inliers or (inl_try == n_global_inliers and rms_try < rms_global_px):
                matches = matches_try
                n_global_inliers = int(inl_try)
                rms_global_px = float(rms_try)
            if n_global_inliers >= 12:
                break

    if matches.shape[0] < 4 and (not strict_astap_iso):
        for tol_mul in (1.0, 1.8, 3.0, 5.0):
            matches_try, inl_try, rms_try = _build_matches_from_transform(
                transform,
                image_positions,
                cat_positions_iso,
                cat_world,
                pixel_tolerance=float(cfg.pixel_tolerance) * float(tol_mul),
                max_matches=600,
            )
            if inl_try > n_global_inliers or (inl_try == n_global_inliers and rms_try < rms_global_px):
                matches = matches_try
                n_global_inliers = int(inl_try)
                rms_global_px = float(rms_try)
            if n_global_inliers >= 12:
                break

    if matches.shape[0] < 4:
        if strict_astap_iso:
            logger.info("near strict astap-iso: skipping geometric consensus gate (mirror mode)")
        else:
            _emit_near_debug_record({
                "event": "zenear_no_consensus",
                "fits": str(fits_path),
                "strict_astap_iso": bool(strict_astap_iso),
                "n_pairs": int(img_pairs.shape[0]),
                "iso_refs": int(iso_refs),
                "n_global_inliers": int(n_global_inliers),
                "rms_global_px": float(rms_global_px) if np.isfinite(rms_global_px) else None,
                "radius_deg": float(radius),
                "approx_scale_arcsec": float(scale_arcsec) if scale_arcsec is not None else None,
                "scale_min_deg": float(scale_min_deg),
                "scale_max_deg": float(scale_max_deg),
                "astap_iso_diag": astap_iso_diag,
            })
            return _failure("no geometric consensus found for metadata solve")
    t_fit0 = time.perf_counter()
    tile_center_hint = (float(ra0), float(dec0))
    tile_center_iso = (float(iso_center_ra), float(iso_center_dec))
    # When the working hypothesis comes from ASTAP-ISO center stepping,
    # keep that solved center for TAN reconstruction. Falling back to the
    # original metadata hint center can inject large WCS residuals.
    tile_center = tile_center_iso if iso_transform is not None else tile_center_hint
    if strict_astap_iso and iso_transform is not None:
        try:
            iso_deg = SimilarityTransform(
                scale=float(getattr(iso_transform, "scale", 0.0)) / 3600.0,
                rotation=float(getattr(iso_transform, "rotation", 0.0)),
                translation=(
                    float(getattr(iso_transform, "translation", (0.0, 0.0))[0]) / 3600.0,
                    float(getattr(iso_transform, "translation", (0.0, 0.0))[1]) / 3600.0,
                ),
                parity=int(getattr(iso_transform, "parity", 1)),
            )
            wcs = tan_from_similarity(iso_deg, image.shape, tile_center=tile_center_iso)
        except Exception:
            if iso_matrix is not None and iso_offset is not None:
                try:
                    wcs, _ = fit_wcs_tan(matches)
                except Exception:
                    wcs = tan_from_similarity(transform_deg, image.shape, tile_center=tile_center)
            else:
                wcs = tan_from_similarity(transform_deg, image.shape, tile_center=tile_center)
    elif iso_matrix is not None and iso_offset is not None:
        wcs = None
        try:
            wcs_ls, _ = fit_wcs_tan(matches)
            ls_pix = _pix_scale_arcsec(wcs_ls)
            tr_pix = float(getattr(transform_deg, "scale", 0.0)) * 3600.0
            if np.isfinite(ls_pix) and tr_pix > 0 and 0.45 <= (ls_pix / tr_pix) <= 2.2:
                wcs = wcs_ls
        except Exception:
            wcs = None
        if wcs is None:
            wcs = tan_from_similarity(transform_deg, image.shape, tile_center=tile_center)
    else:
        wcs = tan_from_similarity(transform_deg, image.shape, tile_center=tile_center)
    # Keep user value as an upper bound, but adapt to available support.
    # On narrow/FOV-limited frames, fixed 60 inliers is often unattainable
    # even for geometrically excellent solutions.
    n_pairs = int(matches.shape[0])
    adaptive_inliers = min(
        int(cfg.quality_inliers),
        max(10, int(0.4 * max(0, n_pairs))),
    )

    if strict_astap_iso:
        # Mirror mode: do not apply ZeSolver validation/sip optimization gates.
        inlier_gate = max(3, int(minimum_quads))
        pix_scale_arcsec_guess = _pix_scale_arcsec(wcs)
        final_wcs = wcs
        final_stats = {
            "quality": "GOOD",
            "success": True,
            "rms_px": float(rms_global_px) if np.isfinite(rms_global_px) else 0.0,
            "inliers": int(max(n_global_inliers, iso_refs)),
            "inliers_raw": int(max(n_global_inliers, iso_refs)),
            "pix_scale_arcsec": float(pix_scale_arcsec_guess) if pix_scale_arcsec_guess is not None else None,
            "reason": "strict_astap_iso_mirror",
        }
    else:
        rms_gate = float(cfg.quality_rms)
        inlier_gate = adaptive_inliers
        thresholds = {"rms_px": rms_gate, "inliers": inlier_gate}
        stats = validate_solution(
            wcs,
            matches,
            thresholds=thresholds,
        )
        final_wcs = wcs
        final_stats = stats
        if cancel_check and cancel_check():
            return _failure("cancelled")
        try:
            ls_wcs, _ = fit_wcs_tan(matches)
        except Exception:
            ls_wcs = None
        else:
            ls_stats = validate_solution(
                ls_wcs,
                matches,
                thresholds=thresholds,
            )
            if ls_stats.get("quality") == "GOOD" and ls_stats.get("rms_px", float("inf")) < final_stats.get("rms_px", float("inf")):
                final_wcs = ls_wcs
                final_stats = ls_stats
        fov_est = approx_fov or (2.0 * np.max(np.hypot(cat_positions[:, 0], cat_positions[:, 1])))
        if final_stats.get("quality") == "GOOD" and needs_sip(final_wcs, final_stats, fov_est):
            for order in range(2, cfg.sip_order + 1):
                if cancel_check and cancel_check():
                    return _failure("cancelled")
                candidate_wcs, _ = fit_wcs_sip(matches, order=order)
                candidate_stats = validate_solution(
                    candidate_wcs,
                    matches,
                    thresholds=thresholds,
                )
                if candidate_stats.get("quality") == "GOOD" and candidate_stats["rms_px"] < final_stats["rms_px"]:
                    final_wcs = candidate_wcs
                    final_stats = candidate_stats
                if not needs_sip(final_wcs, final_stats, fov_est):
                    break
    if final_stats.get("quality") != "GOOD":
        cd = getattr(final_wcs.wcs, "cd", None)
        cd_det = None
        if cd is not None:
            try:
                cd_det = float(np.linalg.det(np.asarray(cd, dtype=float)))
            except Exception:
                cd_det = None
        _emit_near_debug_record({
            "event": "zenear_validation_fail",
            "fits": str(fits_path),
            "reason": str(final_stats.get("reason", "validation_fail")),
            "stats": dict(final_stats),
            "n_pairs": int(n_pairs),
            "global_inliers": int(n_global_inliers) if "n_global_inliers" in locals() else None,
            "inliers": int(final_stats.get("inliers", 0)),
            "req_inliers": int(inlier_gate),
            "transform_scale_degpx": float(getattr(transform, "scale", float("nan"))),
            "transform_rot": float(getattr(transform, "rotation", float("nan"))),
            "transform_parity": int(getattr(transform, "parity", 1)),
            "cd_det": cd_det,
            "radius_deg": float(radius),
            "approx_scale_arcsec": float(scale_arcsec) if scale_arcsec is not None else None,
            "scale_min_deg": float(scale_min_deg),
            "scale_max_deg": float(scale_max_deg),
            "tile_id": candidates[0].get("tile_key") if candidates else None,
            "iso_refs": int(iso_refs),
            "astap_iso_diag": astap_iso_diag,
        })
        return _failure(f"near solution failed validation ({final_stats})")

    if strict_astap_iso:
        pix_scale_arcsec = _pix_scale_arcsec(final_wcs)
        zemo_ok, zemo_reason = True, "strict_astap_iso_acceptance_gate"
        mode = str(getattr(cfg, "strict_acceptance_mode", "diagnostic") or "diagnostic").strip().lower()
        if mode not in {"off", "diagnostic", "enforce"}:
            mode = "diagnostic"
        near_diag = {}
        if mode == "off":
            near_ok, near_reason = True, "strict_acceptance_off"
            near_diag = {"decision": "ACCEPT", "reason": "strict_acceptance_off", "mode": mode}
        else:
            near_diag = validate_strict_astap_iso_candidate(
                final_wcs,
                matches,
                width=int(width),
                height=int(height),
                ra_hint_deg=float(ra0) if ra0 is not None else None,
                dec_hint_deg=float(dec0) if dec0 is not None else None,
                search_radius_deg=float(radius),
                approx_fov_deg=float(approx_fov) if approx_fov is not None else None,
                approx_scale_arcsec=float(scale_arcsec) if scale_arcsec is not None else None,
                pixel_tolerance=float(cfg.pixel_tolerance),
                iso_refs=int(iso_refs),
                min_unique_matches=int(getattr(cfg, "strict_acceptance_min_unique_matches", 12) or 12),
                min_holdout_matches=int(getattr(cfg, "strict_acceptance_min_holdout_matches", 8) or 8),
                max_rms_px=float(getattr(cfg, "strict_acceptance_max_rms_px", 3.0) or 3.0),
                max_cd_cond=float(getattr(cfg, "strict_acceptance_max_cd_cond", 1.0e4) or 1.0e4),
                min_span_fraction=float(getattr(cfg, "strict_acceptance_min_span_fraction", 0.12) or 0.12),
            )
            near_diag["mode"] = mode
            near_ok = str(near_diag.get("decision")) == "ACCEPT" or mode == "diagnostic"
            near_reason = str(near_diag.get("reason", "STRICT_ACCEPTANCE_REJECTED"))
        final_stats["strict_acceptance"] = dict(near_diag)
        final_stats["strict_acceptance_mode"] = mode
    else:
        zemo_ok, zemo_reason, pix_scale_arcsec = validate_wcs_for_zemosaic(final_wcs)
        if not zemo_ok:
            cd = getattr(final_wcs.wcs, "cd", None)
            cd_det = None
            if cd is not None:
                try:
                    cd_det = float(np.linalg.det(np.asarray(cd, dtype=float)))
                except Exception:
                    cd_det = None
            _emit_near_debug_record({
                "event": "zenear_reject_zemosaic",
                "fits": str(fits_path),
                "reason": str(zemo_reason),
                "quality": final_stats.get("quality"),
                "rms_px": float(final_stats.get("rms_px", float("nan"))),
                "inliers": int(final_stats.get("inliers", 0)),
                "req_inliers": int(inlier_gate),
                "n_pairs": int(n_pairs),
                "radius_deg": float(radius),
                "approx_scale_arcsec": float(scale_arcsec) if scale_arcsec is not None else None,
                "cd_det": cd_det,
                "tile_id": candidates[0].get("tile_key") if candidates else None,
            })
            return _failure(f"near solution rejected for zemosaic ({zemo_reason})")

        near_ok, near_reason, near_diag = _near_conformance_check(
            final_wcs,
            width=width,
            height=height,
            ra_hint_deg=ra0,
            dec_hint_deg=dec0,
            search_radius_deg=radius,
            approx_fov_deg=approx_fov,
            approx_scale_arcsec=scale_arcsec,
            scale_min_ratio=float(getattr(cfg, "conformance_scale_min_ratio", 0.60) or 0.70),
            scale_max_ratio=float(getattr(cfg, "conformance_scale_max_ratio", 1.80) or 1.50),
            center_extra_deg=float(getattr(cfg, "conformance_center_extra_deg", 0.6) or 0.6),
            center_fov_mult=float(getattr(cfg, "conformance_center_fov_mult", 1.2) or 1.2),
            center_max_deg=float(getattr(cfg, "conformance_center_max_deg", 0.90) or 0.5),
        )

    cd = getattr(final_wcs.wcs, "cd", None)
    cd_det = None
    cd_cond = None
    if cd is not None:
        try:
            cd_arr = np.asarray(cd, dtype=float)
            cd_det = float(np.linalg.det(cd_arr))
            cd_cond = float(np.linalg.cond(cd_arr))
        except Exception:
            cd_det = None
            cd_cond = None

    debug_record = {
        "event": "zenear_attempt",
        "fits": str(fits_path),
        "quality": final_stats.get("quality"),
        "rms_px": float(final_stats.get("rms_px", float("nan"))),
        "inliers": int(final_stats.get("inliers", 0)),
        "req_inliers": int(inlier_gate),
        "n_pairs": int(n_pairs),
        "radius_deg": float(radius),
        "approx_scale_arcsec": float(scale_arcsec) if scale_arcsec is not None else None,
        "pixscal_arcsec": float(pix_scale_arcsec) if pix_scale_arcsec is not None else None,
        "zemosaic_ok": bool(zemo_ok),
        "zemosaic_reason": str(zemo_reason),
        "conformance_ok": bool(near_ok),
        "conformance_reason": str(near_reason),
        "cd_det": cd_det,
        "cd_cond": cd_cond,
        "diag": near_diag,
        "astap_iso_diag": astap_iso_diag,
        "tile_id": candidates[0].get("tile_key") if candidates else None,
        "near_catalog_provider": provider_telemetry.get("near_catalog_provider"),
    }
    _emit_near_debug_record(debug_record)

    if not near_ok:
        return _failure(f"near solution rejected by conformance gate ({near_reason})")

    header_updates = {
        "SOLVED": 1,
        "QUALITY": final_stats.get("quality", "GOOD"),
        "NEAR_VER": NEAR_SOLVER_VERSION,
        "RMSPX": final_stats.get("rms_px"),
        "INLIERS": final_stats.get("inliers"),
        "REQINL": adaptive_inliers,
        "TILE_ID": candidates[0].get("tile_key") if candidates else None,
        "SOLVMODE": "NEAR",
        "SOLVER": "ZeSolver",
    }
    try:
        if used_transform is not None:
            final_stats["seed_scale"] = float(used_transform.scale)
            final_stats["seed_rotation"] = float(used_transform.rotation)
            final_stats["seed_parity"] = int(getattr(used_transform, "parity", 1))
    except Exception:
        pass
    if provider is not None:
        try:
            refreshed = provider.telemetry()
            refreshed["near_catalog_candidate_tiles"] = provider_telemetry.get("near_catalog_candidate_tiles", 0)
            refreshed["near_catalog_loaded_tiles"] = provider_telemetry.get("near_catalog_loaded_tiles", 0)
            refreshed["near_catalog_loaded_stars"] = provider_telemetry.get("near_catalog_loaded_stars", 0)
            provider_telemetry = refreshed
        except Exception:
            pass
    final_stats.update(provider_telemetry)
    if pix_scale_arcsec is None:
        pix_scale_arcsec = _pix_scale_arcsec(final_wcs)
    if pix_scale_arcsec is not None:
        header_updates["PIXSCAL"] = pix_scale_arcsec
    t_fit_s = time.perf_counter() - t_fit0
    elapsed = time.perf_counter() - start
    header_updates["NEARTIME"] = f"{elapsed:.2f}s"
    if cancel_check and cancel_check():
        return _failure("cancelled")
    t_write0 = time.perf_counter()
    try:
        with fits.open(fits_path, mode="update", memmap=False) as hdul:
            header = hdul[0].header
            apply_wcs_solution_to_header(
                header,
                final_wcs,
                header_updates=header_updates,
                remove_sip_before_write=True,
            )
            hdul.flush()
    except Exception as exc:
        return _failure(f"unable to write WCS to FITS: {exc}")
    t_write_s = time.perf_counter() - t_write0
    logger.info(
        "near solve succeeded for %s (rms=%.3f px, inliers=%d, %.1fs)",
        fits_path.name,
        final_stats.get("rms_px", float("nan")),
        final_stats.get("inliers", 0),
        elapsed,
    )
    logger.info(
        "near timings for %s: detect=%.3fs pair=%.3fs ransac=%.3fs fit=%.3fs write=%.3fs total=%.3fs",
        fits_path.name,
        float(t_detect_s),
        float(t_pair_s),
        float(t_ransac_s),
        float(t_fit_s),
        float(t_write_s),
        float(elapsed),
    )
    return WcsSolution(
        True,
        "near solution found",
        final_wcs,
        final_stats,
        candidates[0].get("tile_key") if candidates else None,
        header_updates,
    )
