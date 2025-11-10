#!/usr/bin/env python3
"""
ZeSolver batch utility: GUI + CLI wrapper around the zewcs290 catalogue reader.

The tool scans a directory for FITS/TIFF/PNG imaging files, detects stars, matches
them with Gaia catalogue tiles via `CatalogDB`, and writes a TAN WCS solution back
to each file (or to a JSON sidecar for raster formats).  When launched without the
`--db-root/--input-dir` arguments it falls back to the PySide6 GUI.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.io import fits


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from importlib.metadata import PackageNotFoundError, version as pkg_version
except ImportError:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore

try:
    APP_VERSION = pkg_version("zewcs290")
except PackageNotFoundError:  # pragma: no cover - running from source tree
    APP_VERSION = "0.0.dev"

try:
    from zewcs290 import CatalogDB
except Exception as exc:  # pragma: no cover - easier failure path for CLI users
    raise SystemExit(
        "Unable to import zewcs290. Make sure you run the script from the repository root "
        "or add it to PYTHONPATH."
    ) from exc

try:
    import astroalign
except ImportError as exc:  # pragma: no cover
    raise SystemExit("astroalign is required. Install the project dependencies first.") from exc

from scipy.ndimage import gaussian_filter
from skimage import color, io as skio
from skimage.feature import peak_local_max
from skimage.transform import rescale

from zesolver.zeblindsolver import (
    BlindSolveResult,
    BlindSolverRuntimeError,
    DEFAULT_DB_SEQUENCE,
    PROFILE_PRESETS,
    blind_solve,
    has_valid_wcs,
)


FITS_EXTENSIONS = {".fit", ".fits", ".fts"}
RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = FITS_EXTENSIONS | RASTER_EXTENSIONS
SIDE_CAR_SUFFIX = ".wcs.json"
FITS_MEMMAP_FORBIDDEN_KEYS = ("BZERO", "BSCALE", "BLANK")

DEFAULT_FOV_DEG = 1.5
DEFAULT_MAX_IMAGE_STARS = 200
DEFAULT_MAX_CATALOG_STARS = 2000
DEFAULT_ALIGNMENT_STARS = 60
DEFAULT_SEARCH_RADIUS_SCALE = 1.8
DEFAULT_SEARCH_RADIUS_ATTEMPTS = 3
GUI_DEFAULT_LANGUAGE = "fr"
GUI_FALLBACK_LANGUAGE = "en"
GUI_LANG_ORDER = ("fr", "en")
GUI_TRANSLATIONS: dict[str, dict[str, str]] = {
    "fr": {
        "language_menu": "Langue",
        "language_action_fr": "Français",
        "language_action_en": "Anglais",
        "window_title": "ZeSolver – Traitement par lot",
        "browse_button": "Parcourir…",
        "database_label": "Base de données",
        "input_label": "Dossier d'images",
        "scan_button": "Analyser les fichiers",
        "options_box": "Paramètres solveur",
        "fov_label": "FOV horizontal (°)",
        "search_scale_label": "Facteur rayon recherche",
        "search_attempts_label": "Tentatives rayon",
        "max_radius_label": "Rayon max (°)",
        "downsample_label": "Downsample",
        "threads_label": "Threads",
        "cache_label": "Cache catalogue",
        "max_files_label": "Limite de fichiers",
        "formats_label": "Extensions suivies",
        "families_label": "Familles ASTAP",
        "overwrite_label": "Réécrire les WCS existants",
        "files_header": "Fichier",
        "status_header": "Statut",
        "details_header": "Détails",
        "log_box": "Journal",
        "run_button": "Résoudre",
        "stop_button": "Stop",
        "status_ready": "Prêt.",
        "dialog_select_directory": "Choisir un dossier",
        "error_select_input": "Sélectionne un dossier contenant tes images.",
        "error_input_missing": "Dossier introuvable: {path}",
        "error_database_required": "Renseigne le dossier base de données.",
        "error_database_missing": "Base introuvable: {path}",
        "error_no_input_dir": "Aucun dossier d'entrée sélectionné.",
        "dialog_config_title": "Configuration",
        "dialog_no_files": "Aucun fichier à résoudre.",
        "info_files_detected": "{count} fichier(s) détecté(s).",
        "processing_count": "Traitement de {count} fichier(s).",
        "log_stop_requested": "Arrêt demandé…",
        "log_processing_done": "Traitement terminé.",
        "runner_start": "Démarrage: {files} fichier(s), {workers} thread(s).",
        "runner_stop_wait": "Arrêt demandé, attente de la fin des tâches…",
        "status_waiting": "en attente",
        "status_solved": "résolu",
        "status_failed": "échec",
        "status_skipped": "ignoré",
        "log_result": "{status}: {path} — {message}",
        "log_result_no_details": "{status}: {path}",
        "log_error_prefix": "ERREUR",
        "special_auto": "Auto",
        "run_info_blind_started": "Blind: démarrage pour {path}",
        "run_info_blind_db": "Blind: base {db}",
        "run_info_blind_succeeded": "Blind réussi ({db}, {elapsed})",
        "run_info_blind_failed": "Blind échec: {message}",
    },
    "en": {
        "language_menu": "Language",
        "language_action_fr": "French",
        "language_action_en": "English",
        "window_title": "ZeSolver – Batch Solver",
        "browse_button": "Browse…",
        "database_label": "Database",
        "input_label": "Image folder",
        "scan_button": "Scan files",
        "options_box": "Solver settings",
        "fov_label": "Horizontal FOV (°)",
        "search_scale_label": "Search radius scale",
        "search_attempts_label": "Radius attempts",
        "max_radius_label": "Max radius (°)",
        "downsample_label": "Downsample",
        "threads_label": "Threads",
        "cache_label": "Catalog cache",
        "max_files_label": "File limit",
        "formats_label": "Watched extensions",
        "families_label": "ASTAP families",
        "overwrite_label": "Overwrite existing WCS",
        "files_header": "File",
        "status_header": "Status",
        "details_header": "Details",
        "log_box": "Log",
        "run_button": "Solve",
        "stop_button": "Stop",
        "status_ready": "Ready.",
        "dialog_select_directory": "Choose a folder",
        "error_select_input": "Select a folder containing your images.",
        "error_input_missing": "Folder not found: {path}",
        "error_database_required": "Provide the catalog database folder.",
        "error_database_missing": "Database not found: {path}",
        "error_no_input_dir": "No input directory selected.",
        "dialog_config_title": "Configuration",
        "dialog_no_files": "No files to solve.",
        "info_files_detected": "Detected {count} file(s).",
        "processing_count": "Processing {count} file(s).",
        "log_stop_requested": "Stop requested…",
        "log_processing_done": "Processing finished.",
        "runner_start": "Starting: {files} file(s), {workers} thread(s).",
        "runner_stop_wait": "Stop requested, waiting for tasks…",
        "status_waiting": "waiting",
        "status_solved": "solved",
        "status_failed": "failed",
        "status_skipped": "skipped",
        "log_result": "{status}: {path} — {message}",
        "log_result_no_details": "{status}: {path}",
        "log_error_prefix": "ERROR",
        "special_auto": "Auto",
        "run_info_blind_started": "Blind solve started for {path}",
        "run_info_blind_db": "Blind: trying {db}",
        "run_info_blind_succeeded": "Blind success ({db}, {elapsed})",
        "run_info_blind_failed": "Blind failed: {message}",
    },
}


class SolveError(RuntimeError):
    """Raised for recoverable solving errors."""

    def __init__(self, message: str, *, skip: bool = False):
        super().__init__(message)
        self.skip = skip


@dataclass(slots=True)
class SolveConfig:
    db_root: Path
    input_dir: Path
    families: Optional[Sequence[str]]
    fov_deg: float = DEFAULT_FOV_DEG
    downsample: int = 1
    overwrite: bool = False
    workers: int = 1
    formats: Sequence[str] = field(default_factory=lambda: tuple(sorted(SUPPORTED_EXTENSIONS)))
    max_files: Optional[int] = None
    cache_size: int = 12
    max_catalog_stars: int = DEFAULT_MAX_CATALOG_STARS
    max_image_stars: int = DEFAULT_MAX_IMAGE_STARS
    max_alignment_stars: int = DEFAULT_ALIGNMENT_STARS
    mag_limit: Optional[float] = None
    search_radius_scale: float = DEFAULT_SEARCH_RADIUS_SCALE
    search_radius_attempts: int = DEFAULT_SEARCH_RADIUS_ATTEMPTS
    max_search_radius_deg: Optional[float] = None
    blind_enabled: bool = True
    blind_db_chain: Optional[Sequence[str]] = None
    blind_profile: Optional[str] = None
    blind_timeout: int = 90
    blind_skip_if_valid: bool = True
    blind_astap_exe: Optional[str] = None
    blind_extra_args: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.families:
            normalized: list[str] = []
            seen: set[str] = set()
            for entry in self.families:
                name = str(entry).strip().lower()
                if not name or name in seen:
                    continue
                seen.add(name)
                normalized.append(name)
            value: Optional[tuple[str, ...]] = tuple(normalized) if normalized else None
            object.__setattr__(self, "families", value)
        if self.blind_db_chain:
            tokens: list[str] = []
            for entry in self.blind_db_chain:
                if entry is None:
                    continue
                for chunk in str(entry).replace(";", ",").split(","):
                    name = chunk.strip()
                    if name:
                        tokens.append(name)
            object.__setattr__(self, "blind_db_chain", tuple(tokens) if tokens else None)
        if self.blind_extra_args:
            object.__setattr__(self, "blind_extra_args", tuple(self.blind_extra_args))
        if self.search_radius_scale < 1.0:
            raise ValueError("search_radius_scale must be >= 1.0")
        if self.search_radius_attempts < 1:
            raise ValueError("search_radius_attempts must be >= 1")
        if self.max_search_radius_deg is not None and self.max_search_radius_deg <= 0:
            raise ValueError("max_search_radius_deg must be positive")
        if self.blind_timeout <= 0:
            raise ValueError("blind_timeout must be positive")


@dataclass(slots=True)
class ImageMetadata:
    path: Path
    kind: str
    width: int
    height: int
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    source: str
    has_wcs: bool
    sidecar_path: Optional[Path] = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class WCSSolution:
    crpix: np.ndarray
    crval: np.ndarray
    cd: np.ndarray
    matches: int
    rms_pixels: Optional[float]


@dataclass(slots=True)
class ImageSolveResult:
    path: Path
    status: str
    message: str
    matched_stars: int = 0
    rms_arcsec: Optional[float] = None
    pixel_scale_arcsec: Optional[float] = None
    metadata_source: Optional[str] = None
    duration_s: Optional[float] = None
    catalog_family: Optional[str] = None
    run_info: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // 2)


def _iter_image_files(input_dir: Path, extensions: Sequence[str]) -> Iterator[Path]:
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def _header_has_wcs(header: fits.Header) -> bool:
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return False
    if any(key in header for key in ("CD1_1", "CD2_2", "CD1_2", "CD2_1")):
        return True
    if "CDELT1" in header and "CDELT2" in header:
        return True
    return False


def _fits_requires_memmap_off(header: fits.Header) -> bool:
    """Astropy cannot memmap scaled data (BZERO/BSCALE/BLANK)."""
    return any(key in header for key in FITS_MEMMAP_FORBIDDEN_KEYS)


def _sexagesimal_to_deg(value: str, is_ra: bool) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    try:
        angle = Angle(text, unit=u.hourangle if is_ra else u.deg)
    except (ValueError, u.UnitsError):
        return None
    return float(angle.degree)


def _parse_angle(value: object, *, is_ra: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return _sexagesimal_to_deg(text, is_ra=is_ra)


def _catalog_offsets(stars: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    dra = (stars["ra_deg"] - ra0 + 540.0) % 360.0 - 180.0
    cos_dec = math.cos(math.radians(dec0))
    xi = dra * cos_dec
    eta = stars["dec_deg"] - dec0
    return np.column_stack((xi, eta)).astype(np.float64)


def _refine_centroid(image: np.ndarray, y: float, x: float, window: int = 7) -> tuple[float, float]:
    h, w = image.shape
    half = max(1, window // 2)
    yc = int(round(y))
    xc = int(round(x))
    y0 = max(0, yc - half)
    y1 = min(h, y0 + 2 * half + 1)
    x0 = max(0, xc - half)
    x1 = min(w, x0 + 2 * half + 1)
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return y, x
    region = region - np.min(region)
    weights = np.clip(region, 0, None)
    total = weights.sum()
    if total <= 0:
        return y, x
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    yc = float((weights * grid_y).sum() / total)
    xc = float((weights * grid_x).sum() / total)
    return yc, xc


def _detect_stars(
    image: np.ndarray,
    *,
    downsample: int,
    max_stars: int,
) -> np.ndarray:
    median = float(np.nanmedian(image))
    if not math.isfinite(median):
        median = 0.0
    data = np.nan_to_num(image, nan=median, posinf=median, neginf=median, copy=False)
    data = data.astype(np.float32, copy=False)
    data -= np.median(data)
    data[data < 0] = 0.0
    scale = 1.0 / max(1, downsample)
    if scale < 1.0:
        work = rescale(
            data,
            scale,
            anti_aliasing=True,
            preserve_range=True,
            channel_axis=None,
        ).astype(np.float32, copy=False)
    else:
        work = data
    smoothed = gaussian_filter(work, sigma=1.2)
    noise = np.std(smoothed)
    threshold = max(noise * 3.5, float(np.percentile(smoothed, 95)))
    if threshold <= 0:
        threshold = float(np.max(smoothed) * 0.25)
    coords = peak_local_max(
        smoothed,
        min_distance=max(2, downsample),
        threshold_abs=threshold,
        num_peaks=max_stars * 2,
    )
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    fluxes = smoothed[coords[:, 0], coords[:, 1]]
    order = np.argsort(fluxes)[::-1][:max_stars]
    coords = coords[order]
    fluxes = fluxes[order]
    scale_y = work.shape[0] / data.shape[0]
    scale_x = work.shape[1] / data.shape[1]
    orig_y = coords[:, 0] / scale_y
    orig_x = coords[:, 1] / scale_x
    refined: List[tuple[float, float]] = []
    for y, x in zip(orig_y, orig_x, strict=False):
        cy, cx = _refine_centroid(data, y, x)
        refined.append((cx, cy))
    refined_arr = np.array(refined, dtype=np.float32)
    return np.column_stack((refined_arr, fluxes[: refined_arr.shape[0]])).astype(np.float32, copy=False)


def _sidecar_candidates(path: Path) -> List[Path]:
    candidates = [
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_suffix(path.suffix + SIDE_CAR_SUFFIX),
        path.with_suffix(".json"),
        path.with_name(path.stem + ".json"),
    ]
    stem = path.stem
    if stem.endswith("_thn"):
        base = stem[:-4]
        for ext in FITS_EXTENSIONS:
            candidates.append(path.with_name(base + ext))
    return candidates


def _load_sidecar_metadata(path: Path) -> tuple[Optional[float], Optional[float], Optional[str]]:
    for candidate in _sidecar_candidates(path):
        if not candidate.exists() or candidate.suffix.lower() in FITS_EXTENSIONS:
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        ra = payload.get("ra_deg") or payload.get("RA")
        dec = payload.get("dec_deg") or payload.get("DEC")
        source = f"sidecar:{candidate.name}"
        if ra is not None and dec is not None:
            return float(ra), float(dec), source
    return None, None, None


def _peer_fits_metadata(path: Path) -> tuple[Optional[float], Optional[float], Optional[str]]:
    candidates: List[Path] = []
    candidates.extend(path.with_suffix(ext) for ext in FITS_EXTENSIONS)
    stem = path.stem
    if stem.endswith("_thn"):
        base = stem[:-4]
        candidates.extend(path.with_name(base + ext) for ext in FITS_EXTENSIONS)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            header = fits.getheader(candidate, ext=0)
        except Exception:
            continue
        ra = _parse_angle(header.get("RA") or header.get("OBJCTRA"), is_ra=True)
        dec = _parse_angle(header.get("DEC") or header.get("OBJCTDEC"), is_ra=False)
        if ra is not None and dec is not None:
            return ra, dec, f"peer:{candidate.name}"
    return None, None, None


class ImageSolver:
    def __init__(self, config: SolveConfig) -> None:
        self.config = config
        self.db = CatalogDB(config.db_root, families=config.families, cache_size=config.cache_size)
        self._db_lock = threading.Lock()
        self._family_candidates = self._init_family_candidates()
        self._family_hint: Optional[str] = None

    def _init_family_candidates(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for tile in self.db.tiles:
            key = tile.spec.key
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        if not ordered:
            ordered = list(self.db.families)
        if not ordered:
            raise RuntimeError("No catalogue families available in the database")
        if self.config.families:
            filtered = [fam for fam in self.config.families if fam in seen]
            missing = [fam for fam in self.config.families if fam not in seen]
            if missing:
                logging.warning("Requested families not found in %s: %s", self.config.db_root, ", ".join(missing))
            if filtered:
                return tuple(filtered)
            logging.info(
                "Falling back to available catalogue families (%s) because requested ones were missing",
                ", ".join(ordered),
            )
        return tuple(ordered)

    def _families_for_attempt(self) -> List[str]:
        order = list(self._family_candidates)
        if self._family_hint and self._family_hint in order:
            order.remove(self._family_hint)
            order.insert(0, self._family_hint)
        return order

    def _radius_candidates(self, base_radius: float) -> List[float]:
        base = max(base_radius, 0.05)
        attempts = max(1, self.config.search_radius_attempts)
        scale = max(1.0, self.config.search_radius_scale)
        limit = self.config.max_search_radius_deg
        radii: List[float] = []
        current = base if limit is None else min(base, limit)
        for _ in range(attempts):
            radii.append(current)
            if scale <= 1.0:
                break
            next_radius = current * scale
            if limit is not None and next_radius >= limit:
                radii.append(limit)
                break
            current = next_radius
        unique: List[float] = []
        for value in radii:
            if not unique or abs(unique[-1] - value) > 1e-6:
                unique.append(value)
        return unique or [base]

    @staticmethod
    def _family_label(family: str) -> str:
        return family.upper()

    @classmethod
    def _family_error(cls, family: str, message: str) -> str:
        return f"{cls._family_label(family)}: {message}"

    def _solve_with_catalog(
        self,
        *,
        path: Path,
        metadata: ImageMetadata,
        peaks: np.ndarray,
        radius: float,
        families: Optional[Sequence[str]],
        label: Optional[str],
        catalog_family: Optional[str],
    ) -> ImageSolveResult:
        with self._db_lock:
            catalog = self.db.query_cone(
                ra_deg=metadata.ra_deg,
                dec_deg=metadata.dec_deg,
                radius_deg=radius,
                families=families,
                mag_limit=self.config.mag_limit,
                max_stars=self.config.max_catalog_stars,
            )
        if catalog.size < 5:
            error = "catalogue returned too few stars in this region"
            raise SolveError(
                self._family_error(catalog_family, error) if catalog_family else error
            )
        order = np.argsort(catalog["mag"])
        catalog = catalog[order[: self.config.max_catalog_stars]]
        plane = _catalog_offsets(catalog, metadata.ra_deg, metadata.dec_deg)
        if plane.shape[0] < 3:
            error = "not enough reference stars after projection"
            raise SolveError(
                self._family_error(catalog_family, error) if catalog_family else error
            )
        image_points = peaks[:, :2].astype(np.float64, copy=False)
        max_points = min(self.config.max_alignment_stars, plane.shape[0], image_points.shape[0])
        if max_points < 3:
            error = "not enough overlap between catalogue and detected stars"
            raise SolveError(
                self._family_error(catalog_family, error) if catalog_family else error
            )
        transform, (used_src, used_dst) = astroalign.find_transform(
            plane,
            image_points,
            max_control_points=max_points,
        )
        projected = astroalign.matrix_transform(used_src, transform.params)
        residuals = np.linalg.norm(projected - used_dst, axis=1)
        rms_px = float(np.sqrt(np.mean(residuals**2))) if residuals.size else None
        solution = self._build_solution(transform, metadata, matches=used_src.shape[0], rms=rms_px)
        pixel_scale_arcsec = self._pixel_scale_arcsec(solution.cd)
        rms_arcsec = rms_px * pixel_scale_arcsec if rms_px is not None else None
        self._write_solution(metadata, solution)
        if label:
            message = f"Solved via {label} ({solution.matches} matches, ~{pixel_scale_arcsec:.2f}\"/px)"
        else:
            message = f"Solved ({solution.matches} matches, ~{pixel_scale_arcsec:.2f}\"/px)"
        if rms_arcsec is not None:
            message += f", rms {rms_arcsec:.2f}\""
        return ImageSolveResult(
            path=path,
            status="solved",
            message=message,
            matched_stars=solution.matches,
            rms_arcsec=rms_arcsec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            metadata_source=metadata.source,
            catalog_family=catalog_family,
        )

    def solve_path(self, path: Path) -> ImageSolveResult:
        start = time.perf_counter()
        run_info: list[tuple[str, dict[str, Any]]] = []
        shortcut = self._try_blind_shortcut(path, run_info)
        if shortcut is not None:
            return shortcut
        metadata: Optional[ImageMetadata] = None
        try:
            data, metadata = self._load_image(path)
            if metadata.has_wcs and not self.config.overwrite:
                raise SolveError("WCS already present (use --overwrite to recompute)", skip=True)
            if metadata.ra_deg is None or metadata.dec_deg is None:
                raise SolveError("Missing RA/DEC metadata", skip=False)
            peaks = _detect_stars(
                data,
                downsample=self.config.downsample,
                max_stars=self.config.max_image_stars,
            )
            if peaks.shape[0] < 5:
                raise SolveError("Not enough stars detected in the frame")
            field_height = self.config.fov_deg * (metadata.height / metadata.width)
            base_radius = 0.55 * math.hypot(self.config.fov_deg, field_height)
            radius_candidates = self._radius_candidates(base_radius)
            final_error: Optional[SolveError] = None
            for radius_index, radius in enumerate(radius_candidates):
                last_error: Optional[SolveError] = None
                combined_label: Optional[str] = None
                combined_catalog_family: Optional[str] = None
                if self.config.families and len(self.config.families) == 1:
                    combined_catalog_family = self.config.families[0]
                    combined_label = self._family_label(combined_catalog_family)
                try:
                    result = self._solve_with_catalog(
                        path=path,
                        metadata=metadata,
                        peaks=peaks,
                        radius=radius,
                        families=self.config.families,
                        label=combined_label,
                        catalog_family=combined_catalog_family,
                    )
                    if result.catalog_family:
                        self._family_hint = result.catalog_family
                    result.duration_s = time.perf_counter() - start
                    result.run_info.extend(run_info)
                    return result
                except SolveError as exc:
                    last_error = exc
                families_to_try = self._families_for_attempt()
                if not families_to_try:
                    if last_error is not None:
                        raise last_error
                    raise SolveError("No catalogue families available in the database")
                for idx, family in enumerate(families_to_try):
                    try:
                        result = self._solve_with_catalog(
                            path=path,
                            metadata=metadata,
                            peaks=peaks,
                            radius=radius,
                            families=(family,),
                            label=self._family_label(family),
                            catalog_family=family,
                        )
                        if self._family_hint != family:
                            if self._family_hint:
                                logging.info(
                                    "Switching preferred catalogue to %s after success on %s",
                                    self._family_label(family),
                                    path.name,
                                )
                            self._family_hint = family
                        result.duration_s = time.perf_counter() - start
                        result.run_info.extend(run_info)
                        return result
                    except SolveError as exc:
                        last_error = exc
                        if idx + 1 < len(families_to_try):
                            logging.info(
                                "%s failed for %s (%s) — trying other families",
                                self._family_label(family),
                                path.name,
                                exc,
                            )
                if last_error:
                    families_label = ", ".join(self._family_label(fam) for fam in families_to_try)
                    composed_error = SolveError(
                        f"All catalogues failed ({families_label}): {last_error}",
                        skip=last_error.skip,
                    )
                    final_error = composed_error
                    if radius_index + 1 < len(radius_candidates):
                        logging.info(
                            "Search radius %.2f° failed for %s (%s) — increasing to %.2f°",
                            radius,
                            path.name,
                            composed_error,
                            radius_candidates[radius_index + 1],
                        )
                    continue
                break
            if final_error:
                raise final_error
            raise SolveError("No catalogue families available in the database")
        except SolveError as exc:
            duration = time.perf_counter() - start
            status = "skipped" if exc.skip else "failed"
            return ImageSolveResult(
                path=path,
                status=status,
                message=str(exc),
                metadata_source=metadata.source if metadata else None,
                duration_s=duration,
                run_info=list(run_info),
            )
        except Exception as exc:  # pragma: no cover - safety net for unexpected failures
            duration = time.perf_counter() - start
            logging.exception("Unexpected error while solving %s", path)
            return ImageSolveResult(
                path=path,
                status="failed",
                message=f"Internal error: {exc}",
                metadata_source=metadata.source if metadata else None,
                duration_s=duration,
                run_info=list(run_info),
            )

    @staticmethod
    def _pixel_scale_arcsec(cd: np.ndarray) -> float:
        scale_x = math.hypot(cd[0, 0], cd[1, 0])
        scale_y = math.hypot(cd[0, 1], cd[1, 1])
        return 3600.0 * 0.5 * (scale_x + scale_y)

    def _build_solution(
        self,
        transform: astroalign.SimilarityTransform,
        metadata: ImageMetadata,
        *,
        matches: int,
        rms: Optional[float],
    ) -> WCSSolution:
        matrix = np.array(transform.params[:2, :2], dtype=np.float64)
        translation = np.array(transform.params[:2, 2], dtype=np.float64)
        try:
            cd = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise SolveError("Degenerate transform (cannot invert matrix)") from exc
        crpix = translation + 1.0
        crval = np.array([metadata.ra_deg, metadata.dec_deg], dtype=np.float64)
        return WCSSolution(crpix=crpix, crval=crval, cd=cd, matches=matches, rms_pixels=rms)

    def _write_solution(self, metadata: ImageMetadata, solution: WCSSolution) -> None:
        if metadata.kind == "fits":
            self._write_fits_solution(metadata.path, solution)
        else:
            target = metadata.sidecar_path or metadata.path.with_suffix(metadata.path.suffix + SIDE_CAR_SUFFIX)
            payload = {
                "crpix": solution.crpix.tolist(),
                "crval": solution.crval.tolist(),
                "cd": solution.cd.tolist(),
                "ctype": ["RA---TAN", "DEC--TAN"],
                "cunit": ["deg", "deg"],
                "matches": solution.matches,
                "rms_pixels": solution.rms_pixels,
                "generated_by": f"zesolver.py {APP_VERSION}",
            }
            target.write_text(json.dumps(payload, indent=2))

    def _write_fits_solution(self, path: Path, solution: WCSSolution) -> None:
        with fits.open(path, mode="update", memmap=False) as hdul:
            header = hdul[0].header
            header["WCSAXES"] = 2
            header["CRPIX1"] = float(solution.crpix[0])
            header["CRPIX2"] = float(solution.crpix[1])
            header["CRVAL1"] = float(solution.crval[0])
            header["CRVAL2"] = float(solution.crval[1])
            header["CD1_1"] = float(solution.cd[0, 0])
            header["CD1_2"] = float(solution.cd[0, 1])
            header["CD2_1"] = float(solution.cd[1, 0])
            header["CD2_2"] = float(solution.cd[1, 1])
            header["CTYPE1"] = "RA---TAN"
            header["CTYPE2"] = "DEC--TAN"
            header["CUNIT1"] = "deg"
            header["CUNIT2"] = "deg"
            header["RADECSYS"] = "ICRS"
            header["EQUINOX"] = 2000.0
            header["ZESOLVER"] = (APP_VERSION, "WCS written by zesolver.py")
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            header.add_history(f"ZeSolver WCS solution at {timestamp}")
            hdul.flush()

    def _load_image(self, path: Path) -> tuple[np.ndarray, ImageMetadata]:
        suffix = path.suffix.lower()
        if suffix in FITS_EXTENSIONS:
            return self._load_fits(path)
        if suffix in RASTER_EXTENSIONS:
            return self._load_raster(path)
        raise SolveError(f"Unsupported file extension {suffix}")

    def _load_fits(self, path: Path) -> tuple[np.ndarray, ImageMetadata]:
        fallback_reason = ""
        for memmap in (True, False):
            try:
                with fits.open(path, memmap=memmap) as hdul:
                    hdu = hdul[0]
                    header = hdu.header
                    if memmap and _fits_requires_memmap_off(header):
                        fallback_reason = "scaled data keywords present"
                        continue
                    has_wcs = _header_has_wcs(header)
                    ra = _parse_angle(header.get("RA") or header.get("OBJCTRA"), is_ra=True)
                    dec = _parse_angle(header.get("DEC") or header.get("OBJCTDEC"), is_ra=False)
                    data = hdu.data
                    if data is None:
                        raise SolveError("Primary HDU has no data")
                    arr = np.asarray(data, dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr[0]
                    if arr.ndim != 2:
                        raise SolveError(f"Unsupported FITS dimensionality: {arr.shape}")
                    height, width = arr.shape
                    meta = ImageMetadata(
                        path=path,
                        kind="fits",
                        width=int(width),
                        height=int(height),
                        ra_deg=ra,
                        dec_deg=dec,
                        source="fits-header",
                        has_wcs=has_wcs,
                    )
                    return np.ascontiguousarray(arr), meta
            except ValueError as exc:
                if memmap and "memory-mapped" in str(exc).lower():
                    fallback_reason = str(exc)
                    continue
                raise
        raise SolveError(
            f"Unable to read FITS data from {path.name}: {fallback_reason or 'memmap fallback failed'}"
        )

    def _load_raster(self, path: Path) -> tuple[np.ndarray, ImageMetadata]:
        image = skio.imread(str(path))
        if image.ndim == 3:
            image = color.rgb2gray(image)
        arr = np.asarray(image, dtype=np.float32)
        height, width = arr.shape
        ra, dec, source = _load_sidecar_metadata(path)
        if ra is None or dec is None:
            ra, dec, peer_source = _peer_fits_metadata(path)
            if peer_source:
                source = peer_source
        meta = ImageMetadata(
            path=path,
            kind="raster",
            width=int(width),
            height=int(height),
            ra_deg=ra,
            dec_deg=dec,
            source=source or "unknown",
            has_wcs=(path.with_suffix(path.suffix + SIDE_CAR_SUFFIX).exists()),
            sidecar_path=path.with_suffix(path.suffix + SIDE_CAR_SUFFIX),
        )
        return np.ascontiguousarray(arr), meta

    def _should_try_blind(self, path: Path) -> bool:
        return self.config.blind_enabled and path.suffix.lower() in FITS_EXTENSIONS

    def _resolve_blind_db_chain(self) -> List[str]:
        entries = list(self.config.blind_db_chain) if self.config.blind_db_chain else list(DEFAULT_DB_SEQUENCE)
        resolved: List[str] = []
        seen: set[str] = set()
        for entry in entries:
            raw = Path(entry).expanduser()
            candidates = [raw]
            if not raw.is_absolute():
                candidates.append((self.config.db_root / entry).expanduser())
            for candidate in candidates:
                if candidate.is_dir():
                    normalized = str(candidate.resolve())
                    if normalized not in seen:
                        seen.add(normalized)
                        resolved.append(normalized)
                    break
            else:
                logging.debug("Blind DB %s not found for %s", entry, self.config.db_root)
        return resolved

    def _try_blind_shortcut(
        self,
        path: Path,
        run_info: list[tuple[str, dict[str, Any]]],
    ) -> Optional[ImageSolveResult]:
        if not self._should_try_blind(path):
            return None
        db_chain = self._resolve_blind_db_chain()
        if not db_chain:
            return None
        try:
            header = fits.getheader(path, ext=0)
        except Exception as exc:
            logging.debug("Blind fallback skipped for %s: %s", path.name, exc)
            return None
        if has_valid_wcs(header):
            return None
        run_info.append(("run_info_blind_started", {"path": path.name}))
        try:
            result = blind_solve(
                fits_path=str(path),
                db_roots=db_chain,
                profile=self.config.blind_profile,
                timeout_sec=self.config.blind_timeout,
                skip_if_valid=self.config.blind_skip_if_valid,
                astap_exe=self.config.blind_astap_exe,
                extra_args=self.config.blind_extra_args,
                log=logging.info,
            )
        except BlindSolverRuntimeError as exc:
            logging.warning("Blind solver failed for %s: %s", path.name, exc)
            run_info.append(("run_info_blind_failed", {"message": str(exc)}))
            return None
        for db in result["tried_dbs"]:
            label = Path(db).name or db
            run_info.append(("run_info_blind_db", {"db": label}))
        if result["success"]:
            db_name = result["used_db"] or (Path(result["tried_dbs"][-1]).name if result["tried_dbs"] else "unknown")
            run_info.append(
                (
                    "run_info_blind_succeeded",
                    {"db": db_name, "elapsed": f"{result['elapsed_sec']:.1f}s"},
                )
            )
            if not self.config.overwrite:
                return ImageSolveResult(
                    path=path,
                    status="solved",
                    message=result["message"],
                    metadata_source="blind",
                    duration_s=result["elapsed_sec"],
                    run_info=list(run_info),
                )
            return None
        run_info.append(("run_info_blind_failed", {"message": result["message"]}))
        return None


class BatchSolver:
    def __init__(self, config: SolveConfig, files: Optional[Sequence[Path]] = None) -> None:
        self.config = config
        self.files: List[Path] = list(files) if files is not None else self._collect_files()
        self.solver = ImageSolver(config)

    def _collect_files(self) -> List[Path]:
        files = list(_iter_image_files(self.config.input_dir, self.config.formats))
        if self.config.max_files:
            files = files[: self.config.max_files]
        return files

    def run(self, cancel_event: Optional[threading.Event] = None) -> Iterator[ImageSolveResult]:
        if not self.files:
            yield ImageSolveResult(
                path=self.config.input_dir,
                status="skipped",
                message="No matching files found",
            )
            return
        workers = max(1, self.config.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.solver.solve_path, path): path for path in self.files}
            try:
                for future in concurrent.futures.as_completed(futures):
                    if cancel_event and cancel_event.is_set():
                        break
                    yield future.result()
            finally:
                if cancel_event and cancel_event.is_set():
                    for future in futures:
                        future.cancel()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ZeSolver batch GUI/CLI")
    parser.add_argument("--db-root", type=Path, help="Directory containing the ASTAP/HNSKY catalogues")
    parser.add_argument("--input-dir", type=Path, help="Directory containing FITS/TIFF/PNG files to solve")
    parser.add_argument("--family", action="append", help="Restrict to specific catalogue families (e.g. d50)")
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG, help="Horizontal field of view in degrees")
    parser.add_argument("--downsample", type=int, default=1, choices=range(1, 5), help="Downsample factor (1-4)")
    parser.add_argument("--workers", type=int, default=_default_worker_count(), help="Worker threads (default: half CPUs)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WCS solutions")
    parser.add_argument("--formats", help="Comma-separated list of file extensions to consider")
    parser.add_argument("--max-files", type=int, help="Limit the number of files to process")
    parser.add_argument("--mag-limit", type=float, help="Optional catalogue magnitude limit")
    parser.add_argument("--cache-size", type=int, default=12, help="Catalog tile cache size")
    parser.add_argument(
        "--search-radius-scale",
        type=float,
        default=DEFAULT_SEARCH_RADIUS_SCALE,
        help="Multiply the search radius by this factor on each retry (>=1.0, default: %(default)s)",
    )
    parser.add_argument(
        "--search-radius-attempts",
        type=int,
        default=DEFAULT_SEARCH_RADIUS_ATTEMPTS,
        help="Maximum number of search radius attempts before giving up (default: %(default)s)",
    )
    parser.add_argument(
        "--max-search-radius-deg",
        type=float,
        help="Optional hard limit (degrees) for the search radius when expanding the cone",
    )
    parser.add_argument(
        "--blind-db",
        "--db",
        dest="blind_db",
        help="Semicolon-separated ASTAP database directories for blind fallback",
    )
    parser.add_argument(
        "--auto-blind-profile",
        choices=sorted(PROFILE_PRESETS.keys()),
        help="Instrument profile hint passed to the blind fallback",
    )
    parser.add_argument(
        "--blind-timeout",
        type=int,
        default=90,
        help="Timeout (seconds) for each blind fallback attempt (default: %(default)s)",
    )
    parser.add_argument("--blind-astap-exe", help="Path to astap/astap.exe for blind fallback")
    parser.add_argument(
        "--blind-extra-args",
        help="Additional arguments forwarded to ASTAP during blind fallback",
    )
    parser.add_argument(
        "--no-blind",
        dest="blind_enabled",
        action="store_false",
        help="Disable the automatic blind fallback when WCS metadata is missing",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--headless", action="store_true", help="Force CLI mode even if arguments are missing")
    parser.add_argument("--gui", action="store_true", help="Force GUI mode even if CLI arguments are provided")
    parser.set_defaults(blind_enabled=True)
    return parser


def _should_launch_gui(args: argparse.Namespace) -> bool:
    if args.gui:
        return True
    if args.headless:
        return False
    return not (args.db_root and args.input_dir)


def _normalize_family_args(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    seen: set[str] = set()
    ordered: List[str] = []
    for raw in values:
        if raw is None:
            continue
        for chunk in raw.replace(";", ",").split(","):
            name = chunk.strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered or None


def _parse_db_chain_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    entries: List[str] = []
    for chunk in value.replace(";", ",").split(","):
        token = chunk.strip()
        if token:
            entries.append(token)
    return entries or None


def _parse_extra_args_arg(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(shlex.split(value))


def _format_run_info_cli(key: str, payload: dict[str, Any], path: Path) -> Optional[str]:
    if key == "run_info_blind_started":
        name = payload.get("path") or path.name
        return f"[blind] starting for {name}"
    if key == "run_info_blind_db":
        db = payload.get("db")
        return f"[blind] trying {db}" if db else None
    if key == "run_info_blind_succeeded":
        db = payload.get("db", "unknown")
        elapsed = payload.get("elapsed", "")
        return f"[blind] success via {db} {elapsed}".strip()
    if key == "run_info_blind_failed":
        return f"[blind] failed: {payload.get('message', 'unknown error')}"
    return None


def run_cli(args: argparse.Namespace) -> int:
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    formats = (
        tuple(f".{chunk.strip().lstrip('.')}".lower() for chunk in args.formats.split(","))
        if args.formats
        else tuple(sorted(SUPPORTED_EXTENSIONS))
    )
    if not args.db_root or not args.input_dir:
        raise SystemExit("--db-root and --input-dir are required in CLI mode (use --gui to launch the GUI)")
    families = _normalize_family_args(args.family)
    blind_db_chain = _parse_db_chain_arg(args.blind_db)
    blind_extra_args = _parse_extra_args_arg(args.blind_extra_args)
    config = SolveConfig(
        db_root=args.db_root.expanduser().resolve(),
        input_dir=args.input_dir.expanduser().resolve(),
        families=families,
        fov_deg=args.fov_deg,
        downsample=args.downsample,
        overwrite=args.overwrite,
        workers=args.workers,
        formats=formats,
        max_files=args.max_files,
        cache_size=args.cache_size,
        mag_limit=args.mag_limit,
        search_radius_scale=args.search_radius_scale,
        search_radius_attempts=args.search_radius_attempts,
        max_search_radius_deg=args.max_search_radius_deg,
        blind_enabled=args.blind_enabled,
        blind_db_chain=blind_db_chain,
        blind_profile=args.auto_blind_profile,
        blind_timeout=args.blind_timeout,
        blind_astap_exe=args.blind_astap_exe,
        blind_extra_args=blind_extra_args,
    )
    logging.info(
        "Starting batch solve in %s (families=%s, workers=%d, downsample=%d)",
        config.input_dir,
        config.families or "auto",
        config.workers,
        config.downsample,
    )
    batch = BatchSolver(config)
    logging.info("Queued %d file(s) for solving", len(batch.files))
    solved = failed = skipped = 0
    start = time.perf_counter()
    for result in batch.run():
        for key, payload in result.run_info:
            message = _format_run_info_cli(key, payload, result.path)
            if message:
                logging.info(message)
        if result.status == "solved":
            solved += 1
            extra = f" [{result.message}]" if result.message else ""
            logging.info("✔ %s%s", result.path, extra)
        elif result.status == "skipped":
            skipped += 1
            logging.info("↷ %s (%s)", result.path, result.message)
        else:
            failed += 1
            logging.error("✖ %s (%s)", result.path, result.message)
    elapsed = time.perf_counter() - start
    logging.info(
        "Done in %.1fs — %d solved, %d skipped, %d failed",
        elapsed,
        solved,
        skipped,
        failed,
    )
    return 0 if failed == 0 else 1


def launch_gui(args: argparse.Namespace) -> int:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("PySide6>=6 is required for the GUI. Install it and retry.") from exc
    prefill_families = _normalize_family_args(args.family)

    class SolveRunner(QtCore.QThread):
        progress = QtCore.Signal(object)
        started = QtCore.Signal(int)
        finished = QtCore.Signal()
        info = QtCore.Signal(str)
        error = QtCore.Signal(str)

        def __init__(
            self,
            config: SolveConfig,
            files: Sequence[Path],
            translator: Callable[..., str],
        ):
            super().__init__()
            self.config = config
            self.files = [path for path in files]
            self._cancel_event = threading.Event()
            self._translate = translator

        def request_cancel(self) -> None:
            self._cancel_event.set()

        def run(self) -> None:  # pragma: no cover - GUI thread
            try:
                batch = BatchSolver(self.config, files=self.files)
            except Exception as exc:
                self.error.emit(str(exc))
                return
            self.started.emit(len(batch.files))
            self.info.emit(
                self._translate(
                    "runner_start",
                    files=len(batch.files),
                    workers=self.config.workers,
                )
            )
            try:
                for result in batch.run(cancel_event=self._cancel_event):
                    self.progress.emit(result)
                    if self._cancel_event.is_set():
                        self.info.emit(self._translate("runner_stop_wait"))
                        break
            except Exception as exc:
                self.error.emit(str(exc))
            finally:
                self.finished.emit()

    class ZeSolverWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._language = GUI_DEFAULT_LANGUAGE
            self.resize(1280, 760)
            self._worker: Optional[SolveRunner] = None
            self._pending_files: List[Path] = []
            self._item_by_path: dict[Path, QtWidgets.QTreeWidgetItem] = {}
            self._current_input_dir: Optional[Path] = None
            self._results_seen = 0
            self._language_actions: dict[str, QtGui.QAction] = {}
            self._build_ui()
            self._prefill_from_args(args)
            self._apply_language()

        # --- UI building helpers -------------------------------------------------
        def _build_ui(self) -> None:
            self._build_menu_bar()
            central = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central)
            layout.addLayout(self._build_paths_row())
            layout.addWidget(self._build_options_box())
            layout.addWidget(self._build_splitter())
            layout.addLayout(self._build_bottom_row())
            self.setCentralWidget(central)

        def _build_menu_bar(self) -> None:
            menu_bar = self.menuBar()
            self.language_menu = menu_bar.addMenu("")
            for code in GUI_LANG_ORDER:
                action = QtGui.QAction(self)
                action.setCheckable(True)
                action.triggered.connect(
                    lambda checked, lang=code: self._switch_language(lang) if checked else None
                )
                self.language_menu.addAction(action)
                self._language_actions[code] = action

        def _build_paths_row(self) -> QtWidgets.QLayout:
            grid = QtWidgets.QGridLayout()
            self.db_label = QtWidgets.QLabel()
            self.db_edit = QtWidgets.QLineEdit()
            self.browse_db_btn = QtWidgets.QPushButton()
            self.browse_db_btn.clicked.connect(lambda: self._pick_directory(self.db_edit))
            grid.addWidget(self.db_label, 0, 0)
            grid.addWidget(self.db_edit, 0, 1)
            grid.addWidget(self.browse_db_btn, 0, 2)
            self.input_label = QtWidgets.QLabel()
            self.input_edit = QtWidgets.QLineEdit()
            self.browse_in_btn = QtWidgets.QPushButton()
            self.browse_in_btn.clicked.connect(
                lambda: self._pick_directory(self.input_edit, trigger_scan=True)
            )
            grid.addWidget(self.input_label, 1, 0)
            grid.addWidget(self.input_edit, 1, 1)
            grid.addWidget(self.browse_in_btn, 1, 2)
            self.scan_btn = QtWidgets.QPushButton()
            self.scan_btn.clicked.connect(self.scan_files)
            grid.addWidget(self.scan_btn, 0, 3, 2, 1)
            return grid

        def _build_options_box(self) -> QtWidgets.QGroupBox:
            self.options_box = QtWidgets.QGroupBox()
            form = QtWidgets.QFormLayout(self.options_box)
            self.fov_spin = QtWidgets.QDoubleSpinBox()
            self.fov_spin.setRange(0.1, 20.0)
            self.fov_spin.setDecimals(2)
            self.fov_spin.setValue(args.fov_deg or DEFAULT_FOV_DEG)
            self.search_scale_spin = QtWidgets.QDoubleSpinBox()
            self.search_scale_spin.setRange(1.0, 10.0)
            self.search_scale_spin.setDecimals(2)
            self.search_scale_spin.setSingleStep(0.1)
            self.search_scale_spin.setValue(args.search_radius_scale or DEFAULT_SEARCH_RADIUS_SCALE)
            self.search_attempts_spin = QtWidgets.QSpinBox()
            self.search_attempts_spin.setRange(1, 10)
            self.search_attempts_spin.setValue(args.search_radius_attempts or DEFAULT_SEARCH_RADIUS_ATTEMPTS)
            self.max_radius_spin = QtWidgets.QDoubleSpinBox()
            self.max_radius_spin.setRange(0.0, 30.0)
            self.max_radius_spin.setDecimals(2)
            self.max_radius_spin.setSingleStep(0.1)
            self.max_radius_spin.setSpecialValueText(GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["special_auto"])
            max_radius = args.max_search_radius_deg or 0.0
            self.max_radius_spin.setValue(max_radius)
            self.downsample_spin = QtWidgets.QSpinBox()
            self.downsample_spin.setRange(1, 4)
            self.downsample_spin.setValue(args.downsample or 1)
            self.workers_spin = QtWidgets.QSpinBox()
            self.workers_spin.setRange(1, max(32, _default_worker_count()))
            self.workers_spin.setValue(args.workers or _default_worker_count())
            self.cache_spin = QtWidgets.QSpinBox()
            self.cache_spin.setRange(2, 64)
            self.cache_spin.setValue(args.cache_size or 12)
            self.max_files_spin = QtWidgets.QSpinBox()
            self.max_files_spin.setRange(0, 10000)
            self.max_files_spin.setValue(args.max_files or 0)
            formats_text = args.formats or ",".join(sorted(SUPPORTED_EXTENSIONS))
            self.formats_edit = QtWidgets.QLineEdit(formats_text)
            self.families_edit = QtWidgets.QLineEdit(
                ",".join(prefill_families) if prefill_families else ""
            )
            self.overwrite_check = QtWidgets.QCheckBox()
            self.overwrite_check.setChecked(args.overwrite)
            self.fov_label_widget = QtWidgets.QLabel()
            self.search_scale_label_widget = QtWidgets.QLabel()
            self.search_attempts_label_widget = QtWidgets.QLabel()
            self.max_radius_label_widget = QtWidgets.QLabel()
            self.downsample_label_widget = QtWidgets.QLabel()
            self.workers_label_widget = QtWidgets.QLabel()
            self.cache_label_widget = QtWidgets.QLabel()
            self.max_files_label_widget = QtWidgets.QLabel()
            self.formats_label_widget = QtWidgets.QLabel()
            self.families_label_widget = QtWidgets.QLabel()
            form.addRow(self.fov_label_widget, self.fov_spin)
            form.addRow(self.search_scale_label_widget, self.search_scale_spin)
            form.addRow(self.search_attempts_label_widget, self.search_attempts_spin)
            form.addRow(self.max_radius_label_widget, self.max_radius_spin)
            form.addRow(self.downsample_label_widget, self.downsample_spin)
            form.addRow(self.workers_label_widget, self.workers_spin)
            form.addRow(self.cache_label_widget, self.cache_spin)
            form.addRow(self.max_files_label_widget, self.max_files_spin)
            form.addRow(self.formats_label_widget, self.formats_edit)
            form.addRow(self.families_label_widget, self.families_edit)
            form.addRow(self.overwrite_check)
            return self.options_box

        def _build_splitter(self) -> QtWidgets.QSplitter:
            splitter = QtWidgets.QSplitter()
            self.files_view = QtWidgets.QTreeWidget()
            self.files_view.setHeaderLabels(
                [
                    GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["files_header"],
                    GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["status_header"],
                    GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["details_header"],
                ]
            )
            self.files_view.setRootIsDecorated(False)
            self.files_view.setAlternatingRowColors(True)
            splitter.addWidget(self.files_view)
            self.log_box = QtWidgets.QGroupBox()
            log_layout = QtWidgets.QVBoxLayout(self.log_box)
            self.log_view = QtWidgets.QPlainTextEdit()
            self.log_view.setReadOnly(True)
            log_layout.addWidget(self.log_view)
            splitter.addWidget(self.log_box)
            splitter.setSizes([800, 400])
            return splitter

        def _build_bottom_row(self) -> QtWidgets.QLayout:
            row = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton()
            self.stop_btn = QtWidgets.QPushButton()
            self.stop_btn.setEnabled(False)
            self.start_btn.clicked.connect(self._start_solving)
            self.stop_btn.clicked.connect(self._stop_solving)
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setMinimum(0)
            self.progress_bar.setValue(0)
            self.status_label = QtWidgets.QLabel("")
            row.addWidget(self.start_btn)
            row.addWidget(self.stop_btn)
            row.addWidget(self.progress_bar, 1)
            row.addWidget(self.status_label)
            return row

        def _switch_language(self, code: str) -> None:
            if code not in GUI_TRANSLATIONS:
                return
            if code == self._language:
                self._apply_language()
                return
            self._language = code
            self._apply_language()

        def _apply_language(self) -> None:
            self.setWindowTitle(self._text("window_title"))
            if hasattr(self, "language_menu"):
                self.language_menu.setTitle(self._text("language_menu"))
                for code, action in self._language_actions.items():
                    label_key = f"language_action_{code}"
                    action.setText(self._text(label_key))
                    action.setChecked(code == self._language)
            browse_label = self._text("browse_button")
            self.browse_db_btn.setText(browse_label)
            self.browse_in_btn.setText(browse_label)
            self.db_label.setText(self._text("database_label"))
            self.input_label.setText(self._text("input_label"))
            self.scan_btn.setText(self._text("scan_button"))
            self.options_box.setTitle(self._text("options_box"))
            self.fov_label_widget.setText(self._text("fov_label"))
            self.search_scale_label_widget.setText(self._text("search_scale_label"))
            self.search_attempts_label_widget.setText(self._text("search_attempts_label"))
            self.max_radius_label_widget.setText(self._text("max_radius_label"))
            self.downsample_label_widget.setText(self._text("downsample_label"))
            self.workers_label_widget.setText(self._text("threads_label"))
            self.cache_label_widget.setText(self._text("cache_label"))
            self.max_files_label_widget.setText(self._text("max_files_label"))
            self.formats_label_widget.setText(self._text("formats_label"))
            self.families_label_widget.setText(self._text("families_label"))
            self.overwrite_check.setText(self._text("overwrite_label"))
            self.files_view.setHeaderLabels(
                [
                    self._text("files_header"),
                    self._text("status_header"),
                    self._text("details_header"),
                ]
            )
            self.log_box.setTitle(self._text("log_box"))
            self.start_btn.setText(self._text("run_button"))
            self.stop_btn.setText(self._text("stop_button"))
            if self._should_reset_status_label():
                self.status_label.setText(self._text("status_ready"))
            self.max_radius_spin.setSpecialValueText(self._text("special_auto"))
            self._retranslate_status_items()

        def _retranslate_status_items(self) -> None:
            for item in self._item_by_path.values():
                status_value = item.data(1, QtCore.Qt.UserRole)
                if isinstance(status_value, str):
                    item.setText(1, self._status_label_for(status_value))

        def _should_reset_status_label(self) -> bool:
            if self._worker is not None:
                return False
            current = self.status_label.text().strip()
            ready_values = {values["status_ready"] for values in GUI_TRANSLATIONS.values()}
            ready_values.add("")
            return current in ready_values

        def _text(self, key: str, **kwargs: object) -> str:
            catalog = GUI_TRANSLATIONS.get(self._language, {})
            template = catalog.get(key)
            if template is None:
                template = GUI_TRANSLATIONS.get(GUI_FALLBACK_LANGUAGE, {}).get(key, key)
            try:
                return template.format(**kwargs)
            except (KeyError, ValueError):
                return template

        def _status_label_for(self, status: str) -> str:
            key = f"status_{status}"
            catalog = GUI_TRANSLATIONS.get(self._language, {})
            value = catalog.get(key)
            if value is None:
                value = GUI_TRANSLATIONS.get(GUI_FALLBACK_LANGUAGE, {}).get(key, status)
            return value

        def _prefill_from_args(self, cli_args: argparse.Namespace) -> None:
            if cli_args.db_root:
                self.db_edit.setText(str(cli_args.db_root))
            if cli_args.input_dir:
                self.input_edit.setText(str(cli_args.input_dir))
                QtCore.QTimer.singleShot(100, self.scan_files)

        # --- Actions -------------------------------------------------------------
        def _pick_directory(self, line_edit: QtWidgets.QLineEdit, *, trigger_scan: bool = False) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self._text("dialog_select_directory"),
            )
            if directory:
                line_edit.setText(directory)
                if trigger_scan:
                    self.scan_files()

        def scan_files(self) -> None:
            try:
                files = self._gather_candidate_files()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))
                return
            self._pending_files = files
            self._refresh_file_list()
            self._log(self._text("info_files_detected", count=len(files)))

        def _gather_candidate_files(self) -> List[Path]:
            path = self.input_edit.text().strip()
            if not path:
                raise ValueError(self._text("error_select_input"))
            directory = Path(path).expanduser()
            if not directory.is_dir():
                raise ValueError(self._text("error_input_missing", path=directory))
            formats = self._parse_formats()
            files = [p.resolve() for p in _iter_image_files(directory, formats)]
            limit = self.max_files_spin.value()
            if limit > 0:
                files = files[:limit]
            self._current_input_dir = directory
            return files

        def _parse_formats(self) -> List[str]:
            text = self.formats_edit.text().replace(";", ",").strip()
            if not text:
                return list(sorted(SUPPORTED_EXTENSIONS))
            formats: List[str] = []
            for chunk in text.split(","):
                chunk = chunk.strip().lower()
                if not chunk:
                    continue
                if not chunk.startswith("."):
                    chunk = f".{chunk}"
                formats.append(chunk)
            return formats

        def _refresh_file_list(self) -> None:
            self.files_view.clear()
            self._item_by_path.clear()
            for path in self._pending_files:
                item = QtWidgets.QTreeWidgetItem(
                    [self._format_path(path), self._status_label_for("waiting"), ""]
                )
                item.setData(1, QtCore.Qt.UserRole, "waiting")
                self.files_view.addTopLevelItem(item)
                self._item_by_path[path.resolve()] = item
            self.files_view.resizeColumnToContents(0)

        def _format_path(self, path: Path) -> str:
            base = self._current_input_dir
            if base:
                try:
                    return str(path.relative_to(base))
                except ValueError:
                    pass
            return path.name

        def _build_config(self) -> SolveConfig:
            db_path = self.db_edit.text().strip()
            if not db_path:
                raise ValueError(self._text("error_database_required"))
            db_root = Path(db_path).expanduser()
            if not db_root.is_dir():
                raise ValueError(self._text("error_database_missing", path=db_root))
            if not self._current_input_dir:
                raise ValueError(self._text("error_no_input_dir"))
            families_text = self.families_edit.text().replace(";", ",").strip()
            families = [
                chunk.strip().lower()
                for chunk in families_text.split(",")
                if chunk.strip()
            ]
            formats = tuple(self._parse_formats())
            max_files = self.max_files_spin.value() or None
            max_radius_value = self.max_radius_spin.value()
            max_radius = max_radius_value if max_radius_value > 0 else None
            return SolveConfig(
                db_root=db_root,
                input_dir=self._current_input_dir,
                families=families or None,
                fov_deg=self.fov_spin.value(),
                downsample=self.downsample_spin.value(),
                overwrite=self.overwrite_check.isChecked(),
                workers=self.workers_spin.value(),
                formats=formats,
                max_files=max_files,
                cache_size=self.cache_spin.value(),
                search_radius_scale=self.search_scale_spin.value(),
                search_radius_attempts=self.search_attempts_spin.value(),
                max_search_radius_deg=max_radius,
                blind_enabled=args.blind_enabled,
                blind_db_chain=_parse_db_chain_arg(args.blind_db),
                blind_profile=args.auto_blind_profile,
                blind_timeout=args.blind_timeout,
                blind_astap_exe=args.blind_astap_exe,
                blind_extra_args=_parse_extra_args_arg(args.blind_extra_args),
            )

        def _start_solving(self) -> None:
            if self._worker is not None:
                return
            if not self._pending_files:
                self.scan_files()
                if not self._pending_files:
                    QtWidgets.QMessageBox.information(
                        self,
                        "ZeSolver",
                        self._text("dialog_no_files"),
                    )
                    return
            try:
                config = self._build_config()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))
                return
            self._results_seen = 0
            target_total = len(self._pending_files)
            limit = self.max_files_spin.value()
            if limit > 0:
                target_total = min(target_total, limit)
            target_total = max(1, target_total)
            self.progress_bar.setMaximum(target_total)
            self.progress_bar.setValue(0)
            self.status_label.setText(f"0 / {target_total}")
            self._set_running(True)
            self._worker = SolveRunner(config, self._pending_files, self._text)
            self._worker.started.connect(self._on_worker_started)
            self._worker.progress.connect(self._on_worker_progress)
            self._worker.info.connect(self._log)
            self._worker.error.connect(self._on_worker_error)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.start()

        def _stop_solving(self) -> None:
            if self._worker:
                self._worker.request_cancel()
                self._log(self._text("log_stop_requested"))

        def _set_running(self, running: bool) -> None:
            self.start_btn.setEnabled(not running)
            self.stop_btn.setEnabled(running)
            self.scan_btn.setEnabled(not running)
            self.db_edit.setEnabled(not running)
            self.input_edit.setEnabled(not running)

        def _on_worker_started(self, total: int) -> None:
            if total == 0:
                self.progress_bar.setMaximum(1)
            self._log(self._text("processing_count", count=total))

        def _on_worker_progress(self, result: ImageSolveResult) -> None:
            if result.run_info:
                for key, payload in result.run_info:
                    self._log(self._text(key, **payload))
            if not result.path.is_dir():
                self._results_seen += 1
                self.progress_bar.setValue(min(self._results_seen, self.progress_bar.maximum()))
                self.status_label.setText(f"{self._results_seen} / {self.progress_bar.maximum()}")
                self._update_item(result)
            status_text = self._status_label_for(result.status)
            if result.message:
                self._log(
                    self._text(
                        "log_result",
                        status=status_text,
                        path=result.path,
                        message=result.message,
                    )
                )
            else:
                self._log(
                    self._text(
                        "log_result_no_details",
                        status=status_text,
                        path=result.path,
                    )
                )

        def _update_item(self, result: ImageSolveResult) -> None:
            key = result.path.resolve()
            item = self._item_by_path.get(key)
            if item is None:
                item = QtWidgets.QTreeWidgetItem([self._format_path(result.path), "", ""])
                self.files_view.addTopLevelItem(item)
                self._item_by_path[key] = item
            item.setText(1, self._status_label_for(result.status))
            item.setData(1, QtCore.Qt.UserRole, result.status)
            item.setText(2, result.message or "")
            color_map = {
                "solved": QtGui.QColor("#2b8a3e"),
                "failed": QtGui.QColor("#c92a2a"),
                "skipped": QtGui.QColor("#5f3dc4"),
            }
            color = color_map.get(result.status)
            if color:
                for idx in range(3):
                    item.setForeground(idx, color)

        def _on_worker_error(self, message: str) -> None:
            QtWidgets.QMessageBox.critical(self, "ZeSolver", message)
            self._log(f"{self._text('log_error_prefix')}: {message}")

        def _on_worker_finished(self) -> None:
            self._log(self._text("log_processing_done"))
            self._set_running(False)
            if self._worker:
                self._worker.deleteLater()
            self._worker = None
            self.status_label.setText(self._text("status_ready"))

        def _log(self, message: str) -> None:
            timestamp = time.strftime("%H:%M:%S")
            self.log_view.appendPlainText(f"[{timestamp}] {message}")
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
            self.statusBar().showMessage(message, 5000)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI hook
            if self._worker:
                self._worker.request_cancel()
                self._worker.wait(1500)
            super().closeEvent(event)

    QtWidgets.QApplication.setApplicationName("ZeSolver")
    QtWidgets.QApplication.setApplicationVersion(APP_VERSION)
    app = QtWidgets.QApplication(sys.argv)
    window = ZeSolverWindow()
    window.show()
    return app.exec()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if _should_launch_gui(args):
        return launch_gui(args)
    return run_cli(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
