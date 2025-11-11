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
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.io import fits


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
LOG_FILE = ROOT_DIR / "zesolver.log"

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
    blind_solve,
    has_valid_wcs,
    near_solve,
)
from zeblindsolver.metadata_solver import NearSolveConfig as NearIndexConfig
from zeblindsolver.db_convert import (
    DEFAULT_MAG_CAP,
    DEFAULT_MAX_QUADS_PER_TILE,
    DEFAULT_MAX_STARS,
    build_index_from_astap,
)
from zeblindsolver.zeblindsolver import SolveConfig as BlindSolveConfig, solve_blind as python_solve_blind
from zeblindsolver.quad_index_builder import validate_index as validate_zeblind_index
from zeblindsolver import presets as preset_utils


FITS_EXTENSIONS = {".fit", ".fits", ".fts"}
RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = FITS_EXTENSIONS | RASTER_EXTENSIONS
SIDE_CAR_SUFFIX = ".wcs.json"
FITS_MEMMAP_FORBIDDEN_KEYS = ("BZERO", "BSCALE", "BLANK")


def _parse_formats_value(value: Optional[str]) -> list[str]:
    text = (value or "").replace(";", ",").strip()
    if not text:
        return list(sorted(SUPPORTED_EXTENSIONS))
    formats: list[str] = []
    for chunk in text.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        if not chunk.startswith("."):
            chunk = f".{chunk}"
        formats.append(chunk)
    if not formats:
        return list(sorted(SUPPORTED_EXTENSIONS))
    return formats

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
        "families_label": "Catalogue",
        "overwrite_label": "Réécrire les WCS existants",
        "blind_label": "Activer le blind solver",
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
        "solver_tab": "Solveur",
        "settings_tab": "Réglages",
        "performance_tab": "Performance",
        "database_tab": "Base de données",
        "database_tab_title": "Base de données",
        "presets_title": "Presets d’instruments",
        "fov_mode_title": "Par FOV",
        "focal_length_mm": "Focale (mm)",
        "pixel_size_um": "Taille pixel (µm)",
        "resolution_px": "Résolution (px)",
        "reducer_factor": "Réducteur (x)",
        "binning": "Binning",
        "compute_button": "Calculer",
        "recommendations_title": "Recommandations d’index",
        "est_scale": "Échantillonnage (""/px)",
        "est_fov": "Champ (°)",
        "mag_cap_suggested": "Magnitude max conseillée",
        "quads_profile": "Profil quads",
        "max_quads_per_tile": "Quads max/tile",
        "spec_warning_unknown": "Spécifications à confirmer (approx.)",
        "database_tab_title": "Base de données",
        "select_db_root": "Dossier database",
        "data_sources": "Sources de données (tuiles HNSKY/ASTAP)",
        "warn_sep_dirs": "L’index doit être dans un dossier différent de database.",
        "settings_db_label": "Base ASTAP",
        "settings_index_label": "Dossier d'index",
        "settings_mag_label": "Magnitude max",
        "settings_max_stars_label": "Étoiles max",
        "settings_max_quads_label": "Quads max",
        "settings_sample_label": "Fichier test (FITS)",
        "settings_save_btn": "Sauvegarder",
        "settings_build_btn": "Construire l'index",
        "settings_run_btn": "Lancer le blind solve",
        "settings_near_btn": "Résolution WCS (Python, sans quads)",
        "settings_blind_group": "Blind solver (Python)",
        "settings_blind_max_stars_label": "Étoiles max (blind)",
        "settings_blind_max_quads_label": "Quads max (blind)",
        "settings_blind_max_candidates_label": "Candidats max (blind)",
        "settings_blind_pixel_tol_label": "Tolérance pixel (blind)",
        "settings_blind_quality_inliers_label": "Inliers mini (blind)",
        "settings_blind_quality_rms_label": "RMS max px (blind)",
        "settings_log": "Journal d'index",
        "settings_saved": "Réglages sauvegardés",
        "settings_index_missing": "Répertoire d'index requis.",
        "settings_sample_required": "Choisis un fichier FITS à tester.",
        "settings_index_result": "Index {status}: {message}",
        "settings_index_health_ok": "Index OK: {tiles} tuiles, {empty} vides ({percent:.1f}%).",
        "settings_index_health_bad": "Index incomplet: {empty}/{tiles} tuiles vides; anneaux touchés: {rings}. Reconstruis l’index.",
        "settings_blind_result": "Blind {status}: {message}",
        "settings_blind_fast_label": "Mode rapide (S-seul, fallback M/L)",
        "settings_near_result": "Near {status}: {message}",
        "settings_perf_near_cache_label": "Cache tuiles (near)",
        "settings_perf_near_max_tiles_label": "Tuiles candidates max (near)",
        "settings_perf_detect_label": "Dispositif détection (GPU/CPU)",
        "settings_perf_io_label": "Concurrence I/O (Auto=0)",
        "settings_perf_near_warm_label": "Near rapide (séquence)",
        "settings_build_start": "Construction de l'index dans {path}… (cela peut prendre plusieurs minutes)",
        "settings_rebuild_title": "Reconstruire l'index ?",
        "settings_rebuild_text": "Un index existe déjà dans {path}. Le reconstruire ?",
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
        "families_label": "Catalog",
        "overwrite_label": "Overwrite existing WCS",
        "blind_label": "Enable blind solver",
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
        "solver_tab": "Solver",
        "settings_tab": "Settings",
        "performance_tab": "Performance",
        "database_tab": "Database",
        "database_tab_title": "Database",
        "presets_title": "Instrument presets",
        "fov_mode_title": "By FOV",
        "focal_length_mm": "Focal length (mm)",
        "pixel_size_um": "Pixel size (µm)",
        "resolution_px": "Resolution (px)",
        "reducer_factor": "Reducer (x)",
        "binning": "Binning",
        "compute_button": "Compute",
        "recommendations_title": "Index recommendations",
        "est_scale": "Pixel scale (""/px)",
        "est_fov": "Field of view (°)",
        "mag_cap_suggested": "Suggested mag cap",
        "quads_profile": "Quads profile",
        "max_quads_per_tile": "Max quads/tile",
        "spec_warning_unknown": "Specifications need confirmation (approx.)",
        "database_tab_title": "Database",
        "select_db_root": "Database folder",
        "data_sources": "Data sources (HNSKY/ASTAP shards)",
        "warn_sep_dirs": "Index must be separate from database.",
        "settings_db_label": "ASTAP database",
        "settings_index_label": "Index root",
        "settings_mag_label": "Max magnitude",
        "settings_max_stars_label": "Max stars",
        "settings_max_quads_label": "Max quads",
        "settings_sample_label": "Sample FITS",
        "settings_save_btn": "Save settings",
        "settings_build_btn": "Build index",
        "settings_run_btn": "Run blind solve",
        "settings_near_btn": "Near solve (Python, no quads)",
        "settings_blind_group": "Blind solver (Python)",
        "settings_blind_max_stars_label": "Max stars (blind)",
        "settings_blind_max_quads_label": "Max quads (blind)",
        "settings_blind_max_candidates_label": "Max candidates (blind)",
        "settings_blind_pixel_tol_label": "Pixel tolerance (blind)",
        "settings_blind_quality_inliers_label": "Min inliers (blind)",
        "settings_blind_quality_rms_label": "Max RMS px (blind)",
        "settings_log": "Index log",
        "settings_saved": "Settings saved",
        "settings_index_missing": "Index directory is required.",
        "settings_sample_required": "Select a FITS file to test.",
        "settings_index_result": "Index {status}: {message}",
        "settings_index_health_ok": "Index OK: {tiles} tiles, {empty} empty ({percent:.1f}%).",
        "settings_index_health_bad": "Index incomplete: {empty}/{tiles} empty tiles; affected rings: {rings}. Rebuild the index.",
        "settings_blind_result": "Blind {status}: {message}",
        "settings_blind_fast_label": "Fast mode (S-only, fallback M/L)",
        "settings_near_result": "Near {status}: {message}",
        "settings_perf_near_cache_label": "Near tile cache",
        "settings_perf_near_max_tiles_label": "Max tile candidates (near)",
        "settings_perf_detect_label": "Star detection device",
        "settings_perf_io_label": "I/O concurrency (Auto=0)",
        "settings_perf_near_warm_label": "Fast near (sequential warm-start)",
        "settings_build_start": "Building index at {path}… (this may take several minutes)",
        "settings_rebuild_title": "Rebuild Index?",
        "settings_rebuild_text": "An index already exists at {path}. Rebuild it?",
    },
}

# Backfill translations for downloads UI (added post hoc) without overriding existing keys
_GUI_DOWNLOADS_I18N = {
    "fr": {
        "downloads_title": "Telechargements",
        "add_to_queue": "Ajouter a la file",
        "start_all": "Tout demarrer",
        "pause_all": "Tout mettre en pause",
        "verify_hashes": "Verifier les empreintes",
        "add_selected": "Ajouter la selection a la file",
    },
    "en": {
        "downloads_title": "Downloads",
        "add_to_queue": "Add to queue",
        "start_all": "Start all",
        "pause_all": "Pause all",
        "verify_hashes": "Verify hashes",
        "add_selected": "Add selected to queue",
    },
}
for _lang, _mapping in _GUI_DOWNLOADS_I18N.items():
    base = GUI_TRANSLATIONS.setdefault(_lang, {})
    for _k, _v in _mapping.items():
        if _k not in base:
            base[_k] = _v

SETTINGS_PATH = Path.home() / ".zesolver_settings.json"


@dataclass
class PersistentSettings:
    db_root: Optional[str] = None
    index_root: Optional[str] = None
    mag_cap: float = DEFAULT_MAG_CAP
    max_stars: int = DEFAULT_MAX_STARS
    max_quads_per_tile: int = DEFAULT_MAX_QUADS_PER_TILE
    sample_fits: Optional[str] = None
    # Preset/FOV persistence
    last_preset_id: Optional[str] = None
    last_fov_focal_mm: float = 0.0
    last_fov_pixel_um: float = 0.0
    last_fov_res_w: int = 0
    last_fov_res_h: int = 0
    last_fov_reducer: float = 1.0
    last_fov_binning: int = 1
    # Blind solver tunables
    blind_max_stars: int = 800
    blind_max_quads: int = 12000
    blind_max_candidates: int = 12
    blind_pixel_tolerance: float = 3.0
    blind_quality_inliers: int = 60
    blind_quality_rms: float = 1.0
    blind_fast_mode: bool = False
    # Near solver performance
    near_max_tile_candidates: int = 48
    near_tile_cache_size: int = 128
    near_detect_backend: str = "auto"  # auto|cpu|cuda
    near_detect_device: int = 0
    io_concurrency: int = 0
    near_warm_start: bool = True
    # Solver panel persisted settings
    solver_fov_deg: float = DEFAULT_FOV_DEG
    solver_search_scale: float = DEFAULT_SEARCH_RADIUS_SCALE
    solver_search_attempts: int = DEFAULT_SEARCH_RADIUS_ATTEMPTS
    solver_max_radius_deg: float = 0.0  # 0 = Auto
    solver_downsample: int = 1
    solver_workers: int = 0  # 0 = auto (half CPUs)
    solver_cache_size: int = 12
    solver_max_files: int = 0
    solver_formats: Optional[str] = None
    solver_family: Optional[str] = None  # lower-case key, None = Auto
    solver_blind_enabled: bool = True
    solver_overwrite: bool = True


def load_persistent_settings() -> PersistentSettings:
    if not SETTINGS_PATH.exists():
        return PersistentSettings()
    try:
        payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return PersistentSettings()
    if not isinstance(payload, dict):
        return PersistentSettings()
    return PersistentSettings(
        db_root=payload.get("db_root"),
        index_root=payload.get("index_root"),
        mag_cap=float(payload.get("mag_cap", DEFAULT_MAG_CAP)),
        max_stars=int(payload.get("max_stars", DEFAULT_MAX_STARS)),
        max_quads_per_tile=int(payload.get("max_quads_per_tile", DEFAULT_MAX_QUADS_PER_TILE)),
        sample_fits=payload.get("sample_fits"),
        last_preset_id=(payload.get("last_preset_id") or None),
        last_fov_focal_mm=float(payload.get("last_fov_focal_mm", 0.0)),
        last_fov_pixel_um=float(payload.get("last_fov_pixel_um", 0.0)),
        last_fov_res_w=int(payload.get("last_fov_res_w", 0)),
        last_fov_res_h=int(payload.get("last_fov_res_h", 0)),
        last_fov_reducer=float(payload.get("last_fov_reducer", 1.0)),
        last_fov_binning=int(payload.get("last_fov_binning", 1)),
        blind_max_stars=int(payload.get("blind_max_stars", 800)),
        blind_max_quads=int(payload.get("blind_max_quads", 12000)),
        blind_max_candidates=int(payload.get("blind_max_candidates", 12)),
        blind_pixel_tolerance=float(payload.get("blind_pixel_tolerance", 3.0)),
        blind_quality_inliers=int(payload.get("blind_quality_inliers", 12)),
        blind_quality_rms=float(payload.get("blind_quality_rms", 1.0)),
        blind_fast_mode=bool(payload.get("blind_fast_mode", False)),
        near_max_tile_candidates=int(payload.get("near_max_tile_candidates", 48)),
        near_tile_cache_size=int(payload.get("near_tile_cache_size", 128)),
        near_detect_backend=str(payload.get("near_detect_backend", "auto")),
        near_detect_device=int(payload.get("near_detect_device", 0)),
        io_concurrency=int(payload.get("io_concurrency", 0)),
        near_warm_start=bool(payload.get("near_warm_start", True)),
        solver_fov_deg=float(payload.get("solver_fov_deg", DEFAULT_FOV_DEG)),
        solver_search_scale=float(payload.get("solver_search_scale", DEFAULT_SEARCH_RADIUS_SCALE)),
        solver_search_attempts=int(payload.get("solver_search_attempts", DEFAULT_SEARCH_RADIUS_ATTEMPTS)),
        solver_max_radius_deg=float(payload.get("solver_max_radius_deg", 0.0)),
        solver_downsample=int(payload.get("solver_downsample", 1)),
        solver_workers=int(payload.get("solver_workers", 0)),
        solver_cache_size=int(payload.get("solver_cache_size", 12)),
        solver_max_files=int(payload.get("solver_max_files", 0)),
        solver_formats=payload.get("solver_formats"),
        solver_family=(payload.get("solver_family") or None),
        solver_blind_enabled=bool(payload.get("solver_blind_enabled", True)),
        solver_overwrite=bool(payload.get("solver_overwrite", True)),
    )


def save_persistent_settings(settings: PersistentSettings) -> None:
    data = asdict(settings)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _configure_logging(level_name: str) -> None:
    """Setup console + file logging (appends to zesolver.log)."""
    level = getattr(logging, (level_name or "").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    configured = getattr(_configure_logging, "_configured", False)
    if configured:
        for handler in root.handlers:
            handler.setLevel(level)
        return
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    file_handler: Optional[logging.Handler] = None
    try:
        # Truncate the log file at each program start
        file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem edge case
        print(f"Warning: unable to open log file {LOG_FILE}: {exc}", file=sys.stderr)
    else:
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    _configure_logging._configured = True  # type: ignore[attr-defined]
    if file_handler:
        root.info("Logging initialized — writing to %s", LOG_FILE)
    else:
        root.info("Logging initialized (no file handler)")


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
    overwrite: bool = True
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
    blind_skip_if_valid: bool = True
    blind_index_path: Optional[Path] = None
    # Limit concurrent disk I/O to avoid thrashing (0 = auto based on workers)
    io_concurrency: int = 0
    # Force periodic GC every N results (0 = disabled)
    gc_interval: int = 0
    # Near solver (index-based) performance knobs
    near_max_tile_candidates: int = 48
    near_tile_cache_size: int = 128
    near_detect_backend: Optional[str] = None
    near_detect_device: Optional[int] = None
    near_warm_start: bool = True

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
        if self.blind_index_path:
            object.__setattr__(self, "blind_index_path", Path(self.blind_index_path).expanduser())
        if self.search_radius_scale < 1.0:
            raise ValueError("search_radius_scale must be >= 1.0")
        if self.search_radius_attempts < 1:
            raise ValueError("search_radius_attempts must be >= 1")
        if self.max_search_radius_deg is not None and self.max_search_radius_deg <= 0:
            raise ValueError("max_search_radius_deg must be positive")


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
        # Sequential near-solver warm start: (scale_deg_per_px, rotation_rad, parity)
        self._near_seed: Optional[tuple[float, float, int]] = None
        # Cache to avoid repeating expensive blind index preflight checks per file
        self._blind_index_checked: bool = False
        self._blind_index_checked_root: Optional[Path] = None
        # Optional cancellation event set by the GUI runner for responsive Stop
        self._cancel_event: Optional[threading.Event] = None
        # Throttle concurrent I/O (reads/writes)
        workers = int(self.config.workers or _default_worker_count())
        io_limit = int(self.config.io_concurrency or 0)
        if io_limit <= 0:
            # Auto‑tune I/O concurrency from a quick throughput probe on the input drive
            try:
                img_limit = self._autotune_io_limit(self.config.input_dir, workers)
                idx_limit = None
                idx_path = getattr(self.config, 'blind_index_path', None)
                if idx_path is not None:
                    try:
                        idx_limit = self._autotune_io_limit(Path(idx_path), workers)
                    except Exception:
                        idx_limit = None
                if idx_limit is not None:
                    io_limit = min(img_limit, idx_limit)
                    logging.info(
                        "I/O auto-tune: input=%d, index=%d -> using %d",
                        img_limit,
                        idx_limit,
                        io_limit,
                    )
                else:
                    io_limit = img_limit
                    logging.info("I/O auto-tune: input=%d", img_limit)
            except Exception:
                io_limit = 5 if workers >= 5 else max(1, (workers + 1) // 2)
        self._io_sema = threading.Semaphore(max(1, io_limit))
        logging.info("I/O concurrency set to %d (workers=%d)", io_limit, workers)

    @staticmethod
    def _autotune_io_limit(base_dir: Path, workers: int) -> int:
        """Estimate disk throughput and choose an I/O concurrency accordingly.

        Reads up to ~4 MiB from a few files to estimate MB/s; maps to a sensible
        number of parallel I/O slots. Keeps it conservative to avoid saturation.
        """
        import itertools
        sample_files = list(itertools.islice(_iter_image_files(base_dir, SUPPORTED_EXTENSIONS), 6))
        if not sample_files:
            return 5 if workers >= 5 else max(1, (workers + 1) // 2)
        total = 0
        start = time.perf_counter()
        target = 4 * 1024 * 1024  # 4 MiB per file max
        for p in sample_files:
            try:
                with open(p, 'rb', buffering=0) as fh:
                    chunk = fh.read(target)
                    total += len(chunk)
            except Exception:
                continue
        elapsed = max(1e-3, time.perf_counter() - start)
        mbps = (total / (1024 * 1024)) / elapsed
        # Map throughput to concurrency caps; NVMe typically > 800 MB/s
        if mbps >= 1500:
            cap = 12
        elif mbps >= 800:
            cap = 10
        elif mbps >= 400:
            cap = 8
        elif mbps >= 200:
            cap = 6
        elif mbps >= 120:
            cap = 5
        else:
            cap = 3
        return min(max(1, cap), max(1, workers))

    def set_cancel_event(self, event: Optional[threading.Event]) -> None:
        self._cancel_event = event

    def _cancelled(self) -> bool:
        return bool(self._cancel_event and self._cancel_event.is_set())

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
        logging.info("[solve] start %s", path.name)
        # Try metadata-assisted (near) solve first; fall back to blind on failure
        metadata: Optional[ImageMetadata] = None
        peaks: Optional[np.ndarray] = None
        try:
            if self._cancelled():
                return ImageSolveResult(path=path, status="skipped", message="cancelled")
            t0 = time.perf_counter()
            data, metadata = self._load_image(path)
            logging.info(
                "[solve] loaded %s: %dx%d, has_wcs=%s, ra=%s, dec=%s (%.2fs)",
                path.name,
                metadata.width,
                metadata.height,
                metadata.has_wcs,
                f"{metadata.ra_deg:.6f}" if metadata.ra_deg is not None else "-",
                f"{metadata.dec_deg:.6f}" if metadata.dec_deg is not None else "-",
                time.perf_counter() - t0,
            )
            if metadata.has_wcs and not self.config.overwrite:
                raise SolveError("WCS already present (use --overwrite to recompute)", skip=True)
            # Prefer the internal index-powered near solver when an index root is configured.
            # This uses Python-only metadata-assisted solving without quads.
            if (
                self.config.blind_index_path
                and path.suffix.lower() in FITS_EXTENSIONS
            ):
                if self._cancelled():
                    return ImageSolveResult(path=path, status="skipped", message="cancelled")
                near_first = self._run_index_near_solver(path)
                if near_first is not None:
                    near_first.duration_s = time.perf_counter() - start
                    near_first.run_info.extend(run_info)
                    return near_first
            if metadata.ra_deg is None or metadata.dec_deg is None:
                # For rasters without metadata: attempt blind via temp FITS bridge
                if (
                    metadata.kind == "raster"
                    and self._should_try_blind(path)
                    and self.config.blind_index_path is not None
                ):
                    bridged = self._run_blind_on_raster(raster_path=path, raster_data=data, run_info=run_info)
                    if bridged is not None:
                        bridged.duration_s = time.perf_counter() - start
                        return bridged
                raise SolveError("Missing RA/DEC metadata", skip=False)
            if self._cancelled():
                return ImageSolveResult(path=path, status="skipped", message="cancelled")
            logging.info("[solve] detecting stars in %s", path.name)
            t1 = time.perf_counter()
            peaks = _detect_stars(
                data,
                downsample=self.config.downsample,
                max_stars=self.config.max_image_stars,
            )
            logging.info("[solve] detected %d stars in %.2fs", 0 if peaks is None else peaks.shape[0], time.perf_counter() - t1)
            if peaks.shape[0] < 5:
                raise SolveError("Not enough stars detected in the frame")
            # Effective FOV for classic catalog-based solve; when GUI FOV is 0 (Auto),
            # fall back to default to avoid a zero-radius search.
            fov_eff = self.config.fov_deg if self.config.fov_deg and self.config.fov_deg > 0 else DEFAULT_FOV_DEG
            field_height = fov_eff * (metadata.height / metadata.width)
            base_radius = 0.55 * math.hypot(fov_eff, field_height)
            radius_candidates = self._radius_candidates(base_radius)
            final_error: Optional[SolveError] = None
            for radius_index, radius in enumerate(radius_candidates):
                if self._cancelled():
                    return ImageSolveResult(path=path, status="skipped", message="cancelled")
                last_error: Optional[SolveError] = None
                combined_label: Optional[str] = None
                combined_catalog_family: Optional[str] = None
                if self.config.families and len(self.config.families) == 1:
                    combined_catalog_family = self.config.families[0]
                    combined_label = self._family_label(combined_catalog_family)
                try:
                    logging.info(
                        "[solve] query radius=%.2f° (attempt %d/%d) for %s",
                        radius,
                        radius_index + 1,
                        len(radius_candidates),
                        path.name,
                    )
                    if self._cancelled():
                        return ImageSolveResult(path=path, status="skipped", message="cancelled")
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
                    if self._cancelled():
                        return ImageSolveResult(path=path, status="skipped", message="cancelled")
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
            if self._cancelled():
                return ImageSolveResult(path=path, status="skipped", message="cancelled")
            blind_result = self._resolve_with_blind_after_failure(
                path=path,
                run_info=run_info,
                cached_result=None,
                ra_hint=metadata.ra_deg if metadata else None,
                dec_hint=metadata.dec_deg if metadata else None,
                metadata=metadata,
                peaks=peaks,
            )
            if blind_result is not None:
                blind_result.duration_s = time.perf_counter() - start
                return blind_result
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
        with self._io_sema:
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
                with self._io_sema:
                    hdul = fits.open(path, memmap=memmap)
                with hdul:
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
                    # Normalize to 2D luminance for star detection
                    arr = np.asarray(data)
                    if arr.ndim == 3:
                        # Handle channel‑last RGB(A) or multi‑plane data
                        if arr.shape[-1] in (3, 4):
                            arr = np.mean(arr[..., :3], axis=-1)
                        elif arr.shape[0] in (3, 4):
                            arr = np.mean(arr[:3, ...], axis=0)
                        else:
                            # Fallback: first plane
                            arr = arr[0]
                    # Safe dtype conversion for NumPy>=2
                    arr = np.asarray(arr)
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32, copy=False)
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
                        extra={"OBJECT": str(header.get("OBJECT", "")).strip() or ""},
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
        with self._io_sema:
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
                # Opportunistically write a sidecar when we recover RA/DEC from a peer FITS
                try:
                    sidecar = path.with_suffix(path.suffix + SIDE_CAR_SUFFIX)
                    payload = {"ra_deg": float(ra), "dec_deg": float(dec), "source": str(peer_source)}
                    sidecar.write_text(json.dumps(payload, indent=2))
                except Exception:
                    pass
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
        # Allow blind on FITS directly, and on rasters via a temporary FITS bridge
        return self.config.blind_enabled and (
            path.suffix.lower() in FITS_EXTENSIONS or path.suffix.lower() in RASTER_EXTENSIONS
        )

    def _run_blind_solver(
        self,
        path: Path,
        run_info: list[tuple[str, dict[str, Any]]],
        *,
        skip_if_header_has_wcs: bool,
        skip_if_valid: bool,
        ra_hint: Optional[float] = None,
        dec_hint: Optional[float] = None,
    ) -> Optional[BlindSolveResult]:
        if self._cancelled():
            return None
        if not self._should_try_blind(path):
            return None
        index_root = self.config.blind_index_path
        if not index_root:
            logging.info("Blind solver skipped for %s: no index root configured", path.name)
            return None
        run_info.append(("run_info_blind_started", {"path": path.name}))
        logging.info(
            "Blind solver attempt for %s (index=%s, skip_if_valid=%s)",
            path.name,
            Path(index_root).name or str(index_root),
            skip_if_valid,
        )
        try:
            # Extra diagnostics: verify index structure before calling the solver
            try:
                root = Path(index_root).expanduser().resolve()
                needs_check = not self._blind_index_checked or (self._blind_index_checked_root != root)
                if needs_check:
                    preflight_start = time.perf_counter()
                    manifest = root / "manifest.json"
                    ht = root / "hash_tables"
                    l = ht / "quads_L.npz"
                    m = ht / "quads_M.npz"
                    s = ht / "quads_S.npz"
                    logging.info(
                        "Blind index check: root=%s manifest=%s L=%s M=%s S=%s",
                        root,
                        "ok" if manifest.exists() else "missing",
                        "ok" if l.exists() else "missing",
                        "ok" if m.exists() else "missing",
                        "ok" if s.exists() else "missing",
                    )
                    # Sanity-check once per run: detect rings with mostly-empty tiles
                    try:
                        health = validate_zeblind_index(root)
                        tiles = int(health.get("manifest_tile_count", 0) or 0)
                        empty = int(health.get("empty_tiles_total", 0) or 0)
                        ratio = float(health.get("empty_ratio_overall", 0.0) or 0.0)
                        rings = health.get("bad_empty_rings") or []
                        ring_str = ",".join(str(r) for r in rings) if rings else "-"
                        logging.info(
                            "Blind index health: %d/%d empty tiles (%.1f%%); affected rings: %s",
                            empty,
                            tiles,
                            100.0 * ratio,
                            ring_str,
                        )
                        if rings:
                            logging.warning(
                                "Index appears incomplete (rings %s mostly empty). Rebuild the index to populate missing tiles.",
                                ring_str,
                            )
                    except Exception:
                        # Non-fatal
                        pass
                    self._blind_index_checked = True
                    self._blind_index_checked_root = root
                    logging.info("Blind preflight completed in %.2fs", time.perf_counter() - preflight_start)
            except Exception:
                # Non-fatal; proceed to solver which will report a clear error
                pass
            if self._cancelled():
                return None
            result = blind_solve(
                fits_path=str(path),
                index_root=str(index_root),
                config=BlindSolveConfig(),
                log=logging.info,
                skip_if_valid=skip_if_valid,
                cancel_check=(self._cancelled if self._cancel_event else None),
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
            logging.info("Blind solver success for %s via %s (%s)", path.name, db_name, result["message"])
        else:
            run_info.append(("run_info_blind_failed", {"message": result["message"]}))
            logging.info("Blind solver failed for %s: %s", path.name, result["message"])
        return result

    def _run_blind_on_raster(
        self,
        *,
        raster_path: Path,
        raster_data: np.ndarray,
        run_info: list[tuple[str, dict[str, Any]]],
    ) -> Optional[ImageSolveResult]:
        """Bridge for blind-solving a raster image by using a temporary FITS.

        On success, writes a JSON WCS sidecar next to the raster and returns a solved result.
        """
        index_root = self.config.blind_index_path
        if not index_root:
            return None
        import tempfile
        from astropy.io import fits as _fits
        tmp_dir = Path(tempfile.gettempdir())
        tmp_fits = tmp_dir / f"zesolver_tmp_{os.getpid()}_{int(time.time()*1000)}.fits"
        try:
            # Minimal FITS with luminance data
            with self._io_sema:
                hdu = _fits.PrimaryHDU(np.ascontiguousarray(raster_data.astype(np.float32, copy=False)))
                hdul = _fits.HDUList([hdu])
                hdul.writeto(tmp_fits, overwrite=True)
                hdul.close()
        except Exception as exc:
            logging.info("Failed to write temporary FITS for raster %s: %s", raster_path.name, exc)
            return None
        try:
            # Reuse blind solver on the temp FITS
            result = self._run_blind_solver(
                tmp_fits,
                run_info,
                skip_if_header_has_wcs=False,
                skip_if_valid=False,
            )
            if not result or not result["success"]:
                return None
            # Extract WCS keywords from the temp FITS to build a JSON sidecar
            try:
                with self._io_sema:
                    hdul = fits.open(tmp_fits, mode="readonly", memmap=False)
                with hdul:
                    hdr = hdul[0].header
                    cd = np.array(
                        [
                            [float(hdr.get("CD1_1", 0.0)), float(hdr.get("CD1_2", 0.0))],
                            [float(hdr.get("CD2_1", 0.0)), float(hdr.get("CD2_2", 0.0))],
                        ],
                        dtype=np.float64,
                    )
                    crpix = np.array(
                        [float(hdr.get("CRPIX1", 0.0)), float(hdr.get("CRPIX2", 0.0))], dtype=np.float64
                    )
                    crval = np.array(
                        [float(hdr.get("CRVAL1", 0.0)), float(hdr.get("CRVAL2", 0.0))], dtype=np.float64
                    )
            except Exception as exc:
                logging.info("Unable to read WCS from temp FITS for %s: %s", raster_path.name, exc)
                return None
            # Write sidecar JSON using existing helper
            pixel_scale_arcsec = self._pixel_scale_arcsec(cd)
            sol = WCSSolution(crpix=crpix, crval=crval, cd=cd, matches=0, rms_pixels=None)
            meta = ImageMetadata(
                path=raster_path,
                kind="raster",
                width=raster_data.shape[1],
                height=raster_data.shape[0],
                ra_deg=None,
                dec_deg=None,
                source="blind",
                has_wcs=False,
                sidecar_path=raster_path.with_suffix(raster_path.suffix + SIDE_CAR_SUFFIX),
            )
            self._write_solution(meta, sol)
            msg = result["message"] or "blind solution (sidecar)"
            return ImageSolveResult(
                path=raster_path,
                status="solved",
                message=msg,
                matched_stars=0,
                rms_arcsec=None,
                pixel_scale_arcsec=pixel_scale_arcsec,
                metadata_source="blind",
                run_info=list(run_info),
            )
        finally:
            try:
                with self._io_sema:
                    tmp_fits.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    @staticmethod
    def _build_blind_result(
        path: Path,
        result: BlindSolveResult,
        run_info: list[tuple[str, dict[str, Any]]],
    ) -> ImageSolveResult:
        return ImageSolveResult(
            path=path,
            status="solved",
            message=result["message"],
            metadata_source="blind",
            duration_s=None,
            run_info=list(run_info),
        )

    def _run_index_near_solver(self, path: Path) -> Optional[ImageSolveResult]:
        """Attempt a metadata-assisted near solve using the Zeblind index.

        Uses the same internal routine as the Settings tab tester (no quads).
        Returns an ImageSolveResult on success, or None on failure.
        """
        try:
            index_root = self.config.blind_index_path
            if not index_root:
                return None
            # Use GUI FOV value as override for near solver when > 0
            fov_override = self.config.fov_deg if self.config.fov_deg and self.config.fov_deg > 0 else None
            family: Optional[str] = None
            if self.config.families and len(self.config.families) == 1:
                family = self.config.families[0]
            # If we have a previous near solution, reduce trials and pass parity hint
            seed = getattr(self, "_near_seed", None)
            ransac_trials = 600 if seed else 1200
            seed_scale = float(seed[0]) if seed else None
            seed_rot = float(seed[1]) if seed else None
            seed_par = int(seed[2]) if seed else 1
            if seed and self.config.near_warm_start:
                logging.info("[ZENEAR] warm-start used (trials=%d)", ransac_trials)
            # Further tighten search margin when OBJECT is present
            obj_name = None
            try:
                obj_name = (metadata.extra.get("OBJECT") if metadata and metadata.extra else None)
            except Exception:
                obj_name = None
            search_margin = 1.2
            if obj_name and metadata.ra_deg is not None and metadata.dec_deg is not None:
                search_margin = 1.05
                logging.info("[ZENEAR] OBJECT=%s -> tighter search margin (%.2f)", obj_name, search_margin)
            near_cfg = NearIndexConfig(
                fov_override_deg=fov_override,
                family=family,
                max_tile_candidates=int(getattr(self.config, "near_max_tile_candidates", 48) or 48),
                tile_cache_size=int(getattr(self.config, "near_tile_cache_size", 128) or 128),
                detect_backend=(getattr(self.config, "near_detect_backend", None) or "auto"),
                detect_device=(getattr(self.config, "near_detect_device", None)),
                ransac_trials=ransac_trials,
                seed_scale_deg=seed_scale,
                seed_rotation=seed_rot,
                seed_parity=seed_par,
                search_margin=search_margin,
            )
            result = near_solve(
                fits_path=str(path),
                index_root=str(index_root),
                config=near_cfg,
                log=logging.info,
                skip_if_valid=False,
                fallback_to_blind=False,
                cancel_check=(self._cancelled if self._cancel_event else None),
            )
        except BlindSolverRuntimeError as exc:
            logging.info("Near solver (index) failed for %s: %s", path.name, exc)
            return None
        if result["success"]:
            message = result.get("message") or "near solution found"
            # Update sequential seed from returned keywords when available
            try:
                kw = result.get("updated_keywords", {}) or {}
                s = float(kw.get("SEED_SCALE")) if kw.get("SEED_SCALE") is not None else None
                r = float(kw.get("SEED_ROT")) if kw.get("SEED_ROT") is not None else None
                p = int(kw.get("SEED_PAR")) if kw.get("SEED_PAR") is not None else 1
                if s and r is not None:
                    self._near_seed = (s, r, p)
            except Exception:
                pass
            return ImageSolveResult(
                path=path,
                status="solved",
                message=message,
                matched_stars=0,
                rms_arcsec=None,
                pixel_scale_arcsec=None,
                metadata_source="near-index",
            )
        return None

    def _try_blind_shortcut(
        self,
        path: Path,
        run_info: list[tuple[str, dict[str, Any]]],
    ) -> tuple[Optional[ImageSolveResult], Optional[BlindSolveResult]]:
        result = self._run_blind_solver(
            path,
            run_info,
            skip_if_header_has_wcs=True,
            skip_if_valid=self.config.blind_skip_if_valid,
        )
        if result is None or not result["success"]:
            return None, None
        if not self.config.overwrite:
            image = self._build_blind_result(path, result, run_info)
            image.duration_s = result["elapsed_sec"]
            return image, None
        return None, result

    def _resolve_with_blind_after_failure(
        self,
        *,
        path: Path,
        run_info: list[tuple[str, dict[str, Any]]],
        cached_result: Optional[BlindSolveResult],
        ra_hint: Optional[float],
        dec_hint: Optional[float],
        metadata: Optional[ImageMetadata],
        peaks: Optional[np.ndarray],
    ) -> Optional[ImageSolveResult]:
        if not self.config.overwrite:
            return None

        if cached_result and cached_result["success"]:
            return self._build_blind_result(path, cached_result, run_info)
        result = self._run_blind_solver(
            path,
            run_info,
            skip_if_header_has_wcs=False,
            skip_if_valid=False,
            ra_hint=ra_hint,
            dec_hint=dec_hint,
        )
        if result and result["success"]:
            return self._build_blind_result(path, result, run_info)
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
        # Propagate cancellation to the solver for cooperative early exit
        try:
            self.solver.set_cancel_event(cancel_event)
        except Exception:
            pass
        workers = max(1, self.config.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            it = iter(self.files)
            inflight: dict[concurrent.futures.Future[ImageSolveResult], Path] = {}
            # Prime up to worker count
            try:
                for _ in range(workers):
                    if cancel_event and cancel_event.is_set():
                        break
                    path = next(it)
                    inflight[pool.submit(self.solver.solve_path, path)] = path
            except StopIteration:
                pass
            try:
                while inflight:
                    future = next(concurrent.futures.as_completed(inflight))
                    path = inflight.pop(future)
                    try:
                        yield future.result()
                    except Exception as exc:
                        yield ImageSolveResult(path=path, status="failed", message=str(exc))
                    if cancel_event and cancel_event.is_set():
                        break
                    try:
                        path = next(it)
                    except StopIteration:
                        continue
                    inflight[pool.submit(self.solver.solve_path, path)] = path
            finally:
                if cancel_event and cancel_event.is_set():
                    for f in inflight.keys():
                        f.cancel()


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
        "--near-max-tile-candidates",
        type=int,
        default=48,
        help="Near-solver: cap intersecting tiles to this many (default: 48)",
    )
    parser.add_argument(
        "--near-tile-cache-size",
        type=int,
        default=128,
        help="Near-solver: in-memory tile cache size shared across runs (default: 128)",
    )
    parser.add_argument(
        "--near-detect-backend",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Near-solver: star detection device (auto|cpu|cuda)",
    )
    parser.add_argument(
        "--near-detect-device",
        type=int,
        help="Near-solver: CUDA device index to use for detection",
    )
    parser.add_argument(
        "--near-warm-start",
        dest="near_warm_start",
        action="store_true",
        default=None,
        help="Enable sequential warm-start for near solver",
    )
    parser.add_argument(
        "--no-near-warm-start",
        dest="near_warm_start",
        action="store_false",
        help="Disable sequential warm-start for near solver",
    )
    parser.add_argument(
        "--io-concurrency",
        type=int,
        default=0,
        help="Limit concurrent disk I/O operations (0 = auto based on workers)",
    )
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=0,
        help="Force a garbage collection every N results (0 = disabled)",
    )
    parser.add_argument(
        "--blind-index",
        type=Path,
        help="Path to the Zeblind index root (manifest + hash tables) used by the internal matcher",
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
    _configure_logging(args.log_level)
    formats = tuple(_parse_formats_value(args.formats))
    if not args.db_root or not args.input_dir:
        raise SystemExit("--db-root and --input-dir are required in CLI mode (use --gui to launch the GUI)")
    families = _normalize_family_args(args.family)
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
        blind_index_path=args.blind_index,
        near_max_tile_candidates=max(1, int(args.near_max_tile_candidates or 48)),
        near_tile_cache_size=max(1, int(args.near_tile_cache_size or 128)),
        near_detect_backend=str(args.near_detect_backend or "auto"),
        near_detect_device=(int(args.near_detect_device) if args.near_detect_device is not None else None),
        io_concurrency=int(args.io_concurrency or 0),
        gc_interval=int(args.gc_interval or 0),
        near_warm_start=(True if args.near_warm_start is None else bool(args.near_warm_start)),
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
    _configure_logging(args.log_level)
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("PySide6>=6 is required for the GUI. Install it and retry.") from exc
    prefill_families = _normalize_family_args(args.family)
    persistent_settings = load_persistent_settings()

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
            # Propagate cancel event for cooperative cancellation
            try:
                batch.solver.set_cancel_event(self._cancel_event)
            except Exception:
                pass
            self.started.emit(len(batch.files))
            self.info.emit(
                self._translate(
                    "runner_start",
                    files=len(batch.files),
                    workers=self.config.workers,
                )
            )
            try:
                import gc as _gc
                processed = 0
                for result in batch.run(cancel_event=self._cancel_event):
                    self.progress.emit(result)
                    processed += 1
                    # Optional periodic GC if requested
                    try:
                        interval = int(getattr(self.config, 'gc_interval', 0) or 0)
                    except Exception:
                        interval = 0
                    if interval > 0 and processed % interval == 0:
                        try:
                            _gc.collect()
                        except Exception:
                            pass
                    if self._cancel_event.is_set():
                        self.info.emit(self._translate("runner_stop_wait"))
                        break
            except Exception as exc:
                self.error.emit(str(exc))
            finally:
                self.finished.emit()

    class FileScanner(QtCore.QThread):
        file_found = QtCore.Signal(str)
        finished = QtCore.Signal(int)

        def __init__(self, root: Path, exts: list[str], limit: Optional[int]) -> None:
            super().__init__()
            self._root = Path(root)
            self._exts = [e.lower() for e in exts]
            self._limit = int(limit) if isinstance(limit, int) and limit > 0 else None
            self._cancel = threading.Event()

        def cancel(self) -> None:
            self._cancel.set()

        def run(self) -> None:
            count = 0
            try:
                for path in _iter_image_files(self._root, self._exts):
                    if self._cancel.is_set():
                        break
                    try:
                        self.file_found.emit(str(path))
                    except Exception:
                        pass
                    count += 1
                    if self._limit and count >= self._limit:
                        break
            except Exception:
                pass
            finally:
                try:
                    self.finished.emit(count)
                except Exception:
                    pass

    class IndexBuilder(QtCore.QThread):
        log = QtCore.Signal(str)
        progress = QtCore.Signal(int, int, str)
        finished = QtCore.Signal(bool, str)

        def __init__(
            self,
            db_root: str,
            index_root: str,
            *,
            mag_cap: float,
            max_stars: int,
            max_quads_per_tile: int,
            quads_only: bool = False,
        ) -> None:
            super().__init__()
            self.db_root = db_root
            self.index_root = index_root
            self.mag_cap = mag_cap
            self.max_stars = max_stars
            self.max_quads_per_tile = max_quads_per_tile
            self.quads_only = quads_only

        def run(self) -> None:
            # Bridge Python logging (INFO+) into the GUI log during the build
            class _ForwardLogHandler(logging.Handler):
                def __init__(self, sink):
                    super().__init__(level=logging.INFO)
                    self._sink = sink
                def emit(self, record: logging.LogRecord) -> None:
                    try:
                        msg = self.format(record)
                    except Exception:
                        msg = record.getMessage()
                    self._sink.emit(msg)

            handler = _ForwardLogHandler(self.log)
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            root_logger = logging.getLogger()
            # Attach only to the root logger to avoid duplicate propagation
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)
            try:
                # Optional quads-only mode (set by caller)
                quads_only = bool(getattr(self, "quads_only", False))
                self.log.emit(f"Index worker started (quads_only={quads_only})")
                logging.info("Index worker started (quads_only=%s)", quads_only)
                if not quads_only:
                    manifest = build_index_from_astap(
                        self.db_root,
                        self.index_root,
                        mag_cap=self.mag_cap,
                        max_stars=self.max_stars,
                        max_quads_per_tile=self.max_quads_per_tile,
                        skip_quads=True,
                    )
                    msg = f"Index built: {manifest}"
                    self.log.emit(msg)
                    logging.info(msg)
                else:
                    msg = "Manifest present; building quad tables only…"
                    self.log.emit(msg)
                    logging.info(msg)
                # Safety net: if quad tables are still missing, (re)build them now.
                try:
                    from zeblindsolver.levels import LEVEL_SPECS
                    from zeblindsolver.quad_index_builder import build_quad_index
                    idx_root = Path(self.index_root).expanduser().resolve()
                    ht = idx_root / "hash_tables"
                    ht.mkdir(parents=True, exist_ok=True)
                    self.log.emit(f"Checking quad tables under: {ht}")
                    logging.info("Checking quad tables under: %s", ht)
                    built_any = False
                    for level in LEVEL_SPECS:
                        out = ht / f"quads_{level.name}.npz"
                        if not out.exists():
                            msg = f"Building missing quad table: {out}"
                            self.log.emit(msg)
                            logging.info(msg)
                            def _cb(done: int, total: int, tile_key: str, lvl=level.name):
                                self.progress.emit(done, total, f"{lvl}:{tile_key}")
                            built = build_quad_index(
                                idx_root,
                                level.name,
                                max_quads_per_tile=self.max_quads_per_tile,
                                on_progress=_cb,
                                workers=max(1, (os.cpu_count() or 1) // 2),
                            )
                            done = f"Built quad table: {built}"
                            self.log.emit(done)
                            logging.info(done)
                            built_any = True
                        else:
                            self.log.emit(f"Quad table present: {out}")
                    # Final presence check
                    missing = [lvl.name for lvl in LEVEL_SPECS if not (ht / f"quads_{lvl.name}.npz").exists()]
                    if missing:
                        raise RuntimeError(f"quad tables missing after build: {', '.join(missing)}")
                    if not built_any:
                        msg = "Quad tables already present; skipped rebuild"
                        self.log.emit(msg)
                        logging.info(msg)
                except Exception as exc:
                    err = f"Quad table build failed: {exc}"
                    self.log.emit(err)
                    logging.error(err)
                    self.finished.emit(False, str(exc))
                    return
                self.finished.emit(True, str(Path(self.index_root).expanduser().resolve() / 'manifest.json'))
            except Exception as exc:
                self.log.emit(f"Index build failed: {exc}")
                self.finished.emit(False, str(exc))
            finally:
                try:
                    root_logger.removeHandler(handler)
                except Exception:
                    pass

    class BlindRunner(QtCore.QThread):
        log = QtCore.Signal(str)
        finished = QtCore.Signal(bool, str)

        def __init__(self, fits_path: str, index_root: str, blind_config: Optional[BlindSolveConfig] = None) -> None:
            super().__init__()
            self.fits_path = fits_path
            self.index_root = index_root
            self._config = blind_config

        def run(self) -> None:
            # Forward Python logging to GUI log during blind run (like index builder)
            class _ForwardLogHandler(logging.Handler):
                def __init__(self, sink):
                    super().__init__(level=logging.INFO)
                    self._sink = sink
                def emit(self, record: logging.LogRecord) -> None:
                    try:
                        msg = self.format(record)
                    except Exception:
                        msg = record.getMessage()
                    self._sink.emit(msg)
            handler = _ForwardLogHandler(self.log)
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            prev_level = root_logger.level
            root_logger.setLevel(logging.INFO)
            try:
                config = self._config or BlindSolveConfig()
                self.log.emit(f"Starting blind solve for {Path(self.fits_path).name}…")
                result = blind_solve(
                    self.fits_path,
                    self.index_root,
                    config=config,
                    log=self.log.emit,
                    skip_if_valid=False,
                )
                if result["success"]:
                    kw = result.get("updated_keywords", {}) or {}
                    rms = kw.get("RMS_PX") or kw.get("RMSPX")
                    inl = kw.get("N_INLIERS") or kw.get("INLIERS")
                    if isinstance(rms, (int, float)) and isinstance(inl, (int, float)):
                        msg = f"Blind solve succeeded (rms={float(rms):.2f} px, inliers={int(inl)})"
                    elif isinstance(rms, (int, float)):
                        msg = f"Blind solve succeeded (rms={float(rms):.2f} px)"
                    else:
                        msg = result.get("message") or "Blind solve succeeded"
                    self.log.emit(msg)
                    self.finished.emit(True, msg)
                else:
                    message = result.get("message") or "no valid solution"
                    self.log.emit(f"Blind solve failed: {message}")
                    self.finished.emit(False, message)
            except Exception as exc:
                self.log.emit(f"Blind solve error: {exc}")
                self.finished.emit(False, str(exc))
            finally:
                try:
                    root_logger.setLevel(prev_level)
                    root_logger.removeHandler(handler)
                except Exception:
                    pass

    class NearRunner(QtCore.QThread):
        log = QtCore.Signal(str)
        finished = QtCore.Signal(bool, str)

        def __init__(self, fits_path: str, index_root: str, *, max_tiles: int = 48, tile_cache: int = 128, detect_backend: str = "auto", detect_device: int | None = None) -> None:
            super().__init__()
            self.fits_path = fits_path
            self.index_root = index_root
            self.max_tiles = int(max_tiles)
            self.tile_cache = int(tile_cache)
            self.detect_backend = str(detect_backend or "auto")
            self.detect_device = detect_device

        def run(self) -> None:
            try:
                cfg = NearIndexConfig(
                    max_tile_candidates=max(1, self.max_tiles),
                    tile_cache_size=max(1, self.tile_cache),
                    detect_backend=self.detect_backend,
                    detect_device=self.detect_device,
                )
                result = near_solve(
                    self.fits_path,
                    self.index_root,
                    config=cfg,
                    skip_if_valid=False,
                    fallback_to_blind=False,
                    log=logging.info,
                )
                if result["success"]:
                    message = result["message"] or "near solve succeeded"
                    self.log.emit(f"Near solve succeeded: {message}")
                    self.finished.emit(True, message)
                else:
                    message = result["message"] or "no valid solution"
                    self.log.emit(f"Near solve failed: {message}")
                    self.finished.emit(False, message)
            except Exception as exc:
                self.log.emit(f"Near solve error: {exc}")
                self.finished.emit(False, str(exc))

    class ZeSolverWindow(QtWidgets.QMainWindow):
        def __init__(self, settings: PersistentSettings) -> None:
            super().__init__()
            self._language = GUI_DEFAULT_LANGUAGE
            self.resize(1280, 760)
            self._worker: Optional[SolveRunner] = None
            self._pending_files: List[Path] = []
            self._item_by_path: dict[Path, QtWidgets.QTreeWidgetItem] = {}
            self._current_input_dir: Optional[Path] = None
            self._results_seen = 0
            self._language_actions: dict[str, QtGui.QAction] = {}
            self._settings = settings
            self._index_worker: Optional[IndexBuilder] = None
            self._blind_worker: Optional[BlindRunner] = None
            self._near_worker: Optional[NearRunner] = None
            self._scanner: Optional[FileScanner] = None
            self._scan_buffer: list[Path] = []
            self._scan_flush_threshold = 250
            self._build_ui()
            self._populate_settings_ui()
            self._prefill_from_args(args)
            self._apply_language()

        # --- UI building helpers -------------------------------------------------
        def _build_ui(self) -> None:
            self._build_menu_bar()
            central = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central)
            layout.addLayout(self._build_paths_row())
            self.tabs = QtWidgets.QTabWidget()
            solver_tab = QtWidgets.QWidget()
            solver_layout = QtWidgets.QVBoxLayout(solver_tab)
            solver_layout.addWidget(self._build_options_box())
            solver_layout.addWidget(self._build_splitter())
            solver_layout.addLayout(self._build_bottom_row())
            self.tabs.addTab(solver_tab, self._text("solver_tab"))
            # Database tab (db_root selection and future download UI)
            self.database_tab = self._build_database_tab()
            self.tabs.addTab(self.database_tab, self._text("database_tab"))
            self.settings_tab = self._build_settings_tab()
            self.tabs.addTab(self.settings_tab, self._text("settings_tab"))
            # Add Performance tab for near-solver tuning
            self.performance_tab = self._build_performance_tab()
            self.tabs.addTab(self.performance_tab, self._text("performance_tab"))
            layout.addWidget(self.tabs)
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
            self.input_label = QtWidgets.QLabel()
            self.input_edit = QtWidgets.QLineEdit()
            self.browse_in_btn = QtWidgets.QPushButton()
            self.browse_in_btn.clicked.connect(
                lambda: self._pick_directory(self.input_edit, trigger_scan=True)
            )
            grid.addWidget(self.input_label, 0, 0)
            grid.addWidget(self.input_edit, 0, 1)
            grid.addWidget(self.browse_in_btn, 0, 2)
            self.scan_btn = QtWidgets.QPushButton()
            self.scan_btn.clicked.connect(self.scan_files)
            grid.addWidget(self.scan_btn, 0, 3)
            return grid

        def _build_database_tab(self) -> QtWidgets.QWidget:
            widget = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(widget)
            form = QtWidgets.QFormLayout()
            # Database root selector (kept in sync with Settings tab)
            self.db_tab_label = QtWidgets.QLabel()
            self.db_tab_edit = QtWidgets.QLineEdit(self._settings.db_root or "")
            self.db_tab_browse = QtWidgets.QPushButton()
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(self.db_tab_edit)
            row_layout.addWidget(self.db_tab_browse)
            form.addRow(self.db_tab_label, row)

            # Data sources (preset URLs)
            self.data_sources_label = QtWidgets.QLabel()
            # Preset list of HNSKY/ASTAP shards (placeholders)
            self.sources_group = QtWidgets.QGroupBox(self._text("data_sources"))
            sources_layout = QtWidgets.QVBoxLayout(self.sources_group)
            self.sources_tree = QtWidgets.QTreeWidget()
            self.sources_tree.setHeaderLabels(["Package", "Size"])
            self.sources_tree.header().setStretchLastSection(True)
            # Placeholder sources (replace with official URLs later)
            preset_sources = [
                {"label": "HNSKY/ASTAP Stars Shard A (placeholder)", "url": "https://example.invalid/astap/stars_A.1476", "size": 120_000_000},
                {"label": "HNSKY/ASTAP Stars Shard B (placeholder)", "url": "https://example.invalid/astap/stars_B.1476", "size": 120_000_000},
                {"label": "HNSKY/ASTAP Stars Shard C (placeholder)", "url": "https://example.invalid/astap/stars_C.1476", "size": 120_000_000},
            ]
            for entry in preset_sources:
                item = QtWidgets.QTreeWidgetItem([entry["label"], f"{entry['size']/1_000_000:.1f} MB"])
                # Store metadata on column 0
                item.setData(0, QtCore.Qt.UserRole, entry)
                item.setCheckState(0, QtCore.Qt.Unchecked)
                self.sources_tree.addTopLevelItem(item)
            sources_layout.addWidget(self.sources_tree)
            # Add-selected button
            self.btn_add_selected = QtWidgets.QPushButton(self._text("add_selected"))
            sources_layout.addWidget(self.btn_add_selected)
            form.addRow(self.sources_group)

            # Downloads group
            self.downloads_group = QtWidgets.QGroupBox()
            self.downloads_group.setTitle(self._text("downloads_title") if hasattr(self, "_text") else "Downloads")
            dl_layout = QtWidgets.QVBoxLayout(self.downloads_group)
            # URLs input
            self.urls_edit = QtWidgets.QPlainTextEdit()
            self.urls_edit.setPlaceholderText("https://… one URL per line")
            dl_layout.addWidget(self.urls_edit)
            # Buttons row
            btn_row = QtWidgets.QHBoxLayout()
            self.btn_add_to_queue = QtWidgets.QPushButton(self._text("add_to_queue") if hasattr(self, "_text") else "Add to queue")
            self.btn_start_all = QtWidgets.QPushButton(self._text("start_all") if hasattr(self, "_text") else "Start all")
            self.btn_pause_all = QtWidgets.QPushButton(self._text("pause_all") if hasattr(self, "_text") else "Pause all")
            self.btn_verify_hashes = QtWidgets.QPushButton(self._text("verify_hashes") if hasattr(self, "_text") else "Verify hashes")
            btn_row.addWidget(self.btn_add_to_queue)
            btn_row.addWidget(self.btn_start_all)
            btn_row.addWidget(self.btn_pause_all)
            btn_row.addWidget(self.btn_verify_hashes)
            btn_row.addStretch(1)
            dl_layout.addLayout(btn_row)
            # Table
            self.downloads_table = QtWidgets.QTableWidget(0, 4)
            self.downloads_table.setHorizontalHeaderLabels(["File", "Size", "Progress", "Status"])
            self.downloads_table.horizontalHeader().setStretchLastSection(True)
            dl_layout.addWidget(self.downloads_table)
            column.addLayout(form)
            column.addWidget(self.downloads_group)

            # Manager and worker
            try:
                from zeblindsolver.downloads import DownloadsManager
                self._downloads_manager = DownloadsManager()
            except Exception:
                self._downloads_manager = None

            class DownloadRunner(QtCore.QThread):
                progress = QtCore.Signal(dict)
                finished = QtCore.Signal()

                def __init__(self, manager, parent=None):
                    super().__init__(parent)
                    self._manager = manager
                    self._stop = threading.Event()

                def stop(self):
                    self._stop.set()

                def run(self):  # pragma: no cover - GUI thread
                    if self._manager is None:
                        return
                    def _emit(item):
                        payload = {
                            "id": item.id,
                            "path": str(item.dest_path),
                            "status": item.status,
                            "done": int(item.bytes_done or 0),
                            "total": int(item.bytes_total or 0) if item.bytes_total else None,
                            "error": item.error,
                        }
                        self.progress.emit(payload)
                    self._manager.run_all(stop_event=self._stop, on_update=_emit)
                    self.finished.emit()

            self._dl_worker = None

            def _append_row_for_item(item) -> None:
                row_idx = self.downloads_table.rowCount()
                self.downloads_table.insertRow(row_idx)
                self.downloads_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(item.dest_path.name))
                size_text = f"{(item.size_hint or 0)/1_000_000:.1f} MB" if item.size_hint else "?"
                self.downloads_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(size_text))
                self.downloads_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem("0%"))
                self.downloads_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(item.status))
                self.downloads_table.scrollToBottom()

            def _update_row(payload: dict) -> None:
                # Find row by filename
                try:
                    name = Path(payload.get("path") or "").name
                    for r in range(self.downloads_table.rowCount()):
                        if self.downloads_table.item(r, 0).text() == name:
                            done = payload.get("done") or 0
                            total = payload.get("total") or 0
                            pct = 0 if not total else int(100 * done / max(1, total))
                            self.downloads_table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{pct}%"))
                            self.downloads_table.setItem(r, 3, QtWidgets.QTableWidgetItem(payload.get("status") or "?"))
                            break
                except Exception:
                    pass

            def _add_to_queue_clicked() -> None:
                try:
                    if not self._downloads_manager:
                        return
                    root = self.db_tab_edit.text().strip() or (self._settings.db_root or "")
                    if not root:
                        QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("error_database_required"))
                        return
                    root_path = Path(root).expanduser()
                    lines = [ln.strip() for ln in self.urls_edit.toPlainText().splitlines()]
                    for url in lines:
                        if not url:
                            continue
                        item = self._downloads_manager.add(url, root_path)
                        _append_row_for_item(item)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))

            def _add_selected_clicked() -> None:
                try:
                    if not self._downloads_manager:
                        return
                    root = self.db_tab_edit.text().strip() or (self._settings.db_root or "")
                    if not root:
                        QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("error_database_required"))
                        return
                    root_path = Path(root).expanduser()
                    for i in range(self.sources_tree.topLevelItemCount()):
                        it = self.sources_tree.topLevelItem(i)
                        if it.checkState(0) == QtCore.Qt.Checked:
                            entry = it.data(0, QtCore.Qt.UserRole)
                            if isinstance(entry, dict):
                                url = entry.get("url") or ""
                                size = int(entry.get("size") or 0)
                                if url:
                                    item = self._downloads_manager.add(url, root_path, size_hint=size)
                                    _append_row_for_item(item)
                                    it.setCheckState(0, QtCore.Qt.Unchecked)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))

            def _start_all_clicked() -> None:
                try:
                    if not self._downloads_manager or self._dl_worker:
                        return
                    self._dl_worker = DownloadRunner(self._downloads_manager)
                    self._dl_worker.progress.connect(_update_row)
                    self._dl_worker.finished.connect(lambda: setattr(self, "_dl_worker", None))
                    self._dl_worker.start()
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))

            def _pause_all_clicked() -> None:
                try:
                    if self._dl_worker:
                        self._dl_worker.stop()
                except Exception:
                    pass

            def _verify_hashes_clicked() -> None:
                try:
                    if not self._downloads_manager:
                        return
                    def _emit(item):
                        _update_row({
                            "path": str(item.dest_path),
                            "status": item.status,
                            "done": item.bytes_done,
                            "total": item.bytes_total,
                        })
                    self._downloads_manager.verify_all(on_update=_emit)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))

            self.btn_add_to_queue.clicked.connect(_add_to_queue_clicked)
            self.btn_add_selected.clicked.connect(_add_selected_clicked)
            self.btn_start_all.clicked.connect(_start_all_clicked)
            self.btn_pause_all.clicked.connect(_pause_all_clicked)
            self.btn_verify_hashes.clicked.connect(_verify_hashes_clicked)

            column.addLayout(form)
            self.db_tab_browse.clicked.connect(lambda: self._pick_settings_directory(self.db_tab_edit))
            # Keep Settings tab DB field in sync
            def _sync_db_text(text: str) -> None:
                try:
                    if hasattr(self, "settings_db_edit"):
                        if self.settings_db_edit.text().strip() != text.strip():
                            self.settings_db_edit.setText(text)
                except Exception:
                    pass
            self.db_tab_edit.textChanged.connect(_sync_db_text)
            return widget

        def _build_options_box(self) -> QtWidgets.QGroupBox:
            self.options_box = QtWidgets.QGroupBox()
            form = QtWidgets.QFormLayout(self.options_box)
            self.fov_spin = QtWidgets.QDoubleSpinBox()
            self.fov_spin.setRange(0.0, 20.0)
            self.fov_spin.setDecimals(2)
            self.fov_spin.setValue(self._settings.solver_fov_deg or args.fov_deg or DEFAULT_FOV_DEG)
            # Treat 0.0 as "Auto" in the UI
            self.fov_spin.setSpecialValueText(GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["special_auto"])
            self.search_scale_spin = QtWidgets.QDoubleSpinBox()
            self.search_scale_spin.setRange(1.0, 10.0)
            self.search_scale_spin.setDecimals(2)
            self.search_scale_spin.setSingleStep(0.1)
            self.search_scale_spin.setValue(self._settings.solver_search_scale or args.search_radius_scale or DEFAULT_SEARCH_RADIUS_SCALE)
            self.search_attempts_spin = QtWidgets.QSpinBox()
            self.search_attempts_spin.setRange(1, 10)
            self.search_attempts_spin.setValue(self._settings.solver_search_attempts or args.search_radius_attempts or DEFAULT_SEARCH_RADIUS_ATTEMPTS)
            self.max_radius_spin = QtWidgets.QDoubleSpinBox()
            self.max_radius_spin.setRange(0.0, 30.0)
            self.max_radius_spin.setDecimals(2)
            self.max_radius_spin.setSingleStep(0.1)
            self.max_radius_spin.setSpecialValueText(GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["special_auto"])
            max_radius = self._settings.solver_max_radius_deg if self._settings.solver_max_radius_deg is not None else 0.0
            self.max_radius_spin.setValue(max_radius)
            self.downsample_spin = QtWidgets.QSpinBox()
            self.downsample_spin.setRange(1, 4)
            self.downsample_spin.setValue(self._settings.solver_downsample or args.downsample or 1)
            self.workers_spin = QtWidgets.QSpinBox()
            self.workers_spin.setRange(1, max(32, _default_worker_count()))
            self.workers_spin.setValue(self._settings.solver_workers or args.workers or _default_worker_count())
            self.cache_spin = QtWidgets.QSpinBox()
            self.cache_spin.setRange(2, 64)
            self.cache_spin.setValue(self._settings.solver_cache_size or args.cache_size or 12)
            self.max_files_spin = QtWidgets.QSpinBox()
            self.max_files_spin.setRange(0, 10000)
            self.max_files_spin.setValue(self._settings.solver_max_files or args.max_files or 0)
            formats_text = self._settings.solver_formats or args.formats or ",".join(sorted(SUPPORTED_EXTENSIONS))
            self.formats_edit = QtWidgets.QLineEdit(formats_text)
            # Catalog family selection (populated from index manifest). First item = Auto
            self.families_combo = QtWidgets.QComboBox()
            self.families_combo.setEditable(False)
            self.families_combo.addItem(GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["special_auto"], "")
            self.overwrite_check = QtWidgets.QCheckBox()
            self.overwrite_check.setChecked(bool(self._settings.solver_overwrite))
            self.blind_check = QtWidgets.QCheckBox()
            self.blind_check.setChecked(bool(self._settings.solver_blind_enabled))
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
            form.addRow(self.families_label_widget, self.families_combo)
            form.addRow(self.blind_check)
            form.addRow(self.overwrite_check)
            return self.options_box

        def _populate_families_from_index(self, index_root_text: str) -> None:
            """Populate the catalog family dropdown from the index manifest.

            Adds an 'Auto' entry first; subsequent entries are sorted unique family keys.
            """
            try:
                root = Path(index_root_text).expanduser().resolve()
            except Exception:
                return
            manifest = root / "manifest.json"
            if not manifest.is_file():
                return
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                return
            tiles = payload.get("tiles") or []
            families: set[str] = set()
            for entry in tiles:
                name = str(entry.get("family", "")).strip().lower()
                if name:
                    families.add(name)
            items = sorted(families)
            # Preserve current selection key if possible
            current = self.families_combo.currentData() if hasattr(self, 'families_combo') else ""
            self.families_combo.blockSignals(True)
            self.families_combo.clear()
            self.families_combo.addItem(GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["special_auto"], "")
            for key in items:
                self.families_combo.addItem(key, key)
            # Restore previous selection if present
            if current and current in items:
                idx = self.families_combo.findData(current)
                if idx >= 0:
                    self.families_combo.setCurrentIndex(idx)
            self.families_combo.blockSignals(False)

        def _build_settings_tab(self) -> QtWidgets.QWidget:
            widget = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(widget)
            form = QtWidgets.QFormLayout()

            self.settings_db_label = QtWidgets.QLabel()
            self.settings_db_edit = QtWidgets.QLineEdit(self._settings.db_root or "")
            self.settings_db_browse = QtWidgets.QPushButton()
            db_row = QtWidgets.QWidget()
            db_layout = QtWidgets.QHBoxLayout(db_row)
            db_layout.setContentsMargins(0, 0, 0, 0)
            db_layout.addWidget(self.settings_db_edit)
            db_layout.addWidget(self.settings_db_browse)
            form.addRow(self.settings_db_label, db_row)

            self.settings_index_label = QtWidgets.QLabel()
            self.settings_index_edit = QtWidgets.QLineEdit(self._settings.index_root or "")
            self.settings_index_browse = QtWidgets.QPushButton()
            index_row = QtWidgets.QWidget()
            index_layout = QtWidgets.QHBoxLayout(index_row)
            index_layout.setContentsMargins(0, 0, 0, 0)
            index_layout.addWidget(self.settings_index_edit)
            index_layout.addWidget(self.settings_index_browse)
            form.addRow(self.settings_index_label, index_row)

            self.settings_mag_label = QtWidgets.QLabel()
            self.settings_mag_spin = QtWidgets.QDoubleSpinBox()
            self.settings_mag_spin.setRange(0.0, 20.0)
            self.settings_mag_spin.setDecimals(2)
            self.settings_mag_spin.setValue(self._settings.mag_cap)
            form.addRow(self.settings_mag_label, self.settings_mag_spin)

            self.settings_max_stars_label = QtWidgets.QLabel()
            self.settings_max_stars_spin = QtWidgets.QSpinBox()
            self.settings_max_stars_spin.setRange(100, 10000)
            self.settings_max_stars_spin.setValue(self._settings.max_stars)
            form.addRow(self.settings_max_stars_label, self.settings_max_stars_spin)

            self.settings_max_quads_label = QtWidgets.QLabel()
            self.settings_max_quads_spin = QtWidgets.QSpinBox()
            self.settings_max_quads_spin.setRange(100, 100000)
            self.settings_max_quads_spin.setValue(self._settings.max_quads_per_tile)
            form.addRow(self.settings_max_quads_label, self.settings_max_quads_spin)

            self.settings_sample_label = QtWidgets.QLabel()
            self.settings_sample_edit = QtWidgets.QLineEdit(self._settings.sample_fits or "")
            self.settings_sample_browse = QtWidgets.QPushButton()
            sample_row = QtWidgets.QWidget()
            sample_layout = QtWidgets.QHBoxLayout(sample_row)
            sample_layout.setContentsMargins(0, 0, 0, 0)
            sample_layout.addWidget(self.settings_sample_edit)
            sample_layout.addWidget(self.settings_sample_browse)
            form.addRow(self.settings_sample_label, sample_row)

            # Presets group
            self.presets_group = QtWidgets.QGroupBox()
            self.presets_group.setTitle(self._text("presets_title"))
            presets_layout = QtWidgets.QVBoxLayout(self.presets_group)
            self.presets_combo = QtWidgets.QComboBox()
            for p in preset_utils.list_presets():
                self.presets_combo.addItem(p.label, p.id)
            self.preset_warning_label = QtWidgets.QLabel()
            self.preset_warning_label.setStyleSheet("color: #c08000;")
            presets_layout.addWidget(self.presets_combo)
            presets_layout.addWidget(self.preset_warning_label)

            # FOV group
            self.fov_group = QtWidgets.QGroupBox()
            self.fov_group.setTitle(self._text("fov_mode_title"))
            fov_form = QtWidgets.QFormLayout(self.fov_group)
            self.fov_focal_spin = QtWidgets.QDoubleSpinBox()
            self.fov_focal_spin.setRange(10.0, 6000.0)
            self.fov_focal_spin.setDecimals(1)
            self.fov_pixel_spin = QtWidgets.QDoubleSpinBox()
            self.fov_pixel_spin.setRange(1.0, 20.0)
            self.fov_pixel_spin.setDecimals(2)
            self.fov_res_w_spin = QtWidgets.QSpinBox()
            self.fov_res_w_spin.setRange(64, 20000)
            self.fov_res_h_spin = QtWidgets.QSpinBox()
            self.fov_res_h_spin.setRange(64, 20000)
            self.fov_reducer_spin = QtWidgets.QDoubleSpinBox()
            self.fov_reducer_spin.setRange(0.2, 2.0)
            self.fov_reducer_spin.setDecimals(2)
            self.fov_reducer_spin.setSingleStep(0.01)
            self.fov_binning_spin = QtWidgets.QSpinBox()
            self.fov_binning_spin.setRange(1, 8)
            # Labels for results
            self.reco_group = QtWidgets.QGroupBox()
            self.reco_group.setTitle(self._text("recommendations_title"))
            reco_form = QtWidgets.QFormLayout(self.reco_group)
            self.reco_scale_label = QtWidgets.QLabel()
            self.reco_scale_value = QtWidgets.QLabel("-")
            self.reco_fov_label = QtWidgets.QLabel()
            self.reco_fov_value = QtWidgets.QLabel("-")
            self.reco_mag_label = QtWidgets.QLabel()
            self.reco_mag_value = QtWidgets.QLabel("-")
            self.reco_quads_label = QtWidgets.QLabel()
            self.reco_quads_value = QtWidgets.QLabel("-")
            self.reco_notes_label = QtWidgets.QLabel("")
            self.reco_notes_label.setWordWrap(True)
            self.compute_button = QtWidgets.QPushButton()

            fov_form.addRow(QtWidgets.QLabel(self._text("focal_length_mm")), self.fov_focal_spin)
            fov_form.addRow(QtWidgets.QLabel(self._text("pixel_size_um")), self.fov_pixel_spin)
            # Resolution as two widgets in one row
            res_row = QtWidgets.QWidget()
            res_layout = QtWidgets.QHBoxLayout(res_row)
            res_layout.setContentsMargins(0, 0, 0, 0)
            res_layout.addWidget(self.fov_res_w_spin)
            res_layout.addWidget(self.fov_res_h_spin)
            fov_form.addRow(QtWidgets.QLabel(self._text("resolution_px")), res_row)
            fov_form.addRow(QtWidgets.QLabel(self._text("reducer_factor")), self.fov_reducer_spin)
            fov_form.addRow(QtWidgets.QLabel(self._text("binning")), self.fov_binning_spin)
            fov_form.addRow(self.compute_button)

            reco_form.addRow(self.reco_scale_label, self.reco_scale_value)
            reco_form.addRow(self.reco_fov_label, self.reco_fov_value)
            reco_form.addRow(self.reco_mag_label, self.reco_mag_value)
            reco_form.addRow(self.reco_quads_label, self.reco_quads_value)
            reco_form.addRow(self.reco_notes_label)

            column.addLayout(form)
            column.addWidget(self.presets_group)
            column.addWidget(self.fov_group)
            column.addWidget(self.reco_group)

            # Blind solver tuning group
            self.blind_group = QtWidgets.QGroupBox()
            self.blind_group.setTitle(self._text("settings_blind_group"))
            blind_form = QtWidgets.QFormLayout(self.blind_group)
            self.settings_blind_max_stars_label = QtWidgets.QLabel()
            self.settings_blind_max_stars_spin = QtWidgets.QSpinBox()
            self.settings_blind_max_stars_spin.setRange(100, 5000)
            self.settings_blind_max_stars_spin.setValue(self._settings.blind_max_stars)
            blind_form.addRow(self.settings_blind_max_stars_label, self.settings_blind_max_stars_spin)

            self.settings_blind_max_quads_label = QtWidgets.QLabel()
            self.settings_blind_max_quads_spin = QtWidgets.QSpinBox()
            self.settings_blind_max_quads_spin.setRange(500, 100000)
            self.settings_blind_max_quads_spin.setSingleStep(500)
            self.settings_blind_max_quads_spin.setValue(self._settings.blind_max_quads)
            blind_form.addRow(self.settings_blind_max_quads_label, self.settings_blind_max_quads_spin)

            self.settings_blind_max_candidates_label = QtWidgets.QLabel()
            self.settings_blind_max_candidates_spin = QtWidgets.QSpinBox()
            self.settings_blind_max_candidates_spin.setRange(4, 64)
            self.settings_blind_max_candidates_spin.setValue(self._settings.blind_max_candidates)
            blind_form.addRow(self.settings_blind_max_candidates_label, self.settings_blind_max_candidates_spin)

            self.settings_blind_pixel_tol_label = QtWidgets.QLabel()
            self.settings_blind_pixel_tol_spin = QtWidgets.QDoubleSpinBox()
            self.settings_blind_pixel_tol_spin.setRange(0.5, 10.0)
            self.settings_blind_pixel_tol_spin.setDecimals(1)
            self.settings_blind_pixel_tol_spin.setSingleStep(0.5)
            self.settings_blind_pixel_tol_spin.setValue(self._settings.blind_pixel_tolerance)
            blind_form.addRow(self.settings_blind_pixel_tol_label, self.settings_blind_pixel_tol_spin)

            self.settings_blind_quality_inliers_label = QtWidgets.QLabel()
            self.settings_blind_quality_inliers_spin = QtWidgets.QSpinBox()
            self.settings_blind_quality_inliers_spin.setRange(4, 200)
            self.settings_blind_quality_inliers_spin.setValue(self._settings.blind_quality_inliers)
            blind_form.addRow(self.settings_blind_quality_inliers_label, self.settings_blind_quality_inliers_spin)

            self.settings_blind_quality_rms_label = QtWidgets.QLabel()
            self.settings_blind_quality_rms_spin = QtWidgets.QDoubleSpinBox()
            self.settings_blind_quality_rms_spin.setRange(0.2, 5.0)
            self.settings_blind_quality_rms_spin.setDecimals(2)
            self.settings_blind_quality_rms_spin.setSingleStep(0.1)
            self.settings_blind_quality_rms_spin.setValue(self._settings.blind_quality_rms)
            blind_form.addRow(self.settings_blind_quality_rms_label, self.settings_blind_quality_rms_spin)
            # Fast mode (S-only then fallback)
            self.settings_blind_fast_check = QtWidgets.QCheckBox()
            self.settings_blind_fast_check.setChecked(self._settings.blind_fast_mode)
            blind_form.addRow(QtWidgets.QLabel(self._text("settings_blind_fast_label")), self.settings_blind_fast_check)

            button_row = QtWidgets.QHBoxLayout()
            self.settings_save_btn = QtWidgets.QPushButton()
            self.settings_save_btn.clicked.connect(self._on_save_settings_clicked)
            self.settings_build_btn = QtWidgets.QPushButton()
            self.settings_build_btn.clicked.connect(self._on_build_index_clicked)
            self.settings_run_blind_btn = QtWidgets.QPushButton()
            self.settings_run_blind_btn.clicked.connect(self._on_run_blind_clicked)
            self.settings_run_near_btn = QtWidgets.QPushButton()
            self.settings_run_near_btn.clicked.connect(self._on_run_near_clicked)
            button_row.addWidget(self.settings_save_btn)
            button_row.addWidget(self.settings_build_btn)
            button_row.addWidget(self.settings_run_blind_btn)
            button_row.addWidget(self.settings_run_near_btn)

            self.settings_log_view = QtWidgets.QPlainTextEdit()
            self.settings_log_view.setReadOnly(True)
            # Cap log memory growth: keep at most 2000 lines
            try:
                self.settings_log_view.document().setMaximumBlockCount(2000)
            except Exception:
                pass
            column.addLayout(form)
            column.addWidget(self.blind_group)
            column.addLayout(button_row)
            self.settings_log_label = QtWidgets.QLabel()
            column.addWidget(self.settings_log_label)
            # Progress bar for index/quads build
            self.settings_progress = QtWidgets.QProgressBar()
            self.settings_progress.setRange(0, 1)
            self.settings_progress.setValue(0)
            self.settings_progress.setFormat("Idle")
            column.addWidget(self.settings_progress)
            column.addWidget(self.settings_log_view)

            self.settings_db_browse.clicked.connect(
                lambda: self._pick_settings_directory(self.settings_db_edit)
            )
            # Keep Database tab DB field in sync when editing in Settings
            def _sync_db_tab_text(text: str) -> None:
                try:
                    if hasattr(self, "db_tab_edit"):
                        if self.db_tab_edit.text().strip() != text.strip():
                            self.db_tab_edit.setText(text)
                except Exception:
                    pass
            self.settings_db_edit.textChanged.connect(_sync_db_tab_text)
            self.settings_index_browse.clicked.connect(
                lambda: self._pick_settings_directory(self.settings_index_edit)
            )
            self.settings_sample_browse.clicked.connect(self._pick_settings_sample)
            # Presets interactions: load preset populates FOV fields
            def _apply_preset(preset_id: str) -> None:
                try:
                    presets = {p.id: p for p in preset_utils.list_presets()}
                    p = presets.get(preset_id)
                    if not p:
                        return
                    self.fov_focal_spin.setValue(p.focal_mm)
                    self.fov_pixel_spin.setValue(p.pixel_um)
                    self.fov_res_w_spin.setValue(p.res_w)
                    self.fov_res_h_spin.setValue(p.res_h)
                    self.fov_reducer_spin.setValue(p.reducer)
                    self.fov_binning_spin.setValue(1)
                    # Warning banner for approximate presets
                    if p.spec_confidence in ("approx", "unknown"):
                        self.preset_warning_label.setText(self._text("spec_warning_unknown"))
                    else:
                        self.preset_warning_label.setText("")
                except Exception:
                    pass

            self.presets_combo.currentIndexChanged.connect(
                lambda idx: _apply_preset(self.presets_combo.itemData(idx))
            )

            self.compute_button.clicked.connect(self._on_compute_fov_clicked)
            return widget

        def _build_performance_tab(self) -> QtWidgets.QWidget:
            widget = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(widget)
            form = QtWidgets.QFormLayout()
            # Near tile cache size
            self.perf_near_cache_label = QtWidgets.QLabel()
            self.perf_near_cache_spin = QtWidgets.QSpinBox()
            self.perf_near_cache_spin.setRange(16, 4096)
            self.perf_near_cache_spin.setSingleStep(16)
            self.perf_near_cache_spin.setValue(int(self._settings.near_tile_cache_size or 128))
            form.addRow(self.perf_near_cache_label, self.perf_near_cache_spin)
            # Near max tile candidates
            self.perf_near_max_tiles_label = QtWidgets.QLabel()
            self.perf_near_max_tiles_spin = QtWidgets.QSpinBox()
            self.perf_near_max_tiles_spin.setRange(4, 256)
            self.perf_near_max_tiles_spin.setValue(int(self._settings.near_max_tile_candidates or 48))
            form.addRow(self.perf_near_max_tiles_label, self.perf_near_max_tiles_spin)
            # Star detection device selection (CPU / CUDA GPUs)
            self.perf_detect_label = QtWidgets.QLabel()
            self.perf_detect_combo = QtWidgets.QComboBox()
            self._populate_detect_devices(self.perf_detect_combo)
            form.addRow(self.perf_detect_label, self.perf_detect_combo)
            # I/O concurrency (0 = Auto)
            self.perf_io_label = QtWidgets.QLabel()
            self.perf_io_spin = QtWidgets.QSpinBox()
            self.perf_io_spin.setRange(0, max(64, _default_worker_count()))
            self.perf_io_spin.setValue(int(getattr(self._settings, 'io_concurrency', 0) or 0))
            self.perf_io_spin.setSpecialValueText(GUI_TRANSLATIONS[GUI_DEFAULT_LANGUAGE]["special_auto"])
            form.addRow(self.perf_io_label, self.perf_io_spin)
            # Near warm-start toggle
            self.perf_near_warm_check = QtWidgets.QCheckBox()
            self.perf_near_warm_check.setChecked(bool(getattr(self._settings, 'near_warm_start', True)))
            form.addRow(QtWidgets.QLabel(self._text("settings_perf_near_warm_label")), self.perf_near_warm_check)
            # Save button
            self.performance_save_btn = QtWidgets.QPushButton()
            self.performance_save_btn.clicked.connect(self._on_save_settings_clicked)
            btn_row = QtWidgets.QHBoxLayout()
            btn_row.addStretch(1)
            btn_row.addWidget(self.performance_save_btn)
            column.addLayout(form)
            column.addLayout(btn_row)
            return widget

        def _populate_detect_devices(self, combo: 'QtWidgets.QComboBox') -> None:
            combo.clear()
            # Always offer CPU
            combo.addItem("CPU", ("cpu", -1))
            # Try CUDA via CuPy
            try:
                import cupy  # type: ignore
                from cupy.cuda import runtime as _rt  # type: ignore
                n = int(_rt.getDeviceCount())
                for i in range(n):
                    props = _rt.getDeviceProperties(i)
                    name = props.get('name') if isinstance(props, dict) else None
                    if isinstance(name, (bytes, bytearray)):
                        name = name.decode(errors='ignore')
                    label = f"CUDA: {name or 'GPU'} (id {i})"
                    combo.addItem(label, ("cuda", i))
            except Exception:
                pass
            # Restore previous selection if possible, otherwise default to first CUDA if present
            backend = (self._settings.near_detect_backend or "auto").lower()
            device = int(self._settings.near_detect_device or 0)
            target_data = None
            if backend == "cuda":
                target_data = ("cuda", device)
            elif backend == "cpu":
                target_data = ("cpu", -1)
            # Try to match stored selection
            if target_data is not None:
                for idx in range(combo.count()):
                    if combo.itemData(idx) == target_data:
                        combo.setCurrentIndex(idx)
                        break
            else:
                # Auto: prefer first CUDA if available
                for idx in range(combo.count()):
                    data = combo.itemData(idx)
                    if isinstance(data, tuple) and data[0] == "cuda":
                        combo.setCurrentIndex(idx)
                        break

        def _pick_settings_directory(self, target: QtWidgets.QLineEdit) -> None:
            opts = QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontUseNativeDialog
            try:
                opts |= QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons
            except Exception:
                pass
            try:
                opts |= QtWidgets.QFileDialog.Option.DontResolveSymlinks
            except Exception:
                pass
            start = target.text().strip() or str(Path.home())
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self._text("dialog_select_directory"),
                start,
                options=opts,
            )
            if directory:
                target.setText(directory)

        def _pick_settings_sample(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self._text("dialog_select_directory"),
                filter="FITS Files (*.fit *.fits *.fts);;All Files (*)",
            )
            if path:
                self.settings_sample_edit.setText(path)

        def _populate_settings_ui(self) -> None:
            settings = self._settings
            self.settings_db_edit.setText(settings.db_root or "")
            self.settings_index_edit.setText(settings.index_root or "")
            self.settings_mag_spin.setValue(settings.mag_cap)
            self.settings_max_stars_spin.setValue(settings.max_stars)
            self.settings_max_quads_spin.setValue(settings.max_quads_per_tile)
            self.settings_sample_edit.setText(settings.sample_fits or "")
            # Blind group
            if hasattr(self, 'settings_blind_max_stars_spin'):
                self.settings_blind_max_stars_spin.setValue(settings.blind_max_stars)
            if hasattr(self, 'settings_blind_max_quads_spin'):
                self.settings_blind_max_quads_spin.setValue(settings.blind_max_quads)
            if hasattr(self, 'settings_blind_max_candidates_spin'):
                self.settings_blind_max_candidates_spin.setValue(settings.blind_max_candidates)
            if hasattr(self, 'settings_blind_pixel_tol_spin'):
                self.settings_blind_pixel_tol_spin.setValue(settings.blind_pixel_tolerance)
            if hasattr(self, 'settings_blind_quality_inliers_spin'):
                self.settings_blind_quality_inliers_spin.setValue(settings.blind_quality_inliers)
            if hasattr(self, 'settings_blind_quality_rms_spin'):
                self.settings_blind_quality_rms_spin.setValue(settings.blind_quality_rms)
            if hasattr(self, 'settings_blind_fast_check'):
                self.settings_blind_fast_check.setChecked(settings.blind_fast_mode)
            # Also refresh the solver tab family dropdown from the chosen index,
            # and restore previously saved family selection if any.
            try:
                self._populate_families_from_index(settings.index_root or "")
                fam = (settings.solver_family or "").strip().lower()
                if fam:
                    idx = self.families_combo.findData(fam)
                    if idx >= 0:
                        self.families_combo.setCurrentIndex(idx)
            except Exception:
                pass

        def _read_settings_from_ui(self) -> PersistentSettings:
            db_root = self.settings_db_edit.text().strip()
            if not db_root:
                raise ValueError(self._text("error_database_required"))
            index_root = self.settings_index_edit.text().strip()
            if not index_root:
                raise ValueError(self._text("settings_index_missing"))
            # Enforce separation between database/ and index/
            try:
                dbp = Path(db_root).expanduser().resolve()
                idxp = Path(index_root).expanduser().resolve()
                same = (idxp == dbp)
                inside = str(idxp).startswith(str(dbp) + os.sep)
                if same or inside:
                    raise ValueError(self._text("warn_sep_dirs"))
            except Exception as _exc:
                # Convert any path-related issue into a user-facing message
                if not isinstance(_exc, ValueError):
                    raise ValueError(self._text("warn_sep_dirs"))
                raise
            # Read detection device from Performance tab
            try:
                sel = self.perf_detect_combo.currentData()
                if isinstance(sel, tuple):
                    backend_sel, dev_sel = sel
                else:
                    backend_sel, dev_sel = ("cpu", -1)
            except Exception:
                backend_sel, dev_sel = ("cpu", -1)
            return PersistentSettings(
                db_root=db_root,
                index_root=index_root,
                mag_cap=float(self.settings_mag_spin.value()),
                max_stars=int(self.settings_max_stars_spin.value()),
                max_quads_per_tile=int(self.settings_max_quads_spin.value()),
                sample_fits=self.settings_sample_edit.text().strip() or None,
                last_preset_id=(self.presets_combo.currentData() if hasattr(self, 'presets_combo') else None),
                last_fov_focal_mm=float(self.fov_focal_spin.value()) if hasattr(self, 'fov_focal_spin') else 0.0,
                last_fov_pixel_um=float(self.fov_pixel_spin.value()) if hasattr(self, 'fov_pixel_spin') else 0.0,
                last_fov_res_w=int(self.fov_res_w_spin.value()) if hasattr(self, 'fov_res_w_spin') else 0,
                last_fov_res_h=int(self.fov_res_h_spin.value()) if hasattr(self, 'fov_res_h_spin') else 0,
                last_fov_reducer=float(self.fov_reducer_spin.value()) if hasattr(self, 'fov_reducer_spin') else 1.0,
                last_fov_binning=int(self.fov_binning_spin.value()) if hasattr(self, 'fov_binning_spin') else 1,
                blind_max_stars=int(self.settings_blind_max_stars_spin.value()),
                blind_max_quads=int(self.settings_blind_max_quads_spin.value()),
                blind_max_candidates=int(self.settings_blind_max_candidates_spin.value()),
                blind_pixel_tolerance=float(self.settings_blind_pixel_tol_spin.value()),
                blind_quality_inliers=int(self.settings_blind_quality_inliers_spin.value()),
                blind_quality_rms=float(self.settings_blind_quality_rms_spin.value()),
                blind_fast_mode=bool(self.settings_blind_fast_check.isChecked()),
                near_max_tile_candidates=int(self.perf_near_max_tiles_spin.value()) if hasattr(self, 'perf_near_max_tiles_spin') else 48,
                near_tile_cache_size=int(self.perf_near_cache_spin.value()) if hasattr(self, 'perf_near_cache_spin') else 128,
                near_detect_backend=str(backend_sel),
                near_detect_device=int(dev_sel if isinstance(dev_sel, int) else 0),
                io_concurrency=int(self.perf_io_spin.value()) if hasattr(self, 'perf_io_spin') else 0,
                near_warm_start=bool(self.perf_near_warm_check.isChecked()) if hasattr(self, 'perf_near_warm_check') else True,
            )

        def _on_compute_fov_clicked(self) -> None:
            try:
                focal = float(self.fov_focal_spin.value())
                pixel = float(self.fov_pixel_spin.value())
                rw = int(self.fov_res_w_spin.value())
                rh = int(self.fov_res_h_spin.value())
                red = float(self.fov_reducer_spin.value())
                binning = int(self.fov_binning_spin.value())
                geo = preset_utils.compute_scale_and_fov(
                    focal, pixel, rw, rh, reducer=red, binning=binning
                )
                rec = preset_utils.recommend_params(
                    geo["scale_arcsec_per_px"], geo["fov_diag_deg"]
                )
                # Update recommendation labels
                self.reco_scale_value.setText(f"{geo['scale_arcsec_per_px']:.3f}")
                self.reco_fov_value.setText(
                    f"{geo['fov_w_deg']:.3f}° × {geo['fov_h_deg']:.3f}° (diag {geo['fov_diag_deg']:.3f}°)"
                )
                self.reco_mag_value.setText(f"{rec['mag_cap']:.2f}")
                self.reco_quads_value.setText(
                    f"{preset_utils.describe_quads_profile(rec['levels'])} / {int(rec['max_quads_per_tile'])}"
                )
                self.reco_notes_label.setText(str(rec.get('notes') or ''))
                # Apply to index build fields for convenience
                try:
                    self.settings_mag_spin.setValue(float(rec["mag_cap"]))
                except Exception:
                    pass
                try:
                    self.settings_max_quads_spin.setValue(int(rec["max_quads_per_tile"]))
                except Exception:
                    pass
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))

        def _log_settings(self, message: str) -> None:
            timestamp = time.strftime("%H:%M:%S")
            self.settings_log_view.appendPlainText(f"[{timestamp}] {message}")
            # Avoid echoing into root logger (prevents feedback via the forwarding handler)

        def _log_index_health(self, index_root_text: str) -> None:
            try:
                from zeblindsolver.quad_index_builder import validate_index
                root = Path(index_root_text).expanduser().resolve()
                health = validate_index(root)
                tiles = int(health.get("manifest_tile_count", 0) or 0)
                empty = int(health.get("empty_tiles_total", 0) or 0)
                ratio = float(health.get("empty_ratio_overall", 0.0) or 0.0)
                rings = health.get("bad_empty_rings") or []
                ring_str = ",".join(str(r) for r in rings) if rings else "-"
                if rings or ratio >= 0.20:
                    self._log_settings(
                        self._text(
                            "settings_index_health_bad",
                            tiles=tiles,
                            empty=empty,
                            percent=100.0 * ratio,
                            rings=ring_str,
                        )
                    )
                else:
                    self._log_settings(
                        self._text(
                            "settings_index_health_ok",
                            tiles=tiles,
                            empty=empty,
                            percent=100.0 * ratio,
                        )
                    )
            except Exception:
                # Non-fatal; keep UI responsive even if index is missing/broken
                pass

        def _on_save_settings_clicked(self) -> None:
            try:
                settings = self._read_settings_from_ui()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))
                return
            self._settings = settings
            save_persistent_settings(settings)
            self._log_settings(self._text("settings_saved"))
            # Refresh families dropdown in solver tab after saving new index root
            try:
                self._populate_families_from_index(settings.index_root or "")
            except Exception:
                pass

        def _on_build_index_clicked(self) -> None:
            if self._index_worker:
                return
            # Ensure user explicitly picks a destination separate from the ASTAP database
            # to avoid mixing `database/` and `index/` content.
            db_root_text = self.settings_db_edit.text().strip()
            if not db_root_text:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("error_database_required"))
                return
            db_root_path = Path(db_root_text).expanduser().resolve()
            current_index = self.settings_index_edit.text().strip()
            need_pick = not current_index
            if not need_pick:
                try:
                    idx_path = Path(current_index).expanduser().resolve()
                    # If index is the same as DB or inside DB, force a new pick
                    same = idx_path == db_root_path
                    inside = str(idx_path).startswith(str(db_root_path) + os.sep)
                    need_pick = same or inside
                except Exception:
                    need_pick = True
            if need_pick:
                # Suggest a sibling "index" next to the DB root
                suggested = db_root_path.parent / "index"
                picked = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    self._text("dialog_select_directory"),
                    str(suggested),
                )
                if not picked:
                    return
                self.settings_index_edit.setText(picked)
            try:
                settings = self._read_settings_from_ui()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))
                return
            # Integrity check: if index looks complete and matches current DB, ask before rebuilding
            idx_path = Path(settings.index_root).expanduser().resolve()
            try:
                from zeblindsolver.quad_index_builder import validate_index
                info = validate_index(idx_path, db_root=Path(settings.db_root).expanduser().resolve())
            except Exception:
                info = {"manifest_ok": False}
            if info.get("manifest_ok"):
                missing = info.get("missing_quads") or []
                db_mismatch = bool(info.get("db_root_mismatch"))
                tile_mismatch = bool(info.get("tile_key_mismatch")) if info.get("tile_key_mismatch") is not None else False
                if not missing and not db_mismatch and not tile_mismatch:
                    # Index appears complete and up to date
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        self._text("settings_rebuild_title"),
                        self._text("settings_rebuild_text", path=str(idx_path)),
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if answer != QtWidgets.QMessageBox.Yes:
                        return
                elif missing and not db_mismatch and not tile_mismatch:
                    # Only quads are missing; propose quads-only build
                    text = f"Missing quad tables: {', '.join(missing)}. Build quads only?"
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        self._text("dialog_config_title"),
                        text,
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.Yes,
                    )
                    if answer == QtWidgets.QMessageBox.Yes:
                        # Run in quads-only mode
                        self._settings = settings
                        save_persistent_settings(settings)
                        self.settings_build_btn.setEnabled(False)
                        if hasattr(self, 'settings_run_blind_btn'):
                            self.settings_run_blind_btn.setEnabled(False)
                        if hasattr(self, 'settings_run_near_btn'):
                            self.settings_run_near_btn.setEnabled(False)
                        self._log_settings(self._text("settings_build_start", path=str(idx_path)))
                        self._index_worker = IndexBuilder(
                            db_root=settings.db_root,
                            index_root=settings.index_root,
                            mag_cap=settings.mag_cap,
                            max_stars=settings.max_stars,
                            max_quads_per_tile=settings.max_quads_per_tile,
                            quads_only=True,
                        )
                        self._index_worker.log.connect(self._log_settings)
                        self._index_worker.progress.connect(self._on_index_progress)
                        self._index_worker.finished.connect(self._on_index_finished)
                        self._index_worker.start()
                        return
            self._settings = settings
            save_persistent_settings(settings)
            # Disable build/run while index build is in progress
            self.settings_build_btn.setEnabled(False)
            if hasattr(self, 'settings_run_blind_btn'):
                self.settings_run_blind_btn.setEnabled(False)
            if hasattr(self, 'settings_run_near_btn'):
                self.settings_run_near_btn.setEnabled(False)
            # Inform user build is long-running
            self._log_settings(self._text("settings_build_start", path=str(Path(settings.index_root).expanduser().resolve())))
            self._index_worker = IndexBuilder(
                db_root=settings.db_root,
                index_root=settings.index_root,
                mag_cap=settings.mag_cap,
                max_stars=settings.max_stars,
                max_quads_per_tile=settings.max_quads_per_tile,
                quads_only=False,
            )
            self._index_worker.log.connect(self._log_settings)
            self._index_worker.progress.connect(self._on_index_progress)
            self._index_worker.finished.connect(self._on_index_finished)
            self._index_worker.start()

        def _on_run_blind_clicked(self) -> None:
            if self._blind_worker:
                return
            sample_path = self.settings_sample_edit.text().strip()
            if not sample_path:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("settings_sample_required"))
                return
            if not Path(sample_path).is_file():
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("error_input_missing", path=sample_path))
                return
            try:
                settings = self._read_settings_from_ui()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))
                return
            self._settings = settings
            save_persistent_settings(settings)
            self.settings_run_blind_btn.setEnabled(False)
            # Build blind config from settings
            blind_cfg = BlindSolveConfig(
                max_candidates=settings.blind_max_candidates,
                max_stars=settings.blind_max_stars,
                max_quads=settings.blind_max_quads,
                sip_order=2,
                quality_rms=settings.blind_quality_rms,
                quality_inliers=settings.blind_quality_inliers,
                pixel_tolerance=settings.blind_pixel_tolerance,
                fast_mode=settings.blind_fast_mode,
                log_level="INFO",
            )
            self._blind_worker = BlindRunner(
                fits_path=sample_path,
                index_root=settings.index_root,
                blind_config=blind_cfg,
            )
            self._blind_worker.log.connect(self._log_settings)
            self._blind_worker.finished.connect(self._on_blind_finished)
            self._blind_worker.start()

        def _on_run_near_clicked(self) -> None:
            if self._near_worker:
                return
            sample_path = self.settings_sample_edit.text().strip()
            if not sample_path:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("settings_sample_required"))
                return
            if not Path(sample_path).is_file():
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("error_input_missing", path=sample_path))
                return
            try:
                settings = self._read_settings_from_ui()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), str(exc))
                return
            self._settings = settings
            save_persistent_settings(settings)
            self.settings_run_near_btn.setEnabled(False)
            # Derive device selection from combo
            try:
                sel = self.perf_detect_combo.currentData()
                if isinstance(sel, tuple):
                    backend_sel, dev_sel = sel
                else:
                    backend_sel, dev_sel = ("cpu", -1)
            except Exception:
                backend_sel, dev_sel = ("cpu", -1)
            self._settings.near_detect_backend = str(backend_sel)
            self._settings.near_detect_device = int(dev_sel if isinstance(dev_sel, int) else 0)
            self._near_worker = NearRunner(
                fits_path=sample_path,
                index_root=settings.index_root,
                max_tiles=int(self.perf_near_max_tiles_spin.value()) if hasattr(self, 'perf_near_max_tiles_spin') else int(self._settings.near_max_tile_candidates or 48),
                tile_cache=int(self.perf_near_cache_spin.value()) if hasattr(self, 'perf_near_cache_spin') else int(self._settings.near_tile_cache_size or 128),
                detect_backend=self._settings.near_detect_backend or "auto",
                detect_device=self._settings.near_detect_device,
            )
            self._near_worker.log.connect(self._log_settings)
            self._near_worker.finished.connect(self._on_near_finished)
            self._near_worker.start()

        def _on_index_finished(self, success: bool, message: str) -> None:
            status = "ok" if success else "failed"
            self._log_settings(self._text("settings_index_result", status=status, message=message))
            # Surface index health after a build/rebuild completes
            try:
                if success:
                    self._log_index_health(self.settings_index_edit.text().strip())
                    # Refresh families dropdown in main solver tab
                    try:
                        self._populate_families_from_index(self.settings_index_edit.text().strip())
                    except Exception:
                        pass
            except Exception:
                pass
            self.settings_build_btn.setEnabled(True)
            if hasattr(self, 'settings_run_blind_btn'):
                self.settings_run_blind_btn.setEnabled(True)
            if hasattr(self, 'settings_run_near_btn'):
                self.settings_run_near_btn.setEnabled(True)
            self._index_worker = None
            # Reset progress bar
            try:
                self.settings_progress.setRange(0, 1)
                self.settings_progress.setValue(0)
                self.settings_progress.setFormat("Idle")
            except Exception:
                pass

        def _on_index_progress(self, value: int, total: int, label: str) -> None:
            try:
                self.settings_progress.setRange(0, total)
                self.settings_progress.setValue(value)
                self.settings_progress.setFormat(f"{label}  {value}/{total}")
            except Exception:
                pass

        def _on_blind_finished(self, success: bool, message: str) -> None:
            status = "ok" if success else "failed"
            self._log_settings(self._text("settings_blind_result", status=status, message=message))
            self.settings_run_blind_btn.setEnabled(True)
            self._blind_worker = None

        def _on_near_finished(self, success: bool, message: str) -> None:
            status = "ok" if success else "failed"
            self._log_settings(self._text("settings_near_result", status=status, message=message))
            self.settings_run_near_btn.setEnabled(True)
            self._near_worker = None

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
            # Cap main log growth: keep at most 5000 lines
            try:
                self.log_view.document().setMaximumBlockCount(5000)
            except Exception:
                pass
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
            self.browse_in_btn.setText(browse_label)
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
            # Populate catalog families from index manifest if available
            try:
                self._populate_families_from_index(self._settings.index_root or "")
                # Restore saved selection
                fam = (self._settings.solver_family or "").strip().lower()
                if fam:
                    idx = self.families_combo.findData(fam)
                    if idx >= 0:
                        self.families_combo.setCurrentIndex(idx)
            except Exception:
                pass
            self.blind_check.setText(self._text("blind_label"))
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
            if hasattr(self, "tabs"):
                try:
                    self.tabs.setTabText(0, self._text("solver_tab"))
                    self.tabs.setTabText(1, self._text("database_tab"))
                    self.tabs.setTabText(2, self._text("settings_tab"))
                    self.tabs.setTabText(3, self._text("performance_tab"))
                except Exception:
                    pass
            browse_label = self._text("browse_button")
            if hasattr(self, "settings_db_label"):
                self.settings_db_label.setText(self._text("settings_db_label"))
                self.settings_db_browse.setText(browse_label)
            if hasattr(self, "settings_index_label"):
                self.settings_index_label.setText(self._text("settings_index_label"))
                self.settings_index_browse.setText(browse_label)
            # Database tab labels
            if hasattr(self, "db_tab_label"):
                self.db_tab_label.setText(self._text("select_db_root"))
                self.db_tab_browse.setText(browse_label)
            if hasattr(self, "data_sources_label"):
                self.data_sources_label.setText(self._text("data_sources"))
            if hasattr(self, "sources_group"):
                self.sources_group.setTitle(self._text("data_sources"))
            # Downloads group labels
            if hasattr(self, "downloads_group"):
                self.downloads_group.setTitle(self._text("downloads_title"))
            if hasattr(self, "btn_add_to_queue"):
                self.btn_add_to_queue.setText(self._text("add_to_queue"))
            if hasattr(self, "btn_start_all"):
                self.btn_start_all.setText(self._text("start_all"))
            if hasattr(self, "btn_pause_all"):
                self.btn_pause_all.setText(self._text("pause_all"))
            if hasattr(self, "btn_verify_hashes"):
                self.btn_verify_hashes.setText(self._text("verify_hashes"))
            if hasattr(self, "btn_add_selected"):
                self.btn_add_selected.setText(self._text("add_selected"))
            if hasattr(self, "settings_mag_label"):
                self.settings_mag_label.setText(self._text("settings_mag_label"))
            if hasattr(self, "settings_max_stars_label"):
                self.settings_max_stars_label.setText(self._text("settings_max_stars_label"))
            if hasattr(self, "settings_max_quads_label"):
                self.settings_max_quads_label.setText(self._text("settings_max_quads_label"))
            if hasattr(self, "blind_group"):
                self.blind_group.setTitle(self._text("settings_blind_group"))
            # Presets/FOV groups
            if hasattr(self, "presets_group"):
                self.presets_group.setTitle(self._text("presets_title"))
            if hasattr(self, "fov_group"):
                self.fov_group.setTitle(self._text("fov_mode_title"))
            if hasattr(self, "compute_button"):
                self.compute_button.setText(self._text("compute_button"))
            if hasattr(self, "reco_group"):
                self.reco_group.setTitle(self._text("recommendations_title"))
                self.reco_scale_label.setText(self._text("est_scale"))
                self.reco_fov_label.setText(self._text("est_fov"))
                self.reco_mag_label.setText(self._text("mag_cap_suggested"))
                self.reco_quads_label.setText(self._text("quads_profile"))
            if hasattr(self, "settings_blind_max_stars_label"):
                self.settings_blind_max_stars_label.setText(self._text("settings_blind_max_stars_label"))
            if hasattr(self, "settings_blind_max_quads_label"):
                self.settings_blind_max_quads_label.setText(self._text("settings_blind_max_quads_label"))
            if hasattr(self, "settings_blind_max_candidates_label"):
                self.settings_blind_max_candidates_label.setText(self._text("settings_blind_max_candidates_label"))
            if hasattr(self, "settings_blind_pixel_tol_label"):
                self.settings_blind_pixel_tol_label.setText(self._text("settings_blind_pixel_tol_label"))
            if hasattr(self, "settings_blind_quality_inliers_label"):
                self.settings_blind_quality_inliers_label.setText(self._text("settings_blind_quality_inliers_label"))
            if hasattr(self, "settings_blind_quality_rms_label"):
                self.settings_blind_quality_rms_label.setText(self._text("settings_blind_quality_rms_label"))
            if hasattr(self, "settings_sample_label"):
                self.settings_sample_label.setText(self._text("settings_sample_label"))
                self.settings_sample_browse.setText(browse_label)
            if hasattr(self, "settings_save_btn"):
                self.settings_save_btn.setText(self._text("settings_save_btn"))
            if hasattr(self, "settings_build_btn"):
                self.settings_build_btn.setText(self._text("settings_build_btn"))
            if hasattr(self, "settings_run_blind_btn"):
                self.settings_run_blind_btn.setText(self._text("settings_run_btn"))
            if hasattr(self, "settings_run_near_btn"):
                self.settings_run_near_btn.setText(self._text("settings_near_btn"))
            if hasattr(self, "settings_log_label"):
                self.settings_log_label.setText(self._text("settings_log"))
            # Performance tab labels
            if hasattr(self, "perf_near_cache_label"):
                self.perf_near_cache_label.setText(self._text("settings_perf_near_cache_label"))
            if hasattr(self, "perf_near_max_tiles_label"):
                self.perf_near_max_tiles_label.setText(self._text("settings_perf_near_max_tiles_label"))
            if hasattr(self, "perf_detect_label"):
                self.perf_detect_label.setText(self._text("settings_perf_detect_label"))
            if hasattr(self, "perf_io_label"):
                self.perf_io_label.setText(self._text("settings_perf_io_label"))
            if hasattr(self, "perf_near_warm_check"):
                # Checkboxes show their own text; set it here
                self.perf_near_warm_check.setText(self._text("settings_perf_near_warm_label"))
            if hasattr(self, "performance_save_btn"):
                self.performance_save_btn.setText(self._text("settings_save_btn"))
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
                db_root = str(cli_args.db_root)
                self._settings.db_root = db_root
                self.settings_db_edit.setText(db_root)
            if cli_args.blind_index:
                index_root = str(cli_args.blind_index)
                self._settings.index_root = index_root
                self.settings_index_edit.setText(index_root)
                # Perform a quick sanity-check and surface advice in the settings log
                self._log_index_health(index_root)
            if cli_args.input_dir:
                self.input_edit.setText(str(cli_args.input_dir))
                QtCore.QTimer.singleShot(100, self.scan_files)

        # --- Actions -------------------------------------------------------------
        def _pick_directory(self, line_edit: QtWidgets.QLineEdit, *, trigger_scan: bool = False) -> None:
            opts = QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontUseNativeDialog
            try:
                opts |= QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons
            except Exception:
                pass
            try:
                opts |= QtWidgets.QFileDialog.Option.DontResolveSymlinks
            except Exception:
                pass
            start = line_edit.text().strip() or str(Path.home())
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self._text("dialog_select_directory"),
                start,
                options=opts,
            )
            if directory:
                line_edit.setText(directory)
                if trigger_scan:
                    self.scan_files()

        def scan_files(self) -> None:
            # Cancel previous scan if any
            try:
                if self._scanner and self._scanner.isRunning():
                    self._scanner.cancel()
            except Exception:
                pass
            # Validate directory
            path = self.input_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("error_select_input"))
                return
            directory = Path(path).expanduser()
            if not directory.is_dir():
                QtWidgets.QMessageBox.warning(self, self._text("dialog_config_title"), self._text("error_input_missing", path=directory))
                return
            # Prepare UI
            self._pending_files = []
            self._current_input_dir = directory
            self.files_view.clear()
            self._item_by_path.clear()
            self.files_view.setSortingEnabled(False)
            self._scan_buffer.clear()
            # Build scanner
            formats = self._parse_formats()
            limit = self.max_files_spin.value()
            limit = limit if limit > 0 else None
            self._scanner = FileScanner(directory, formats, limit)
            self._scanner.file_found.connect(self._on_scan_file_found)
            self._scanner.finished.connect(self._on_scan_finished)
            self._scanner.start()
            # Give immediate feedback
            self._log(self._text("info_files_detected", count=0))

        def _on_scan_file_found(self, path_text: str) -> None:
            try:
                p = Path(path_text).resolve()
                self._pending_files.append(p)
                self._scan_buffer.append(p)
            except Exception:
                return
            # Flush in chunks to keep UI responsive
            if len(self._scan_buffer) >= self._scan_flush_threshold:
                self._flush_scan_buffer()
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            # Update info line periodically
            if len(self._pending_files) % 500 == 0:
                self._log(self._text("info_files_detected", count=len(self._pending_files)))

        def _on_scan_finished(self, count: int) -> None:
            self._flush_scan_buffer()
            self._log(self._text("info_files_detected", count=len(self._pending_files)))

        def _flush_scan_buffer(self) -> None:
            if not self._scan_buffer:
                return
            self.files_view.setUpdatesEnabled(False)
            try:
                items = []
                for path in self._scan_buffer:
                    item = QtWidgets.QTreeWidgetItem(
                        [self._format_path(path), self._status_label_for("waiting"), ""]
                    )
                    item.setData(1, QtCore.Qt.UserRole, "waiting")
                    items.append(item)
                if items:
                    self.files_view.addTopLevelItems(items)
                    self.files_view.resizeColumnToContents(0)
            except Exception:
                pass
            finally:
                self.files_view.setUpdatesEnabled(True)
                self._scan_buffer.clear()

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
            return _parse_formats_value(self.formats_edit.text())

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
            db_root_text = self._settings.db_root
            if not db_root_text:
                raise ValueError(self._text("error_database_required"))
            db_root = Path(db_root_text).expanduser()
            if not db_root.is_dir():
                raise ValueError(self._text("error_database_missing", path=db_root))
            if not self._current_input_dir:
                raise ValueError(self._text("error_no_input_dir"))
            # Family selection via dropdown ('Auto' → no restriction)
            selected_family = self.families_combo.currentData()
            families = [str(selected_family).strip().lower()] if selected_family else []
            formats = tuple(self._parse_formats())
            max_files = self.max_files_spin.value() or None
            max_radius_value = self.max_radius_spin.value()
            max_radius = max_radius_value if max_radius_value > 0 else None
            index_root_text = self._settings.index_root
            if not index_root_text:
                raise ValueError(self._text("settings_index_missing"))
            index_root = Path(index_root_text).expanduser()
            if not index_root.is_dir():
                raise ValueError(self._text("settings_index_missing"))
            # Persist solver panel + performance settings for next runs
            try:
                self._settings.solver_fov_deg = float(self.fov_spin.value())
                self._settings.solver_search_scale = float(self.search_scale_spin.value())
                self._settings.solver_search_attempts = int(self.search_attempts_spin.value())
                self._settings.solver_max_radius_deg = float(self.max_radius_spin.value())
                self._settings.solver_downsample = int(self.downsample_spin.value())
                self._settings.solver_workers = int(self.workers_spin.value())
                self._settings.solver_cache_size = int(self.cache_spin.value())
                self._settings.solver_max_files = int(self.max_files_spin.value())
                self._settings.solver_formats = self.formats_edit.text().strip() or None
                sel_fam = self.families_combo.currentData() or ""
                self._settings.solver_family = str(sel_fam).strip().lower() or None
                self._settings.solver_blind_enabled = bool(self.blind_check.isChecked())
                self._settings.solver_overwrite = bool(self.overwrite_check.isChecked())
                # Pull performance tab values if present
                if hasattr(self, 'perf_near_max_tiles_spin'):
                    self._settings.near_max_tile_candidates = int(self.perf_near_max_tiles_spin.value())
                if hasattr(self, 'perf_near_cache_spin'):
                    self._settings.near_tile_cache_size = int(self.perf_near_cache_spin.value())
                save_persistent_settings(self._settings)
            except Exception:
                pass
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
                blind_enabled=self.blind_check.isChecked(),
                blind_index_path=index_root,
                near_max_tile_candidates=int(self._settings.near_max_tile_candidates or 48),
                near_tile_cache_size=int(self._settings.near_tile_cache_size or 128),
                near_detect_backend=str(self._settings.near_detect_backend or "auto"),
                near_detect_device=int(self._settings.near_detect_device) if self._settings.near_detect_device is not None else None,
                io_concurrency=int(self._settings.io_concurrency or 0),
                near_warm_start=bool(self._settings.near_warm_start),
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
            logging.info(message)
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
    window = ZeSolverWindow(persistent_settings)
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
