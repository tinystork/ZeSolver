from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from zeblindsolver.db_convert import (
    DEFAULT_MAG_CAP,
    DEFAULT_MAX_QUADS_PER_TILE,
    DEFAULT_MAX_STARS,
)

# GUI + solver defaults used across both the CLI entry point and the tests
DEFAULT_FOV_DEG = 1.5
DEFAULT_SEARCH_RADIUS_SCALE = 1.8
DEFAULT_SEARCH_RADIUS_ATTEMPTS = 3

SETTINGS_PATH = Path.home() / ".zesolver_settings.json"
# Increment when the on-disk settings layout or recommended defaults change
SETTINGS_SCHEMA_VERSION = 5

QUAD_STORAGE_CHOICES = ("npz", "npz_uncompressed", "npy")
TILE_COMPRESSION_CHOICES = ("compressed", "uncompressed")


@dataclass
class PersistentSettings:
    schema_version: int = SETTINGS_SCHEMA_VERSION
    db_root: Optional[str] = None
    index_root: Optional[str] = None
    mag_cap: float = DEFAULT_MAG_CAP
    max_stars: int = DEFAULT_MAX_STARS
    max_quads_per_tile: int = DEFAULT_MAX_QUADS_PER_TILE
    quad_storage: str = QUAD_STORAGE_CHOICES[0]
    tile_compression: str = TILE_COMPRESSION_CHOICES[0]
    log_level: str = "INFO"
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
    blind_max_stars: int = 500
    blind_max_quads: int = 8000
    blind_max_candidates: int = 10
    blind_pixel_tolerance: float = 2.5
    blind_quality_inliers: int = 40
    blind_quality_rms: float = 1.2
    blind_fast_mode: bool = True
    # Near solver performance
    near_max_tile_candidates: int = 48
    near_tile_cache_size: int = 128
    near_detect_backend: str = "auto"  # auto|cpu|cuda
    near_detect_device: int = 0
    io_concurrency: int = 0
    near_warm_start: bool = True
    # Near (fast) solver quality and tuning
    near_quality_inliers: int = 60
    near_quality_rms: float = 1.0
    near_pixel_tolerance: float = 3.0
    near_ransac_trials: int = 1200
    near_max_img_stars: int = 800
    near_max_cat_stars: int = 2000
    near_try_parity_flip: bool = True
    near_search_margin: float = 1.2
    dev_bucket_limit_override: int = 0
    dev_vote_percentile: int = 40
    dev_detect_k_sigma: float = 3.0
    dev_detect_min_area: int = 5
    dev_bucket_cap_S: int = 6000
    dev_bucket_cap_M: int = 4096
    dev_bucket_cap_L: int = 8192
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
    solver_hint_ra_deg: Optional[float] = None
    solver_hint_dec_deg: Optional[float] = None
    solver_hint_radius_deg: Optional[float] = None
    solver_hint_focal_mm: Optional[float] = None
    solver_hint_pixel_um: Optional[float] = None
    solver_hint_resolution_arcsec: Optional[float] = None
    solver_hint_resolution_min_arcsec: Optional[float] = None
    solver_hint_resolution_max_arcsec: Optional[float] = None
    # Solver backend selection + Astrometry.net web backend
    solver_backend: str = "local"  # "local" or "astrometry"
    astrometry_api_url: str = "https://nova.astrometry.net/api"
    astrometry_api_key: Optional[str] = None
    astrometry_parallel_jobs: int = 2
    astrometry_timeout_s: int = 600
    astrometry_use_hints: bool = True
    astrometry_fallback_local: bool = True


def _resolve_settings_path() -> Path:
    """Return the active settings path, honoring runtime overrides."""
    pkg = sys.modules.get("zesolver")
    if pkg is not None:
        override = getattr(pkg, "SETTINGS_PATH", None)
        if override:
            try:
                return Path(override).expanduser()
            except Exception:
                pass
    return SETTINGS_PATH


def load_persistent_settings() -> PersistentSettings:
    path = _resolve_settings_path()
    if not path.exists():
        return PersistentSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return PersistentSettings()
    if not isinstance(payload, dict):
        return PersistentSettings()

    def _float_or_none(value: object) -> Optional[float]:
        if value in (None, "", False):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_choice(value: object, choices: tuple[str, ...], default: str) -> str:
        candidate = default
        if isinstance(value, str) and value.strip():
            candidate = value.strip().lower()
        if candidate not in choices:
            return default
        return candidate

    settings = PersistentSettings(
        schema_version=int(payload.get("schema_version", 1)),
        db_root=payload.get("db_root"),
        index_root=payload.get("index_root"),
        mag_cap=float(payload.get("mag_cap", DEFAULT_MAG_CAP)),
        max_stars=int(payload.get("max_stars", DEFAULT_MAX_STARS)),
        max_quads_per_tile=int(payload.get("max_quads_per_tile", DEFAULT_MAX_QUADS_PER_TILE)),
        quad_storage=_normalize_choice(payload.get("quad_storage"), QUAD_STORAGE_CHOICES, QUAD_STORAGE_CHOICES[0]),
        tile_compression=_normalize_choice(payload.get("tile_compression"), TILE_COMPRESSION_CHOICES, TILE_COMPRESSION_CHOICES[0]),
        log_level=str(payload.get("log_level", "INFO") or "INFO").upper(),
        sample_fits=payload.get("sample_fits"),
        last_preset_id=(payload.get("last_preset_id") or None),
        last_fov_focal_mm=float(payload.get("last_fov_focal_mm", 0.0)),
        last_fov_pixel_um=float(payload.get("last_fov_pixel_um", 0.0)),
        last_fov_res_w=int(payload.get("last_fov_res_w", 0)),
        last_fov_res_h=int(payload.get("last_fov_res_h", 0)),
        last_fov_reducer=float(payload.get("last_fov_reducer", 1.0)),
        last_fov_binning=int(payload.get("last_fov_binning", 1)),
        blind_max_stars=int(payload.get("blind_max_stars", 500)),
        blind_max_quads=int(payload.get("blind_max_quads", 8000)),
        blind_max_candidates=int(payload.get("blind_max_candidates", 10)),
        blind_pixel_tolerance=float(payload.get("blind_pixel_tolerance", 2.5)),
        blind_quality_inliers=int(payload.get("blind_quality_inliers", 40)),
        blind_quality_rms=float(payload.get("blind_quality_rms", 1.2)),
        blind_fast_mode=bool(payload.get("blind_fast_mode", True)),
        near_max_tile_candidates=int(payload.get("near_max_tile_candidates", 48)),
        near_tile_cache_size=int(payload.get("near_tile_cache_size", 128)),
        near_detect_backend=str(payload.get("near_detect_backend", "auto")),
        near_detect_device=int(payload.get("near_detect_device", 0)),
        io_concurrency=int(payload.get("io_concurrency", 0)),
        near_warm_start=bool(payload.get("near_warm_start", True)),
        near_quality_inliers=int(payload.get("near_quality_inliers", 60)),
        near_quality_rms=float(payload.get("near_quality_rms", 1.0)),
        near_pixel_tolerance=float(payload.get("near_pixel_tolerance", 3.0)),
        near_ransac_trials=int(payload.get("near_ransac_trials", 1200)),
        near_max_img_stars=int(payload.get("near_max_img_stars", 800)),
        near_max_cat_stars=int(payload.get("near_max_cat_stars", 2000)),
        near_try_parity_flip=bool(payload.get("near_try_parity_flip", True)),
        near_search_margin=float(payload.get("near_search_margin", 1.2)),
        dev_bucket_limit_override=int(payload.get("dev_bucket_limit_override", 0)),
        dev_vote_percentile=int(payload.get("dev_vote_percentile", 40)),
        dev_detect_k_sigma=float(payload.get("dev_detect_k_sigma", 3.0)),
        dev_detect_min_area=int(payload.get("dev_detect_min_area", 5)),
        dev_bucket_cap_S=int(payload.get("dev_bucket_cap_S", 6000)),
        dev_bucket_cap_M=int(payload.get("dev_bucket_cap_M", 4096)),
        dev_bucket_cap_L=int(payload.get("dev_bucket_cap_L", 8192)),
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
        solver_hint_ra_deg=_float_or_none(payload.get("solver_hint_ra_deg")),
        solver_hint_dec_deg=_float_or_none(payload.get("solver_hint_dec_deg")),
        solver_hint_radius_deg=_float_or_none(payload.get("solver_hint_radius_deg")),
        solver_hint_focal_mm=_float_or_none(payload.get("solver_hint_focal_mm")),
        solver_hint_pixel_um=_float_or_none(payload.get("solver_hint_pixel_um")),
        solver_hint_resolution_arcsec=_float_or_none(payload.get("solver_hint_resolution_arcsec")),
        solver_hint_resolution_min_arcsec=_float_or_none(payload.get("solver_hint_resolution_min_arcsec")),
        solver_hint_resolution_max_arcsec=_float_or_none(payload.get("solver_hint_resolution_max_arcsec")),
        solver_backend=str(payload.get("solver_backend", "local") or "local"),
        astrometry_api_url=str(payload.get("astrometry_api_url", "https://nova.astrometry.net/api") or "https://nova.astrometry.net/api"),
        astrometry_api_key=(payload.get("astrometry_api_key") or None),
        astrometry_parallel_jobs=int(payload.get("astrometry_parallel_jobs", 2)),
        astrometry_timeout_s=int(payload.get("astrometry_timeout_s", 600)),
        astrometry_use_hints=bool(payload.get("astrometry_use_hints", True)),
        astrometry_fallback_local=bool(payload.get("astrometry_fallback_local", True)),
    )
    migrated, updated = _migrate_settings_if_needed(settings)
    if updated:
        try:
            save_persistent_settings(migrated)
        except Exception:
            pass
        return migrated
    return settings


def save_persistent_settings(settings: PersistentSettings) -> None:
    path = _resolve_settings_path()
    data = asdict(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _migrate_settings_if_needed(settings: PersistentSettings) -> tuple[PersistentSettings, bool]:
    """Upgrade persisted settings to the current schema and defaults."""
    changed = False
    try:
        current_version = int(getattr(settings, "schema_version", 1) or 1)
    except Exception:
        current_version = 1

    # Legacy defaults shipped previously
    LEGACY = {
        "max_stars": 800,
        "max_quads": 12000,
        "max_candidates": 12,
        "pixel_tol": 3.0,
        "quality_inliers": {60, 12},
        "quality_rms": 1.0,
        "fast_mode": False,
    }
    # New recommended defaults
    NEW = {
        "max_stars": 500,
        "max_quads": 8000,
        "max_candidates": 10,
        "pixel_tol": 2.5,
        "quality_inliers": 40,
        "quality_rms": 1.2,
        "fast_mode": True,
    }

    if current_version < SETTINGS_SCHEMA_VERSION:
        legacy_match = (
            settings.blind_max_stars == LEGACY["max_stars"]
            and settings.blind_max_quads == LEGACY["max_quads"]
            and settings.blind_max_candidates == LEGACY["max_candidates"]
            and abs(settings.blind_pixel_tolerance - LEGACY["pixel_tol"]) < 1e-6
            and (settings.blind_quality_inliers in LEGACY["quality_inliers"])
            and abs(settings.blind_quality_rms - LEGACY["quality_rms"]) < 1e-6
            and settings.blind_fast_mode == LEGACY["fast_mode"]
        )
        if legacy_match:
            settings.blind_max_stars = NEW["max_stars"]
            settings.blind_max_quads = NEW["max_quads"]
            settings.blind_max_candidates = NEW["max_candidates"]
            settings.blind_pixel_tolerance = NEW["pixel_tol"]
            settings.blind_quality_inliers = NEW["quality_inliers"]
            settings.blind_quality_rms = NEW["quality_rms"]
            settings.blind_fast_mode = NEW["fast_mode"]
            changed = True
        else:
            if settings.blind_quality_inliers <= 20:
                settings.blind_quality_inliers = max(20, NEW["quality_inliers"])
                changed = True
            if settings.blind_fast_mode is False:
                if (
                    settings.blind_max_candidates >= 12
                    or settings.blind_pixel_tolerance >= 3.0
                    or settings.blind_max_quads >= 12000
                ):
                    settings.blind_fast_mode = True
                    changed = True
        settings.schema_version = SETTINGS_SCHEMA_VERSION
        changed = True

    return settings, changed


__all__ = [
    "DEFAULT_FOV_DEG",
    "DEFAULT_SEARCH_RADIUS_ATTEMPTS",
    "DEFAULT_SEARCH_RADIUS_SCALE",
    "QUAD_STORAGE_CHOICES",
    "TILE_COMPRESSION_CHOICES",
    "PersistentSettings",
    "SETTINGS_PATH",
    "SETTINGS_SCHEMA_VERSION",
    "load_persistent_settings",
    "save_persistent_settings",
]
