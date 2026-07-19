from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .astap_db_reader import TileMeta, iter_tiles, load_tile_stars
from .db_convert import DEFAULT_MAG_CAP, DEFAULT_MAX_STARS
from .projections import project_tan
from .quad_index_4d import (
    ASTROMETRY_AB_CODE_4D_DEFAULT_TOL,
    ASTROMETRY_AB_CODE_4D_SCHEMA,
    ASTROMETRY_AB_CODE_4D_VERSION,
    Quad4DPayloadTile,
    build_4d_index_from_payload_tiles,
)

BUILDER_VERSION = "astap_direct_4d_v1"
PROJECTION_IMPLEMENTATION = "zeblindsolver.projections.project_tan"
PROJECTION_VERSION = "tan_v1"
TAN_CENTER_POLICY = "cartesian_center_then_layout_fallback"
QUAD_SCHEMA = ASTROMETRY_AB_CODE_4D_SCHEMA
QUAD_VERSION = ASTROMETRY_AB_CODE_4D_VERSION


@dataclass(frozen=True, slots=True)
class AstapTileMaterializationConfig:
    mag_cap: float | None = DEFAULT_MAG_CAP
    source_max_stars: int = DEFAULT_MAX_STARS
    source_star_truncation_mode: str = "native_prefix"
    tan_center_policy: str = TAN_CENTER_POLICY
    max_stars_per_tile: int = 400


@dataclass(frozen=True, slots=True)
class Astap4DBuildConfig:
    family: str = "d50"
    tile_keys: tuple[str, ...] = field(default_factory=tuple)
    level: str = "S"
    mag_cap: float | None = DEFAULT_MAG_CAP
    source_max_stars: int = DEFAULT_MAX_STARS
    source_star_truncation_mode: str = "native_prefix"
    tan_center_policy: str = TAN_CENTER_POLICY
    max_stars_per_tile: int = 400
    max_quads_per_tile: int = 8000
    sampler_tag: str = "catalog_ring_coverage"
    code_tol_recommended: float = ASTROMETRY_AB_CODE_4D_DEFAULT_TOL
    dtype: str = "float32"
    projection_implementation: str = PROJECTION_IMPLEMENTATION
    projection_version: str = PROJECTION_VERSION
    quad_schema: str = QUAD_SCHEMA
    quad_version: int = QUAD_VERSION
    builder_version: str = BUILDER_VERSION

    def materialization_config(self) -> AstapTileMaterializationConfig:
        return AstapTileMaterializationConfig(
            mag_cap=self.mag_cap,
            source_max_stars=self.source_max_stars,
            source_star_truncation_mode=self.source_star_truncation_mode,
            tan_center_policy=self.tan_center_policy,
            max_stars_per_tile=self.max_stars_per_tile,
        )


@dataclass(frozen=True, slots=True)
class Quad4DSourceTile:
    tile_key: str
    family: str
    tile_code: str
    center_ra_deg: float
    center_dec_deg: float
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray
    x_deg: np.ndarray
    y_deg: np.ndarray
    source_star_indices: np.ndarray
    pre_mag_count: int
    post_mag_count: int
    post_source_limit_count: int
    post_4d_limit_count: int
    tan_center_policy: str
    used_layout_fallback: bool = False

    def as_payload_tile(self) -> Quad4DPayloadTile:
        return Quad4DPayloadTile(
            tile_key=self.tile_key,
            ra_deg=self.ra_deg,
            dec_deg=self.dec_deg,
            mag=self.mag,
            x_deg=self.x_deg,
            y_deg=self.y_deg,
        )


def _cartesian_center(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[float, float]:
    if ra_deg.size == 0:
        return 0.0, 0.0
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    cos_dec = np.cos(dec_rad)
    x = np.sum(cos_dec * np.cos(ra_rad))
    y = np.sum(cos_dec * np.sin(ra_rad))
    z = np.sum(np.sin(dec_rad))
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return float(np.mean(ra_deg)), float(np.mean(dec_deg))
    vx = x / norm
    vy = y / norm
    vz = z / norm
    ra = math.degrees(math.atan2(vy, vx)) % 360.0
    vz = min(1.0, max(-1.0, vz))
    dec = math.degrees(math.asin(vz))
    return ra, dec


def _validate_materialization_config(config: AstapTileMaterializationConfig) -> None:
    mode = str(config.source_star_truncation_mode or "").strip().lower()
    if mode not in {"native_prefix", "brightest_mag"}:
        raise ValueError("source_star_truncation_mode must be 'native_prefix' or 'brightest_mag'")
    if int(config.source_max_stars) < 0:
        raise ValueError("source_max_stars must be >= 0")
    if int(config.max_stars_per_tile) < 0:
        raise ValueError("max_stars_per_tile must be >= 0")
    if str(config.tan_center_policy) != TAN_CENTER_POLICY:
        raise ValueError(f"unsupported tan_center_policy: {config.tan_center_policy!r}")


def materialize_astap_tile_for_4d(
    db_root: Path | str,
    tile_meta: TileMeta,
    *,
    config: AstapTileMaterializationConfig | None = None,
) -> Quad4DSourceTile:
    cfg = config or AstapTileMaterializationConfig()
    _validate_materialization_config(cfg)
    stars = load_tile_stars(db_root, tile_meta)
    total = int(stars.size)
    used_layout_fallback = False
    if total:
        ra = stars["ra_deg"].astype(np.float64, copy=False)
        dec = stars["dec_deg"].astype(np.float64, copy=False)
        center_ra_deg, center_dec_deg = _cartesian_center(ra, dec)
        x_deg, y_deg = project_tan(ra, dec, center_ra_deg, center_dec_deg)
        valid = np.isfinite(x_deg) & np.isfinite(y_deg)
        if not valid.any():
            center_ra_deg = tile_meta.center_ra_deg
            center_dec_deg = tile_meta.center_dec_deg
            x_deg, y_deg = project_tan(ra, dec, center_ra_deg, center_dec_deg)
            valid = np.isfinite(x_deg) & np.isfinite(y_deg)
            used_layout_fallback = True
        stars = stars[valid]
        ra = ra[valid]
        dec = dec[valid]
        x_deg = x_deg[valid]
        y_deg = y_deg[valid]
    else:
        center_ra_deg = tile_meta.center_ra_deg
        center_dec_deg = tile_meta.center_dec_deg
        ra = np.empty(0, dtype=np.float64)
        dec = np.empty(0, dtype=np.float64)
        x_deg = np.empty(0, dtype=np.float32)
        y_deg = np.empty(0, dtype=np.float32)

    pre_mag_count = int(stars.size)
    if cfg.mag_cap is not None:
        mask_mag = stars["mag"] <= float(cfg.mag_cap)
        stars = stars[mask_mag]
        x_deg = x_deg[mask_mag]
        y_deg = y_deg[mask_mag]
        ra = ra[mask_mag]
        dec = dec[mask_mag]
    post_mag_count = int(stars.size)
    sweep_rank = np.arange(int(stars.size), dtype=np.int32)
    max_source = int(cfg.source_max_stars)
    if stars.size > max_source:
        if str(cfg.source_star_truncation_mode).strip().lower() == "brightest_mag":
            order = np.argsort(stars["mag"])[:max_source]
        else:
            order = np.arange(max_source, dtype=np.int64)
        stars = stars[order]
        x_deg = x_deg[order]
        y_deg = y_deg[order]
        ra = ra[order]
        dec = dec[order]
        sweep_rank = sweep_rank[order]
    post_source_limit_count = int(stars.size)
    if int(cfg.max_stars_per_tile) > 0:
        post_4d_limit_count = int(min(post_source_limit_count, int(cfg.max_stars_per_tile)))
    else:
        post_4d_limit_count = post_source_limit_count
    return Quad4DSourceTile(
        tile_key=tile_meta.key,
        family=tile_meta.family,
        tile_code=tile_meta.tile_code,
        center_ra_deg=float(center_ra_deg),
        center_dec_deg=float(center_dec_deg),
        ra_deg=stars["ra_deg"].astype(np.float64, copy=False),
        dec_deg=stars["dec_deg"].astype(np.float64, copy=False),
        mag=stars["mag"].astype(np.float32, copy=False),
        x_deg=x_deg.astype(np.float32, copy=False),
        y_deg=y_deg.astype(np.float32, copy=False),
        source_star_indices=sweep_rank.astype(np.int32, copy=False),
        pre_mag_count=pre_mag_count,
        post_mag_count=post_mag_count,
        post_source_limit_count=post_source_limit_count,
        post_4d_limit_count=post_4d_limit_count,
        tan_center_policy=str(cfg.tan_center_policy),
        used_layout_fallback=used_layout_fallback,
    )


def _canonical_source_fingerprint(tiles: Iterable[Quad4DSourceTile], config: Astap4DBuildConfig) -> str:
    h = hashlib.sha256()
    stable_config = {
        "family": config.family,
        "tile_keys": list(config.tile_keys),
        "mag_cap": config.mag_cap,
        "source_max_stars": int(config.source_max_stars),
        "source_star_truncation_mode": config.source_star_truncation_mode,
        "tan_center_policy": config.tan_center_policy,
        "max_stars_per_tile": int(config.max_stars_per_tile),
    }
    h.update(json.dumps(stable_config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for tile in sorted(tiles, key=lambda item: item.tile_key):
        h.update(tile.tile_key.encode("utf-8"))
        for arr in (tile.ra_deg, tile.dec_deg, tile.mag, tile.x_deg, tile.y_deg, tile.source_star_indices):
            contiguous = np.ascontiguousarray(arr)
            h.update(json.dumps({"shape": contiguous.shape, "dtype": str(contiguous.dtype)}, sort_keys=True).encode("utf-8"))
            h.update(contiguous.tobytes(order="C"))
    return h.hexdigest()


def _select_tile_metas(db_root: Path | str, config: Astap4DBuildConfig) -> list[TileMeta]:
    wanted = tuple(str(v) for v in config.tile_keys)
    if not wanted:
        raise ValueError("tile_keys must be provided explicitly")
    metas = [meta for meta in iter_tiles(db_root) if meta.family == str(config.family) and meta.key in set(wanted)]
    by_key = {meta.key: meta for meta in metas}
    missing = [key for key in wanted if key not in by_key]
    if missing:
        raise KeyError(f"tile(s) not found in ASTAP catalog: {missing}")
    return [by_key[key] for key in wanted]


def build_4d_index_from_astap(
    db_root: Path | str,
    out_path: Path | str,
    *,
    config: Astap4DBuildConfig,
    overwrite: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> Path:
    output = Path(out_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if config.quad_schema != ASTROMETRY_AB_CODE_4D_SCHEMA or int(config.quad_version) != ASTROMETRY_AB_CODE_4D_VERSION:
        raise ValueError("unsupported quad schema/version")
    materialization_config = config.materialization_config()
    metas = _select_tile_metas(db_root, config)
    source_tiles: list[Quad4DSourceTile] = []
    for ordinal, meta in enumerate(metas, start=1):
        if cancel_callback is not None and bool(cancel_callback()):
            raise RuntimeError("build_cancelled")
        tile = materialize_astap_tile_for_4d(db_root, meta, config=materialization_config)
        source_tiles.append(tile)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "materialized_tile",
                    "tile_key": tile.tile_key,
                    "ordinal": ordinal,
                    "total": len(metas),
                    "stars": tile.post_source_limit_count,
                }
            )
    source_fingerprint = _canonical_source_fingerprint(source_tiles, config)
    build_parameters = {
        "level": str(config.level),
        "mag_cap": config.mag_cap,
        "source_max_stars": int(config.source_max_stars),
        "source_star_truncation_mode": str(config.source_star_truncation_mode),
        "tan_center_policy": str(config.tan_center_policy),
        "max_stars_per_tile": int(config.max_stars_per_tile),
        "max_quads_per_tile": int(config.max_quads_per_tile),
        "sampler_tag": str(config.sampler_tag),
        "code_tol_recommended": float(config.code_tol_recommended),
        "dtype": str(config.dtype),
        "projection_implementation": str(config.projection_implementation),
        "projection_version": str(config.projection_version),
        "quad_schema": str(config.quad_schema),
        "quad_version": int(config.quad_version),
        "builder_version": str(config.builder_version),
    }
    metadata_extra = {
        "source_catalog": "astap_raw",
        "source_db_root": str(Path(db_root).expanduser().resolve()),
        "source_family": str(config.family),
        "source_tiles": [
            {
                "tile_key": tile.tile_key,
                "family": tile.family,
                "tile_code": tile.tile_code,
                "center_ra_deg": tile.center_ra_deg,
                "center_dec_deg": tile.center_dec_deg,
                "pre_mag_count": tile.pre_mag_count,
                "post_mag_count": tile.post_mag_count,
                "post_source_limit_count": tile.post_source_limit_count,
                "post_4d_limit_count": tile.post_4d_limit_count,
                "used_layout_fallback": tile.used_layout_fallback,
            }
            for tile in source_tiles
        ],
        "source_fingerprint": source_fingerprint,
        "build_parameters": build_parameters,
        "provenance_fingerprint": source_fingerprint,
        "builder_version": str(config.builder_version),
    }
    result = build_4d_index_from_payload_tiles(
        output,
        tiles=[tile.as_payload_tile() for tile in source_tiles],
        level=config.level,
        max_stars_per_tile=config.max_stars_per_tile,
        max_quads_per_tile=config.max_quads_per_tile,
        sampler_tag=config.sampler_tag,
        code_tol_recommended=config.code_tol_recommended,
        dtype=config.dtype,
        source_catalog="astap_raw",
        metadata_extra=metadata_extra,
    )
    if progress_callback is not None:
        progress_callback({"stage": "written", "path": str(result)})
    return result


__all__ = [
    "Astap4DBuildConfig",
    "AstapTileMaterializationConfig",
    "BUILDER_VERSION",
    "Quad4DSourceTile",
    "build_4d_index_from_astap",
    "materialize_astap_tile_for_4d",
]
