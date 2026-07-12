from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

from .asterisms import sample_quads
from .quad_code_diagnostic import build_astrometry_quad_records

ASTROMETRY_AB_CODE_4D_SCHEMA = "astrometry_ab_code_4d_v1"
ASTROMETRY_AB_CODE_4D_VERSION = 1
ASTROMETRY_AB_CODE_4D_DEFAULT_TOL = 0.015


@dataclass(frozen=True, slots=True)
class Quad4DSearchHit:
    image_record_index: int
    catalog_record_index: int
    code_distance: float
    image_quad_indices: tuple[int, int, int, int]
    catalog_quad_indices: tuple[int, int, int, int]
    tile_key: str


@dataclass(frozen=True, slots=True)
class Quad4DIndex:
    path: Path
    metadata: dict[str, Any]
    codes_4d: np.ndarray
    quad_star_indices: np.ndarray
    source_quad_indices: np.ndarray
    tile_keys: tuple[str, ...]
    tile_key_indices: np.ndarray
    catalog_ra_dec: np.ndarray
    catalog_xy: np.ndarray
    ratio_hashes: np.ndarray
    tree: cKDTree | None

    @classmethod
    def load(cls, path: Path | str) -> "Quad4DIndex":
        index_path = Path(path).expanduser().resolve()
        with np.load(index_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"][0]))
            schema = str(metadata.get("schema", ""))
            if schema != ASTROMETRY_AB_CODE_4D_SCHEMA:
                raise RuntimeError(f"quad 4D schema {schema!r} incompatible with {ASTROMETRY_AB_CODE_4D_SCHEMA}")
            codes = np.asarray(data["codes_4d"], dtype=np.float64)
            tile_keys = tuple(str(v) for v in data["tile_keys"].astype(str).tolist())
            index = cls(
                path=index_path,
                metadata=metadata,
                codes_4d=codes,
                quad_star_indices=np.asarray(data["quad_star_indices"], dtype=np.int32),
                source_quad_indices=np.asarray(data["source_quad_indices"], dtype=np.int32),
                tile_keys=tile_keys,
                tile_key_indices=np.asarray(data["tile_key_indices"], dtype=np.int32),
                catalog_ra_dec=np.asarray(data["catalog_ra_dec"], dtype=np.float64),
                catalog_xy=np.asarray(data["catalog_xy"], dtype=np.float64),
                ratio_hashes=np.asarray(data["ratio_hashes"], dtype=np.int64) if "ratio_hashes" in data else np.full(codes.shape[0], -1, dtype=np.int64),
                tree=cKDTree(codes) if codes.shape[0] else None,
            )
        return index

    def search_records(
        self,
        image_records: Iterable[Any],
        *,
        code_tol: float | None = None,
        max_hits: int = 2000,
        max_hits_per_image_quad: int = 8,
    ) -> list[Quad4DSearchHit]:
        if self.tree is None:
            return []
        tol = float(self.metadata.get("code_tol_recommended", ASTROMETRY_AB_CODE_4D_DEFAULT_TOL) if code_tol is None else code_tol)
        seen: set[tuple[int, int]] = set()
        hits: list[Quad4DSearchHit] = []
        for image_record_index, image_record in enumerate(image_records):
            code = np.asarray(image_record.code, dtype=np.float64)
            neighbor_ids = self.tree.query_ball_point(code, r=tol)
            if not neighbor_ids:
                continue
            ranked = sorted(
                ((int(idx), float(np.linalg.norm(code - self.codes_4d[int(idx)]))) for idx in neighbor_ids),
                key=lambda item: item[1],
            )
            for catalog_record_index, distance in ranked[: max(0, int(max_hits_per_image_quad))]:
                key = (int(image_record.source_quad_index), int(catalog_record_index))
                if key in seen:
                    continue
                seen.add(key)
                tile_idx = int(self.tile_key_indices[catalog_record_index])
                tile_key = self.tile_keys[tile_idx] if 0 <= tile_idx < len(self.tile_keys) else ""
                hits.append(
                    Quad4DSearchHit(
                        image_record_index=int(image_record_index),
                        catalog_record_index=int(catalog_record_index),
                        code_distance=float(distance),
                        image_quad_indices=tuple(int(v) for v in image_record.ordered_indices),
                        catalog_quad_indices=tuple(int(v) for v in self.quad_star_indices[catalog_record_index]),
                        tile_key=str(tile_key),
                    )
                )
                if len(hits) >= int(max_hits):
                    return hits
        return hits


def _metadata_array(metadata: dict[str, Any]) -> np.ndarray:
    text = json.dumps(metadata, sort_keys=True)
    return np.asarray([text], dtype=f"<U{len(text)}")


def _load_manifest(index_root: Path) -> dict[str, Any]:
    return json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))


def _tile_entries(index_root: Path, tile_keys: Iterable[str]) -> list[dict[str, Any]]:
    wanted = {str(v) for v in tile_keys}
    manifest = _load_manifest(index_root)
    entries: list[dict[str, Any]] = []
    for entry in list(manifest.get("tiles") or []):
        if str(entry.get("tile_key") or "") in wanted:
            entries.append(dict(entry))
    missing = sorted(wanted - {str(entry.get("tile_key") or "") for entry in entries})
    if missing:
        raise KeyError(f"tile(s) not found in manifest: {missing}")
    return entries


def build_experimental_4d_index(
    index_root: Path | str,
    out_path: Path | str,
    *,
    tile_keys: Iterable[str],
    level: str = "S",
    max_stars_per_tile: int = 400,
    max_quads_per_tile: int = 8000,
    sampler_tag: str = "catalog_ring_coverage",
    code_tol_recommended: float = ASTROMETRY_AB_CODE_4D_DEFAULT_TOL,
    dtype: str = "float32",
) -> Path:
    root = Path(index_root).expanduser().resolve()
    output = Path(out_path).expanduser().resolve()
    entries = _tile_entries(root, tile_keys)
    all_codes: list[np.ndarray] = []
    all_quads: list[np.ndarray] = []
    all_source_indices: list[np.ndarray] = []
    all_tile_indices: list[np.ndarray] = []
    all_ratio_hashes: list[np.ndarray] = []
    all_ra_dec: list[np.ndarray] = []
    all_xy: list[np.ndarray] = []
    tile_key_values: list[str] = []
    star_offset = 0
    for tile_idx, entry in enumerate(entries):
        tile_key = str(entry.get("tile_key") or "")
        tile_path = root / str(entry.get("tile_file") or "")
        with np.load(tile_path, allow_pickle=False) as data:
            x = np.asarray(data["x_deg"], dtype=np.float64)
            y = np.asarray(data["y_deg"], dtype=np.float64)
            ra = np.asarray(data["ra_deg"], dtype=np.float64)
            dec = np.asarray(data["dec_deg"], dtype=np.float64)
            mag = np.asarray(data["mag"], dtype=np.float64)
        order = np.argsort(mag, kind="stable")
        if max_stars_per_tile > 0:
            order = order[: int(max_stars_per_tile)]
        positions = np.column_stack((x[order], y[order])).astype(np.float64, copy=False)
        stars = np.zeros(order.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
        stars["x"] = positions[:, 0].astype(np.float32)
        stars["y"] = positions[:, 1].astype(np.float32)
        stars["mag"] = mag[order].astype(np.float32)
        quads = sample_quads(stars, max_quads=int(max_quads_per_tile), strategy=str(sampler_tag))
        records = build_astrometry_quad_records(quads, positions)
        if not records:
            continue
        codes = np.vstack([record.code for record in records])
        local_quads = np.asarray([record.ordered_indices for record in records], dtype=np.int32)
        all_codes.append(codes.astype(dtype, copy=False))
        all_quads.append((local_quads + star_offset).astype(np.int32, copy=False))
        all_source_indices.append(np.asarray([record.source_quad_index for record in records], dtype=np.int32))
        all_tile_indices.append(np.full(len(records), tile_idx, dtype=np.int32))
        all_ratio_hashes.append(np.asarray([-1 if record.ratio_hash is None else int(record.ratio_hash) for record in records], dtype=np.int64))
        all_ra_dec.append(np.column_stack((ra[order], dec[order])).astype(np.float64, copy=False))
        all_xy.append(positions.astype(np.float64, copy=False))
        tile_key_values.append(tile_key)
        star_offset += int(order.shape[0])
    if not all_codes:
        raise RuntimeError("no 4D codes generated")
    codes_4d = np.vstack(all_codes)
    quad_star_indices = np.vstack(all_quads)
    source_quad_indices = np.concatenate(all_source_indices)
    tile_key_indices = np.concatenate(all_tile_indices)
    ratio_hashes = np.concatenate(all_ratio_hashes)
    catalog_ra_dec = np.vstack(all_ra_dec)
    catalog_xy = np.vstack(all_xy)
    metadata = {
        "schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "version": ASTROMETRY_AB_CODE_4D_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_index_root": str(root),
        "source_catalog": "tile_npz:x_deg/y_deg/ra_deg/dec_deg",
        "level": str(level),
        "sampler_tag": str(sampler_tag),
        "tile_keys": list(tile_key_values),
        "max_stars_per_tile": int(max_stars_per_tile),
        "max_quads_per_tile": int(max_quads_per_tile),
        "code_tol_recommended": float(code_tol_recommended),
        "dtype": str(dtype),
        "entry_count": int(codes_4d.shape[0]),
        "star_count": int(catalog_ra_dec.shape[0]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        codes_4d=codes_4d,
        quad_star_indices=quad_star_indices,
        source_quad_indices=source_quad_indices,
        tile_key_indices=tile_key_indices,
        ratio_hashes=ratio_hashes,
        tile_keys=np.asarray(tile_key_values, dtype=f"<U{max(1, max(len(v) for v in tile_key_values))}"),
        catalog_ra_dec=catalog_ra_dec,
        catalog_xy=catalog_xy,
        metadata=_metadata_array(metadata),
    )
    return output
