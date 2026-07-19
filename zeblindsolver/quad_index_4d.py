from __future__ import annotations

import hashlib
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
_SCIENTIFIC_METADATA_EXCLUDES = frozenset(
    {
        "generated_at",
        "source_index_root",
        "source_db_root",
        "source_paths",
        "out_path",
    }
)


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


@dataclass(frozen=True, slots=True)
class Quad4DPayloadTile:
    tile_key: str
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray
    x_deg: np.ndarray
    y_deg: np.ndarray


def _metadata_array(metadata: dict[str, Any]) -> np.ndarray:
    text = json.dumps(metadata, sort_keys=True)
    return np.asarray([text], dtype=f"<U{len(text)}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _scientific_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in metadata.items() if str(k) not in _SCIENTIFIC_METADATA_EXCLUDES}


def scientific_payload_fingerprint(index: Quad4DIndex | Path | str) -> str:
    """Return a deterministic scientific hash for a 4D payload.

    The fingerprint covers array names, shapes, dtypes and bytes plus canonical
    metadata with timestamps and machine-local source paths removed. It is
    intentionally independent of ZIP container metadata in the NPZ file.
    """

    loaded = index if isinstance(index, Quad4DIndex) else Quad4DIndex.load(index)
    arrays = {
        "codes_4d": loaded.codes_4d,
        "quad_star_indices": loaded.quad_star_indices,
        "source_quad_indices": loaded.source_quad_indices,
        "tile_key_indices": loaded.tile_key_indices,
        "ratio_hashes": loaded.ratio_hashes,
        "tile_keys": np.asarray(loaded.tile_keys, dtype=f"<U{max(1, max((len(v) for v in loaded.tile_keys), default=1))}"),
        "catalog_ra_dec": loaded.catalog_ra_dec,
        "catalog_xy": loaded.catalog_xy,
    }
    h = hashlib.sha256()
    h.update(_canonical_json(_scientific_metadata(loaded.metadata)))
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(_canonical_json({"name": name, "shape": arr.shape, "dtype": str(arr.dtype)}))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def compare_4d_indexes(left: Quad4DIndex | Path | str, right: Quad4DIndex | Path | str) -> dict[str, Any]:
    lhs = left if isinstance(left, Quad4DIndex) else Quad4DIndex.load(left)
    rhs = right if isinstance(right, Quad4DIndex) else Quad4DIndex.load(right)
    arrays = (
        "codes_4d",
        "quad_star_indices",
        "source_quad_indices",
        "tile_key_indices",
        "ratio_hashes",
        "catalog_ra_dec",
        "catalog_xy",
    )
    array_reports: dict[str, dict[str, Any]] = {}
    exact = True
    for name in arrays:
        a = np.asarray(getattr(lhs, name))
        b = np.asarray(getattr(rhs, name))
        report: dict[str, Any] = {
            "left_shape": list(a.shape),
            "right_shape": list(b.shape),
            "left_dtype": str(a.dtype),
            "right_dtype": str(b.dtype),
            "equal": False,
        }
        if a.shape == b.shape:
            equal = bool(np.array_equal(a, b))
            report["equal"] = equal
            if not equal:
                diff_mask = a != b
                if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
                    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    report["max_abs_diff"] = float(np.nanmax(diff)) if diff.size else 0.0
                    denom = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
                    rel = np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)
                    report["max_rel_diff"] = float(np.nanmax(rel)) if rel.size else 0.0
                coords = np.argwhere(diff_mask)
                if coords.size:
                    first = tuple(int(v) for v in coords[0])
                    report["first_difference_index"] = list(first)
                    report["left_value"] = np.asarray(a[first]).item()
                    report["right_value"] = np.asarray(b[first]).item()
        array_reports[name] = report
        exact = exact and bool(report["equal"])
    tile_keys_equal = lhs.tile_keys == rhs.tile_keys
    return {
        "exact": bool(exact and tile_keys_equal),
        "left_fingerprint": scientific_payload_fingerprint(lhs),
        "right_fingerprint": scientific_payload_fingerprint(rhs),
        "tile_keys_equal": bool(tile_keys_equal),
        "left_tile_keys": list(lhs.tile_keys),
        "right_tile_keys": list(rhs.tile_keys),
        "arrays": array_reports,
    }


def build_4d_index_from_payload_tiles(
    out_path: Path | str,
    *,
    tiles: Iterable[Quad4DPayloadTile],
    level: str = "S",
    max_stars_per_tile: int = 400,
    max_quads_per_tile: int = 8000,
    sampler_tag: str = "catalog_ring_coverage",
    code_tol_recommended: float = ASTROMETRY_AB_CODE_4D_DEFAULT_TOL,
    dtype: str = "float32",
    source_catalog: str,
    metadata_extra: dict[str, Any] | None = None,
) -> Path:
    output = Path(out_path).expanduser().resolve()
    tile_list = list(tiles)
    all_codes: list[np.ndarray] = []
    all_quads: list[np.ndarray] = []
    all_source_indices: list[np.ndarray] = []
    all_tile_indices: list[np.ndarray] = []
    all_ratio_hashes: list[np.ndarray] = []
    all_ra_dec: list[np.ndarray] = []
    all_xy: list[np.ndarray] = []
    tile_key_values: list[str] = []
    star_offset = 0
    for tile in tile_list:
        tile_key = str(tile.tile_key)
        x = np.asarray(tile.x_deg, dtype=np.float64)
        y = np.asarray(tile.y_deg, dtype=np.float64)
        ra = np.asarray(tile.ra_deg, dtype=np.float64)
        dec = np.asarray(tile.dec_deg, dtype=np.float64)
        mag = np.asarray(tile.mag, dtype=np.float64)
        if not (x.shape == y.shape == ra.shape == dec.shape == mag.shape):
            raise ValueError(f"tile {tile_key}: source arrays must have matching 1-D shapes")
        if x.ndim != 1:
            raise ValueError(f"tile {tile_key}: source arrays must be one-dimensional")
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
        emitted_tile_idx = len(tile_key_values)
        tile_key_values.append(tile_key)
        codes = np.vstack([record.code for record in records])
        local_quads = np.asarray([record.ordered_indices for record in records], dtype=np.int32)
        all_codes.append(codes.astype(dtype, copy=False))
        all_quads.append((local_quads + star_offset).astype(np.int32, copy=False))
        all_source_indices.append(np.asarray([record.source_quad_index for record in records], dtype=np.int32))
        all_tile_indices.append(np.full(len(records), emitted_tile_idx, dtype=np.int32))
        all_ratio_hashes.append(np.asarray([-1 if record.ratio_hash is None else int(record.ratio_hash) for record in records], dtype=np.int64))
        all_ra_dec.append(np.column_stack((ra[order], dec[order])).astype(np.float64, copy=False))
        all_xy.append(positions.astype(np.float64, copy=False))
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
        "source_catalog": str(source_catalog),
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
    if metadata_extra:
        metadata.update(dict(metadata_extra))
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
    tiles: list[Quad4DPayloadTile] = []
    for entry in entries:
        tile_key = str(entry.get("tile_key") or "")
        tile_path = root / str(entry.get("tile_file") or "")
        with np.load(tile_path, allow_pickle=False) as data:
            tiles.append(
                Quad4DPayloadTile(
                    tile_key=tile_key,
                    x_deg=np.asarray(data["x_deg"], dtype=np.float64),
                    y_deg=np.asarray(data["y_deg"], dtype=np.float64),
                    ra_deg=np.asarray(data["ra_deg"], dtype=np.float64),
                    dec_deg=np.asarray(data["dec_deg"], dtype=np.float64),
                    mag=np.asarray(data["mag"], dtype=np.float64),
                )
            )
    return build_4d_index_from_payload_tiles(
        output,
        tiles=tiles,
        level=level,
        max_stars_per_tile=max_stars_per_tile,
        max_quads_per_tile=max_quads_per_tile,
        sampler_tag=sampler_tag,
        code_tol_recommended=code_tol_recommended,
        dtype=dtype,
        source_catalog="tile_npz:x_deg/y_deg/ra_deg/dec_deg",
        metadata_extra={"source_index_root": str(root)},
    )
