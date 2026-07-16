from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex


MANIFEST_SCHEMA = "zeblind.astrometry_4d_index_manifest.v1"
MANIFEST_VERSION = 1


class IndexManifestError(RuntimeError):
    """Base class for strict 4D index manifest failures."""


class IndexManifestSchemaError(IndexManifestError):
    """Raised when the manifest JSON shape or version is invalid."""


class IndexManifestIntegrityError(IndexManifestError):
    """Raised when paths, checksums or duplicate constraints fail."""


class IndexManifestCompatibilityError(IndexManifestError):
    """Raised when an index file is incompatible with its manifest entry."""


@dataclass(frozen=True)
class Loaded4DIndexEntry:
    id: str
    path: Path
    filename: str
    quad_schema: str
    index_version: int
    level: str
    tile_keys: tuple[str, ...]
    star_count: int
    quad_count: int
    sampler_tag: str
    sha256: str
    code_tol_recommended: float | None
    catalog_source: str | None
    metadata: dict[str, Any]
    manifest_entry: dict[str, Any]


@dataclass(frozen=True)
class Loaded4DManifest:
    manifest_path: Path
    schema: str
    manifest_version: int
    entries: tuple[Loaded4DIndexEntry, ...]

    @property
    def enabled_index_paths(self) -> tuple[Path, ...]:
        return tuple(entry.path for entry in self.entries)

    @property
    def index_ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    @property
    def tile_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for entry in self.entries:
            keys.extend(entry.tile_keys)
        return tuple(keys)

    @property
    def checksums(self) -> dict[str, str]:
        return {entry.id: entry.sha256 for entry in self.entries}

    @property
    def metadata(self) -> dict[str, dict[str, Any]]:
        return {entry.id: dict(entry.metadata) for entry in self.entries}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IndexManifestSchemaError(f"manifest_absent: {path}") from exc
    except Exception as exc:
        raise IndexManifestSchemaError(f"manifest_json_invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise IndexManifestSchemaError("manifest_json_invalid: expected object")
    return payload


def _resolve_manifest_path(path_value: object, *, manifest_path: Path, index_root: Path | None) -> Path:
    raw_text = str(path_value or "").strip()
    if not raw_text:
        raise IndexManifestSchemaError("manifest_entry_path_missing")
    raw = Path(raw_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    base = index_root.expanduser().resolve() if index_root is not None else manifest_path.parent
    return (base / raw).resolve()


def _as_int(value: object, *, field: str, entry_id: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise IndexManifestSchemaError(f"manifest_{field}_invalid: {entry_id}") from exc


def _as_tile_tuple(value: object, *, entry_id: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise IndexManifestSchemaError(f"manifest_tile_keys_invalid: {entry_id}")
    tiles = tuple(str(v).strip() for v in value if str(v).strip())
    if not tiles:
        raise IndexManifestSchemaError(f"manifest_tile_keys_invalid: {entry_id}")
    return tiles


def load_4d_index_manifest(path: Path | str, *, index_root: Path | str | None = None) -> Loaded4DManifest:
    manifest_path = Path(path).expanduser().resolve()
    root = Path(index_root).expanduser().resolve() if index_root is not None else None
    payload = _read_json(manifest_path)
    schema = str(payload.get("schema") or "")
    if schema != MANIFEST_SCHEMA:
        raise IndexManifestSchemaError(f"manifest_schema_invalid: {schema!r}")
    version = _as_int(payload.get("manifest_version", -1), field="version", entry_id="<manifest>")
    if version != MANIFEST_VERSION:
        raise IndexManifestSchemaError(f"manifest_version_invalid: {version!r}")
    raw_entries = payload.get("indexes")
    if not isinstance(raw_entries, list):
        raise IndexManifestSchemaError("manifest_indexes_invalid: expected list")

    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_tiles: set[str] = set()
    entries: list[Loaded4DIndexEntry] = []
    for i, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise IndexManifestSchemaError(f"manifest_entry_invalid[{i}]: expected object")
        entry_id = str(raw_entry.get("id") or "").strip()
        if not entry_id:
            raise IndexManifestSchemaError(f"manifest_entry_id_missing[{i}]")
        if entry_id in seen_ids:
            raise IndexManifestIntegrityError(f"manifest_duplicate_id: {entry_id}")
        seen_ids.add(entry_id)
        if not bool(raw_entry.get("enabled", True)):
            continue

        index_path = _resolve_manifest_path(raw_entry.get("path"), manifest_path=manifest_path, index_root=root)
        if index_path in seen_paths:
            raise IndexManifestIntegrityError(f"manifest_duplicate_path: {index_path}")
        seen_paths.add(index_path)
        if not index_path.exists():
            raise IndexManifestIntegrityError(f"manifest_index_absent: {index_path}")
        expected_sha = str(raw_entry.get("sha256") or "").strip().lower()
        actual_sha = sha256_file(index_path).lower()
        if expected_sha and expected_sha != actual_sha:
            raise IndexManifestIntegrityError(f"manifest_sha256_mismatch: {entry_id}")
        try:
            index = Quad4DIndex.load(index_path)
        except Exception as exc:
            raise IndexManifestCompatibilityError(f"manifest_index_incompatible: {entry_id}: {exc}") from exc

        metadata = dict(index.metadata)
        expected_schema = str(raw_entry.get("quad_schema") or "")
        if expected_schema != ASTROMETRY_AB_CODE_4D_SCHEMA:
            raise IndexManifestCompatibilityError(f"manifest_quad_schema_invalid: {entry_id}: {expected_schema!r}")
        if str(metadata.get("schema") or "") != ASTROMETRY_AB_CODE_4D_SCHEMA:
            raise IndexManifestCompatibilityError(f"manifest_index_schema_invalid: {entry_id}: {metadata.get('schema')!r}")
        expected_version = _as_int(raw_entry.get("index_version", -1), field="index_version", entry_id=entry_id)
        actual_version = int(metadata.get("version", -1))
        if actual_version != expected_version:
            raise IndexManifestCompatibilityError(f"manifest_index_version_mismatch: {entry_id}")
        expected_tiles = _as_tile_tuple(raw_entry.get("tile_keys"), entry_id=entry_id)
        actual_tiles = tuple(str(v) for v in index.tile_keys)
        if expected_tiles != actual_tiles:
            raise IndexManifestCompatibilityError(f"manifest_tile_keys_mismatch: {entry_id}")
        for tile in actual_tiles:
            if tile in seen_tiles:
                raise IndexManifestIntegrityError(f"manifest_duplicate_tile: {tile}")
            seen_tiles.add(tile)
        star_count = int(index.catalog_ra_dec.shape[0])
        quad_count = int(index.codes_4d.shape[0])
        if _as_int(raw_entry.get("star_count", -1), field="star_count", entry_id=entry_id) != star_count:
            raise IndexManifestCompatibilityError(f"manifest_star_count_mismatch: {entry_id}")
        if _as_int(raw_entry.get("quad_count", -1), field="quad_count", entry_id=entry_id) != quad_count:
            raise IndexManifestCompatibilityError(f"manifest_quad_count_mismatch: {entry_id}")
        sampler = str(raw_entry.get("sampler_tag") or "")
        actual_sampler = str(metadata.get("sampler_tag") or "")
        if sampler != actual_sampler:
            raise IndexManifestCompatibilityError(f"manifest_sampler_mismatch: {entry_id}")

        manifest_entry = dict(raw_entry)
        manifest_entry["resolved_path"] = str(index_path)
        manifest_entry["actual_sha256"] = actual_sha
        entries.append(
            Loaded4DIndexEntry(
                id=entry_id,
                path=index_path,
                filename=str(raw_entry.get("filename") or index_path.name),
                quad_schema=expected_schema,
                index_version=actual_version,
                level=str(raw_entry.get("level") or metadata.get("level") or ""),
                tile_keys=actual_tiles,
                star_count=star_count,
                quad_count=quad_count,
                sampler_tag=actual_sampler,
                sha256=actual_sha,
                code_tol_recommended=(
                    float(raw_entry["code_tol_recommended"])
                    if raw_entry.get("code_tol_recommended") is not None
                    else None
                ),
                catalog_source=(str(raw_entry.get("catalog_source")) if raw_entry.get("catalog_source") is not None else None),
                metadata=metadata,
                manifest_entry=manifest_entry,
            )
        )

    return Loaded4DManifest(
        manifest_path=manifest_path,
        schema=schema,
        manifest_version=version,
        entries=tuple(entries),
    )


__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "IndexManifestCompatibilityError",
    "IndexManifestError",
    "IndexManifestIntegrityError",
    "IndexManifestSchemaError",
    "Loaded4DIndexEntry",
    "Loaded4DManifest",
    "load_4d_index_manifest",
    "sha256_file",
]
