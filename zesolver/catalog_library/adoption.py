"""Non-destructive adoption planning for existing catalogue installations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from zeblindsolver.index_manifest_4d import IndexManifestError, load_4d_index_manifest, sha256_file
from zewcs290.catalog290 import FAMILY_SPECS
from zewcs290.layouts import get_layout

from .coverage import coverage_for_tiles, merge_coverages
from .models import (
    CatalogCompatibilityResource,
    CatalogCoverage,
    CatalogDataStatus,
    CatalogIndex,
    CatalogIssue,
    CatalogLibraryAdoptionPlanResult,
    CatalogLibraryAdoptionTelemetry,
    CatalogRepairAction,
    CatalogSource,
    CatalogStatus,
    CoverageStatus,
    FingerprintPolicy,
    IntegrityFile,
    IssueSeverity,
    ParameterCompleteness,
    PathKind,
    PathReference,
    SourceShard,
)

ProgressCallback = Callable[[int, int, Path], None]

REPAIR_VERIFY_SOURCE_SHA256 = "VERIFY_SOURCE_SHA256"
REPAIR_LOCATE_MISSING_SOURCE = "LOCATE_MISSING_SOURCE"
REPAIR_LOCATE_MISSING_INDEX = "LOCATE_MISSING_INDEX"
REPAIR_REBUILD_BLIND4D_INDEX = "REBUILD_BLIND4D_INDEX"
REPAIR_REGENERATE_STRICT_4D_MANIFEST = "REGENERATE_STRICT_4D_MANIFEST"
REPAIR_REMOVE_STALE_COMPATIBILITY_REFERENCE = "REMOVE_STALE_COMPATIBILITY_REFERENCE"
REPAIR_RECOMPUTE_COVERAGE = "RECOMPUTE_COVERAGE"

_SCIENTIFIC_TIMESTAMP_KEYS = {"generated_at", "created_at", "mtime_ns", "mtime", "last_modified"}


class CatalogLibraryAdoptionPlan:
    """Factory for read-only adoption plans.

    The planner inspects explicit paths and returns an in-memory manifest
    preview.  It never writes, moves, copies, deletes or rebuilds catalogue
    resources.
    """

    @classmethod
    def reference_existing(
        cls,
        *,
        library_root: str | Path,
        astap_roots: str | Path | Iterable[str | Path] | None = None,
        blind4d_manifest: str | Path | None = None,
        legacy_index_root: str | Path | None = None,
        fingerprint_policy: str | FingerprintPolicy = FingerprintPolicy.FAST,
        include_legacy: bool = True,
        progress_callback: ProgressCallback | None = None,
        generated_at: str | None = None,
    ) -> CatalogLibraryAdoptionPlanResult:
        policy = _coerce_policy(fingerprint_policy)
        root = Path(library_root).expanduser().resolve()
        warnings: list[CatalogIssue] = []
        errors: list[CatalogIssue] = []
        repair_actions: list[CatalogRepairAction] = []
        source_hashed_count = 0
        index_hashed_count = 0

        sources: list[CatalogSource] = []
        source_tiles: dict[str, set[str]] = {}
        for root_index, astap_root in enumerate(_iter_roots(astap_roots)):
            discovered, hashed = _discover_astap_root(
                astap_root,
                root_index=root_index,
                policy=policy,
                progress_callback=progress_callback,
                warnings=warnings,
                errors=errors,
                repair_actions=repair_actions,
            )
            source_hashed_count += hashed
            sources.extend(discovered)
            for source in discovered:
                source_tiles.setdefault(source.family, set()).update(
                    shard.tile_code or "" for shard in source.shards if shard.tile_code
                )

        indexes: list[CatalogIndex] = []
        if blind4d_manifest is not None:
            discovered, hashed = _discover_blind4d_manifest(
                blind4d_manifest,
                sources=tuple(sources),
                source_tiles=source_tiles,
                warnings=warnings,
                errors=errors,
                repair_actions=repair_actions,
            )
            indexes.extend(discovered)
            index_hashed_count += hashed

        compatibility_resources: tuple[CatalogCompatibilityResource, ...] = ()
        if legacy_index_root is not None and include_legacy:
            compatibility_resources = (
                _compatibility_resource(legacy_index_root, warnings=warnings, repair_actions=repair_actions),
            )

        coverage = _adoption_coverage(tuple(sources), tuple(indexes))
        status = _adoption_status(tuple(sources), tuple(indexes), tuple(errors), coverage)
        manifest_preview = _manifest_preview(
            library_root=root,
            status=status,
            sources=tuple(sources),
            indexes=tuple(indexes),
            compatibility_resources=compatibility_resources,
            coverage=coverage,
            fingerprint_policy=policy,
            generated_at=generated_at,
        )
        telemetry = CatalogLibraryAdoptionTelemetry(
            fingerprint_policy=policy,
            source_file_count=sum(len(source.shards) for source in sources),
            source_hashed_count=source_hashed_count,
            index_file_count=sum(len(index.derived_files) or len(index.integrity_files) for index in indexes),
            index_hashed_count=index_hashed_count,
            builder_called=False,
            files_written=0,
        )
        return CatalogLibraryAdoptionPlanResult(
            library_root=root,
            status=status,
            sources=tuple(sources),
            indexes=tuple(indexes),
            compatibility_resources=compatibility_resources,
            coverage=coverage,
            warnings=tuple(warnings),
            errors=tuple(errors),
            repair_actions=tuple(repair_actions),
            manifest_preview=manifest_preview,
            telemetry=telemetry,
        )


def canonical_provenance_fingerprint(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA256 over scientific provenance fields only."""

    return hashlib.sha256(_canonical_json(_strip_non_scientific(payload)).encode("utf-8")).hexdigest()


def _discover_astap_root(
    astap_root: str | Path,
    *,
    root_index: int,
    policy: FingerprintPolicy,
    progress_callback: ProgressCallback | None,
    warnings: list[CatalogIssue],
    errors: list[CatalogIssue],
    repair_actions: list[CatalogRepairAction],
) -> tuple[list[CatalogSource], int]:
    root = Path(astap_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        errors.append(_issue("SOURCE_PATH_MISSING", IssueSeverity.ERROR, root, "astap"))
        repair_actions.append(
            _repair(
                REPAIR_LOCATE_MISSING_SOURCE,
                IssueSeverity.ERROR,
                "astap",
                f"ASTAP root is missing: {root}",
                ("choose_existing_astap_root",),
                "P1D-2B",
                automatic=False,
            )
        )
        return [], 0
    sources: list[CatalogSource] = []
    hashed = 0
    suffix = "" if root_index == 0 else f"-{root_index + 1}"
    for family, spec in sorted(FAMILY_SPECS.items()):
        files = sorted(root.glob(spec.glob_pattern()), key=lambda path: path.name)
        if not files:
            continue
        total_tiles = _layout_tile_count(family)
        shards: list[SourceShard] = []
        for index, path in enumerate(files, start=1):
            sha = None
            if policy is FingerprintPolicy.FULL:
                if progress_callback is not None:
                    progress_callback(index, len(files), path)
                sha = sha256_file(path)
                hashed += 1
            stat = path.stat()
            shards.append(
                SourceShard(
                    path=path.name,
                    family=family,
                    tile_code=_tile_code_from_path(path),
                    size_bytes=int(stat.st_size),
                    sha256=sha,
                    mtime_ns=int(stat.st_mtime_ns),
                    status=CatalogDataStatus.FULL_VERIFIED if sha else CatalogDataStatus.FAST_VERIFIED,
                    resolved_path=path,
                )
            )
        coverage = CatalogCoverage(
            status=CoverageStatus.FULL if len(files) >= total_tiles else CoverageStatus.PARTIAL,
            all_sky=len(files) >= total_tiles,
            families=(family,),
            tile_keys=tuple(f"{family}_{shard.tile_code}" for shard in shards if shard.tile_code),
            dec_min_deg=-90.0 if len(files) >= total_tiles else None,
            dec_max_deg=90.0 if len(files) >= total_tiles else None,
            ra_segments_deg=((0.0, 360.0),) if len(files) >= total_tiles else (),
            covered_tiles=len(files),
            total_tiles=total_tiles,
            fraction=(len(files) / total_tiles) if total_tiles else None,
            provenance="adoption-source-scan",
        )
        fingerprint_payload = {
            "family": family,
            "format": spec.format_name,
            "layout": _layout_name(family),
            "policy": policy.value,
            "files": [
                {"path": shard.path, "size_bytes": shard.size_bytes, "sha256": shard.sha256}
                for shard in shards
            ],
        }
        sources.append(
            CatalogSource(
                id=f"astap-{family}{suffix}",
                kind="astap_hnsky",
                family=family,
                format=spec.format_name,
                path=PathReference(PathKind.EXTERNAL_REFERENCE, str(root), root),
                tile_count=len(files),
                layout=_layout_name(family),
                coverage=coverage,
                integrity_files=tuple(
                    IntegrityFile(path=str(shard.resolved_path), sha256=shard.sha256, size_bytes=shard.size_bytes, resolved_path=shard.resolved_path)
                    for shard in shards
                ),
                status=CatalogDataStatus.FULL_VERIFIED if policy is FingerprintPolicy.FULL else CatalogDataStatus.FAST_VERIFIED,
                provenance={
                    "adoption_mode": "external-reference",
                    "layout_fingerprint": canonical_provenance_fingerprint(fingerprint_payload),
                    "fingerprint_policy": policy.value,
                },
                families=(family,),
                mode="external-reference",
                path_policy="external_reference",
                shards=tuple(shards),
                layout_fingerprint=canonical_provenance_fingerprint(fingerprint_payload),
                reader_version="zewcs290.catalog290",
                fingerprint_policy=policy,
            )
        )
        if policy is FingerprintPolicy.FAST:
            repair_actions.append(
                _repair(
                    REPAIR_VERIFY_SOURCE_SHA256,
                    IssueSeverity.INFO,
                    f"astap-{family}{suffix}",
                    "FAST adoption did not compute cryptographic hashes for source shards.",
                    ("source_files_present",),
                    "P1D-2B",
                    automatic=False,
                )
            )
    return sources, hashed


def _discover_blind4d_manifest(
    manifest_path: str | Path,
    *,
    sources: tuple[CatalogSource, ...],
    source_tiles: dict[str, set[str]],
    warnings: list[CatalogIssue],
    errors: list[CatalogIssue],
    repair_actions: list[CatalogRepairAction],
) -> tuple[list[CatalogIndex], int]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        errors.append(_issue("BLIND4D_MANIFEST_MISSING", IssueSeverity.ERROR, path, "blind4d"))
        repair_actions.append(
            _repair(
                REPAIR_LOCATE_MISSING_INDEX,
                IssueSeverity.ERROR,
                "blind4d",
                f"Strict Blind 4D manifest is missing: {path}",
                ("choose_existing_4d_manifest",),
                "P1D-2B",
                automatic=False,
            )
        )
        return [], 0
    try:
        loaded = load_4d_index_manifest(path)
    except IndexManifestError as exc:
        errors.append(_issue("BLIND4D_MANIFEST_INVALID", IssueSeverity.ERROR, path, "blind4d", message=str(exc)))
        repair_actions.append(
            _repair(
                REPAIR_REGENERATE_STRICT_4D_MANIFEST,
                IssueSeverity.ERROR,
                "blind4d",
                str(exc),
                ("valid_index_npz_files",),
                "P1D-3_OR_LATER",
                automatic=False,
            )
        )
        return [], 0

    source_ids_by_family: dict[str, tuple[str, ...]] = {}
    for source in sources:
        source_ids_by_family.setdefault(source.family, tuple())
        source_ids_by_family[source.family] = (*source_ids_by_family[source.family], source.id)

    indexes: list[CatalogIndex] = []
    for entry in loaded.entries:
        family = _family_from_tiles(entry.tile_keys)
        missing_tiles = tuple(
            tile for tile in entry.tile_keys if family and tile.split("_", 1)[-1] not in source_tiles.get(family, set())
        )
        if missing_tiles and family in source_ids_by_family:
            warnings.append(_issue("INDEX_SOURCE_TILE_ABSENT", IssueSeverity.WARNING, entry.path, entry.id))
            repair_actions.append(
                _repair(
                    REPAIR_LOCATE_MISSING_SOURCE,
                    IssueSeverity.WARNING,
                    entry.id,
                    f"Index references source tiles absent from discovered ASTAP root: {', '.join(missing_tiles)}",
                    ("complete_astap_family",),
                    "P1D-2B",
                    automatic=False,
                )
            )
        source_ids = source_ids_by_family.get(family or "", ())
        if not source_ids:
            warnings.append(_issue("INDEX_SOURCE_UNKNOWN", IssueSeverity.WARNING, entry.path, entry.id))
            repair_actions.append(
                _repair(
                    REPAIR_LOCATE_MISSING_SOURCE,
                    IssueSeverity.WARNING,
                    entry.id,
                    "Blind 4D index has no discovered ASTAP source family.",
                    ("select_astap_root",),
                    "P1D-2B",
                    automatic=False,
                )
            )
        parameters = _build_parameters(entry.manifest_entry, entry.metadata)
        parameter_status = _parameter_status(parameters)
        if parameter_status is not ParameterCompleteness.KNOWN:
            warnings.append(_issue("INDEX_PARAMETERS_INCOMPLETE", IssueSeverity.WARNING, entry.path, entry.id))
            repair_actions.append(
                _repair(
                    REPAIR_REBUILD_BLIND4D_INDEX,
                    IssueSeverity.WARNING,
                    entry.id,
                    "Historical index metadata is incomplete for deterministic rebuild.",
                    ("complete_source_provenance", "builder_available"),
                    "P1D-3_OR_LATER",
                    automatic=False,
                )
            )
        fingerprint_payload = {
            "engine": "blind4d",
            "schema": entry.quad_schema,
            "source_ids": sorted(source_ids),
            "source_tiles": sorted(entry.tile_keys),
            "families": tuple(filter(None, (family,))),
            "parameters": parameters,
            "derived_files": [{"filename": entry.filename, "sha256": entry.sha256, "size_bytes": entry.path.stat().st_size}],
        }
        coverage = coverage_for_tiles(
            family=family or "unknown",
            tile_keys=entry.tile_keys,
            total_tiles=_layout_tile_count(family) if family else None,
            provenance="adoption-blind4d",
        )
        indexes.append(
            CatalogIndex(
                id=entry.id,
                engine="blind4d",
                schema=entry.quad_schema,
                path=PathReference(PathKind.EXTERNAL_REFERENCE, str(entry.path), entry.path),
                manifest_path=PathReference(PathKind.EXTERNAL_REFERENCE, str(path), path),
                source_ids=source_ids,
                source_tiles=entry.tile_keys,
                coverage=coverage,
                integrity_files=(IntegrityFile(path=str(entry.path), sha256=entry.sha256, size_bytes=entry.path.stat().st_size, resolved_path=entry.path),),
                status=CatalogDataStatus.FULL_VERIFIED,
                algorithm_version=entry.quad_schema,
                parameters=parameters,
                compatibility={
                    "strict_manifest": str(path),
                    "strict_manifest_sha256": sha256_file(path),
                    "runtime_loader": "zeblindsolver.index_manifest_4d.load_4d_index_manifest",
                },
                category="product",
                families=tuple(filter(None, (family,))),
                derived_files=(IntegrityFile(path=str(entry.path), sha256=entry.sha256, size_bytes=entry.path.stat().st_size, resolved_path=entry.path),),
                source_file_refs=tuple(entry.tile_keys),
                build_parameters=parameters,
                parameter_status=parameter_status,
                provenance_fingerprint=canonical_provenance_fingerprint(fingerprint_payload),
                reconstruction_status="KNOWN" if parameter_status is ParameterCompleteness.KNOWN and source_ids else "PARTIAL",
            )
        )
    if indexes and not merge_coverages(index.coverage for index in indexes).all_sky:
        warnings.append(_issue("BLIND4D_COVERAGE_PARTIAL", IssueSeverity.WARNING, path, "blind4d"))
        repair_actions.append(
            _repair(
                REPAIR_RECOMPUTE_COVERAGE,
                IssueSeverity.INFO,
                "blind4d",
                "Blind 4D coverage is partial and must remain explicit.",
                ("complete_index_inventory",),
                "P1D-2B",
                automatic=True,
            )
        )
    return indexes, len(indexes)


def _compatibility_resource(
    legacy_index_root: str | Path,
    *,
    warnings: list[CatalogIssue],
    repair_actions: list[CatalogRepairAction],
) -> CatalogCompatibilityResource:
    root = Path(legacy_index_root).expanduser().resolve()
    manifest = root / "manifest.json"
    status = CatalogDataStatus.FAST_VERIFIED if root.exists() and manifest.exists() else CatalogDataStatus.MISSING
    if status is CatalogDataStatus.MISSING:
        warnings.append(_issue("LEGACY_INDEX_REFERENCE_STALE", IssueSeverity.WARNING, root, "legacy-index"))
        repair_actions.append(
            _repair(
                REPAIR_REMOVE_STALE_COMPATIBILITY_REFERENCE,
                IssueSeverity.WARNING,
                "legacy-index",
                "Legacy compatibility index root is missing or lacks manifest.json.",
                ("confirm_legacy_reference_unused",),
                "P1D-2B",
                automatic=False,
            )
        )
    files: list[IntegrityFile] = []
    if root.exists():
        for rel in ("manifest.json", "tiles", "hash_tables"):
            path = root / rel
            if path.exists():
                files.append(IntegrityFile(path=rel, size_bytes=path.stat().st_size if path.is_file() else None, resolved_path=path))
    return CatalogCompatibilityResource(
        id="legacy-index",
        category="compatibility",
        path=root,
        manifest_path=manifest if manifest.exists() else None,
        files=tuple(files),
        status=status,
        notes="Historical index retained only for rollback/diagnostics; not a normal source or Blind 4D coverage proof.",
    )


def _manifest_preview(
    *,
    library_root: Path,
    status: CatalogStatus,
    sources: tuple[CatalogSource, ...],
    indexes: tuple[CatalogIndex, ...],
    compatibility_resources: tuple[CatalogCompatibilityResource, ...],
    coverage: CatalogCoverage,
    fingerprint_policy: FingerprintPolicy,
    generated_at: str | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "library_id": "adopted-existing",
        "created_at": generated_at,
        "created_by": "CatalogLibraryAdoptionPlan.reference_existing",
        "minimum_zesolver_version": None,
        "status": status.value,
        "sources": [_source_payload(source) for source in sorted(sources, key=lambda item: item.id)],
        "derived_indexes": [_index_payload(index) for index in sorted(indexes, key=lambda item: item.id)],
        "coverage": _coverage_payload(coverage),
        "integrity": {"manifest_sha256": None, "checksum_algorithm": "sha256"},
        "provenance": {
            "adoption_mode": "REFERENCE_EXISTING",
            "fingerprint_policy": fingerprint_policy.value,
            "canonicalization": "sort_keys JSON, compact separators, generated timestamps and mtimes excluded from provenance fingerprints",
            "library_root": str(library_root),
        },
    }
    if compatibility_resources:
        payload["compatibility_resources"] = [_compatibility_payload(item) for item in compatibility_resources]
    return payload


def _source_payload(source: CatalogSource) -> dict[str, Any]:
    return {
        "id": source.id,
        "kind": source.kind,
        "family": source.family,
        "families": list(source.families or (source.family,)),
        "format": source.format,
        "mode": source.mode,
        "path_policy": source.path_policy,
        "path": _path_payload(source.path),
        "tile_count": source.tile_count,
        "layout": source.layout,
        "layout_fingerprint": source.layout_fingerprint,
        "reader_version": source.reader_version,
        "fingerprint_policy": source.fingerprint_policy.value if source.fingerprint_policy else None,
        "coverage": _coverage_payload(source.coverage),
        "shards": [
            {
                "path": shard.path,
                "family": shard.family,
                "tile_code": shard.tile_code,
                "size_bytes": shard.size_bytes,
                "sha256": shard.sha256,
                "mtime_ns": shard.mtime_ns,
                "status": shard.status.value,
            }
            for shard in sorted(source.shards, key=lambda item: item.path)
        ],
        "integrity": {
            "checksum_algorithm": "sha256",
            "files": [
                {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes, "mtime_ns": item.mtime_ns}
                for item in source.integrity_files
            ],
        },
        "provenance": dict(source.provenance),
        "status": source.status.value,
    }


def _index_payload(index: CatalogIndex) -> dict[str, Any]:
    return {
        "id": index.id,
        "engine": index.engine,
        "category": index.category,
        "schema": index.schema,
        "algorithm_version": index.algorithm_version,
        "path": _path_payload(index.path),
        "manifest_path": _path_payload(index.manifest_path) if index.manifest_path else None,
        "source_ids": list(index.source_ids),
        "source_tiles": list(index.source_tiles),
        "families": list(index.families),
        "parameters": dict(index.parameters),
        "build_parameters": dict(index.build_parameters),
        "parameter_status": index.parameter_status.value,
        "provenance_fingerprint": index.provenance_fingerprint,
        "reconstruction_status": index.reconstruction_status,
        "scale_range_arcsec": (
            {"min": index.scale_range_arcsec[0], "max": index.scale_range_arcsec[1]}
            if index.scale_range_arcsec is not None
            else None
        ),
        "coverage": _coverage_payload(index.coverage),
        "integrity": {
            "checksum_algorithm": "sha256",
            "files": [
                {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes, "mtime_ns": item.mtime_ns}
                for item in index.integrity_files
            ],
        },
        "derived_files": [
            {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes, "mtime_ns": item.mtime_ns}
            for item in index.derived_files
        ],
        "source_file_refs": list(index.source_file_refs),
        "compatibility": dict(index.compatibility),
        "status": index.status.value,
    }


def _coverage_payload(coverage: CatalogCoverage) -> dict[str, Any]:
    return {
        "status": coverage.status.value,
        "all_sky": coverage.all_sky,
        "families": list(coverage.families),
        "tile_keys": list(coverage.tile_keys),
        "dec_min_deg": coverage.dec_min_deg,
        "dec_max_deg": coverage.dec_max_deg,
        "ra_segments_deg": [list(item) for item in coverage.ra_segments_deg],
        "covered_tiles": coverage.covered_tiles,
        "total_tiles": coverage.total_tiles,
        "fraction": coverage.fraction,
        "notes": coverage.notes,
    }


def _path_payload(ref: PathReference) -> dict[str, str]:
    return {"kind": ref.kind.value, "value": ref.value}


def _compatibility_payload(item: CatalogCompatibilityResource) -> dict[str, Any]:
    return {
        "id": item.id,
        "category": item.category,
        "path": {"kind": "external_reference", "value": str(item.path)},
        "manifest_path": {"kind": "external_reference", "value": str(item.manifest_path)} if item.manifest_path else None,
        "files": [{"path": file.path, "size_bytes": file.size_bytes, "sha256": file.sha256} for file in item.files],
        "status": item.status.value,
        "notes": item.notes,
    }


def _build_parameters(manifest_entry: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    raw = {**metadata, **manifest_entry}
    return {
        "level": raw.get("level"),
        "mag_cap": raw.get("mag_cap"),
        "max_stars_per_tile": raw.get("max_stars_per_tile"),
        "max_quads_per_tile": raw.get("max_quads_per_tile"),
        "sampler_tag": raw.get("sampler_tag"),
        "code_tol_recommended": raw.get("code_tol_recommended"),
        "dtype": raw.get("dtype"),
        "tan_center_policy": raw.get("tan_center_policy"),
        "star_ordering_policy": raw.get("star_ordering_policy"),
        "star_truncation_policy": raw.get("star_truncation_policy"),
        "projection_implementation": raw.get("projection_implementation"),
        "projection_version": raw.get("projection_version"),
        "quad_schema": raw.get("quad_schema") or raw.get("schema"),
        "quad_version": raw.get("index_version") or raw.get("version"),
        "builder_version": raw.get("builder_version") or raw.get("builder_commit"),
        "catalog_source": raw.get("catalog_source") or raw.get("source_catalog"),
    }


def _parameter_status(parameters: dict[str, Any]) -> ParameterCompleteness:
    values = [value for value in parameters.values() if value is not None]
    if not values:
        return ParameterCompleteness.UNKNOWN
    required = (
        "level",
        "mag_cap",
        "max_stars_per_tile",
        "max_quads_per_tile",
        "sampler_tag",
        "code_tol_recommended",
        "dtype",
        "tan_center_policy",
        "star_ordering_policy",
        "star_truncation_policy",
        "projection_implementation",
        "projection_version",
        "quad_schema",
        "quad_version",
        "builder_version",
    )
    return ParameterCompleteness.KNOWN if all(parameters.get(key) is not None for key in required) else ParameterCompleteness.PARTIAL


def _adoption_coverage(sources: tuple[CatalogSource, ...], indexes: tuple[CatalogIndex, ...]) -> CatalogCoverage:
    if indexes:
        return merge_coverages(index.coverage for index in indexes)
    if sources:
        return merge_coverages(source.coverage for source in sources)
    return CatalogCoverage(status=CoverageStatus.MISSING, provenance="adoption")


def _adoption_status(
    sources: tuple[CatalogSource, ...],
    indexes: tuple[CatalogIndex, ...],
    errors: tuple[CatalogIssue, ...],
    coverage: CatalogCoverage,
) -> CatalogStatus:
    if errors:
        return CatalogStatus.CORRUPT if any("SHA256" in issue.code or "INVALID" in issue.code for issue in errors) else CatalogStatus.MISSING
    if sources and indexes:
        return CatalogStatus.READY_FULL if coverage.all_sky else CatalogStatus.READY_PARTIAL
    if sources:
        return CatalogStatus.SOURCE_ONLY
    if indexes:
        return CatalogStatus.BLIND4D_ONLY
    return CatalogStatus.MISSING


def _coerce_policy(value: str | FingerprintPolicy) -> FingerprintPolicy:
    if isinstance(value, FingerprintPolicy):
        return value
    return FingerprintPolicy(str(value).strip().lower())


def _iter_roots(value: str | Path | Iterable[str | Path] | None) -> tuple[Path | str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (value,)
    return tuple(value)


def _tile_code_from_path(path: Path) -> str | None:
    stem = path.stem
    if "_" not in stem:
        return None
    return stem.split("_", 1)[1]


def _family_from_tiles(tile_keys: tuple[str, ...]) -> str | None:
    for tile_key in tile_keys:
        if "_" in tile_key:
            return tile_key.split("_", 1)[0].lower()
    return None


def _layout_name(family: str) -> str:
    spec = FAMILY_SPECS[family]
    return "hnsky_1476" if spec.extension == "1476" else "hnsky_290"


def _layout_tile_count(family: str | None) -> int:
    if not family or family not in FAMILY_SPECS:
        return 0
    return int(sum(ring.ra_cells for ring in get_layout(_layout_name(family)).iter_rings()))


def _issue(
    code: str,
    severity: IssueSeverity,
    path: Path,
    component_id: str,
    *,
    message: str | None = None,
) -> CatalogIssue:
    return CatalogIssue(code=code, severity=severity, message=message or f"{code}: {component_id}", path=path, component_id=component_id)


def _repair(
    code: str,
    severity: IssueSeverity,
    resource_id: str,
    reason: str,
    preconditions: tuple[str, ...],
    execution_phase: str,
    *,
    automatic: bool,
) -> CatalogRepairAction:
    return CatalogRepairAction(
        code=code,
        severity=severity,
        resource_id=resource_id,
        reason=reason,
        preconditions=preconditions,
        execution_phase=execution_phase,
        automatic=automatic,
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _strip_non_scientific(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_non_scientific(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _SCIENTIFIC_TIMESTAMP_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_non_scientific(item) for item in value]
    return value


__all__ = [
    "CatalogLibraryAdoptionPlan",
    "REPAIR_LOCATE_MISSING_INDEX",
    "REPAIR_LOCATE_MISSING_SOURCE",
    "REPAIR_REBUILD_BLIND4D_INDEX",
    "REPAIR_RECOMPUTE_COVERAGE",
    "REPAIR_REGENERATE_STRICT_4D_MANIFEST",
    "REPAIR_REMOVE_STALE_COMPATIBILITY_REFERENCE",
    "REPAIR_VERIFY_SOURCE_SHA256",
    "canonical_provenance_fingerprint",
]
