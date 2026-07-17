"""Catalog resource resolution for the solver pipeline.

This module deliberately resolves only catalogue/index resources. It does not
own solver thresholds, candidate limits, quad parameters or GUI options.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from zeblindsolver.index_manifest_4d import IndexManifestError, load_4d_index_manifest

from .catalog_library import (
    Blind4DIndexDescriptor,
    CatalogLibrary,
    CatalogStatus,
    CatalogValidationReport,
    CoverageStatus,
    IssueSeverity,
    NearCatalogDescriptor,
    discover_existing,
)
from .catalog_library.models import CatalogCoverage


ENVIRONMENT_CATALOG_KEYS = (
    "ZESOLVER_ASTAP_ROOT",
    "ZESOLVER_BLIND4D_MANIFEST",
    "ZEBLIND_4D_MANIFEST",
    "ZESOLVER_LEGACY_INDEX_ROOT",
)


class CatalogResourceResolutionError(RuntimeError):
    """Raised when explicitly requested catalogue resources cannot be used."""


@dataclass(frozen=True, slots=True)
class SolverCatalogResources:
    library_path: Path | None
    library_status: CatalogStatus | None
    near: NearCatalogDescriptor | None
    blind4d_indexes: tuple[Blind4DIndexDescriptor, ...]
    blind4d_runtime_paths: tuple[Path, ...]
    blind4d_manifest_path: Path | None
    legacy_index_root: Path | None
    source: str
    warnings: tuple[str, ...]
    catalog_library_id: str | None = None
    coverage: CatalogCoverage | None = None
    all_sky_blind4d: bool = False

    @property
    def near_available(self) -> bool:
        return self.near is not None

    @property
    def blind4d_available(self) -> bool:
        return bool(self.blind4d_runtime_paths)

    @property
    def blind4d_index_count(self) -> int:
        return len(self.blind4d_runtime_paths)

    @property
    def blind4d_coverage_fraction(self) -> float | None:
        if self.coverage is None:
            return None
        return self.coverage.fraction

    def telemetry(self, *, include_paths: bool = False) -> dict[str, object]:
        coverage = self.coverage
        data: dict[str, object] = {
            "catalog_source": self.source,
            "catalog_library_id": self.catalog_library_id,
            "catalog_library_status": self.library_status.value if self.library_status else None,
            "near_family": list(self.near.families) if self.near else [],
            "blind4d_index_count": self.blind4d_index_count,
            "blind4d_all_sky": bool(self.all_sky_blind4d),
            "blind4d_coverage_fraction": self.blind4d_coverage_fraction,
            "catalog_profile": "catalog-library-v1" if self.source == "library" else self.source,
            "coverage_status": coverage.status.value if coverage else None,
            "coverage_tile_count": coverage.covered_tiles if coverage else 0,
            "coverage_total_tiles": coverage.total_tiles if coverage else None,
            "warnings": list(self.warnings),
        }
        if include_paths:
            data.update(
                {
                    "catalog_library_path": str(self.library_path) if self.library_path else None,
                    "blind4d_manifest_path": str(self.blind4d_manifest_path) if self.blind4d_manifest_path else None,
                    "legacy_index_root": str(self.legacy_index_root) if self.legacy_index_root else None,
                }
            )
        return data


def resolve_catalog_resources(
    *,
    catalog_library: CatalogLibrary | str | Path | None = None,
    legacy_db_root: str | Path | None = None,
    legacy_families: tuple[str, ...] | list[str] | None = None,
    legacy_blind4d_manifest: str | Path | None = None,
    legacy_index_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    enable_environment_discovery: bool = True,
    allow_legacy_fallback_on_invalid_library: bool = False,
) -> SolverCatalogResources:
    """Resolve catalogue resources in the P1C priority order."""

    if catalog_library is not None:
        try:
            library = _coerce_library(catalog_library)
            return _resources_from_library(library)
        except Exception as exc:
            if not allow_legacy_fallback_on_invalid_library:
                raise CatalogResourceResolutionError(f"catalog_library_invalid: {exc}") from exc
            legacy = _resources_from_legacy(
                legacy_db_root=legacy_db_root,
                legacy_families=legacy_families,
                legacy_blind4d_manifest=legacy_blind4d_manifest,
                legacy_index_root=legacy_index_root,
                source="legacy",
            )
            return _with_warning(legacy, f"catalog_library_invalid_fell_back_to_legacy: {exc}")

    if legacy_db_root is not None or legacy_blind4d_manifest is not None or legacy_index_root is not None:
        return _resources_from_legacy(
            legacy_db_root=legacy_db_root,
            legacy_families=legacy_families,
            legacy_blind4d_manifest=legacy_blind4d_manifest,
            legacy_index_root=legacy_index_root,
            source="legacy",
        )

    if enable_environment_discovery and _has_environment_catalog_hint(env):
        return _resources_from_environment(env)

    return SolverCatalogResources(
        library_path=None,
        library_status=None,
        near=None,
        blind4d_indexes=(),
        blind4d_runtime_paths=(),
        blind4d_manifest_path=None,
        legacy_index_root=None,
        source="none",
        warnings=("catalog_resources_absent",),
        catalog_library_id=None,
        coverage=None,
        all_sky_blind4d=False,
    )


def _coerce_library(value: CatalogLibrary | str | Path) -> CatalogLibrary:
    if isinstance(value, CatalogLibrary):
        return value
    return CatalogLibrary.open(value)


def _resources_from_library(library: CatalogLibrary) -> SolverCatalogResources:
    report = library.validate()
    _raise_if_library_unusable(report)
    warnings = _warnings_from_report(report)
    near = library.near_source() if report.capabilities.near else None
    blind_indexes = library.blind4d_indexes() if report.capabilities.blind4d else ()
    manifest_path = _library_4d_manifest_path(library, blind_indexes)
    if report.status is CatalogStatus.READY_PARTIAL:
        warnings.append("catalog_library_ready_partial")
    if report.capabilities.blind4d and not report.capabilities.all_sky_blind4d:
        warnings.append("blind4d_coverage_not_all_sky")
    return SolverCatalogResources(
        library_path=library.root,
        library_status=report.status,
        near=near,
        blind4d_indexes=blind_indexes,
        blind4d_runtime_paths=tuple(index.path for index in blind_indexes),
        blind4d_manifest_path=manifest_path,
        legacy_index_root=None,
        source="library",
        warnings=tuple(dict.fromkeys(warnings)),
        catalog_library_id=library.manifest.library_id,
        coverage=report.coverage,
        all_sky_blind4d=bool(report.capabilities.all_sky_blind4d),
    )


def _raise_if_library_unusable(report: CatalogValidationReport) -> None:
    hard_error_codes = (
        "CORRUPT",
        "SHA256_MISMATCH",
        "INCOMPATIBLE",
    )
    for issue in report.issues:
        if issue.severity in {IssueSeverity.ERROR, IssueSeverity.FATAL} and any(token in issue.code for token in hard_error_codes):
            raise CatalogResourceResolutionError(f"{report.status.value}: {issue.code}")
    if report.status in {CatalogStatus.CORRUPT, CatalogStatus.INCOMPATIBLE, CatalogStatus.MISSING}:
        issue_codes = ", ".join(issue.code for issue in report.issues) or report.status.value
        raise CatalogResourceResolutionError(f"{report.status.value}: {issue_codes}")


def _warnings_from_report(report: CatalogValidationReport) -> list[str]:
    result: list[str] = []
    for issue in report.issues:
        if issue.severity in {IssueSeverity.INFO, IssueSeverity.WARNING}:
            result.append(issue.code)
    return result


def _library_4d_manifest_path(
    library: CatalogLibrary,
    descriptors: tuple[Blind4DIndexDescriptor, ...],
) -> Path | None:
    if not descriptors:
        return None
    descriptor_ids = {descriptor.id for descriptor in descriptors}
    manifest_paths: list[Path] = []
    for index in library.manifest.derived_indexes:
        if index.engine != "blind4d" or index.id not in descriptor_ids or index.manifest_path is None:
            continue
        manifest_paths.append(index.manifest_path.resolved)
    unique = tuple(dict.fromkeys(path.resolve() for path in manifest_paths))
    if len(unique) == 1:
        return unique[0]
    return None


def _resources_from_legacy(
    *,
    legacy_db_root: str | Path | None,
    legacy_families: tuple[str, ...] | list[str] | None,
    legacy_blind4d_manifest: str | Path | None,
    legacy_index_root: str | Path | None,
    source: str,
) -> SolverCatalogResources:
    near = None
    if legacy_db_root is not None:
        near = NearCatalogDescriptor(
            root=Path(legacy_db_root).expanduser(),
            families=_normalize_families(legacy_families),
            formats=(),
            coverage=CatalogCoverage(status=CoverageStatus.UNKNOWN, provenance=source),
            external_reference=True,
        )
    blind_indexes: tuple[Blind4DIndexDescriptor, ...] = ()
    manifest_path = Path(legacy_blind4d_manifest).expanduser() if legacy_blind4d_manifest is not None else None
    warnings: list[str] = []
    if manifest_path is not None:
        try:
            loaded = load_4d_index_manifest(manifest_path)
        except IndexManifestError as exc:
            raise CatalogResourceResolutionError(f"legacy_blind4d_manifest_invalid: {exc}") from exc
        blind_indexes = tuple(
            Blind4DIndexDescriptor(
                id=entry.id,
                path=entry.path,
                family=_family_from_tiles(entry.tile_keys),
                tile_keys=entry.tile_keys,
                sha256=entry.sha256,
                coverage=CatalogCoverage(
                    status=CoverageStatus.PARTIAL,
                    all_sky=False,
                    families=tuple(filter(None, (_family_from_tiles(entry.tile_keys),))),
                    tile_keys=entry.tile_keys,
                    covered_tiles=len(entry.tile_keys),
                    provenance=source,
                ),
                schema=entry.quad_schema,
                enabled=True,
            )
            for entry in loaded.entries
        )
        warnings.append("legacy_blind4d_manifest_used")
    return SolverCatalogResources(
        library_path=None,
        library_status=None,
        near=near,
        blind4d_indexes=blind_indexes,
        blind4d_runtime_paths=tuple(index.path for index in blind_indexes),
        blind4d_manifest_path=manifest_path,
        legacy_index_root=Path(legacy_index_root).expanduser() if legacy_index_root is not None else None,
        source=source,
        warnings=tuple(warnings),
        catalog_library_id=None,
        coverage=_merge_descriptor_coverage(blind_indexes),
        all_sky_blind4d=False,
    )


def _resources_from_environment(env: Mapping[str, str] | None) -> SolverCatalogResources:
    discovery = discover_existing(env=dict(env) if env is not None else None)
    near = None
    if discovery.astap_root is not None and discovery.families:
        near = NearCatalogDescriptor(
            root=discovery.astap_root,
            families=discovery.families,
            formats=(),
            coverage=CatalogCoverage(status=CoverageStatus.UNKNOWN, provenance="environment"),
            external_reference=True,
        )
    manifest_path = discovery.blind4d_manifest
    return SolverCatalogResources(
        library_path=None,
        library_status=discovery.status,
        near=near,
        blind4d_indexes=discovery.blind4d_indexes,
        blind4d_runtime_paths=tuple(index.path for index in discovery.blind4d_indexes),
        blind4d_manifest_path=manifest_path,
        legacy_index_root=discovery.legacy_index_root,
        source="environment",
        warnings=tuple(issue.code for issue in discovery.issues),
        catalog_library_id=None,
        coverage=_merge_descriptor_coverage(discovery.blind4d_indexes),
        all_sky_blind4d=False,
    )


def _has_environment_catalog_hint(env: Mapping[str, str] | None) -> bool:
    env_map = env if env is not None else {}
    if env is None:
        import os

        env_map = os.environ
    return any(str(env_map.get(key) or "").strip() for key in ENVIRONMENT_CATALOG_KEYS)


def _normalize_families(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        text = str(value).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _family_from_tiles(tile_keys: tuple[str, ...]) -> str | None:
    for tile_key in tile_keys:
        if "_" in tile_key:
            return tile_key.split("_", 1)[0].lower()
    return None


def _merge_descriptor_coverage(indexes: tuple[Blind4DIndexDescriptor, ...]) -> CatalogCoverage | None:
    if not indexes:
        return None
    tile_keys: list[str] = []
    families: list[str] = []
    for index in indexes:
        tile_keys.extend(index.tile_keys)
        if index.family:
            families.append(index.family)
    return CatalogCoverage(
        status=CoverageStatus.PARTIAL,
        all_sky=False,
        families=tuple(dict.fromkeys(families)),
        tile_keys=tuple(dict.fromkeys(tile_keys)),
        covered_tiles=len(tuple(dict.fromkeys(tile_keys))),
        provenance="runtime-descriptors",
    )


def _with_warning(resources: SolverCatalogResources, warning: str) -> SolverCatalogResources:
    return SolverCatalogResources(
        library_path=resources.library_path,
        library_status=resources.library_status,
        near=resources.near,
        blind4d_indexes=resources.blind4d_indexes,
        blind4d_runtime_paths=resources.blind4d_runtime_paths,
        blind4d_manifest_path=resources.blind4d_manifest_path,
        legacy_index_root=resources.legacy_index_root,
        source=resources.source,
        warnings=tuple(dict.fromkeys((*resources.warnings, warning))),
        catalog_library_id=resources.catalog_library_id,
        coverage=resources.coverage,
        all_sky_blind4d=resources.all_sky_blind4d,
    )


__all__ = [
    "CatalogResourceResolutionError",
    "SolverCatalogResources",
    "resolve_catalog_resources",
]
