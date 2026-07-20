"""Read-only descriptors bridging CatalogLibrary to existing solver inputs."""

from __future__ import annotations

from .models import Blind4DIndexDescriptor, CatalogDataStatus, NearCatalogDescriptor
from .manifest import CatalogIncompleteError


def near_source_descriptor(library) -> NearCatalogDescriptor:
    report = library.validate()
    if not report.capabilities.near:
        raise CatalogIncompleteError("NEAR_SOURCE_UNAVAILABLE")
    sources = [
        source
        for source in library.manifest.sources
        if source.kind == "astap_hnsky"
        and source.path.resolved.exists()
        and source.status not in {CatalogDataStatus.MISSING, CatalogDataStatus.CORRUPT, CatalogDataStatus.INCOMPATIBLE}
    ]
    if not sources:
        raise CatalogIncompleteError("NEAR_SOURCE_UNAVAILABLE")
    root = sources[0].path.resolved
    return NearCatalogDescriptor(
        root=root,
        families=tuple(source.family for source in sources),
        formats=tuple(source.format for source in sources),
        coverage=report.coverage,
        external_reference=any(source.path.external_reference for source in sources),
    )


def blind4d_index_descriptors(library) -> tuple[Blind4DIndexDescriptor, ...]:
    report = library.validate()
    if not report.capabilities.blind4d:
        raise CatalogIncompleteError("BLIND4D_INDEXES_UNAVAILABLE")
    bad_ids = {
        issue.component_id
        for issue in report.issues
        if issue.component_id and issue.code.startswith("INDEX_") and issue.severity.value in {"ERROR", "FATAL"}
    }
    descriptors: list[Blind4DIndexDescriptor] = []
    for index in library.manifest.derived_indexes:
        if index.engine != "blind4d" or index.id in bad_ids or not index.path.resolved.exists():
            continue
        sha = None
        if index.integrity_files:
            sha = index.integrity_files[0].sha256
        family = None
        for tile_key in index.source_tiles:
            if "_" in tile_key:
                family = tile_key.split("_", 1)[0].lower()
                break
        descriptors.append(
            Blind4DIndexDescriptor(
                id=index.id,
                path=index.path.resolved,
                family=family,
                tile_keys=index.source_tiles,
                sha256=sha,
                coverage=index.coverage,
                schema=index.schema,
                enabled=True,
            )
        )
    if not descriptors:
        raise CatalogIncompleteError("BLIND4D_INDEXES_UNAVAILABLE")
    return tuple(descriptors)
