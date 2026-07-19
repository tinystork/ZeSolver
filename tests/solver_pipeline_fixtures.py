from __future__ import annotations

from pathlib import Path

from astropy.wcs import WCS

from catalog_resource_helpers import strict_entry, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_library import Blind4DIndexDescriptor, CatalogCoverage, CoverageStatus, NearCatalogDescriptor
from zesolver.catalog_resources import SolverCatalogResources


def sample_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [184.62777087563467, 47.298584781528135]
    wcs.wcs.crpix = [8.0, 8.0]
    scale = 2.371713806045426 / 3600.0
    wcs.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    return wcs


def empty_resources() -> SolverCatalogResources:
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
    )


def near_resources(tmp_path: Path, *, blind_count: int = 0) -> SolverCatalogResources:
    coverage = CatalogCoverage(
        status=CoverageStatus.PARTIAL if blind_count else CoverageStatus.UNKNOWN,
        all_sky=False,
        covered_tiles=blind_count,
        total_tiles=1476 if blind_count else None,
        fraction=(blind_count / 1476.0) if blind_count else None,
        provenance="test",
    )
    blind_items: list[Blind4DIndexDescriptor] = []
    manifest_path = None
    if blind_count:
        entries: list[dict[str, object]] = []
        for idx in range(blind_count):
            tile_key = f"d50_{idx}"
            index_path = write_fake_4d_index(tmp_path / f"{tile_key}_S_q.npz", tile_key)
            index_id = f"d50_{idx}"
            entries.append(strict_entry(index_id, index_path, tile_key))
            blind_items.append(
                Blind4DIndexDescriptor(
                    id=index_id,
                    path=index_path,
                    family="d50",
                    tile_keys=(tile_key,),
                    sha256=entries[-1]["sha256"],
                    coverage=coverage,
                    schema="astrometry_ab_code_4d_v1",
                )
            )
        manifest_path = write_strict_manifest(tmp_path / "manifest.json", entries)
    blind = tuple(blind_items)
    return SolverCatalogResources(
        library_path=None,
        library_status=None,
        near=NearCatalogDescriptor(
            root=tmp_path,
            families=("d50",),
            formats=(),
            coverage=CatalogCoverage(status=CoverageStatus.UNKNOWN, provenance="test"),
            external_reference=True,
        ),
        blind4d_indexes=blind,
        blind4d_runtime_paths=tuple(item.path for item in blind),
        blind4d_manifest_path=manifest_path,
        legacy_index_root=tmp_path,
        source="legacy",
        warnings=("blind4d_coverage_not_all_sky",) if blind else (),
        coverage=coverage if blind else None,
        all_sky_blind4d=False,
    )
