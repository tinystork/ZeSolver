from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_library.discovery import discover_existing
from zesolver.catalog_library.models import CatalogStatus


def _write_manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = root / "d50_2823.npz"
    index.write_bytes(b"index")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "manifest_version": 1,
                "indexes": [
                    {
                        "id": "d50_2823",
                        "enabled": True,
                        "path": "d50_2823.npz",
                        "sha256": None,
                        "quad_schema": "astrometry_ab_code_4d_v1",
                        "tile_keys": ["d50_2823"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_astap(root: Path, *, family: str, tile: str = "1501") -> Path:
    ra = np.asarray([12.00, 12.03, 12.06, 12.10], dtype=np.float64)
    dec = np.asarray([1.00, 1.06, 0.98, 1.12], dtype=np.float64)
    write_astap_1476_tile(root, family=family, tile_code=tile, ra_deg=ra, dec_deg=dec)
    return root / f"{family}_{tile}.1476"


def test_discovery_with_no_paths_is_missing() -> None:
    result = discover_existing(env={})

    assert result.status == CatalogStatus.MISSING
    assert result.families == ()
    assert result.blind4d_indexes == ()


def test_discovery_finds_astap_families(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    (astap / "d50_2823.1476").write_bytes(b"tile")

    result = discover_existing(astap_root=astap, env={})

    assert result.status == CatalogStatus.NEAR_ONLY
    assert result.families == ("d50",)


def test_discovery_finds_blind4d_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    result = discover_existing(blind4d_manifest=manifest, env={})

    assert result.status == CatalogStatus.BLIND4D_ONLY
    assert len(result.blind4d_indexes) == 1
    assert result.blind4d_indexes[0].tile_keys == ("d50_2823",)


def test_discovery_finds_astap_and_blind4d(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    (astap / "d50_2823.1476").write_bytes(b"tile")
    manifest = _write_manifest(tmp_path)

    result = discover_existing(astap_root=astap, blind4d_manifest=manifest, env={})

    assert result.status == CatalogStatus.READY_PARTIAL


def test_discovery_uses_legacy_blind4d_env_alias(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    result = discover_existing(env={"ZEBLIND_4D_MANIFEST": str(manifest)})

    assert result.status == CatalogStatus.BLIND4D_ONLY


def test_discovery_warns_when_blind4d_env_vars_conflict(tmp_path: Path) -> None:
    primary = _write_manifest(tmp_path / "primary")
    legacy = _write_manifest(tmp_path / "legacy")

    result = discover_existing(
        env={
            "ZESOLVER_BLIND4D_MANIFEST": str(primary),
            "ZEBLIND_4D_MANIFEST": str(legacy),
        }
    )

    assert result.blind4d_manifest == primary
    assert any(issue.code == "BLIND4D_ENV_CONFLICT" for issue in result.issues)


def test_discovery_current_bundled_4d_manifest_is_partial_blind4d_only() -> None:
    manifest = Path(__file__).resolve().parents[1] / "config/zeblind_4d_experimental_manifest.json"

    result = discover_existing(blind4d_manifest=manifest, env={})
    if not result.blind4d_indexes:
        pytest.skip("bundled 4D NPZ indexes are not present in this checkout")

    assert result.status == CatalogStatus.BLIND4D_ONLY
    assert len(result.blind4d_indexes) == 6
    assert result.blind4d_indexes[0].coverage.total_tiles == 1476
    assert result.blind4d_indexes[0].coverage.all_sky is False


def test_discovery_detects_only_installed_astap_families_case_insensitive(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    path = _write_astap(astap, family="d50")
    path.rename(astap / "D50_1501.1476")
    _write_astap(astap, family="d20")
    (astap / "notes.txt").write_text("ignore", encoding="utf-8")

    discovery = discover_existing(astap_root=astap)

    assert discovery.status == CatalogStatus.NEAR_ONLY
    assert discovery.families == ("d20", "d50")
    assert [source["family"] for source in discovery.candidate_manifest["sources"]] == ["d20", "d50"]


def test_discovery_d50_subdirectory_does_not_require_absent_families(tmp_path: Path) -> None:
    astap = tmp_path / "program-files-astap"
    d50_dir = astap / "D50"
    path = _write_astap(d50_dir, family="d50")
    path.rename(d50_dir / "D50_1501.1476")

    discovery = discover_existing(astap_root=astap)

    assert discovery.status == CatalogStatus.NEAR_ONLY
    assert discovery.families == ("d50",)
    assert not discovery.issues
