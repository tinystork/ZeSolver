from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_resource_helpers import write_fake_4d_index
from zeblindsolver import presets
from zeblindsolver.index_manifest_4d import sha256_file
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zesolver.catalog_library.verification_cache import (
    FULL_LEVEL,
    STATUS_VALID,
    build_lightweight_catalog_fingerprint,
    catalog_verification_record,
    restore_cached_catalog_verification,
)
from zesolver.engine_selection import select_engine
from zesolver.gui_pipeline.settings_adapter import build_engine_selection_request, build_gui_solve_request_from_legacy_config
from zesolver.settings_store import PersistentSettings, load_persistent_settings, save_persistent_settings


def _library(root: Path, *, tiles: tuple[str, ...] = ("d50_0001", "d50_0002"), order: tuple[str, ...] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indexes = []
    runtime_order = []
    for idx, tile in enumerate(tiles):
        path = write_fake_4d_index(root / "indexes" / f"direct-d50-fixed32-{idx:03d}.npz", tile)
        index_id = f"direct-d50-fixed32-{idx:03d}"
        runtime_order.append(index_id)
        indexes.append(
            {
                "id": index_id,
                "engine": "blind4d",
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "algorithm_version": ASTROMETRY_AB_CODE_4D_SCHEMA,
                "path": {"kind": "relative", "value": f"indexes/{path.name}"},
                "manifest_path": None,
                "source_ids": ["astap-d50"],
                "source_tiles": [tile],
                "coverage": {
                    "status": "PARTIAL",
                    "all_sky": False,
                    "families": ["d50"],
                    "tile_keys": [tile],
                    "covered_tiles": 1,
                    "total_tiles": len(tiles),
                    "fraction": 1.0 / len(tiles),
                },
                "integrity": {"files": [{"path": f"indexes/{path.name}", "sha256": sha256_file(path), "size_bytes": path.stat().st_size}]},
                "derived_files": [{"path": f"indexes/{path.name}", "sha256": sha256_file(path), "size_bytes": path.stat().st_size}],
                "status": "FULL_VERIFIED",
                "category": "product",
                "provenance_fingerprint": f"prov-{idx}",
            }
        )
    source_dir = root / "sources" / "astap" / "d50"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "d50_0001.1476").write_bytes(b"fixture")
    payload = {
        "schema_version": 1,
        "library_id": "s5e-test-library",
        "created_at": "2026-07-25T00:00:00Z",
        "created_by": "tests",
        "minimum_zesolver_version": None,
        "status": "READY_FULL" if len(tiles) > 1 else "READY_PARTIAL",
        "sources": [
            {
                "id": "astap-d50",
                "kind": "astap_hnsky",
                "family": "d50",
                "format": "1476-5",
                "path": {"kind": "relative", "value": "sources/astap/d50"},
                "tile_count": len(tiles),
                "layout": "hnsky_1476",
                "coverage": {"status": "FULL", "all_sky": True, "families": ["d50"], "tile_keys": list(tiles)},
                "integrity": {"files": []},
                "status": "FAST_VERIFIED",
            }
        ],
        "derived_indexes": indexes,
        "coverage": {
            "status": "FULL" if len(tiles) > 1 else "PARTIAL",
            "all_sky": len(tiles) > 1,
            "families": ["d50"],
            "tile_keys": list(tiles),
            "covered_tiles": len(tiles),
            "total_tiles": len(tiles),
            "fraction": 1.0,
        },
        "integrity": {"checksum_algorithm": "sha256"},
        "provenance": {"kind": "s5e-fixture"},
        "runtime_order": {"blind4d": list(order or tuple(runtime_order))},
    }
    (root / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def test_instrument_s50_settings_roundtrip_and_corruption_is_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.settings_store as store

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store, "_resolve_settings_path", lambda: settings_file)
    s50 = {item.id: item for item in presets.list_presets()}["seestar_s50"]
    save_persistent_settings(
        PersistentSettings(
            last_preset_id="seestar_s50",
            last_fov_focal_mm=s50.focal_mm,
            last_fov_pixel_um=s50.pixel_um,
            last_fov_res_w=s50.res_w,
            last_fov_res_h=s50.res_h,
            last_fov_reducer=s50.reducer,
            last_fov_binning=1,
        )
    )
    payload = json.loads(settings_file.read_text(encoding="utf-8"))
    payload["last_fov_pixel_um"] = "broken"
    payload["last_fov_res_w"] = -5
    settings_file.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_persistent_settings()
    assert loaded.last_preset_id == "seestar_s50"
    assert loaded.last_fov_focal_mm == pytest.approx(250.0)
    assert loaded.last_fov_pixel_um == pytest.approx(0.0)
    assert loaded.last_fov_res_w == 0
    assert loaded.last_fov_res_h == 1920


def test_catalog_full_cache_reused_and_payloads_not_hashed(tmp_path: Path) -> None:
    root = _library(tmp_path / "library")
    fp = build_lightweight_catalog_fingerprint(root)
    record = catalog_verification_record(fingerprint=fp, verification_level=FULL_LEVEL, verification_status=STATUS_VALID)
    restored = restore_cached_catalog_verification(root, record)
    assert restored.cache_reused is True
    assert restored.message == "Vérifiée — cache valide"
    assert restored.fingerprint is not None
    assert restored.fingerprint.payload_hash_count == 0
    assert restored.fingerprint.blind4d_index_count == 2
    assert restored.fingerprint.covered_tiles == 2
    assert restored.fingerprint.all_sky is True


def test_catalog_cache_invalidates_catalog_manifest_order_shard_and_coverage(tmp_path: Path) -> None:
    root = _library(tmp_path / "library")
    fp = build_lightweight_catalog_fingerprint(root)
    record = catalog_verification_record(fingerprint=fp, verification_level=FULL_LEVEL, verification_status=STATUS_VALID)

    payload = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    payload["runtime_order"]["blind4d"] = list(reversed(payload["runtime_order"]["blind4d"]))
    (root / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert restore_cached_catalog_verification(root, record).cache_reused is False

    root = _library(tmp_path / "library2")
    fp = build_lightweight_catalog_fingerprint(root)
    record = catalog_verification_record(fingerprint=fp, verification_level=FULL_LEVEL, verification_status=STATUS_VALID)
    first = next((root / "indexes").glob("*.npz"))
    first.write_bytes(first.read_bytes() + b"x")
    assert restore_cached_catalog_verification(root, record).cache_reused is False

    root = _library(tmp_path / "library3", tiles=("d50_0001",))
    fp = build_lightweight_catalog_fingerprint(root)
    assert fp.all_sky is True
    payload = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    payload["derived_indexes"][0]["coverage"]["total_tiles"] = 2
    payload["coverage"]["total_tiles"] = 2
    payload["coverage"]["all_sky"] = False
    payload["coverage"]["status"] = "PARTIAL"
    (root / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    changed = build_lightweight_catalog_fingerprint(root)
    assert changed.all_sky is False


def test_engine_selection_uses_final_catalog_resources_all_sky() -> None:
    class Config:
        input_dir = Path("/tmp")
        catalog_library_path = Path("/catalog")
        overwrite = True
        workers = 2
        blind_enabled = True
        astrometry_fallback_after_blind = False
        astrometry_api_key = None
        formats = ("fit",)
        max_files = None
        log_level = "INFO"
        fov_deg = 1.5
        downsample = 1

    class Resources:
        all_sky_blind4d = True

    request = build_gui_solve_request_from_legacy_config([Path("m31.fit")], Config(), catalog_resources=Resources())
    selected = select_engine(build_engine_selection_request(request))
    assert request.blind4d_all_sky is True
    assert "blind4d_coverage_partial_not_all_sky" not in selected.warnings
