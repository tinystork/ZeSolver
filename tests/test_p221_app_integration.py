from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zeblindsolver.index_manifest_4d import (
    IndexManifestError,
    IndexManifestIntegrityError,
    IndexManifestSchemaError,
    load_4d_index_manifest,
    sha256_file,
)
from zeblindsolver.profiles import (
    HISTORICAL_PROFILE,
    ZEBLIND_4D_EXPERIMENTAL_PROFILE,
    get_solver_profile,
)
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import SolveConfig, _astrometry_4d_runtime_requested


def _load_zesolver_entrypoint():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metadata_array(metadata: dict[str, object]) -> np.ndarray:
    text = json.dumps(metadata, sort_keys=True)
    return np.asarray([text], dtype=f"<U{len(text)}")


def _write_fake_index(path: Path, tile_key: str = "d50_TEST", *, schema: str = ASTROMETRY_AB_CODE_4D_SCHEMA) -> Path:
    metadata = {
        "schema": schema,
        "version": 1,
        "level": "S",
        "sampler_tag": "catalog_ring_coverage",
        "code_tol_recommended": 0.015,
        "source_catalog": "unit-test",
        "generated_at": "2026-07-13T00:00:00Z",
        "max_stars_per_tile": 4,
        "max_quads_per_tile": 1,
        "entry_count": 1,
        "star_count": 4,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        codes_4d=np.asarray([[0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        quad_star_indices=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        source_quad_indices=np.asarray([0], dtype=np.int32),
        tile_key_indices=np.asarray([0], dtype=np.int32),
        ratio_hashes=np.asarray([-1], dtype=np.int64),
        tile_keys=np.asarray([tile_key], dtype=f"<U{len(tile_key)}"),
        catalog_ra_dec=np.asarray([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]], dtype=np.float64),
        catalog_xy=np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.3], [0.4, 0.5]], dtype=np.float64),
        metadata=_metadata_array(metadata),
    )
    return path


def _entry(index_id: str, path: Path, tile_key: str = "d50_TEST", *, enabled: bool = True) -> dict[str, object]:
    return {
        "id": index_id,
        "enabled": enabled,
        "path": path.name,
        "filename": path.name,
        "quad_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "index_version": 1,
        "level": "S",
        "tile_keys": [tile_key],
        "star_count": 4,
        "quad_count": 1,
        "sampler_tag": "catalog_ring_coverage",
        "code_tol_recommended": 0.015,
        "catalog_source": "unit-test",
        "sha256": sha256_file(path) if path.exists() else "0" * 64,
    }


def _manifest(path: Path, entries: list[dict[str, object]], *, version: int = 1) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "manifest_version": version,
                "indexes": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_package_manifest_valid_relative_disabled_and_order(tmp_path: Path) -> None:
    idx1 = _write_fake_index(tmp_path / "a.npz", "d50_A")
    idx2 = _write_fake_index(tmp_path / "b.npz", "d50_B")
    disabled = dict(_entry("disabled", tmp_path / "missing.npz", "d50_MISSING", enabled=False))
    manifest = _manifest(tmp_path / "manifest.json", [disabled, _entry("a", idx1, "d50_A"), _entry("b", idx2, "d50_B")])

    loaded = load_4d_index_manifest(manifest)

    assert loaded.index_ids == ("a", "b")
    assert loaded.enabled_index_paths == (idx1.resolve(), idx2.resolve())
    assert loaded.tile_keys == ("d50_A", "d50_B")
    assert loaded.checksums["a"] == sha256_file(idx1)


def test_package_manifest_rejects_absent_bad_json_version_checksum_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(IndexManifestSchemaError, match="manifest_absent"):
        load_4d_index_manifest(tmp_path / "absent.json")
    invalid = tmp_path / "bad.json"
    invalid.write_text("{ nope", encoding="utf-8")
    with pytest.raises(IndexManifestSchemaError, match="manifest_json_invalid"):
        load_4d_index_manifest(invalid)

    idx1 = _write_fake_index(tmp_path / "a.npz", "d50_A")
    idx2 = _write_fake_index(tmp_path / "b.npz", "d50_B")
    with pytest.raises(IndexManifestSchemaError, match="manifest_version_invalid"):
        load_4d_index_manifest(_manifest(tmp_path / "bad_version.json", [_entry("a", idx1, "d50_A")], version=99))
    bad_sha = dict(_entry("a", idx1, "d50_A"))
    bad_sha["sha256"] = "0" * 64
    with pytest.raises(IndexManifestIntegrityError, match="manifest_sha256_mismatch"):
        load_4d_index_manifest(_manifest(tmp_path / "bad_sha.json", [bad_sha]))
    with pytest.raises(IndexManifestIntegrityError, match="manifest_duplicate_id"):
        load_4d_index_manifest(_manifest(tmp_path / "dup_id.json", [_entry("a", idx1, "d50_A"), _entry("a", idx2, "d50_B")]))
    with pytest.raises(IndexManifestIntegrityError, match="manifest_duplicate_path"):
        load_4d_index_manifest(_manifest(tmp_path / "dup_path.json", [_entry("a", idx1, "d50_A"), _entry("b", idx1, "d50_A")]))
    idx3 = _write_fake_index(tmp_path / "c.npz", "d50_A")
    with pytest.raises(IndexManifestIntegrityError, match="manifest_duplicate_tile"):
        load_4d_index_manifest(_manifest(tmp_path / "dup_tile.json", [_entry("a", idx1, "d50_A"), _entry("c", idx3, "d50_A")]))


def test_solver_profile_contract_and_historical_default(tmp_path: Path) -> None:
    idx = _write_fake_index(tmp_path / "a.npz", "d50_A")
    manifest = load_4d_index_manifest(_manifest(tmp_path / "manifest.json", [_entry("a", idx, "d50_A")]))

    historical = get_solver_profile(HISTORICAL_PROFILE).apply_to_config(SolveConfig())
    assert not _astrometry_4d_runtime_requested(historical)
    assert historical.quad_hash_schema == "opposite_edge_ratio_8bit_v1"

    cfg = get_solver_profile(ZEBLIND_4D_EXPERIMENTAL_PROFILE).apply_to_config(SolveConfig(), index_paths=manifest.enabled_index_paths)
    assert _astrometry_4d_runtime_requested(cfg)
    assert cfg.quad_hash_schema == ASTROMETRY_AB_CODE_4D_SCHEMA
    assert cfg.max_stars == 120
    assert cfg.max_quads == 2500
    assert cfg.quality_inliers == 40
    assert cfg.quality_rms == pytest.approx(1.2)
    assert cfg.blind_astrometry_4d_match_radius_px == pytest.approx(3.0)
    assert cfg.blind_astrometry_4d_validation_catalog_policy == "union_candidate_tiles"
    assert cfg.blind_astrometry_4d_accept_policy == "best_within_budget"
    assert cfg.blind_astrometry_4d_max_hypotheses == 2000
    assert cfg.blind_astrometry_4d_max_accepts == 64
    assert cfg.blind_global_hard_budget_s == pytest.approx(0.0)
    assert cfg.blind_astrometry_4d_search_budget_s == pytest.approx(45.0)
    assert cfg.ra_hint_deg is None and cfg.dec_hint_deg is None


def test_app_settings_profile_persistence_and_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    zs = importlib.import_module("zesolver")
    from zeblindsolver import presets

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(zs, "SETTINGS_PATH", settings_file, raising=True)

    defaults = zs.load_persistent_settings()
    assert defaults.blind_backend_profile == ZEBLIND_4D_EXPERIMENTAL_PROFILE
    assert defaults.blind_4d_manifest_path is None

    settings = zs.PersistentSettings(
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        blind_4d_manifest_path=str(tmp_path / "manifest.json"),
    )
    zs.save_persistent_settings(settings)
    loaded = zs.load_persistent_settings()
    assert loaded.blind_backend_profile == ZEBLIND_4D_EXPERIMENTAL_PROFILE
    assert loaded.blind_4d_manifest_path == str(tmp_path / "manifest.json")

    by_id = {preset.id: preset for preset in presets.list_presets()}
    assert by_id["seestar_s30"].focal_mm == pytest.approx(150.0)
    assert presets.compute_scale_and_fov(150.0, 2.9, 1920, 1080)["scale_arcsec_per_px"] == pytest.approx(3.99, rel=0.04)
    assert by_id["seestar_s50"].focal_mm == pytest.approx(250.0)
    assert presets.compute_scale_and_fov(250.0, 2.9, 1080, 1920)["scale_arcsec_per_px"] == pytest.approx(2.39, rel=0.04)


def test_app_build_blind_config_requires_manifest_only_for_4d(tmp_path: Path) -> None:
    pytest.importorskip("astroalign")
    zs = _load_zesolver_entrypoint()
    historical = zs.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=None,
        blind_backend_profile=HISTORICAL_PROFILE,
    )
    assert not _astrometry_4d_runtime_requested(zs.build_blind_solve_config(historical))

    cfg = zs.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=None,
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
    )
    with pytest.raises(IndexManifestError, match="blind_4d_manifest_required"):
        zs.build_blind_solve_config(cfg)

    idx = _write_fake_index(tmp_path / "a.npz", "d50_A")
    loaded = load_4d_index_manifest(_manifest(tmp_path / "manifest.json", [_entry("a", idx, "d50_A")]))
    cfg = zs.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=None,
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        blind_4d_loaded_manifest=loaded,
        hint_ra_deg=12.3,
        hint_dec_deg=45.6,
    )
    blind_cfg = zs.build_blind_solve_config(cfg, loaded_manifest=loaded)
    assert _astrometry_4d_runtime_requested(blind_cfg)
    assert blind_cfg.blind_astrometry_4d_index_paths == tuple(str(p) for p in loaded.enabled_index_paths)
    assert blind_cfg.ra_hint_deg is None and blind_cfg.dec_hint_deg is None
