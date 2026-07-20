from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import SolveConfig, _astrometry_4d_runtime_requested

import tools.diagnose_p220_4d_mixed_index_pool as p220


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


def _manifest_entry(index_id: str, path: Path, tile_key: str = "d50_TEST") -> dict[str, object]:
    return {
        "id": index_id,
        "enabled": True,
        "path": str(path),
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
        "sha256": p220.sha256_file(path) if path.exists() else "0" * 64,
    }


def _write_manifest(path: Path, entries: list[dict[str, object]], *, version: int = 1) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": p220.MANIFEST_SCHEMA,
                "manifest_version": version,
                "indexes": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_valid_and_order_deterministic(tmp_path: Path) -> None:
    idx1 = _write_fake_index(tmp_path / "a.npz", "d50_A")
    idx2 = _write_fake_index(tmp_path / "b.npz", "d50_B")
    manifest = _write_manifest(tmp_path / "manifest.json", [_manifest_entry("a", idx1, "d50_A"), _manifest_entry("b", idx2, "d50_B")])

    paths, entries = p220.load_index_manifest(manifest)

    assert paths == [idx1.resolve(), idx2.resolve()]
    assert [entry["id"] for entry in entries] == ["a", "b"]


def test_manifest_disabled_entry_ignored(tmp_path: Path) -> None:
    idx = _write_fake_index(tmp_path / "a.npz", "d50_A")
    disabled = dict(_manifest_entry("disabled", tmp_path / "missing.npz", "d50_MISSING"))
    disabled["enabled"] = False
    manifest = _write_manifest(tmp_path / "manifest.json", [disabled, _manifest_entry("a", idx, "d50_A")])

    paths, entries = p220.load_index_manifest(manifest)

    assert paths == [idx.resolve()]
    assert [entry["id"] for entry in entries] == ["a"]


def test_manifest_absent_and_invalid_json_rejected(tmp_path: Path) -> None:
    with pytest.raises(p220.ManifestError, match="manifest_json_invalid"):
        p220.load_index_manifest(tmp_path / "absent.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{ nope", encoding="utf-8")
    with pytest.raises(p220.ManifestError, match="manifest_json_invalid"):
        p220.load_index_manifest(invalid)


def test_manifest_bad_version_rejected(tmp_path: Path) -> None:
    idx = _write_fake_index(tmp_path / "a.npz")
    manifest = _write_manifest(tmp_path / "manifest.json", [_manifest_entry("a", idx)], version=999)

    with pytest.raises(p220.ManifestError, match="manifest_version_invalid"):
        p220.load_index_manifest(manifest)


def test_manifest_missing_checksum_schema_and_metadata_rejected(tmp_path: Path) -> None:
    idx = _write_fake_index(tmp_path / "a.npz", "d50_A")
    missing = dict(_manifest_entry("missing", tmp_path / "missing.npz", "d50_A"))
    manifest = _write_manifest(tmp_path / "missing.json", [missing])
    with pytest.raises(p220.ManifestError, match="manifest_index_absent"):
        p220.load_index_manifest(manifest)

    bad_sha = dict(_manifest_entry("a", idx, "d50_A"))
    bad_sha["sha256"] = "0" * 64
    manifest = _write_manifest(tmp_path / "bad_sha.json", [bad_sha])
    with pytest.raises(p220.ManifestError, match="manifest_sha256_mismatch"):
        p220.load_index_manifest(manifest)

    bad_schema = dict(_manifest_entry("a", idx, "d50_A"))
    bad_schema["quad_schema"] = "bad"
    manifest = _write_manifest(tmp_path / "bad_schema.json", [bad_schema])
    with pytest.raises(p220.ManifestError, match="manifest_quad_schema_invalid"):
        p220.load_index_manifest(manifest)

    bad_tile = dict(_manifest_entry("a", idx, "d50_A"))
    bad_tile["tile_keys"] = ["d50_WRONG"]
    manifest = _write_manifest(tmp_path / "bad_tile.json", [bad_tile])
    with pytest.raises(p220.ManifestError, match="manifest_tile_keys_mismatch"):
        p220.load_index_manifest(manifest)


def test_manifest_duplicates_rejected(tmp_path: Path) -> None:
    idx1 = _write_fake_index(tmp_path / "a.npz", "d50_A")
    idx2 = _write_fake_index(tmp_path / "b.npz", "d50_B")
    with pytest.raises(p220.ManifestError, match="manifest_duplicate_id"):
        p220.load_index_manifest(_write_manifest(tmp_path / "dup_id.json", [_manifest_entry("a", idx1, "d50_A"), _manifest_entry("a", idx2, "d50_B")]))
    with pytest.raises(p220.ManifestError, match="manifest_duplicate_path"):
        p220.load_index_manifest(_write_manifest(tmp_path / "dup_path.json", [_manifest_entry("a", idx1, "d50_A"), _manifest_entry("b", idx1, "d50_A")]))
    idx3 = _write_fake_index(tmp_path / "c.npz", "d50_A")
    with pytest.raises(p220.ManifestError, match="manifest_duplicate_tile"):
        p220.load_index_manifest(_write_manifest(tmp_path / "dup_tile.json", [_manifest_entry("a", idx1, "d50_A"), _manifest_entry("c", idx3, "d50_A")]))


def test_runtime_prepare_strips_position_and_identity_hints(tmp_path: Path) -> None:
    source = tmp_path / "source.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((8, 8), dtype=np.float32))
    hdu.header["RA"] = 12.3
    hdu.header["DEC"] = 45.6
    hdu.header["OBJCTRA"] = "00 01 02"
    hdu.header["OBJECT"] = "M TEST"
    hdu.header["INSTRUME"] = "Seestar S50"
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRVAL1"] = 12.3
    hdu.header["CRVAL2"] = 45.6
    hdu.header["CRPIX1"] = 4.0
    hdu.header["CRPIX2"] = 4.0
    hdu.header["CD1_1"] = -0.0006
    hdu.header["CD1_2"] = 0.0
    hdu.header["CD2_1"] = 0.0
    hdu.header["CD2_2"] = 0.0006
    hdu.writeto(source)

    work, audit = p220._runtime_work_fits(source, tmp_path / "work", "case")
    header = fits.getheader(work)

    assert audit["forbidden_keys_remaining"] == []
    assert not audit["has_celestial_wcs_after_strip"]
    for key in ("RA", "DEC", "OBJCTRA", "OBJECT", "CTYPE1", "CRVAL1", "CD1_1"):
        assert key not in header
    assert header["INSTRUME"] == "Seestar S50"


def test_backend_defaults_and_manifest_errors_do_not_enable_4d() -> None:
    assert not _astrometry_4d_runtime_requested(SolveConfig())
    assert SolveConfig().blind_astrometry_4d_index_paths == ()
