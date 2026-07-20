from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.audit_catalog_library import build_report, main


def _write_fake_4d_index(path: Path, *, tile_key: str = "d50_2823") -> str:
    metadata = {
        "schema": "astrometry_ab_code_4d_v1",
        "version": 1,
        "tile_keys": [tile_key],
        "sampler_tag": "catalog_ring_coverage",
    }
    text = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(path, metadata=np.asarray([text], dtype=f"<U{len(text)}"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_catalog_library_scans_astap_filenames(tmp_path: Path) -> None:
    astap_root = tmp_path / "astap"
    astap_root.mkdir()
    (astap_root / "d50_2823.1476").write_bytes(b"not decoded by scanner")
    (astap_root / "g05_0101.290").write_bytes(b"not decoded by scanner")

    report = build_report(astap_root, None, None)

    assert report["astap"]["status"] == "READY_PARTIAL"
    assert report["astap"]["families"]["d50"]["tile_count"] == 1
    assert report["astap"]["families"]["d50"]["status"] == "PARTIAL"
    assert report["astap"]["families"]["g05"]["tile_count"] == 1


def test_audit_catalog_library_reads_4d_manifest_without_rebuilding(tmp_path: Path) -> None:
    index_path = tmp_path / "index.npz"
    sha = _write_fake_4d_index(index_path)
    manifest = {
        "schema": "zeblind.astrometry_4d_index_manifest.v1",
        "manifest_version": 1,
        "indexes": [
            {
                "id": "d50_2823_test",
                "enabled": True,
                "path": "index.npz",
                "sha256": sha,
                "quad_schema": "astrometry_ab_code_4d_v1",
                "index_version": 1,
                "level": "S",
                "tile_keys": ["d50_2823"],
                "star_count": 2000,
                "quad_count": 40000,
                "sampler_tag": "catalog_ring_coverage",
                "catalog_source": "tile_npz:x_deg/y_deg/ra_deg/dec_deg",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_report(None, manifest_path, None)

    assert report["blind4d"]["status"] == "READY_PARTIAL"
    assert report["blind4d"]["coverage"]["all_sky"] is False
    assert report["blind4d"]["coverage"]["families"]["d50"]["tile_count"] == 1
    assert report["blind4d"]["indexes"][0]["sha256_ok"] is True


def test_audit_catalog_library_reports_corrupt_4d_sha(tmp_path: Path) -> None:
    index_path = tmp_path / "index.npz"
    _write_fake_4d_index(index_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "manifest_version": 1,
                "indexes": [
                    {
                        "id": "bad_sha",
                        "path": "index.npz",
                        "sha256": "0" * 64,
                        "tile_keys": ["d50_2823"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_report(None, manifest_path, None)

    assert report["blind4d"]["status"] == "CORRUPT"
    assert any(issue["code"] == "BLIND4D_INDEX_SHA256_MISMATCH" for issue in report["blind4d"]["issues"])


def test_audit_catalog_library_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"

    code = main(["--output-json", str(out_json), "--output-md", str(out_md)])

    assert code == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "Catalog Library Audit" in out_md.read_text(encoding="utf-8")


def test_catalog_manifest_architecture_examples_are_valid_json() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "docs/architecture/catalog_manifest_schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "docs/architecture/catalog_manifest_example.json").read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == 1
    assert example["schema_version"] == 1
    assert example["status"] == "READY_PARTIAL"
    assert example["derived_indexes"][0]["coverage"]["all_sky"] is False
