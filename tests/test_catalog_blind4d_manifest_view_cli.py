from __future__ import annotations

import json
from pathlib import Path

from catalog_resource_helpers import write_catalog_library, write_fake_4d_index
from tools.generate_blind4d_manifest_view import main
from zeblindsolver.index_manifest_4d import load_4d_index_manifest


def _library(tmp_path: Path) -> Path:
    idx = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    root = write_catalog_library(tmp_path / "library", index_paths=[idx])
    payload = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    payload["runtime_order"] = {"blind4d": ["blind4d-0"]}
    (root / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def test_generate_view_cli_preview_is_read_only(tmp_path: Path) -> None:
    root = _library(tmp_path)
    report = tmp_path / "report.json"

    code = main(["--catalog-library", str(root), "--report-json", str(report)])

    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "READY"
    assert payload["materialized_path"] is None
    assert not (tmp_path / "view.json").exists()


def test_generate_view_cli_write_requires_explicit_flag(tmp_path: Path) -> None:
    root = _library(tmp_path)
    out = tmp_path / "view.json"
    report = tmp_path / "report.json"

    code = main(["--catalog-library", str(root), "--output", str(out), "--report-json", str(report)])

    assert code == 0
    assert not out.exists()


def test_generate_view_cli_materializes_when_requested(tmp_path: Path) -> None:
    root = _library(tmp_path)
    out = tmp_path / "view.json"

    code = main(["--catalog-library", str(root), "--output", str(out), "--write"])

    assert code == 0
    loaded = load_4d_index_manifest(out)
    assert loaded.index_ids == ("blind4d-0",)
