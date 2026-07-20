from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.adopt_catalog_library import main


def _astap(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "d50_2823.1476").write_bytes(b"tile")
    return root


def test_cli_preview_is_read_only_by_default(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    report = tmp_path / "preview.json"
    code = main([
        "--library-root",
        str(library),
        "--astap-root",
        str(_astap(tmp_path / "astap")),
        "--report-json",
        str(report),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["mode"] == "preview"
    assert payload["commit"] is None
    assert payload["plan"]["status"] == "SOURCE_ONLY"
    assert not (library / "catalog.json").exists()


def test_cli_write_create_requires_explicit_write(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    report = tmp_path / "write.json"
    code = main([
        "--library-root",
        str(library),
        "--astap-root",
        str(_astap(tmp_path / "astap")),
        "--write",
        "--report-json",
        str(report),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["commit"]["status"] == "CREATED"
    assert (library / "catalog.json").exists()


def test_cli_replace_requires_replace_flag_and_expected_sha(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    astap = _astap(tmp_path / "astap")
    main(["--library-root", str(library), "--astap-root", str(astap), "--write"])
    existing_sha = hashlib.sha256((library / "catalog.json").read_bytes()).hexdigest()
    report = tmp_path / "replace.json"
    code = main([
        "--library-root",
        str(library),
        "--astap-root",
        str(astap),
        "--write",
        "--replace-existing",
        "--expected-existing-sha256",
        existing_sha,
        "--report-json",
        str(report),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["commit"]["status"] == "NO_CHANGE"


def test_cli_replace_flag_without_write_is_error_and_read_only(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    report = tmp_path / "error.json"
    code = main([
        "--library-root",
        str(library),
        "--astap-root",
        str(_astap(tmp_path / "astap")),
        "--replace-existing",
        "--report-json",
        str(report),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert code == 2
    assert payload["error"]["code"] == "CATALOG_ADOPTION_PLAN_INVALID"
    assert not (library / "catalog.json").exists()
