from __future__ import annotations

from pathlib import Path

import pytest

from corpus_loader import (
    CorpusDataCorrupt,
    CorpusDataMissing,
    CorpusError,
    load_manifest,
    sha256_file,
    iter_cases,
)


def test_regression_manifest_schema_and_unique_ids() -> None:
    manifest = load_manifest()

    assert manifest["schema_version"] == 1
    cases = list(iter_cases(manifest=manifest, enabled_only=False))
    assert cases
    assert len({case.id for case in cases}) == len(cases)
    assert {case.mode for case in cases} >= {"near", "blind4d", "pipeline"}


def test_regression_manifest_cases_have_provenance_and_explicit_roots() -> None:
    for case in iter_cases(enabled_only=False):
        assert case.raw.get("source_report")
        assert case.raw.get("root_env")
        assert not str(case.raw.get("relative_path", "")).startswith("/home/")


def test_regression_manifest_missing_env_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    case = next(iter_cases(mode="near"))
    monkeypatch.delenv(case.env, raising=False)

    with pytest.raises(CorpusDataMissing, match=case.env):
        case.resolve_path()


def test_regression_manifest_detects_corrupt_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = root / "frame.fit"
    payload.write_bytes(b"not a real fits")
    monkeypatch.setenv("ZESOLVER_CORPUS_ROOT", str(root))
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "id": "corrupt",
                "enabled": True,
                "source_group": "unit",
                "root_env": "ZESOLVER_CORPUS_ROOT",
                "relative_path": "frame.fit",
                "sha256": "0" * 64,
                "solver_mode": "near",
                "expected_status": "solved",
            }
        ],
    }
    case = next(iter_cases(manifest=manifest))

    assert sha256_file(payload) != "0" * 64
    with pytest.raises(CorpusDataCorrupt):
        case.resolve_path()


def test_regression_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version":1,"cases":[{"id":"x","solver_mode":"near"},{"id":"x","solver_mode":"blind4d"}]}',
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="duplicate"):
        load_manifest(path)
