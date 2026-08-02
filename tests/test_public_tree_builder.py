from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_public_tree.py"
SPEC = importlib.util.spec_from_file_location("build_public_tree", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build_public_tree = importlib.util.module_from_spec(SPEC)
sys.modules["build_public_tree"] = build_public_tree
SPEC.loader.exec_module(build_public_tree)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _make_clean_test_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "checkout", "-q", "-b", "test")
    (repo / "packaging").mkdir()
    (repo / "public.txt").write_text("public\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "secret.py").write_text("secret\n", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools" / "internal.py").write_text("internal\n", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pkg" / "__pycache__").mkdir()
    (repo / "pkg" / "__pycache__" / "runtime.pyc").write_bytes(b"cache")
    (repo / "packaging" / "public_manifest.txt").write_text(
        "\n".join(
            [
                "public.txt",
                "pkg/**",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "user.name=Test",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    return repo


def test_public_tree_builder_exports_only_manifest_allowlist(tmp_path: Path) -> None:
    repo = _make_clean_test_repo(tmp_path)
    destination = tmp_path / "public"

    report = build_public_tree.build_public_tree(
        source=repo,
        destination=destination,
        manifest=Path("packaging/public_manifest.txt"),
        check_only=False,
    )

    assert destination.joinpath("public.txt").read_text(encoding="utf-8") == "public\n"
    assert destination.joinpath("pkg", "runtime.py").is_file()
    assert destination.joinpath("ZESOLVER_SOURCE_REVISION").is_file()
    assert not destination.joinpath("tests").exists()
    assert not destination.joinpath("tools").exists()
    assert not destination.joinpath("pkg", "__pycache__").exists()
    assert "public.txt" in report.copied_files
    assert "pkg/runtime.py" in report.copied_files


def test_public_tree_builder_rejects_dirty_source(tmp_path: Path) -> None:
    repo = _make_clean_test_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(build_public_tree.PublicTreeError, match="source must be clean"):
        build_public_tree.build_public_tree(
            source=repo,
            destination=tmp_path / "public",
            manifest=Path("packaging/public_manifest.txt"),
            check_only=True,
        )


def test_public_manifest_rejects_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "bad_manifest.txt"
    manifest.write_text("../secret\n", encoding="utf-8")

    with pytest.raises(build_public_tree.PublicTreeError, match="invalid manifest path"):
        build_public_tree.read_manifest(manifest)


def test_public_manifest_includes_ballad_documents() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "packaging" / "public_manifest.txt"

    selected = build_public_tree.expand_manifest(repo, build_public_tree.read_manifest(manifest))

    assert "ZeSolver_Ballad.md" in selected
    assert "la_ballade_de_ZeSolver.md" in selected
