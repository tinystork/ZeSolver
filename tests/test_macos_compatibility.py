from __future__ import annotations

import hashlib
import io
import multiprocessing
import os
from pathlib import Path

import pytest

from catalog_resource_helpers import write_catalog_library, write_fake_4d_index, write_strict_manifest, strict_entry
from zesolver.catalog_library import CatalogLibrary
from zesolver.catalog_library.distribution import ResumableAssetDownloader
from zesolver.catalog_library.paths import (
    default_cache_root,
    default_library_parent,
    file_manager_command,
    validate_library_parent,
)
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.macos_preflight import _check_cupy_optional, _check_process_pool, _check_temp_paths_and_fits


class _Manifest:
    library_id = "ZeSolver D50"
    version = "1.1.0"


def _spawn_square(value: int) -> int:
    return value * value


def test_macos_user_cache_library_paths_accept_spaces_unicode_and_app_bundle_rejection(tmp_path: Path) -> None:
    home = tmp_path / "Utilisateur Étoile"
    env = {"HOME": str(home)}

    assert default_cache_root(platform_name="Darwin", env=env) == home / "Library" / "Caches" / "ZeSolver" / "catalogs"
    assert default_library_parent(platform_name="Darwin", env=env) == home / "ZeSolverCatalog" / "libraries"

    parent = home / "Documents" / "Bibliothèques ZeSolver"
    result = validate_library_parent(parent, _Manifest(), platform_name="Darwin")
    assert result.ok
    assert result.destination.name == "ZeSolver-D50-v1.1.0"

    app_parent = home / "Applications" / "ZeSolver.app" / "Contents" / "Resources" / "Catalogs"
    rejected = validate_library_parent(app_parent, _Manifest(), platform_name="Darwin", probe=False)
    assert not rejected.ok
    assert rejected.code == "DISTRIBUTION_DESTINATION_INSIDE_APPLICATION"


def test_file_manager_command_routes_darwin_to_open_without_shell(tmp_path: Path) -> None:
    path = tmp_path / "Dossier avec espaces é"
    path.mkdir()

    command = file_manager_command(path, platform_name="darwin", os_name="posix")

    assert command == ["open", str(path)]


def test_file_manager_command_routes_linux_and_windows_without_confusing_macos(tmp_path: Path) -> None:
    path = tmp_path / "folder"
    path.mkdir()

    assert file_manager_command(path, platform_name="linux", os_name="posix") == ["xdg-open", str(path)]
    assert file_manager_command(path, platform_name="win32", os_name="nt") is None


def test_spawn_context_runs_representative_worker() -> None:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        assert pool.apply(_spawn_square, (9,)) == 81


def test_macos_preflight_spawn_and_temp_fits_checks() -> None:
    ok_pool, detail_pool = _check_process_pool()
    ok_paths, detail_paths = _check_temp_paths_and_fits()

    assert ok_pool, detail_pool
    assert "spawn" in detail_pool
    assert ok_paths, detail_paths


def test_qt_offscreen_widget_and_theme_can_start_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from zesolver.gui_theme import ThemeController

    app = QtWidgets.QApplication.instance()
    if app is not None and not isinstance(app, QtWidgets.QApplication):
        pytest.skip("QCoreApplication already active")
    app = app or QtWidgets.QApplication([])
    controller = ThemeController(app, initial_mode="system", system_scheme_getter=lambda: "light")
    controller.apply("dark", persist=False)
    widget = QtWidgets.QWidget()
    widget.setWindowTitle("macOS CI smoke")
    widget.show()
    app.processEvents()
    widget.close()
    controller.apply("system", persist=False)
    app.setProperty("zesolver_system_palette", None)
    app.processEvents()

    assert app.property("zesolver_theme_mode") == "system"


def test_catalog_library_fixture_resolves_on_macos_style_unicode_path(tmp_path: Path) -> None:
    root = tmp_path / "Bibliothèque ZeSolver é"
    index = write_fake_4d_index(tmp_path / "d50_TEST_S_q.npz", "d50_TEST")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("idx", index, "d50_TEST")])
    write_catalog_library(root, index_paths=[index], strict_manifest_path=manifest)

    library = CatalogLibrary.open(root)
    report = library.validate()
    resources = resolve_catalog_resources(catalog_library=library)

    assert report.capabilities.near is True
    assert resources.source == "library"
    assert resources.near_available is True
    assert resources.blind4d_available is True


class _Response:
    status = 206
    headers = {"Content-Range": "bytes 4-10/11", "ETag": '"macos-fixture"'}

    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class _Backend:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.headers_seen: list[dict[str, str]] = []

    def open(self, _url: str, *, headers: dict[str, str] | None = None) -> _Response:
        self.headers_seen.append(dict(headers or {}))
        return _Response(self.payload)


def test_download_resume_works_under_path_with_spaces_unicode_and_apostrophe(tmp_path: Path) -> None:
    content = b"hello-macos"
    root = tmp_path / "Cache téléchargement é 'test'"
    destination = root / "asset.zip"
    part = destination.with_suffix(".zip.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(content[:4])
    backend = _Backend(content[4:])

    result = ResumableAssetDownloader(http_backend=backend, retries=0, chunk_size=64 * 1024).download(
        url="https://example.invalid/asset.zip",
        destination=destination,
        expected_size=len(content),
        expected_sha256=hashlib.sha256(content).hexdigest(),
        component_id="metadata",
        version="test",
        overall_current=1,
        overall_total=1,
    )

    assert result == destination
    assert destination.read_bytes() == content
    assert backend.headers_seen[0]["Range"] == "bytes=4-"


def test_cupy_absence_is_optional() -> None:
    ok, detail = _check_cupy_optional()

    assert ok
    assert detail
