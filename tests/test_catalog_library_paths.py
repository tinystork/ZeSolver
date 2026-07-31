from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from shutil import _ntuple_diskusage

from zesolver.catalog_library.paths import (
    build_storage_plan,
    cache_reclaimable_bytes,
    cleanup_distribution_cache,
    default_cache_root,
    default_library_parent,
    format_bytes_binary,
    resolve_library_destination,
    validate_library_parent,
)


@dataclass(frozen=True)
class _Manifest:
    library_id: str = "zesolver-d50"
    version: str = "1.1.0"


@dataclass(frozen=True)
class _Component:
    id: str
    asset: str
    size_bytes: int
    sha256: str
    installed_size_bytes: int | None = None


@dataclass(frozen=True)
class _Plan:
    cache_dir: Path
    destination: Path
    components: tuple[_Component, ...]
    installed_size_bytes: int | None = None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_default_paths_are_native_user_locations_by_platform() -> None:
    win_env = {"USERPROFILE": r"C:\Users\Alice", "LOCALAPPDATA": r"C:\Users\Alice\AppData\Local"}
    assert str(default_library_parent(platform_name="Windows", env=win_env)).replace("/", "\\").endswith(
        r"C:\Users\Alice\ZeSolverCatalog\libraries"
    )
    assert str(default_cache_root(platform_name="Windows", env=win_env)).replace("/", "\\").endswith(
        r"C:\Users\Alice\AppData\Local\ZeSolver\catalogs"
    )
    fallback = str(default_cache_root(platform_name="Windows", env={"USERPROFILE": r"C:\Users\Alice"})).replace("/", "\\")
    assert fallback.endswith(r"C:\Users\Alice\AppData\Local\ZeSolver\catalogs")

    assert default_library_parent(platform_name="Linux", env={"HOME": "/home/alice"}) == Path("/home/alice/ZeSolverCatalog/libraries")
    assert default_cache_root(platform_name="Linux", env={"HOME": "/home/alice", "XDG_CACHE_HOME": "/cache"}) == Path(
        "/cache/ZeSolver/catalogs"
    )
    assert default_cache_root(platform_name="Linux", env={"HOME": "/home/alice"}) == Path("/home/alice/.cache/ZeSolver/catalogs")

    assert default_library_parent(platform_name="Darwin", env={"HOME": "/Users/alice"}) == Path("/Users/alice/ZeSolverCatalog/libraries")
    assert default_cache_root(platform_name="Darwin", env={"HOME": "/Users/alice"}) == Path(
        "/Users/alice/Library/Caches/ZeSolver/catalogs"
    )


def test_destination_is_manifest_derived_under_parent(tmp_path: Path) -> None:
    destination = resolve_library_destination(_Manifest(library_id="ZeSolver D50", version="1.1.0"), tmp_path / "libs")
    assert destination == tmp_path / "libs" / "ZeSolver-D50-v1.1.0"


def test_validate_library_parent_rejects_conflict_cache_and_application_paths(tmp_path: Path) -> None:
    manifest = _Manifest()
    cache_root = tmp_path / "cache"
    app_root = tmp_path / "app"
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    assert validate_library_parent(libraries, manifest, cache_root=cache_root, application_roots=[app_root]).ok

    destination = resolve_library_destination(manifest, libraries)
    destination.mkdir()
    conflict = validate_library_parent(libraries, manifest, cache_root=cache_root, application_roots=[app_root])
    assert not conflict.ok
    assert conflict.code == "DISTRIBUTION_DESTINATION_CONFLICT"

    inside_cache = validate_library_parent(cache_root, manifest, cache_root=cache_root, application_roots=[app_root], probe=False)
    assert not inside_cache.ok
    assert inside_cache.code == "DISTRIBUTION_DESTINATION_INSIDE_CACHE"

    inside_app = validate_library_parent(app_root / "data", manifest, cache_root=cache_root, application_roots=[app_root], probe=False)
    assert not inside_app.ok
    assert inside_app.code == "DISTRIBUTION_DESTINATION_INSIDE_APPLICATION"


def test_storage_plan_counts_validated_cache_and_partials_without_double_counting(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    destination = tmp_path / "libraries" / "zesolver-d50-v1.1.0"
    cache.mkdir(parents=True)
    (tmp_path / "libraries").mkdir()
    good_data = b"abc"
    (cache / "good.zip").write_bytes(good_data)
    (cache / "partial.zip.part").write_bytes(b"12")
    components = (
        _Component("good", "good.zip", len(good_data), _sha(good_data)),
        _Component("partial", "partial.zip", 10, "0" * 64),
        _Component("missing", "missing.zip", 5, "1" * 64),
    )
    plan = _Plan(cache, destination, components, installed_size_bytes=100)
    storage = build_storage_plan(
        plan,
        same_filesystem_func=lambda _a, _b: True,
        volume_key_func=lambda _path: "dev:1",
        disk_usage_func=lambda _path: _ntuple_diskusage(total=1_000_000_000, used=0, free=1_000_000_000),
    )

    assert storage.cached_verified_bytes == 3
    assert storage.partial_bytes == 2
    assert storage.download_remaining_bytes == 13
    assert storage.installed_size_bytes == 100
    assert storage.same_volume is True
    assert len(storage.requirements) == 1
    assert storage.sufficient


def test_storage_plan_splits_cache_and_destination_volumes(tmp_path: Path) -> None:
    component = _Component("asset", "asset.zip", 20, "0" * 64)
    plan = _Plan(tmp_path / "cache", tmp_path / "dest" / "zesolver-d50-v1.1.0", (component,), installed_size_bytes=30)
    storage = build_storage_plan(
        plan,
        same_filesystem_func=lambda _a, _b: False,
        volume_key_func=lambda path: "cache" if "cache" in str(path) else "dest",
        disk_usage_func=lambda _path: _ntuple_diskusage(total=1_000_000_000, used=0, free=1_000_000_000),
    )

    assert storage.same_volume is False
    assert [item.role for item in storage.requirements] == ["cache", "library"]
    assert all(item.sufficient for item in storage.requirements)


def test_cache_cleanup_removes_only_current_validated_assets_and_preserves_partials(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "asset.zip").write_bytes(b"asset")
    (cache / "active.zip").write_bytes(b"active")
    (cache / "active.zip.part").write_bytes(b"partial")
    (tmp_path / "other-version.zip").write_bytes(b"other")
    components = (
        _Component("asset", "asset.zip", 5, _sha(b"asset")),
        _Component("active", "active.zip", 6, _sha(b"active")),
    )

    assert cache_reclaimable_bytes(cache, components) == 11
    removed = cleanup_distribution_cache(cache, components)

    assert removed == 5
    assert not (cache / "asset.zip").exists()
    assert (cache / "active.zip").exists()
    assert (cache / "active.zip.part").exists()


def test_format_bytes_binary_uses_requested_units() -> None:
    assert format_bytes_binary(1024) == "1,00 KiB"
    assert format_bytes_binary(1024**3).endswith("Gio")
