"""Cross-platform paths and storage preflight for ZeSolver catalog libraries."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping

from zeblindsolver.index_manifest_4d import sha256_file


MIB = 1024 * 1024
SAFETY_MARGIN_MIN_BYTES = 256 * MIB
SAFETY_MARGIN_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class PathValidationResult:
    ok: bool
    parent: Path
    destination: Path
    code: str | None = None
    message: str = ""
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class DistributionVolumeRequirement:
    role: str
    path: Path
    volume: str
    required_bytes: int
    available_bytes: int | None
    sufficient: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DistributionStoragePlan:
    cache_dir: Path
    destination: Path
    total_download_bytes: int
    cached_verified_bytes: int
    partial_bytes: int
    download_remaining_bytes: int
    installed_size_bytes: int
    safety_margin_bytes: int
    same_volume: bool
    requirements: tuple[DistributionVolumeRequirement, ...]

    @property
    def sufficient(self) -> bool:
        return all(item.sufficient for item in self.requirements)

    @property
    def temporary_peak_bytes(self) -> int:
        return max((item.required_bytes for item in self.requirements), default=0)


def default_library_parent(
    *,
    platform_name: str | None = None,
    home: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the user-visible default parent for installed libraries."""

    system = _system(platform_name)
    values = dict(os.environ if env is None else env)
    home_path = _home_path(home=home, env=values, system=system)
    if system == "windows":
        return home_path / "ZeSolverCatalog" / "libraries"
    return home_path / "ZeSolverCatalog" / "libraries"


def default_cache_root(
    *,
    platform_name: str | None = None,
    home: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the native per-user cache root for downloaded distribution assets."""

    system = _system(platform_name)
    values = dict(os.environ if env is None else env)
    home_path = _home_path(home=home, env=values, system=system)
    if system == "windows":
        base = values.get("LOCALAPPDATA")
        if base:
            return Path(base) / "ZeSolver" / "catalogs"
        return home_path / "AppData" / "Local" / "ZeSolver" / "catalogs"
    if system == "macos":
        return home_path / "Library" / "Caches" / "ZeSolver" / "catalogs"
    xdg = values.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else home_path / ".cache") / "ZeSolver" / "catalogs"


def resolve_library_destination(manifest: Any, parent: str | Path | None = None) -> Path:
    library_id = str(getattr(manifest, "library_id", "") or "").strip()
    version = str(getattr(manifest, "version", "") or "").strip()
    if not library_id or not version:
        raise ValueError("manifest must expose library_id and version")
    base = Path(parent).expanduser() if parent is not None else default_library_parent()
    return base / _slug(f"{library_id}-v{version}")


def validate_library_parent(
    parent: str | Path,
    manifest: Any,
    *,
    cache_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    application_roots: Iterable[str | Path] = (),
    platform_name: str | None = None,
    probe: bool = True,
) -> PathValidationResult:
    system = _system(platform_name)
    raw_parent = Path(parent).expanduser()
    destination = resolve_library_destination(manifest, raw_parent)
    code: str | None = None
    message = ""

    if not _is_absolute_for_system(raw_parent, system):
        code, message = "DISTRIBUTION_DESTINATION_INVALID", "Installation parent must be an absolute path."
    elif raw_parent.name.endswith(".partial") or ".partial-" in raw_parent.name or destination.name.endswith(".partial") or ".partial-" in destination.name:
        code, message = "DISTRIBUTION_DESTINATION_INVALID", "Installation path cannot be a partial staging path."
    elif system == "windows" and _windows_path_has_reserved_component(raw_parent):
        code, message = "DISTRIBUTION_DESTINATION_INVALID", "Installation path contains a reserved Windows component."
    elif raw_parent.exists() and not raw_parent.is_dir():
        code, message = "DISTRIBUTION_DESTINATION_INVALID", "Installation parent is occupied by a file."
    elif destination.exists():
        code, message = "DISTRIBUTION_DESTINATION_CONFLICT", "Final library destination already exists."
    elif cache_dir is not None and _same_or_inside(destination, Path(cache_dir).expanduser(), system=system):
        code, message = "DISTRIBUTION_DESTINATION_INSIDE_CACHE", "Final destination cannot be inside the download cache."
    elif cache_root is not None and _same_or_inside(destination, Path(cache_root).expanduser(), system=system):
        code, message = "DISTRIBUTION_DESTINATION_INSIDE_CACHE", "Final destination cannot be inside the download cache."
    elif any(_same_or_inside(destination, Path(root).expanduser(), system=system) for root in application_roots):
        code, message = "DISTRIBUTION_DESTINATION_INSIDE_APPLICATION", "Final destination cannot be inside the application directory."
    elif system == "macos" and any(part.lower().endswith(".app") for part in destination.parts):
        code, message = "DISTRIBUTION_DESTINATION_INSIDE_APPLICATION", "Final destination cannot be inside a macOS application bundle."

    if code is not None:
        return PathValidationResult(False, raw_parent, destination, code=code, message=message)

    try:
        raw_parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return PathValidationResult(
            False,
            raw_parent,
            destination,
            code="DISTRIBUTION_DESTINATION_NOT_WRITABLE",
            message=str(exc),
        )
    if not os.access(raw_parent, os.W_OK):
        return PathValidationResult(
            False,
            raw_parent,
            destination,
            code="DISTRIBUTION_DESTINATION_NOT_WRITABLE",
            message=f"Installation parent is not writable: {raw_parent}",
        )
    if probe:
        probe_dir = raw_parent / f".zesolver-write-test-{uuid.uuid4().hex[:10]}"
        probe_renamed = raw_parent / f".zesolver-write-test-{uuid.uuid4().hex[:10]}.ok"
        try:
            probe_dir.mkdir()
            probe_dir.replace(probe_renamed)
        except Exception as exc:
            return PathValidationResult(
                False,
                raw_parent,
                destination,
                code="DISTRIBUTION_DESTINATION_NOT_WRITABLE",
                message=str(exc),
            )
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
            shutil.rmtree(probe_renamed, ignore_errors=True)
    warning = "network_or_removable_storage" if _looks_network_or_removable(raw_parent, system) else None
    return PathValidationResult(True, raw_parent, destination, warning=warning)


def same_filesystem(
    left: str | Path,
    right: str | Path,
    *,
    platform_name: str | None = None,
    stat_func: Callable[[Path], os.stat_result] | None = None,
) -> bool:
    system = _system(platform_name)
    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    if system == "windows":
        left_drive = PureWindowsPath(str(left_path)).drive.lower()
        right_drive = PureWindowsPath(str(right_path)).drive.lower()
        if left_drive or right_drive:
            return left_drive == right_drive
    stat = stat_func or (lambda path: path.stat())
    try:
        return stat(_nearest_existing(left_path)).st_dev == stat(_nearest_existing(right_path)).st_dev
    except Exception:
        return _volume_fallback_key(left_path, system) == _volume_fallback_key(right_path, system)


def volume_key(
    path: str | Path,
    *,
    platform_name: str | None = None,
    stat_func: Callable[[Path], os.stat_result] | None = None,
) -> str:
    system = _system(platform_name)
    item = Path(path).expanduser()
    if system == "windows":
        drive = PureWindowsPath(str(item)).drive
        if drive:
            return drive.rstrip("\\/") or drive
    stat = stat_func or (lambda path: path.stat())
    try:
        st = stat(_nearest_existing(item))
        return f"dev:{st.st_dev}"
    except Exception:
        return _volume_fallback_key(item, system)


def build_storage_plan(
    install_plan: Any,
    *,
    disk_usage_func: Callable[[Path], shutil._ntuple_diskusage] | None = None,
    same_filesystem_func: Callable[[Path, Path], bool] | None = None,
    volume_key_func: Callable[[Path], str] | None = None,
) -> DistributionStoragePlan:
    cache_dir = Path(getattr(install_plan, "cache_dir")).expanduser()
    destination = Path(getattr(install_plan, "destination")).expanduser()
    components = tuple(getattr(install_plan, "components", ()) or ())
    total = sum(int(getattr(component, "size_bytes", 0) or 0) for component in components)
    cached_verified = 0
    partial = 0
    remaining = 0
    for component in components:
        size = int(getattr(component, "size_bytes", 0) or 0)
        expected_sha = str(getattr(component, "sha256", "") or "").lower()
        asset_name = str(getattr(component, "asset", "") or "")
        final = cache_dir / asset_name
        part = final.with_suffix(final.suffix + ".part")
        if final.is_file() and final.stat().st_size == size and expected_sha and sha256_file(final).lower() == expected_sha:
            cached_verified += size
            continue
        part_size = part.stat().st_size if part.is_file() else 0
        part_size = max(0, min(size, int(part_size)))
        partial += part_size
        remaining += max(0, size - part_size)
    installed = int(getattr(install_plan, "installed_size_bytes", None) or 0)
    if installed <= 0:
        installed = sum(int(getattr(component, "installed_size_bytes", 0) or 0) for component in components)
    if installed <= 0:
        installed = total
    same_volume = (
        same_filesystem_func(cache_dir, destination)
        if same_filesystem_func is not None
        else same_filesystem(cache_dir, destination)
    )
    margin_base = remaining + partial + installed
    margin = max(SAFETY_MARGIN_MIN_BYTES, int(margin_base * SAFETY_MARGIN_RATIO))
    usage = disk_usage_func or shutil.disk_usage
    volume = volume_key_func or (lambda path: volume_key(path))

    def _available(path: Path) -> int | None:
        try:
            return int(usage(_nearest_existing(path)).free)
        except Exception:
            return None

    requirements: list[DistributionVolumeRequirement] = []
    if same_volume:
        required = remaining + partial + installed + margin
        available = _available(destination.parent)
        requirements.append(
            DistributionVolumeRequirement(
                role="cache+library",
                path=destination.parent,
                volume=str(volume(destination.parent)),
                required_bytes=required,
                available_bytes=available,
                sufficient=available is not None and available >= required,
            )
        )
    else:
        cache_required = remaining + partial + margin
        cache_available = _available(cache_dir)
        requirements.append(
            DistributionVolumeRequirement(
                role="cache",
                path=cache_dir,
                volume=str(volume(cache_dir)),
                required_bytes=cache_required,
                available_bytes=cache_available,
                sufficient=cache_available is not None and cache_available >= cache_required,
            )
        )
        dest_required = installed + margin
        dest_available = _available(destination.parent)
        requirements.append(
            DistributionVolumeRequirement(
                role="library",
                path=destination.parent,
                volume=str(volume(destination.parent)),
                required_bytes=dest_required,
                available_bytes=dest_available,
                sufficient=dest_available is not None and dest_available >= dest_required,
            )
        )
    return DistributionStoragePlan(
        cache_dir=cache_dir,
        destination=destination,
        total_download_bytes=total,
        cached_verified_bytes=cached_verified,
        partial_bytes=partial,
        download_remaining_bytes=remaining,
        installed_size_bytes=installed,
        safety_margin_bytes=margin,
        same_volume=same_volume,
        requirements=tuple(requirements),
    )


def cache_reclaimable_bytes(cache_dir: str | Path, components: Iterable[Any]) -> int:
    root = Path(cache_dir).expanduser()
    total = 0
    for component in components:
        path = root / str(getattr(component, "asset", "") or "")
        if path.is_file():
            total += path.stat().st_size
    return total


def cleanup_distribution_cache(cache_dir: str | Path, components: Iterable[Any], *, active: bool = False) -> int:
    if active:
        raise RuntimeError("DISTRIBUTION_CACHE_CLEANUP_FAILED: distribution operation active")
    root = Path(cache_dir).expanduser()
    removed = 0
    for component in components:
        path = root / str(getattr(component, "asset", "") or "")
        part = path.with_suffix(path.suffix + ".part")
        if part.exists():
            continue
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            removed += size
    return removed


def file_manager_command(
    path: str | Path,
    *,
    platform_name: str | None = None,
    os_name: str | None = None,
) -> list[str] | None:
    target = Path(path).expanduser()
    if (os_name or os.name) == "nt":
        return None
    if (platform_name or sys.platform) == "darwin":
        return ["open", str(target)]
    return ["xdg-open", str(target)]


def open_in_file_manager(path: str | Path) -> bool:
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(str(target))
    command = file_manager_command(target)
    if command is None:
        os.startfile(str(target))  # type: ignore[attr-defined]
        return True
    subprocess.Popen(command)
    return True


def format_bytes_binary(value: int | None) -> str:
    if value is None:
        return "?"
    amount = float(max(0, int(value)))
    units = ("B", "KiB", "MiB", "Gio", "Tio")
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} B"
    return f"{amount:.2f} {unit}".replace(".", ",")


def _system(platform_name: str | None) -> str:
    value = (platform_name or platform.system() or sys.platform).strip().lower()
    if value.startswith(("win", "msys", "cygwin")):
        return "windows"
    if value in {"darwin", "mac", "macos"} or value.startswith("mac"):
        return "macos"
    return "linux"


def _home_path(*, home: str | Path | None, env: Mapping[str, str], system: str) -> Path:
    if home is not None:
        return Path(home).expanduser()
    if system == "windows" and env.get("USERPROFILE"):
        return Path(env["USERPROFILE"])
    if env.get("HOME"):
        return Path(env["HOME"]).expanduser()
    return Path.home()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return slug or "zesolver-library"


def _is_absolute_for_system(path: Path, system: str) -> bool:
    if system == "windows":
        return PureWindowsPath(str(path)).is_absolute()
    return path.is_absolute()


def _windows_path_has_reserved_component(path: Path) -> bool:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in PureWindowsPath(str(path)).parts:
        name = part.rstrip(" .").upper()
        if name in reserved:
            return True
    return False


def _same_or_inside(child: Path, parent: Path, *, system: str) -> bool:
    if system == "windows":
        child_text = str(PureWindowsPath(str(child))).rstrip("\\/").lower()
        parent_text = str(PureWindowsPath(str(parent))).rstrip("\\/").lower()
        return child_text == parent_text or child_text.startswith(parent_text + "\\")
    try:
        child_resolved = child.resolve(strict=False)
        parent_resolved = parent.resolve(strict=False)
        return child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    except Exception:
        child_text = str(child).rstrip("/")
        parent_text = str(parent).rstrip("/")
        return child_text == parent_text or child_text.startswith(parent_text + "/")


def _looks_network_or_removable(path: Path, system: str) -> bool:
    text = str(path)
    if system == "windows":
        return text.startswith("\\\\")
    return text.startswith(("/media/", "/mnt/", "/Volumes/"))


def _nearest_existing(path: Path) -> Path:
    current = Path(path).expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _volume_fallback_key(path: Path, system: str) -> str:
    if system == "windows":
        drive = PureWindowsPath(str(path)).drive
        return drive.lower() or str(PureWindowsPath(str(path)).anchor).lower()
    parts = Path(path).parts
    return parts[0] if parts else str(path)
