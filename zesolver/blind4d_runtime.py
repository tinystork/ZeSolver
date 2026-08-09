from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_4D_MANIFEST_RELATIVE = Path("config") / "zeblind_4d_experimental_manifest.json"
ENV_4D_MANIFEST_PATH = "ZEBLIND_4D_MANIFEST"


def _package_config_root() -> Path | None:
    """Return the package resource root for config access (importlib.resources)."""
    try:
        from importlib.resources import files as _resource_files
        return Path(str(_resource_files("zesolver")))
    except Exception:
        return None


def runtime_resource_dirs() -> list[Path]:
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        try:
            roots.append(Path(meipass))
        except Exception:
            pass
    pkg = _package_config_root()
    if pkg is not None:
        roots.append(pkg)
    # Also include parent dir for source-tree fallback
    source_root = Path(__file__).resolve().parents[1]
    roots.append(source_root)
    seen: set[str] = set()
    ordered: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(root)
    return ordered


def resolve_default_4d_manifest_path(explicit: Path | str | None = None) -> Path:
    """Resolve the experimental 4D manifest without scanning arbitrary folders."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get(ENV_4D_MANIFEST_PATH)
    if env_value:
        return Path(env_value).expanduser().resolve()
    for base in runtime_resource_dirs():
        candidate = base / DEFAULT_4D_MANIFEST_RELATIVE
        if candidate.is_file():
            return candidate.resolve()
    # Final fallback: source-tree location
    source_root = Path(__file__).resolve().parents[1]
    return (source_root / DEFAULT_4D_MANIFEST_RELATIVE).resolve()


__all__ = [
    "DEFAULT_4D_MANIFEST_RELATIVE",
    "ENV_4D_MANIFEST_PATH",
    "resolve_default_4d_manifest_path",
    "runtime_resource_dirs",
]
