from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zeblindsolver.index_manifest_4d import IndexManifestError, load_4d_index_manifest
from zeblindsolver.quad_index_builder import validate_index as validate_legacy_near_index


@dataclass(frozen=True, slots=True)
class GuiCatalogPathValidation:
    ok: bool
    code: str
    message: str
    path: Path | None = None


def _path(value: str | Path | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


def _has_astap_shards(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(path.glob("*.1476")) or any(path.glob("*.290"))
    except Exception:
        return False


def validate_catalog_library_root(value: str | Path | None) -> GuiCatalogPathValidation:
    path = _path(value)
    if path is None:
        return GuiCatalogPathValidation(False, "CATALOG_LIBRARY_PATH_MISSING", "Choose a ZeSolver library root.")
    if path.is_file():
        return GuiCatalogPathValidation(
            False,
            "CATALOG_LIBRARY_ROOT_REQUIRED",
            "A ZeSolver library expects the folder containing catalog.json, not a file.",
            path,
        )
    if not path.exists():
        return GuiCatalogPathValidation(False, "CATALOG_LIBRARY_PATH_MISSING", "The selected library folder does not exist.", path)
    if (path / "catalog.json").is_file():
        return GuiCatalogPathValidation(True, "CATALOG_LIBRARY_OK", "ZeSolver library root.", path)
    if (path / "manifest.json").is_file():
        return GuiCatalogPathValidation(
            False,
            "LEGACY_NEAR_INDEX_USED_AS_CATALOG_LIBRARY",
            "This folder looks like a historical Near index. A ZeSolver library must contain catalog.json.",
            path,
        )
    if _has_astap_shards(path):
        return GuiCatalogPathValidation(
            False,
            "ASTAP_SOURCE_USED_AS_CATALOG_LIBRARY",
            "This folder looks like an ASTAP/HNSKY source. A ZeSolver library must contain catalog.json.",
            path,
        )
    return GuiCatalogPathValidation(False, "CATALOG_LIBRARY_MANIFEST_MISSING", "catalog.json not found in the selected library folder.", path)


def validate_astap_root(value: str | Path | None) -> GuiCatalogPathValidation:
    path = _path(value)
    if path is None:
        return GuiCatalogPathValidation(False, "ASTAP_ROOT_MISSING", "Choose a historical ASTAP source folder.")
    if path.is_file():
        return GuiCatalogPathValidation(False, "ASTAP_ROOT_DIRECTORY_REQUIRED", "The ASTAP source field expects a folder.", path)
    if not path.exists():
        return GuiCatalogPathValidation(False, "ASTAP_ROOT_MISSING", "The ASTAP source folder does not exist.", path)
    if _has_astap_shards(path):
        return GuiCatalogPathValidation(True, "ASTAP_ROOT_OK", "ASTAP/HNSKY source folder.", path)
    return GuiCatalogPathValidation(False, "ASTAP_ROOT_NO_SHARDS", "Expected *.1476 or *.290 files in the historical ASTAP source folder.", path)


def validate_legacy_near_index_root(value: str | Path | None) -> GuiCatalogPathValidation:
    path = _path(value)
    if path is None:
        return GuiCatalogPathValidation(False, "LEGACY_NEAR_INDEX_MISSING", "Choose a historical Near index folder.")
    if path.is_file():
        return GuiCatalogPathValidation(False, "LEGACY_NEAR_INDEX_DIRECTORY_REQUIRED", "The historical Near index field expects a folder.", path)
    if not path.exists():
        return GuiCatalogPathValidation(False, "LEGACY_NEAR_INDEX_MISSING", "The historical Near index folder does not exist.", path)
    if (path / "catalog.json").is_file() and not (path / "manifest.json").is_file():
        return GuiCatalogPathValidation(
            False,
            "CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX",
            "This folder is a ZeSolver library, not a historical Near index. Use it in the ZeSolver library field.",
            path,
        )
    if not (path / "manifest.json").is_file():
        return GuiCatalogPathValidation(False, "LEGACY_NEAR_MANIFEST_MISSING", "Expected manifest.json in the historical Near index folder.", path)
    try:
        info = validate_legacy_near_index(path)
    except Exception as exc:
        return GuiCatalogPathValidation(False, "LEGACY_NEAR_INDEX_INVALID", f"Historical Near index invalid: {exc}", path)
    if not bool(info.get("manifest_ok")):
        return GuiCatalogPathValidation(False, "LEGACY_NEAR_INDEX_INVALID", "Historical Near index manifest is not compatible.", path)
    return GuiCatalogPathValidation(True, "LEGACY_NEAR_INDEX_OK", "Historical Near index folder.", path)


def validate_blind4d_manifest_file(value: str | Path | None) -> GuiCatalogPathValidation:
    path = _path(value)
    if path is None:
        return GuiCatalogPathValidation(False, "BLIND4D_MANIFEST_MISSING", "Choose an external Blind 4D manifest JSON file.")
    if path.is_dir():
        return GuiCatalogPathValidation(False, "BLIND4D_MANIFEST_FILE_REQUIRED", "The external Blind 4D manifest field expects a JSON file, not a folder.", path)
    if not path.exists():
        return GuiCatalogPathValidation(False, "BLIND4D_MANIFEST_MISSING", "The external Blind 4D manifest file does not exist.", path)
    try:
        load_4d_index_manifest(path)
    except IndexManifestError as exc:
        return GuiCatalogPathValidation(False, "BLIND4D_EXTERNAL_MANIFEST_INVALID", f"External Blind 4D manifest invalid: {exc}", path)
    return GuiCatalogPathValidation(True, "BLIND4D_EXTERNAL_MANIFEST_OK", "External Blind 4D manifest.", path)

