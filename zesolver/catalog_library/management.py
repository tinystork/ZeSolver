"""Non-GUI management facade for ZeSolver catalogue libraries."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap
from zeblindsolver.index_manifest_4d import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    IndexManifestError,
    load_4d_index_manifest,
    sha256_file,
)
from zeblindsolver.quad_index_4d import Quad4DIndex
from zeblindsolver.astap_db_reader import iter_tiles
from zewcs290.catalog290 import CatalogFamilySpec, FAMILY_SPECS

from .adoption import CatalogLibraryAdoptionPlan
from .atomic_adoption import CatalogLibraryAdoptionWriter
from .blind4d_view import build_blind4d_manifest_view
from .manifest import CatalogLibrary, CatalogLibraryError
from .models import CatalogStatus


ProgressCallback = Callable[["LibraryManagementProgress"], None]
CancelCallback = Callable[[], bool]

PACKAGE_METADATA_NAMES = (
    "zesolver-library-package.json",
    "library_package.json",
)


class CatalogLibraryManagementError(RuntimeError):
    """Raised for user-facing catalogue management failures."""


class CatalogLibraryManagementCancelled(CatalogLibraryManagementError):
    """Raised when a management operation is cancelled cooperatively."""


@dataclass(frozen=True, slots=True)
class LibraryManagementProgress:
    stage: str
    message: str = ""
    family: str | None = None
    overall_current: int = 0
    overall_total: int = 0
    step_current: int = 0
    step_total: int = 0


@dataclass(frozen=True, slots=True)
class AstapFamilyInfo:
    family: str
    root: Path
    shard_count: int
    size_bytes: int
    status: str


@dataclass(frozen=True, slots=True)
class LibraryCreateOptions:
    astap_root: Path
    destination: Path
    families: tuple[str, ...] = ()
    storage_policy: str = "reference"
    mode: str = "standard"
    mag_cap: float | None = None
    source_max_stars: int | None = None
    max_stars_per_tile: int | None = None
    max_quads_per_tile: int | None = None
    workers: int | None = None
    quad_storage: str = "npz"
    compression: str = "compressed"


@dataclass(frozen=True, slots=True)
class LibraryInstallOptions:
    package_path: Path
    destination: Path


@dataclass(frozen=True, slots=True)
class LibraryOperationResult:
    library_root: Path
    status: CatalogStatus
    catalog_json: Path
    blind4d_manifest: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LibraryDiagnosticItem:
    element: str
    state: str
    detail: str


@dataclass(frozen=True, slots=True)
class LibraryRepairPlan:
    library_root: Path
    actions: tuple[str, ...]
    items: tuple[LibraryDiagnosticItem, ...]


@dataclass(frozen=True, slots=True)
class LibraryAnalysisResult:
    library_root: Path
    status: str
    items: tuple[LibraryDiagnosticItem, ...]
    repair_plan: LibraryRepairPlan


class CatalogLibraryManagementService:
    """High-level library lifecycle service used by GUI and tests.

    The service intentionally delegates manifest adoption, final validation and
    Blind 4D view validation to the existing catalogue APIs.
    """

    def __init__(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
        disk_usage: Callable[[Path], shutil._ntuple_diskusage] | None = None,
    ) -> None:
        self._progress_callback = progress_callback
        self._cancel_callback = cancel_callback
        self._disk_usage = disk_usage or shutil.disk_usage

    def detect_astap_families(self, root: str | Path) -> tuple[AstapFamilyInfo, ...]:
        base = Path(root).expanduser().resolve()
        if not base.exists() or not base.is_dir():
            raise CatalogLibraryManagementError(f"ASTAP_SOURCE_MISSING: {base}")
        candidates = [base]
        candidates.extend(path for path in sorted(base.iterdir()) if path.is_dir())
        by_family: dict[str, AstapFamilyInfo] = {}
        for candidate in candidates:
            for family, spec in sorted(FAMILY_SPECS.items()):
                files = _find_family_files(candidate, spec)
                if not files:
                    continue
                size = sum(path.stat().st_size for path in files if path.is_file())
                current = by_family.get(family)
                info = AstapFamilyInfo(
                    family=family,
                    root=candidate,
                    shard_count=len(files),
                    size_bytes=size,
                    status="ready",
                )
                if current is None or info.shard_count > current.shard_count:
                    by_family[family] = info
        return tuple(by_family[key] for key in sorted(by_family))

    def create_from_astap(self, options: LibraryCreateOptions) -> LibraryOperationResult:
        destination = Path(options.destination).expanduser().resolve()
        source_root = Path(options.astap_root).expanduser().resolve()
        families = self._selected_families(source_root, options.families)
        self._check_cancelled()
        self._validate_create_paths(source_root, destination)
        self._ensure_destination_available(destination)

        staging = self._staging_path(destination)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        created_final = False
        try:
            self._emit("analyze_sources", "Analysing ASTAP sources", overall_current=1, overall_total=7)
            astap_roots = self._prepare_sources(
                source_root,
                staging,
                families=families,
                storage_policy=options.storage_policy,
            )
            self._check_cancelled()
            self._emit("build_blind4d", "Building Blind 4D indexes", overall_current=4, overall_total=7)
            blind_manifest = self._build_blind4d_indexes(staging, astap_roots, families=families, options=options)
            self._check_cancelled()
            self._emit("publish", "Publishing library", overall_current=6, overall_total=7)
            self._publish_staging(staging, destination)
            created_final = True
            final_astap_roots = self._final_astap_roots(destination, source_root, astap_roots, options.storage_policy)
            final_blind_manifest = destination / "indexes" / "blind4d" / "strict_4d_manifest.json"
            if blind_manifest is not None and not final_blind_manifest.exists():
                raise CatalogLibraryManagementError("BLIND4D_MANIFEST_NOT_PUBLISHED")
            self._write_catalog(
                destination,
                astap_roots=final_astap_roots,
                families=families,
                blind4d_manifest=final_blind_manifest if final_blind_manifest.exists() else None,
                generated_by="CatalogLibraryManagementService.create_from_astap",
            )
            result = self._validate_result(destination, final_blind_manifest if final_blind_manifest.exists() else None)
            self._emit("complete", "Library ready", overall_current=7, overall_total=7)
            return result
        except CatalogLibraryManagementCancelled:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if created_final and destination.exists() and not (destination / "catalog.json").exists():
                shutil.rmtree(destination, ignore_errors=True)
            raise
        except Exception as exc:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if created_final and destination.exists() and not (destination / "catalog.json").exists():
                shutil.rmtree(destination, ignore_errors=True)
            if isinstance(exc, CatalogLibraryManagementError):
                raise
            raise CatalogLibraryManagementError(str(exc)) from exc

    def install_package(self, options: LibraryInstallOptions) -> LibraryOperationResult:
        package_path = Path(options.package_path).expanduser().resolve()
        destination = Path(options.destination).expanduser().resolve()
        if not package_path.exists():
            raise CatalogLibraryManagementError(f"PACKAGE_MISSING: {package_path}")
        self._ensure_destination_available(destination)
        staging = self._staging_path(destination)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            self._emit("read_package", "Reading package metadata", overall_current=1, overall_total=6)
            extracted = staging / "package"
            metadata, library_root = self._materialize_package(package_path, extracted)
            self._check_package_metadata(metadata)
            self._check_free_space(destination, int(metadata.get("installed_size_bytes") or _directory_size(library_root)))
            self._emit("verify_hashes", "Verifying package hashes", overall_current=3, overall_total=6)
            self._verify_package_hashes(library_root, metadata)
            self._emit("validate_library", "Validating package library", overall_current=4, overall_total=6)
            library = CatalogLibrary.open(library_root)
            report = library.validate()
            if report.status in {CatalogStatus.CORRUPT, CatalogStatus.INCOMPATIBLE, CatalogStatus.MISSING}:
                raise CatalogLibraryManagementError(f"PACKAGE_LIBRARY_INVALID: {report.status.value}")
            try:
                if report.capabilities.blind4d:
                    view = build_blind4d_manifest_view(library)
                    if not view.valid:
                        raise CatalogLibraryManagementError(
                            "PACKAGE_BLIND4D_VIEW_INVALID: "
                            + ", ".join(issue.code for issue in view.errors)
                        )
            except Exception as exc:
                raise CatalogLibraryManagementError(f"PACKAGE_BLIND4D_VIEW_INVALID: {exc}") from exc
            self._emit("publish", "Installing library", overall_current=5, overall_total=6)
            self._publish_staging(library_root, destination)
            result = self._validate_result(destination, None, metadata=metadata)
            self._emit("complete", "Library installed", overall_current=6, overall_total=6)
            return result
        except Exception as exc:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, CatalogLibraryManagementError):
                raise
            raise CatalogLibraryManagementError(str(exc)) from exc
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def analyze_library(self, library_root: str | Path) -> LibraryAnalysisResult:
        root = Path(library_root).expanduser().resolve()
        items: list[LibraryDiagnosticItem] = []
        actions: list[str] = []
        try:
            library = CatalogLibrary.open(root)
            report = library.validate()
            items.append(LibraryDiagnosticItem("catalog.json", "ready", f"schema v{library.manifest.schema_version}"))
            items.append(
                LibraryDiagnosticItem(
                    "ASTAP sources",
                    "ready" if report.capabilities.near else "missing",
                    ", ".join(sorted({source.family for source in library.manifest.sources})) or "-",
                )
            )
            items.append(LibraryDiagnosticItem("ZeNear", "ready" if report.capabilities.near else "unavailable", "ASTAP-native source"))
            blind_state = "ready" if report.capabilities.blind4d else "missing"
            items.append(LibraryDiagnosticItem("Blind 4D", blind_state, f"{len(library.manifest.derived_indexes)} index(es)"))
            items.append(
                LibraryDiagnosticItem(
                    "Global Blind 4D coverage",
                    "yes" if report.capabilities.all_sky_blind4d else "no",
                    report.coverage.status.value,
                )
            )
            if report.issues:
                detail = "; ".join(f"{issue.code}:{issue.severity.value}" for issue in report.issues)
                items.append(LibraryDiagnosticItem("Integrity", "warning" if report.status not in {CatalogStatus.CORRUPT, CatalogStatus.INCOMPATIBLE} else "invalid", detail))
            else:
                items.append(LibraryDiagnosticItem("Integrity", "valid", "No validation issues"))
            if report.capabilities.near and not report.capabilities.blind4d:
                actions.append("build_missing_blind4d")
            if any(issue.code.endswith("MISSING") for issue in report.issues):
                actions.append("locate_missing_resources")
            status = report.status.value
        except CatalogLibraryError as exc:
            items.append(LibraryDiagnosticItem("catalog.json", "invalid", str(exc)))
            actions.append("choose_valid_library")
            status = "INVALID"
        plan = LibraryRepairPlan(root, tuple(dict.fromkeys(actions)), tuple(items))
        return LibraryAnalysisResult(root, status, tuple(items), plan)

    def repair_library(self, plan: LibraryRepairPlan) -> LibraryOperationResult:
        root = Path(plan.library_root).expanduser().resolve()
        actions = set(plan.actions)
        if not actions:
            return self._validate_result(root, None)
        if "build_missing_blind4d" not in actions:
            raise CatalogLibraryManagementError("NO_AUTOMATIC_REPAIR_AVAILABLE")
        library = CatalogLibrary.open(root)
        astap_roots = tuple(source.path.resolved for source in library.manifest.sources)
        families = tuple(sorted({source.family for source in library.manifest.sources}))
        if not astap_roots or not families:
            raise CatalogLibraryManagementError("REPAIR_ASTAP_SOURCE_MISSING")
        staging = self._staging_path(root / "repair")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            options = LibraryCreateOptions(astap_root=astap_roots[0], destination=root, families=families)
            manifest = self._build_blind4d_indexes(staging, astap_roots, families=families, options=options)
            target_dir = root / "indexes" / "blind4d"
            if target_dir.exists():
                raise CatalogLibraryManagementError("REPAIR_TARGET_ALREADY_EXISTS")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging / "indexes" / "blind4d"), str(target_dir))
            final_manifest = target_dir / "strict_4d_manifest.json"
            self._write_catalog(
                root,
                astap_roots=astap_roots,
                families=families,
                blind4d_manifest=final_manifest if manifest is not None else None,
                mode="replace",
                generated_by="CatalogLibraryManagementService.repair_library",
            )
            return self._validate_result(root, final_manifest)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _selected_families(self, source_root: Path, requested: Iterable[str]) -> tuple[str, ...]:
        detected = self.detect_astap_families(source_root)
        available = {item.family: item for item in detected}
        families = _normalize_requested_families(requested) or tuple(available)
        if not families:
            raise CatalogLibraryManagementError(
                f"ASTAP_NO_FAMILIES_DETECTED: no supported ASTAP tiles found under {source_root}. "
                "Install at least one supported ASTAP base or choose another root."
            )
        missing = [family for family in families if family not in available]
        if missing:
            raise CatalogLibraryManagementError(
                "ASTAP_FAMILY_MISSING: requested family {families} was not found under {root}; "
                "deselect it or install the corresponding ASTAP base.".format(
                    families=", ".join(missing),
                    root=source_root,
                )
            )
        return families

    def _prepare_sources(self, source_root: Path, staging: Path, *, families: tuple[str, ...], storage_policy: str) -> tuple[Path, ...]:
        policy = str(storage_policy or "reference").strip().lower()
        detected = {item.family: item for item in self.detect_astap_families(source_root)}
        if policy == "reference":
            return tuple(detected[family].root for family in families)
        if policy != "copy":
            raise CatalogLibraryManagementError(f"STORAGE_POLICY_INVALID: {storage_policy}")
        target = staging / "sources" / "astap"
        target.mkdir(parents=True, exist_ok=True)
        for family in families:
            info = detected[family]
            spec = FAMILY_SPECS[family]
            for path in sorted(info.root.glob(spec.glob_pattern()), key=lambda item: item.name):
                self._check_cancelled()
                shutil.copy2(path, target / path.name)
        return (target,)

    def _build_blind4d_indexes(
        self,
        staging: Path,
        astap_roots: tuple[Path, ...],
        *,
        families: tuple[str, ...],
        options: LibraryCreateOptions,
    ) -> Path | None:
        entries: list[dict[str, Any]] = []
        out_dir = staging / "indexes" / "blind4d"
        out_dir.mkdir(parents=True, exist_ok=True)
        for family_index, family in enumerate(families, start=1):
            root = self._root_for_family(astap_roots, family)
            tile_keys = tuple(meta.key for meta in iter_tiles(root, families=(family,)) if meta.family == family)
            if not tile_keys:
                raise CatalogLibraryManagementError(
                    f"ASTAP_FAMILY_EMPTY: selected family {family} has no usable tiles under {root}"
                )
            output = out_dir / f"{family}_4d.npz"
            defaults = Astap4DBuildConfig()
            config = Astap4DBuildConfig(
                family=family,
                tile_keys=tile_keys,
                mag_cap=defaults.mag_cap if options.mag_cap is None else options.mag_cap,
                source_max_stars=defaults.source_max_stars if options.source_max_stars is None else options.source_max_stars,
                max_stars_per_tile=defaults.max_stars_per_tile if options.max_stars_per_tile is None else options.max_stars_per_tile,
                max_quads_per_tile=defaults.max_quads_per_tile if options.max_quads_per_tile is None else options.max_quads_per_tile,
            )

            def _progress(event: dict[str, Any], *, fam: str = family, fam_index: int = family_index) -> None:
                self._emit(
                    "build_blind4d",
                    str(event.get("stage") or "building"),
                    family=fam,
                    overall_current=fam_index,
                    overall_total=len(families),
                    step_current=int(event.get("ordinal") or 0),
                    step_total=int(event.get("total") or 0),
                )

            result = build_4d_index_from_astap(
                root,
                output,
                config=config,
                progress_callback=_progress,
                cancel_callback=self._is_cancelled,
            )
            loaded = Quad4DIndex.load(result)
            entries.append(
                {
                    "id": f"direct-{family}",
                    "enabled": True,
                    "path": result.name,
                    "filename": result.name,
                    "quad_schema": loaded.metadata["schema"],
                    "index_version": int(loaded.metadata["version"]),
                    "level": loaded.metadata["level"],
                    "tile_keys": list(loaded.tile_keys),
                    "star_count": int(loaded.catalog_ra_dec.shape[0]),
                    "quad_count": int(loaded.codes_4d.shape[0]),
                    "sampler_tag": loaded.metadata["sampler_tag"],
                    "sha256": sha256_file(result),
                    "code_tol_recommended": loaded.metadata.get("code_tol_recommended"),
                    "catalog_source": "astap_raw",
                }
            )
        if not entries:
            return None
        manifest = out_dir / "strict_4d_manifest.json"
        payload = {"schema": MANIFEST_SCHEMA, "manifest_version": MANIFEST_VERSION, "indexes": entries}
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        load_4d_index_manifest(manifest)
        return manifest

    def _root_for_family(self, roots: tuple[Path, ...], family: str) -> Path:
        spec = FAMILY_SPECS[family]
        for root in roots:
            if _find_family_files(root, spec):
                return root
        raise CatalogLibraryManagementError(f"ASTAP_FAMILY_ROOT_MISSING: {family}")

    def _write_catalog(
        self,
        library_root: Path,
        *,
        astap_roots: tuple[Path, ...],
        families: tuple[str, ...],
        blind4d_manifest: Path | None,
        mode: str = "create",
        generated_by: str,
    ) -> None:
        plan = CatalogLibraryAdoptionPlan.reference_existing(
            library_root=library_root,
            astap_roots=astap_roots,
            astap_families=families,
            blind4d_manifest=blind4d_manifest,
            fingerprint_policy="fast",
            generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        payload = dict(plan.manifest_preview)
        payload["library_id"] = _library_id_from_root(library_root)
        payload["created_by"] = generated_by
        payload.setdefault("provenance", {})
        payload["provenance"] = {
            **dict(payload["provenance"]),
            "astap_credit": "ASTAP star database by Han Kleijn",
            "astrometry_credit": "Astrometry.net concepts and components credited to the upstream project",
            "redistribution_notice": "Package redistribution must preserve upstream licences and notices.",
        }
        plan = type(plan)(
            library_root=plan.library_root,
            status=plan.status,
            sources=plan.sources,
            indexes=plan.indexes,
            compatibility_resources=plan.compatibility_resources,
            coverage=plan.coverage,
            warnings=plan.warnings,
            errors=plan.errors,
            repair_actions=plan.repair_actions,
            manifest_preview=payload,
            telemetry=plan.telemetry,
        )
        expected_sha = sha256_file(library_root / "catalog.json") if mode == "replace" and (library_root / "catalog.json").is_file() else None
        CatalogLibraryAdoptionWriter.commit(plan, mode=mode, expected_existing_sha256=expected_sha)

    def _validate_result(self, root: Path, blind_manifest: Path | None, *, metadata: dict[str, Any] | None = None) -> LibraryOperationResult:
        library = CatalogLibrary.open(root)
        report = library.validate()
        if report.status in {CatalogStatus.CORRUPT, CatalogStatus.INCOMPATIBLE, CatalogStatus.MISSING}:
            raise CatalogLibraryManagementError(f"LIBRARY_INVALID: {report.status.value}")
        if blind_manifest is not None:
            try:
                view = build_blind4d_manifest_view(library)
                if not view.valid:
                    raise CatalogLibraryManagementError(
                        "BLIND4D_VIEW_INVALID: "
                        + ", ".join(issue.code for issue in view.errors)
                    )
            except (CatalogLibraryError, IndexManifestError, Exception) as exc:
                raise CatalogLibraryManagementError(f"BLIND4D_VIEW_INVALID: {exc}") from exc
        return LibraryOperationResult(
            library_root=root,
            status=report.status,
            catalog_json=root / "catalog.json",
            blind4d_manifest=blind_manifest,
            metadata=dict(metadata or {}),
        )

    def _materialize_package(self, package_path: Path, target: Path) -> tuple[dict[str, Any], Path]:
        if package_path.is_dir():
            metadata = self._read_package_metadata(package_path)
            library_root = self._package_library_root(package_path)
            copied = target / "library"
            shutil.copytree(library_root, copied, symlinks=False)
            return metadata, copied
        target.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(package_path):
            self._extract_zip(package_path, target)
        elif tarfile.is_tarfile(package_path):
            self._extract_tar(package_path, target)
        else:
            raise CatalogLibraryManagementError("PACKAGE_ARCHIVE_UNSUPPORTED")
        metadata = self._read_package_metadata(target)
        return metadata, self._package_library_root(target)

    def _read_package_metadata(self, root: Path) -> dict[str, Any]:
        for name in PACKAGE_METADATA_NAMES:
            path = root / name
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise CatalogLibraryManagementError("PACKAGE_METADATA_INVALID")
                return payload
        raise CatalogLibraryManagementError("PACKAGE_METADATA_MISSING")

    def _package_library_root(self, root: Path) -> Path:
        candidates = [root / "library", root]
        for candidate in candidates:
            if (candidate / "catalog.json").is_file():
                return candidate
        raise CatalogLibraryManagementError("PACKAGE_LIBRARY_MISSING")

    def _check_package_metadata(self, metadata: dict[str, Any]) -> None:
        required = (
            "library_id",
            "version",
            "format_version",
            "astap_families",
            "near_coverage",
            "blind4d_coverage",
            "all_sky_blind4d",
            "installed_size_bytes",
            "sha256",
            "provenance",
            "astap_credit",
            "astrometry_credit",
            "license",
            "generated_at",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise CatalogLibraryManagementError(f"PACKAGE_METADATA_INCOMPLETE: {', '.join(missing)}")
        if int(metadata.get("format_version") or 0) < 1:
            raise CatalogLibraryManagementError("PACKAGE_FORMAT_INCOMPATIBLE")

    def _verify_package_hashes(self, library_root: Path, metadata: dict[str, Any]) -> None:
        hashes = metadata.get("sha256")
        if not isinstance(hashes, dict):
            raise CatalogLibraryManagementError("PACKAGE_HASHES_INVALID")
        for rel, expected in sorted(hashes.items()):
            rel_path = Path(str(rel))
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise CatalogLibraryManagementError(f"PACKAGE_HASH_PATH_INVALID: {rel}")
            path = (library_root / rel_path).resolve()
            if not _is_relative_to(path, library_root.resolve()):
                raise CatalogLibraryManagementError(f"PACKAGE_HASH_PATH_INVALID: {rel}")
            if not path.is_file():
                raise CatalogLibraryManagementError(f"PACKAGE_HASH_FILE_MISSING: {rel}")
            actual = sha256_file(path)
            if actual.lower() != str(expected).lower():
                raise CatalogLibraryManagementError(f"PACKAGE_SHA256_MISMATCH: {rel}")

    def _extract_zip(self, archive: Path, target: Path) -> None:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                self._validate_archive_member(info.filename, is_dir=info.is_dir(), is_symlink=_zip_is_symlink(info))
            zf.extractall(target)

    def _extract_tar(self, archive: Path, target: Path) -> None:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                self._validate_archive_member(member.name, is_dir=member.isdir(), is_symlink=member.issym() or member.islnk())
            tf.extractall(target)

    def _validate_archive_member(self, name: str, *, is_dir: bool, is_symlink: bool) -> None:
        path = Path(name)
        if not name or path.is_absolute() or ".." in path.parts:
            raise CatalogLibraryManagementError(f"PACKAGE_ARCHIVE_UNSAFE_PATH: {name}")
        if is_symlink:
            raise CatalogLibraryManagementError(f"PACKAGE_ARCHIVE_SYMLINK_FORBIDDEN: {name}")
        if not is_dir and any(part == "" for part in path.parts):
            raise CatalogLibraryManagementError(f"PACKAGE_ARCHIVE_MALFORMED_PATH: {name}")

    def _validate_create_paths(self, source: Path, destination: Path) -> None:
        if source == destination:
            raise CatalogLibraryManagementError("DESTINATION_OVERLAPS_SOURCE")
        if _is_relative_to(destination, source):
            raise CatalogLibraryManagementError("DESTINATION_INSIDE_SOURCE")
        if _is_relative_to(source, destination):
            raise CatalogLibraryManagementError("SOURCE_INSIDE_DESTINATION")
        parent = destination.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if not os.access(parent, os.W_OK):
            raise CatalogLibraryManagementError(f"DESTINATION_PARENT_NOT_WRITABLE: {parent}")

    def _ensure_destination_available(self, destination: Path) -> None:
        if destination.exists():
            if destination.is_dir() and not any(destination.iterdir()):
                destination.rmdir()
            else:
                raise CatalogLibraryManagementError(f"DESTINATION_EXISTS: {destination}")

    def _check_free_space(self, destination: Path, required_bytes: int) -> None:
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        free = int(self._disk_usage(parent).free)
        if required_bytes > free:
            raise CatalogLibraryManagementError("DISK_SPACE_INSUFFICIENT")

    def _publish_staging(self, staging_root: Path, destination: Path) -> None:
        self._ensure_destination_available(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_root), str(destination))

    def _staging_path(self, destination: Path) -> Path:
        return destination.parent / f"{destination.name}.partial-{uuid.uuid4().hex[:10]}"

    def _final_astap_roots(self, destination: Path, source_root: Path, astap_roots: tuple[Path, ...], storage_policy: str) -> tuple[Path, ...]:
        if str(storage_policy or "reference").strip().lower() == "copy":
            return (destination / "sources" / "astap",)
        return astap_roots or (source_root,)

    def _emit(
        self,
        stage: str,
        message: str,
        *,
        family: str | None = None,
        overall_current: int = 0,
        overall_total: int = 0,
        step_current: int = 0,
        step_total: int = 0,
    ) -> None:
        if self._progress_callback is not None:
            self._progress_callback(
                LibraryManagementProgress(
                    stage=stage,
                    message=message,
                    family=family,
                    overall_current=overall_current,
                    overall_total=overall_total,
                    step_current=step_current,
                    step_total=step_total,
                )
            )

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_callback and self._cancel_callback())

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise CatalogLibraryManagementCancelled("CONSTRUCTION_CANCELLED")


def _library_id_from_root(root: Path) -> str:
    name = root.name.strip().lower().replace(" ", "-") or "zesolver-library"
    return "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_"}) or "zesolver-library"


def _normalize_requested_families(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        family = str(value).strip().lower()
        if not family or family in seen:
            continue
        seen.add(family)
        result.append(family)
    return tuple(result)


def _find_family_files(root: Path, spec: CatalogFamilySpec) -> tuple[Path, ...]:
    expected_prefix = f"{spec.prefix}_".lower()
    expected_suffix = f".{spec.extension}".lower()
    matches = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name.lower().startswith(expected_prefix)
        and path.name.lower().endswith(expected_suffix)
    ]
    return tuple(sorted(matches, key=lambda item: item.name.lower()))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == 0o120000


__all__ = [
    "AstapFamilyInfo",
    "CatalogLibraryManagementCancelled",
    "CatalogLibraryManagementError",
    "CatalogLibraryManagementService",
    "LibraryAnalysisResult",
    "LibraryCreateOptions",
    "LibraryDiagnosticItem",
    "LibraryInstallOptions",
    "LibraryManagementProgress",
    "LibraryOperationResult",
    "LibraryRepairPlan",
]
