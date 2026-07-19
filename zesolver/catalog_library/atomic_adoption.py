"""Atomic writer for explicit CatalogLibrary adoption plans."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .manifest import CatalogLibrary, CatalogManifestError, manifest_from_payload
from .models import (
    CatalogAdoptionCommitStatus,
    CatalogAdoptionWriteMode,
    CatalogIssue,
    CatalogLibraryAdoptionPlanResult,
    CatalogLibraryAdoptionResult,
    CatalogStatus,
    IssueSeverity,
)

CATALOG_ADOPTION_PLAN_INVALID = "CATALOG_ADOPTION_PLAN_INVALID"
CATALOG_ADOPTION_PLAN_HAS_ERRORS = "CATALOG_ADOPTION_PLAN_HAS_ERRORS"
CATALOG_ADOPTION_TARGET_EXISTS = "CATALOG_ADOPTION_TARGET_EXISTS"
CATALOG_ADOPTION_TARGET_MISSING = "CATALOG_ADOPTION_TARGET_MISSING"
CATALOG_ADOPTION_CONFLICT = "CATALOG_ADOPTION_CONFLICT"
CATALOG_ADOPTION_LOCKED = "CATALOG_ADOPTION_LOCKED"
CATALOG_ADOPTION_TARGET_SYMLINK = "CATALOG_ADOPTION_TARGET_SYMLINK"
CATALOG_ADOPTION_TEMP_WRITE_FAILED = "CATALOG_ADOPTION_TEMP_WRITE_FAILED"
CATALOG_ADOPTION_VALIDATION_FAILED = "CATALOG_ADOPTION_VALIDATION_FAILED"
CATALOG_ADOPTION_REPLACE_FAILED = "CATALOG_ADOPTION_REPLACE_FAILED"
CATALOG_ADOPTION_ROLLBACK_FAILED = "CATALOG_ADOPTION_ROLLBACK_FAILED"
CATALOG_ADOPTION_READ_ONLY = "CATALOG_ADOPTION_READ_ONLY"

_LOCK_NAME = ".catalog-adoption.lock"


class CatalogLibraryAdoptionError(RuntimeError):
    """Stable adoption writer failure."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        result: CatalogLibraryAdoptionResult | None = None,
    ) -> None:
        self.code = code
        self.result = result
        super().__init__(message or code)


class CatalogLibraryAdoptionWriter:
    """Commit a prebuilt adoption plan to `<library_root>/catalog.json`."""

    @classmethod
    def commit(
        cls,
        plan: CatalogLibraryAdoptionPlanResult,
        *,
        mode: str | CatalogAdoptionWriteMode = CatalogAdoptionWriteMode.CREATE_NEW,
        expected_existing_sha256: str | None = None,
        create_backup: bool = True,
    ) -> CatalogLibraryAdoptionResult:
        write_mode = _coerce_mode(mode)
        library_root = _plan_library_root(plan)
        catalog_path = library_root / "catalog.json"
        serialized = _serialize_manifest(plan.manifest_preview)
        manifest_sha = _sha256_bytes(serialized)
        _preflight_plan(plan, library_root=library_root, catalog_path=catalog_path)
        if not library_root.exists() or not library_root.is_dir():
            raise _error(CATALOG_ADOPTION_READ_ONLY, write_mode, library_root, catalog_path, "library root must already exist")
        if catalog_path.is_symlink():
            raise _error(CATALOG_ADOPTION_TARGET_SYMLINK, write_mode, library_root, catalog_path)
        previous_sha = _sha256_file(catalog_path) if catalog_path.exists() and catalog_path.is_file() else None
        if previous_sha == manifest_sha:
            return CatalogLibraryAdoptionResult(
                status=CatalogAdoptionCommitStatus.NO_CHANGE,
                mode=write_mode,
                library_root=library_root,
                catalog_path=catalog_path,
                unchanged=True,
                manifest_sha256=manifest_sha,
                previous_sha256=previous_sha,
                lock_used=False,
                atomic_replace_used=False,
                post_write_validation=True,
                files_written=0,
                warnings=tuple(plan.warnings),
                telemetry=_telemetry(plan, target_exists=True, fast_limit=_fast_limit(plan)),
            )
        _check_mode(write_mode, catalog_path, expected_existing_sha256, previous_sha, library_root)

        lock = _AdoptionLock(library_root)
        temp_path: Path | None = None
        backup_path: Path | None = None
        lock.acquire()
        try:
            latest_sha = _sha256_file(catalog_path) if catalog_path.exists() and catalog_path.is_file() else None
            if latest_sha == manifest_sha:
                return CatalogLibraryAdoptionResult(
                    status=CatalogAdoptionCommitStatus.NO_CHANGE,
                    mode=write_mode,
                    library_root=library_root,
                    catalog_path=catalog_path,
                    unchanged=True,
                    manifest_sha256=manifest_sha,
                    previous_sha256=latest_sha,
                    lock_used=True,
                    post_write_validation=True,
                    files_written=0,
                    warnings=tuple(plan.warnings),
                    telemetry=_telemetry(plan, target_exists=True, fast_limit=_fast_limit(plan)),
                )
            _check_mode(write_mode, catalog_path, expected_existing_sha256, latest_sha, library_root)
            _check_resource_drift(plan)
            temp_path = _write_temp_manifest(library_root, serialized)
            _validate_payload_at_path(plan.manifest_preview, library_root=library_root, manifest_path=temp_path, full=True)
            if write_mode is CatalogAdoptionWriteMode.REPLACE_EXISTING:
                if not create_backup:
                    raise _error(CATALOG_ADOPTION_PLAN_INVALID, write_mode, library_root, catalog_path, "replacement requires backup")
                backup_path = _create_backup(catalog_path, latest_sha or "")
            try:
                _atomic_replace(temp_path, catalog_path)
                temp_path = None
                _fsync_dir(library_root)
            except Exception as exc:
                if backup_path is not None and backup_path.exists():
                    _attempt_rollback(backup_path, catalog_path, write_mode, library_root)
                raise _error(CATALOG_ADOPTION_REPLACE_FAILED, write_mode, library_root, catalog_path, str(exc)) from exc
            post_ok = False
            rollback_performed = False
            try:
                report = CatalogLibrary.open(library_root).validate()
                post_ok = report.status not in {CatalogStatus.CORRUPT, CatalogStatus.INCOMPATIBLE, CatalogStatus.MISSING}
                if not post_ok:
                    raise CatalogManifestError(f"post_write_status_invalid: {report.status.value}")
            except Exception as exc:
                if backup_path is not None and backup_path.exists():
                    try:
                        _attempt_rollback(backup_path, catalog_path, write_mode, library_root)
                        rollback_performed = True
                    except Exception as rollback_exc:
                        result = _failure_result(
                            write_mode,
                            library_root,
                            catalog_path,
                            backup_path=backup_path,
                            manifest_sha=manifest_sha,
                            previous_sha=latest_sha,
                            lock_used=True,
                            rollback_performed=True,
                            code=CATALOG_ADOPTION_ROLLBACK_FAILED,
                        )
                        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_ROLLBACK_FAILED, str(rollback_exc), result=result) from rollback_exc
                    result = _failure_result(
                        write_mode,
                        library_root,
                        catalog_path,
                        backup_path=backup_path,
                        manifest_sha=manifest_sha,
                        previous_sha=latest_sha,
                        lock_used=True,
                        rollback_performed=True,
                        code=CATALOG_ADOPTION_VALIDATION_FAILED,
                    )
                    raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_VALIDATION_FAILED, str(exc), result=result) from exc
                try:
                    if catalog_path.exists() and not catalog_path.is_symlink():
                        catalog_path.unlink()
                finally:
                    result = _failure_result(
                        write_mode,
                        library_root,
                        catalog_path,
                        manifest_sha=manifest_sha,
                        previous_sha=latest_sha,
                        lock_used=True,
                        rollback_performed=rollback_performed,
                        code=CATALOG_ADOPTION_VALIDATION_FAILED,
                    )
                raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_VALIDATION_FAILED, str(exc), result=result) from exc
            return CatalogLibraryAdoptionResult(
                status=CatalogAdoptionCommitStatus.CREATED if write_mode is CatalogAdoptionWriteMode.CREATE_NEW else CatalogAdoptionCommitStatus.REPLACED,
                mode=write_mode,
                library_root=library_root,
                catalog_path=catalog_path,
                created=write_mode is CatalogAdoptionWriteMode.CREATE_NEW,
                replaced=write_mode is CatalogAdoptionWriteMode.REPLACE_EXISTING,
                backup_path=backup_path,
                manifest_sha256=manifest_sha,
                previous_sha256=latest_sha,
                lock_used=True,
                atomic_replace_used=True,
                post_write_validation=post_ok,
                rollback_performed=False,
                files_written=2 if backup_path else 1,
                warnings=tuple(plan.warnings),
                telemetry=_telemetry(plan, target_exists=latest_sha is not None, fast_limit=_fast_limit(plan)),
            )
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
            lock.release()


def _coerce_mode(mode: str | CatalogAdoptionWriteMode) -> CatalogAdoptionWriteMode:
    if isinstance(mode, CatalogAdoptionWriteMode):
        return mode
    raw = str(mode).strip().lower().replace("-", "_")
    if raw in {"create", "create_new", "new"}:
        return CatalogAdoptionWriteMode.CREATE_NEW
    if raw in {"replace", "replace_existing"}:
        return CatalogAdoptionWriteMode.REPLACE_EXISTING
    raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_PLAN_INVALID, f"invalid adoption mode: {mode}")


def _plan_library_root(plan: CatalogLibraryAdoptionPlanResult) -> Path:
    if getattr(plan, "library_root", None) is not None:
        return Path(plan.library_root).expanduser().resolve()
    raw = plan.manifest_preview.get("provenance", {}).get("library_root")
    if not raw:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_PLAN_INVALID, "plan has no library_root")
    return Path(str(raw)).expanduser().resolve()


def _preflight_plan(plan: CatalogLibraryAdoptionPlanResult, *, library_root: Path, catalog_path: Path) -> None:
    if plan.errors:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_PLAN_HAS_ERRORS, ", ".join(issue.code for issue in plan.errors))
    if plan.status in {CatalogStatus.CORRUPT, CatalogStatus.INCOMPATIBLE, CatalogStatus.MISSING}:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_PLAN_INVALID, f"plan status cannot be adopted: {plan.status.value}")
    if catalog_path != library_root / "catalog.json":
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_PLAN_INVALID, "target must be library_root/catalog.json")
    preview_root = Path(str(plan.manifest_preview.get("provenance", {}).get("library_root") or "")).expanduser().resolve()
    if preview_root != library_root:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_PLAN_INVALID, "manifest preview library_root does not match plan")
    _validate_payload_at_path(plan.manifest_preview, library_root=library_root, manifest_path=catalog_path)


def _check_mode(
    mode: CatalogAdoptionWriteMode,
    catalog_path: Path,
    expected_existing_sha256: str | None,
    previous_sha: str | None,
    library_root: Path,
) -> None:
    if mode is CatalogAdoptionWriteMode.CREATE_NEW:
        if previous_sha is not None or catalog_path.exists():
            raise _error(CATALOG_ADOPTION_TARGET_EXISTS, mode, library_root, catalog_path)
        return
    if previous_sha is None:
        raise _error(CATALOG_ADOPTION_TARGET_MISSING, mode, library_root, catalog_path)
    if not expected_existing_sha256:
        raise _error(CATALOG_ADOPTION_CONFLICT, mode, library_root, catalog_path, "expected_existing_sha256 is required")
    if previous_sha.lower() != expected_existing_sha256.lower():
        raise _error(CATALOG_ADOPTION_CONFLICT, mode, library_root, catalog_path, "existing manifest changed")


def _check_resource_drift(plan: CatalogLibraryAdoptionPlanResult) -> None:
    for source in plan.sources:
        if not source.path.resolved.exists() or not source.path.resolved.is_dir():
            raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"source disappeared: {source.id}")
        for shard in source.shards:
            if shard.resolved_path is None or not shard.resolved_path.exists():
                raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"source shard disappeared: {shard.path}")
            if shard.sha256 and _sha256_file(shard.resolved_path).lower() != shard.sha256.lower():
                raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"source shard changed: {shard.path}")
    for index in plan.indexes:
        if not index.path.resolved.exists():
            raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"index disappeared: {index.id}")
        if index.manifest_path is not None:
            if not index.manifest_path.resolved.exists():
                raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"strict manifest disappeared: {index.id}")
            expected_manifest_sha = str(index.compatibility.get("strict_manifest_sha256") or "")
            if expected_manifest_sha and _sha256_file(index.manifest_path.resolved).lower() != expected_manifest_sha.lower():
                raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"strict manifest changed: {index.id}")
        for file in (*index.integrity_files, *index.derived_files):
            if file.resolved_path is None or not file.resolved_path.exists():
                raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"index file disappeared: {index.id}")
            if file.sha256 and _sha256_file(file.resolved_path).lower() != file.sha256.lower():
                raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"index file changed: {index.id}")


def _serialize_manifest(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _write_temp_manifest(library_root: Path, data: bytes) -> Path:
    temp_path = library_root / f".catalog.json.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(temp_path, flags, 0o644)
        try:
            with os.fdopen(fd, "wb", closefd=True) as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    except PermissionError as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_READ_ONLY, str(exc)) from exc
    except Exception as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_TEMP_WRITE_FAILED, str(exc)) from exc
    return temp_path


def _validate_payload_at_path(payload: dict[str, Any], *, library_root: Path, manifest_path: Path, full: bool = False) -> None:
    try:
        library = CatalogLibrary(manifest_from_payload(payload, root=library_root, manifest_path=manifest_path))
    except Exception as exc:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_VALIDATION_FAILED, str(exc)) from exc
    if not full:
        return
    try:
        report = library.validate()
    except Exception as exc:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_VALIDATION_FAILED, str(exc)) from exc
    if report.status in {CatalogStatus.CORRUPT, CatalogStatus.INCOMPATIBLE, CatalogStatus.MISSING}:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_VALIDATION_FAILED, report.status.value)


def _create_backup(catalog_path: Path, previous_sha: str) -> Path:
    backup = catalog_path.with_name(f"catalog.json.backup.{previous_sha}.json")
    if backup.exists():
        if _sha256_file(backup) != previous_sha:
            raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_CONFLICT, f"backup path exists with different content: {backup.name}")
        return backup
    try:
        shutil.copy2(catalog_path, backup)
        with backup.open("rb") as fh:
            os.fsync(fh.fileno())
    except PermissionError as exc:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_READ_ONLY, str(exc)) from exc
    except Exception as exc:
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_REPLACE_FAILED, f"backup failed: {exc}") from exc
    return backup


def _atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def _attempt_rollback(backup_path: Path, catalog_path: Path, mode: CatalogAdoptionWriteMode, library_root: Path) -> None:
    rollback_temp = library_root / f".catalog.rollback.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(backup_path, rollback_temp)
        _atomic_replace(rollback_temp, catalog_path)
        _fsync_dir(library_root)
        CatalogLibrary.open(library_root).validate()
    except Exception as exc:
        try:
            rollback_temp.unlink()
        except FileNotFoundError:
            pass
        raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_ROLLBACK_FAILED, str(exc)) from exc


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _error(
    code: str,
    mode: CatalogAdoptionWriteMode,
    library_root: Path,
    catalog_path: Path,
    message: str | None = None,
) -> CatalogLibraryAdoptionError:
    return CatalogLibraryAdoptionError(
        code,
        message or code,
        result=_failure_result(
            mode,
            library_root,
            catalog_path,
            code=code,
        ),
    )


def _failure_result(
    mode: CatalogAdoptionWriteMode,
    library_root: Path,
    catalog_path: Path,
    *,
    code: str,
    backup_path: Path | None = None,
    manifest_sha: str | None = None,
    previous_sha: str | None = None,
    lock_used: bool = False,
    rollback_performed: bool = False,
) -> CatalogLibraryAdoptionResult:
    return CatalogLibraryAdoptionResult(
        status=CatalogAdoptionCommitStatus.ROLLED_BACK if rollback_performed else CatalogAdoptionCommitStatus.FAILED,
        mode=mode,
        library_root=library_root,
        catalog_path=catalog_path,
        backup_path=backup_path,
        manifest_sha256=manifest_sha,
        previous_sha256=previous_sha,
        lock_used=lock_used,
        rollback_performed=rollback_performed,
        errors=(
            CatalogIssue(
                code=code,
                severity=IssueSeverity.ERROR,
                message=code,
                path=catalog_path,
                component_id="catalog_adoption",
            ),
        ),
    )


def _telemetry(plan: CatalogLibraryAdoptionPlanResult, *, target_exists: bool, fast_limit: bool) -> dict[str, Any]:
    return {
        "adoption_writer": "atomic-v1",
        "target_exists": bool(target_exists),
        "fingerprint_policy": plan.telemetry.fingerprint_policy.value,
        "source_file_count": plan.telemetry.source_file_count,
        "source_hashed_count": plan.telemetry.source_hashed_count,
        "index_file_count": plan.telemetry.index_file_count,
        "index_hashed_count": plan.telemetry.index_hashed_count,
        "fast_source_mutation_limit": bool(fast_limit),
        "builder_called": False,
    }


def _fast_limit(plan: CatalogLibraryAdoptionPlanResult) -> bool:
    return plan.telemetry.source_file_count > plan.telemetry.source_hashed_count


class _AdoptionLock:
    def __init__(self, library_root: Path) -> None:
        self.path = library_root / _LOCK_NAME
        self._fd: int | None = None

    def acquire(self) -> None:
        payload = json.dumps({"pid": os.getpid(), "purpose": "catalog_adoption"}, sort_keys=True).encode("utf-8")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_LOCKED, str(self.path)) from exc
        except PermissionError as exc:
            raise CatalogLibraryAdoptionError(CATALOG_ADOPTION_READ_ONLY, str(exc)) from exc
        os.write(fd, payload)
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "CATALOG_ADOPTION_CONFLICT",
    "CATALOG_ADOPTION_LOCKED",
    "CATALOG_ADOPTION_PLAN_HAS_ERRORS",
    "CATALOG_ADOPTION_PLAN_INVALID",
    "CATALOG_ADOPTION_READ_ONLY",
    "CATALOG_ADOPTION_REPLACE_FAILED",
    "CATALOG_ADOPTION_ROLLBACK_FAILED",
    "CATALOG_ADOPTION_TARGET_EXISTS",
    "CATALOG_ADOPTION_TARGET_MISSING",
    "CATALOG_ADOPTION_TARGET_SYMLINK",
    "CATALOG_ADOPTION_TEMP_WRITE_FAILED",
    "CATALOG_ADOPTION_VALIDATION_FAILED",
    "CatalogLibraryAdoptionError",
    "CatalogLibraryAdoptionWriter",
]
