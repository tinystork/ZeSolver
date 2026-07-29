from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Protocol

from .core.terminal_reasons import is_unresolved_move_eligible, normalize_terminal_reason_code
from .output_contract import UNRESOLVED_DIRECTORY_NAME, is_inside_unresolved_directory


class UnresolvedResult(Protocol):
    input_path: Path
    output_path: Path | None
    status: object
    backend: str | None
    error: str | None
    terminal_reason_code: str | None


@dataclass(frozen=True, slots=True)
class UnresolvedMoveRecord:
    original_relative_path: str
    destination_relative_path: str | None
    reason_code: str | None
    near_result: str | None
    blind_result: str | None
    web_fallback_result: str | None
    move_status: str
    move_error: str | None


@dataclass(frozen=True, slots=True)
class UnresolvedMoveSummary:
    eligible: int
    moved: int
    move_failed: int
    directory: Path | None
    manifest_path: Path | None
    records: tuple[UnresolvedMoveRecord, ...]

    def telemetry(self, *, requested: bool) -> dict[str, object]:
        return {
            "move_unresolved_requested": bool(requested),
            "unresolved_eligible": self.eligible,
            "unresolved_moved": self.moved,
            "unresolved_move_failed": self.move_failed,
            "unresolved_directory": str(self.directory) if self.directory else None,
            "unresolved_manifest": str(self.manifest_path) if self.manifest_path else None,
        }


def collect_unresolved_eligible(results: Iterable[UnresolvedResult]) -> tuple[UnresolvedResult, ...]:
    out: list[UnresolvedResult] = []
    for result in results:
        reason = normalize_terminal_reason_code(getattr(result, "terminal_reason_code", None))
        status_obj = getattr(result, "status", "")
        status = str(getattr(status_obj, "value", status_obj)).upper()
        if status != "UNSOLVED" and not (status == "FAILED" and is_unresolved_move_eligible(reason)):
            continue
        if bool(getattr(result, "wcs_written", False)):
            continue
        if is_unresolved_move_eligible(reason):
            out.append(result)
    return tuple(out)


def move_unresolved_results(
    *,
    input_root: Path,
    results: Iterable[UnresolvedResult],
    terminal_status: str,
    requested: bool,
    run_id: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    log_warning=None,
) -> UnresolvedMoveSummary:
    eligible = collect_unresolved_eligible(results)
    root = Path(input_root).expanduser()
    target_root = root / UNRESOLVED_DIRECTORY_NAME
    if not requested or str(terminal_status) != "completed":
        return UnresolvedMoveSummary(
            eligible=len(eligible),
            moved=0,
            move_failed=0,
            directory=(target_root if requested and eligible else None),
            manifest_path=None,
            records=(),
        )

    records: list[UnresolvedMoveRecord] = []
    moved = 0
    move_failed = 0
    target_root.mkdir(parents=True, exist_ok=True)
    for result in eligible:
        source = Path(result.output_path or result.input_path)
        original_rel = _relative_to(source, root)
        if is_inside_unresolved_directory(source):
            records.append(
                UnresolvedMoveRecord(
                    original_relative_path=original_rel,
                    destination_relative_path=None,
                    reason_code=normalize_terminal_reason_code(getattr(result, "terminal_reason_code", None)),
                    near_result=_near_result(result),
                    blind_result=_blind_result(result),
                    web_fallback_result=None,
                    move_status="skipped_already_unresolved",
                    move_error=None,
                )
            )
            continue
        destination = _collision_free_path(target_root / original_rel)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _move_file(source, destination)
            for sidecar in _known_sidecars(source):
                if sidecar.exists():
                    sidecar_dest = destination.with_name(destination.name + sidecar.name[len(source.name) :])
                    _move_file(sidecar, _collision_free_path(sidecar_dest))
            moved += 1
            records.append(
                UnresolvedMoveRecord(
                    original_relative_path=original_rel,
                    destination_relative_path=_relative_to(destination, root),
                    reason_code=normalize_terminal_reason_code(getattr(result, "terminal_reason_code", None)),
                    near_result=_near_result(result),
                    blind_result=_blind_result(result),
                    web_fallback_result=None,
                    move_status="moved",
                    move_error=None,
                )
            )
        except Exception as exc:
            move_failed += 1
            records.append(
                UnresolvedMoveRecord(
                    original_relative_path=original_rel,
                    destination_relative_path=_relative_to(destination, root),
                    reason_code=normalize_terminal_reason_code(getattr(result, "terminal_reason_code", None)),
                    near_result=_near_result(result),
                    blind_result=_blind_result(result),
                    web_fallback_result=None,
                    move_status="failed",
                    move_error=str(exc),
                )
            )

    manifest_path = None
    if records:
        manifest_path = target_root / f"unresolved_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            _write_manifest(
                manifest_path,
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                input_root=root,
                records=records,
                eligible=len(eligible),
                moved=moved,
                move_failed=move_failed,
            )
        except Exception as exc:
            if callable(log_warning):
                log_warning(f"Unresolved manifest write skipped: {exc}")
            manifest_path = None
    return UnresolvedMoveSummary(
        eligible=len(eligible),
        moved=moved,
        move_failed=move_failed,
        directory=target_root,
        manifest_path=manifest_path,
        records=tuple(records),
    )


def _known_sidecars(source: Path) -> tuple[Path, ...]:
    return (Path(str(source) + ".wcs.json"), Path(str(source) + ".meta.json"))


def _move_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(str(source))
    if destination.exists():
        raise FileExistsError(str(destination))
    try:
        source.replace(destination)
    except OSError:
        shutil.move(str(source), str(destination))


def _collision_free_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10000):
        candidate = path.with_name(f"{stem}__{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"too many collisions for {path}")


def _relative_to(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return Path(path).name


def _near_result(result: UnresolvedResult) -> str | None:
    reason = normalize_terminal_reason_code(getattr(result, "terminal_reason_code", None))
    if reason in {"NEAR_UNRESOLVED_BLIND_UNAVAILABLE", "ALL_ENABLED_SOLVERS_EXHAUSTED"}:
        return "failed"
    return None


def _blind_result(result: UnresolvedResult) -> str | None:
    reason = normalize_terminal_reason_code(getattr(result, "terminal_reason_code", None))
    if reason == "NEAR_UNRESOLVED_BLIND_UNAVAILABLE":
        return "unavailable"
    if reason == "ALL_ENABLED_SOLVERS_EXHAUSTED":
        return "failed"
    return None


def _write_manifest(
    path: Path,
    *,
    run_id: int | None,
    started_at: str | None,
    finished_at: str | None,
    input_root: Path,
    records: list[UnresolvedMoveRecord],
    eligible: int,
    moved: int,
    move_failed: int,
) -> None:
    payload: dict[str, object] = {
        "schema": "zesolver.unresolved.v1",
        "run": {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "input_root": ".",
        },
        "summary": {
            "eligible": eligible,
            "moved": moved,
            "move_failed": move_failed,
        },
        "files": [asdict(record) for record in records],
    }
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
