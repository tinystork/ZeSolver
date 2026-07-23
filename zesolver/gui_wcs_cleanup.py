from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PySide6 import QtCore


ProcessFitsCallable = Callable[..., tuple[int, int]]


@dataclass(frozen=True, slots=True)
class WcsCleanupConfig:
    files: tuple[Path, ...]
    dry_run: bool = False
    backup: bool = False
    only_if_wcs: bool = True
    all_hdus: bool = False

    @classmethod
    def from_files(
        cls,
        files: Sequence[Path],
        *,
        dry_run: bool = False,
        backup: bool = False,
        only_if_wcs: bool = True,
        all_hdus: bool = False,
    ) -> "WcsCleanupConfig":
        return cls(
            files=tuple(Path(path) for path in files),
            dry_run=bool(dry_run),
            backup=bool(backup),
            only_if_wcs=bool(only_if_wcs),
            all_hdus=bool(all_hdus),
        )


@dataclass(frozen=True, slots=True)
class WcsCleanupFileResult:
    path: Path
    deleted_cards: int
    edited_hdus: int
    changed: bool
    status: str


@dataclass(frozen=True, slots=True)
class WcsCleanupProgress:
    completed: int
    total: int
    remaining: int
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class WcsCleanupError:
    path: Path | None
    operation: str
    message: str
    final_status: str


@dataclass(frozen=True, slots=True)
class WcsCleanupSummary:
    planned: int
    processed: int
    changed_files: int
    deleted_cards: int
    errors: int
    remaining: int
    duration_s: float
    terminal_status: str


class WcsCleanupRunner(QtCore.QThread):
    cleanup_started = QtCore.Signal(object)
    file_result = QtCore.Signal(object)
    progress = QtCore.Signal(object)
    warning = QtCore.Signal(object)
    file_error = QtCore.Signal(object)
    fatal_error = QtCore.Signal(object)
    completed = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)

    def __init__(
        self,
        config: WcsCleanupConfig,
        *,
        process_fits_func: ProcessFitsCallable | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._cancel_event = threading.Event()
        self._process_fits_func = process_fits_func
        self._terminal_emitted = False

    def request_cancel(self) -> None:
        logging.info("WCS cleanup cancellation requested")
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _emit_terminal(self, signal, summary: WcsCleanupSummary) -> None:
        if self._terminal_emitted:
            logging.warning("Duplicate WCS cleanup terminal ignored: %s", summary.terminal_status)
            return
        self._terminal_emitted = True
        signal.emit(summary)

    def _process_fits(self) -> ProcessFitsCallable:
        if self._process_fits_func is not None:
            return self._process_fits_func
        from zewcscleaner import process_fits

        return process_fits

    def run(self) -> None:  # pragma: no cover - exercised through Qt tests
        started_at = time.perf_counter()
        total = len(self.config.files)
        processed = 0
        changed_files = 0
        deleted_cards = 0
        errors = 0
        logging.info("WCS cleanup started planned=%s", total)
        self.cleanup_started.emit(WcsCleanupProgress(completed=0, total=total, remaining=total))
        try:
            process_fits = self._process_fits()
            for path in self.config.files:
                if self._cancel_event.is_set():
                    remaining = max(0, total - processed)
                    summary = WcsCleanupSummary(
                        planned=total,
                        processed=processed,
                        changed_files=changed_files,
                        deleted_cards=deleted_cards,
                        errors=errors,
                        remaining=remaining,
                        duration_s=time.perf_counter() - started_at,
                        terminal_status="cancelled",
                    )
                    logging.info(
                        "WCS cleanup cancelled planned=%s processed=%s remaining=%s",
                        total,
                        processed,
                        remaining,
                    )
                    self._emit_terminal(self.cancelled, summary)
                    return
                try:
                    deleted, edited_hdus = process_fits(
                        str(path),
                        dry_run=self.config.dry_run,
                        backup=self.config.backup,
                        only_if_wcs=self.config.only_if_wcs,
                        all_hdus=self.config.all_hdus,
                    )
                except Exception as exc:
                    errors += 1
                    err = WcsCleanupError(
                        path=path,
                        operation="process_fits",
                        message=str(exc),
                        final_status="failed",
                    )
                    logging.warning("WCS cleanup file failed path=%s error=%s", path, exc)
                    self.file_error.emit(err)
                    summary = WcsCleanupSummary(
                        planned=total,
                        processed=processed,
                        changed_files=changed_files,
                        deleted_cards=deleted_cards,
                        errors=errors,
                        remaining=max(0, total - processed),
                        duration_s=time.perf_counter() - started_at,
                        terminal_status="failed",
                    )
                    self._emit_terminal(self.fatal_error, err)
                    logging.info("WCS cleanup failed planned=%s processed=%s errors=%s", total, processed, errors)
                    return
                processed += 1
                deleted_i = int(deleted)
                edited_i = int(edited_hdus)
                deleted_cards += deleted_i
                changed = edited_i > 0
                if changed:
                    changed_files += 1
                result = WcsCleanupFileResult(
                    path=path,
                    deleted_cards=deleted_i,
                    edited_hdus=edited_i,
                    changed=changed,
                    status="changed" if changed else "unchanged",
                )
                self.file_result.emit(result)
                progress = WcsCleanupProgress(
                    completed=processed,
                    total=total,
                    remaining=max(0, total - processed),
                    path=path,
                )
                logging.info(
                    "WCS cleanup progress completed=%s total=%s remaining=%s path=%s",
                    progress.completed,
                    progress.total,
                    progress.remaining,
                    path,
                )
                self.progress.emit(progress)
            summary = WcsCleanupSummary(
                planned=total,
                processed=processed,
                changed_files=changed_files,
                deleted_cards=deleted_cards,
                errors=errors,
                remaining=0,
                duration_s=time.perf_counter() - started_at,
                terminal_status="completed",
            )
            logging.info(
                "WCS cleanup completed planned=%s processed=%s changed=%s cards=%s duration=%.3fs",
                total,
                processed,
                changed_files,
                deleted_cards,
                summary.duration_s,
            )
            self._emit_terminal(self.completed, summary)
        except Exception as exc:
            errors += 1
            err = WcsCleanupError(
                path=None,
                operation="wcs_cleanup_worker",
                message=str(exc),
                final_status="failed",
            )
            logging.exception("WCS cleanup fatal error")
            self._emit_terminal(self.fatal_error, err)
