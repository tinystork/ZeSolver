from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zesolver.core.batch import BatchProgress
from zesolver.core.models import SolveResult, SolveStatus


@dataclass(frozen=True, slots=True)
class GuiProgress:
    total: int
    completed: int
    solved: int
    failed: int
    skipped: int
    cancelled: int
    current_path: Path | None = None
    current_phase: str | None = None


def gui_progress_from_batch(result: SolveResult | None, progress: BatchProgress, *, phase: str | None = None) -> GuiProgress:
    completed = progress.solved + progress.failed + progress.skipped + progress.cancelled
    current = result.input_path if result is not None else None
    return GuiProgress(
        total=max(0, int(progress.total)),
        completed=max(0, min(int(progress.total), completed)),
        solved=max(0, int(progress.solved)),
        failed=max(0, int(progress.failed)),
        skipped=max(0, int(progress.skipped)),
        cancelled=max(0, int(progress.cancelled)),
        current_path=current,
        current_phase=phase,
    )


def gui_progress_from_results(total: int, results: tuple[object, ...], *, current_path: Path | None = None, phase: str | None = None) -> GuiProgress:
    solved = 0
    failed = 0
    skipped = 0
    cancelled = 0
    for item in results:
        status = getattr(item, "status", None)
        if status is SolveStatus.SOLVED or str(status).upper() == "SOLVED" or str(status).lower() in {"solved", "wcs"}:
            solved += 1
        elif status is SolveStatus.CANCELLED or str(status).upper() == "CANCELLED":
            cancelled += 1
        elif str(status).upper() in {"SKIPPED", "INVALID_INPUT"} or str(status).lower() == "skipped":
            skipped += 1
        else:
            failed += 1
    completed = solved + failed + skipped + cancelled
    return GuiProgress(
        total=max(0, int(total)),
        completed=max(0, min(int(total), completed)),
        solved=solved,
        failed=failed,
        skipped=skipped,
        cancelled=cancelled,
        current_path=current_path,
        current_phase=phase,
    )
