from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from zesolver.core.models import SolveRequest, SolveResult


@dataclass(frozen=True, slots=True)
class BatchSolveRequest:
    requests: tuple[SolveRequest, ...]
    workers: int = 1
    io_concurrency: int = 1
    preserve_order: bool = True
    stop_on_error: bool = False
    blind_enabled: bool = True
    cancel_token: object | None = None
    startup_stagger_ms: int = 0
    input_root: Path | None = None
    move_unresolved_files: bool = False
    run_id: int | None = None
    started_at: str | None = None


@dataclass(frozen=True, slots=True)
class BatchProgress:
    total: int
    queued: int
    running: int
    solved: int
    failed: int
    skipped: int
    cancelled: int


@dataclass(frozen=True, slots=True)
class BatchSolveResult:
    results: tuple[SolveResult, ...]
    progress: BatchProgress
    cancelled: bool
    duration_s: float
    telemetry: Mapping[str, object] | None = None
