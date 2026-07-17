from __future__ import annotations

from dataclasses import dataclass

from zesolver.core.models import SolveRequest, SolveResult


@dataclass(frozen=True, slots=True)
class BatchSolveRequest:
    requests: tuple[SolveRequest, ...]
    workers: int = 1
    io_concurrency: int = 1
    preserve_order: bool = True
    stop_on_error: bool = False
    cancel_token: object | None = None


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
