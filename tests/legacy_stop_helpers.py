from __future__ import annotations

import multiprocessing
import os
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Callable


def load_zesolver_app():
    root = Path(__file__).resolve().parents[1]
    name = "zesolver_app_stop_tests"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, root / "zesolver.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CooperativeImageSolver:
    started_dir: str = ""
    app_module = None

    def __init__(self, config) -> None:
        self._cancel_event = None

    def set_cancel_event(self, event) -> None:
        self._cancel_event = event

    def solve_path(self, path: Path, *, allow_blind_fallback: bool = True):
        zs = self.app_module

        Path(self.started_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.started_dir) / f"{os.getpid()}-{Path(path).name}.started").write_text("started")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self._cancel_event is not None and self._cancel_event.is_set():
                return zs.ImageSolveResult(path=Path(path), status="cancelled", message="cancelled")
            time.sleep(0.01)
        return zs.ImageSolveResult(path=Path(path), status="failed", message="timeout")

    def solve_path_blind_only(self, path: Path, *, near_failure=None):
        zs = self.app_module

        return zs.ImageSolveResult(path=Path(path), status="failed", message="blind should not run")


class BlockingImageSolver(CooperativeImageSolver):
    def solve_path(self, path: Path, *, allow_blind_fallback: bool = True):
        zs = self.app_module

        Path(self.started_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.started_dir) / f"{os.getpid()}-{Path(path).name}.started").write_text("started")
        time.sleep(30)
        return zs.ImageSolveResult(path=Path(path), status="solved", message="late")


class ImmediateImageSolver(CooperativeImageSolver):
    def solve_path(self, path: Path, *, allow_blind_fallback: bool = True):
        zs = self.app_module

        Path(self.started_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.started_dir) / f"{os.getpid()}-{Path(path).name}.started").write_text("started")
        return zs.ImageSolveResult(path=Path(path), status="solved", message="ok")


def make_config(zs, tmp_path: Path, *, workers: int = 2):
    return zs.SolveConfig(
        input_dir=tmp_path,
        db_root=tmp_path,
        families=("d50",),
        blind_index_path=tmp_path,
        workers=workers,
        blind_enabled=False,
        overwrite=True,
        formats=(".fit",),
    )


def make_paths(tmp_path: Path, count: int) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in range(count):
        path = tmp_path / f"frame_{idx:02d}.fit"
        path.write_bytes(b"synthetic")
        paths.append(path)
    return paths


def wait_for_started(started_dir: Path, count: int, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(list(started_dir.glob("*.started"))) >= count:
            return
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {count} started workers")


def run_batch_in_thread(batch, cancel_event: threading.Event) -> tuple[threading.Thread, list]:
    results: list = []

    def target() -> None:
        results.extend(list(batch.run(cancel_event=cancel_event)))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, results


def active_child_count() -> int:
    return len(multiprocessing.active_children())


def run_legacy_stop_case(
    monkeypatch,
    tmp_path: Path,
    solver_cls,
    *,
    files: int,
    workers: int,
    wait_started: int,
    grace: str = "0.5",
) -> tuple[float, list]:
    zs = load_zesolver_app()

    started_dir = tmp_path / "started"
    solver_cls.started_dir = str(started_dir)
    solver_cls.app_module = zs
    monkeypatch.setattr(zs, "ImageSolver", solver_cls)
    monkeypatch.setenv("ZE_NEAR_PARALLEL_MODE", "process")
    monkeypatch.setenv("ZE_STOP_GRACE_PERIOD_S", grace)
    paths = make_paths(tmp_path, files)
    batch = zs.BatchSolver(make_config(zs, tmp_path, workers=workers), files=paths)
    cancel_event = threading.Event()
    thread, results = run_batch_in_thread(batch, cancel_event)
    wait_for_started(started_dir, wait_started)
    t0 = time.perf_counter()
    cancel_event.set()
    thread.join(timeout=8)
    elapsed = time.perf_counter() - t0
    assert not thread.is_alive()
    assert len(results) == len(paths)
    assert len({item.path for item in results}) == len(paths)
    assert all(item.status == "cancelled" for item in results)
    return elapsed, results
