from __future__ import annotations

import threading

from tests.legacy_stop_helpers import (
    BlockingImageSolver,
    ImmediateImageSolver,
    load_zesolver_app,
    make_config,
    make_paths,
    run_legacy_stop_case,
)


def test_new_run_creates_fresh_executor_after_stop(monkeypatch, tmp_path) -> None:
    run_legacy_stop_case(
        monkeypatch,
        tmp_path / "first",
        BlockingImageSolver,
        files=3,
        workers=2,
        wait_started=2,
        grace="0.2",
    )

    zs = load_zesolver_app()

    second_root = tmp_path / "second"
    second_root.mkdir()
    started_dir = second_root / "started"
    ImmediateImageSolver.started_dir = str(started_dir)
    ImmediateImageSolver.app_module = zs
    monkeypatch.setattr(zs, "ImageSolver", ImmediateImageSolver)
    monkeypatch.setenv("ZE_NEAR_PARALLEL_MODE", "process")
    paths = make_paths(second_root, 3)
    batch = zs.BatchSolver(make_config(zs, second_root, workers=2), files=paths)
    results = list(batch.run(cancel_event=threading.Event()))
    assert len(results) == 3
    assert all(item.status == "solved" for item in results)
