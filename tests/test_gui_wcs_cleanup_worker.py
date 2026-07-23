from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from tests.p3av3_helpers import write_fits


QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from astropy.io import fits

from zesolver.gui_wcs_cleanup import WcsCleanupConfig, WcsCleanupRunner
from zewcscleaner import process_fits


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pump_until(app, predicate, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached before timeout")


def _run_worker(worker: WcsCleanupRunner, app):
    worker.start()
    _pump_until(app, lambda: worker.isFinished(), timeout_s=5.0)
    worker.wait(1000)
    app.processEvents()


def test_wcs_cleanup_worker_preserves_order_progress_and_single_terminal(tmp_path: Path) -> None:
    app = _app()
    files = [tmp_path / f"{idx}.fit" for idx in range(3)]
    calls = []

    def fake_process(path: str, **_kwargs):
        calls.append(Path(path))
        return (10 + len(calls), 1)

    worker = WcsCleanupRunner(WcsCleanupConfig.from_files(files), process_fits_func=fake_process)
    results = []
    progress = []
    terminals = []
    worker.file_result.connect(results.append)
    worker.progress.connect(progress.append)
    worker.completed.connect(lambda summary: terminals.append(summary.terminal_status))
    worker.cancelled.connect(lambda summary: terminals.append(summary.terminal_status))
    worker.fatal_error.connect(lambda error: terminals.append(error.final_status))

    _run_worker(worker, app)

    assert calls == files
    assert [event.path for event in results] == files
    assert [event.completed for event in progress] == [1, 2, 3]
    assert [event.remaining for event in progress] == [2, 1, 0]
    assert terminals == ["completed"]
    assert worker.isFinished()


def test_wcs_cleanup_worker_reports_file_exception_and_stops(tmp_path: Path) -> None:
    app = _app()
    files = [tmp_path / "a.fit", tmp_path / "b.fit", tmp_path / "c.fit"]
    calls = []

    def fake_process(path: str, **_kwargs):
        calls.append(Path(path))
        if Path(path).name == "b.fit":
            raise RuntimeError("boom")
        return (1, 1)

    worker = WcsCleanupRunner(WcsCleanupConfig.from_files(files), process_fits_func=fake_process)
    file_errors = []
    fatal_errors = []
    terminals = []
    worker.file_error.connect(file_errors.append)
    worker.fatal_error.connect(lambda error: (fatal_errors.append(error), terminals.append("fatal")))
    worker.completed.connect(lambda summary: terminals.append(summary.terminal_status))
    worker.cancelled.connect(lambda summary: terminals.append(summary.terminal_status))

    _run_worker(worker, app)

    assert calls == files[:2]
    assert len(file_errors) == 1
    assert file_errors[0].path == files[1]
    assert file_errors[0].operation == "process_fits"
    assert "boom" in file_errors[0].message
    assert len(fatal_errors) == 1
    assert terminals == ["fatal"]


def test_wcs_cleanup_worker_cancels_between_files(tmp_path: Path) -> None:
    app = _app()
    files = [tmp_path / "a.fit", tmp_path / "b.fit", tmp_path / "c.fit"]
    calls = []
    release_current = threading.Event()

    def fake_process(path: str, **_kwargs):
        calls.append(Path(path))
        release_current.wait(timeout=2)
        return (1, 1)

    worker = WcsCleanupRunner(WcsCleanupConfig.from_files(files), process_fits_func=fake_process)
    cancelled = []
    worker.cancelled.connect(cancelled.append)
    worker.start()

    _pump_until(app, lambda: len(calls) == 1)
    worker.request_cancel()
    release_current.set()
    _pump_until(app, lambda: worker.isFinished(), timeout_s=5.0)
    worker.wait(1000)
    app.processEvents()

    assert calls == files[:1]
    assert len(cancelled) == 1
    assert cancelled[0].processed == 1
    assert cancelled[0].remaining == 2


def test_wcs_cleanup_runs_process_fits_outside_main_thread_and_slots_in_main(tmp_path: Path) -> None:
    app = _app()
    file_path = tmp_path / "a.fit"
    worker_thread_ids = []
    slot_thread_is_main = []
    timer_fired = []
    release = threading.Event()

    def fake_process(path: str, **_kwargs):
        worker_thread_ids.append(threading.get_ident())
        release.wait(timeout=2)
        return (1, 1)

    worker = WcsCleanupRunner(WcsCleanupConfig.from_files([file_path]), process_fits_func=fake_process)
    worker.file_result.connect(lambda _result: slot_thread_is_main.append(QtCore.QThread.currentThread() is app.thread()))
    QtCore.QTimer.singleShot(0, lambda: timer_fired.append(True))
    worker.start()

    _pump_until(app, lambda: bool(worker_thread_ids) and bool(timer_fired))
    assert worker_thread_ids[0] != threading.get_ident()
    release.set()
    _pump_until(app, lambda: worker.isFinished(), timeout_s=5.0)
    worker.wait(1000)
    app.processEvents()

    assert slot_thread_is_main == [True]


def test_wcs_cleanup_real_fits_preserves_pixels_hdus_and_backup(tmp_path: Path) -> None:
    path = write_fits(tmp_path / "with_wcs.fit", primary_wcs=True, extension_wcs=True)
    with fits.open(path) as hdul:
        before_data = [hdu.data.copy() if hdu.data is not None else None for hdu in hdul]
        before_hdus = len(hdul)
        before_ext_header = hdul[1].header.copy()

    deleted, edited = process_fits(str(path), dry_run=False, backup=True, only_if_wcs=True, all_hdus=False)

    assert deleted > 0
    assert edited == 1
    backup = Path(str(path) + ".bak")
    assert backup.exists()
    with fits.open(path) as hdul:
        assert len(hdul) == before_hdus
        assert "CTYPE1" not in hdul[0].header
        assert "CTYPE2" not in hdul[0].header
        assert "CTYPE1" in hdul[1].header
        for index, before in enumerate(before_data):
            if before is None:
                assert hdul[index].data is None
            else:
                assert (hdul[index].data == before).all()
        assert hdul[1].header["CTYPE1"] == before_ext_header["CTYPE1"]
    with fits.open(backup) as hdul:
        assert len(hdul) == before_hdus
        assert "CTYPE1" in hdul[0].header
