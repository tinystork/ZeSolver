from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from catalog_resource_helpers import write_catalog_library
from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "measure_s6a1c_native_threading.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("measure_s6a1c_native_threading", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _threadpool_counts() -> list[int]:
    from threadpoolctl import threadpool_info

    return [int(item.get("num_threads", 0) or 0) for item in threadpool_info()]


def _require_native_pool() -> list[int]:
    import numpy  # noqa: F401

    counts = _threadpool_counts()
    if not counts:
        pytest.skip("no threadpoolctl-compatible native pool is loaded")
    return counts


def test_s6a1c_threadpool_info_is_accessible() -> None:
    import threadpoolctl
    from threadpoolctl import threadpool_info

    assert tuple(int(part) for part in threadpoolctl.__version__.split(".")[:2]) >= (3, 6)
    assert isinstance(threadpool_info(), list)


def test_s6a1c_threadpool_limits_one_two_and_restore_after_success() -> None:
    initial = _require_native_pool()
    tool = _load_tool_module()

    with tool.native_thread_context("1", "all"):
        limited = _threadpool_counts()
        assert limited
        assert all(value == 1 for value in limited)

    assert _threadpool_counts() == initial

    with tool.native_thread_context("2", "all"):
        limited = _threadpool_counts()
        assert limited
        assert all(value == 2 for value in limited)

    assert _threadpool_counts() == initial


def test_s6a1c_threadpool_limits_restore_after_exception() -> None:
    initial = _require_native_pool()
    tool = _load_tool_module()

    with pytest.raises(RuntimeError):
        with tool.native_thread_context("1", "all"):
            assert all(value == 1 for value in _threadpool_counts())
            raise RuntimeError("synthetic")

    assert _threadpool_counts() == initial


def test_s6a1c_baseline_default_context_does_not_change_limits() -> None:
    initial = _require_native_pool()
    tool = _load_tool_module()

    with tool.native_thread_context("default", "all"):
        assert _threadpool_counts() == initial

    assert _threadpool_counts() == initial


def test_s6a1c_stop_under_native_limit_restores_and_closes_runtime(tmp_path: Path, monkeypatch) -> None:
    initial = _require_native_pool()
    tool = _load_tool_module()
    library = _library_with_astap_tile(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library)
    files = tuple(_fits(tmp_path / f"frame_{idx}.fit") for idx in range(8))
    cancel = threading.Event()
    emitted: list[str] = []

    def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
        del fits_path, index_root, catalog_provider, kwargs
        for _ in range(10):
            if cancel.is_set():
                return {"success": False, "message": "synthetic cancelled", "stats": {}, "wrote_wcs": False}
            time.sleep(0.005)
        return {"success": False, "message": "synthetic miss", "stats": {}, "wrote_wcs": False}

    def on_result(result) -> None:
        emitted.append(str(result.status))
        cancel.set()

    monkeypatch.setattr("zesolver.core.pipeline.near_solve", fake_near_solve)
    request = build_gui_solve_request(files, _state(resources, workers=6), cancel_token=cancel)

    with tool.native_thread_context("1", "all"):
        assert all(value == 1 for value in _threadpool_counts())
        summary = PipelineGuiRunner(result_callback=on_result).run(request)

    assert emitted
    assert summary.cancelled is True
    assert len(summary.results) == len(files)
    assert summary.telemetry["counters"]["near_catalog_runtime_closed"] == 1
    assert _threadpool_counts() == initial


def test_s6a1c_tool_outputs_complete_json_from_fresh_subprocess(tmp_path: Path) -> None:
    library = _library_with_astap_tile(tmp_path)
    inputs = [_fits(tmp_path / f"input_{idx}.fit") for idx in range(2)]
    out = tmp_path / "s6a1c.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--input-dir",
            str(tmp_path),
            "--catalog-library",
            str(library),
            "--workers",
            "2",
            "--native-threads",
            "1",
            "--max-files",
            str(len(inputs)),
            "--repeat",
            "1",
            "--max-loadavg1",
            "999",
            "--stub-near",
            "--json-output",
            str(out),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=90,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["runs"]
    row = payload["runs"][0]
    assert row["valid"] is True
    assert row["workers"] == 2
    assert row["native_threads"] == "1"
    assert row["threadpoolctl_version"]
    assert "after_numpy" in row["threadpool_info"]
    assert "during_limit" in row["threadpool_info"]
    assert "after_restore" in row["threadpool_info"]
    assert isinstance(row["process_thread_sampling_supported"], bool)
    if row["process_thread_sampling_supported"]:
        assert row["threads_process_peak"] is not None
    else:
        assert row["threads_process_peak"] is None
    assert row["ru_nvcsw_delta"] >= 0
    assert sum(int(value) for value in row["statuses"].values()) == len(inputs)
    assert row["counters"]["near_catalog_runtime_closed"] == 1


def _fits(path: Path) -> Path:
    fits.PrimaryHDU(data=np.ones((16, 16), dtype=np.uint16)).writeto(path)
    return path


def _library_with_astap_tile(tmp_path: Path) -> Path:
    root = write_catalog_library(tmp_path / "library", include_source=True, index_paths=[])
    write_astap_1476_tile(
        root / "sources" / "astap" / "d50",
        family="d50",
        tile_code="2823",
        ra_deg=np.asarray([184.6, 184.7], dtype=np.float64),
        dec_deg=np.asarray([47.2, 47.3], dtype=np.float64),
    )
    return root


def _state(resources, *, workers: int) -> GuiSettingsState:
    return GuiSettingsState(
        workers=workers,
        preserve_order=True,
        use_blind=False,
        catalog_resources=resources,
        legacy_config=object(),
    )
