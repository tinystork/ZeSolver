from __future__ import annotations

import threading

import numpy as np
import pytest

from zeblindsolver import metadata_solver as ms
from zesolver.resource_telemetry import BatchResourceTelemetry


def _image() -> np.ndarray:
    data = np.ones((96, 96), dtype=np.float32) * 1000.0
    yy, xx = np.ogrid[:96, :96]
    data += 5000.0 * np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    return data


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    ms.reset_zenear_gpu_runtime_state()
    yield
    ms.reset_zenear_gpu_runtime_state()


def test_s6a2b_cancel_after_slot_acquire_does_not_start_gpu_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: {"device": 0, "device_count": 1})
    called = False
    checks = 0

    def cancel_after_acquire() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    def _must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("GPU section started after cancel")

    monkeypatch.setattr(ms, "_astap_compatible_mean_bin_image_cuda", _must_not_run)

    with pytest.raises(RuntimeError, match="cancelled_after_gpu_slot_acquired"):
        ms.detect_stars_astap_strict(
            _image(),
            backend="cuda",
            gpu_slots=1,
            bin_factor=1,
            max_stars=4,
            cancel_check=cancel_after_acquire,
        )

    assert called is False


def test_s6a2b_telemetry_counts_detection_and_gpu_section_after_cancel() -> None:
    telemetry = BatchResourceTelemetry()
    cancel_at = telemetry.mark_cancel_requested(source="test")

    telemetry.record_near_detection(
        {
            "backend_requested": "auto",
            "backend_selected": "cuda",
            "backend_used": "cuda",
            "device_used": 0,
            "detect_started_at": cancel_at - 1.0,
            "gpu_section_started_at": cancel_at - 0.5,
            "gpu_section_finished_at": cancel_at + 0.5,
        }
    )
    telemetry.record_near_detection(
        {
            "backend_requested": "auto",
            "backend_selected": "cuda",
            "backend_used": "cuda",
            "device_used": 0,
            "detect_started_at": cancel_at + 0.1,
            "gpu_section_started_at": cancel_at + 0.2,
            "gpu_section_finished_at": cancel_at + 0.3,
        }
    )

    summary = telemetry.near_detection_summary(terminal_status="cancelled")
    assert summary["detections_started_after_cancel"] == 1
    assert summary["gpu_sections_started_after_cancel"] == 1
    assert summary["gpu_sections_finished_after_cancel"] == 1


def test_s6a2b_batch_reset_clears_gpu_disabled_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: (_ for _ in ()).throw(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")))
    first = ms.detect_stars_astap_strict(_image(), backend="cuda", bin_factor=1, max_stars=4)
    assert first.gpu_disabled_for_batch is True

    ms.reset_zenear_gpu_runtime_state()
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: {"device": 0, "device_count": 1})
    monkeypatch.setattr(
        ms,
        "_astap_compatible_mean_bin_image_cuda",
        lambda image, factor, *, crop=1.0, device=None: (*ms.astap_compatible_mean_bin_image(image, factor, crop=crop), {}),
    )
    second = ms.detect_stars_astap_strict(_image(), backend="auto", bin_factor=1, max_stars=4)

    assert second.backend_selected == "cuda"
    assert second.backend_used == "cuda"
    assert second.gpu_disabled_for_batch is False
