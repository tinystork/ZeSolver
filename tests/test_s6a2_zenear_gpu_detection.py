from __future__ import annotations

import numpy as np
import pytest

from zeblindsolver import metadata_solver as ms


def _synthetic_field() -> np.ndarray:
    image = np.ones((220, 220), dtype=np.float32) * 1000.0
    yy, xx = np.ogrid[:220, :220]
    for y, x, amp in ((60, 70, 5000), (120, 130, 7000), (160, 40, 6000), (80, 170, 6500)):
        image += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    return image


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    ms.reset_zenear_gpu_runtime_state()
    yield
    ms.reset_zenear_gpu_runtime_state()


def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ms,
        "_cuda_runtime_probe",
        lambda device=None: {
            "device": 0 if device is None else int(device),
            "device_count": 1,
            "device_name": "S6A2 fake GPU",
        },
    )

    def _bin(image, factor, *, crop=1.0, device=None):
        binned, used = ms.astap_compatible_mean_bin_image(image, factor, crop=crop)
        return binned, used, {
            "transfer_to_gpu_s": 0.001,
            "gpu_compute_s": 0.002,
            "transfer_to_cpu_s": 0.0015,
            "vram_before_bytes": 10,
            "vram_peak_bytes": 20,
            "vram_after_bytes": 12,
            "cupy_pool_used_bytes": 4,
            "cupy_pool_reserved_bytes": 32,
        }

    monkeypatch.setattr(ms, "_astap_compatible_mean_bin_image_cuda", _bin)


def test_s6a2_cpu_strict_detection_contract() -> None:
    stars, diag = ms.astap_adaptive_image_detection(_synthetic_field(), backend="cpu", bin_factor=1, max_stars=20)

    assert stars.dtype == ms._STAR_DETECT_DTYPE
    assert int(stars.size) == 4
    assert diag["backend_requested"] == "cpu"
    assert diag["backend_selected"] == "cpu"
    assert diag["backend_used"] == "cpu"
    assert diag["fallback_used"] is False


def test_s6a2_fake_cuda_preserves_source_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    image = _synthetic_field()

    cpu, cpu_diag = ms.astap_adaptive_image_detection(image, backend="cpu", bin_factor=1, max_stars=20)
    gpu, gpu_diag = ms.astap_adaptive_image_detection(image, backend="cuda", device=0, gpu_slots=1, bin_factor=1, max_stars=20)

    assert gpu_diag["backend_requested"] == "cuda"
    assert gpu_diag["backend_selected"] == "cuda"
    assert gpu_diag["backend_used"] == "cuda"
    assert gpu_diag["device_used"] == 0
    assert gpu_diag["gpu_stage"] == "mean_bin"
    assert gpu_diag["duration_transfer_to_gpu_s"] > 0
    assert gpu_diag["duration_gpu_compute_s"] > 0
    assert gpu_diag["duration_transfer_to_cpu_s"] > 0
    assert int(cpu_diag["selected_count"]) == int(gpu_diag["selected_count"])
    np.testing.assert_array_equal(gpu["x"], cpu["x"])
    np.testing.assert_array_equal(gpu["y"], cpu["y"])
    np.testing.assert_array_equal(gpu["flux"], cpu["flux"])


def test_s6a2_auto_selects_cuda_when_runtime_probe_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)

    _stars, diag = ms.astap_adaptive_image_detection(_synthetic_field(), backend="auto", bin_factor=1, max_stars=20)

    assert diag["backend_requested"] == "auto"
    assert diag["backend_selected"] == "cuda"
    assert diag["backend_used"] == "cuda"
