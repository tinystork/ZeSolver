from __future__ import annotations

import numpy as np
import pytest

from zeblindsolver import metadata_solver as ms


def _image() -> np.ndarray:
    data = np.ones((160, 160), dtype=np.float32) * 1000.0
    yy, xx = np.ogrid[:160, :160]
    data += 6000.0 * np.exp(-((xx - 80) ** 2 + (yy - 70) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    return data


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    ms.reset_zenear_gpu_runtime_state()
    yield
    ms.reset_zenear_gpu_runtime_state()


def test_s6a2_auto_falls_back_to_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: (_ for _ in ()).throw(RuntimeError("no CUDA device detected")))

    result = ms.detect_stars_astap_strict(_image(), backend="auto", bin_factor=1, max_stars=10)

    assert result.backend_requested == "auto"
    assert result.backend_selected == "cpu"
    assert result.backend_used == "cpu"
    assert result.fallback_used is True
    assert "no CUDA device" in str(result.fallback_reason)
    assert result.sources.dtype == ms._STAR_DETECT_DTYPE


def test_s6a2_explicit_cuda_invalid_device_falls_back_and_disables_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: (_ for _ in ()).throw(RuntimeError("CUDA device 7 out of range (devices=1)")))

    result = ms.detect_stars_astap_strict(_image(), backend="cuda", device=7, bin_factor=1, max_stars=10)

    assert result.backend_selected == "cuda"
    assert result.backend_used == "cpu"
    assert result.fallback_used is True
    assert result.gpu_disabled_for_batch is True
    assert "out of range" in str(result.gpu_disabled_reason)


def test_s6a2_gpu_exception_after_slot_reprocesses_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: {"device": 0, "device_count": 1})
    monkeypatch.setattr(
        ms,
        "_astap_compatible_mean_bin_image_cuda",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kernel launch failed")),
    )

    cpu = ms.detect_stars_astap_strict(_image(), backend="cpu", bin_factor=1, max_stars=10)
    result = ms.detect_stars_astap_strict(_image(), backend="cuda", bin_factor=1, max_stars=10)

    assert result.backend_selected == "cuda"
    assert result.backend_used == "cpu"
    assert result.fallback_used is True
    assert "kernel launch failed" in str(result.fallback_reason)
    np.testing.assert_array_equal(result.sources, cpu.sources)


def test_s6a2_oom_fallback_counts_as_structural_for_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: {"device": 0, "device_count": 1})
    monkeypatch.setattr(
        ms,
        "_astap_compatible_mean_bin_image_cuda",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")),
    )

    first = ms.detect_stars_astap_strict(_image(), backend="cuda", bin_factor=1, max_stars=10)
    second = ms.detect_stars_astap_strict(_image(), backend="auto", bin_factor=1, max_stars=10)

    assert first.backend_used == "cpu"
    assert first.fallback_used is True
    assert first.gpu_disabled_for_batch is True
    assert second.backend_selected == "cpu"
    assert second.backend_used == "cpu"
    assert second.fallback_used is True
    assert "OUT_OF_MEMORY" in str(second.fallback_reason)
