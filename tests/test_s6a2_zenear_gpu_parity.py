from __future__ import annotations

import numpy as np
import pytest

from zeblindsolver import metadata_solver as ms


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    ms.reset_zenear_gpu_runtime_state()
    yield
    ms.reset_zenear_gpu_runtime_state()


def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: {"device": 0, "device_count": 1})

    def _bin(image, factor, *, crop=1.0, device=None):
        binned, used = ms.astap_compatible_mean_bin_image(image, factor, crop=crop)
        return binned, used, {"transfer_to_gpu_s": 0.0, "gpu_compute_s": 0.0, "transfer_to_cpu_s": 0.0}

    monkeypatch.setattr(ms, "_astap_compatible_mean_bin_image_cuda", _bin)


def _field(kind: str) -> np.ndarray:
    data = np.ones((180, 180), dtype=np.float32) * 1000.0
    if kind == "empty":
        return data
    yy, xx = np.ogrid[:180, :180]
    sources = [(45, 50, 5000), (100, 120, 7500), (140, 70, 6200)]
    if kind == "edge":
        sources.append((18, 24, 9000))
    if kind == "equal_flux":
        sources = [(50, 50, 6000), (90, 90, 6000), (130, 110, 6000)]
    for y, x, amp in sources:
        data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    if kind == "nan_inf":
        data[1, 1] = np.nan
        data[2, 2] = np.inf
    return data


@pytest.mark.parametrize("kind", ["empty", "normal", "edge", "equal_flux"])
def test_s6a2_cpu_cuda_source_parity_on_synthetic_fields(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    _fake_cuda(monkeypatch)
    image = _field(kind)

    cpu = ms.detect_stars_astap_strict(image, backend="cpu", bin_factor=1, max_stars=20)
    gpu = ms.detect_stars_astap_strict(image, backend="cuda", bin_factor=1, max_stars=20)

    assert gpu.backend_used == "cuda"
    assert int(cpu.sources.size) == int(gpu.sources.size)
    np.testing.assert_array_equal(gpu.sources["x"], cpu.sources["x"])
    np.testing.assert_array_equal(gpu.sources["y"], cpu.sources["y"])
    np.testing.assert_array_equal(gpu.sources["flux"], cpu.sources["flux"])


def test_s6a2_integer_fits_style_input_keeps_numpy_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    image = np.rint(_field("normal")).astype(np.uint16)

    result = ms.detect_stars_astap_strict(image, backend="cuda", bin_factor=1, max_stars=20)

    assert result.backend_used == "cuda"
    assert isinstance(result.sources, np.ndarray)
    assert result.sources.dtype == ms._STAR_DETECT_DTYPE
