from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from zeblindsolver import metadata_solver as ms
from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration


def _blank_image() -> np.ndarray:
    return np.ones((80, 80), dtype=np.float32) * 1000.0


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    ms.reset_zenear_gpu_runtime_state()
    yield
    ms.reset_zenear_gpu_runtime_state()


def test_s6a2_product_settings_route_device_and_slots() -> None:
    product = ProductSettings(gpu_mode="cuda", near_detect_device=2, near_detect_gpu_slots=3)
    cfg = build_solver_configuration(product_settings=product, runtime_options=RuntimeOptions())

    assert cfg.legacy_solve_config_values["near_detect_backend"] == "cuda"
    assert cfg.legacy_solve_config_values["near_detect_device"] == 2
    assert cfg.legacy_solve_config_values["near_detect_gpu_slots"] == 3


def test_s6a2_product_settings_are_immutable_with_gpu_fields() -> None:
    product = ProductSettings()
    changed = replace(product, gpu_mode="cpu", near_detect_device=None, near_detect_gpu_slots=1)

    assert changed.gpu_mode == "cpu"
    assert changed.near_detect_gpu_slots == 1


def test_s6a2_gpu_slots_bound_concurrent_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: {"device": 0, "device_count": 1})
    active = 0
    max_active = 0
    lock = threading.Lock()
    release = threading.Event()

    def _slow_bin(image, factor, *, crop=1.0, device=None):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        release.wait(timeout=2.0)
        with lock:
            active -= 1
        binned, used = ms.astap_compatible_mean_bin_image(image, factor, crop=crop)
        return binned, used, {"transfer_to_gpu_s": 0.0, "gpu_compute_s": 0.01, "transfer_to_cpu_s": 0.0}

    monkeypatch.setattr(ms, "_astap_compatible_mean_bin_image_cuda", _slow_bin)
    threads = [
        threading.Thread(
            target=ms.detect_stars_astap_strict,
            kwargs={"image": _blank_image(), "backend": "cuda", "gpu_slots": 2, "bin_factor": 1, "max_stars": 1},
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    release.set()
    for thread in threads:
        thread.join(timeout=3.0)

    assert max_active == 2
    assert all(not thread.is_alive() for thread in threads)


def test_s6a2_stop_while_waiting_for_gpu_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_cuda_runtime_probe", lambda device=None: {"device": 0, "device_count": 1})
    release = threading.Event()

    def _slow_bin(image, factor, *, crop=1.0, device=None):
        release.wait(timeout=2.0)
        binned, used = ms.astap_compatible_mean_bin_image(image, factor, crop=crop)
        return binned, used, {}

    monkeypatch.setattr(ms, "_astap_compatible_mean_bin_image_cuda", _slow_bin)
    first = threading.Thread(
        target=ms.detect_stars_astap_strict,
        kwargs={"image": _blank_image(), "backend": "cuda", "gpu_slots": 1, "bin_factor": 1, "max_stars": 1},
    )
    first.start()
    time.sleep(0.1)

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="cancelled_waiting_for_gpu_slot"):
        ms.detect_stars_astap_strict(
            _blank_image(),
            backend="cuda",
            gpu_slots=1,
            bin_factor=1,
            max_stars=1,
            cancel_check=cancelled.is_set,
        )
    release.set()
    first.join(timeout=3.0)
