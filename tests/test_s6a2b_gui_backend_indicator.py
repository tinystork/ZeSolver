from __future__ import annotations

from zesolver.gui_pipeline.pipeline_runner import _near_backend_indicator_from_telemetry


def test_s6a2b_gui_indicator_formats_cuda_cpu_and_fallback() -> None:
    assert (
        _near_backend_indicator_from_telemetry(
            {
                "near_detection": {
                    "requested": "auto",
                    "selected_last": "cuda",
                    "backends_used": ["cuda"],
                    "devices_used": [0],
                    "fallbacks": 0,
                }
            }
        )
        == "ZeNear : Auto -> CUDA - GPU 0"
    )
    assert (
        _near_backend_indicator_from_telemetry(
            {
                "near_detection": {
                    "requested": "auto",
                    "selected_last": "cpu",
                    "backends_used": ["cpu"],
                    "devices_used": [],
                    "fallbacks": 0,
                }
            }
        )
        == "ZeNear : Auto -> CPU"
    )
    assert (
        _near_backend_indicator_from_telemetry(
            {
                "near_detection": {
                    "requested": "auto",
                    "selected_last": "cuda",
                    "backends_used": ["cpu"],
                    "devices_used": [],
                    "fallbacks": 1,
                }
            }
        )
        == "ZeNear : Auto -> CPU - fallback CUDA"
    )


def test_s6a2b_gui_indicator_returns_none_before_backend_used() -> None:
    assert _near_backend_indicator_from_telemetry({"near_detection": {"requested": "auto"}}) is None
