from __future__ import annotations

import json
import logging

from zesolver.resource_telemetry import (
    BatchResourceTelemetry,
    build_run_telemetry_payload,
    record_near_detection,
    reset_active_batch_telemetry,
    set_active_batch_telemetry,
    write_run_telemetry_sidecar,
)


def test_s6a2b_near_detection_active_and_summary_are_aggregated(caplog) -> None:
    telemetry = BatchResourceTelemetry()
    token = set_active_batch_telemetry(telemetry)
    try:
        with caplog.at_level(logging.INFO):
            record_near_detection(
                {
                    "backend_requested": "auto",
                    "backend_selected": "cuda",
                    "backend_used": "cuda",
                    "device_used": 0,
                    "gpu_slots": 1,
                    "duration_total_s": 0.25,
                    "duration_transfer_to_gpu_s": 0.01,
                    "duration_gpu_compute_s": 0.02,
                    "duration_transfer_to_cpu_s": 0.003,
                    "gpu_slot_wait_s": 0.004,
                    "vram_peak_bytes": 1234,
                    "cupy_pool_reserved_bytes": 512,
                }
            )
            record_near_detection(
                {
                    "backend_requested": "auto",
                    "backend_selected": "cuda",
                    "backend_used": "cuda",
                    "device_used": 0,
                    "gpu_slots": 1,
                    "duration_total_s": 0.35,
                    "gpu_slot_wait_s": 0.006,
                }
            )
        summary = telemetry.near_detection_summary(terminal_status="completed")
    finally:
        reset_active_batch_telemetry(token)

    assert summary["requested"] == "auto"
    assert summary["selected_initial"] == "cuda"
    assert summary["backends_used"] == ["cuda"]
    assert summary["devices_used"] == [0]
    assert summary["images_cuda"] == 2
    assert summary["images_cpu"] == 0
    assert summary["fallbacks"] == 0
    assert summary["detect_duration_ms"]["median"] == 300.0
    assert summary["gpu_slot_wait_ms"]["p95"] == 6.0
    assert summary["transfer_h2d_ms_total"] == 10.0
    assert summary["gpu_compute_ms_total"] == 20.0
    assert summary["transfer_d2h_ms_total"] == 3.0
    assert summary["vram_peak_bytes"] == 1234
    assert sum("ZeNear detection active" in item.message for item in caplog.records) == 1


def test_s6a2b_fallback_is_warning_and_reason_is_aggregated(caplog) -> None:
    telemetry = BatchResourceTelemetry()
    token = set_active_batch_telemetry(telemetry)
    try:
        with caplog.at_level(logging.WARNING):
            record_near_detection(
                {
                    "backend_requested": "cuda",
                    "backend_selected": "cuda",
                    "backend_used": "cpu",
                    "fallback_used": True,
                    "fallback_reason": "CUDA_ERROR_OUT_OF_MEMORY",
                    "gpu_disabled_for_batch": True,
                    "gpu_disabled_reason": "CUDA_ERROR_OUT_OF_MEMORY",
                }
            )
        summary = telemetry.near_detection_summary(terminal_status="cancelled")
    finally:
        reset_active_batch_telemetry(token)

    assert summary["images_cpu"] == 1
    assert summary["fallbacks"] == 1
    assert summary["fallback_reasons"] == {"CUDA_ERROR_OUT_OF_MEMORY": 1}
    assert summary["gpu_disabled_for_batch"] is True
    assert any("ZeNear CUDA fallback" in item.message for item in caplog.records)


def test_s6a2b_sidecar_json_uses_log_basename_and_schema(tmp_path) -> None:
    telemetry = BatchResourceTelemetry()
    telemetry.record_near_detection(
        {
            "backend_requested": "auto",
            "backend_selected": "cpu",
            "backend_used": "cpu",
            "duration_total_s": 0.1,
        }
    )
    payload = build_run_telemetry_payload(
        run_id=7,
        started_at="2026-07-26T10:00:00+0200",
        finished_at="2026-07-26T10:00:10+0200",
        duration_s=10.0,
        terminal_status="completed",
        planned=3,
        solved=3,
        failed=0,
        cancelled=0,
        telemetry=telemetry.snapshot(),
    )
    log_path = tmp_path / "zesolver_run_20260726_100000.log"
    log_path.write_text("log\n", encoding="utf-8")

    sidecar = write_run_telemetry_sidecar(log_path, payload)
    data = json.loads(sidecar.read_text(encoding="utf-8"))

    assert sidecar.name == "zesolver_run_20260726_100000.telemetry.json"
    assert data["schema"] == "zesolver.run_telemetry.v1"
    assert data["run"]["terminal_status"] == "completed"
    assert data["input"]["processed"] == 3
    assert data["near_detection"]["images_cpu"] == 1


def test_s6a2b_sidecar_payload_supports_cancelled_and_failed() -> None:
    for terminal, solved, failed, cancelled in (("cancelled", 1, 0, 2), ("failed", 1, 1, 0)):
        payload = build_run_telemetry_payload(
            run_id=1,
            started_at=None,
            finished_at=None,
            duration_s=None,
            terminal_status=terminal,
            planned=3,
            solved=solved,
            failed=failed,
            cancelled=cancelled,
            telemetry={},
        )
        assert payload["run"]["terminal_status"] == terminal
        assert payload["input"]["processed"] == solved + failed + cancelled
