from __future__ import annotations

import pytest

from zesolver.engine_selection import EngineMode, EngineSelectionRequest, select_engine


def test_auto_fits_local_simple_uses_pipeline() -> None:
    selected = select_engine(EngineSelectionRequest(input_path="m106.fit", backend="local", blind4d_all_sky=True))
    assert selected.selected_mode == EngineMode.PIPELINE
    assert selected.supported is True
    assert selected.reason


def test_auto_fits_local_near_blind_uses_pipeline() -> None:
    selected = select_engine(EngineSelectionRequest(input_path="m106.fits", backend="near_blind4d", blind4d_all_sky=True))
    assert selected.selected_mode == EngineMode.PIPELINE


def test_auto_batch_fits_threads_uses_pipeline() -> None:
    selected = select_engine(
        EngineSelectionRequest(input_path="m106.fts", is_batch=True, workers=4, worker_strategy="threads", blind4d_all_sky=True)
    )
    assert selected.selected_mode == EngineMode.PIPELINE


@pytest.mark.parametrize("path", ["frame.tif", "frame.png", "frame.jpeg"])
def test_auto_raster_uses_legacy(path: str) -> None:
    selected = select_engine(EngineSelectionRequest(input_path=path, blind4d_all_sky=True))
    assert selected.selected_mode == EngineMode.LEGACY
    assert selected.supported is True
    assert "raster_not_supported_by_pipeline" in selected.reason


def test_auto_web_backend_uses_legacy() -> None:
    selected = select_engine(EngineSelectionRequest(input_path="m106.fit", backend="astrometry.net", blind4d_all_sky=True))
    assert selected.selected_mode == EngineMode.LEGACY
    assert "web" in selected.reason


def test_auto_fallback_web_uses_legacy() -> None:
    selected = select_engine(EngineSelectionRequest(input_path="m106.fit", fallback_web=True, blind4d_all_sky=True))
    assert selected.selected_mode == EngineMode.LEGACY
    assert "fallback_web_requires_legacy" in selected.reason


def test_pipeline_explicit_raster_is_rejected_without_fallback() -> None:
    selected = select_engine(EngineSelectionRequest(input_path="frame.tiff", requested_mode=EngineMode.PIPELINE, blind4d_all_sky=True))
    assert selected.selected_mode == EngineMode.PIPELINE
    assert selected.supported is False
    assert "pipeline_unsupported" in selected.reason


def test_pipeline_explicit_web_is_rejected_without_fallback() -> None:
    selected = select_engine(
        EngineSelectionRequest(input_path="m106.fit", requested_mode=EngineMode.PIPELINE, backend="web", blind4d_all_sky=True)
    )
    assert selected.selected_mode == EngineMode.PIPELINE
    assert selected.supported is False
    assert "web" in selected.reason


def test_legacy_explicit_uses_legacy() -> None:
    selected = select_engine(EngineSelectionRequest(input_path="m106.fit", requested_mode="legacy"))
    assert selected.selected_mode == EngineMode.LEGACY
    assert selected.supported is True
    assert selected.reason == "legacy_requested"


def test_unknown_capability_uses_legacy_in_auto() -> None:
    selected = select_engine(
        EngineSelectionRequest(input_path="m106.fit", unknown_capabilities=("process_near",), blind4d_all_sky=True)
    )
    assert selected.selected_mode == EngineMode.LEGACY
    assert "unknown_capability:process_near" in selected.reason


def test_reason_is_always_present() -> None:
    cases = (
        EngineSelectionRequest(input_path="m106.fit"),
        EngineSelectionRequest(input_path="frame.png"),
        EngineSelectionRequest(input_path="m106.fit", requested_mode=EngineMode.PIPELINE),
        EngineSelectionRequest(input_path="m106.fit", requested_mode=EngineMode.LEGACY),
    )
    assert all(select_engine(case).reason for case in cases)


def test_partial_coverage_is_warning_not_all_sky_promotion() -> None:
    selected = select_engine(EngineSelectionRequest(input_path="m106.fit", backend="near_blind4d", blind4d_all_sky=False))
    assert selected.selected_mode == EngineMode.PIPELINE
    assert "blind4d_coverage_partial_not_all_sky" in selected.warnings
