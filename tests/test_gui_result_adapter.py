from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from zesolver.core.models import SolveResult, SolveStatus
from zesolver.gui_pipeline.result_adapter import gui_result_from_legacy, gui_result_from_solve_result


def test_pipeline_result_preserves_status_and_details() -> None:
    result = SolveResult(
        request_id="r1",
        input_path=Path("a.fit"),
        output_path=None,
        status=SolveStatus.SOLVED,
        backend="NEAR",
        wcs_written=True,
        center_ra_deg=None,
        center_dec_deg=None,
        pixel_scale_arcsec=1.2,
        orientation_deg=None,
        parity=None,
        inliers=42,
        rms_px=0.8,
        profile_ids={},
        catalog_status="legacy",
        warnings=("w",),
        error=None,
    )
    gui = gui_result_from_solve_result(result)
    assert gui.status == "SOLVED"
    assert gui.backend == "NEAR"
    assert gui.inliers == 42
    assert gui.legacy_status == "solved"


def test_legacy_result_maps_without_losing_message() -> None:
    raw = SimpleNamespace(path=Path("a.fit"), status="failed", message="no solution", run_info=[])
    gui = gui_result_from_legacy(raw)
    assert gui.status == "FAILED"
    assert gui.errors == ("no solution",)
    assert gui.legacy_status == "failed"
