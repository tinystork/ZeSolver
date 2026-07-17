from __future__ import annotations

from pathlib import Path

from zesolver.core import SolveRequest, SolveResult, SolveStatus


def test_solver_pipeline_models_are_explicit_and_immutable() -> None:
    request = SolveRequest(
        input_path=Path("in.fit"),
        output_path=Path("out.fit"),
        overwrite_wcs=False,
        metadata_overrides={"OBJECT": "M 106"},
        request_id="req-1",
    )

    assert request.request_id == "req-1"
    assert SolveStatus.SOLVED.value == "SOLVED"
    assert {status.value for status in SolveStatus} >= {
        "SOLVED",
        "UNSOLVED",
        "REJECTED_FALSE_SOLUTION",
        "INVALID_INPUT",
        "CATALOG_UNAVAILABLE",
        "CANCELLED",
        "FAILED",
    }

    result = SolveResult(
        request_id=request.request_id,
        input_path=request.input_path,
        output_path=request.output_path,
        status=SolveStatus.UNSOLVED,
        backend=None,
        wcs_written=False,
        center_ra_deg=None,
        center_dec_deg=None,
        pixel_scale_arcsec=None,
        orientation_deg=None,
        parity=None,
        inliers=None,
        rms_px=None,
        profile_ids={"near": "zenear-v1", "blind": "zeblind4d-v1", "pipeline": "pipeline-v1"},
        catalog_status="legacy",
        warnings=(),
        error="no solution",
    )

    assert result.profile_ids["pipeline"] == "pipeline-v1"
    assert result.status is SolveStatus.UNSOLVED
