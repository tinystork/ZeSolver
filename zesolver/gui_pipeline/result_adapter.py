from __future__ import annotations

from pathlib import Path

from zesolver.core.models import SolveResult

from .requests import GuiFileResult


def gui_result_from_solve_result(result: SolveResult, *, selected_engine=None) -> GuiFileResult:
    error = result.error
    message = error or result.status.value
    return GuiFileResult(
        path=Path(result.output_path or result.input_path),
        status=result.status.value,
        message=message,
        backend=result.backend,
        inliers=result.inliers,
        rms_px=result.rms_px,
        pixel_scale_arcsec=result.pixel_scale_arcsec,
        wcs_written=result.wcs_written,
        warnings=tuple(result.warnings or ()),
        errors=(() if error is None else (str(error),)),
        selected_engine=selected_engine,
    )


def gui_result_from_legacy(result: object, *, selected_engine=None) -> GuiFileResult:
    status = str(getattr(result, "status", "failed") or "failed").strip().lower()
    if status == "solved":
        public_status = "SOLVED"
    elif status == "skipped":
        public_status = "SKIPPED"
    elif status == "wcs":
        public_status = "SOLVED"
    else:
        public_status = "FAILED"
    message = str(getattr(result, "message", "") or public_status)
    run_info = tuple(getattr(result, "run_info", ()) or ())
    return GuiFileResult(
        path=Path(getattr(result, "path", "")),
        status=public_status,
        message=message,
        backend=getattr(result, "metadata_source", None),
        inliers=int(getattr(result, "matched_stars", 0) or 0) or None,
        rms_px=None,
        pixel_scale_arcsec=getattr(result, "pixel_scale_arcsec", None),
        wcs_written=(status in {"solved", "wcs"}),
        errors=(() if status in {"solved", "skipped", "wcs"} else (message,)),
        run_info=run_info,
        selected_engine=selected_engine,
    )
