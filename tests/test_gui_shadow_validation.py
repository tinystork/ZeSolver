from __future__ import annotations

from pathlib import Path

from zesolver.engine_selection import EngineMode
from zesolver.gui_pipeline.requests import GuiFileResult, GuiRunSummary, GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request
from zesolver.gui_pipeline.shadow_validation import run_shadow_validation


def test_shadow_validation_uses_copies(tmp_path: Path) -> None:
    src = tmp_path / "a.fit"
    src.write_bytes(b"pixels")
    request = build_gui_solve_request([src], GuiSettingsState())

    def run(req):
        assert req.input_paths[0] != src
        return GuiRunSummary(
            EngineMode.PIPELINE,
            "ok",
            (GuiFileResult(req.input_paths[0], "SOLVED", "ok", backend="NEAR", wcs_written=True),),
            False,
            0.0,
        )

    comparison = run_shadow_validation(request, pipeline_run=run, legacy_run=run)
    assert comparison.ok
    assert src.read_bytes() == b"pixels"
