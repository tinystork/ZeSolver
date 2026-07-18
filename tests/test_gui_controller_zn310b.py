from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.engine_selection import EngineMode
from zesolver.gui_pipeline.controller import GuiSolveController
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


ROOT = Path(__file__).resolve().parents[1]


def _required_path(env_key: str) -> Path:
    raw = os.environ.get(env_key)
    if not raw:
        pytest.skip(f"{env_key} is not set")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"{env_key} path is absent: {path}")
    return path


@pytest.mark.corpus
@pytest.mark.slow
@pytest.mark.blind4d
def test_gui_controller_routes_zn310b_to_pipeline(tmp_path: Path) -> None:
    zn_root = _required_path("ZESOLVER_ZN310B_ROOT")
    astap_root = _required_path("ZESOLVER_ASTAP_ROOT")
    manifest_path = _required_path("ZESOLVER_BLIND4D_MANIFEST")
    legacy_root = Path(os.environ.get("ZESOLVER_LEGACY_INDEX_ROOT") or manifest_path.parent).expanduser()
    resources = resolve_catalog_resources(
        legacy_db_root=astap_root,
        legacy_families=("d50",),
        legacy_blind4d_manifest=manifest_path,
        legacy_index_root=legacy_root,
        enable_environment_discovery=False,
    )
    manifest = json.loads((ROOT / "reports" / "zenear_zn310b_gui_manifest.json").read_text(encoding="utf-8"))
    copied: list[Path] = []
    for item in manifest["items"]:
        src = zn_root / "gui_mixed" / str(item["gui_filename"])
        dst = tmp_path / str(item["gui_filename"])
        dst.write_bytes(src.read_bytes())
        copied.append(dst)

    request = build_gui_solve_request(
        copied,
        GuiSettingsState(
            engine_mode=EngineMode.AUTO,
            backend="local",
            workers=1,
            use_blind=True,
            use_web_fallback=False,
            catalog_resources=resources,
        ),
    )
    controller = GuiSolveController(
        pipeline_runner_factory=lambda: PipelineGuiRunner(),
        legacy_runner_factory=lambda: pytest.fail("legacy route should not be selected"),
    )
    summary = controller.run(request)
    assert summary.selected_engine is EngineMode.PIPELINE
    assert len(summary.results) == 8
    assert sum(1 for item in summary.results if item.status == "SOLVED") == 8
    assert [item.backend for item in summary.results[:3]] == ["NEAR", "NEAR", "NEAR"]
    assert [item.backend for item in summary.results[3:]] == ["BLIND4D"] * 5
