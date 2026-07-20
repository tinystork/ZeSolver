from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .requests import GuiRunSummary, GuiSolveRequest


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    pipeline: GuiRunSummary
    legacy: GuiRunSummary
    status_match: bool
    backend_match: bool
    wcs_written_match: bool
    result_count_match: bool

    @property
    def ok(self) -> bool:
        return self.status_match and self.backend_match and self.wcs_written_match and self.result_count_match


def run_shadow_validation(
    request: GuiSolveRequest,
    *,
    pipeline_run: Callable[[GuiSolveRequest], GuiRunSummary],
    legacy_run: Callable[[GuiSolveRequest], GuiRunSummary],
) -> ShadowComparison:
    with tempfile.TemporaryDirectory(prefix="zesolver_gui_shadow_") as tmp:
        root = Path(tmp)
        pipeline_paths = _copy_inputs(request.input_paths, root / "pipeline")
        legacy_paths = _copy_inputs(request.input_paths, root / "legacy")
        pipeline_summary = pipeline_run(_replace_paths(request, pipeline_paths))
        legacy_summary = legacy_run(_replace_paths(request, legacy_paths))
    return ShadowComparison(
        pipeline=pipeline_summary,
        legacy=legacy_summary,
        status_match=tuple(r.status for r in pipeline_summary.results) == tuple(r.status for r in legacy_summary.results),
        backend_match=tuple(r.backend for r in pipeline_summary.results) == tuple(r.backend for r in legacy_summary.results),
        wcs_written_match=tuple(r.wcs_written for r in pipeline_summary.results) == tuple(r.wcs_written for r in legacy_summary.results),
        result_count_match=len(pipeline_summary.results) == len(legacy_summary.results),
    )


def _copy_inputs(paths: tuple[Path, ...], root: Path) -> tuple[Path, ...]:
    root.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for idx, path in enumerate(paths):
        dst = root / f"{idx}_{Path(path).name}"
        shutil.copy2(path, dst)
        copied.append(dst)
    return tuple(copied)


def _replace_paths(request: GuiSolveRequest, paths: tuple[Path, ...]) -> GuiSolveRequest:
    from dataclasses import replace

    return replace(request, input_paths=paths)
