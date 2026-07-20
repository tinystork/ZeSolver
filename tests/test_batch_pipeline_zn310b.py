from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.core import SolveRequest, SolveStatus, SolverPipeline
from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.settings import ProductSettings, RuntimeOptions


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
def test_batch_pipeline_zn310b_routes_eight_cases(tmp_path: Path) -> None:
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
    requests = tuple(
        SolveRequest(
            zn_root / "gui_mixed" / str(item["gui_filename"]),
            tmp_path / str(item["gui_filename"]),
            True,
            request_id=str(item["case_id"]),
        )
        for item in manifest["items"]
    )

    def make_pipeline(phase: str) -> SolverPipeline:
        return SolverPipeline(
            product_settings=ProductSettings(
                blind_enabled=(phase == "blind"),
                blind_only=(phase == "blind"),
                web_fallback=False,
            ),
            runtime_options=RuntimeOptions(),
            catalog_resources=resources,
        )

    result = BatchSolverPipeline(solver_pipeline_factory=make_pipeline).solve(
        BatchSolveRequest(requests=requests, workers=1, preserve_order=True)
    )

    assert len(result.results) == 8
    assert result.progress.solved == 8
    assert result.progress.failed == 0
    assert result.progress.cancelled == 0
    assert [item.backend for item in result.results[:3]] == ["NEAR", "NEAR", "NEAR"]
    assert [item.backend for item in result.results[3:]] == ["BLIND4D"] * 5
    assert all(item.status is SolveStatus.SOLVED for item in result.results)
