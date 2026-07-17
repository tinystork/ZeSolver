from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from corpus_loader import CorpusDataMissing, iter_cases


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zesolver_app_regression", ROOT / "zesolver.py")
assert SPEC is not None and SPEC.loader is not None
zesolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zesolver
SPEC.loader.exec_module(zesolver)


def _solver(tmp_path: Path):
    solver = object.__new__(zesolver.ImageSolver)
    solver.config = zesolver.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=("d50",),
        blind_index_path=tmp_path,
        workers=1,
        blind_enabled=True,
        near_defer_blind_fallback=False,
    )
    solver._cancel_event = None
    solver._near_seed = None
    solver._near_warmstart_parallel_notified = False
    return solver


def _metadata(path: Path):
    return zesolver.ImageMetadata(
        path=path,
        kind="fits",
        width=1080,
        height=1920,
        ra_deg=184.0,
        dec_deg=47.0,
        source="header",
        has_wcs=False,
    )


def test_pipeline_near_success_does_not_request_inline_blind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def fake_near_solve(*args, fallback_to_blind: bool, **kwargs):
        calls.append(fallback_to_blind)
        return {"success": True, "message": "near solution found"}

    monkeypatch.setattr(zesolver, "near_solve", fake_near_solve)
    path = tmp_path / "frame.fit"

    result = _solver(tmp_path)._run_index_near_solver(path, _metadata(path), allow_blind_fallback=True)

    assert result.status == "solved"
    assert calls == [False]


def test_pipeline_4d_profile_defers_failed_near_to_batch_blind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def fake_near_solve(*args, fallback_to_blind: bool, **kwargs):
        calls.append(fallback_to_blind)
        return {"success": False, "message": "near failed"}

    monkeypatch.setattr(zesolver, "near_solve", fake_near_solve)
    path = tmp_path / "frame.fit"

    result = _solver(tmp_path)._run_index_near_solver(path, _metadata(path), allow_blind_fallback=True)

    assert result is None
    assert calls == [False, False]


def test_pipeline_historical_profile_keeps_inline_blind_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def fake_near_solve(*args, fallback_to_blind: bool, **kwargs):
        calls.append(fallback_to_blind)
        return {"success": False, "message": "near failed"}

    monkeypatch.setattr(zesolver, "near_solve", fake_near_solve)
    solver = _solver(tmp_path)
    solver.config.blind_backend_profile = zesolver.HISTORICAL_PROFILE
    path = tmp_path / "frame.fit"

    result = solver._run_index_near_solver(path, _metadata(path), allow_blind_fallback=True)

    assert result is None
    assert calls == [False, False, True]


@pytest.mark.corpus
@pytest.mark.slow
@pytest.mark.blind4d
def test_pipeline_zn310b_cases_resolve_paths_or_skip_explicitly() -> None:
    for case in iter_cases(mode="pipeline"):
        try:
            path = case.resolve_path()
        except CorpusDataMissing as exc:
            pytest.skip(str(exc))
        assert path.exists()
