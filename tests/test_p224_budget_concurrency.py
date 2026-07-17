from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from zeblindsolver.index_manifest_4d import load_4d_index_manifest
from zeblindsolver.profiles import HISTORICAL_PROFILE, ZEBLIND_4D_EXPERIMENTAL_PROFILE, get_solver_profile
from zeblindsolver.zeblindsolver import SolveConfig, SolveStopReason, solve_blind

from test_p221_app_integration import _entry, _manifest, _write_fake_index


def _load_zesolver_entrypoint():
    path = Path(__file__).resolve().parents[1] / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_for_p224_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p224_4d_profile_maps_max_wall_to_route_budget(tmp_path: Path) -> None:
    idx = _write_fake_index(tmp_path / "a.npz", "d50_A")
    manifest = load_4d_index_manifest(_manifest(tmp_path / "manifest.json", [_entry("a", idx, "d50_A")]))

    cfg = get_solver_profile(ZEBLIND_4D_EXPERIMENTAL_PROFILE).apply_to_config(
        SolveConfig(),
        index_paths=manifest.enabled_index_paths,
    )

    assert cfg.blind_global_hard_budget_s == pytest.approx(0.0)
    assert cfg.blind_astrometry_4d_search_budget_s == pytest.approx(45.0)
    assert cfg.blind_astrometry_4d_max_hypotheses == 2000
    assert cfg.blind_astrometry_4d_max_accepts == 64


def test_p224_historical_profile_keeps_global_budget_semantics() -> None:
    cfg = SolveConfig(blind_global_hard_budget_s=12.5)
    out = get_solver_profile(HISTORICAL_PROFILE).apply_to_config(cfg)

    assert out.blind_global_hard_budget_s == pytest.approx(12.5)
    assert out.blind_astrometry_4d_search_budget_s == pytest.approx(0.0)


def test_p224_app_build_config_passes_route_budget(tmp_path: Path) -> None:
    pytest.importorskip("astroalign")
    zs = _load_zesolver_entrypoint()
    idx = _write_fake_index(tmp_path / "a.npz", "d50_A")
    loaded = load_4d_index_manifest(_manifest(tmp_path / "manifest.json", [_entry("a", idx, "d50_A")]))
    cfg = zs.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=None,
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        blind_4d_loaded_manifest=loaded,
    )

    blind_cfg = zs.build_blind_solve_config(cfg, loaded_manifest=loaded)

    assert blind_cfg.blind_global_hard_budget_s == pytest.approx(0.0)
    assert blind_cfg.blind_astrometry_4d_search_budget_s == pytest.approx(45.0)


def test_p224_4d_worker_policy_prefers_one_worker_and_respects_override(monkeypatch: pytest.MonkeyPatch) -> None:
    zs = _load_zesolver_entrypoint()
    monkeypatch.delenv("ZE_BLIND_WORKERS", raising=False)

    assert zs._auto_blind_worker_count(
        8,
        ram_gb=64.0,
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
    ) == 1
    assert zs._auto_blind_worker_count(8, ram_gb=64.0, blind_backend_profile=HISTORICAL_PROFILE) == 6

    monkeypatch.setenv("ZE_BLIND_WORKERS", "2")
    assert zs._auto_blind_worker_count(
        8,
        ram_gb=64.0,
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
    ) == 2


def test_p224_blind_attempt_budget_is_not_reported_as_user_cancelled(tmp_path: Path) -> None:
    fits_path = tmp_path / "scene.fits"
    fits.PrimaryHDU(data=np.zeros((16, 16), dtype=np.float32)).writeto(fits_path)

    result = solve_blind(
        str(fits_path),
        str(tmp_path),
        config=SolveConfig(blind_global_hard_budget_s=1.0e-9),
    )

    assert not result.success
    assert "blind attempt budget exceeded" in result.message
    assert result.stats["blind_attempt_budget_s"] == pytest.approx(1.0e-9)
    assert result.stats.get("astrometry_4d_stop_reason") != SolveStopReason.USER_CANCELLED.value


def test_p224_stop_reason_values_are_stable() -> None:
    assert SolveStopReason.USER_CANCELLED.value == "user_cancelled"
    assert SolveStopReason.BLIND_ATTEMPT_BUDGET_EXCEEDED.value == "blind_attempt_budget_exceeded"
    assert SolveStopReason.ASTROMETRY_4D_SEARCH_BUDGET_EXCEEDED.value == "astrometry_4d_search_budget_exceeded"
