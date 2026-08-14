"""Contract tests for the v1 API's local-only network guarantee.

These tests pin the honest public contract introduced after removing the
previously-lying ``NetworkPolicy.ALLOWED`` member: API 1.0 is local-only, and
the v1 solve path must never inherit a persisted GUI web-fallback preference
(``use_web_fallback`` / ``astrometry_fallback_after_blind``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astropy.io import fits

from zesolver.api.v1 import (
    BackendPolicy,
    GpuPolicy,
    InvalidRequestError,
    NetworkPolicy,
    SolveOptions,
    SolveRequest,
    SolverRuntime,
    create_solver_runtime,
)
from zesolver.api.v1._adapters import _build_configuration, run_solve
from zesolver.core.models import EngineSolveResult, SolveStatus as InternalStatus


def _write_clean_fits(path: Path) -> Path:
    fits.writeto(path, np.zeros((64, 64), dtype=np.float32), overwrite=True)
    return path


# ---------------------------------------------------------------------------
# NetworkPolicy: API 1.0 is local-only
# ---------------------------------------------------------------------------


def test_network_policy_has_only_disabled() -> None:
    assert {m.value for m in NetworkPolicy} == {"disabled"}
    assert not hasattr(NetworkPolicy, "ALLOWED")


def test_network_policy_cannot_instantiate_allowed() -> None:
    with pytest.raises(ValueError):
        NetworkPolicy("allowed")


def test_solve_options_rejects_non_disabled_network_policy() -> None:
    with pytest.raises(InvalidRequestError):
        SolveOptions(network_policy="allowed")
    with pytest.raises(InvalidRequestError):
        SolveOptions(network_policy="disabled")
    with pytest.raises(InvalidRequestError):
        SolveOptions(network_policy=None)  # type: ignore[arg-type]


def test_solve_options_accepts_disabled_network_policy() -> None:
    assert SolveOptions().network_policy is NetworkPolicy.DISABLED
    assert (
        SolveOptions(network_policy=NetworkPolicy.DISABLED).network_policy
        is NetworkPolicy.DISABLED
    )


def test_runtime_rejects_non_disabled_network_policy() -> None:
    with pytest.raises(InvalidRequestError):
        SolverRuntime(network_policy="allowed")  # type: ignore[arg-type]
    with pytest.raises(InvalidRequestError):
        create_solver_runtime(network_policy="allowed")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Source of truth: the v1 solve path builds local-only ProductSettings
# ---------------------------------------------------------------------------


def test_v1_configuration_forces_web_fallback_off(tmp_path: Path) -> None:
    request = SolveRequest(tmp_path / "in.fits")
    cfg = _build_configuration(request, None, GpuPolicy.AUTO, NetworkPolicy.DISABLED, None)
    assert cfg.product_settings.web_fallback is False
    assert cfg.legacy_solve_config_values["astrometry_fallback_after_blind"] is False


def test_v1_ignores_persisted_gui_web_fallback(tmp_path: Path) -> None:
    from zesolver.gui_pipeline.requests import GuiSettingsState
    from zesolver.gui_pipeline.settings_adapter import build_product_settings

    # A persisted GUI state that WOULD enable web fallback in the legacy path.
    gui_state = GuiSettingsState(use_web_fallback=True, astrometry_api_key="dummy")
    assert build_product_settings(gui_state).web_fallback is True

    # The v1 API never reads that state: it builds its own local-only settings.
    cfg = _build_configuration(
        SolveRequest(tmp_path / "in.fits"),
        None,
        GpuPolicy.AUTO,
        NetworkPolicy.DISABLED,
        None,
    )
    assert cfg.product_settings.web_fallback is False
    assert cfg.legacy_solve_config_values["astrometry_fallback_after_blind"] is False


def test_v1_solve_passes_local_only_configuration_to_engine(tmp_path: Path) -> None:
    captured: dict = {}

    def near(internal_req, resources, configuration, shared_near, cancel_check):
        captured["configuration"] = configuration
        return EngineSolveResult(status=InternalStatus.UNSOLVED, backend="NEAR", error="no_solution")

    resources = SimpleNamespace(near_available=True, blind4d_available=False)
    run_solve(
        SolveRequest(
            _write_clean_fits(tmp_path / "in.fits"),
            options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY),
        ),
        resources=resources,
        near_shared=None,
        blind_selection=None,
        gpu_policy=GpuPolicy.AUTO,
        network_policy=NetworkPolicy.DISABLED,
        resources_path=None,
        near_solver=near,
        blind_solver=None,
        cancellation=None,
        progress=None,
        prep_cache={},
    )

    # The engine really was invoked, so it received the v1-built configuration.
    assert "configuration" in captured
    configuration = captured["configuration"]
    assert configuration.product_settings.web_fallback is False
    assert configuration.legacy_solve_config_values["astrometry_fallback_after_blind"] is False
