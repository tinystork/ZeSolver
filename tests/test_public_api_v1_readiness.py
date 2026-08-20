"""Tests for the public ZeSolver API v1 readiness / configuration surface.

These tests are a *consumer sentinel*: they import only from
``zesolver.api.v1`` (plus stdlib helpers), never the internal catalog/settings
modules, except where a test must inject a specific internal exception type to
prove it never leaks.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zesolver.api.v1 import (
    API_VERSION,
    CapabilityAvailability,
    CapabilityUnavailableReason,
    InvalidRequestError,
    ReadinessReport,
    ZeSolverApiError,
    readiness,
)

ROOT = Path(__file__).resolve().parents[1]

# The ``readiness`` name is exported as a *function* by ``zesolver.api.v1``,
# shadowing the submodule attribute of the same name.  Fetch the actual module
# object explicitly so tests can monkeypatch its private helpers.
readiness_module = importlib.import_module("zesolver.api.v1.readiness")


def _fake_resources(*, near: bool, blind: bool, source: str) -> SimpleNamespace:
    return SimpleNamespace(near_available=near, blind4d_available=blind, source=source)


# ---------------------------------------------------------------------------
# (1) Resource -> capability mapping (resolve path monkeypatched)
# ---------------------------------------------------------------------------


def test_mapping_near_and_blind_is_operational(monkeypatch) -> None:
    monkeypatch.setattr(readiness_module, "_load_settings", lambda _p: SimpleNamespace())
    monkeypatch.setattr(
        readiness_module,
        "_resolve_resources",
        lambda settings, env=None: _fake_resources(near=True, blind=True, source="library"),
    )
    report = readiness()
    assert isinstance(report, ReadinessReport)
    assert report.api_version == API_VERSION
    assert report.operational is True
    assert report.configuration_needed is False
    assert report.catalog_source == "library"
    by_id = {c.id: c for c in report.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.AVAILABLE
    assert by_id["blind_solve"].availability is CapabilityAvailability.AVAILABLE
    assert by_id["wcs_write"].availability is CapabilityAvailability.AVAILABLE


def test_mapping_near_only_is_operational(monkeypatch) -> None:
    monkeypatch.setattr(readiness_module, "_load_settings", lambda _p: SimpleNamespace())
    monkeypatch.setattr(
        readiness_module,
        "_resolve_resources",
        lambda settings, env=None: _fake_resources(near=True, blind=False, source="legacy"),
    )
    report = readiness()
    assert report.operational is True
    assert report.configuration_needed is False
    by_id = {c.id: c for c in report.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.AVAILABLE
    assert by_id["blind_solve"].availability is CapabilityAvailability.UNAVAILABLE
    assert by_id["blind_solve"].unavailable_reason is CapabilityUnavailableReason.MISSING_RESOURCE
    assert by_id["wcs_write"].availability is CapabilityAvailability.AVAILABLE


def test_mapping_none_is_not_operational_and_needs_configuration(monkeypatch) -> None:
    monkeypatch.setattr(readiness_module, "_load_settings", lambda _p: SimpleNamespace())
    monkeypatch.setattr(
        readiness_module,
        "_resolve_resources",
        lambda settings, env=None: _fake_resources(near=False, blind=False, source="none"),
    )
    report = readiness()
    assert report.operational is False
    assert report.configuration_needed is True
    assert report.catalog_source == "none"
    by_id = {c.id: c for c in report.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.UNAVAILABLE
    assert by_id["blind_solve"].availability is CapabilityAvailability.UNAVAILABLE
    assert by_id["wcs_write"].availability is CapabilityAvailability.AVAILABLE


# ---------------------------------------------------------------------------
# (2) Real persisted settings with a missing catalog library path
# ---------------------------------------------------------------------------


def test_settings_catalog_library_path_missing_is_not_operational(
    tmp_path: Path, monkeypatch
) -> None:
    import zesolver
    import zesolver.settings_store as store

    missing = tmp_path / "no_such_library"
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"catalog_library_path": str(missing)}), encoding="utf-8"
    )
    monkeypatch.setattr(zesolver, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)

    report = readiness()

    assert report.operational is False
    assert report.configuration_needed is True
    assert report.catalog_source == "none"
    assert report.message == "ZeSolver catalog resources are not configured"
    # Stable message: no internal exception type or path leaked.
    assert "CatalogResourceResolutionError" not in (report.message or "")
    assert str(missing) not in (report.message or "")
    by_id = {c.id: c for c in report.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.UNAVAILABLE
    assert by_id["near_solve"].unavailable_reason is CapabilityUnavailableReason.MISSING_RESOURCE


# ---------------------------------------------------------------------------
# (3) Environment discovery with missing hint paths
# ---------------------------------------------------------------------------


def test_environment_hints_missing_is_not_operational(tmp_path: Path, monkeypatch) -> None:
    import zesolver
    import zesolver.settings_store as store

    settings_file = tmp_path / "empty_settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(zesolver, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)

    env = {
        "ZESOLVER_ASTAP_ROOT": str(tmp_path / "no_such_astap"),
        "ZESOLVER_BLIND4D_MANIFEST": str(tmp_path / "no_such_manifest.json"),
    }
    report = readiness(env=env)

    assert report.operational is False
    assert report.configuration_needed is True
    assert report.catalog_source == "environment"


# ---------------------------------------------------------------------------
# (4) Resolution exceptions never leak
# ---------------------------------------------------------------------------


def test_unexpected_exception_yields_non_operational_without_leak(monkeypatch) -> None:
    def _boom(settings, env=None):
        raise RuntimeError("internal invariant failure")

    monkeypatch.setattr(readiness_module, "_load_settings", lambda _p: SimpleNamespace())
    monkeypatch.setattr(readiness_module, "_resolve_resources", _boom)

    report = readiness()

    assert isinstance(report, ReadinessReport)
    assert report.operational is False
    assert report.configuration_needed is False
    assert report.message == "ZeSolver readiness could not be determined"
    assert "internal invariant failure" not in (report.message or "")
    assert "RuntimeError" not in (report.message or "")


def test_catalog_resolution_error_is_a_configuration_miss(monkeypatch) -> None:
    from zesolver.catalog_resources import CatalogResourceResolutionError

    def _boom(settings, env=None):
        raise CatalogResourceResolutionError("catalog_library_invalid")

    monkeypatch.setattr(readiness_module, "_load_settings", lambda _p: SimpleNamespace())
    monkeypatch.setattr(readiness_module, "_resolve_resources", _boom)

    report = readiness()

    assert report.operational is False
    assert report.configuration_needed is True
    assert report.catalog_source == "none"
    assert report.message == "ZeSolver catalog resources are not configured"
    assert "catalog_library_invalid" not in (report.message or "")


def test_readiness_rejects_non_positive_timeout() -> None:
    with pytest.raises(InvalidRequestError):
        readiness(timeout_s=0.0)
    with pytest.raises(InvalidRequestError):
        readiness(timeout_s=-1.0)


# ---------------------------------------------------------------------------
# (5) open_configuration launches the public launcher non-blocking
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    value = "zesolver._app:main"


class _FakeProc:
    def wait(self):
        raise AssertionError("wait() must not be called (non-blocking)")

    def communicate(self):
        raise AssertionError("communicate() must not be called (non-blocking)")

    def poll(self):
        raise AssertionError("poll() must not be called (non-blocking)")


def _record_popen(monkeypatch):
    calls: dict = {}

    def fake_popen(args, **kwargs):
        calls["args"] = list(args)
        calls["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(readiness_module.subprocess, "Popen", fake_popen)
    return calls


def test_open_configuration_fallback_launcher(monkeypatch) -> None:
    monkeypatch.setattr(readiness_module, "_gui_entry_point", lambda: None)
    calls = _record_popen(monkeypatch)

    result = readiness_module.open_configuration()

    assert result is None
    assert calls["args"] == [sys.executable, "-c", "from zesolver._app import main; main()"]
    kw = calls["kwargs"]
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["stdout"] is subprocess.DEVNULL
    assert kw["stderr"] is subprocess.DEVNULL
    assert kw["start_new_session"] is True


def test_open_configuration_entry_point_launcher(monkeypatch) -> None:
    monkeypatch.setattr(readiness_module, "_gui_entry_point", lambda: _FakeEntryPoint())
    calls = _record_popen(monkeypatch)

    readiness_module.open_configuration()

    assert calls["args"] == [sys.executable, "-c", "from zesolver._app import main; main()"]


def test_open_configuration_raises_on_launch_failure(monkeypatch) -> None:
    def _boom(args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(readiness_module, "_gui_entry_point", lambda: None)
    monkeypatch.setattr(readiness_module.subprocess, "Popen", _boom)

    with pytest.raises(ZeSolverApiError, match="unable to launch ZeSolver configuration"):
        readiness_module.open_configuration()


# ---------------------------------------------------------------------------
# (6) Import boundary stays lightweight
# ---------------------------------------------------------------------------


def test_importing_readiness_stays_lightweight() -> None:
    script = r"""
import json
import sys

import zesolver.api.v1 as v1
import zesolver.api.v1.readiness

payload = {
    "api_version": v1.API_VERSION,
    "catalog_loaded": "zesolver.catalog_resources" in sys.modules,
    "settings_loaded": "zesolver.settings_store" in sys.modules,
    "engine_loaded": "zesolver.zeblindsolver" in sys.modules,
    "cupy_loaded": any(m == "cupy" or m.startswith("cupy.") for m in sys.modules),
    "qt_loaded": any(m.startswith("PySide6") or m.startswith("PyQt") for m in sys.modules),
}
print(json.dumps(payload, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["api_version"] == "1.1"
    assert payload["catalog_loaded"] is False
    assert payload["settings_loaded"] is False
    assert payload["engine_loaded"] is False
    assert payload["cupy_loaded"] is False
    assert payload["qt_loaded"] is False


# ---------------------------------------------------------------------------
# (3b) Legacy near catalog root existence check (report-level truth layer)
# ---------------------------------------------------------------------------


def test_settings_db_root_missing_is_not_operational(tmp_path: Path, monkeypatch) -> None:
    import zesolver
    import zesolver.settings_store as store

    missing = tmp_path / "no_such_db"
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"db_root": str(missing)}), encoding="utf-8")
    monkeypatch.setattr(zesolver, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)

    report = readiness()

    assert report.operational is False
    assert report.configuration_needed is True
    assert report.catalog_source == "legacy"
    assert report.message == "ZeSolver catalog resources are not configured"
    # Stable message: the configured path never leaks.
    assert str(missing) not in (report.message or "")
    by_id = {c.id: c for c in report.capabilities}
    near = by_id["near_solve"]
    assert near.availability is CapabilityAvailability.UNAVAILABLE
    assert near.unavailable_reason is CapabilityUnavailableReason.MISSING_RESOURCE
    assert near.detail == "near catalog root does not exist"


def test_settings_db_root_existing_is_operational(tmp_path: Path, monkeypatch) -> None:
    import zesolver
    import zesolver.settings_store as store

    existing = tmp_path / "db"
    existing.mkdir()
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"db_root": str(existing)}), encoding="utf-8")
    monkeypatch.setattr(zesolver, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)

    report = readiness()

    # Existence is the criterion (parity with runtime acquisition), not content.
    assert report.operational is True
    assert report.configuration_needed is False
    assert report.catalog_source == "legacy"
    by_id = {c.id: c for c in report.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.AVAILABLE


def test_mapping_legacy_near_root_missing_is_not_operational(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(readiness_module, "_load_settings", lambda _p: SimpleNamespace())
    missing = tmp_path / "no_such_near"
    fake = SimpleNamespace(
        near_available=True,
        blind4d_available=False,
        source="legacy",
        near=SimpleNamespace(root=missing),
    )
    monkeypatch.setattr(
        readiness_module,
        "_resolve_resources",
        lambda settings, env=None: fake,
    )

    report = readiness()

    assert report.operational is False
    assert report.configuration_needed is True
    assert report.catalog_source == "legacy"
    assert report.message == "ZeSolver catalog resources are not configured"
    by_id = {c.id: c for c in report.capabilities}
    near = by_id["near_solve"]
    assert near.availability is CapabilityAvailability.UNAVAILABLE
    assert near.unavailable_reason is CapabilityUnavailableReason.MISSING_RESOURCE
    assert near.detail == "near catalog root does not exist"
