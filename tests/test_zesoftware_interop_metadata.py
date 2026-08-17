"""Anti-drift and wheel-inclusion tests for ZeSolver interop metadata.

``zesolver/zesoftware_interop.json`` is deployment-time metadata owned by
ZeSolver (see the ZeSoftware Interoperability Rules).  It is not product
behavior, so it must stay in lockstep with the public API source of truth in
``zesolver/api/v1/models.py`` / ``zesolver/api/v1/probe.py``.

These tests never import ZeAlfie and never inspect any sibling repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from importlib import resources
from pathlib import Path

import pytest

from zesolver.api.v1.models import API_VERSION, _SUPPORTED_CAPABILITIES
from zesolver.api.v1.probe import get_api_info

INTEROP_FILENAME = "zesoftware_interop.json"
INTEROP_WHEEL_PATH = f"zesolver/{INTEROP_FILENAME}"


def _load_interop_json() -> dict:
    text = (
        resources.files("zesolver")
        .joinpath(INTEROP_FILENAME)
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


@pytest.fixture(scope="module")
def interop() -> dict:
    return _load_interop_json()


@pytest.fixture(scope="module")
def provides(interop: dict) -> dict:
    entries = interop["provides"]
    assert len(entries) == 1
    return entries[0]


# ---------------------------------------------------------------------------
# Schema and identity
# ---------------------------------------------------------------------------


def test_schema_is_interop_v1(interop: dict) -> None:
    assert interop["schema"] == "zesoftware.interop.v1"


def test_product_identity(interop: dict) -> None:
    assert interop["product_id"] == "zesolver"
    assert interop["distribution_name"] == "ZeSolver"


def test_consumes_is_empty(interop: dict) -> None:
    assert interop["consumes"] == []


# ---------------------------------------------------------------------------
# Anti-drift against the public API source of truth
# ---------------------------------------------------------------------------


def test_api_module_matches_public_namespace(provides: dict) -> None:
    assert provides["api_module"] == "zesolver.api.v1"


def test_api_version_matches_models_constant(provides: dict) -> None:
    assert provides["api_version"] == API_VERSION


def test_capabilities_match_models_constant(provides: dict) -> None:
    # Same set, and order preserved from the constants tuple.
    assert provides["capabilities"] == list(_SUPPORTED_CAPABILITIES)
    assert set(provides["capabilities"]) == set(_SUPPORTED_CAPABILITIES)


def test_provider_facts_agree_with_get_api_info(provides: dict) -> None:
    info = get_api_info()
    assert provides["api_version"] == info.api_version
    assert list(provides["capabilities"]) == list(info.supported_capabilities)
    assert set(provides["capabilities"]) == set(info.supported_capabilities)


# ---------------------------------------------------------------------------
# Wheel inclusion / zip inspection (no product import from the wheel)
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pip_wheel_build_args() -> list[str]:
    # Avoid a network round-trip for build isolation when the build backend is
    # already importable; otherwise let pip provision it in isolation.
    try:
        import setuptools  # noqa: F401
        import wheel  # noqa: F401
    except ImportError:
        return []
    return ["--no-build-isolation"]


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    outdir = tmp_path_factory.mktemp("wheel")
    root = _project_root()
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(root),
        "--no-deps",
        *_pip_wheel_build_args(),
        "-w",
        str(outdir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            "wheel build failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    wheels = list(outdir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_wheel_contains_exactly_one_interop_json(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as zf:
        matches = [name for name in zf.namelist() if name.endswith(INTEROP_FILENAME)]
    assert matches == [INTEROP_WHEEL_PATH]


def test_wheel_interop_json_parseable_without_importing_product_code(
    wheel_path: Path,
) -> None:
    # Read straight from the wheel zip; no ``import zesolver`` from the wheel.
    with zipfile.ZipFile(wheel_path) as zf:
        raw = zf.read(INTEROP_WHEEL_PATH)
    data = json.loads(raw.decode("utf-8"))

    assert data["schema"] == "zesoftware.interop.v1"
    assert data["product_id"] == "zesolver"
    assert data["distribution_name"] == "ZeSolver"
    assert len(data["provides"]) == 1
    assert data["provides"][0]["api_module"] == "zesolver.api.v1"
    assert data["provides"][0]["api_version"] == API_VERSION
    assert set(data["provides"][0]["capabilities"]) == set(_SUPPORTED_CAPABILITIES)
    assert data["consumes"] == []


def test_wheel_install_then_api_import_passes(
    tmp_path: Path, wheel_path: Path
) -> None:
    """Install the built wheel standalone and import the public API.

    The wheel is installed with ``--no-deps --target`` into an isolated
    directory (dependencies come from the test environment) and the import
    runs from a foreign CWD with ``PYTHONPATH`` pinned to that target, so the
    imported ``zesolver`` provably comes from the wheel, not from the source
    checkout.
    """
    target = tmp_path / "site"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheel_path),
            "--no-deps",
            "--target",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, f"wheel install failed:\n{install.stderr}"

    script = (
        "import sys\n"
        "import zesolver\n"
        "assert zesolver.__file__.startswith(sys.argv[1]), zesolver.__file__\n"
        "from zesolver.api import v1\n"
        "assert v1.API_VERSION == '1.0'\n"
        "from zesolver.api.v1.probe import get_api_info\n"
        "info = get_api_info()\n"
        "assert info.api_version == '1.0'\n"
        "assert set(info.supported_capabilities) == "
        "{'near_solve', 'blind_solve', 'wcs_write', 'cancel', 'gpu'}\n"
        "print('wheel API import OK:', zesolver.__file__)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target)
    probe = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert probe.returncode == 0, f"wheel API import failed:\n{probe.stderr}"
    assert "wheel API import OK" in probe.stdout
