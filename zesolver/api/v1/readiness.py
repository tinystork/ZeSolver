"""Readiness and configuration access for the ZeSolver v1 public API.

:func:`readiness` reports whether the installed ZeSolver is actually
*operational* (at least one usable solve backend plus the intrinsic WCS write
capability) using the *same resource-discovery rules as the real runtime*, and
never leaks an internal catalog-resolution exception.

:func:`open_configuration` launches the public ZeSolver configuration GUI in a
detached, non-blocking subprocess.

Import-boundary rule: this module must stay lightweight.  The heavy catalog and
settings machinery is imported lazily inside the functions, never at module
import time, so ``import zesolver.api.v1`` remains cheap.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Mapping

from .errors import InvalidRequestError, ZeSolverApiError
from .models import (
    API_VERSION,
    CapabilityAvailability,
    CapabilityState,
    CapabilityUnavailableReason,
    ReadinessReport,
)
from .probe import _product_version

# Stable, consumer-facing readiness messages.  Deliberately free of internal
# exception text, catalog paths, and catalog details so consumers can rely on
# them for display and branching.
MSG_OPERATIONAL = "ZeSolver is operational"
MSG_CONFIGURATION_NEEDED = "ZeSolver catalog resources are not configured"
MSG_READINESS_UNKNOWN = "ZeSolver readiness could not be determined"


def readiness(
    *,
    settings_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 5.0,
) -> ReadinessReport:
    """Report whether ZeSolver is operational, using real runtime discovery rules.

    Discovery order (identical to the real runtime):

    1. If ``settings_path`` is provided it is used directly (tests/embedders);
       otherwise :func:`zesolver.settings_store.load_persistent_settings` is
       used, honoring the existing ``zesolver.SETTINGS_PATH`` override.
    2. The persisted catalog fields (``catalog_library_path``, ``db_root``,
       ``blind_4d_manifest_path``, ``index_root``) are mapped onto
       :func:`zesolver.catalog_resources.resolve_catalog_resources` exactly like
       the runtime's ``resolve_catalog_resources_for_config``.
    3. If nothing is configured, discovery falls back to the environment catalog
       hints (``enable_environment_discovery=True``), like the public v1 runtime
       with ``resources_path=None``.

    Internal resolution failures are never raised: any such failure yields a
    non-operational :class:`ReadinessReport` with a stable message.
    """
    if timeout_s <= 0:
        raise InvalidRequestError("timeout_s must be > 0")

    product_version = _product_version()

    try:
        settings = _load_settings(settings_path)
    except Exception:
        return _report(
            product_version=product_version,
            operational=False,
            configuration_needed=False,
            catalog_source="none",
            capabilities=_unavailable_capabilities(),
            message=MSG_READINESS_UNKNOWN,
        )

    from zesolver.catalog_resources import CatalogResourceResolutionError

    try:
        resources = _resolve_resources(settings, env=env)
    except CatalogResourceResolutionError:
        # Explicitly-configured catalog resources could not be used: a
        # configuration problem the consumer can fix via open_configuration().
        return _report(
            product_version=product_version,
            operational=False,
            configuration_needed=True,
            catalog_source="none",
            capabilities=_unavailable_capabilities(),
            message=MSG_CONFIGURATION_NEEDED,
        )
    except Exception:
        # Unexpected internal failure: non-operational, but *not* reported as a
        # simple configuration miss (the cause is unknown).
        return _report(
            product_version=product_version,
            operational=False,
            configuration_needed=False,
            catalog_source="none",
            capabilities=_unavailable_capabilities(),
            message=MSG_READINESS_UNKNOWN,
        )

    return _report_from_resources(resources, product_version=product_version)


def open_configuration() -> None:
    """Launch the public ZeSolver configuration GUI, non-blocking.

    Prefers the installed ``zesolver`` ``gui_scripts`` entry point (resolved
    through :mod:`importlib.metadata`); otherwise falls back to
    ``[sys.executable, "-c", "from zesolver._app import main; main()"]``.

    The subprocess is detached (new session), with stdout/stderr/stdin bound to
    ``DEVNULL``, and this function returns immediately without waiting.  A
    failed launch raises :class:`ZeSolverApiError` with a stable message.
    """
    cmd = _configuration_launcher_cmd()
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        raise ZeSolverApiError("unable to launch ZeSolver configuration") from exc


# ---------------------------------------------------------------------------
# Settings loading
# ---------------------------------------------------------------------------


def _load_settings(settings_path: Path | None):
    """Return the active :class:`~zesolver.settings_store.PersistentSettings`."""
    from zesolver.settings_store import load_persistent_settings

    if settings_path is None:
        return load_persistent_settings()

    # Explicit path (tests/embedders): honor it through the documented
    # ``zesolver.SETTINGS_PATH`` override for the duration of the load.
    import zesolver as pkg

    path = Path(settings_path)
    had_attr = "SETTINGS_PATH" in pkg.__dict__
    previous = pkg.__dict__.get("SETTINGS_PATH")
    pkg.SETTINGS_PATH = path
    try:
        return load_persistent_settings()
    finally:
        if had_attr:
            pkg.SETTINGS_PATH = previous
        else:
            pkg.__dict__.pop("SETTINGS_PATH", None)


# ---------------------------------------------------------------------------
# Resource discovery (same rules as the real runtime)
# ---------------------------------------------------------------------------


def _resolve_resources(settings, env: Mapping[str, str] | None):
    from zesolver.catalog_resources import resolve_catalog_resources

    catalog_library = getattr(settings, "catalog_library_path", None)
    db_root = getattr(settings, "db_root", None)
    families = tuple(getattr(settings, "db_family_cache", None) or ())
    blind4d_manifest = getattr(settings, "blind_4d_manifest_path", None)
    index_root = getattr(settings, "index_root", None)

    configured = any((catalog_library, db_root, blind4d_manifest, index_root))
    if configured:
        return resolve_catalog_resources(
            catalog_library=catalog_library,
            legacy_db_root=db_root,
            legacy_families=families,
            legacy_blind4d_manifest=blind4d_manifest,
            legacy_index_root=index_root,
            env=env,
            enable_environment_discovery=False,
        )
    return resolve_catalog_resources(
        env=env,
        enable_environment_discovery=True,
    )


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def _report_from_resources(resources, *, product_version: str | None) -> ReadinessReport:
    near_available = bool(getattr(resources, "near_available", False))
    near_root_missing = _legacy_near_root_missing(resources)
    if near_root_missing:
        near_available = False
    blind_available = bool(getattr(resources, "blind4d_available", False))

    capabilities = (
        _near_capability(
            near_available,
            detail="near catalog root does not exist" if near_root_missing else None,
        ),
        _blind_capability(blind_available),
        _wcs_write_capability(),
    )

    operational = near_available or blind_available
    source = getattr(resources, "source", None)

    return ReadinessReport(
        api_version=API_VERSION,
        product_version=product_version,
        operational=operational,
        configuration_needed=not operational,
        capabilities=capabilities,
        catalog_source=source,
        message=MSG_OPERATIONAL if operational else MSG_CONFIGURATION_NEEDED,
    )


def _legacy_near_root_missing(resources) -> bool:
    """Report whether a legacy near catalog points at a non-existent directory.

    ``resolve_catalog_resources`` builds a legacy
    :class:`~zesolver.catalog_library.models.NearCatalogDescriptor` from
    ``legacy_db_root`` without validating that the path exists (parity with the
    runtime), so a persisted ``db_root`` pointing at a missing directory would
    otherwise surface as an operational near backend even though acquisition
    would fail at solve time.  This is a *report-level* existence check only:
    discovery order and runtime behavior are untouched.
    """
    if getattr(resources, "source", None) != "legacy":
        return False
    near = getattr(resources, "near", None)
    if near is None:
        return False
    root = getattr(near, "root", None)
    if root is None:
        return False
    return not Path(root).expanduser().is_dir()


def _report(
    *,
    product_version: str | None,
    operational: bool,
    configuration_needed: bool,
    catalog_source: str | None,
    capabilities: tuple[CapabilityState, ...],
    message: str,
) -> ReadinessReport:
    return ReadinessReport(
        api_version=API_VERSION,
        product_version=product_version,
        operational=operational,
        configuration_needed=configuration_needed,
        capabilities=capabilities,
        catalog_source=catalog_source,
        message=message,
    )


def _near_capability(available: bool, *, detail: str | None = None) -> CapabilityState:
    if available:
        return CapabilityState("near_solve", True, CapabilityAvailability.AVAILABLE)
    return CapabilityState(
        "near_solve",
        True,
        CapabilityAvailability.UNAVAILABLE,
        CapabilityUnavailableReason.MISSING_RESOURCE,
        detail=detail if detail is not None else "near catalog resources are missing",
    )


def _blind_capability(available: bool) -> CapabilityState:
    if available:
        return CapabilityState("blind_solve", True, CapabilityAvailability.AVAILABLE)
    return CapabilityState(
        "blind_solve",
        True,
        CapabilityAvailability.UNAVAILABLE,
        CapabilityUnavailableReason.MISSING_RESOURCE,
        detail="blind 4D index resources are missing",
    )


def _wcs_write_capability() -> CapabilityState:
    return CapabilityState("wcs_write", True, CapabilityAvailability.AVAILABLE)


def _unavailable_capabilities() -> tuple[CapabilityState, ...]:
    return (
        _near_capability(False),
        _blind_capability(False),
        _wcs_write_capability(),
    )


# ---------------------------------------------------------------------------
# Configuration launcher
# ---------------------------------------------------------------------------


def _gui_entry_point():
    """Return the public ``zesolver`` gui_scripts entry point, if installed."""
    try:
        from importlib.metadata import entry_points
    except Exception:
        return None
    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            matches = list(eps.select(group="gui_scripts", name="zesolver"))
        else:  # pragma: no cover - Python < 3.10 compatibility
            matches = [
                ep
                for ep in eps.get("gui_scripts", ())
                if getattr(ep, "name", None) == "zesolver"
            ]
    except Exception:
        return None
    return matches[0] if matches else None


def _configuration_launcher_cmd() -> list[str]:
    entry = _gui_entry_point()
    if entry is not None:
        value = (getattr(entry, "value", None) or "").strip()
        module_name, sep, attr = value.partition(":")
        module_name = module_name.strip()
        attr = attr.strip()
        if sep and module_name and attr:
            code = f"from {module_name} import {attr}; {attr}()"
            return [sys.executable, "-c", code]
    return [sys.executable, "-c", "from zesolver._app import main; main()"]
