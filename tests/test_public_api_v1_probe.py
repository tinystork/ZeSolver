"""Tests for the public ZeSolver API v1 probe / metadata surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from zesolver.api.v1 import (
    API_VERSION,
    ApiInfo,
    CapabilityAvailability,
    CapabilityUnavailableReason,
    InvalidRequestError,
    get_api_info,
    probe,
)


def test_get_api_info_is_static() -> None:
    info = get_api_info()
    assert isinstance(info, ApiInfo)
    assert info.api_version == API_VERSION
    assert info.supported_capabilities == (
        "near_solve",
        "blind_solve",
        "wcs_write",
        "gpu",
        "cancel",
    )
    # Static only: no dynamic capability state fields on ApiInfo.
    assert not hasattr(info, "capabilities")
    assert not hasattr(info, "warnings")


def test_probe_default_marks_expensive_checks_not_checked() -> None:
    result = probe(check_gpu=False, check_catalogs=False)
    assert result.api_version == API_VERSION
    by_id = {c.id: c for c in result.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.NOT_CHECKED
    assert by_id["blind_solve"].availability is CapabilityAvailability.NOT_CHECKED
    assert by_id["gpu"].availability is CapabilityAvailability.NOT_CHECKED
    # Cheap, always-known capabilities are reported as available.
    assert by_id["wcs_write"].availability is CapabilityAvailability.AVAILABLE
    assert by_id["cancel"].availability is CapabilityAvailability.AVAILABLE


def test_probe_rejects_non_positive_timeout() -> None:
    with pytest.raises(InvalidRequestError):
        probe(timeout_s=0.0)
    with pytest.raises(InvalidRequestError):
        probe(timeout_s=-1.0)


def test_probe_check_gpu_reports_checked_not_not_checked() -> None:
    result = probe(check_gpu=True, check_catalogs=False)
    gpu = next(c for c in result.capabilities if c.id == "gpu")
    assert gpu.availability is not CapabilityAvailability.NOT_CHECKED
    assert gpu.availability in (
        CapabilityAvailability.AVAILABLE,
        CapabilityAvailability.UNAVAILABLE,
    )


def test_probe_check_catalogs_with_missing_path_reports_unavailable() -> None:
    result = probe(
        check_catalogs=True,
        check_gpu=False,
        resources_path=Path("/nonexistent/catalog/library"),
    )
    by_id = {c.id: c for c in result.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.UNAVAILABLE
    assert by_id["blind_solve"].availability is CapabilityAvailability.UNAVAILABLE
    assert by_id["near_solve"].unavailable_reason is CapabilityUnavailableReason.MISSING_RESOURCE


def test_probe_check_catalogs_without_path_reports_not_checked() -> None:
    result = probe(check_catalogs=True, check_gpu=False, resources_path=None)
    by_id = {c.id: c for c in result.capabilities}
    assert by_id["near_solve"].availability is CapabilityAvailability.NOT_CHECKED
    assert by_id["blind_solve"].availability is CapabilityAvailability.NOT_CHECKED
