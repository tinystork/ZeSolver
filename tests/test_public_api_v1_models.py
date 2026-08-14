"""Tests for the public ZeSolver API v1 data models and contract surface."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from astropy.io import fits
from astropy.wcs import WCS

from zesolver.api.v1 import (
    API_MAJOR,
    API_MINOR,
    API_VERSION,
    BackendPolicy,
    CanonicalWcsHeader,
    CapabilityAvailability,
    CapabilityUnavailableReason,
    FailureCode,
    GpuPolicy,
    InvalidRequestError,
    NetworkPolicy,
    ProgressPhase,
    SolveHints,
    SolveOptions,
    SolveRequest,
    SolveStatus,
    WritePolicy,
)


# ---------------------------------------------------------------------------
# Version contract
# ---------------------------------------------------------------------------


def test_api_version_is_single_source_of_truth() -> None:
    assert API_VERSION == "1.0"
    assert API_MAJOR == 1
    assert API_MINOR == 0
    assert f"{API_MAJOR}.{API_MINOR}" == API_VERSION


# ---------------------------------------------------------------------------
# Policy enums
# ---------------------------------------------------------------------------


def test_write_policy_has_only_two_members_and_no_no_write() -> None:
    assert {m.value for m in WritePolicy} == {"overwrite_input", "write_copy"}
    assert not hasattr(WritePolicy, "NO_WRITE")


def test_backend_policy_is_non_ambiguous() -> None:
    assert {m.value for m in BackendPolicy} == {"auto", "near_only", "blind_only"}


def test_network_policy_default_disabled() -> None:
    assert {m.value for m in NetworkPolicy} == {"disabled", "allowed"}
    assert SolveOptions().network_policy is NetworkPolicy.DISABLED


def test_gpu_policy_is_runtime_scoped_not_per_solve() -> None:
    # GPU policy must NOT live on the per-request SolveOptions.
    option_fields = {f.name for f in fields(SolveOptions)}
    assert "gpu_policy" not in option_fields
    assert {m.value for m in GpuPolicy} == {"auto", "disabled", "required"}


def test_solve_options_default_write_policy_is_overwrite_input() -> None:
    assert SolveOptions().write_policy is WritePolicy.OVERWRITE_INPUT


# ---------------------------------------------------------------------------
# Wire value contract (lowercase snake_case, stable for public interop)
# ---------------------------------------------------------------------------


def test_enum_values_are_lowercase_snake_case() -> None:
    # Member *names* stay uppercase; the str *values* are the stable wire form.
    assert {m.value for m in CapabilityAvailability} == {
        "available",
        "unavailable",
        "not_checked",
    }
    assert {m.value for m in CapabilityUnavailableReason} == {
        "missing_resource",
        "backend_unavailable",
        "policy_disabled",
        "gpu_unavailable",
        "network_unavailable",
        "license_or_auth_required",
        "unsupported_platform",
        "unknown",
    }
    assert {m.value for m in FailureCode} == {
        "invalid_input",
        "unsupported_input",
        "existing_wcs_invalid",
        "no_solution",
        "missing_resource",
        "backend_unavailable",
        "policy_disabled",
        "timeout",
        "wcs_invalid",
        "write_failed",
    }
    assert {m.value for m in ProgressPhase} == {
        "preparing",
        "solving",
        "writing",
        "finalizing",
    }


def test_enum_member_names_preserved_uppercase() -> None:
    assert {m.name for m in CapabilityAvailability} == {
        "AVAILABLE",
        "UNAVAILABLE",
        "NOT_CHECKED",
    }
    assert {m.name for m in GpuPolicy} == {"AUTO", "DISABLED", "REQUIRED"}
    assert {m.name for m in SolveStatus} == {
        "SOLVED",
        "SKIPPED_EXISTING_WCS",
        "FAILED",
        "CANCELLED",
    }


# ---------------------------------------------------------------------------
# SolveHints validation
# ---------------------------------------------------------------------------


def test_hints_require_ra_and_dec_together() -> None:
    with pytest.raises(InvalidRequestError):
        SolveHints(ra_deg=10.0)
    with pytest.raises(InvalidRequestError):
        SolveHints(dec_deg=10.0)
    assert SolveHints(ra_deg=10.0, dec_deg=20.0).ra_deg == 10.0


def test_hints_validate_ra_dec_ranges() -> None:
    with pytest.raises(InvalidRequestError):
        SolveHints(ra_deg=-1.0, dec_deg=0.0)
    with pytest.raises(InvalidRequestError):
        SolveHints(ra_deg=360.0, dec_deg=0.0)
    with pytest.raises(InvalidRequestError):
        SolveHints(ra_deg=0.0, dec_deg=-91.0)
    with pytest.raises(InvalidRequestError):
        SolveHints(ra_deg=0.0, dec_deg=91.0)


def test_hints_radius_requires_center_and_positive() -> None:
    with pytest.raises(InvalidRequestError):
        SolveHints(radius_deg=1.0)  # no center
    with pytest.raises(InvalidRequestError):
        SolveHints(ra_deg=0.0, dec_deg=0.0, radius_deg=0.0)
    assert SolveHints(ra_deg=0.0, dec_deg=0.0, radius_deg=1.0).radius_deg == 1.0


@pytest.mark.parametrize(
    "name", ["pixel_scale_arcsec", "fov_deg", "focal_length_mm", "pixel_size_um"]
)
def test_hints_positive_only(name: str) -> None:
    with pytest.raises(InvalidRequestError):
        SolveHints(**{name: 0.0})


# ---------------------------------------------------------------------------
# SolveOptions validation
# ---------------------------------------------------------------------------


def test_options_timeout_must_be_positive() -> None:
    with pytest.raises(InvalidRequestError):
        SolveOptions(timeout_s=0.0)
    with pytest.raises(InvalidRequestError):
        SolveOptions(timeout_s=-1.0)


def test_write_copy_requires_output_path() -> None:
    with pytest.raises(InvalidRequestError):
        SolveOptions(write_policy=WritePolicy.WRITE_COPY)


def test_overwrite_input_cannot_have_output_path() -> None:
    with pytest.raises(InvalidRequestError):
        SolveOptions(
            write_policy=WritePolicy.OVERWRITE_INPUT,
            output_path=Path("/tmp/out.fits"),
        )


def test_write_copy_accepts_output_path() -> None:
    opts = SolveOptions(write_policy=WritePolicy.WRITE_COPY, output_path="/tmp/out.fits")
    assert opts.output_path == Path("/tmp/out.fits")


# ---------------------------------------------------------------------------
# SolveRequest validation
# ---------------------------------------------------------------------------


def test_request_output_must_differ_from_input(tmp_path: Path) -> None:
    p = tmp_path / "a.fits"
    with pytest.raises(InvalidRequestError):
        SolveRequest(
            p,
            options=SolveOptions(
                write_policy=WritePolicy.WRITE_COPY, output_path=p
            ),
        )


def test_request_coerces_input_to_path(tmp_path: Path) -> None:
    req = SolveRequest(str(tmp_path / "a.fits"))
    assert isinstance(req.input_path, Path)


# ---------------------------------------------------------------------------
# Canonical WCS transport
# ---------------------------------------------------------------------------


def _full_cd_header() -> fits.Header:
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 100.0
    hdr["CRVAL2"] = 30.0
    hdr["CRPIX1"] = 50.5
    hdr["CRPIX2"] = 50.5
    hdr["CD1_1"] = -0.0003
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 0.0003
    hdr["COMMENT"] = "first comment"
    hdr["HISTORY"] = "a history line"
    return hdr


def test_canonical_wcs_header_is_serialized_cards_not_live_wcs() -> None:
    # The dataclass carries only a format tag and card strings — never a WCS.
    canonical_fields = {f.name for f in fields(CanonicalWcsHeader)}
    assert canonical_fields == {"format", "cards"}
    assert CanonicalWcsHeader().format == "fits-header-cards-v1"


def test_canonical_wcs_header_round_trip_preserves_order_and_repeats() -> None:
    source = _full_cd_header()
    canonical = CanonicalWcsHeader.from_fits_header(source)
    # order- and repeat-preserving: COMMENT/HISTORY cards must survive.
    joined = "\n".join(canonical.cards)
    assert "first comment" in joined
    assert "a history line" in joined
    rebuilt = canonical.to_fits_header()
    assert rebuilt["CRVAL1"] == 100.0
    assert rebuilt["CRVAL2"] == 30.0
    # Reconstructed WCS must be usable and have a celestial component.
    w = canonical.to_astropy_wcs()
    assert w.has_celestial


def test_canonical_wcs_header_rejects_unknown_format() -> None:
    with pytest.raises(InvalidRequestError):
        CanonicalWcsHeader(format="bogus")  # type: ignore[arg-type]


def test_canonical_wcs_header_accepts_passthrough_wcs(tmp_path: Path) -> None:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [1.0, 2.0]
    w.wcs.crpix = [10.0, 10.0]
    w.wcs.cd = [[-0.0003, 0.0], [0.0, 0.0003]]
    canonical = CanonicalWcsHeader.from_fits_header(w.to_header())
    assert canonical.cards
    assert canonical.to_astropy_wcs().has_celestial


# ---------------------------------------------------------------------------
# SolveStatus / FailureCode
# ---------------------------------------------------------------------------


def test_solve_status_members() -> None:
    assert {m.value for m in SolveStatus} == {
        "solved",
        "skipped_existing_wcs",
        "failed",
        "cancelled",
    }


def test_failure_code_does_not_duplicate_cancelled() -> None:
    assert "cancelled" not in {m.value for m in FailureCode}
