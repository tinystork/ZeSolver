"""Public data models for the ZeSolver v1 API.

These models are deliberately free of heavy runtime dependencies: no Qt, no
CuPy, no live Astropy WCS objects, no locks, no callbacks and no subprocess
handles.  They are intended to be pickle- and process-friendly so that they can
cross a process/adapter boundary when ZeMosaic uses a process-based adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from .errors import InvalidRequestError

# ---------------------------------------------------------------------------
# API version
# ---------------------------------------------------------------------------

API_VERSION = "1.0"
"""Single source of truth for the public API version."""


def _api_version_parts() -> tuple[int, int]:
    major_text, _, minor_text = API_VERSION.partition(".")
    try:
        return int(major_text), int(minor_text or "0")
    except ValueError:  # pragma: no cover - defensive only
        return 0, 0


# Derived exclusively from ``API_VERSION`` (never an independent constant).
API_MAJOR, API_MINOR = _api_version_parts()

_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "near_solve",
    "blind_solve",
    "wcs_write",
    "gpu",
    "cancel",
)


@dataclass(frozen=True, slots=True)
class ApiInfo:
    """Static API metadata.  Contains no runtime/dynamic state."""

    api_version: str
    product_version: str | None
    supported_capabilities: tuple[str, ...]


# ---------------------------------------------------------------------------
# Capability model
# ---------------------------------------------------------------------------


class CapabilityAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_CHECKED = "not_checked"


class CapabilityUnavailableReason(str, Enum):
    MISSING_RESOURCE = "missing_resource"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    POLICY_DISABLED = "policy_disabled"
    GPU_UNAVAILABLE = "gpu_unavailable"
    NETWORK_UNAVAILABLE = "network_unavailable"
    LICENSE_OR_AUTH_REQUIRED = "license_or_auth_required"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CapabilityState:
    """The negotiated state of a single named capability."""

    id: str
    supported: bool
    availability: CapabilityAvailability
    unavailable_reason: CapabilityUnavailableReason | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    """The result of :func:`zesolver.api.v1.probe`."""

    api_version: str
    product_version: str | None
    capabilities: tuple[CapabilityState, ...]
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class GpuPolicy(str, Enum):
    """GPU usage policy, owned by the *runtime* (never per-solve options)."""

    AUTO = "auto"
    DISABLED = "disabled"
    REQUIRED = "required"


class NetworkPolicy(str, Enum):
    """Network policy.  ``DISABLED`` is the default everywhere in v1."""

    DISABLED = "disabled"
    ALLOWED = "allowed"


class BackendPolicy(str, Enum):
    """Which solver backends may be used for a request."""

    AUTO = "auto"
    NEAR_ONLY = "near_only"
    BLIND_ONLY = "blind_only"


class WritePolicy(str, Enum):
    """How a solved WCS is written to disk.

    API 1.0 exposes only these two policies.  There is no ``NO_WRITE`` and no
    temp-copy simulation of one.
    """

    OVERWRITE_INPUT = "overwrite_input"
    WRITE_COPY = "write_copy"


# ---------------------------------------------------------------------------
# Solve hints / options / request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolveHints:
    """Optional plate-solving hints.  No generic ``metadata_overrides`` bag."""

    ra_deg: float | None = None
    dec_deg: float | None = None
    radius_deg: float | None = None
    pixel_scale_arcsec: float | None = None
    fov_deg: float | None = None
    focal_length_mm: float | None = None
    pixel_size_um: float | None = None

    def __post_init__(self) -> None:
        if (self.ra_deg is None) != (self.dec_deg is None):
            raise InvalidRequestError(
                "ra_deg and dec_deg must be provided together"
            )
        if self.ra_deg is not None:
            if not (0.0 <= float(self.ra_deg) < 360.0):
                raise InvalidRequestError("ra_deg must be in [0, 360)")
        if self.dec_deg is not None:
            if not (-90.0 <= float(self.dec_deg) <= 90.0):
                raise InvalidRequestError("dec_deg must be in [-90, 90]")
        if self.radius_deg is not None:
            if float(self.radius_deg) <= 0.0:
                raise InvalidRequestError("radius_deg must be > 0")
            if self.ra_deg is None:
                raise InvalidRequestError(
                    "radius_deg requires ra_deg/dec_deg (a center)"
                )
        for name in ("pixel_scale_arcsec", "fov_deg", "focal_length_mm", "pixel_size_um"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise InvalidRequestError(f"{name} must be > 0")


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Per-request solve options.  GPU policy is *not* here (it is runtime-scoped)."""

    backend_policy: BackendPolicy = BackendPolicy.AUTO
    network_policy: NetworkPolicy = NetworkPolicy.DISABLED
    write_policy: WritePolicy = WritePolicy.OVERWRITE_INPUT
    output_path: Path | None = None
    overwrite_existing_wcs: bool = False
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_s is not None and float(self.timeout_s) <= 0.0:
            raise InvalidRequestError("timeout_s must be > 0")
        if self.output_path is not None and not isinstance(self.output_path, Path):
            object.__setattr__(self, "output_path", Path(self.output_path))
        if self.write_policy is WritePolicy.WRITE_COPY:
            if self.output_path is None:
                raise InvalidRequestError(
                    "WRITE_COPY requires an output_path"
                )
        elif self.write_policy is WritePolicy.OVERWRITE_INPUT:
            if self.output_path is not None:
                raise InvalidRequestError(
                    "OVERWRITE_INPUT cannot be combined with output_path"
                )


@dataclass(frozen=True, slots=True)
class SolveRequest:
    """A single solve request."""

    input_path: Path
    hints: SolveHints = SolveHints()
    options: SolveOptions = SolveOptions()

    def __post_init__(self) -> None:
        input_path = Path(self.input_path)
        object.__setattr__(self, "input_path", input_path)
        if self.options.output_path is not None:
            output_path = self.options.output_path
            if _same_path(input_path, output_path):
                raise InvalidRequestError(
                    "output_path must differ from input_path"
                )


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.expanduser().resolve() == b.expanduser().resolve()
    except OSError:  # pragma: no cover - path may not exist yet
        return str(a.expanduser().absolute()) == str(b.expanduser().absolute())


# ---------------------------------------------------------------------------
# WCS transport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalWcsHeader:
    """A transport-safe WCS header.

    The primary public WCS transport is *not* a live :class:`astropy.wcs.WCS`.
    It is an ordered tuple of FITS header card strings that preserves order and
    repeatable cards (``COMMENT``/``HISTORY``) and reconstructs a FITS header.
    """

    format: Literal["fits-header-cards-v1"] = "fits-header-cards-v1"
    cards: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.format != "fits-header-cards-v1":
            raise InvalidRequestError(
                f"unsupported wcs header format: {self.format!r}"
            )
        object.__setattr__(self, "cards", tuple(str(c) for c in self.cards))

    @classmethod
    def from_fits_header(cls, header) -> "CanonicalWcsHeader":
        """Build from an Astropy ``Header`` (order- and repeat-preserving)."""
        cards: list[str] = []
        for card in header.cards:
            image = getattr(card, "image", None)
            cards.append(image if isinstance(image, str) and image else str(card))
        return cls(cards=tuple(cards))

    def to_fits_header(self):
        """Reconstruct an Astropy ``fits.Header``.  Requires Astropy."""
        from astropy.io import fits

        if not self.cards:
            return fits.Header()
        return fits.Header([fits.Card.fromstring(card) for card in self.cards])

    def to_astropy_wcs(self):
        """Reconstruct an Astropy ``WCS``.  Requires Astropy."""
        from astropy.wcs import WCS

        return WCS(self.to_fits_header(), relax=True)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class SolveStatus(str, Enum):
    SOLVED = "solved"
    SKIPPED_EXISTING_WCS = "skipped_existing_wcs"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureCode(str, Enum):
    """Stable, control-flow-safe failure categories.

    ``CANCELLED`` is deliberately *not* duplicated here (it is a
    :class:`SolveStatus`, not a failure code).
    """

    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_INPUT = "unsupported_input"
    EXISTING_WCS_INVALID = "existing_wcs_invalid"
    NO_SOLUTION = "no_solution"
    MISSING_RESOURCE = "missing_resource"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    POLICY_DISABLED = "policy_disabled"
    TIMEOUT = "timeout"
    WCS_INVALID = "wcs_invalid"
    WRITE_FAILED = "write_failed"


@dataclass(frozen=True, slots=True)
class SolveResult:
    """The outcome of a single solve.  Never carries live Astropy WCS."""

    status: SolveStatus
    input_path: Path
    output_path: Path | None
    wcs_header: CanonicalWcsHeader | None
    backend_used: str | None
    failure_code: FailureCode | None
    diagnostic_code: str | None
    message: str | None
    warnings: tuple[str, ...] = ()
    elapsed_s: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    pixel_scale_arcsec: float | None = None
    orientation_deg: float | None = None


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class ProgressPhase(str, Enum):
    PREPARING = "preparing"
    SOLVING = "solving"
    WRITING = "writing"
    FINALIZING = "finalizing"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A progress notification.  ``message`` is non-stable diagnostic text."""

    phase: ProgressPhase
    message: str | None = None
    current: int | None = None
    total: int | None = None
