from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from astropy.wcs import WCS


class SolveStatus(str, Enum):
    SOLVED = "SOLVED"
    UNSOLVED = "UNSOLVED"
    REJECTED_FALSE_SOLUTION = "REJECTED_FALSE_SOLUTION"
    INVALID_INPUT = "INVALID_INPUT"
    CATALOG_UNAVAILABLE = "CATALOG_UNAVAILABLE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class SolveRequest:
    input_path: Path
    output_path: Path | None
    overwrite_wcs: bool
    metadata_overrides: Mapping[str, object] = field(default_factory=dict)
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class EngineSolveResult:
    status: SolveStatus
    backend: str | None = None
    wcs: WCS | None = None
    wcs_written: bool = False
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None
    pixel_scale_arcsec: float | None = None
    orientation_deg: float | None = None
    parity: str | None = None
    inliers: int | None = None
    rms_px: float | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    raw: Mapping[str, object] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.status is SolveStatus.SOLVED


@dataclass(frozen=True, slots=True)
class SolveResult:
    request_id: str | None
    input_path: Path
    output_path: Path | None
    status: SolveStatus
    backend: str | None
    wcs_written: bool
    center_ra_deg: float | None
    center_dec_deg: float | None
    pixel_scale_arcsec: float | None
    orientation_deg: float | None
    parity: str | None
    inliers: int | None
    rms_px: float | None
    profile_ids: Mapping[str, str]
    catalog_status: str | None
    warnings: tuple[str, ...]
    error: str | None
