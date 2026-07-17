from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import SolveRequest


@dataclass(frozen=True, slots=True)
class BlindSolveRequest:
    input_path: Path
    output_path: Path | None
    overwrite_wcs: bool
    ra_hint_deg: float | None
    dec_hint_deg: float | None
    radius_hint_deg: float | None
    focal_length_mm: float | None
    pixel_size_um: float | None
    pixel_scale_arcsec: float | None
    pixel_scale_min_arcsec: float | None
    pixel_scale_max_arcsec: float | None
    profile_id: str
    request_id: str | None = None

    @classmethod
    def from_solve_request(cls, request: SolveRequest, *, configuration) -> "BlindSolveRequest":
        values = configuration.legacy_solve_config_values
        return cls(
            input_path=request.input_path,
            output_path=request.output_path,
            overwrite_wcs=request.overwrite_wcs,
            ra_hint_deg=_float_or_none(values.get("hint_ra_deg")),
            dec_hint_deg=_float_or_none(values.get("hint_dec_deg")),
            radius_hint_deg=_float_or_none(values.get("hint_radius_deg")),
            focal_length_mm=_float_or_none(values.get("hint_focal_mm")),
            pixel_size_um=_float_or_none(values.get("hint_pixel_um")),
            pixel_scale_arcsec=_float_or_none(values.get("hint_resolution_arcsec")),
            pixel_scale_min_arcsec=_float_or_none(values.get("hint_resolution_min_arcsec")),
            pixel_scale_max_arcsec=_float_or_none(values.get("hint_resolution_max_arcsec")),
            profile_id=configuration.blind_profile.profile_id,
            request_id=request.request_id,
        )


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
