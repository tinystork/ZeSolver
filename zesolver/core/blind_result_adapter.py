from __future__ import annotations

from pathlib import Path
from typing import Mapping

from astropy.io import fits
from astropy.wcs import WCS

from .models import EngineSolveResult, SolveStatus
from .result_adapter import pixel_scale_arcsec_from_wcs


def engine_result_from_blind_result(result: Mapping[str, object], *, solved_path: Path) -> EngineSolveResult:
    stats = result.get("stats")
    stats = stats if isinstance(stats, Mapping) else {}
    raw = dict(result)
    raw["stats"] = dict(stats)
    success = bool(result.get("success"))
    if not success:
        return EngineSolveResult(
            status=SolveStatus.UNSOLVED,
            backend="BLIND4D",
            warnings=_warnings_from_stats(stats),
            error=str(result.get("message") or "blind_failed"),
            raw=raw,
        )

    try:
        with fits.open(solved_path, memmap=False) as hdul:
            wcs_obj = WCS(hdul[0].header, naxis=2, relax=True)
    except Exception as exc:
        return EngineSolveResult(
            status=SolveStatus.FAILED,
            backend="BLIND4D",
            error=f"blind_wcs_read_failed: {exc}",
            raw=raw,
        )
    if not bool(getattr(wcs_obj, "has_celestial", getattr(wcs_obj, "is_celestial", False))):
        return EngineSolveResult(
            status=SolveStatus.FAILED,
            backend="BLIND4D",
            error="blind_wcs_missing_after_success",
            raw=raw,
        )

    validation = stats.get("astrometry_4d_best_accepted_validation")
    validation = validation if isinstance(validation, Mapping) else {}
    raw["header_updates"] = dict(result.get("updated_keywords") or {})
    return EngineSolveResult(
        status=SolveStatus.SOLVED,
        backend="BLIND4D",
        wcs=wcs_obj,
        wcs_written=False,
        center_ra_deg=_float_or_none(stats.get("center_ra_deg")),
        center_dec_deg=_float_or_none(stats.get("center_dec_deg")),
        pixel_scale_arcsec=_float_or_none(stats.get("pix_scale_arcsec")) or pixel_scale_arcsec_from_wcs(wcs_obj),
        inliers=_int_or_none(validation.get("inliers", stats.get("astrometry_4d_best_inliers"))),
        rms_px=_float_or_none(validation.get("rms_px", stats.get("astrometry_4d_best_rms_px"))),
        warnings=_warnings_from_stats(stats),
        raw=raw,
    )


def _warnings_from_stats(stats: Mapping[str, object]) -> tuple[str, ...]:
    warnings: list[str] = []
    if stats.get("blind4d_preflight_ok") is False:
        warnings.append("blind4d_preflight_failed")
    if stats.get("astrometry_4d_runtime_enabled") is False:
        warnings.append("blind4d_runtime_disabled")
    return tuple(warnings)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
