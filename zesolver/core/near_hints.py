"""Deterministic Near hint resolution (pure, no GUI, no I/O).

A single source of truth for turning three candidate hint sources — an explicit
user override, the acquisition FITS metadata, and the instrument preset — into
the effective Near hints and the source of each field.

Priority, without exception and independent of the GUI/API path::

    override > fits > preset > fallback/none

Used by both the GUI pipeline and the API path so they produce identical results
for identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

# Source labels (observability).  ``none`` = no hint resolved / not applied.
SOURCE_OVERRIDE = "override"
SOURCE_FITS = "fits"
SOURCE_PRESET = "preset"
SOURCE_NONE = "none"

_SOURCES = (SOURCE_OVERRIDE, SOURCE_FITS, SOURCE_PRESET, SOURCE_NONE)

# 206.265 arcsec/radian * 1e-3 (mm -> m) * 1e-6 (um -> m) == 206.265.
_ARCSEC_PER_RAD = 206.26480624709636


def preset_scale_from_focal_pixel(focal_mm: float, pixel_um: float, binning: float = 1.0) -> float:
    """Physical plate scale (arcsec/px) from focal length and pixel size.

    ``scale = 206.265 * pixel_um / focal_mm`` (the same relation used by the FITS
    ``estimate_scale_and_fov``), optionally multiplied by a binning factor (binned
    pixels are larger on the sky).
    """
    return float(_ARCSEC_PER_RAD * float(pixel_um) / float(focal_mm)) * float(binning)


@dataclass(frozen=True, slots=True)
class NearHintResolution:
    """Effective Near hints plus the source of each field."""

    center_ra_deg: float | None
    center_dec_deg: float | None
    scale_arcsec: float | None
    radius_deg: float | None
    fov_deg: float | None
    center_source: str
    scale_source: str
    radius_source: str
    fov_source: str

    def as_dict(self) -> dict:
        return {
            "center_ra_deg": self.center_ra_deg,
            "center_dec_deg": self.center_dec_deg,
            "scale_arcsec": self.scale_arcsec,
            "radius_deg": self.radius_deg,
            "fov_deg": self.fov_deg,
            "center_source": self.center_source,
            "scale_source": self.scale_source,
            "radius_source": self.radius_source,
            "fov_source": self.fov_source,
        }


def _pick(*, override, fits, preset) -> tuple[object, str]:
    """First non-None wins: override > fits > preset.  Returns (value, source)."""
    if override is not None:
        return override, SOURCE_OVERRIDE
    if fits is not None:
        return fits, SOURCE_FITS
    if preset is not None:
        return preset, SOURCE_PRESET
    return None, SOURCE_NONE


def resolve_near_hints(
    *,
    # explicit user overrides (highest priority)
    override_ra_deg: float | None = None,
    override_dec_deg: float | None = None,
    override_scale_arcsec: float | None = None,
    override_radius_deg: float | None = None,
    override_fov_deg: float | None = None,
    # acquisition FITS metadata (the real measurement)
    fits_ra_deg: float | None = None,
    fits_dec_deg: float | None = None,
    fits_scale_arcsec: float | None = None,
    # instrument preset (lowest non-fallback priority)
    preset_ra_deg: float | None = None,
    preset_dec_deg: float | None = None,
    preset_focal_mm: float | None = None,
    preset_pixel_um: float | None = None,
    preset_binning: float = 1.0,
    preset_radius_deg: float | None = None,
) -> NearHintResolution:
    """Resolve effective Near hints with a documented, deterministic priority.

    * **center**: the RA/DEC pair is resolved as a pair — the pair is used only if
      both components are present at the winning level.
    * **scale**: override (direct arcsec/px) > FITS scale > preset focal+px.
    * **radius**: an explicit radius override only; never derived from FITS or
      silently from the preset.
    * **fov**: explicit FOV override only; never applied silently from a preset.
    """
    # --- center (pair) -----------------------------------------------------
    center_ra = center_dec = None
    center_source = SOURCE_NONE
    if override_ra_deg is not None and override_dec_deg is not None:
        center_ra, center_dec, center_source = float(override_ra_deg), float(override_dec_deg), SOURCE_OVERRIDE
    elif fits_ra_deg is not None and fits_dec_deg is not None:
        center_ra, center_dec, center_source = float(fits_ra_deg), float(fits_dec_deg), SOURCE_FITS
    elif preset_ra_deg is not None and preset_dec_deg is not None:
        center_ra, center_dec, center_source = float(preset_ra_deg), float(preset_dec_deg), SOURCE_PRESET

    # --- scale -------------------------------------------------------------
    scale_arcsec = None
    scale_source = SOURCE_NONE
    if override_scale_arcsec is not None:
        scale_arcsec, scale_source = float(override_scale_arcsec), SOURCE_OVERRIDE
    elif fits_scale_arcsec is not None:
        scale_arcsec, scale_source = float(fits_scale_arcsec), SOURCE_FITS
    elif preset_focal_mm is not None and preset_pixel_um is not None:
        scale_arcsec = preset_scale_from_focal_pixel(preset_focal_mm, preset_pixel_um, preset_binning)
        scale_source = SOURCE_PRESET

    # --- radius (override only) -------------------------------------------
    radius_deg = None
    radius_source = SOURCE_NONE
    if override_radius_deg is not None:
        radius_deg = float(override_radius_deg)
        radius_source = SOURCE_OVERRIDE

    # --- fov (override only) ----------------------------------------------
    fov_deg = None
    fov_source = SOURCE_NONE
    if override_fov_deg is not None:
        fov_deg = float(override_fov_deg)
        fov_source = SOURCE_OVERRIDE

    return NearHintResolution(
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        scale_arcsec=scale_arcsec,
        radius_deg=radius_deg,
        fov_deg=fov_deg,
        center_source=center_source,
        scale_source=scale_source,
        radius_source=radius_source,
        fov_source=fov_source,
    )


def instrument_hint_applied(
    *,
    center_source: str,
    scale_source: str,
    radius_source: str,
    fov_source: str,
) -> bool:
    """True when at least one hint field was actually consumed from the preset.

    The GUI claim ``global_instrument_hint_applied`` is derived from this (real
    consumption) rather than from ``instrument_mode != "auto"``.
    """
    return any(s == SOURCE_PRESET for s in (center_source, scale_source, radius_source, fov_source))


__all__ = [
    "NearHintResolution",
    "resolve_near_hints",
    "preset_scale_from_focal_pixel",
    "instrument_hint_applied",
    "SOURCE_OVERRIDE",
    "SOURCE_FITS",
    "SOURCE_PRESET",
    "SOURCE_NONE",
]
