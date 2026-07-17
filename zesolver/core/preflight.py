from __future__ import annotations

from dataclasses import dataclass

from astropy.io import fits
from astropy.wcs import WCS

from zesolver.catalog_resources import SolverCatalogResources

from .models import SolveRequest, SolveStatus


@dataclass(frozen=True, slots=True)
class PreflightResult:
    ok: bool
    status: SolveStatus | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    has_existing_wcs: bool = False
    image_shape: tuple[int, ...] | None = None
    image_dtype: str | None = None


def run_preflight(
    request: SolveRequest,
    *,
    catalog_resources: SolverCatalogResources | None = None,
    require_catalog: bool = True,
) -> PreflightResult:
    path = request.input_path
    if not path.exists():
        return PreflightResult(ok=False, status=SolveStatus.INVALID_INPUT, error=f"input_missing: {path}")
    if not path.is_file():
        return PreflightResult(ok=False, status=SolveStatus.INVALID_INPUT, error=f"input_not_file: {path}")

    try:
        with fits.open(path, memmap=False) as hdul:
            image_hdu = next((hdu for hdu in hdul if hdu.data is not None), None)
            if image_hdu is None:
                return PreflightResult(ok=False, status=SolveStatus.INVALID_INPUT, error="fits_image_missing")
            data = image_hdu.data
            shape = tuple(int(v) for v in data.shape)
            if len(shape) < 2 or shape[-1] <= 0 or shape[-2] <= 0:
                return PreflightResult(ok=False, status=SolveStatus.INVALID_INPUT, error=f"fits_invalid_dimensions: {shape}")
            has_wcs = any(bool(WCS(hdu.header, naxis=2, relax=True).has_celestial) for hdu in hdul)
    except Exception as exc:
        return PreflightResult(ok=False, status=SolveStatus.INVALID_INPUT, error=f"fits_unreadable: {exc}")

    if has_wcs and not request.overwrite_wcs:
        return PreflightResult(
            ok=False,
            status=SolveStatus.INVALID_INPUT,
            error="existing_wcs_overwrite_forbidden",
            has_existing_wcs=True,
            image_shape=shape,
            image_dtype=str(data.dtype),
        )

    warnings: list[str] = []
    if catalog_resources is not None:
        warnings.extend(catalog_resources.warnings)
        if require_catalog and not catalog_resources.near_available and not catalog_resources.blind4d_available:
            return PreflightResult(
                ok=False,
                status=SolveStatus.CATALOG_UNAVAILABLE,
                warnings=tuple(dict.fromkeys(warnings)),
                error="catalog_resources_unavailable",
                has_existing_wcs=has_wcs,
                image_shape=shape,
                image_dtype=str(data.dtype),
            )

    return PreflightResult(
        ok=True,
        warnings=tuple(dict.fromkeys(warnings)),
        has_existing_wcs=has_wcs,
        image_shape=shape,
        image_dtype=str(data.dtype),
    )
