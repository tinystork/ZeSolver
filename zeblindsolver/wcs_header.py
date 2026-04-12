from __future__ import annotations

import math
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


_BASE_WCS_KEYS = {
    "WCSAXES",
    "CTYPE1", "CTYPE2",
    "CRVAL1", "CRVAL2",
    "CRPIX1", "CRPIX2",
    "CUNIT1", "CUNIT2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "CDELT1", "CDELT2",
    "CROTA1", "CROTA2",
    "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE",
}


def _is_sip_key(key: str) -> bool:
    k = str(key).upper()
    return (
        k in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}
        or k.startswith("A_")
        or k.startswith("B_")
        or k.startswith("AP_")
        or k.startswith("BP_")
    )


def sanitize_wcs_keywords(header: fits.Header, *, remove_sip: bool = True) -> int:
    removed = 0
    for key in list(header.keys()):
        ku = str(key).upper()
        if ku in _BASE_WCS_KEYS or (remove_sip and _is_sip_key(ku)):
            try:
                del header[key]
                removed += 1
            except Exception:
                pass
    return removed


def _extract_cd_matrix(wcs_obj: WCS) -> np.ndarray | None:
    try:
        cd_attr = getattr(wcs_obj.wcs, "cd", None)
    except Exception:
        cd_attr = None
    if cd_attr is not None:
        cd = np.asarray(cd_attr, dtype=np.float64)
        if cd.ndim == 2 and cd.shape[0] >= 2 and cd.shape[1] >= 2:
            return cd[:2, :2]
    try:
        pc_attr = getattr(wcs_obj.wcs, "pc", None)
        cdelt_attr = getattr(wcs_obj.wcs, "cdelt", None)
    except Exception:
        pc_attr = None
        cdelt_attr = None
    if pc_attr is not None and cdelt_attr is not None:
        pc = np.asarray(pc_attr, dtype=np.float64)
        cdelt = np.asarray(cdelt_attr, dtype=np.float64)
        if pc.ndim == 2 and pc.shape[0] >= 2 and pc.shape[1] >= 2 and cdelt.size >= 2:
            return pc[:2, :2] @ np.diag(cdelt[:2])
    return None


def apply_wcs_solution_to_header(
    header: fits.Header,
    wcs_obj: WCS,
    *,
    header_updates: dict[str, Any] | None = None,
    remove_sip_before_write: bool = True,
) -> None:
    """Write a canonical, ZeMosaic-compatible celestial WCS into *header*."""

    sanitize_wcs_keywords(header, remove_sip=remove_sip_before_write)

    wcs_header = wcs_obj.to_header(relax=True)
    for key, value in wcs_header.items():
        header[key] = value

    # Prefer explicit CD matrix to avoid ambiguity across readers.
    cd = _extract_cd_matrix(wcs_obj)
    if cd is not None and np.all(np.isfinite(cd)):
        header["CD1_1"] = float(cd[0, 0])
        header["CD1_2"] = float(cd[0, 1])
        header["CD2_1"] = float(cd[1, 0])
        header["CD2_2"] = float(cd[1, 1])
        for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2", "CROTA1", "CROTA2"):
            if k in header:
                try:
                    del header[k]
                except Exception:
                    pass

    # Keep key celestial metadata explicit.
    header["WCSAXES"] = 2
    if not str(header.get("RADESYS", "")).strip():
        header["RADESYS"] = "ICRS"

    for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
        if key in header:
            try:
                val = float(header[key])
                if math.isfinite(val):
                    header[key] = val
            except Exception:
                pass

    if header_updates:
        for key, value in header_updates.items():
            if value is not None:
                header[key] = value
