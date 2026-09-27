from __future__ import annotations

from astropy.io import fits
from astropy.wcs import WCS

_IMAGE_HDU_TYPES = (fits.PrimaryHDU, fits.ImageHDU)


def is_wcs_probe_eligible_hdu(hdu) -> bool:
    """Structural eligibility for a 2D celestial WCS probe.

    An HDU is eligible only when it is an image HDU carrying at least two
    image axes (NAXIS >= 2).  An HDU with fewer than two axes cannot
    structurally represent a 2D image/WCS and must never be force-probed.
    Header-only: does not load pixel data.
    """
    if not isinstance(hdu, _IMAGE_HDU_TYPES):
        return False
    return int(hdu.header.get("NAXIS", 0) or 0) >= 2


def any_hdu_has_celestial_wcs(hdul) -> bool:
    """True when any structurally eligible HDU carries a celestial 2D WCS.

    Exceptions raised while probing an ELIGIBLE HDU are NOT swallowed: a
    genuinely corrupt 2D header must still surface as invalid input.
    """
    return any(
        bool(WCS(hdu.header, naxis=2, relax=True).has_celestial)
        for hdu in hdul
        if is_wcs_probe_eligible_hdu(hdu)
    )
