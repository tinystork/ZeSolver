from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.wcs_header import apply_wcs_solution_to_header


@dataclass(frozen=True, slots=True)
class WcsWriteResult:
    ok: bool
    path: Path | None
    wcs_written: bool
    pixels_unchanged: bool
    error: str | None = None


def pixel_fingerprint(path: Path) -> str:
    with fits.open(path, memmap=False) as hdul:
        h = hashlib.sha256()
        for hdu in hdul:
            if hdu.data is None:
                continue
            arr = np.ascontiguousarray(hdu.data)
            h.update(str(arr.dtype).encode("ascii"))
            h.update(str(tuple(arr.shape)).encode("ascii"))
            h.update(arr.tobytes())
        return h.hexdigest()


def write_wcs_safely(
    *,
    input_path: Path,
    output_path: Path | None,
    wcs: WCS,
    overwrite_wcs: bool,
    header_updates: dict[str, object] | None = None,
    verify_pixels: bool = True,
) -> WcsWriteResult:
    target = output_path or input_path
    try:
        if target != input_path:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, target)
        before = pixel_fingerprint(target) if verify_pixels else ""
        with fits.open(target, mode="update", memmap=False) as hdul:
            if not overwrite_wcs and any(bool(WCS(hdu.header, naxis=2, relax=True).has_celestial) for hdu in hdul):
                return WcsWriteResult(False, target, False, True, "existing_wcs_overwrite_forbidden")
            apply_wcs_solution_to_header(hdul[0].header, wcs, header_updates=header_updates or {"SOLVED": 1, "SOLVER": "ZeSolver"})
            hdul.flush(output_verify="exception")
        after = pixel_fingerprint(target) if verify_pixels else before
        if verify_pixels and before != after:
            return WcsWriteResult(False, target, True, False, "pixels_changed")
        with fits.open(target, memmap=False) as hdul:
            if not bool(WCS(hdul[0].header, naxis=2, relax=True).has_celestial):
                return WcsWriteResult(False, target, True, True, "written_wcs_invalid")
        return WcsWriteResult(True, target, True, True)
    except Exception as exc:
        return WcsWriteResult(False, target, False, True, str(exc))
