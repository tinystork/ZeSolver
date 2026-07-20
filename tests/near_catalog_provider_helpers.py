from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from zeblindsolver.projections import project_tan
from zewcs290.catalog290 import DEC_SCALE, RA_SCALE


def _encode_dec_bytes(dec_deg: float) -> tuple[int, int, int]:
    dec_i = int(round(float(dec_deg) / DEC_SCALE))
    if not (-(1 << 23) <= dec_i < (1 << 23)):
        raise ValueError(f"DEC outside ASTAP synthetic range: {dec_deg}")
    return dec_i & 0xFF, (dec_i >> 8) & 0xFF, (dec_i >> 16)


def write_astap_1476_tile(
    root: Path,
    *,
    family: str = "d50",
    tile_code: str = "1501",
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    mag: np.ndarray | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{family}_{tile_code}.1476"
    mag_arr = np.asarray(mag if mag is not None else np.full(len(ra_deg), 10.0), dtype=np.float32)
    header = bytearray(110)
    desc = b"SYNTHETIC_ASTAP_TILE"
    header[: len(desc)] = desc
    header[-1] = 5
    payload = bytearray()
    for ra_v, dec_v, mag_v in zip(np.asarray(ra_deg, dtype=np.float64), np.asarray(dec_deg, dtype=np.float64), mag_arr):
        dec0, dec1, dec9 = _encode_dec_bytes(float(dec_v))
        mag_byte = int(round(float(mag_v) * 10.0 + 16.0))
        payload += bytes([0xFF, 0xFF, 0xFF, (dec9 + 128) & 0xFF, mag_byte & 0xFF])
        ra_i = int(round((float(ra_v) % 360.0) / RA_SCALE))
        ra_i = max(0, min((1 << 24) - 1, ra_i))
        payload += bytes([ra_i & 0xFF, (ra_i >> 8) & 0xFF, (ra_i >> 16) & 0xFF, dec0, dec1])
    path.write_bytes(bytes(header) + bytes(payload))
    return path


def write_legacy_index_from_tile(
    index_root: Path,
    *,
    tile_key: str,
    family: str,
    tile_code: str,
    center_ra_deg: float,
    center_dec_deg: float,
    bounds: dict,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    mag: np.ndarray,
    db_root: Path | None = None,
) -> Path:
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tiles_dir / f"{tile_key}.npz"
    x_deg, y_deg = project_tan(ra_deg, dec_deg, center_ra_deg, center_dec_deg)
    np.savez(
        tile_path,
        ra_deg=np.asarray(ra_deg, dtype=np.float64),
        dec_deg=np.asarray(dec_deg, dtype=np.float64),
        mag=np.asarray(mag, dtype=np.float32),
        x_deg=np.asarray(x_deg, dtype=np.float32),
        y_deg=np.asarray(y_deg, dtype=np.float32),
    )
    manifest = {
        "db_root": str(db_root) if db_root is not None else "",
        "tiles": [
            {
                "tile_key": tile_key,
                "tile_file": f"tiles/{tile_path.name}",
                "family": family,
                "tile_code": tile_code,
                "center_ra_deg": center_ra_deg,
                "center_dec_deg": center_dec_deg,
                "bounds": bounds,
            }
        ],
    }
    (index_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return index_root


def make_synthetic_near_fits(path: Path, *, center_ra: float, center_dec: float, star_px: np.ndarray) -> None:
    image = np.zeros((200, 200), dtype=np.float32)
    for x, y in np.asarray(star_px, dtype=np.float64):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        image[max(0, yi - 1) : yi + 2, max(0, xi - 1) : xi + 2] = 5000.0
    header = fits.Header()
    header["RA"] = float(center_ra)
    header["DEC"] = float(center_dec)
    header["FOCALLEN"] = 150.0
    header["XPIXSZ"] = 3.76
    header["YPIXSZ"] = 3.76
    fits.PrimaryHDU(data=image, header=header).writeto(path)
