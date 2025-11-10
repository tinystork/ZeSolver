"""Blind ASTAP-based WCS solver used by ZeSolver."""
from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.io import fits

try:  # pragma: no cover - fallback for older Python versions
    from importlib.metadata import PackageNotFoundError, version as pkg_version
except ImportError:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore[misc]
    def pkg_version(_: str) -> str:  # type: ignore[override]
        return "0.0.dev"

try:  # pragma: no cover - when running from source tree
    __version__ = pkg_version("zewcs290")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.dev"

ZEBLIND_VERSION = __version__
# Default ASTAP database order (can be overridden by callers)
DEFAULT_DB_SEQUENCE = ["D80", "D50", "V50", "D20", "D05", "G05", "W08", "H18"]
DEFAULT_RADIUS_TOLERANCE = 200
PROFILE_PRESETS: dict[str, dict[str, Optional[float] | tuple[Optional[float], Optional[float]]]] = {
    "S50": {"scale": 2.39, "fov": (0.72, 1.28)},
    "S30": {"scale": 1.55, "fov": (0.54, 0.92)},
    "CUSTOM": {"scale": None, "fov": (None, None)},
}


class BlindSolveResult(TypedDict):
    success: bool
    message: str
    elapsed_sec: float
    tried_dbs: List[str]
    used_db: Optional[str]
    wrote_wcs: bool
    updated_keywords: dict[str, Any]
    output_path: str


class BlindSolverRuntimeError(RuntimeError):
    exit_code = 2


class InvalidInputError(BlindSolverRuntimeError):
    exit_code = 2


class AstapNotFoundError(BlindSolverRuntimeError):
    exit_code = 3


class AstapExecutionError(BlindSolverRuntimeError):
    exit_code = 4


def _default_log(message: str) -> None:
    print(message, flush=True)


def _normalize_db_chain(db_roots: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for chunk in db_roots:
        text = str(chunk).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            normalized.append(str(path))
        else:
            normalized.append(text)
    return normalized


def _resolve_astap_executable(explicit: Optional[str]) -> str:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return str(candidate)
        if candidate.suffix == "" and candidate.parent.exists():
            maybe = candidate.with_suffix(".exe")
            if maybe.is_file():
                return str(maybe)
        raise AstapNotFoundError(f"ASTAP executable not found at {candidate}")
    exe_name = "astap.exe" if os.name == "nt" else "astap"
    discovered = shutil.which(exe_name)
    if discovered:
        return discovered
    raise AstapNotFoundError("ASTAP executable not found in PATH")


def _parse_header_angle(value: object, *, is_ra: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        number = float(value)
        return number
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        try:
            angle = Angle(text, unit=u.hourangle if is_ra else u.deg)
            return float(angle.degree)
        except (ValueError, u.UnitsError):
            return None


def _ra_deg_to_hours(value: float) -> Optional[float]:
    try:
        hours = (float(value) % 360.0) / 15.0
    except (TypeError, ValueError):
        return None
    if math.isnan(hours):
        return None
    return hours


def _dec_deg_to_spd(value: float) -> Optional[float]:
    try:
        spd = 90.0 + float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(spd):
        return None
    return min(max(spd, 0.0), 180.0)


def has_valid_wcs(header: fits.Header) -> bool:
    required = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CTYPE1", "CTYPE2")
    if any(key not in header for key in required):
        return False
    radesys = str(header.get("RADESYS", "")).strip()
    if not radesys:
        return False
    ctype1 = str(header.get("CTYPE1", "")).upper()
    ctype2 = str(header.get("CTYPE2", "")).upper()
    if not any(token in ctype1 for token in ("RA", "GLON")):
        return False
    if not any(token in ctype2 for token in ("DEC", "GLAT")):
        return False
    cd_keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if any(key not in header for key in cd_keys):
        return False
    try:
        cd = np.array(
            [
                [float(header["CD1_1"]), float(header["CD1_2"])],
                [float(header["CD2_1"]), float(header["CD2_2"])],
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cd)):
        return False
    if abs(np.linalg.det(cd)) < 1e-12:
        return False
    for key in ("CDELT1", "CDELT2"):
        if key in header:
            try:
                if math.isclose(float(header[key]), 1.0, rel_tol=0, abs_tol=1e-9):
                    return False
            except (TypeError, ValueError):
                return False
    for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def sanitize_wcs(header: fits.Header) -> int:
    removed = 0
    keys = [
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CDELT1",
        "CDELT2",
        "CROTA1",
        "CROTA2",
        "RADESYS",
        "EQUINOX",
        "LONPOLE",
        "LATPOLE",
    ]
    for key in keys:
        if key in header:
            del header[key]
            removed += 1
    return removed


def estimate_scale_and_fov(
    header: fits.Header,
    width: int,
    height: int,
) -> tuple[Optional[float], tuple[Optional[float], Optional[float]]]:
    focal_len = header.get("FOCALLEN") or header.get("FOCLEN") or header.get("FOCALLENGTH")
    if focal_len is None:
        return None, (None, None)
    try:
        focal_len = float(focal_len)
    except (TypeError, ValueError):
        return None, (None, None)
    if focal_len <= 0:
        return None, (None, None)
    pix_x = header.get("XPIXSZ") or header.get("PIXSIZE1")
    pix_y = header.get("YPIXSZ") or header.get("PIXSIZE2")
    if pix_x is None or pix_y is None:
        return None, (None, None)
    try:
        pix_um = float((float(pix_x) + float(pix_y)) / 2.0)
    except (TypeError, ValueError):
        return None, (None, None)
    if pix_um <= 0:
        return None, (None, None)
    scale = 206.265 * pix_um / float(focal_len)
    fov_x = scale * width / 3600.0
    fov_y = scale * height / 3600.0
    return float(scale), (float(fov_x), float(fov_y))


def _extract_bayer_green(data: np.ndarray, pattern: str) -> np.ndarray:
    pattern = (pattern or "").strip().upper()
    if len(pattern) != 4 or data.ndim != 2:
        return data
    mask = np.zeros_like(data, dtype=bool)
    mapping = {(0, 0): pattern[0], (0, 1): pattern[1], (1, 0): pattern[2], (1, 1): pattern[3]}
    for (dy, dx), channel in mapping.items():
        if channel == "G":
            mask[dy::2, dx::2] = True
    if not mask.any():
        return data
    result = np.array(data, copy=True)
    green_values = result[mask]
    if green_values.size == 0:
        return data
    mean_green = float(np.mean(green_values))
    if math.isfinite(mean_green):
        result[~mask] = mean_green
    return result


def to_luminance_for_solve(hdu: fits.PrimaryHDU) -> np.ndarray:
    data = hdu.data
    if data is None:
        raise InvalidInputError("FITS HDU has no data to generate luminance")
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=0)
    elif arr.ndim == 2:
        bayer = hdu.header.get("BAYERPAT") or hdu.header.get("BAYERP")
        arr = _extract_bayer_green(np.asarray(arr), bayer)
    else:
        raise InvalidInputError(f"Unsupported FITS data shape for luminance: {arr.shape}")
    arr = np.asarray(arr, dtype=np.float32)
    try:
        min_val = float(np.nanmin(arr))
    except (ValueError, FloatingPointError):
        min_val = 0.0
    if math.isfinite(min_val):
        arr -= min_val
    arr = np.nan_to_num(arr, copy=False)
    high = float(np.nanpercentile(arr, 99.5)) if arr.size else 1.0
    if not math.isfinite(high) or high <= 0:
        high = 1.0
    arr = np.clip(arr / high, 0.0, 1.0)
    return arr.astype(np.float32, copy=False)


def _infer_profile(header: fits.Header) -> Optional[str]:
    for key in ("INSTRUME", "TELESCOP", "OBSERVER", "CREATOR"):
        text = str(header.get(key, "")).upper()
        if "S50" in text:
            return "S50"
        if "S30" in text:
            return "S30"
    return None


def _format_db_label(value: str) -> str:
    path = Path(value)
    if path.name:
        return path.name
    return value


def blind_solve(
    fits_path: str,
    db_roots: Sequence[str],
    astap_exe: Optional[str] = None,
    profile: Optional[str] = None,
    timeout_sec: int = 90,
    skip_if_valid: bool = True,
    ra_hint: Optional[float] = None,
    dec_hint: Optional[float] = None,
    write_to: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    log: Optional[Callable[[str], None]] = None,
    verbose: bool = False,
) -> BlindSolveResult:
    logger = log or _default_log
    start = time.perf_counter()
    input_path = Path(fits_path).expanduser()
    if not input_path.is_file():
        raise InvalidInputError(f"Input FITS not found: {input_path}")
    db_chain = _normalize_db_chain(db_roots)
    if not db_chain:
        raise InvalidInputError("At least one ASTAP database path is required")
    work_path = Path(write_to).expanduser() if write_to else input_path
    if write_to:
        work_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, work_path)
    logger("[ZEBLIND] starting")
    logger(f"[ZEBLIND] db_chain={'/'.join(_format_db_label(db) for db in db_chain)}")
    if profile:
        logger(f"[ZEBLIND] profile={profile}")
    astap_path: Optional[str] = None
    tried_dbs: List[str] = []
    used_db: Optional[str] = None
    updated_keywords: dict[str, Any] = {}
    wrote_wcs = False
    removed_keys = 0
    scale_hint: Optional[float] = None
    fov_hint: tuple[Optional[float], Optional[float]] = (None, None)
    luminance_backup: Optional[np.ndarray] = None
    luminance_used = False
    profile_name = profile
    hint_ra_deg = float(ra_hint) if ra_hint is not None else None
    hint_dec_deg = float(dec_hint) if dec_hint is not None else None
    try:
        with fits.open(work_path, mode="update", memmap=False) as hdul:
            primary = hdul[0]
            header = primary.header
            if profile_name is None:
                profile_name = _infer_profile(header)
            data = primary.data
            if data is None:
                raise InvalidInputError("Primary HDU has no data")
            height, width = data.shape[-2], data.shape[-1]
            if skip_if_valid and has_valid_wcs(header):
                elapsed = time.perf_counter() - start
                message = "skipped: already valid"
                logger(f"[ZEBLIND] {message}")
                return BlindSolveResult(
                    success=True,
                    message=message,
                    elapsed_sec=elapsed,
                    tried_dbs=[],
                    used_db=None,
                    wrote_wcs=False,
                    updated_keywords={},
                    output_path=str(work_path),
                )
            removed_keys = sanitize_wcs(header)
            logger(f"[ZEBLIND] sanitized_wcs keys_removed={removed_keys}")
            scale_hint, fov_hint = estimate_scale_and_fov(header, width, height)
            preset = PROFILE_PRESETS.get(profile_name.upper() if profile_name else "", {})
            preset_scale = preset.get("scale")  # type: ignore[assignment]
            if isinstance(preset_scale, (int, float)) and preset_scale > 0:
                scale_hint = float(preset_scale)
            preset_fov = preset.get("fov")  # type: ignore[assignment]
            if isinstance(preset_fov, tuple) and preset_fov[0] is not None:
                fov_hint = (preset_fov[0], preset_fov[1])
            if scale_hint:
                fov_x, fov_y = fov_hint
                logger(
                    f"[ZEBLIND] scale_est={scale_hint:.2f} asec/px"
                    + (
                        f" fov≈{fov_x or 0:.2f}x{fov_y or 0:.2f} deg"
                        if fov_x and fov_y
                        else ""
                    )
                )
            needs_luminance = data.ndim != 2 or header.get("BAYERPAT")
            if needs_luminance:
                luminance_backup = np.array(data, copy=True)
                primary.data = to_luminance_for_solve(primary)
                luminance_used = True
                logger("[ZEBLIND] generated luminance plane for CFA data")
            if hint_ra_deg is None:
                hint_ra_deg = _parse_header_angle(header.get("RA") or header.get("OBJCTRA"), is_ra=True)
            if hint_dec_deg is None:
                hint_dec_deg = _parse_header_angle(header.get("DEC") or header.get("OBJCTDEC"), is_ra=False)
            if hint_ra_deg is not None or hint_dec_deg is not None:
                ra_text = f"{hint_ra_deg:.3f}°" if hint_ra_deg is not None else "?"
                dec_text = f"{hint_dec_deg:.3f}°" if hint_dec_deg is not None else "?"
                logger(f"[ZEBLIND] hints ra={ra_text} dec={dec_text}")
            hdul.flush()
        astap_path = _resolve_astap_executable(astap_exe)
        logger(f"[ZEBLIND] astap={astap_path}")
        for db in db_chain:
            tried_dbs.append(db)
            label = _format_db_label(db)
            logger(f"[ZEBLIND] trying ASTAP db={label} timeout={timeout_sec}s")
            cmd = [
                astap_path,
                "-f",
                str(work_path),
                "-z",
                db,
                "-r",
                str(DEFAULT_RADIUS_TOLERANCE),
                "-update",
            ]
            ra_hours = _ra_deg_to_hours(hint_ra_deg) if hint_ra_deg is not None else None
            spd = _dec_deg_to_spd(hint_dec_deg) if hint_dec_deg is not None else None
            if ra_hours is not None:
                cmd.extend(["-ra", f"{ra_hours:.6f}"])
            if spd is not None:
                cmd.extend(["-spd", f"{spd:.6f}"])
            if extra_args:
                cmd.extend(extra_args)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise AstapExecutionError(f"ASTAP timed out after {timeout_sec}s on {label}") from exc
            except OSError as exc:
                raise AstapExecutionError(f"Failed to launch ASTAP: {exc}") from exc
            if verbose:
                if proc.stdout:
                    logger(f"[ZEBLIND] astap stdout ({label}): {proc.stdout.strip()}")
                if proc.stderr:
                    logger(f"[ZEBLIND] astap stderr ({label}): {proc.stderr.strip()}")
            if proc.returncode != 0:
                logger(f"[ZEBLIND] fail db={label} code={proc.returncode}")
                continue
            with fits.open(work_path, mode="update", memmap=False) as hdul:
                header = hdul[0].header
                if not has_valid_wcs(header):
                    logger(f"[ZEBLIND] ASTAP reported success but WCS invalid for db={label}")
                    continue
                header.setdefault("RADESYS", "ICRS")
                header.setdefault("EQUINOX", 2000.0)
                header["SOLVED"] = (1, "Solved via zeblindsolver")
                header["ZESOLVER_HINT"] = (1, "ZeSolver blind fallback marker")
                header["ZEBLINDVER"] = (ZEBLIND_VERSION, "zeblindsolver version")
                header["USED_DB"] = (label, "ASTAP database used for blind solve")
                updated_keywords.update(
                    {
                        "SOLVED": 1,
                        "ZESOLVER_HINT": 1,
                        "ZEBLINDVER": ZEBLIND_VERSION,
                        "USED_DB": label,
                    }
                )
                if scale_hint:
                    header["SCALE_EST"] = (float(scale_hint), "Estimated scale (arcsec/px)")
                    updated_keywords["SCALE_EST"] = float(scale_hint)
                if profile_name:
                    header["PROFILE"] = (profile_name, "Instrument profile hint")
                    updated_keywords["PROFILE"] = profile_name
                hdul.flush()
            used_db = label
            wrote_wcs = True
            duration = time.perf_counter() - start
            logger(f"[ZEBLIND] success db={label} elapsed={duration:.1f}s")
            message = f"Solved via ASTAP {label}"
            return BlindSolveResult(
                success=True,
                message=message,
                elapsed_sec=duration,
                tried_dbs=tried_dbs,
                used_db=used_db,
                wrote_wcs=wrote_wcs,
                updated_keywords=updated_keywords,
                output_path=str(work_path),
            )
        elapsed = time.perf_counter() - start
        return BlindSolveResult(
            success=False,
            message="Blind solve failed for all databases",
            elapsed_sec=elapsed,
            tried_dbs=tried_dbs,
            used_db=None,
            wrote_wcs=False,
            updated_keywords={},
            output_path=str(work_path),
        )
    finally:
        if luminance_used and luminance_backup is not None:
            with fits.open(work_path, mode="update", memmap=False) as hdul:
                hdul[0].data = luminance_backup
                hdul.flush()


def _split_db_argument(text: str) -> List[str]:
    parts: List[str] = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind WCS solver around ASTAP")
    parser.add_argument("--input", required=True, help="Input FITS file")
    parser.add_argument("--db", required=True, help="Semicolon separated ASTAP database directories")
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS.keys()), help="Instrument profile hint")
    parser.add_argument("--timeout", type=int, default=90, help="Per-database timeout in seconds")
    parser.add_argument("--skip-if-valid", dest="skip_if_valid", action="store_true", default=True, help="Skip solving if the FITS already holds a valid WCS")
    parser.add_argument("--no-skip-if-valid", dest="skip_if_valid", action="store_false", help="Force solving even if an existing WCS is detected")
    parser.add_argument("--ra-hint", type=float, help="Optional RA hint in degrees (0-360)")
    parser.add_argument("--dec-hint", type=float, help="Optional DEC hint in degrees (-90..+90)")
    parser.add_argument("--write-to", help="Optional output FITS path (defaults to in-place update)")
    parser.add_argument("--astap-exe", help="Path to astap.exe/astap binary")
    parser.add_argument("--extra-astap-args", help="Additional arguments passed to ASTAP")
    parser.add_argument("--verbose", action="store_true", help="Print ASTAP stdout/stderr")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    db_chain = _split_db_argument(args.db)
    if not db_chain:
        parser.error("--db requires at least one entry")
    extra_args = shlex.split(args.extra_astap_args) if args.extra_astap_args else None
    try:
        result = blind_solve(
            fits_path=args.input,
            db_roots=db_chain,
            astap_exe=args.astap_exe,
            profile=args.profile,
            timeout_sec=args.timeout,
            skip_if_valid=args.skip_if_valid,
            ra_hint=args.ra_hint,
            dec_hint=args.dec_hint,
            write_to=args.write_to,
            extra_args=extra_args,
            verbose=args.verbose,
        )
    except BlindSolverRuntimeError as exc:
        print(f"[ZEBLIND] error: {exc}", file=sys.stderr)
        return exc.exit_code
    status = 0 if result["success"] else 1
    print(f"[ZEBLIND] {result['message']}")
    if result["success"] and result.get("used_db"):
        print(
            f"[ZEBLIND] used_db={result['used_db']} elapsed={result['elapsed_sec']:.1f}s",
            flush=True,
        )
    elif not result["success"]:
        print("[ZEBLIND] tried_dbs=" + ",".join(result["tried_dbs"]), file=sys.stderr)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
