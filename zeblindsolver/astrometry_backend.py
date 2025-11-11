from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np
from astropy.io import fits

from .fits_utils import estimate_scale_and_fov as _estimate_scale_and_fov
from .fits_utils import parse_angle as _parse_angle
from .astrometry_client import AstrometryClient, AstrometryClientError, parse_wcs_bytes
try:
    # Internal blind solver for fallback
    from .zeblindsolver import solve_blind as _solve_blind_internal
    from .zeblindsolver import SolveConfig as _BlindSolveConfig
except Exception:  # pragma: no cover - optional import at runtime only
    _solve_blind_internal = None  # type: ignore
    _BlindSolveConfig = None  # type: ignore


_UPLOAD_PRIVACY_FLAGS = {
    "allow_commercial_use": "n",
    "allow_modifications": "n",
    "publicly_visible": "n",
}

_HEADER_COPY_KEYS = ("OBJECT", "DATE-OBS", "EXPTIME", "FILTER", "INSTRUME", "TELESCOP")
_LUMINANCE_COEFFS = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 1, 3)


def _default_log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class AstrometryConfig:
    api_url: str
    api_key: str
    parallel_jobs: int = 2
    timeout_s: int = 600
    use_hints: bool = True
    fallback_local: bool = True
    scale_tolerance_percent: float = 20.0
    # For fallback
    index_root: Optional[str] = None


@dataclass
class JobResult:
    path: Path
    success: bool
    message: str


def _extract_hints(header: fits.Header, width: int, height: int, scale_tolerance_percent: float) -> dict:
    ra = _parse_angle(header.get("CRVAL1") or header.get("RA") or header.get("OBJCTRA"), is_ra=True)
    dec = _parse_angle(header.get("CRVAL2") or header.get("DEC") or header.get("OBJCTDEC"), is_ra=False)
    scale_arcsec, (fov_w_deg, fov_h_deg) = _estimate_scale_and_fov(header, width, height)
    radius = None
    if fov_w_deg and fov_h_deg:
        radius = 0.5 * float(max(fov_w_deg, fov_h_deg))
    hints: dict[str, Any] = {}
    if ra is not None and dec is not None:
        hints["center_ra"] = float(ra)
        hints["center_dec"] = float(dec)
    if scale_arcsec and scale_arcsec > 0:
        tol = max(0.0, float(scale_tolerance_percent))
        frac = tol / 100.0 if tol > 0 else 0.0
        scale_lower = scale_arcsec * (1.0 - frac)
        scale_upper = scale_arcsec * (1.0 + frac)
        hints["scale_units"] = "arcsecperpix"
        hints["scale_lower"] = max(1e-6, scale_lower)
        hints["scale_upper"] = max(scale_lower, scale_upper)
    if radius and radius > 0:
        hints["radius"] = float(min(30.0, max(0.05, radius)))
    return hints


def _prepare_submission_image(data: np.ndarray, header: Optional[fits.Header]) -> Path:
    img = np.asarray(data)
    if img is None:
        raise ValueError("image data missing")
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
        img = np.sum(img.astype(np.float32) * _LUMINANCE_COEFFS, axis=2, dtype=np.float32)
    elif img.ndim == 2:
        img = img.astype(np.float32)
    else:
        raise ValueError(f"unsupported image shape {img.shape!r}")
    img = np.nan_to_num(img, copy=False)
    min_v = float(np.min(img)) if img.size else 0.0
    max_v = float(np.max(img)) if img.size else 0.0
    if not np.isfinite(min_v):
        min_v = 0.0
    if not np.isfinite(max_v):
        max_v = min_v
    if max_v > min_v:
        norm = (img - min_v) / (max_v - min_v)
    else:
        norm = np.zeros_like(img, dtype=np.float32)
    data_uint16 = (np.clip(norm, 0.0, 1.0) * 65535.0).astype(np.uint16)
    data_int16 = (data_uint16.astype(np.int32) - 32768).astype(np.int16)
    header_out = fits.Header()
    header_out["SIMPLE"] = True
    header_out["BITPIX"] = 16
    header_out["BSCALE"] = 1
    header_out["BZERO"] = 32768
    header_out["NAXIS"] = 2
    header_out["NAXIS1"] = data_int16.shape[1]
    header_out["NAXIS2"] = data_int16.shape[0]
    for key in _HEADER_COPY_KEYS:
        if header and key in header:
            header_out[key] = header[key]
    fd, tmp_name = tempfile.mkstemp(prefix="zesolver_astrometry_", suffix=".fits")
    os.close(fd)
    tmp_path = Path(tmp_name)
    fits.writeto(tmp_path, data_int16, header=header_out, overwrite=True, output_verify="silentfix")
    return tmp_path


def _write_wcs_header(fits_path: Path, cards: Dict[str, Any], *, backend_label: str = "astrometry.net") -> None:
    with fits.open(fits_path, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        # Merge returned WCS cards
        for k, v in cards.items():
            try:
                hdr[k] = v
            except Exception:
                continue
        # Stamp additional provenance keys
        hdr["SOLVED"] = (True, "WCS solved")
        hdr["BACKEND"] = (backend_label, "WCS backend")
        # Keep compatibility with project’s blind version tag when present
        try:
            from .zeblindsolver import ZEBLIND_VERSION  # type: ignore

            hdr.setdefault("BLINDVER", ZEBLIND_VERSION)
            hdr.setdefault("ZBLNDVER", ZEBLIND_VERSION)
        except Exception:
            pass
        hdul.flush()


def solve_single(
    fits_path: Path | str,
    cfg: AstrometryConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> JobResult:
    log_fn = log or _default_log
    path = Path(fits_path).expanduser().resolve()
    if not path.is_file():
        return JobResult(path, False, "file not found")
    # Login
    try:
        client = AstrometryClient(cfg.api_url)
        client.login(cfg.api_key)
    except AstrometryClientError as exc:
        return JobResult(path, False, f"login failed: {exc}")
    header: Optional[fits.Header] = None
    img_data: Optional[np.ndarray] = None
    h = w = 0
    try:
        with fits.open(path, mode="readonly", memmap=False) as hdul:
            header = hdul[0].header.copy()
            if hdul[0].data is not None:
                img_data = np.asarray(hdul[0].data)
                if img_data.ndim >= 2:
                    h = int(img_data.shape[-2])
                    w = int(img_data.shape[-1])
    except Exception as exc:
        log_fn(f"Astrometry prep: unable to read FITS header/data ({exc})")
    hints: dict[str, Any] = dict(_UPLOAD_PRIVACY_FLAGS)
    if cfg.use_hints and header is not None and h > 0 and w > 0:
        try:
            hints.update(_extract_hints(header, w, h, cfg.scale_tolerance_percent))
        except Exception as exc:
            log_fn(f"Astrometry prep: unable to extract hints ({exc})")
    prepared_path = path
    temp_path: Optional[Path] = None
    if img_data is not None:
        try:
            prepared_path = _prepare_submission_image(img_data, header)
            temp_path = prepared_path
        except Exception as exc:
            log_fn(f"Astrometry prep: fallback to original FITS ({exc})")
    # Submit
    try:
        sub = client.submit_fits(str(prepared_path), hints=hints)
        job_id = client.poll_submission_for_job(int(sub.get("subid") or sub.get("submissionid")), timeout=max(30.0, float(cfg.timeout_s) * 0.5))
        info = client.wait_for_job(job_id, timeout=float(cfg.timeout_s))
        status = str(info.get("status") or info.get("job_status") or "").lower()
        if status not in {"success"}:
            return JobResult(path, False, f"job {job_id} {status or 'failed'}")
        raw = client.download_wcs(job_id)
        cards = parse_wcs_bytes(raw)
        if not cards:
            return JobResult(path, False, "no WCS returned")
        _write_wcs_header(path, cards)
        return JobResult(path, True, f"job {job_id} solved")
    except AstrometryClientError as exc:
        return JobResult(path, False, str(exc))
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def solve_batch(
    files: Iterable[Path | str],
    cfg: AstrometryConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
    progress_hook: Optional[Callable[[int, str], None]] = None,
) -> Iterator[JobResult]:
    paths = [Path(p).expanduser().resolve() for p in files]
    if not paths:
        return iter(())

    total = len(paths)

    def _report(idx: int, message: str) -> None:
        if progress_hook:
            try:
                progress_hook(idx, message)
            except Exception:
                pass
        if log:
            try:
                log(message)
            except Exception:
                pass

    for idx, path in enumerate(paths, start=1):
        temp_path: Optional[Path] = None
        try:
            client = AstrometryClient(cfg.api_url)
            client.login(cfg.api_key)
            header: Optional[fits.Header] = None
            img_data: Optional[np.ndarray] = None
            h = w = 0
            try:
                with fits.open(path, mode="readonly", memmap=False) as hdul:
                    header = hdul[0].header.copy()
                    if hdul[0].data is not None:
                        img_data = np.asarray(hdul[0].data)
                        if img_data.ndim >= 2:
                            h = int(img_data.shape[-2])
                            w = int(img_data.shape[-1])
            except Exception as exc:
                _report(idx, f"[{idx}/{total}] {path.name}: lecture FITS impossible ({exc})")
            hints: dict[str, Any] = dict(_UPLOAD_PRIVACY_FLAGS)
            if cfg.use_hints and header is not None and h > 0 and w > 0:
                try:
                    hints.update(_extract_hints(header, w, h, cfg.scale_tolerance_percent))
                except Exception as exc:
                    _report(idx, f"[{idx}/{total}] {path.name}: extraction des hints impossible ({exc})")
            prepared_path = path
            if img_data is not None:
                try:
                    prepared_path = _prepare_submission_image(img_data, header)
                    temp_path = prepared_path
                except Exception as exc:
                    _report(idx, f"[{idx}/{total}] {path.name}: fallback sur FITS original ({exc})")
            _report(idx, f"[{idx}/{total}] {path.name}: uploading")
            sub = client.submit_fits(str(prepared_path), hints=hints)
            job_id = client.poll_submission_for_job(int(sub.get("subid") or sub.get("submissionid")), timeout=max(30.0, float(cfg.timeout_s) * 0.5))
            _report(idx, f"[{idx}/{total}] {path.name}: job {job_id} en file")
            info = client.wait_for_job(job_id, timeout=float(cfg.timeout_s))
            status = str(info.get("status") or info.get("job_status") or "").lower()
            if status != "success":
                _report(idx, f"[{idx}/{total}] {path.name}: job {job_id} {status or 'failed'}")
                # Optional fallback to local blind solver
                if cfg.fallback_local and cfg.index_root and _solve_blind_internal and _BlindSolveConfig:
                    _report(idx, f"[{idx}/{total}] {path.name}: fallback local (ZeBlind)")
                    try:
                        sol = _solve_blind_internal(str(path), str(Path(cfg.index_root).expanduser()), config=_BlindSolveConfig())
                        if sol.success:
                            _report(idx, f"[{idx}/{total}] {path.name}: fallback résolu")
                            yield JobResult(path, True, "fallback solved")
                            continue
                        else:
                            _report(idx, f"[{idx}/{total}] {path.name}: fallback échec ({sol.message})")
                    except Exception as exc:
                        _report(idx, f"[{idx}/{total}] {path.name}: fallback erreur ({exc})")
                yield JobResult(path, False, f"job {job_id} {status or 'failed'}")
                continue
            raw = client.download_wcs(job_id)
            cards = parse_wcs_bytes(raw)
            if not cards:
                _report(idx, f"[{idx}/{total}] {path.name}: job {job_id} sans WCS")
                yield JobResult(path, False, "no WCS returned")
                continue
            _write_wcs_header(path, cards)
            _report(idx, f"[{idx}/{total}] {path.name}: job {job_id} résolu")
            yield JobResult(path, True, f"job {job_id} solved")
        except AstrometryClientError as exc:
            _report(idx, f"[{idx}/{total}] {path.name}: {exc}")
            # Optional fallback on API error
            if cfg.fallback_local and cfg.index_root and _solve_blind_internal and _BlindSolveConfig:
                _report(idx, f"[{idx}/{total}] {path.name}: fallback local (ZeBlind)")
                try:
                    sol = _solve_blind_internal(str(path), str(Path(cfg.index_root).expanduser()), config=_BlindSolveConfig())
                    if sol.success:
                        _report(idx, f"[{idx}/{total}] {path.name}: fallback résolu")
                        yield JobResult(path, True, "fallback solved")
                        continue
                    else:
                        _report(idx, f"[{idx}/{total}] {path.name}: fallback échec ({sol.message})")
                except Exception as exc2:
                    _report(idx, f"[{idx}/{total}] {path.name}: fallback erreur ({exc2})")
            yield JobResult(path, False, str(exc))
        except Exception as exc:
            _report(idx, f"[{idx}/{total}] {path.name}: {exc}")
            yield JobResult(path, False, str(exc))
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
