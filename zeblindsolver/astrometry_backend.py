from __future__ import annotations

import math
import os
import time
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
    # For fallback
    index_root: Optional[str] = None


@dataclass
class JobResult:
    path: Path
    success: bool
    message: str


def _extract_hints(header: fits.Header, width: int, height: int) -> dict:
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
        # Tolerant scale bounds
        hints["scale_units"] = "arcsecperpix"
        hints["scale_lower"] = max(1e-6, scale_arcsec * 0.6)
        hints["scale_upper"] = scale_arcsec * 1.6
    if radius and radius > 0:
        hints["radius"] = float(min(30.0, max(0.05, radius)))
    return hints


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
    # Prepare hints
    hints: dict[str, Any] = {}
    try:
        with fits.open(path, mode="readonly", memmap=False) as hdul:
            hdr = hdul[0].header
            h, w = int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]) if hdul[0].data is not None else (0, 0)
        if cfg.use_hints:
            hints = _extract_hints(hdr, w, h)
    except Exception:
        hints = {}
    # Submit
    try:
        sub = client.submit_fits(str(path), hints=hints)
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
        try:
            client = AstrometryClient(cfg.api_url)
            client.login(cfg.api_key)
            with fits.open(path, mode="readonly", memmap=False) as hdul:
                hdr = hdul[0].header
                h, w = int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]) if hdul[0].data is not None else (0, 0)
            hints = _extract_hints(hdr, w, h) if cfg.use_hints else {}
            _report(idx, f"[{idx}/{total}] {path.name}: uploading")
            sub = client.submit_fits(str(path), hints=hints)
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
