from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "corpus" / "manifest.json"


class CorpusError(RuntimeError):
    pass


class CorpusDataMissing(CorpusError):
    pass


class CorpusDataCorrupt(CorpusError):
    pass


class WcsValidationError(AssertionError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class CorpusCase:
    raw: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def mode(self) -> str:
        return str(self.raw["solver_mode"])

    @property
    def group(self) -> str:
        return str(self.raw["source_group"])

    @property
    def enabled(self) -> bool:
        return bool(self.raw.get("enabled", True))

    @property
    def env(self) -> str:
        return str(self.raw.get("root_env") or "ZESOLVER_CORPUS_ROOT")

    @property
    def relative_path(self) -> str:
        return str(self.raw.get("relative_path") or "")

    def resolve_path(self) -> Path:
        root = os.environ.get(self.env)
        if not root:
            raise CorpusDataMissing(f"{self.env} is not set for corpus case {self.id}")
        path = Path(root) / self.relative_path
        if not path.exists():
            raise CorpusDataMissing(f"corpus case {self.id} missing file: {path}")
        expected = self.raw.get("sha256")
        if expected:
            actual = sha256_file(path)
            if actual.lower() != str(expected).lower():
                raise CorpusDataCorrupt(f"corpus case {self.id} SHA256 mismatch: {actual} != {expected}")
        return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise CorpusError(f"unsupported corpus manifest schema: {data.get('schema_version')!r}")
    ids = [case.get("id") for case in data.get("cases", [])]
    if len(ids) != len(set(ids)):
        raise CorpusError("duplicate corpus case id")
    return data


def iter_cases(
    *,
    manifest: dict[str, Any] | None = None,
    mode: str | None = None,
    group: str | None = None,
    enabled_only: bool = True,
) -> Iterable[CorpusCase]:
    data = manifest or load_manifest()
    for raw in data.get("cases", []):
        case = CorpusCase(raw)
        if enabled_only and not case.enabled:
            continue
        if mode and case.mode != mode:
            continue
        if group and case.group != group:
            continue
        yield case


def angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1r, dec1r, ra2r, dec2r = map(math.radians, [ra1, dec1, ra2, dec2])
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = math.sin(ddec / 2.0) ** 2 + math.cos(dec1r) * math.cos(dec2r) * math.sin(dra / 2.0) ** 2
    return math.degrees(2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))) * 3600.0


def wcs_pixel_scale_arcsec(wcs: WCS) -> float:
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    finite = scales[np.isfinite(scales)]
    if finite.size < 2:
        raise WcsValidationError("SCALE_OUT_OF_TOLERANCE", "pixel scale is not finite")
    return float(np.mean(np.abs(finite[:2])))


def wcs_orientation_deg(wcs: WCS) -> float:
    cd = getattr(wcs.wcs, "cd", None)
    if cd is None:
        pc = getattr(wcs.wcs, "pc", None)
        cdelt = getattr(wcs.wcs, "cdelt", None)
        if pc is None or cdelt is None:
            raise WcsValidationError("INVALID_WCS_MATRIX", "missing CD/PC matrix")
        cd = np.asarray(pc, dtype=float) @ np.diag(np.asarray(cdelt, dtype=float))
    cd = np.asarray(cd, dtype=float)[:2, :2]
    if not np.all(np.isfinite(cd)) or abs(float(np.linalg.det(cd))) < 1e-16:
        raise WcsValidationError("INVALID_WCS_MATRIX", "non-finite or degenerate CD matrix")
    return float(math.degrees(math.atan2(cd[1, 0], cd[0, 0])))


def wcs_parity(wcs: WCS) -> str:
    cd = getattr(wcs.wcs, "cd", None)
    if cd is None:
        pc = getattr(wcs.wcs, "pc", None)
        cdelt = getattr(wcs.wcs, "cdelt", None)
        if pc is None or cdelt is None:
            raise WcsValidationError("INVALID_WCS_MATRIX", "missing CD/PC matrix")
        cd = np.asarray(pc, dtype=float) @ np.diag(np.asarray(cdelt, dtype=float))
    det = float(np.linalg.det(np.asarray(cd, dtype=float)[:2, :2]))
    if not math.isfinite(det) or abs(det) < 1e-16:
        raise WcsValidationError("INVALID_WCS_MATRIX", "degenerate CD determinant")
    return "negative" if det < 0 else "positive"


def validate_wcs_against_case(wcs: WCS, case: CorpusCase, *, inliers: int | None = None, rms_px: float | None = None) -> None:
    if wcs is None or not bool(getattr(wcs, "is_celestial", False)):
        raise WcsValidationError("INVALID_WCS_MATRIX", "missing celestial WCS")
    crval = np.asarray(wcs.wcs.crval, dtype=float)
    if crval.size < 2 or not np.all(np.isfinite(crval[:2])):
        raise WcsValidationError("INVALID_WCS_MATRIX", "non-finite CRVAL")

    raw = case.raw
    if raw.get("expected_center_ra_deg") is not None and raw.get("expected_center_dec_deg") is not None:
        sep = angular_sep_arcsec(
            float(raw["expected_center_ra_deg"]),
            float(raw["expected_center_dec_deg"]),
            float(crval[0]),
            float(crval[1]),
        )
        tol = float(raw.get("center_tolerance_arcsec") or 0.0)
        if sep > tol:
            raise WcsValidationError("CENTER_OUT_OF_TOLERANCE", f"{sep:.3f} arcsec > {tol:.3f}")

    if raw.get("expected_pixel_scale_arcsec") is not None:
        scale = wcs_pixel_scale_arcsec(wcs)
        expected = float(raw["expected_pixel_scale_arcsec"])
        pct = abs(scale - expected) / expected * 100.0
        tol = float(raw.get("pixel_scale_tolerance_percent") or 0.0)
        if pct > tol:
            raise WcsValidationError("SCALE_OUT_OF_TOLERANCE", f"{pct:.3f}% > {tol:.3f}%")

    if raw.get("expected_parity") is not None and wcs_parity(wcs) != raw["expected_parity"]:
        raise WcsValidationError("PARITY_MISMATCH")

    if raw.get("minimum_inliers") is not None and inliers is not None and inliers < int(raw["minimum_inliers"]):
        raise WcsValidationError("INSUFFICIENT_INLIERS", f"{inliers} < {raw['minimum_inliers']}")

    if raw.get("maximum_rms_px") is not None and rms_px is not None and rms_px > float(raw["maximum_rms_px"]):
        raise WcsValidationError("RMS_TOO_HIGH", f"{rms_px:.3f} > {raw['maximum_rms_px']}")
