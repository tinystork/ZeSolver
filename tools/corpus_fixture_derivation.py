#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


WCS_EXACT_KEYS = {
    "WCSAXES",
    "LONPOLE",
    "LATPOLE",
    "RADESYS",
    "EQUINOX",
    "MJDREF",
}

SOLVER_EXACT_KEYS = {
    "SOLVED",
    "SOLVER",
    "SOLVMODE",
    "SOLVBACK",
    "SOLVSTAT",
    "SOLVMSG",
    "SOLVRMS",
    "SOLVINLR",
}

CHECKSUM_KEYS = {"CHECKSUM", "DATASUM"}

LEGITIMATE_INPUT_METADATA = {
    "RA",
    "DEC",
    "OBJCTRA",
    "OBJCTDEC",
    "OBJRA",
    "OBJDEC",
    "OBJ_RA",
    "OBJ_DEC",
    "OBJECT",
    "FOCALLEN",
    "FOCLEN",
    "FOCUSPOS",
    "PIXSIZE",
    "XPIXSZ",
    "YPIXSZ",
    "DATE",
    "DATE-OBS",
    "EXPTIME",
    "EXPOSURE",
    "FILTER",
    "GAIN",
    "OFFSET",
    "INSTRUME",
    "TELESCOP",
    "BAYERPAT",
}


@dataclass(frozen=True, slots=True)
class HduAudit:
    index: int
    name: str
    data_shape: tuple[int, ...] | None
    data_dtype: str | None
    pixel_hash: str | None
    has_celestial_wcs: bool
    wcs_cards: tuple[str, ...]
    solver_cards: tuple[str, ...]
    checksum_cards: tuple[str, ...]
    retained_input_metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class FixtureDerivationReport:
    source_path: str
    output_path: str
    source_sha256: str
    output_sha256: str
    source_hdus: tuple[HduAudit, ...]
    output_hdus: tuple[HduAudit, ...]
    removed_cards_by_hdu: tuple[tuple[int, tuple[str, ...]], ...]
    pixels_unchanged: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(data: Any) -> str:
    arr = np.ascontiguousarray(data)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _is_axis_wcs_key(key: str) -> bool:
    return bool(
        re.match(r"^(CRPIX|CRVAL|CTYPE|CUNIT|CDELT|CROTA|CNAME)\d+[A-Z]?$", key)
        or re.match(r"^(CD|PC|PV|PS)\d+_\d+[A-Z]?$", key)
    )


def _is_sip_key(key: str) -> bool:
    return bool(
        key in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}
        or re.match(r"^(A|B|AP|BP)_\d+_\d+$", key)
    )


def _is_distortion_key(key: str) -> bool:
    return bool(
        re.match(r"^(CPDIS|DP|DQ)\d+[A-Z]?$", key)
        or re.match(r"^(DP|DQ)\d+\.\w+", key)
    )


def is_wcs_card(key: str) -> bool:
    ku = str(key).upper()
    return ku in WCS_EXACT_KEYS or _is_axis_wcs_key(ku) or _is_sip_key(ku) or _is_distortion_key(ku)


def is_solver_card(key: str, value: Any = None) -> bool:
    ku = str(key).upper()
    if ku in SOLVER_EXACT_KEYS:
        return True
    if ku.startswith("SOLV") and ku not in {"SOLUTION"}:
        return True
    if isinstance(value, str) and "ZESOLVER" in value.upper():
        return True
    return False


def audit_fits(path: Path) -> tuple[HduAudit, ...]:
    audits: list[HduAudit] = []
    with fits.open(path, memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            header = hdu.header
            data = hdu.data
            wcs_cards = tuple(key for key in header.keys() if is_wcs_card(str(key)))
            solver_cards = tuple(key for key in header.keys() if is_solver_card(str(key), header.get(key)))
            checksum_cards = tuple(key for key in header.keys() if str(key).upper() in CHECKSUM_KEYS)
            retained = {
                key: str(header.get(key))
                for key in header.keys()
                if str(key).upper() in LEGITIMATE_INPUT_METADATA
            }
            audits.append(
                HduAudit(
                    index=index,
                    name=str(hdu.name),
                    data_shape=None if data is None else tuple(int(v) for v in data.shape),
                    data_dtype=None if data is None else str(data.dtype),
                    pixel_hash=None if data is None else pixel_hash(data),
                    has_celestial_wcs=bool(WCS(header, naxis=2, relax=True).has_celestial),
                    wcs_cards=wcs_cards,
                    solver_cards=solver_cards,
                    checksum_cards=checksum_cards,
                    retained_input_metadata=retained,
                )
            )
    return tuple(audits)


def strip_solver_wcs_cards(source: Path, output: Path) -> FixtureDerivationReport:
    source = Path(source)
    output = Path(output)
    source_audit = audit_fits(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    removed_by_hdu: list[tuple[int, tuple[str, ...]]] = []
    with fits.open(source, memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            removed: list[str] = []
            for key in list(hdu.header.keys()):
                value = hdu.header.get(key)
                ku = str(key).upper()
                if ku in CHECKSUM_KEYS or is_wcs_card(ku) or is_solver_card(ku, value):
                    del hdu.header[key]
                    removed.append(str(key))
            removed_by_hdu.append((index, tuple(removed)))
        hdul.writeto(output, overwrite=True, checksum=False, output_verify="exception")

    output_audit = audit_fits(output)
    if any(hdu.has_celestial_wcs for hdu in output_audit):
        raise RuntimeError("canonical fixture still contains celestial WCS")
    if any(hdu.solver_cards for hdu in output_audit):
        raise RuntimeError("canonical fixture still contains solver cards")
    if any(hdu.checksum_cards for hdu in output_audit):
        raise RuntimeError("canonical fixture still contains checksum cards")

    source_hashes = [h.pixel_hash for h in source_audit if h.pixel_hash is not None]
    output_hashes = [h.pixel_hash for h in output_audit if h.pixel_hash is not None]
    pixels_unchanged = source_hashes == output_hashes
    if not pixels_unchanged:
        raise RuntimeError("canonical fixture changed image pixels")

    return FixtureDerivationReport(
        source_path=str(source),
        output_path=str(output),
        source_sha256=sha256_file(source),
        output_sha256=sha256_file(output),
        source_hdus=source_audit,
        output_hdus=output_audit,
        removed_cards_by_hdu=tuple(removed_by_hdu),
        pixels_unchanged=pixels_unchanged,
    )


def report_to_jsonable(report: FixtureDerivationReport) -> dict[str, Any]:
    return asdict(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive deterministic clean FITS corpus fixtures.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)

    report = strip_solver_wcs_cards(args.source, args.output)
    payload = report_to_jsonable(report)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
