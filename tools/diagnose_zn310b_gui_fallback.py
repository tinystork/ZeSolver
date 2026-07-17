#!/usr/bin/env python3
"""Analyze a ZN3.10B GUI log against the hybrid fallback manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_md(path: Path, title: str, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True, default=str)}\n```\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_events(log_path: Path | None) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {}
    if log_path is None or not log_path.exists():
        return events
    pattern = re.compile(r"ZN310B_EVENT\s+(\{.*\})")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        try:
            payload = json.loads(m.group(1))
        except Exception:
            continue
        name = str(payload.get("case_filename") or "")
        if name:
            events.setdefault(name, []).append(payload)
    return events


def angular_sep_arcsec(a: tuple[float, float], b: tuple[float, float]) -> float:
    ra1, dec1 = map(math.radians, a)
    ra2, dec2 = map(math.radians, b)
    c = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 3600.0


def fits_center(path: Path) -> tuple[float, float] | None:
    try:
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header
            data = hdul[0].data
            if data is None:
                return None
            w = WCS(header)
            if not w.has_celestial:
                return None
            height, width = np.asarray(data).shape[-2:]
            ra, dec = w.pixel_to_world_values(width / 2.0, height / 2.0)
            return float(ra), float(dec)
    except Exception:
        return None


def oracle_center(sidecar: Path) -> tuple[float, float] | None:
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if data.get("center_ra_deg") is not None and data.get("center_dec_deg") is not None:
        return float(data["center_ra_deg"]), float(data["center_dec_deg"])
    raw = data.get("raw") or {}
    wcs = raw.get("wcs") or {}
    crval = wcs.get("crval")
    if isinstance(crval, list) and len(crval) >= 2:
        return float(crval[0]), float(crval[1])
    return None


def classify(item: dict[str, Any], final_path: Path, events: list[dict[str, Any]], source_sha_after: str | None) -> dict[str, Any]:
    near_events = [e for e in events if e.get("event") == "near_result"]
    blind_events = [e for e in events if e.get("event") == "blind4d_result"]
    historical = any(bool(e.get("historical_blind_called")) for e in events)
    web = any(bool(e.get("astrometry_web_called")) for e in events)
    source_ok = source_sha_after in {None, item.get("source_SHA256")}
    final_center = fits_center(final_path)
    expected = oracle_center(Path(item["oracle_sidecar"]))
    sep = angular_sep_arcsec(final_center, expected) if final_center and expected else None
    wcs_confirmed = sep is not None and sep < 15.0
    near_success = any(bool(e.get("near_success")) for e in near_events)
    blind_success = any(bool(e.get("blind4d_success")) for e in blind_events)
    blind_calls = sum(int(e.get("blind4d_call_count") or (1 if e.get("blind4d_called") else 0)) for e in blind_events)

    if historical:
        cls = "HISTORICAL_BACKEND_CALLED"
    elif web:
        cls = "ASTROMETRY_WEB_CALLED"
    elif not source_ok:
        cls = "ORIGINAL_MODIFIED"
    elif final_center is None:
        cls = "OUTPUT_WCS_MISSING"
    elif item["variant"] == "CONTROL":
        if blind_calls:
            cls = "CONTROL_UNEXPECTED_4D"
        elif near_success and wcs_confirmed:
            cls = "CONTROL_NEAR_CORRECT"
        else:
            cls = "UNRESOLVED"
    elif item["variant"] == "NOHINT":
        if near_success:
            cls = "NOHINT_NEAR_UNEXPECTED_SUCCESS"
        elif blind_success and blind_calls == 1 and wcs_confirmed:
            cls = "NOHINT_4D_CORRECT"
        elif blind_calls:
            cls = "BLIND4D_WRONG" if not wcs_confirmed else "BLIND4D_FAILED"
        else:
            cls = "UNRESOLVED"
    else:
        if near_success and not blind_calls:
            cls = "BADHINT_NEAR_WRONG_ACCEPTED"
        elif blind_success and blind_calls == 1 and wcs_confirmed:
            cls = "BADHINT_4D_CORRECT"
        elif blind_calls:
            cls = "BLIND4D_WRONG" if not wcs_confirmed else "BLIND4D_FAILED"
        else:
            cls = "UNRESOLVED"
    return {
        "gui_filename": item["gui_filename"],
        "variant": item["variant"],
        "classification": cls,
        "near_events": near_events,
        "blind4d_events": blind_events,
        "blind4d_call_count": blind_calls,
        "final_wcs_center": final_center,
        "oracle_center": expected,
        "center_separation_arcsec": sep,
        "wcs_confirmed": wcs_confirmed,
        "source_sha_after": source_sha_after,
        "original_unmodified": source_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=ROOT / "reports" / "zenear_zn310b_gui_manifest.json")
    ap.add_argument("--oracle-root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=ROOT / "reports" / "zenear_zn310b_gui_result.json")
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    events = parse_events(args.log)
    rows: dict[str, Any] = {}
    for item in manifest.get("items", []):
        final_path = Path(item["gui_path"])
        source_path = Path(item["source_path"])
        source_after = sha256_file(source_path) if source_path.exists() else None
        rows[item["gui_filename"]] = classify(item, final_path, events.get(item["gui_filename"], []), source_after)
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    payload = {
        "log": str(args.log) if args.log else None,
        "manifest": str(args.manifest),
        "cases": rows,
        "summary": {
            "counts": counts,
            "historical_blind_called": counts.get("HISTORICAL_BACKEND_CALLED", 0),
            "astrometry_web_called": counts.get("ASTROMETRY_WEB_CALLED", 0),
            "verdict": "NOT_RUN" if not events else ("PASS" if not any(k in counts for k in ("HISTORICAL_BACKEND_CALLED", "ASTROMETRY_WEB_CALLED", "BADHINT_NEAR_WRONG_ACCEPTED", "BLIND4D_WRONG", "OUTPUT_WCS_MISSING", "ORIGINAL_MODIFIED", "UNRESOLVED")) else "FAIL"),
        },
    }
    write_json(args.output, payload)
    write_md(args.output.with_suffix(".md"), "ZN3.10B GUI result", payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
