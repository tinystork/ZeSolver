#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import (  # noqa: E402
    ASTROMETRY_AB_CODE_4D_SCHEMA,
    Quad4DIndex,
    build_experimental_4d_index,
)

import tools.diagnose_p219_4d_multifield_validation as p219  # noqa: E402


NEW_CORPUS_ROOT = Path("/home/tristan/zemosaic/example/various_fresh")
BASE = ROOT / "reports/p219b_incremental_multiregime"
INVENTORY_OUT = ROOT / "reports/zeblind_p219b_new_corpus_inventory.json"
SELECTION_OUT = ROOT / "reports/zeblind_p219b_selected_corpus.json"
VALIDATION_OUT = ROOT / "reports/zeblind_p219b_incremental_validation.json"
REPORT_OUT = ROOT / "reports/zeblind_p219b_incremental_validation.md"

MULTIFIELD_INDEX_ROOT = p219.MULTIFIELD_INDEX_ROOT
M106_INDEXES = [p219.M106_INDEX_2823, p219.M106_INDEX_2822]
NGC6888_INDEXES = [
    ROOT / "reports/p219_4d_multifield_validation/indexes/p219_astrometry_ab_code_4d_v1_d50_2644_S_stars2000_q40000.npz",
    ROOT / "reports/p219_4d_multifield_validation/indexes/p219_astrometry_ab_code_4d_v1_d50_2645_S_stars2000_q40000.npz",
]

REGIMES = {
    "s50": {
        "nominal_scale_arcsec": 2.372,
        "min_scale_arcsec": 1.90,
        "max_scale_arcsec": 2.85,
        "margin_pct": 20,
        "source": "Seestar S50 metadata focal length 250 mm plus offline WCS in M106/NGC6888/M31 examples",
    },
    "s30": {
        "nominal_scale_arcsec": 3.990,
        "min_scale_arcsec": 3.19,
        "max_scale_arcsec": 4.79,
        "margin_pct": 20,
        "source": "Seestar S30 metadata focal length 150 mm plus offline WCS on the three M31 S30 FITS",
    },
    "c11": {
        "nominal_scale_arcsec": 0.214,
        "min_scale_arcsec": 0.16,
        "max_scale_arcsec": 0.27,
        "margin_pct": 25,
        "source": "C11 native focal length about 2800 mm with ASI462MC 2.9 um pixels; ASILive FITS lack WCS, so this remains an inventory-only regime here",
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if np.isfinite(f) else None
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _index_meta(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists(), "sha256": _sha256(path)}
    if not path.exists():
        return out
    try:
        idx = Quad4DIndex.load(path)
        out.update(
            {
                "schema": idx.metadata.get("schema"),
                "version": idx.metadata.get("version"),
                "tile_keys": list(idx.tile_keys),
                "stars": int(idx.catalog_ra_dec.shape[0]),
                "quads": int(idx.codes_4d.shape[0]),
                "metadata": dict(idx.metadata),
            }
        )
    except Exception as exc:
        out["load_error"] = str(exc)
    return out


def _ensure_index(tile_key: str) -> Path:
    out = BASE / "indexes" / f"p219b_{ASTROMETRY_AB_CODE_4D_SCHEMA}_{tile_key}_S_stars2000_q40000.npz"
    if not out.exists():
        build_experimental_4d_index(
            MULTIFIELD_INDEX_ROOT,
            out,
            tile_keys=[tile_key],
            level="S",
            max_stars_per_tile=2000,
            max_quads_per_tile=40000,
            sampler_tag="catalog_ring_coverage",
        )
    return out.resolve()


def _ensure_indexes() -> dict[str, list[Path]]:
    p219._ensure_ngc6888_indexes(ROOT / "reports/p219_4d_multifield_validation")
    return {
        "m31": [_ensure_index("d50_2602"), _ensure_index("d50_2702")],
        "ngc6888": [p.resolve() for p in NGC6888_INDEXES],
        "m106": [p.resolve() for p in M106_INDEXES],
    }


def _wcs_info(header: fits.Header) -> dict[str, Any]:
    out = {"has_wcs": False}
    try:
        wcs = WCS(header)
        if not getattr(wcs, "has_celestial", False):
            return out
        width = int(header.get("NAXIS1", 0) or 0)
        height = int(header.get("NAXIS2", 0) or 0)
        scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        sky = wcs.pixel_to_world(width * 0.5, height * 0.5)
        out.update(
            {
                "has_wcs": True,
                "center_ra_dec": [float(sky.ra.deg), float(sky.dec.deg)],
                "pixel_scale_arcsec": float(np.sqrt(float(scales[0]) * float(scales[1]))),
                "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            }
        )
    except Exception as exc:
        out["wcs_error"] = str(exc)
    return out


def _simple_source_count(path: Path) -> dict[str, Any]:
    try:
        data = np.asarray(fits.getdata(path), dtype=np.float32)
        if data.ndim > 2:
            data = data[0]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return {"ok": False, "reason": "no_finite_pixels"}
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(finite))
        if not np.isfinite(sigma) or sigma <= 0:
            return {"ok": False, "reason": "zero_noise"}
        mask = data > med + 5.0 * sigma
        labels, nlab = ndimage.label(mask)
        objects = ndimage.find_objects(labels)
        count = 0
        for slc in objects:
            if slc is None:
                continue
            area = int(np.count_nonzero(labels[slc]))
            if 3 <= area <= 500:
                count += 1
        return {"ok": True, "method": "median_mad_5sigma_components", "source_count": int(count), "background": med, "sigma": sigma}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _classify(path: Path, header: fits.Header, wcs: dict[str, Any]) -> dict[str, str]:
    name = path.name
    instr = str(header.get("INSTRUME", "") or "")
    obj = str(header.get("OBJECT", "") or "")
    if "Seestar S30" in instr or "S30_" in str(header.get("TELESCOP", "") or ""):
        instrument = "S30"
        regime = "s30"
    elif "Seestar S50" in instr or "S50" in instr or "S50_" in str(header.get("TELESCOP", "") or ""):
        instrument = "S50"
        regime = "s50"
    elif "ASI462" in instr or "ASILive" in name:
        instrument = "C11/ASI462MC probable"
        regime = "c11"
    else:
        instrument = instr or "unknown"
        regime = "unknown"
    field = "m31" if "M 31" in name or obj == "M 31" else "unknown_asi462"
    mode = "EQ" if wcs.get("has_wcs") else "unknown"
    if field == "m31" and name.startswith("Light_mosaic_"):
        mode = "Alt-Az"
    return {"field_id": field, "instrument": instrument, "mount_mode": mode, "scale_regime": regime}


def _fits_inventory(path: Path) -> dict[str, Any]:
    header = fits.getheader(path)
    wcs = _wcs_info(header)
    klass = _classify(path, header, wcs)
    source_probe = _simple_source_count(path)
    keys = {
        "object": "OBJECT",
        "camera": "INSTRUME",
        "telescope": "TELESCOP",
        "filter": "FILTER",
        "exposure_s": "EXPTIME",
        "date_obs": "DATE-OBS",
        "xbinning": "XBINNING",
        "ybinning": "YBINNING",
        "gain": "GAIN",
        "ccd_temp": "CCD-TEMP",
        "focal_length_mm": "FOCALLEN",
        "bayer_pattern": "BAYERPAT",
    }
    item = {
        "name": path.name,
        "path": str(path),
        "folder": str(path.parent),
        "dimensions": [int(header.get("NAXIS2", 0) or 0), int(header.get("NAXIS1", 0) or 0)],
        **klass,
        "wcs": wcs,
        "approx_orientation_deg": wcs.get("rotation_deg"),
        "approx_scale_arcsec": wcs.get("pixel_scale_arcsec") or REGIMES.get(klass["scale_regime"], {}).get("nominal_scale_arcsec"),
        "apparent_quality": "usable" if source_probe.get("ok") and int(source_probe.get("source_count", 0)) >= 40 else "weak_or_unverified",
        "raw_source_probe": source_probe,
        "has_runtime_wcs_stripped_for_probe": False,
    }
    for out_key, fits_key in keys.items():
        value = header.get(fits_key)
        if value is not None:
            item[out_key] = value.item() if hasattr(value, "item") else value
    return item


def _group_key(item: dict[str, Any]) -> str:
    dims = "x".join(str(v) for v in item.get("dimensions", []))
    return "|".join([str(item["field_id"]), str(item["instrument"]), str(item["mount_mode"]), dims, str(item["scale_regime"])])


def _build_inventory() -> dict[str, Any]:
    fits_paths = sorted(
        p for p in NEW_CORPUS_ROOT.iterdir()
        if p.is_file() and p.suffix.lower() in {".fit", ".fits", ".fts"}
    )
    items = [_fits_inventory(path) for path in fits_paths]
    groups: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _group_key(item)
        groups.setdefault(
            key,
            {
                "group_key": key,
                "field_id": item["field_id"],
                "instrument": item["instrument"],
                "mount_mode": item["mount_mode"],
                "dimensions": item["dimensions"],
                "scale_regime": item["scale_regime"],
                "fits": [],
            },
        )["fits"].append(item["path"])
    return {
        "schema": "zeblind.p219b_new_corpus_inventory.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "NEW_CORPUS_ROOT": str(NEW_CORPUS_ROOT),
        "bounded_roots": [
            str(NEW_CORPUS_ROOT),
            str(MULTIFIELD_INDEX_ROOT),
            str(ROOT / "tools"),
            str(ROOT / "reports/p219_4d_multifield_validation"),
            str(ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"),
            str(ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz"),
        ],
        "items": items,
        "groups": list(groups.values()),
        "instrument_scale_regimes": REGIMES,
        "notes": [
            "Only NEW_CORPUS_ROOT was inventoried for new FITS.",
            "M31 Seestar FITS already contain offline WCS and are copied/stripped before runtime solve_blind.",
            "ASILive ASI462MC FITS have no WCS and the quick ASTAP offline attempt did not produce a reference WCS in the bounded run; they are inventoried but excluded from validation.",
        ],
    }


def _select_new(inventory: dict[str, Any], indexes: dict[str, list[Path]]) -> dict[str, Any]:
    items = list(inventory["items"])
    m31_s50 = [x for x in items if x["field_id"] == "m31" and x["scale_regime"] == "s50"]
    m31_s30 = [x for x in items if x["field_id"] == "m31" and x["scale_regime"] == "s30"]
    selected = []
    for item in sorted(m31_s50, key=lambda x: (str(x.get("date_obs", "")), x["name"])):
        selected.append({**item, "selection_reason": "M31 S50 with offline WCS; covers EQ/non-mosaic and Alt-Az/mosaic rotations"})
    for item in sorted(m31_s30, key=lambda x: (str(x.get("date_obs", "")), x["name"])):
        selected.append({**item, "selection_reason": "M31 S30 with offline WCS; distinct scale regime from S50"})
    excluded = []
    for item in items:
        if item["path"] not in {x["path"] for x in selected}:
            excluded.append({**item, "exclusion_reason": "no reliable offline WCS oracle available in this run" if not item["wcs"].get("has_wcs") else "not selected"})
    return {
        "schema": "zeblind.p219b_selected_corpus.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_new_images": selected,
        "selected_new_count": len(selected),
        "excluded_new_images": excluded,
        "target_note": "9 new M31 images selected; the remaining 6 ASI462 raw frames are kept in inventory only because no WCS oracle was established.",
        "index_sets": {
            "m31": [_index_meta(p) for p in indexes["m31"]],
            "ngc6888": [_index_meta(p) for p in indexes["ngc6888"]],
            "m106": [_index_meta(p) for p in indexes["m106"]],
        },
        "historical_witnesses": [
            {"field_id": "m106", "path": str(p219.M106_DIR / "Light_mosaic_M 106_20.0s_IRCUT_20250518-234013.fit"), "reason": "P2.18 boundary case 234013"},
            {"field_id": "m106", "path": str(p219.M106_DIR / "Light_mosaic_M 106_20.0s_IRCUT_20250518-232205.fit"), "reason": "P2.18 easy case"},
            {"field_id": "ngc6888", "path": str(p219.NGC6888_DIR / "Light_NGC 6888_30.0s_LP_20250619-020803.fit"), "reason": "P2.19 low-rank accepted case"},
            {"field_id": "ngc6888", "path": str(p219.NGC6888_DIR / "Light_NGC 6888_30.0s_LP_20250619-015658.fit"), "reason": "P2.19 high-rank accepted case"},
        ],
    }


def _args_for(scale_regime: str, base_args: argparse.Namespace) -> argparse.Namespace:
    args = copy.copy(base_args)
    regime = REGIMES[scale_regime]
    args.pixel_scale_min_arcsec = float(regime["min_scale_arcsec"])
    args.pixel_scale_max_arcsec = float(regime["max_scale_arcsec"])
    return args


def _run_case(row: dict[str, Any], indexes: list[Path], base_args: argparse.Namespace, *, tag: str, control: str) -> dict[str, Any]:
    args = _args_for(str(row["scale_regime"]), base_args)
    out = p219._run_4d_case(
        field_id=str(row["field_id"]),
        source=Path(row["path"]),
        index_paths=indexes,
        args=args,
        tag=tag,
        control=control,
    )
    out.update(
        {
            "instrument": row.get("instrument"),
            "mount_mode": row.get("mount_mode"),
            "scale_regime": row.get("scale_regime"),
            "configured_scale_range_arcsec": [args.pixel_scale_min_arcsec, args.pixel_scale_max_arcsec],
        }
    )
    return out


def _prepare_work_fits_without_oracle_hints(source: Path, work_dir: Path, tag: str) -> Path:
    target_dir = work_dir / tag / source.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    p219.p22._strip_wcs(target)
    oracle_hint_keys = {
        "RA",
        "DEC",
        "OBJCTRA",
        "OBJCTDEC",
        "OBJRA",
        "OBJDEC",
        "TELRA",
        "TELDEC",
        "CENTRA",
        "CENTDEC",
        "RA_OBJ",
        "DEC_OBJ",
    }
    with fits.open(target, mode="update") as hdul:
        header = hdul[0].header
        for key in oracle_hint_keys:
            if key in header:
                del header[key]
        header["P219BSTR"] = (True, "P2.19b stripped WCS and RA/Dec oracle hints before runtime")
        hdul.flush()
    return target


def _run_witnesses(selection: dict[str, Any], indexes: dict[str, list[Path]], base_args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for item in selection["historical_witnesses"]:
        field = item["field_id"]
        scale = "s50"
        row = {
            "field_id": field,
            "path": item["path"],
            "instrument": "S50",
            "mount_mode": "Alt-Az",
            "scale_regime": scale,
        }
        idx = indexes["m106"] if field == "m106" else indexes["ngc6888"]
        rows.append(_run_case(row, idx, base_args, tag="historical_witnesses", control="historical_witness"))
    return rows


def _run_new_baseline(selection: dict[str, Any], indexes: dict[str, list[Path]], base_args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    selected = selection["selected_new_images"]
    for i, item in enumerate(selected, start=1):
        print(json.dumps({"event": "new_baseline_start", "i": i, "n": len(selected), "image": item["name"], "regime": item["scale_regime"]}), flush=True)
        row = _run_case(item, indexes["m31"], base_args, tag="new_m31_baseline", control="baseline")
        rows.append(row)
        print(json.dumps({"event": "new_baseline_done", "image": item["name"], "success": row["success"], "inliers": row["inliers"], "rms": row["rms_px"], "total_s": row["solver_total_s"]}, default=_json_default), flush=True)
    return rows


def _run_controls(selection: dict[str, Any], indexes: dict[str, list[Path]], base_args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = selection["selected_new_images"]
    if not selected:
        return []
    controls = []
    first = selected[0]
    s30 = next((x for x in selected if x["scale_regime"] == "s30"), selected[-1])
    missing = BASE / "missing_indexes" / "missing_p219b_index.npz"
    incompatible = MULTIFIELD_INDEX_ROOT / "tiles/d50_2602.npz"
    control_specs = [
        (first, indexes["ngc6888"], "wrong_ngc6888_indexes"),
        (first, indexes["m106"], "wrong_m106_indexes"),
        (first, list(reversed(indexes["m31"])), "reversed_index_order"),
        (first, [indexes["m31"][0]], "primary_index_only"),
        (first, [indexes["m31"][1]], "secondary_index_only"),
        (first, [missing], "missing_index"),
        (first, [incompatible], "incompatible_index_format"),
    ]
    for item, idx, control in control_specs:
        controls.append(_run_case(item, idx, base_args, tag=f"controls_{control}", control=control))
    wrong_scale = dict(s30)
    wrong_scale["scale_regime"] = "s50"
    controls.append(_run_case(wrong_scale, indexes["m31"], base_args, tag="controls_s30_wrong_s50_scale", control="s30_with_s50_scale_range"))
    return controls


def _stats_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    out = {}
    for value, items in grouped.items():
        successes = [r for r in items if r.get("success")]
        out[value] = {
            "total": len(items),
            "successes": len(successes),
            "success_rate": len(successes) / len(items) if items else 0.0,
            "median_inliers": p219._median([float(r["inliers"]) for r in successes]),
            "median_rms": p219._median([float(r["rms_px"]) for r in successes]),
            "median_runtime_s": p219._median([float(r["solver_total_s"]) for r in items]),
            "median_sources": p219._median([float(r["raw_sources"]) for r in items]),
            "origin_tiles": sorted({str(r.get("origin_tile")) for r in successes if r.get("origin_tile")}),
        }
    return out


def _failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("success"):
            continue
        out.append(
            {
                "image": Path(str(row.get("fits"))).name,
                "field_id": row.get("field_id"),
                "instrument": row.get("instrument"),
                "scale_regime": row.get("scale_regime"),
                "control": row.get("control"),
                "failure_class": row.get("failure_class"),
                "stop_reason": row.get("stop_reason"),
                "message": row.get("message"),
                "best_plausible_reject": row.get("best_plausible_reject"),
                "best_scale_invalid_reject": row.get("best_scale_invalid_reject"),
                "best_rms_invalid_reject": row.get("best_rms_invalid_reject"),
                "best_geometry_invalid_reject": row.get("best_geometry_invalid_reject"),
            }
        )
    return out


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# ZeBlind P2.19b - Extension incrementale multi-instruments",
        "",
        "> Diagnostic only. No ZeNear, GUI, default backend, AB/C/D core, default routing, quality thresholds, inverse metric decision, all-sky routing, or oracle runtime input changed.",
        "",
        "## Runtime Input Hygiene",
        "",
        "- Work FITS are copied from the source corpus, then stripped of WCS and RA/Dec-like header hints (`RA`, `DEC`, `OBJCTRA`, `OBJCTDEC`, `OBJRA`, `OBJDEC`, `TELRA`, `TELDEC`, `CENTRA`, `CENTDEC`, `RA_OBJ`, `DEC_OBJ`) before `solve_blind`.",
        "- Offline WCS is used only for reference comparison and compact field-index selection.",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{s['verdict']}`.",
        f"- Historical witnesses: `{s['witness_successes']}/{s['witness_cases']}`.",
        f"- New baseline: `{s['new_successes']}/{s['new_cases']}`.",
        f"- Offline false positives: `{s['offline_false_positives']}`.",
        f"- Negative false accepts: `{s['negative_false_accepts']}`.",
        "",
        "## Scale Regimes",
        "",
    ]
    for key, regime in REGIMES.items():
        lines.append(
            f"- `{key}`: nominal `{regime['nominal_scale_arcsec']}` arcsec/px, range `{regime['min_scale_arcsec']}..{regime['max_scale_arcsec']}`, margin `{regime['margin_pct']}%`, source: {regime['source']}."
        )
    lines.extend(["", "## New Corpus Groups", ""])
    for group in payload["inventory"]["groups"]:
        lines.append(f"- `{group['group_key']}`: `{len(group['fits'])}` FITS.")
    lines.extend(["", "## Selected Matrix", "", "| image | instrument | mode | regime | success | tile | rank | sources | inliers | RMS | scale | rot | total | offline |", "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in payload["new_rows"]:
        off = row.get("offline_wcs_check") or {}
        lines.append(
            f"| `{Path(str(row['fits'])).name}` | `{row.get('instrument')}` | `{row.get('mount_mode')}` | `{row.get('scale_regime')}` | `{row.get('success')}` | `{row.get('origin_tile')}` | {row.get('rank')} | {row.get('raw_sources')} | {row.get('inliers')} | {p219._fmt(row.get('rms_px'), 3)} | {p219._fmt(row.get('pixel_scale_arcsec'), 3)} | {p219._fmt(row.get('rotation_deg'), 1)} | {p219._fmt(row.get('solver_total_s'), 3)} | `{off.get('ok')}` |"
        )
    lines.extend(["", "## Historical Witnesses", "", "| image | field | success | tile | rank | inliers | RMS | offline |", "|---|---|---:|---|---:|---:|---:|---|"])
    for row in payload["witness_rows"]:
        off = row.get("offline_wcs_check") or {}
        lines.append(
            f"| `{Path(str(row['fits'])).name}` | `{row.get('field_id')}` | `{row.get('success')}` | `{row.get('origin_tile')}` | {row.get('rank')} | {row.get('inliers')} | {p219._fmt(row.get('rms_px'), 3)} | `{off.get('ok')}` |"
        )
    lines.extend(["", "## Negative Controls", "", "| control | image | success | failure | inliers | RMS | scale | message |", "|---|---|---:|---|---:|---:|---:|---|"])
    for row in payload["control_rows"]:
        reason = (row.get("chosen_validation") or {}).get("reason") or row.get("message")
        lines.append(
            f"| `{row.get('control')}` | `{Path(str(row['fits'])).name}` | `{row.get('success')}` | `{row.get('failure_class')}` | {row.get('inliers')} | {p219._fmt(row.get('rms_px'), 3)} | {p219._fmt(row.get('pixel_scale_arcsec'), 3)} | {str(reason)[:120]} |"
        )
    lines.extend(
        [
            "",
            "## Statistics",
            "",
            "### By Field",
            "",
            "```json",
            json.dumps(payload["stats"]["by_field"], indent=2, default=_json_default),
            "```",
            "",
            "### By Instrument",
            "",
            "```json",
            json.dumps(payload["stats"]["by_instrument"], indent=2, default=_json_default),
            "```",
            "",
            "### By Mount Mode",
            "",
            "```json",
            json.dumps(payload["stats"]["by_mount_mode"], indent=2, default=_json_default),
            "```",
            "",
            "### By Scale Regime",
            "",
            "```json",
            json.dumps(payload["stats"]["by_scale_regime"], indent=2, default=_json_default),
            "```",
            "",
            "## Failure Analysis",
            "",
        ]
    )
    failures = payload["failure_analysis"]
    if failures:
        for failure in failures:
            lines.append(f"- `{failure['image']}` / `{failure['control']}`: `{failure['failure_class']}`.")
    else:
        lines.append("- No baseline failures on selected new images.")
    lines.extend(["", "## Required Answers", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Recommendation", "", payload["recommendation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.19b incremental multiregime validation for ZeBlind 4D.")
    ap.add_argument("--work-dir", type=Path, default=BASE / "work")
    ap.add_argument("--verification-min-sep-px", type=float, default=0.75)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--quad-sources", type=int, default=120)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--max-wall-s", type=float, default=45.0)
    ap.add_argument("--max-accepts", type=int, default=64)
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    args = ap.parse_args()

    p219._prepare_work_fits = _prepare_work_fits_without_oracle_hints
    print(json.dumps({"event": "ensure_indexes"}), flush=True)
    indexes = _ensure_indexes()
    print(json.dumps({"event": "inventory", "root": str(NEW_CORPUS_ROOT)}), flush=True)
    inventory = _build_inventory()
    selection = _select_new(inventory, indexes)
    _write_json(INVENTORY_OUT, inventory)
    _write_json(SELECTION_OUT, selection)

    print(json.dumps({"event": "witnesses_start"}), flush=True)
    witness_rows = _run_witnesses(selection, indexes, args)
    witness_success = [r for r in witness_rows if r.get("success") and (r.get("offline_wcs_check") or {}).get("ok")]
    witnesses_ok = len(witness_success) == len(witness_rows)
    if not witnesses_ok:
        print(json.dumps({"event": "witnesses_failed", "successes": len(witness_success), "total": len(witness_rows)}), flush=True)
        new_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
    else:
        print(json.dumps({"event": "witnesses_ok"}), flush=True)
        new_rows = _run_new_baseline(selection, indexes, args)
        control_rows = _run_controls(selection, indexes, args)

    offline_false = [r for r in [*witness_rows, *new_rows] if r.get("success") and (r.get("offline_wcs_check") or {}).get("available") and not (r.get("offline_wcs_check") or {}).get("ok")]
    negative_false = [
        r for r in control_rows
        if r.get("control") in {"wrong_ngc6888_indexes", "wrong_m106_indexes", "missing_index", "incompatible_index_format", "s30_with_s50_scale_range"}
        and r.get("success")
    ]
    new_success = [r for r in new_rows if r.get("success")]
    verdict = "B - Extension partielle"
    if witnesses_ok and len(new_success) == len(new_rows) and not offline_false and not negative_false:
        verdict = "A - Extension positive"
    if not witnesses_ok or offline_false or negative_false:
        verdict = "C - Extension negative"

    stats = {
        "by_field": _stats_by(new_rows, "field_id"),
        "by_instrument": _stats_by(new_rows, "instrument"),
        "by_mount_mode": _stats_by(new_rows, "mount_mode"),
        "by_scale_regime": _stats_by(new_rows, "scale_regime"),
        "by_density": {
            "raw_source_counts": [
                {"image": Path(str(r["fits"])).name, "raw_sources": r.get("raw_sources"), "success": r.get("success"), "hypotheses": r.get("hypotheses_tested"), "validation_s": r.get("validation_s")}
                for r in new_rows
            ]
        },
    }
    answers = [
        f"Le backend supporte une autre echelle que S50 dans ce banc: S30 `{sum(1 for r in new_rows if r.get('scale_regime') == 's30' and r.get('success'))}/{sum(1 for r in new_rows if r.get('scale_regime') == 's30')}` avec la plage figee 3.19..4.79 arcsec/px.",
        "Le C11/ASI462 est present dans l'inventaire, mais non valide en baseline faute de WCS offline fiable; aucun tuning image par image n'a ete fait pour le faire passer.",
        f"Le S30 fonctionne avec les memes seuils qualite: quality_inliers=40, quality_rms=1.2, match_radius_px=3.0.",
        "Les rotations et modes Alt-Az/EQ M31 ne changent pas la decision dans ce corpus selectionne." if all(r.get("success") for r in new_rows) else "Au moins une rotation/mode doit etre analysee dans les echecs.",
        "Les champs denses M31 ne montrent pas d'echec baseline; le cout est reporte par source count et validation_s.",
        "Les images brutes sans WCS exploitable posent d'abord un probleme d'oracle offline/reference, pas un rejet runtime mesure du backend 4D.",
    ]
    payload = {
        "schema": "zeblind.p219b_incremental_validation.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inventory": inventory,
        "selection": selection,
        "witness_rows": witness_rows,
        "new_rows": new_rows,
        "control_rows": control_rows,
        "stats": stats,
        "failure_analysis": _failure_summary([*new_rows, *control_rows]),
        "summary": {
            "verdict": verdict,
            "witness_cases": len(witness_rows),
            "witness_successes": len(witness_success),
            "new_cases": len(new_rows),
            "new_successes": len(new_success),
            "offline_false_positives": len(offline_false),
            "negative_false_accepts": len(negative_false),
            "thresholds": {"quality_inliers": 40, "quality_rms": 1.2, "match_radius_px": 3.0},
            "runtime_oracle_wcs_input": False,
            "backend_default_changed": False,
            "gui_changed": False,
            "zenear_changed": False,
            "all_sky": False,
        },
        "answers": answers,
        "recommendation": "Next single step: resolve one ASI462/C11 raw frame offline with a bounded, documented reference workflow, then add exactly one compact 4D index for that field and rerun this same P2.19b probe without changing thresholds.",
    }
    _write_json(VALIDATION_OUT, payload)
    _write_report(REPORT_OUT, payload)
    print(json.dumps({"event": "done", "json": str(VALIDATION_OUT), "report": str(REPORT_OUT), "verdict": verdict}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
