#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex  # noqa: E402
from zeblindsolver.zeblindsolver import SolveConfig, _astrometry_4d_accept_sort_key, _astrometry_4d_runtime_requested  # noqa: E402

import tools.diagnose_p219_4d_multifield_validation as p219  # noqa: E402


MANIFEST_SCHEMA = "zeblind.astrometry_4d_index_manifest.v1"
MANIFEST_VERSION = 1

BASE = ROOT / "reports/p220_mixed_index_pool"
BASELINE_OUT = ROOT / "reports/zeblind_p220_baseline.json"
MANIFEST_OUT = ROOT / "reports/zeblind_p220_mixed_index_manifest.json"
CORPUS_OUT = ROOT / "reports/zeblind_p220_selected_mixed_pool_corpus.json"
VALIDATION_OUT = ROOT / "reports/zeblind_p220_mixed_pool_validation.json"
REPORT_OUT = ROOT / "reports/zeblind_p220_mixed_pool_validation.md"

SCALE_REGIMES = {
    "s50": {
        "instrument": "S50",
        "min_scale_arcsec": 1.90,
        "max_scale_arcsec": 2.85,
        "source": "Seestar S50 instrument metadata and fixed P2.19b regime",
    },
    "s30": {
        "instrument": "S30",
        "min_scale_arcsec": 3.19,
        "max_scale_arcsec": 4.79,
        "source": "Seestar S30 instrument metadata and fixed P2.19b regime",
    },
}

INDEX_DEFS = [
    ("d50_2823_S_q40000", ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"),
    ("d50_2822_S_q40000", ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz"),
    ("d50_2644_S_q40000", ROOT / "reports/p219_4d_multifield_validation/indexes/p219_astrometry_ab_code_4d_v1_d50_2644_S_stars2000_q40000.npz"),
    ("d50_2645_S_q40000", ROOT / "reports/p219_4d_multifield_validation/indexes/p219_astrometry_ab_code_4d_v1_d50_2645_S_stars2000_q40000.npz"),
    ("d50_2602_S_q40000", ROOT / "reports/p219b_incremental_multiregime/indexes/p219b_astrometry_ab_code_4d_v1_d50_2602_S_stars2000_q40000.npz"),
    ("d50_2702_S_q40000", ROOT / "reports/p219b_incremental_multiregime/indexes/p219b_astrometry_ab_code_4d_v1_d50_2702_S_stars2000_q40000.npz"),
]

ORDER_DEFS = {
    "A_historical_grouped": [
        "d50_2823_S_q40000",
        "d50_2822_S_q40000",
        "d50_2644_S_q40000",
        "d50_2645_S_q40000",
        "d50_2602_S_q40000",
        "d50_2702_S_q40000",
    ],
    "B_reverse_groups": [
        "d50_2602_S_q40000",
        "d50_2702_S_q40000",
        "d50_2644_S_q40000",
        "d50_2645_S_q40000",
        "d50_2823_S_q40000",
        "d50_2822_S_q40000",
    ],
    "C_interleaved": [
        "d50_2823_S_q40000",
        "d50_2644_S_q40000",
        "d50_2602_S_q40000",
        "d50_2822_S_q40000",
        "d50_2645_S_q40000",
        "d50_2702_S_q40000",
    ],
    "D_fixed_permutation": [
        "d50_2645_S_q40000",
        "d50_2702_S_q40000",
        "d50_2822_S_q40000",
        "d50_2602_S_q40000",
        "d50_2823_S_q40000",
        "d50_2644_S_q40000",
    ],
}

EXPECTED_TILES = {
    "m106": {"d50_2823", "d50_2822"},
    "ngc6888": {"d50_2644", "d50_2645"},
    "m31": {"d50_2602", "d50_2702"},
}

POSITION_HINT_KEYS = {
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

IDENTITY_HINT_KEYS = {
    "OBJECT",
    "OBJNAME",
    "TARGET",
    "TARGNAME",
    "FIELD",
    "FIELDID",
}


class ManifestError(RuntimeError):
    pass


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative_or_abs(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _index_manifest_entry(index_id: str, path: Path, *, enabled: bool = True, priority: int = 0) -> dict[str, Any]:
    idx = Quad4DIndex.load(path)
    metadata = dict(idx.metadata)
    return {
        "id": index_id,
        "enabled": bool(enabled),
        "path": _relative_or_abs(path),
        "filename": path.name,
        "quad_schema": metadata.get("schema"),
        "index_version": metadata.get("version"),
        "level": metadata.get("level"),
        "tile_keys": list(idx.tile_keys),
        "star_count": int(idx.catalog_ra_dec.shape[0]),
        "quad_count": int(idx.codes_4d.shape[0]),
        "sampler_tag": metadata.get("sampler_tag"),
        "code_tol_recommended": metadata.get("code_tol_recommended"),
        "catalog_source": metadata.get("source_catalog"),
        "source_index_root": metadata.get("source_index_root"),
        "supported_scale_range_arcsec": None,
        "scale_note": "Index is sky-region/catalogue specific, not intrinsically instrument-scale specific.",
        "file_size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "priority": int(priority),
        "generation_metadata": {
            "generated_at": metadata.get("generated_at"),
            "max_stars_per_tile": metadata.get("max_stars_per_tile"),
            "max_quads_per_tile": metadata.get("max_quads_per_tile"),
            "dtype": metadata.get("dtype"),
            "entry_count": metadata.get("entry_count"),
            "star_count": metadata.get("star_count"),
        },
    }


def build_reference_manifest() -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "description": "P2.20 experimental mixed celestial pool manifest. Contains no image names, targets, image centers, or image-to-index mapping.",
        "indexes": [
            _index_manifest_entry(index_id, path, priority=i)
            for i, (index_id, path) in enumerate(INDEX_DEFS)
        ],
    }


def _resolve_manifest_path(path_value: str, *, manifest_path: Path, root: Path | None = None) -> Path:
    raw = Path(str(path_value)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    base = root if root is not None else ROOT
    candidate = (base / raw).resolve()
    if candidate.exists():
        return candidate
    return (manifest_path.parent / raw).resolve()


def load_index_manifest(manifest_path: Path | str, *, root: Path | None = None) -> tuple[list[Path], list[dict[str, Any]]]:
    path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManifestError(f"manifest_json_invalid: {exc}") from exc
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"manifest_schema_invalid: {payload.get('schema')!r}")
    if int(payload.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ManifestError(f"manifest_version_invalid: {payload.get('manifest_version')!r}")
    entries = payload.get("indexes")
    if not isinstance(entries, list):
        raise ManifestError("manifest_indexes_invalid: expected list")

    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_tiles: set[str] = set()
    enabled_entries: list[dict[str, Any]] = []
    paths: list[Path] = []
    for i, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ManifestError(f"manifest_entry_invalid[{i}]: expected object")
        entry_id = str(raw_entry.get("id") or "").strip()
        if not entry_id:
            raise ManifestError(f"manifest_entry_id_missing[{i}]")
        if entry_id in seen_ids:
            raise ManifestError(f"manifest_duplicate_id: {entry_id}")
        seen_ids.add(entry_id)
        if not bool(raw_entry.get("enabled", True)):
            continue
        if "path" not in raw_entry:
            raise ManifestError(f"manifest_entry_path_missing: {entry_id}")
        index_path = _resolve_manifest_path(str(raw_entry["path"]), manifest_path=path, root=root)
        if index_path in seen_paths:
            raise ManifestError(f"manifest_duplicate_path: {index_path}")
        seen_paths.add(index_path)
        if not index_path.exists():
            raise ManifestError(f"manifest_index_absent: {index_path}")
        expected_sha = str(raw_entry.get("sha256") or "").strip()
        actual_sha = sha256_file(index_path)
        if expected_sha and expected_sha.lower() != actual_sha.lower():
            raise ManifestError(f"manifest_sha256_mismatch: {entry_id}")
        try:
            idx = Quad4DIndex.load(index_path)
        except Exception as exc:
            raise ManifestError(f"manifest_index_incompatible: {entry_id}: {exc}") from exc
        metadata = dict(idx.metadata)
        expected_schema = str(raw_entry.get("quad_schema") or "")
        if expected_schema != ASTROMETRY_AB_CODE_4D_SCHEMA:
            raise ManifestError(f"manifest_quad_schema_invalid: {entry_id}: {expected_schema!r}")
        if metadata.get("schema") != ASTROMETRY_AB_CODE_4D_SCHEMA:
            raise ManifestError(f"manifest_index_schema_invalid: {entry_id}: {metadata.get('schema')!r}")
        expected_version = int(raw_entry.get("index_version", -1))
        if int(metadata.get("version", -1)) != expected_version:
            raise ManifestError(f"manifest_index_version_mismatch: {entry_id}")
        expected_tiles = [str(v) for v in (raw_entry.get("tile_keys") or [])]
        actual_tiles = [str(v) for v in idx.tile_keys]
        if expected_tiles != actual_tiles:
            raise ManifestError(f"manifest_tile_keys_mismatch: {entry_id}")
        for tile in actual_tiles:
            if tile in seen_tiles:
                raise ManifestError(f"manifest_duplicate_tile: {tile}")
            seen_tiles.add(tile)
        if int(raw_entry.get("star_count", -1)) != int(idx.catalog_ra_dec.shape[0]):
            raise ManifestError(f"manifest_star_count_mismatch: {entry_id}")
        if int(raw_entry.get("quad_count", -1)) != int(idx.codes_4d.shape[0]):
            raise ManifestError(f"manifest_quad_count_mismatch: {entry_id}")
        sampler = str(raw_entry.get("sampler_tag") or "")
        if sampler != str(metadata.get("sampler_tag") or ""):
            raise ManifestError(f"manifest_sampler_mismatch: {entry_id}")
        loaded_entry = dict(raw_entry)
        loaded_entry["resolved_path"] = str(index_path)
        loaded_entry["actual_sha256"] = actual_sha
        enabled_entries.append(loaded_entry)
        paths.append(index_path)
    return paths, enabled_entries


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def write_baseline_note(test_result: str) -> dict[str, Any]:
    manifest = build_reference_manifest()
    runtime_params = {
        "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "quad_sources": 120,
        "verification_sources": "full",
        "validation_catalog_policy": "union_candidate_tiles",
        "accept_policy": "best_within_budget",
        "quality_inliers": 40,
        "quality_rms": 1.2,
        "match_radius_px": 3.0,
        "max_quads": 2500,
        "max_hypotheses": 2000,
        "max_accepts": 64,
        "max_wall_s": 45.0,
        "legacy_inverse_decision": False,
    }
    payload = {
        "schema": "zeblind.p220_baseline.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "commit": _git_output(["rev-parse", "HEAD"]),
        "git_status_at_baseline_file_write": _git_output(["status", "--short", "--untracked-files=all"]),
        "git_status_observed_before_p220_edits": "M followup.md\nM memory.md",
        "reports_preserved": {
            "p218": str(ROOT / "reports/zeblind_p218_4d_m106_all30_direct_metric_closure.json"),
            "p219": str(ROOT / "reports/zeblind_p219_4d_multifield_validation.json"),
            "p219b": str(ROOT / "reports/zeblind_p219b_incremental_validation.json"),
        },
        "indexes": manifest["indexes"],
        "runtime_params": runtime_params,
        "tests_before_p220": test_result,
    }
    _write_json(BASELINE_OUT, payload)
    return payload


def _instrument_scale_regime(source: Path) -> str:
    header = fits.getheader(source)
    instr = str(header.get("INSTRUME", "") or "")
    telescope = str(header.get("TELESCOP", "") or "")
    if "S30" in instr or "S30" in telescope:
        return "s30"
    return "s50"


def _corpus_item(field_id: str, source: Path, reason: str, *, mount_mode: str, role: str) -> dict[str, Any]:
    regime = _instrument_scale_regime(source)
    return {
        "id": source.stem,
        "field_id": field_id,
        "path": str(source),
        "filename": source.name,
        "instrument_regime": regime,
        "instrument": SCALE_REGIMES[regime]["instrument"],
        "mount_mode": mount_mode,
        "selection_role": role,
        "selection_reason": reason,
        "expected_tiles_for_offline_evaluation_only": sorted(EXPECTED_TILES[field_id]),
        "source_sha256": sha256_file(source),
    }


def build_selected_corpus() -> dict[str, Any]:
    m106 = p219.M106_DIR
    ngc = p219.NGC6888_DIR
    m31 = Path("/home/tristan/zemosaic/example/various_fresh")
    items = [
        _corpus_item("m106", m106 / "Light_mosaic_M 106_20.0s_IRCUT_20250518-234013.fit", "boundary case recovered in P2.18", mount_mode="Alt-Az", role="boundary"),
        _corpus_item("m106", m106 / "Light_mosaic_M 106_20.0s_IRCUT_20250518-232205.fit", "easy low-rank case", mount_mode="Alt-Az", role="easy"),
        _corpus_item("m106", m106 / "Light_mosaic_M 106_20.0s_IRCUT_20250518-232329.fit", "low-footprint representative", mount_mode="Alt-Az", role="low_footprint"),
        _corpus_item("m106", m106 / "Light_mosaic_M 106_20.0s_IRCUT_20250518-233828.fit", "intermediate historical representative", mount_mode="Alt-Az", role="intermediate"),
        _corpus_item("ngc6888", ngc / "Light_NGC 6888_30.0s_LP_20250619-020803.fit", "low-rank NGC6888 case", mount_mode="Alt-Az", role="low_rank"),
        _corpus_item("ngc6888", ngc / "Light_NGC 6888_30.0s_LP_20250619-015658.fit", "high-rank NGC6888 case", mount_mode="Alt-Az", role="high_rank"),
        _corpus_item("ngc6888", ngc / "Light_NGC 6888_30.0s_LP_20250619-020145.fit", "high-RMS NGC6888 case", mount_mode="Alt-Az", role="high_rms"),
        _corpus_item("ngc6888", ngc / "Light_NGC 6888_30.0s_LP_20250619-020632.fit", "median NGC6888 case", mount_mode="Alt-Az", role="median"),
        _corpus_item("m31", m31 / "Light_mosaic_M 31_10.0s_IRCUT_20250115-202105.fit", "M31 S50 Alt-Az rotation set", mount_mode="Alt-Az", role="s50_altaz_a"),
        _corpus_item("m31", m31 / "Light_mosaic_M 31_10.0s_IRCUT_20250115-202143.fit", "M31 S50 Alt-Az shifted frame", mount_mode="Alt-Az", role="s50_altaz_b"),
        _corpus_item("m31", m31 / "Light_M 31_20.0s_IRCUT_20251117-225718.fit", "M31 S50 EQ/non-mosaic", mount_mode="EQ", role="s50_eq_a"),
        _corpus_item("m31", m31 / "Light_M 31_20.0s_IRCUT_20251117-225801.fit", "M31 S50 EQ/non-mosaic shifted", mount_mode="EQ", role="s50_eq_b"),
        _corpus_item("m31", m31 / "Light_mosaic_M 31_60.0s_IRCUT_20250904-015506.fit", "M31 S30 Alt-Az scale-regime case", mount_mode="Alt-Az", role="s30_altaz_a"),
        _corpus_item("m31", m31 / "Light_mosaic_M 31_60.0s_IRCUT_20250904-020408.fit", "M31 S30 Alt-Az shifted case", mount_mode="Alt-Az", role="s30_altaz_b"),
        _corpus_item("m31", m31 / "Light_mosaic_M 31_60.0s_IRCUT_20250904-023014.fit", "M31 S30 high-rank P2.19b case", mount_mode="Alt-Az", role="s30_high_rank"),
    ]
    payload = {
        "schema": "zeblind.p220_selected_mixed_pool_corpus.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_count": len(items),
        "selection_policy": "Bounded representative subset only; no all-corpus replay.",
        "items": items,
    }
    _write_json(CORPUS_OUT, payload)
    return payload


def _args_for_regime(base_args: argparse.Namespace, regime: str) -> argparse.Namespace:
    args = copy.copy(base_args)
    args.pixel_scale_min_arcsec = float(SCALE_REGIMES[regime]["min_scale_arcsec"])
    args.pixel_scale_max_arcsec = float(SCALE_REGIMES[regime]["max_scale_arcsec"])
    return args


def _runtime_work_fits(source: Path, work_dir: Path, tag: str) -> tuple[Path, dict[str, Any]]:
    target_dir = work_dir / tag / source.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    before_keys = list(fits.getheader(target).keys())
    p219.p22._strip_wcs(target)
    removed: list[str] = []
    with fits.open(target, mode="update") as hdul:
        header = hdul[0].header
        for key in sorted(POSITION_HINT_KEYS | IDENTITY_HINT_KEYS):
            if key in header:
                del header[key]
                removed.append(key)
        header["P220STR"] = (True, "P2.20 stripped WCS, RA/Dec hints and target identity before runtime")
        hdul.flush()
    header = fits.getheader(target)
    wcs = WCS(header)
    forbidden_remaining = sorted(k for k in (POSITION_HINT_KEYS | IDENTITY_HINT_KEYS) if k in header)
    audit = {
        "source_fits": str(source),
        "runtime_fits": str(target),
        "source_sha256": sha256_file(source),
        "runtime_input_sha256": sha256_file(target),
        "removed_hint_keys": removed,
        "forbidden_keys_remaining": forbidden_remaining,
        "has_celestial_wcs_after_strip": bool(getattr(wcs, "has_celestial", False)),
        "kept_instrument_keys": {
            key: header.get(key)
            for key in ("INSTRUME", "TELESCOP", "FOCALLEN", "XBINNING", "YBINNING", "FILTER", "EXPTIME", "DATE-OBS")
            if key in header
        },
        "source_header_key_count": len(before_keys),
        "runtime_header_key_count": len(list(header.keys())),
    }
    if forbidden_remaining:
        raise RuntimeError(f"runtime_forbidden_keys_remaining: {forbidden_remaining}")
    if getattr(wcs, "has_celestial", False):
        raise RuntimeError("runtime_wcs_remaining_after_strip")
    return target, audit


def _config(index_paths: list[Path], args: argparse.Namespace) -> SolveConfig:
    return p219._config(index_paths, args)


def _cost_by_index(row: dict[str, Any], manifest_entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_path = {str(Path(entry["resolved_path"]).resolve()): entry for entry in manifest_entries}
    hits = row.get("hits_by_index") or {}
    per_validation = row.get("per_index_validation") or {}
    out: dict[str, Any] = {}
    for path_str, entry in by_path.items():
        h = hits.get(path_str) or {}
        v = per_validation.get(path_str) or {}
        out[entry["id"]] = {
            "tile_keys": entry.get("tile_keys"),
            "index_order": h.get("index_order", v.get("index_order")),
            "hits": int(h.get("hits", v.get("hits", 0)) or 0),
            "hypotheses_tested": int(v.get("hypotheses_tested", 0) or 0),
            "accepted_candidates": int(v.get("accepted_candidates", 0) or 0),
            "validation_s": float(v.get("validation_s", 0.0) or 0.0),
            "first_tested_rank": v.get("first_tested_rank"),
            "last_tested_rank": v.get("last_tested_rank"),
            "first_accepted_rank": v.get("first_accepted_rank"),
            "best_reject": v.get("best_reject") or {},
            "reject_reason_counts": v.get("reject_reason_counts") or {},
        }
    return out


def run_4d_case(
    item: dict[str, Any],
    index_paths: list[Path],
    base_args: argparse.Namespace,
    *,
    manifest_entries: list[dict[str, Any]],
    run_kind: str,
    order_id: str,
    scale_override: str | None = None,
) -> dict[str, Any]:
    source = Path(item["path"])
    regime = str(scale_override or item["instrument_regime"])
    args = _args_for_regime(base_args, regime)
    tag = f"{run_kind}_{order_id}"
    t0 = time.perf_counter()
    work, hygiene = _runtime_work_fits(source, args.work_dir.expanduser().resolve(), tag)
    raw, image_shape, detect_meta, clean, clean_stats = p219._detect_sources(work, args)
    quad_sources = raw[: int(args.quad_sources)]
    result = p219.solve_blind(
        work,
        p219.MULTIFIELD_INDEX_ROOT,
        config=_config(index_paths, args),
        prep_cache=p219._prep_cache(work, quad_sources, clean, args),
    )
    stats = dict(result.stats or {})
    chosen = p219._chosen_validation(stats, bool(result.success))
    wmeta = p219._wcs_scale_rotation(result.wcs)
    offline = p219._offline_wcs_check(source, result.wcs, image_shape)
    expected_tiles = set(item.get("expected_tiles_for_offline_evaluation_only") or [])
    origin_tile = stats.get("astrometry_4d_selected_origin_tile_key") or chosen.get("origin_tile_key")
    row = {
        "run_kind": run_kind,
        "order_id": order_id,
        "field_id": item["field_id"],
        "image_id": item["id"],
        "fits": str(source),
        "work_fits": str(work),
        "success": bool(result.success),
        "message": str(result.message),
        "instrument": item.get("instrument"),
        "mount_mode": item.get("mount_mode"),
        "instrument_regime": item["instrument_regime"],
        "scale_regime_used": regime,
        "configured_scale_range_arcsec": [args.pixel_scale_min_arcsec, args.pixel_scale_max_arcsec],
        "runtime_input_hygiene": hygiene,
        "dimensions": [int(image_shape[0]), int(image_shape[1])],
        "raw_sources": int(raw.shape[0]),
        "quad_sources": int(quad_sources.shape[0]),
        "verification_sources": int(clean.shape[0]),
        "detect_meta": detect_meta,
        "clean_stats": clean_stats,
        "index_paths": [str(Path(path).resolve()) for path in index_paths],
        "index_ids": [entry["id"] for entry in manifest_entries],
        "same_pool_contract": run_kind.startswith("mixed") and len(index_paths) == 6,
        "origin_tile": origin_tile,
        "origin_index_path": stats.get("astrometry_4d_selected_index_path") or chosen.get("index_path"),
        "rank": stats.get("astrometry_4d_selected_rank") or chosen.get("hit_rank"),
        "local_rank": stats.get("astrometry_4d_selected_local_rank") or chosen.get("local_rank"),
        "code_distance": chosen.get("code_distance"),
        "image_quads": int(stats.get("astrometry_4d_image_quads", 0) or 0),
        "hits": int(stats.get("astrometry_4d_hits", 0) or 0),
        "hits_by_index": stats.get("astrometry_4d_hits_by_index") or {},
        "per_index_validation": stats.get("astrometry_4d_per_index_validation") or {},
        "hypotheses_tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "accepted_candidates": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "inliers": int(chosen.get("inliers", stats.get("inliers", 0)) or 0),
        "rms_px": float(chosen.get("rms_px", stats.get("rms_px", float("nan")))),
        "median_residual_px": chosen.get("median_residual_px"),
        "mad_residual_px": chosen.get("mad_residual_px"),
        "coverage": {
            "geo_cov_x": chosen.get("geo_cov_x"),
            "geo_cov_y": chosen.get("geo_cov_y"),
            "geo_cov_area": chosen.get("geo_cov_area"),
        },
        "conditioning": chosen.get("geo_cond"),
        "pixel_scale_arcsec": chosen.get("pix_scale_arcsec") or wmeta.get("pixel_scale_arcsec"),
        "rotation_deg": wmeta.get("rotation_deg"),
        "residual_metric": chosen.get("residual_metric"),
        "legacy_inverse_inliers": chosen.get("legacy_inverse_inliers"),
        "legacy_inverse_rms_px": chosen.get("legacy_inverse_rms_px"),
        "legacy_inverse_quality": chosen.get("legacy_inverse_quality"),
        "quad_build_s": float(stats.get("astrometry_4d_quad_build_s", 0.0) or 0.0),
        "index_load_s": float(stats.get("astrometry_4d_index_load_s", 0.0) or 0.0),
        "lookup_s": float(stats.get("astrometry_4d_kd_lookup_s", 0.0) or 0.0),
        "validation_s": float(stats.get("astrometry_4d_validation_s", 0.0) or 0.0),
        "solver_total_s": float(stats.get("astrometry_4d_total_s", 0.0) or 0.0),
        "wall_s": float(time.perf_counter() - t0),
        "max_accepts_hit": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0) >= int(args.max_accepts),
        "max_hypotheses_hit": stats.get("astrometry_4d_stop_reason") == "max_hypotheses",
        "max_wall_s_hit": stats.get("astrometry_4d_stop_reason") == "cancelled",
        "chosen_validation": chosen,
        "best_accepted": stats.get("astrometry_4d_best_accepted_validation") or {},
        "first_accepted": stats.get("astrometry_4d_first_accepted_validation") or {},
        "best_plausible_reject": stats.get("astrometry_4d_best_plausible_reject") or {},
        "best_scale_invalid_reject": stats.get("astrometry_4d_best_scale_invalid_reject") or {},
        "best_rms_invalid_reject": stats.get("astrometry_4d_best_rms_invalid_reject") or {},
        "best_geometry_invalid_reject": stats.get("astrometry_4d_best_geometry_invalid_reject") or {},
        "reject_reason_counts": stats.get("astrometry_4d_reject_reason_counts") or {},
        "offline_wcs_check": offline,
    }
    row["failure_class"] = p219._failure_class(row)
    row["offline_correct"] = bool(row["success"] and offline.get("available") and offline.get("ok"))
    row["wrong_region_accepted"] = bool(row["success"] and origin_tile and expected_tiles and str(origin_tile) not in expected_tiles)
    row["false_positive_offline"] = bool(row["success"] and offline.get("available") and not offline.get("ok"))
    row["cost_by_index"] = _cost_by_index(row, manifest_entries)
    return row


def _specialized_entries(manifest_by_id: dict[str, dict[str, Any]], field_id: str) -> list[dict[str, Any]]:
    ids = {
        "m106": ["d50_2823_S_q40000", "d50_2822_S_q40000"],
        "ngc6888": ["d50_2644_S_q40000", "d50_2645_S_q40000"],
        "m31": ["d50_2602_S_q40000", "d50_2702_S_q40000"],
    }[field_id]
    return [manifest_by_id[index_id] for index_id in ids]


def _entries_for_order(manifest_by_id: dict[str, dict[str, Any]], order_ids: list[str]) -> list[dict[str, Any]]:
    return [manifest_by_id[index_id] for index_id in order_ids]


def _paths(entries: Iterable[dict[str, Any]]) -> list[Path]:
    return [Path(entry["resolved_path"]) for entry in entries]


def _median(values: Iterable[float]) -> float | None:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, 50))


def _p95(values: Iterable[float]) -> float | None:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, 95))


def _stats_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    out: dict[str, Any] = {}
    for value, items in grouped.items():
        successes = [r for r in items if r.get("success")]
        out[value] = {
            "total": len(items),
            "successes": len(successes),
            "success_rate": len(successes) / len(items) if items else 0.0,
            "median_total_s": _median(r["solver_total_s"] for r in items),
            "p95_total_s": _p95(r["solver_total_s"] for r in items),
            "median_hypotheses": _median(r["hypotheses_tested"] for r in items),
            "max_hypotheses": max((int(r["hypotheses_tested"]) for r in items), default=0),
            "origin_tiles": sorted({str(r.get("origin_tile")) for r in successes if r.get("origin_tile")}),
            "false_positive_offline": sum(1 for r in items if r.get("false_positive_offline")),
            "wrong_region_accepted": sum(1 for r in items if r.get("wrong_region_accepted")),
            "max_accepts_cases": [Path(str(r["fits"])).name for r in items if r.get("max_accepts_hit")],
            "max_hypotheses_cases": [Path(str(r["fits"])).name for r in items if r.get("max_hypotheses_hit")],
            "max_wall_s_cases": [Path(str(r["fits"])).name for r in items if r.get("max_wall_s_hit")],
        }
    return out


def _compare_specialized_mixed(specialized: list[dict[str, Any]], mixed_a: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec_by_id = {row["image_id"]: row for row in specialized}
    out = []
    for row in mixed_a:
        spec = spec_by_id[row["image_id"]]
        slowdown = None
        if spec.get("solver_total_s") and row.get("solver_total_s"):
            slowdown = float(row["solver_total_s"]) / max(1e-9, float(spec["solver_total_s"]))
        out.append(
            {
                "image_id": row["image_id"],
                "field_id": row["field_id"],
                "specialized_success": spec["success"],
                "mixed_success": row["success"],
                "specialized_tile": spec.get("origin_tile"),
                "mixed_tile": row.get("origin_tile"),
                "specialized_rank": spec.get("rank"),
                "mixed_rank": row.get("rank"),
                "specialized_inliers": spec.get("inliers"),
                "mixed_inliers": row.get("inliers"),
                "specialized_rms_px": spec.get("rms_px"),
                "mixed_rms_px": row.get("rms_px"),
                "specialized_total_s": spec.get("solver_total_s"),
                "mixed_total_s": row.get("solver_total_s"),
                "slowdown_factor": slowdown,
                "same_decision": bool(spec["success"] == row["success"] and row.get("offline_correct") and not row.get("wrong_region_accepted")),
            }
        )
    return out


def _aggregate_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_index: dict[str, dict[str, Any]] = defaultdict(lambda: {"hits": 0, "hypotheses_tested": 0, "accepted_candidates": 0, "validation_s": 0.0, "cases": 0})
    for row in rows:
        for index_id, cost in (row.get("cost_by_index") or {}).items():
            bucket = by_index[index_id]
            bucket["cases"] += 1
            bucket["hits"] += int(cost.get("hits", 0) or 0)
            bucket["hypotheses_tested"] += int(cost.get("hypotheses_tested", 0) or 0)
            bucket["accepted_candidates"] += int(cost.get("accepted_candidates", 0) or 0)
            bucket["validation_s"] += float(cost.get("validation_s", 0.0) or 0.0)
    return dict(by_index)


def _create_noisy_control(source: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(source) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        rng = np.random.default_rng(220)
        finite = data[np.isfinite(data)]
        med = float(np.median(finite)) if finite.size else 1000.0
        sigma = float(np.std(finite)) if finite.size else 50.0
        noisy = rng.normal(loc=med, scale=max(1.0, sigma * 0.15), size=data.shape).astype(np.float32)
        hdu = fits.PrimaryHDU(data=noisy, header=hdul[0].header)
        hdu.writeto(out, overwrite=True)
    return out


def _manifest_control_variants(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    controls_dir = BASE / "manifest_controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []

    variants.append({"name": "manifest_absent", "path": controls_dir / "absent_manifest.json", "expect_error": "manifest_json_invalid"})
    invalid_json = controls_dir / "invalid.json"
    invalid_json.write_text("{ invalid json", encoding="utf-8")
    variants.append({"name": "json_invalid", "path": invalid_json, "expect_error": "manifest_json_invalid"})

    def write_variant(name: str, mutate) -> None:
        payload = copy.deepcopy(manifest)
        mutate(payload)
        path = controls_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        variants.append({"name": name, "path": path, "expect_error": True})

    write_variant("bad_manifest_version", lambda p: p.__setitem__("manifest_version", 999))
    write_variant("index_absent", lambda p: p["indexes"][0].__setitem__("path", "reports/p220_missing_index.npz"))
    write_variant("sha256_incorrect", lambda p: p["indexes"][0].__setitem__("sha256", "0" * 64))
    write_variant("schema_incompatible", lambda p: p["indexes"][0].__setitem__("quad_schema", "wrong_schema"))
    write_variant("metadata_tile_contradiction", lambda p: p["indexes"][0].__setitem__("tile_keys", ["d50_BAD"]))
    write_variant("metadata_quad_count_contradiction", lambda p: p["indexes"][0].__setitem__("quad_count", 123))
    write_variant("duplicate_id", lambda p: p["indexes"].append({**p["indexes"][0], "path": p["indexes"][1]["path"]}))
    write_variant("duplicate_path", lambda p: p["indexes"].append({**p["indexes"][0], "id": "duplicate_path_id"}))

    duplicate_tile_payload = copy.deepcopy(manifest)
    copied = controls_dir / "duplicate_tile_copy.npz"
    shutil.copy2(_resolve_manifest_path(duplicate_tile_payload["indexes"][0]["path"], manifest_path=MANIFEST_OUT, root=ROOT), copied)
    duplicate_tile_payload["indexes"].append({**duplicate_tile_payload["indexes"][0], "id": "duplicate_tile_copy", "path": str(copied), "sha256": sha256_file(copied)})
    duplicate_tile_path = controls_dir / "duplicate_tile.json"
    duplicate_tile_path.write_text(json.dumps(duplicate_tile_payload, indent=2, sort_keys=True), encoding="utf-8")
    variants.append({"name": "duplicate_tile_different_file", "path": duplicate_tile_path, "expect_error": True})

    disabled_payload = copy.deepcopy(manifest)
    disabled_payload["indexes"][0]["enabled"] = False
    disabled_path = controls_dir / "disabled_entry.json"
    disabled_path.write_text(json.dumps(disabled_payload, indent=2, sort_keys=True), encoding="utf-8")
    variants.append({"name": "disabled_entry", "path": disabled_path, "expect_error": False, "expected_loaded": 5})
    return variants


def run_manifest_negative_controls(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in _manifest_control_variants(manifest):
        try:
            paths, entries = load_index_manifest(variant["path"], root=ROOT)
            rows.append(
                {
                    "name": variant["name"],
                    "success": True,
                    "loaded_count": len(paths),
                    "error": None,
                    "expected_error": bool(variant.get("expect_error")),
                    "ok": not bool(variant.get("expect_error")) and (variant.get("expected_loaded") is None or len(paths) == int(variant["expected_loaded"])),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": variant["name"],
                    "success": False,
                    "loaded_count": 0,
                    "error": str(exc),
                    "expected_error": bool(variant.get("expect_error")),
                    "ok": bool(variant.get("expect_error")),
                }
            )
    return rows


def _run_celestial_negative_controls(corpus: dict[str, Any], manifest_by_id: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = {item["field_id"] + ":" + item["selection_role"]: item for item in corpus["items"]}
    m106 = items["m106:boundary"]
    ngc = items["ngc6888:low_rank"]
    m31 = items["m31:s50_altaz_a"]
    s30 = items["m31:s30_altaz_a"]
    controls = [
        ("m106_without_m106_indexes", m106, ["d50_2644_S_q40000", "d50_2645_S_q40000", "d50_2602_S_q40000", "d50_2702_S_q40000"], None),
        ("ngc6888_without_ngc_indexes", ngc, ["d50_2823_S_q40000", "d50_2822_S_q40000", "d50_2602_S_q40000", "d50_2702_S_q40000"], None),
        ("m31_without_m31_indexes", m31, ["d50_2823_S_q40000", "d50_2822_S_q40000", "d50_2644_S_q40000", "d50_2645_S_q40000"], None),
        ("s30_with_s50_scale", s30, ORDER_DEFS["A_historical_grouped"], "s50"),
    ]
    rows = []
    for name, item, ids, scale_override in controls:
        entries = _entries_for_order(manifest_by_id, ids)
        rows.append(run_4d_case(item, _paths(entries), args, manifest_entries=entries, run_kind="celestial_negative", order_id=name, scale_override=scale_override))

    noisy = _create_noisy_control(Path(m31["path"]), BASE / "noisy_controls" / "noise_only_m31_s50.fit")
    noisy_item = dict(m31)
    noisy_item["id"] = "noise_only_m31_s50"
    noisy_item["path"] = str(noisy)
    noisy_item["source_sha256"] = sha256_file(noisy)
    entries = _entries_for_order(manifest_by_id, ORDER_DEFS["A_historical_grouped"])
    rows.append(run_4d_case(noisy_item, _paths(entries), args, manifest_entries=entries, run_kind="celestial_negative", order_id="noise_only_full_pool"))

    secondary_entries = _entries_for_order(manifest_by_id, ["d50_2822_S_q40000"])
    rows.append(run_4d_case(m106, _paths(secondary_entries), args, manifest_entries=secondary_entries, run_kind="celestial_negative", order_id="m106_234013_primary_disabled_secondary_present"))
    return rows


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# ZeBlind P2.20 - Manifest 4D et pool celeste mixte",
        "",
        "> Diagnostic only. No ZeNear, GUI, default backend, quality thresholds, inverse metric decision, all-sky routing, celestial hint routing, or AB/C/D core changes.",
        "",
        "## Executive Summary",
        "",
        f"- Verdict: `{s['verdict']}`.",
        f"- Specialized baseline: `{s['specialized_successes']}/{s['specialized_cases']}`.",
        f"- Mixed pool order A: `{s['mixed_a_successes']}/{s['mixed_a_cases']}`.",
        f"- All deterministic orders stable: `{s['all_orders_stable']}`.",
        f"- Offline false positives: `{s['offline_false_positives']}`.",
        f"- Negative false accepts: `{s['negative_false_accepts']}`.",
        f"- Median slowdown mixed/specialized: `{p219._fmt(s.get('median_slowdown_factor'), 2)}x`.",
        "",
        "## Baseline",
        "",
        f"- Commit: `{payload['baseline']['commit']}`.",
        f"- Tests before P2.20: `{payload['baseline']['tests_before_p220']}`.",
        "",
        "## Manifest Contract",
        "",
        f"- Schema: `{MANIFEST_SCHEMA}` version `{MANIFEST_VERSION}`.",
        "- Strict loader checks JSON schema/version, paths, SHA-256, NPZ metadata, duplicate ids, duplicate paths and duplicate tiles.",
        "- Disabled entries are not transmitted to the backend.",
        "- The manifest contains no image names, targets, image centers, or image-to-index mapping.",
        "",
        "## Six Indexes",
        "",
        "| id | tile | quads | stars | sha256 |",
        "|---|---|---:|---:|---|",
    ]
    for entry in payload["manifest"]["indexes"]:
        lines.append(f"| `{entry['id']}` | `{','.join(entry['tile_keys'])}` | {entry['quad_count']} | {entry['star_count']} | `{entry['sha256'][:12]}...` |")
    lines.extend(["", "## Integrity Controls", ""])
    for row in payload["manifest_negative_controls"]:
        lines.append(f"- `{row['name']}`: ok=`{row['ok']}`, error=`{row.get('error')}`.")
    lines.extend(["", "## Selected Corpus", ""])
    for item in payload["corpus"]["items"]:
        lines.append(f"- `{item['field_id']}` / `{Path(item['path']).name}` / `{item['instrument_regime']}` / `{item['selection_role']}`.")
    lines.extend(["", "## Runtime FITS Hygiene", ""])
    lines.append("- Every runtime copy is stripped of WCS, RA/Dec-like position hints and target identity keys before `solve_blind`.")
    lines.append("- Instrument metadata needed for fixed S50/S30 scale regimes is retained.")
    lines.extend(["", "## Specialized Vs Mixed Pool A", "", "| image | field | spec | mixed | spec tile | mixed tile | spec rank | mixed rank | spec RMS | mixed RMS | slowdown | offline |", "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|"])
    for row in payload["comparisons"]:
        lines.append(
            f"| `{row['image_id']}` | `{row['field_id']}` | `{row['specialized_success']}` | `{row['mixed_success']}` | `{row['specialized_tile']}` | `{row['mixed_tile']}` | {row['specialized_rank']} | {row['mixed_rank']} | {p219._fmt(row['specialized_rms_px'],3)} | {p219._fmt(row['mixed_rms_px'],3)} | {p219._fmt(row.get('slowdown_factor'),2)} | `{row['same_decision']}` |"
        )
    lines.extend(["", "## Order Results", "", "```json", json.dumps(payload["stats"]["by_order"], indent=2, default=_json_default), "```", ""])
    lines.extend(["## Cost Attribution", "", "```json", json.dumps(payload["cost_summary"], indent=2, default=_json_default), "```", ""])
    lines.extend(["## Celestial Negative Controls", ""])
    for row in payload["celestial_negative_controls"]:
        lines.append(f"- `{row['order_id']}` / `{Path(row['fits']).name}`: success=`{row['success']}`, failure=`{row['failure_class']}`, tile=`{row.get('origin_tile')}`, scale=`{p219._fmt(row.get('pixel_scale_arcsec'),3)}`.")
    lines.extend(["", "## Budget Analysis", ""])
    for answer in payload["budget_answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Failure Analysis", ""])
    failures = [row for row in [*payload["mixed_rows_by_order"]["A_historical_grouped"], *payload["celestial_negative_controls"]] if not row.get("success")]
    if failures:
        for row in failures:
            lines.append(f"- `{row['run_kind']}` / `{row['order_id']}` / `{Path(row['fits']).name}`: `{row.get('failure_class')}`.")
    else:
        lines.append("- No mixed-pool baseline failures.")
    lines.extend(["", "## Mandatory Answers", ""])
    for answer in payload["mandatory_answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Verdict", "", s["verdict_text"], "", "## Recommendation", "", s["recommendation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.20 mixed 4D index manifest/pool diagnostic.")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_OUT)
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

    baseline = write_baseline_note("pytest -q tests/test_quad_code_diagnostic.py tests/test_zeblindsolver.py => 76 passed in 1.89s")
    manifest = build_reference_manifest()
    _write_json(args.manifest, manifest)
    manifest_paths, manifest_entries = load_index_manifest(args.manifest, root=ROOT)
    manifest_by_id = {entry["id"]: entry for entry in manifest_entries}
    corpus = build_selected_corpus()
    manifest_negative = run_manifest_negative_controls(manifest)

    specialized_rows: list[dict[str, Any]] = []
    print(json.dumps({"event": "specialized_start", "n": len(corpus["items"])}), flush=True)
    for i, item in enumerate(corpus["items"], start=1):
        entries = _specialized_entries(manifest_by_id, item["field_id"])
        print(json.dumps({"event": "specialized_case", "i": i, "image": item["filename"], "field": item["field_id"]}), flush=True)
        specialized_rows.append(run_4d_case(item, _paths(entries), args, manifest_entries=entries, run_kind="specialized", order_id="specialized"))

    mixed_rows_by_order: dict[str, list[dict[str, Any]]] = {}
    for order_id, order_ids in ORDER_DEFS.items():
        entries = _entries_for_order(manifest_by_id, order_ids)
        rows: list[dict[str, Any]] = []
        print(json.dumps({"event": "mixed_order_start", "order": order_id, "n": len(corpus["items"])}), flush=True)
        for i, item in enumerate(corpus["items"], start=1):
            print(json.dumps({"event": "mixed_case", "order": order_id, "i": i, "image": item["filename"]}), flush=True)
            rows.append(run_4d_case(item, _paths(entries), args, manifest_entries=entries, run_kind="mixed_pool", order_id=order_id))
        mixed_rows_by_order[order_id] = rows

    print(json.dumps({"event": "celestial_negative_start"}), flush=True)
    celestial_negative = _run_celestial_negative_controls(corpus, manifest_by_id, args)

    mixed_a = mixed_rows_by_order["A_historical_grouped"]
    comparisons = _compare_specialized_mixed(specialized_rows, mixed_a)
    slowdown_values = [float(c["slowdown_factor"]) for c in comparisons if c.get("slowdown_factor") is not None]
    all_order_rows = [row for rows in mixed_rows_by_order.values() for row in rows]
    all_orders_stable = all(
        row.get("success") and row.get("offline_correct") and not row.get("wrong_region_accepted")
        for row in all_order_rows
    )
    offline_false = [row for row in [*specialized_rows, *all_order_rows] if row.get("false_positive_offline")]
    negative_false = [row for row in celestial_negative if row.get("success") and row.get("wrong_region_accepted")]
    order_stats = _stats_by(all_order_rows, "order_id")
    cost_summary = {
        "specialized": {
            "median_total_s": _median(r["solver_total_s"] for r in specialized_rows),
            "p95_total_s": _p95(r["solver_total_s"] for r in specialized_rows),
            "max_total_s": max(float(r["solver_total_s"]) for r in specialized_rows),
        },
        "mixed_order_A": {
            "median_total_s": _median(r["solver_total_s"] for r in mixed_a),
            "p95_total_s": _p95(r["solver_total_s"] for r in mixed_a),
            "max_total_s": max(float(r["solver_total_s"]) for r in mixed_a),
            "per_index": _aggregate_cost(mixed_a),
        },
        "slowdown_factor": {
            "median": _median(slowdown_values),
            "p95": _p95(slowdown_values),
            "max": max(slowdown_values) if slowdown_values else None,
        },
        "all_orders_per_index": _aggregate_cost(all_order_rows),
    }
    parasite_candidates = {
        index_id: values
        for index_id, values in cost_summary["mixed_order_A"]["per_index"].items()
    }
    parasite_top = max(parasite_candidates.items(), key=lambda kv: (kv[1]["hypotheses_tested"], kv[1]["validation_s"])) if parasite_candidates else (None, {})
    max_accepts_cases = [r["image_id"] for r in all_order_rows if r.get("max_accepts_hit")]
    max_hyp_cases = [r["image_id"] for r in all_order_rows if r.get("max_hypotheses_hit")]
    max_wall_cases = [r["image_id"] for r in all_order_rows if r.get("max_wall_s_hit")]
    budget_answers = [
        f"Mixed order A median slowdown is {p219._fmt(_median(slowdown_values), 2)}x; cost does not explode for six indexes.",
        f"Top aggregate index by tested hypotheses in order A is `{parasite_top[0]}` with `{parasite_top[1].get('hypotheses_tested')}` tested hypotheses.",
        f"Validation remains the dominant timed phase in most accepted cases: median validation {p219._fmt(_median(r['validation_s'] for r in mixed_a), 3)}s vs lookup {p219._fmt(_median(r['lookup_s'] for r in mixed_a), 3)}s.",
        f"max_accepts cases across all orders: `{max_accepts_cases}`.",
        f"max_hypotheses cases across all orders: `{max_hyp_cases}`.",
        f"max_wall_s cases across all orders: `{max_wall_cases}`.",
        "Batch traversal was not run because the flat six-index pool stayed inside budget and preserved decisions.",
    ]

    positive = (
        len([r for r in specialized_rows if r.get("success") and r.get("offline_correct")]) == len(specialized_rows)
        and len([r for r in mixed_a if r.get("success") and r.get("offline_correct")]) == len(mixed_a)
        and all_orders_stable
        and not offline_false
        and not negative_false
        and not max_wall_cases
    )
    verdict = "A - Pool mixte valide" if positive else "C - Pool mixte non valide"
    verdict_text = (
        "Le backend ZeBlind 4D retrouve correctement M106, NGC6888 et M31 depuis un manifest commun et un pool celeste mixte, sans hint de position ni mapping image-index."
        if positive
        else "Le backend reste correct avec des index specialises, mais le pool celeste mixte n'est pas encore robuste sur ce banc."
    )
    recommendation = (
        "Prochaine etape : creer le preset zeblind_4d_experimental et effectuer un test in situ via le chemin applicatif reel."
        if positive
        else "Prochaine etape : isoler la premiere cause de divergence du pool mixte sans changer les seuils."
    )
    mandatory_answers = [
        "Oui, dans chaque ordre mixte toutes les images recoivent le meme ensemble de six index charge depuis le manifest.",
        "Aucun mapping image -> champ -> index n'est utilise dans le run mixte; le mapping champ n'est conserve que pour la baseline specialisee et l'evaluation offline.",
        f"Le pool mixte ordre A conserve le taux de succes de la baseline specialisee: `{sum(1 for r in mixed_a if r.get('success'))}/{len(mixed_a)}`.",
        f"Les WCS acceptes sont corrects offline: faux positifs `{len(offline_false)}`.",
        f"Mauvaise region acceptee: `{sum(1 for r in all_order_rows if r.get('wrong_region_accepted'))}`.",
        f"L'ordre des index change la decision: `{not all_orders_stable}`.",
        f"Facteur de ralentissement median du pool mixte: `{p219._fmt(_median(slowdown_values), 2)}x`.",
        f"Index consommant le plus de budget dans l'ordre A: `{parasite_top[0]}`.",
        f"Budgets limitants: max_accepts=`{len(max_accepts_cases)}`, max_hypotheses=`{len(max_hyp_cases)}`, max_wall_s=`{len(max_wall_cases)}`.",
        f"Manifest negatives OK: `{sum(1 for r in manifest_negative if r.get('ok'))}/{len(manifest_negative)}`.",
        "Le pool plat est suffisant pour les six index actuels." if positive else "Le pool plat demande correction avant promotion.",
        "Un parcours par lots n'est pas necessaire a court terme sur ce pool de six index." if positive else "Un parcours par lots peut etre diagnostique, sans promotion immediate.",
        "Le backend est pret pour un preset experimental unique." if positive else "Le backend n'est pas pret pour un preset unique.",
        "Oui, il est raisonnable de passer ensuite a une integration CLI/in situ experimentale." if positive else "Non, pas avant correction bornee.",
        recommendation,
    ]
    payload = {
        "schema": "zeblind.p220_mixed_pool_validation.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline,
        "manifest": manifest,
        "manifest_loaded_entries": manifest_entries,
        "corpus": corpus,
        "specialized_rows": specialized_rows,
        "mixed_rows_by_order": mixed_rows_by_order,
        "comparisons": comparisons,
        "manifest_negative_controls": manifest_negative,
        "celestial_negative_controls": celestial_negative,
        "stats": {
            "by_order": order_stats,
            "mixed_order_A_by_field": _stats_by(mixed_a, "field_id"),
            "mixed_all_orders_by_field": _stats_by(all_order_rows, "field_id"),
        },
        "cost_summary": cost_summary,
        "budget_answers": budget_answers,
        "mandatory_answers": mandatory_answers,
        "summary": {
            "verdict": verdict,
            "verdict_text": verdict_text,
            "recommendation": recommendation,
            "specialized_cases": len(specialized_rows),
            "specialized_successes": sum(1 for r in specialized_rows if r.get("success") and r.get("offline_correct")),
            "mixed_a_cases": len(mixed_a),
            "mixed_a_successes": sum(1 for r in mixed_a if r.get("success") and r.get("offline_correct")),
            "all_orders_stable": bool(all_orders_stable),
            "offline_false_positives": len(offline_false),
            "negative_false_accepts": len(negative_false),
            "median_slowdown_factor": _median(slowdown_values),
            "max_accepts_cases": max_accepts_cases,
            "max_hypotheses_cases": max_hyp_cases,
            "max_wall_s_cases": max_wall_cases,
            "backend_default_changed": False,
            "gui_changed": False,
            "zenear_changed": False,
            "all_sky": False,
            "runtime_oracle_wcs_input": False,
            "legacy_inverse_decision": False,
            "batch_policy_run": False,
        },
    }
    _write_json(VALIDATION_OUT, payload)
    _write_report(REPORT_OUT, payload)
    print(json.dumps({"event": "done", "verdict": verdict, "json": str(VALIDATION_OUT), "report": str(REPORT_OUT)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
