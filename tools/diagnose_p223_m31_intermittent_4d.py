#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.index_manifest_4d import load_4d_index_manifest, sha256_file as manifest_sha256_file  # noqa: E402
from zeblindsolver.quad_index_4d import Quad4DIndex  # noqa: E402
from zeblindsolver.zeblindsolver import SolveConfig as BlindSolveConfig  # noqa: E402
from zeblindsolver.zeblindsolver import solve_blind  # noqa: E402


MANIFEST = ROOT / "config/zeblind_4d_experimental_manifest.json"
WORK = ROOT / "reports/p223_m31_intermittent_4d"
BASELINE_OUT = ROOT / "reports/zeblind_p223_baseline.json"
CORPUS_OUT = ROOT / "reports/zeblind_p223_corpus.json"
REPRO_OUT = ROOT / "reports/zeblind_p223_reproduction.json"
TRAVERSAL_OUT = ROOT / "reports/zeblind_p223_candidate_traversal.json"
VALIDATION_COST_OUT = ROOT / "reports/zeblind_p223_validation_cost.json"
ORACLE_OUT = ROOT / "reports/zeblind_p223_oracle_quad_audit.json"
UPSTREAM_OUT = ROOT / "reports/zeblind_p223_upstream_comparison.md"
REPORT_OUT = ROOT / "reports/zeblind_p223_m31_failure_autopsy.md"

ANDROTEST = Path("/home/tristan/zemosaic/example/androtest")
M31_IMAGES = [
    "Light_M 31_11_30.0s_IRCUT_20250922-230409.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230510.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230650.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230720.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230853.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231350.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231844.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231915.fit",
]

M31_TILES = {"d50_2602", "d50_2702"}
LOCAL_GROUPS = (
    "d50_2823,d50_2822",
    "d50_2644,d50_2645",
    "d50_2602,d50_2702",
)

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
IDENTITY_HINT_KEYS = {"OBJECT", "OBJNAME", "TARGET", "TARGNAME", "FIELD", "FIELDID"}
WCS_KEYS = {
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
    "CDELT1",
    "CDELT2",
    "CROTA1",
    "CROTA2",
    "RADESYS",
    "RADECSYS",
    "EQUINOX",
    "LONPOLE",
    "LATPOLE",
    "WCSAXES",
    "SOLVED",
    "DBSET",
    "TILE_ID",
    "RMSPX",
    "INLIERS",
    "PIXSCAL",
    "SIPORD",
    "QUALITY",
    "USED_DB",
    "SOLVER",
    "SOLVMODE",
    "BLINDVER",
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
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _load_zesolver_module() -> Any:
    spec = importlib.util.spec_from_file_location("zesolver_app_p223", ROOT / "zesolver.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load zesolver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numpy", "scipy", "astropy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            out[name] = f"ERROR: {exc}"
    return out


def _has_celestial_wcs(path: Path) -> bool:
    try:
        return bool(WCS(fits.getheader(path)).has_celestial)
    except Exception:
        return False


def _strip_runtime_copy(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    removed: list[str] = []
    with fits.open(target, mode="update", memmap=False) as hdul:
        header = hdul[0].header
        for key in sorted(WCS_KEYS | POSITION_HINT_KEYS | IDENTITY_HINT_KEYS):
            if key in header:
                removed.append(key)
                del header[key]
        for key in list(header.keys()):
            if key.startswith(("PV", "A_", "B_", "AP_", "BP_", "SIP")):
                removed.append(key)
                del header[key]
        hdul.flush()
    header = fits.getheader(target)
    forbidden = sorted(k for k in (POSITION_HINT_KEYS | IDENTITY_HINT_KEYS | WCS_KEYS) if k in header)
    return {
        "source": str(source),
        "runtime": str(target),
        "source_sha256": _sha256(source),
        "runtime_sha256": _sha256(target),
        "removed_keys": sorted(set(removed)),
        "forbidden_keys_remaining": forbidden,
        "has_celestial_wcs_after_strip": _has_celestial_wcs(target),
    }


def _offline_wcs_check(source: Path, candidate_wcs: WCS | None, image_shape: tuple[int, int]) -> dict[str, Any]:
    if candidate_wcs is None:
        return {"available": _has_celestial_wcs(source), "ok": False, "reason": "no_candidate_wcs"}
    try:
        ref = WCS(fits.getheader(source))
        if not ref.has_celestial:
            return {"available": False, "ok": False, "reason": "source_has_no_celestial_wcs"}
        h, w = image_shape
        pts = np.asarray(
            [
                [w / 2.0, h / 2.0],
                [0.0, 0.0],
                [w - 1.0, 0.0],
                [0.0, h - 1.0],
                [w - 1.0, h - 1.0],
            ],
            dtype=np.float64,
        )
        got = candidate_wcs.all_pix2world(pts, 0)
        exp = ref.all_pix2world(pts, 0)
        cos_dec = np.cos(np.deg2rad(exp[:, 1]))
        dra = (got[:, 0] - exp[:, 0]) * cos_dec * 3600.0
        ddec = (got[:, 1] - exp[:, 1]) * 3600.0
        sep = np.hypot(dra, ddec)
        scales = proj_plane_pixel_scales(candidate_wcs.celestial) * 3600.0
        return {
            "available": True,
            "ok": bool(float(sep[0]) < 60.0 and float(np.max(sep[1:])) < 180.0),
            "center_sep_arcsec": float(sep[0]),
            "corner_max_sep_arcsec": float(np.max(sep[1:])),
            "pixel_scale_arcsec": float(np.sqrt(float(scales[0]) * float(scales[1]))),
        }
    except Exception as exc:
        return {"available": True, "ok": False, "reason": str(exc)}


def _image_shape(path: Path) -> tuple[int, int]:
    shape = fits.getdata(path, memmap=False).shape
    return int(shape[-2]), int(shape[-1])


def _app_base_config(zs: Any, manifest: Any) -> Any:
    return zs.SolveConfig(
        db_root=ROOT,
        input_dir=WORK,
        families=None,
        workers=1,
        blind_enabled=True,
        blind_only=True,
        blind_skip_if_valid=False,
        blind_backend_profile="zeblind_4d_experimental",
        blind_4d_manifest_path=MANIFEST,
        blind_4d_loaded_manifest=manifest,
        hint_focal_mm=250.0,
        hint_pixel_um=2.9,
        hint_resolution_arcsec=2.39,
        hint_resolution_min_arcsec=1.79,
        hint_resolution_max_arcsec=2.99,
        blind_quality_inliers=40,
        blind_quality_rms=1.2,
        blind_max_stars=500,
        blind_max_quads=8000,
        blind_max_candidates=10,
        blind_fast_mode=True,
        log_level="INFO",
    )


def _effective_blind_config(zs: Any, manifest: Any, *, index_paths: Iterable[Path] | None = None) -> BlindSolveConfig:
    app_cfg = _app_base_config(zs, manifest)
    if index_paths is None:
        cfg = zs.build_blind_solve_config(app_cfg, loaded_manifest=manifest)
    else:
        from zeblindsolver.profiles import get_solver_profile

        base = zs.build_blind_solve_config(app_cfg, loaded_manifest=manifest)
        cfg = get_solver_profile("zeblind_4d_experimental").apply_to_config(base, index_paths=tuple(index_paths))
    return cfg


def _variant_config(
    base: BlindSolveConfig,
    *,
    index_paths: tuple[Path, ...],
    budget_s: float = 45.0,
    order_policy: str = "index_order",
    groups: tuple[str, ...] = (),
    reuse_tree: bool = False,
    skip_legacy: bool = False,
    skip_mono: bool = False,
    max_hits_per_quad: int = 8,
) -> BlindSolveConfig:
    return dataclasses.replace(
        base,
        blind_astrometry_4d_index_paths=tuple(str(path) for path in index_paths),
        max_stars=120,
        max_quads=2500,
        quality_inliers=40,
        quality_rms=1.2,
        pixel_tolerance=2.5,
        blind_global_hard_budget_s=float(budget_s),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_accept_policy="best_within_budget",
        blind_astrometry_4d_match_radius_px=3.0,
        blind_astrometry_4d_max_hypotheses=2000,
        blind_astrometry_4d_max_accepts=64,
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
        blind_astrometry_4d_image_strategy="log_spaced",
        blind_astrometry_4d_code_tol=0.015,
        blind_astrometry_4d_max_hits=2000,
        blind_astrometry_4d_max_hits_per_image_quad=int(max_hits_per_quad),
        blind_astrometry_4d_diagnostic_candidate_order_policy=str(order_policy),
        blind_astrometry_4d_diagnostic_validation_catalog_groups=tuple(groups),
        blind_astrometry_4d_diagnostic_reuse_image_kdtree=bool(reuse_tree),
        blind_astrometry_4d_diagnostic_skip_legacy_inverse=bool(skip_legacy),
        blind_astrometry_4d_diagnostic_skip_mono_validation=bool(skip_mono),
        ra_hint_deg=None,
        dec_hint_deg=None,
        radius_hint_deg=None,
    )


def _short_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "quad_hash_schema",
        "astrometry_4d_runtime_enabled",
        "astrometry_4d_hits",
        "astrometry_4d_candidates",
        "astrometry_4d_hits_tested",
        "astrometry_4d_accepted_candidates",
        "astrometry_4d_stop_reason",
        "astrometry_4d_first_plausible_rank",
        "astrometry_4d_first_accepted_rank",
        "astrometry_4d_selected_origin_tile_key",
        "astrometry_4d_selected_rank",
        "astrometry_4d_selected_local_rank",
        "astrometry_4d_selected_index_path",
        "astrometry_4d_total_s",
        "astrometry_4d_quad_build_s",
        "astrometry_4d_kd_lookup_s",
        "astrometry_4d_validation_s",
        "astrometry_4d_validation_cost_s",
        "inliers",
        "rms_px",
        "pix_scale_arcsec",
    ]
    return {key: stats.get(key) for key in keys if key in stats}


def _run_case(
    *,
    label: str,
    source: Path,
    base_runtime: Path,
    cfg: BlindSolveConfig,
    index_root: Path,
) -> dict[str, Any]:
    runtime = WORK / "runs" / label / source.name
    hygiene = _strip_runtime_copy(base_runtime, runtime)
    start = time.perf_counter()
    result = solve_blind(runtime, index_root, config=cfg, cancel_check=None)
    elapsed = time.perf_counter() - start
    stats = dict(result.stats or {})
    offline = _offline_wcs_check(source, result.wcs, _image_shape(source))
    row = {
        "label": label,
        "source": str(source),
        "runtime": str(runtime),
        "hygiene": hygiene,
        "success": bool(result.success),
        "message": result.message,
        "elapsed_s": float(elapsed),
        "offline_wcs_check": offline,
        "offline_correct": bool(result.success and offline.get("ok")),
        "false_positive_offline": bool(result.success and offline.get("available") and not offline.get("ok")),
        "stats": stats,
        "summary": _short_stats(stats),
    }
    print(
        f"[p223] {label} {source.name}: success={row['success']} offline={row['offline_correct']} "
        f"stop={stats.get('astrometry_4d_stop_reason')} hits={stats.get('astrometry_4d_hits')} "
        f"tested={stats.get('astrometry_4d_hits_tested')} tile={stats.get('astrometry_4d_selected_origin_tile_key')} "
        f"rms={stats.get('rms_px')} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return row


def _index_by_tile(entries: Iterable[Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for entry in entries:
        for tile in entry.tile_keys:
            out[str(tile)] = entry.path
    return out


def _summarize_per_index(row: dict[str, Any]) -> dict[str, Any]:
    stats = row.get("stats") or {}
    ranges = stats.get("astrometry_4d_candidate_ranges_by_index") or {}
    validation = stats.get("astrometry_4d_per_index_validation") or {}
    out: dict[str, Any] = {}
    for path, payload in ranges.items():
        merged = dict(payload or {})
        merged.update(validation.get(path) or {})
        out[Path(path).stem] = merged
    return out


def _first_m31_rank(row: dict[str, Any]) -> dict[str, Any]:
    per = _summarize_per_index(row)
    m31 = {k: v for k, v in per.items() if any(tile in M31_TILES for tile in (v.get("tile_keys") or []))}
    first_candidate = min((int(v["first_global_rank"]) for v in m31.values() if v.get("first_global_rank") is not None), default=None)
    first_tested = min((int(v["first_tested_rank"]) for v in m31.values() if v.get("first_tested_rank") is not None), default=None)
    first_accepted = min((int(v["first_accepted_rank"]) for v in m31.values() if v.get("first_accepted_rank") is not None), default=None)
    return {
        "first_m31_candidate_rank": first_candidate,
        "first_m31_tested_rank": first_tested,
        "first_m31_accepted_rank": first_accepted,
        "m31_per_index": m31,
    }


def _build_baseline(zs: Any, manifest: Any, base_cfg: BlindSolveConfig) -> dict[str, Any]:
    entries = []
    for entry in manifest.entries:
        entries.append(
            {
                "id": entry.id,
                "path": str(entry.path),
                "tile_keys": list(entry.tile_keys),
                "manifest_sha256": entry.sha256,
                "actual_sha256": manifest_sha256_file(entry.path),
                "quad_count": entry.quad_count,
                "star_count": entry.star_count,
                "metadata": entry.metadata,
            }
        )
    effective = dataclasses.asdict(base_cfg)
    keys = {
        "quad_hash_schema": effective.get("quad_hash_schema"),
        "quad_sources": effective.get("max_stars"),
        "verification_sources": "full",
        "validation_catalog_policy": effective.get("blind_astrometry_4d_validation_catalog_policy"),
        "accept_policy": effective.get("blind_astrometry_4d_accept_policy"),
        "quality_inliers": effective.get("quality_inliers"),
        "quality_rms": effective.get("quality_rms"),
        "match_radius_px": effective.get("blind_astrometry_4d_match_radius_px"),
        "max_quads": effective.get("max_quads"),
        "max_hypotheses": effective.get("blind_astrometry_4d_max_hypotheses"),
        "max_accepts": effective.get("blind_astrometry_4d_max_accepts"),
        "max_wall_s": effective.get("blind_global_hard_budget_s"),
        "max_hits_per_image_quad": effective.get("blind_astrometry_4d_max_hits_per_image_quad"),
    }
    expected = {
        "quad_hash_schema": "astrometry_ab_code_4d_v1",
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
        "max_hits_per_image_quad": 8,
    }
    divergences = {k: {"expected": v, "actual": keys.get(k)} for k, v in expected.items() if keys.get(k) != v}
    payload = {
        "schema": "zeblind.p223_baseline.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "git_status": _git(["status", "--short", "--untracked-files=all"]),
        "python": sys.version,
        "packages": _package_versions(),
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "workers": 1,
        "manifest": str(MANIFEST),
        "manifest_tile_order": list(manifest.tile_keys),
        "indexes": entries,
        "effective_config_checked_via_build_blind_solve_config": keys,
        "effective_config_full": effective,
        "required_config_divergences": divergences,
    }
    _write_json(BASELINE_OUT, payload)
    return payload


def _build_corpus() -> dict[str, Any]:
    items = []
    for name in M31_IMAGES:
        source = ANDROTEST / name
        runtime = WORK / "corpus" / "runtime" / name
        hygiene = _strip_runtime_copy(source, runtime)
        oracle = WORK / "corpus" / "oracle" / name
        oracle.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, oracle)
        items.append(
            {
                "filename": name,
                "source": str(source),
                "oracle": str(oracle),
                "runtime": str(runtime),
                "source_sha256": _sha256(source),
                "oracle_sha256": _sha256(oracle),
                "runtime_sha256": _sha256(runtime),
                "source_has_wcs": _has_celestial_wcs(source),
                "runtime_hygiene": hygiene,
            }
        )
    payload = {
        "schema": "zeblind.p223_corpus.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(ANDROTEST),
        "selection_policy": "Only the eight FITS named in P2.23 are copied; no directory-wide traversal.",
        "items": items,
    }
    _write_json(CORPUS_OUT, payload)
    return payload


def _matrix_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "successes": sum(1 for row in rows if row.get("success")),
        "offline_correct": sum(1 for row in rows if row.get("offline_correct")),
        "false_positive_offline": sum(1 for row in rows if row.get("false_positive_offline")),
        "by_image": {
            Path(row["source"]).name: {
                "success": row.get("success"),
                "offline_correct": row.get("offline_correct"),
                "stop": (row.get("stats") or {}).get("astrometry_4d_stop_reason"),
                "hits": (row.get("stats") or {}).get("astrometry_4d_hits"),
                "tested": (row.get("stats") or {}).get("astrometry_4d_hits_tested"),
                "accepted": (row.get("stats") or {}).get("astrometry_4d_accepted_candidates"),
                "tile": (row.get("stats") or {}).get("astrometry_4d_selected_origin_tile_key"),
                "rank": (row.get("stats") or {}).get("astrometry_4d_selected_rank"),
                "inliers": (row.get("stats") or {}).get("inliers"),
                "rms_px": (row.get("stats") or {}).get("rms_px"),
                **_first_m31_rank(row),
            }
            for row in rows
        },
    }


def _oracle_audit(repro_rows: list[dict[str, Any]], matrix_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    failed = [row for row in repro_rows if not row.get("success")]
    a2_by_name = {Path(row["source"]).name: row for row in matrix_rows.get("A2_m31_pair_120s", [])}
    cap_by_name = {Path(row["source"]).name: row for row in matrix_rows.get("neighbor_cap64_m31_pair_120s", [])}
    out_rows = []
    for row in failed:
        name = Path(row["source"]).name
        stats = row.get("stats") or {}
        a2 = a2_by_name.get(name, {})
        cap = cap_by_name.get(name, {})
        first = _first_m31_rank(row)
        classification = "other"
        if a2.get("success"):
            if first.get("first_m31_tested_rank") is None:
                classification = "true_candidate_generated_but_not_reached"
            elif stats.get("astrometry_4d_stop_reason") == "hard_budget_exceeded":
                classification = "true_candidate_valid_but_budget_expired"
            else:
                classification = "true_candidate_reached_but_validation_failed"
        elif cap.get("success"):
            classification = "true_neighbor_rank_gt_8"
        elif (a2.get("stats") or {}).get("astrometry_4d_hits", 0):
            classification = "true_candidate_reached_but_validation_failed"
        else:
            classification = "true_quad_absent_from_sampled_quads"
        out_rows.append(
            {
                "filename": name,
                "baseline": row.get("summary"),
                "baseline_m31": first,
                "a2_m31_pair_120s": a2.get("summary"),
                "cap64_m31_pair_120s": cap.get("summary"),
                "classification": classification,
                "oracle_note": "Offline WCS is used only for validating candidate WCS correctness; no runtime hints are passed.",
            }
        )
    payload = {
        "schema": "zeblind.p223_oracle_quad_audit.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method_limit": "This audit classifies failures through runtime candidate telemetry plus 120s/cap64 diagnostic probes; it does not use oracle RA/Dec to route runtime solving.",
        "failures": out_rows,
    }
    _write_json(ORACLE_OUT, payload)
    return payload


def _write_upstream_comparison() -> None:
    lines = [
        "# P2.23 Astrometry.net Conceptual Comparison",
        "",
        "| Theme | Astrometry.net local source | Behavior | ZeBlind 4D behavior | P2.23 status |",
        "|---|---|---|---|---|",
        "| Quad code / lookup | `astrometry-net-main/include/astrometry/codefile.h`, `astrometry-net-main/util/multiindex.c`, `astrometry-net-main/util/index.c` | Continuous quad code indexed and searched through KD/range structures. | `zeblindsolver/quad_index_4d.py` uses `cKDTree.query_ball_point` over AB/C/D code. | Difference mostly closed by 4D backend; not the primary suspected cause here unless cap/source audit proves otherwise. |",
        "| Candidate traversal | `astrometry-net-main/solver/solver.c` symbols `solver_handle_hit`, `verify_hit` | Solver interleaves generated image quads and index hits through a sequential verify loop. | Current ZeBlind 4D sorts by `index_order`, then `code_distance`, then local rank unless diagnostic policies alter it. | Causal if C1/C2 reaches M31 candidates within 45s while C0 does not. |",
        "| Verification | `astrometry-net-main/solver/verify.c` symbols `verify_hit`, `real_verify_star_lists` | Sequential evidence/log-odds style verification and tweak/refit path. | Direct catalog-world2pix matching with fixed inlier/RMS thresholds; legacy inverse/mono are telemetry in this route. | Conceptual difference; causal only if path/cost matrices show delayed valid candidates. |",
        "| Source preprocessing | `astrometry-net-main/util/uniformize.py`, `solver/image2xy.py` | Source lists can be spatially uniformized and progressively consumed. | P2.23 profile uses `diagnostic_unfiltered`, `quad_sources=120`, `verification_sources=full`, `log_spaced` quads. | Plausible if oracle/source audit shows true quads absent from 120/2500. |",
        "| Multi-index validation catalog | `astrometry-net-main/util/multiindex.c` | Indexes are distinct search sources; candidate verification is local to candidate/index context. | `union_candidate_tiles` currently unions all loaded index catalogs unless diagnostic groups are supplied. | Causal if D1 preserves WCS/inliers and fixes budget. |",
        "| Stop policy | `astrometry-net-main/solver/solver.c` | Stop after sufficient evidence, with log-odds controls. | `best_within_budget` continues until budget, `max_accepts`, `max_hypotheses`, or exhaustion. | Plausible secondary cost difference; not promoted in P2.23. |",
        "",
        "P2.23 deliberately does not port Astrometry.net verification. The comparison is used only to label demonstrated vs plausible differences.",
        "",
    ]
    UPSTREAM_OUT.write_text("\n".join(lines), encoding="utf-8")


def _write_report(payload: dict[str, Any]) -> None:
    matrices = payload["matrices"]
    repro = matrices["B1_full_pool_45s_run1"]
    failures = [name for name, row in repro["by_image"].items() if not row["success"]]
    successes = [name for name, row in repro["by_image"].items() if row["success"]]
    cost_rows = payload["validation_cost"]["by_variant"]
    order_causal = any(
        not repro["by_image"].get(name, {}).get("success") and matrices.get("C2_round_robin_45s", {}).get("by_image", {}).get(name, {}).get("success")
        for name in failures
    ) or any(
        not repro["by_image"].get(name, {}).get("success") and matrices.get("C1_global_distance_45s", {}).get("by_image", {}).get(name, {}).get("success")
        for name in failures
    )
    groups_causal = any(
        not repro["by_image"].get(name, {}).get("success") and matrices.get("D1_local_groups_45s", {}).get("by_image", {}).get(name, {}).get("success")
        for name in failures
    )
    cost_causal = any(
        not repro["by_image"].get(name, {}).get("success") and matrices.get("E2_minimal_direct_45s", {}).get("by_image", {}).get(name, {}).get("success")
        for name in failures
    )
    if order_causal or groups_causal or cost_causal:
        verdict = "A - Parcours et validation causaux"
        recommendation = "P2.24: appliquer d'abord un parcours equitable du pool couple a une validation par groupes locaux, puis alleger le chemin chaud mesure."
    else:
        verdict = "E - Non resolu"
        recommendation = "Ne modifier aucun parametre produit; completer l'audit oracle source-list/cap huit avec un trace vrai-quad plus profond."
    lines = [
        "# P2.23 - Autopsie M31 ZeBlind 4D intermittent",
        "",
        "## Resume executif",
        "",
        f"- Verdict: `{verdict}`.",
        f"- Baseline sequentielle B1: `{repro['successes']}/{repro['total']}` succes, faux positifs offline `{repro['false_positive_offline']}`.",
        f"- Succes baseline: {', '.join('`' + s + '`' for s in successes) if successes else '`aucun`'}.",
        f"- Echecs baseline: {', '.join('`' + s + '`' for s in failures) if failures else '`aucun`'}.",
        f"- Correction unique recommandee P2.24: {recommendation}",
        "",
        "## Configuration effective",
        "",
        f"- Commit: `{payload['baseline']['commit']}` sur branche `{payload['baseline']['branch']}`.",
        f"- Manifest: `{payload['baseline']['manifest']}`.",
        f"- Divergences config requise: `{len(payload['baseline']['required_config_divergences'])}`.",
        "",
        "## Matrices",
        "",
        "| matrice | succes | offline OK | faux positifs |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in matrices.items():
        lines.append(f"| `{name}` | {summary['successes']}/{summary['total']} | {summary['offline_correct']} | {summary['false_positive_offline']} |")
    lines.extend(["", "## Chronologie et cout", ""])
    for name, costs in cost_rows.items():
        lines.append(
            f"- `{name}`: hypotheses `{costs.get('tested_total')}`, validation `{costs.get('validation_s_total'):.3f}s`, "
            f"cout moyen `{costs.get('mean_hypothesis_s'):.4f}s`, composants `{costs.get('components')}`."
        )
    lines.extend(["", "## Audit oracle des echecs", ""])
    for row in payload["oracle_audit"]["failures"]:
        lines.append(f"- `{row['filename']}`: `{row['classification']}`; baseline M31 `{row['baseline_m31']}`.")
    lines.extend(
        [
            "",
            "## Questions obligatoires",
            "",
            f"1. Les cinq echecs sont-ils reproductibles en mode sequentiel ? `{len(failures) == 5}` ({len(failures)} echecs observes).",
            "2. Le bon candidat M31 existe-t-il dans leurs hits ? Voir `first_m31_candidate_rank` et matrices A2/cap64 dans `zeblind_p223_oracle_quad_audit.json`.",
            "3. Le bon candidat est-il atteint avant le timeout ? Voir `first_m31_tested_rank` par image.",
            "4. Quel index consomme le plus de temps parasite ? Voir `zeblind_p223_candidate_traversal.json`, section per-index `validation_s`.",
            f"5. Le tri par `index_order` est-il causal ? `{order_causal}`.",
            f"6. Un tri global ou equitable ameliore-t-il le resultat ? `{order_causal}`.",
            f"7. L'union globale des six catalogues est-elle necessaire ? `{not groups_causal}` dans cette matrice; D1 mesure le groupe local.",
            "8. Les groupes locaux conservent-ils les inliers et le WCS ? Voir matrice D1, offline false positives doit rester 0.",
            "9. Quel est le cout moyen reel d'une hypothese ? Voir section cout ci-dessus.",
            "10. Combien de ce cout vient de legacy/mono/telemetrie ? Voir composants `legacy_inverse_validation`, `mono_*` dans `zeblind_p223_validation_cost.json`.",
            "11. Le KD-tree image est-il reconstruit par hypothese ? Oui dans E0; E1 active le cache diagnostique.",
            "12. Les 120 sources contiennent-elles assez d'etoiles oracle-matchables ? Classement par image dans `zeblind_p223_oracle_quad_audit.json`.",
            "13. Un vrai quad se trouve-t-il parmi les 2500 quads ? Infere par A2/cap64; trace exhaustif vrai-quad non promu produit.",
            "14. Le vrai voisin depasse-t-il parfois le rang 8 ? Mesure par `neighbor_cap64_m31_pair_120s`.",
            "15. Les hashes/codes AB/C/D sont-ils reellement incorrects ? Non demontre par P2.23 si A2/cap64 produit des WCS corrects.",
            "16. Quelle difference avec Astrometry.net est causalement demontree ? Voir verdict et matrices C/D/E.",
            "17. Quelle difference reste seulement conceptuelle ? Voir `zeblind_p223_upstream_comparison.md`.",
            f"18. Correctif unique recommande P2.24: {recommendation}",
            "",
            "## Artefacts",
            "",
            f"- JSON baseline: `{BASELINE_OUT}`",
            f"- JSON corpus: `{CORPUS_OUT}`",
            f"- JSON reproduction: `{REPRO_OUT}`",
            f"- JSON traversal: `{TRAVERSAL_OUT}`",
            f"- JSON validation cost: `{VALIDATION_COST_OUT}`",
            f"- JSON oracle audit: `{ORACLE_OUT}`",
            f"- Upstream comparison: `{UPSTREAM_OUT}`",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.23 M31 intermittent ZeBlind 4D autopsy")
    ap.add_argument("--skip-long", action="store_true", help="Run only baseline reproduction plus cheap reports")
    ap.add_argument("--only", choices=("all", "baseline"), default="all")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    manifest = load_4d_index_manifest(MANIFEST)
    zs = _load_zesolver_module()
    base_cfg = _effective_blind_config(zs, manifest)
    base_cfg = _variant_config(base_cfg, index_paths=tuple(manifest.enabled_index_paths), budget_s=45.0)
    baseline = _build_baseline(zs, manifest, base_cfg)
    corpus = _build_corpus()
    _write_upstream_comparison()
    if args.only == "baseline":
        return 0

    by_tile = _index_by_tile(manifest.entries)
    full_pool = tuple(manifest.enabled_index_paths)
    m31_pair = tuple(by_tile[tile] for tile in ("d50_2602", "d50_2702"))
    index_root = MANIFEST.parent
    sources = [ANDROTEST / name for name in M31_IMAGES]
    runtime_sources = {Path(item["source"]).name: Path(item["runtime"]) for item in corpus["items"]}

    variants: list[tuple[str, BlindSolveConfig, list[Path]]] = [
        ("B1_full_pool_45s_run1", _variant_config(base_cfg, index_paths=full_pool, budget_s=45.0), sources),
        ("B1_full_pool_45s_run2", _variant_config(base_cfg, index_paths=full_pool, budget_s=45.0), sources),
    ]
    if not args.skip_long:
        variants.extend(
            [
                ("A1_m31_pair_45s", _variant_config(base_cfg, index_paths=m31_pair, budget_s=45.0), sources),
                ("A2_m31_pair_120s", _variant_config(base_cfg, index_paths=m31_pair, budget_s=120.0), sources),
                ("B2_full_pool_120s", _variant_config(base_cfg, index_paths=full_pool, budget_s=120.0), sources),
                ("C1_global_distance_45s", _variant_config(base_cfg, index_paths=full_pool, budget_s=45.0, order_policy="global_distance"), sources),
                ("C2_round_robin_45s", _variant_config(base_cfg, index_paths=full_pool, budget_s=45.0, order_policy="round_robin_local_rank"), sources),
                ("D1_local_groups_45s", _variant_config(base_cfg, index_paths=full_pool, budget_s=45.0, groups=LOCAL_GROUPS), sources),
                ("E1_cached_tree_45s", _variant_config(base_cfg, index_paths=full_pool, budget_s=45.0, reuse_tree=True), sources),
                (
                    "E2_minimal_direct_45s",
                    _variant_config(base_cfg, index_paths=full_pool, budget_s=45.0, groups=LOCAL_GROUPS, reuse_tree=True, skip_legacy=True, skip_mono=True),
                    sources,
                ),
                ("neighbor_cap64_m31_pair_120s", _variant_config(base_cfg, index_paths=m31_pair, budget_s=120.0, max_hits_per_quad=64), sources),
            ]
        )

    matrix_rows: dict[str, list[dict[str, Any]]] = {}
    for label, cfg, variant_sources in variants:
        rows = []
        for source in variant_sources:
            rows.append(
                _run_case(
                    label=label,
                    source=source,
                    base_runtime=runtime_sources[source.name],
                    cfg=cfg,
                    index_root=index_root,
                )
            )
        matrix_rows[label] = rows

    matrices = {label: _matrix_summary(rows) for label, rows in matrix_rows.items()}
    reproduction = {
        "schema": "zeblind.p223_reproduction.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": {
            "run1": matrix_rows["B1_full_pool_45s_run1"],
            "run2": matrix_rows["B1_full_pool_45s_run2"],
        },
        "summary": {
            "run1": matrices["B1_full_pool_45s_run1"],
            "run2": matrices["B1_full_pool_45s_run2"],
            "same_success_vector": [
                row1.get("success") == row2.get("success")
                for row1, row2 in zip(matrix_rows["B1_full_pool_45s_run1"], matrix_rows["B1_full_pool_45s_run2"])
            ],
        },
    }
    _write_json(REPRO_OUT, reproduction)

    traversal = {
        "schema": "zeblind.p223_candidate_traversal.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "matrices": {
            label: {
                Path(row["source"]).name: {
                    "summary": row.get("summary"),
                    "first_m31": _first_m31_rank(row),
                    "per_index": _summarize_per_index(row),
                }
                for row in rows
            }
            for label, rows in matrix_rows.items()
        },
    }
    _write_json(TRAVERSAL_OUT, traversal)

    validation_cost = {"schema": "zeblind.p223_validation_cost.v1", "generated_at": datetime.now().isoformat(timespec="seconds"), "by_variant": {}}
    for label, rows in matrix_rows.items():
        tested_total = sum(int((row.get("stats") or {}).get("astrometry_4d_hits_tested", 0) or 0) for row in rows)
        validation_total = sum(float((row.get("stats") or {}).get("astrometry_4d_validation_s", 0.0) or 0.0) for row in rows)
        components: dict[str, float] = {}
        for row in rows:
            for key, value in ((row.get("stats") or {}).get("astrometry_4d_validation_cost_s") or {}).items():
                components[str(key)] = components.get(str(key), 0.0) + float(value or 0.0)
        validation_cost["by_variant"][label] = {
            "tested_total": int(tested_total),
            "validation_s_total": float(validation_total),
            "mean_hypothesis_s": float(validation_total / tested_total) if tested_total else 0.0,
            "components": components,
        }
    _write_json(VALIDATION_COST_OUT, validation_cost)

    oracle = _oracle_audit(matrix_rows["B1_full_pool_45s_run1"], matrix_rows)
    payload = {
        "baseline": baseline,
        "corpus": corpus,
        "matrices": matrices,
        "validation_cost": validation_cost,
        "oracle_audit": oracle,
    }
    _write_report(payload)
    print(f"[p223] wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
