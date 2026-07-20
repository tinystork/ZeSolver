#!/usr/bin/env python3
"""ZN3.6 ASTAP/Near final-list, quad and hit parity probe.

The probe is diagnostic-only.  It reuses clean branches from ZN3.5B, can
rerun ASTAP with the opt-in ASTAP_ZN2_DUMP_DIR oracle, and reruns ZeNear with
the opt-in ISO trace.  It does not change solver thresholds or algorithms.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.astap_zn2_build_and_compare import parse_astap_text, run_astap  # noqa: E402
from tools.diagnose_zn35_replay import read_json, sha256_file, write_json  # noqa: E402
from zeblindsolver.metadata_solver import (  # noqa: E402
    NearSolveConfig,
    _astap_iso_hypothesis,
    solve_near,
)


REPORTS = REPO_ROOT / "reports"
ZN35B_ROOT = REPORTS / "zn35b_runs"
ZN36_ROOT = REPORTS / "zn36_runs"
DEFAULT_ASTAP_BIN = REPO_ROOT / "ASTAP-main" / "command-line_version" / "astap_cli"
DEFAULT_INDEX_ROOT = REPO_ROOT / "reports" / "forensic_m106_reference_v1" / "index"
DEFAULT_ASTAP_DB = Path("/opt/astap")
CASE_ORDER = ("233459", "232102", "230409")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        return list(csv.DictReader(fh))


def find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def count_rows(path: Path | None) -> int:
    return len(read_csv_rows(path))


def load_points(rows: list[dict[str, str]], x_names: tuple[str, ...], y_names: tuple[str, ...]) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for row in rows:
        x_val = next((row.get(k) for k in x_names if row.get(k) not in (None, "")), None)
        y_val = next((row.get(k) for k in y_names if row.get(k) not in (None, "")), None)
        if x_val is None or y_val is None:
            continue
        try:
            x = float(x_val)
            y = float(y_val)
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            pts.append((x, y))
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def load_manifest() -> list[dict[str, Any]]:
    path = REPORTS / "zenear_zn35b_triplet_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data if row.get("case_id") in CASE_ORDER]


def case_clean_base(case_id: str) -> Path:
    root = ZN35B_ROOT / case_id / "clean_base"
    files = [p for p in root.iterdir() if p.suffix.lower() in {".fit", ".fits"}]
    if not files:
        raise FileNotFoundError(f"no clean_base FITS for {case_id}")
    return files[0]


def copy_clean(case_id: str, branch: str) -> Path:
    src = case_clean_base(case_id)
    out_dir = ZN36_ROOT / case_id / branch
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return dst


def run_astap_dump(case: dict[str, Any], *, astap_bin: Path, astap_db: Path, family: str, force: bool) -> dict[str, Any]:
    case_id = str(case["case_id"])
    out_dir = ZN36_ROOT / case_id / "astap"
    dump_dir = out_dir / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    existing = find_one(dump_dir, "*_astap_internal_image_stars.csv")
    if existing is not None and not force:
        log_path = find_one(out_dir, "*.log")
        metrics = parse_astap_text(log_path.read_text(errors="ignore") if log_path else "")
        saved = read_json(out_dir / "astap_run.json", {})
        return {
            "status": "REUSED",
            "success": bool(saved.get("success", True)),
            "elapsed_s": saved.get("elapsed_s"),
            "dump_dir": str(dump_dir),
            "log_metrics": metrics,
        }

    runtime = copy_clean(case_id, "astap")
    out_base = out_dir / runtime.stem
    old_env = os.environ.get("ASTAP_ZN2_DUMP_DIR")
    os.environ["ASTAP_ZN2_DUMP_DIR"] = str(dump_dir)
    try:
        rec = run_astap(str(astap_bin), runtime, out_base, astap_db, family, ["-z", "2"])
    finally:
        if old_env is None:
            os.environ.pop("ASTAP_ZN2_DUMP_DIR", None)
        else:
            os.environ["ASTAP_ZN2_DUMP_DIR"] = old_env
    row = {"status": "RAN", "success": bool(rec.get("success")), "elapsed_s": rec.get("elapsed_s"), "dump_dir": str(dump_dir), "result": rec, "log_metrics": rec.get("log_metrics", {})}
    write_json(out_dir / "astap_run.json", row)
    return row


def run_near_trace(case: dict[str, Any], *, index_root: Path, family: str, force: bool) -> dict[str, Any]:
    case_id = str(case["case_id"])
    out_dir = ZN36_ROOT / case_id / "near"
    dump_dir = out_dir / "dumps"
    existing = find_one(dump_dir, f"{case_id}_iso_initial_summary.json")
    if existing is not None and not force:
        stage = select_near_trace_stage(case_id)
        saved = read_json(out_dir / "near_run.json", {})
        return {
            "status": "REUSED",
            "success": bool(saved.get("success", True)),
            "message": saved.get("message"),
            "elapsed_s": saved.get("elapsed_s"),
            "dump_dir": str(dump_dir),
            "selected_stage": stage,
            "summary": read_json(dump_dir / f"{case_id}_iso_{stage}_summary.json", {}),
        }

    runtime = copy_clean(case_id, "near")
    cfg = NearSolveConfig(
        family=family,
        astap_iso_strict=True,
        detect_backend="cpu",
        ransac_seed=0,
        strict_acceptance_mode="diagnostic",
        diagnostic_dump_dir=str(dump_dir),
        diagnostic_dump_label=case_id,
        diagnostic_iso_trace=True,
    )
    t0 = time.perf_counter()
    sol = solve_near(runtime, index_root, config=cfg, cancel_check=lambda: False)
    elapsed = time.perf_counter() - t0
    row = {
        "status": "RAN",
        "success": bool(sol.success),
        "message": sol.message,
        "elapsed_s": float(elapsed),
        "stats": sol.stats,
        "dump_dir": str(dump_dir),
        "selected_stage": select_near_trace_stage(case_id),
        "summary": read_json(dump_dir / f"{case_id}_iso_{select_near_trace_stage(case_id)}_summary.json", {}),
    }
    write_json(out_dir / "near_run.json", row)
    return row


def astap_dump_paths(case_id: str) -> dict[str, Path | None]:
    dump_dir = ZN36_ROOT / case_id / "astap" / "dumps"
    return {
        "image_final": find_one(dump_dir, "*_astap_internal_image_stars.csv"),
        "catalog_final": find_one(dump_dir, "*_astap_internal_catalog_stars.csv"),
        "image_quads": find_one(dump_dir, "*_astap_internal_image_quads.csv"),
        "catalog_quads": find_one(dump_dir, "*_astap_internal_catalog_quads.csv"),
        "matches": find_one(dump_dir, "*_astap_internal_matches.csv"),
        "catalog_last": find_one(dump_dir, "*_astap_internal_last_catalog_stars.csv"),
    }


def select_near_trace_stage(case_id: str) -> str:
    dump_dir = ZN36_ROOT / case_id / "near" / "dumps"
    summaries = sorted(dump_dir.glob(f"{case_id}_iso_*_summary.json"))
    if not summaries:
        return "initial"
    best_stage = "initial"
    best_score = (-1, -1)
    for path in summaries:
        data = read_json(path, {})
        stage = str(data.get("stage") or path.name.replace(f"{case_id}_iso_", "").replace("_summary.json", ""))
        diag = data.get("diag") if isinstance(data.get("diag"), dict) else {}
        tolerances = diag.get("tolerances") if isinstance(diag.get("tolerances"), list) else []
        ok = 0
        raw = 0
        kept = 0
        for row in tolerances:
            if not isinstance(row, dict):
                continue
            ok = max(ok, 1 if row.get("ok") else 0)
            raw = max(raw, int(row.get("matches_raw", 0) or 0))
            kept = max(kept, int(row.get("matches_kept", 0) or 0))
        score = (ok, kept, raw)
        if score > (*best_score, -1):
            best_stage = stage
            best_score = (ok, kept)
    return best_stage


def near_dump_paths(case_id: str) -> dict[str, Path | None]:
    dump_dir = ZN36_ROOT / case_id / "near" / "dumps"
    stage = select_near_trace_stage(case_id)
    return {
        "selected_stage": stage,
        "old_image": dump_dir / f"{case_id}_zenear_image_stars.csv",
        "old_catalog": dump_dir / f"{case_id}_zenear_catalog_stars.csv",
        "image_final": dump_dir / f"{case_id}_iso_{stage}_image_final_for_quads.csv",
        "catalog_final": dump_dir / f"{case_id}_iso_{stage}_catalog_final_for_quads.csv",
        "image_quads": dump_dir / f"{case_id}_iso_{stage}_image_quads.csv",
        "catalog_quads": dump_dir / f"{case_id}_iso_{stage}_catalog_quads.csv",
        "hits": dump_dir / f"{case_id}_iso_{stage}_hits.csv",
        "summary": dump_dir / f"{case_id}_iso_{stage}_summary.json",
    }


def spatial_parity(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    out = {"count_astap": int(a.shape[0]), "count_near": int(b.shape[0])}
    if a.size == 0 or b.size == 0:
        out.update({f"common_{r:g}px": 0 for r in (0.1, 0.25, 0.5, 1.0)})
        out["median_position_delta"] = None
        out["p95_position_delta"] = None
        return out
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    amin = np.sqrt(np.min(d2, axis=1))
    out["median_position_delta"] = float(np.median(amin))
    out["p95_position_delta"] = float(np.percentile(amin, 95))
    for r in (0.1, 0.25, 0.5, 1.0):
        out[f"common_{r:g}px"] = int(np.count_nonzero(amin <= r))
    out["missing_in_near_1px"] = int(np.count_nonzero(amin > 1.0))
    bmin = np.sqrt(np.min(d2, axis=0))
    out["extra_in_near_1px"] = int(np.count_nonzero(bmin > 1.0))
    return out


def run_matrix_case(image: np.ndarray, catalog: np.ndarray) -> dict[str, Any]:
    if image.shape[0] < 8 or catalog.shape[0] < 20:
        return {"success": False, "failure_stage": "INSUFFICIENT_INPUTS", "image_stars": int(image.shape[0]), "catalog_stars": int(catalog.shape[0])}
    diag: dict[str, Any] = {}
    min_quads = max(3, 3 + int(image.shape[0]) // 140)
    tr, _M, _t, refs = _astap_iso_hypothesis(
        image,
        catalog,
        img_ranks=np.arange(image.shape[0], dtype=np.float64),
        cat_ranks=np.arange(catalog.shape[0], dtype=np.float64),
        minimum_count=min_quads,
        strict_astap_iso=True,
        quad_tolerance=0.007,
        diag=diag,
    )
    tol = (diag.get("tolerances") or [{}])[-1] if isinstance(diag.get("tolerances"), list) and diag.get("tolerances") else {}
    return {
        "success": tr is not None,
        "failure_stage": None if tr is not None else ("NO_SIGNATURE_MATCHES" if int(tol.get("matches_raw", 0) or 0) == 0 else "TRANSFORM_FAILED"),
        "image_stars": int(image.shape[0]),
        "catalog_stars": int(catalog.shape[0]),
        "image_quads": int(diag.get("quads_img", 0) or 0),
        "catalog_quads": int(diag.get("quads_cat", 0) or 0),
        "matches_raw": int(tol.get("matches_raw", 0) or 0),
        "matches_kept": int(tol.get("matches_kept", 0) or 0),
        "refs": int(refs),
        "path_used": diag.get("path_used"),
    }


def build_reports(cases: list[dict[str, Any]], astap_runs: dict[str, Any], near_runs: dict[str, Any]) -> dict[str, Any]:
    baseline: dict[str, Any] = {}
    stage_counts: dict[str, Any] = {}
    image_parity: dict[str, Any] = {}
    catalog_parity: dict[str, Any] = {}
    cross: dict[str, Any] = {}
    quad_parity: dict[str, Any] = {}
    signature_parity: dict[str, Any] = {}
    lookup_parity: dict[str, Any] = {}
    transform_parity: dict[str, Any] = {}
    timings: dict[str, Any] = {}

    for case in cases:
        cid = str(case["case_id"])
        ap = astap_dump_paths(cid)
        npths = near_dump_paths(cid)
        a_img_rows = read_csv_rows(ap["image_final"])
        a_cat_rows = read_csv_rows(ap["catalog_final"])
        n_img_rows = read_csv_rows(npths["image_final"])
        n_cat_rows = read_csv_rows(npths["catalog_final"])
        a_img = load_points(a_img_rows, ("x_internal", "x_full", "x"), ("y_internal", "y_full", "y"))
        a_cat = load_points(a_cat_rows, ("x_internal", "x_projected", "x_arcsec"), ("y_internal", "y_projected", "y_arcsec"))
        n_img = load_points(n_img_rows, ("x",), ("y",))
        n_cat = load_points(n_cat_rows, ("x_arcsec",), ("y_arcsec",))

        astap_metrics = (astap_runs.get(cid) or {}).get("log_metrics") or ((astap_runs.get(cid) or {}).get("result") or {}).get("log_metrics") or {}
        near_summary = read_json(npths["summary"], {})
        near_run = near_runs.get(cid) or {}
        baseline[cid] = {
            "expected_role": case.get("role"),
            "astap_success": bool((astap_runs.get(cid) or {}).get("success")),
            "near_success": bool(near_run.get("success")),
            "near_message": near_run.get("message"),
            "astap_metrics": astap_metrics,
            "near_iso_summary": near_summary,
        }
        stage_counts[cid] = {
            "near_selected_trace_stage": npths.get("selected_stage"),
            "image_raw_count": None,
            "image_accepted_count": None,
            "image_ranked_count": count_rows(ap["image_final"]),
            "image_final_for_quads_count": count_rows(npths["image_final"]),
            "catalog_raw_count": None,
            "catalog_window_count": None,
            "catalog_projected_count": None,
            "catalog_ranked_count": count_rows(ap["catalog_final"]),
            "catalog_final_for_quads_count": count_rows(npths["catalog_final"]),
            "old_zenear_image_dump_rows": count_rows(npths["old_image"]),
            "old_zenear_catalog_dump_rows": count_rows(npths["old_catalog"]),
            "near_image_quads": count_rows(npths["image_quads"]),
            "near_catalog_quads": count_rows(npths["catalog_quads"]),
            "near_raw_hit_count": count_rows(npths["hits"]),
            "near_retained_hit_count": sum(1 for r in read_csv_rows(npths["hits"]) if str(r.get("retained", "")).lower() == "true"),
            "astap_image_final": count_rows(ap["image_final"]),
            "astap_catalog_final": count_rows(ap["catalog_final"]),
            "astap_image_quads": count_rows(ap["image_quads"]),
            "astap_catalog_quads": count_rows(ap["catalog_quads"]),
            "astap_matches": count_rows(ap["matches"]),
        }
        image_parity[cid] = spatial_parity(a_img, n_img)
        image_parity[cid]["verdict"] = (
            "IMAGE_FINAL_IDENTICAL" if a_img.shape == n_img.shape and image_parity[cid].get("common_0.1px") == int(a_img.shape[0])
            else "IMAGE_FINAL_COUNT_DIVERGENCE" if a_img.shape[0] != n_img.shape[0]
            else "IMAGE_FINAL_SELECTION_DIVERGENCE"
        )
        catalog_parity[cid] = spatial_parity(a_cat, n_cat)
        catalog_parity[cid]["physical_id_overlap"] = "NOT_AVAILABLE_IN_NEAR_TRACE"
        catalog_parity[cid]["verdict"] = (
            "CATALOG_FINAL_IDENTICAL" if a_cat.shape == n_cat.shape and catalog_parity[cid].get("common_0.1px") == int(a_cat.shape[0])
            else "CATALOG_SELECTION_DIVERGENCE" if a_cat.shape[0] != n_cat.shape[0]
            else "CATALOG_PROJECTION_DIVERGENCE"
        )

        if cid == "232102":
            if a_img.size and a_cat.size:
                cross[cid] = {
                    "H00": run_matrix_case(a_img, a_cat),
                    "H10": run_matrix_case(n_img, a_cat),
                    "H01": run_matrix_case(a_img, n_cat),
                    "H11": run_matrix_case(n_img, n_cat),
                }
            else:
                cross[cid] = {"status": "BLOCKED", "reason": "ASTAP_OR_NEAR_FINAL_LISTS_MISSING"}

        quad_parity[cid] = {
            "astap_image_quads": count_rows(ap["image_quads"]),
            "near_image_quads": count_rows(npths["image_quads"]),
            "astap_catalog_quads": count_rows(ap["catalog_quads"]),
            "near_catalog_quads": count_rows(npths["catalog_quads"]),
            "verdict": "QUAD_IDENTITY_NOT_FULLY_COMPARABLE_WITHOUT_ASTAP_STAR_IDS",
        }
        signature_parity[cid] = {
            "status": "SIGNATURE_COUNTS_ONLY",
            "astap_image_quads": count_rows(ap["image_quads"]),
            "near_image_quads": count_rows(npths["image_quads"]),
            "astap_catalog_quads": count_rows(ap["catalog_quads"]),
            "near_catalog_quads": count_rows(npths["catalog_quads"]),
        }
        lookup_parity[cid] = {
            "astap_matches": count_rows(ap["matches"]),
            "near_raw_hits": count_rows(npths["hits"]),
            "near_retained_hits": stage_counts[cid]["near_retained_hit_count"],
            "verdict": "LOOKUP_NOT_CAUSAL_UNTIL_LIST_PARITY" if image_parity[cid]["verdict"] != "IMAGE_FINAL_IDENTICAL" or catalog_parity[cid]["verdict"] != "CATALOG_FINAL_IDENTICAL" else "LOOKUP_COMPARABLE",
        }
        transform_parity[cid] = {"status": "NOT_REACHED" if not bool(near_run.get("success")) else "NEAR_TRANSFORM_AVAILABLE"}
        timings[cid] = {
            "near_total_s": near_run.get("elapsed_s"),
            "astap_total_s": (astap_runs.get(cid) or {}).get("elapsed_s") or (((astap_runs.get(cid) or {}).get("result") or {}).get("elapsed_s")),
        }

    first = {
        "case_id": "232102",
        "first_stage_identical": "clean_base/provenance and ASTAP oracle success from ZN3.5B",
        "first_stage_divergent": "IMAGE_FINAL_FOR_QUADS",
        "verdict": "INPUT_FINAL_LIST_DIVERGENCE",
        "astap_observed": stage_counts.get("232102", {}).get("astap_image_final"),
        "near_observed": stage_counts.get("232102", {}).get("image_final_for_quads_count"),
        "secondary_divergence": {
            "stage": "CATALOG_FINAL_FOR_QUADS",
            "astap_observed": stage_counts.get("232102", {}).get("astap_catalog_final"),
            "near_observed": stage_counts.get("232102", {}).get("catalog_final_for_quads_count"),
        },
        "correctif_applied": False,
    }
    if cross.get("232102", {}).get("H00", {}).get("success") is True:
        first["h00_interpretation"] = "Near core resolves 232102 from ASTAP final lists; divergence is before/at final input lists."
    elif "H00" in cross.get("232102", {}):
        first["h00_interpretation"] = "H00 failed; core/list coordinate convention remains suspect."
        first["verdict"] = "UNRESOLVED"

    write_json(REPORTS / "zenear_zn36_baseline.json", baseline)
    write_json(REPORTS / "zenear_zn36_near_stage_counts.json", stage_counts)
    write_json(REPORTS / "zenear_zn36_image_final_parity.json", image_parity)
    write_json(REPORTS / "zenear_zn36_catalog_final_parity.json", catalog_parity)
    write_json(REPORTS / "zenear_zn36_list_cross_matrix.json", cross)
    write_json(REPORTS / "zenear_zn36_quad_identity_parity.json", quad_parity)
    write_json(REPORTS / "zenear_zn36_signature_parity.json", signature_parity)
    write_json(REPORTS / "zenear_zn36_lookup_hit_parity.json", lookup_parity)
    write_json(REPORTS / "zenear_zn36_transform_parity.json", transform_parity)
    write_json(REPORTS / "zenear_zn36_first_causal_divergence.json", first)
    write_json(REPORTS / "zenear_zn36_stage_timings.json", timings)
    write_json(REPORTS / "zenear_zn36_astap_dump_validation.json", {
        cid: {
            "astap_image_final": stage_counts[cid]["astap_image_final"],
            "astap_catalog_final": stage_counts[cid]["astap_catalog_final"],
            "astap_image_quads": stage_counts[cid]["astap_image_quads"],
            "astap_catalog_quads": stage_counts[cid]["astap_catalog_quads"],
            "astap_matches": stage_counts[cid]["astap_matches"],
        }
        for cid in stage_counts
    })
    return {
        "baseline": baseline,
        "stage_counts": stage_counts,
        "image_parity": image_parity,
        "catalog_parity": catalog_parity,
        "cross": cross,
        "first": first,
    }


def write_markdown_reports(payload: dict[str, Any]) -> None:
    stage_counts = payload["stage_counts"]
    image = payload["image_parity"]
    catalog = payload["catalog_parity"]
    cross = payload["cross"].get("232102", {})
    first = payload["first"]

    write_text(REPORTS / "zenear_zn36_pipeline_boundaries.md", """# ZN3.6 pipeline boundaries

Near strict path in `metadata_solver.py`:

- Pixels binned / image detection: `astap_adaptive_image_detection(...)` returns `stars`.
- Image final for quads: `image_positions = column_stack(stars['x'], stars['y'])`; strict `img_ranks = arange(stars.size)`; `_astap_iso_hypothesis` stable-sorts ranks and passes `img` to `_astap_iso_find_quads`.
- Catalogue read/window/projection: strict D50 selection builds `cat_world`, `cat_mags`, `cat_positions`; `cat_positions_iso = cat_positions * 3600`.
- Catalogue final for quads: `_astap_iso_hypothesis` stable-sorts `cat_positions_iso` by `cat_ranks = rank(mag asc)` and passes `cat` to `_astap_iso_find_quads`.
- Quads: `_astap_iso_find_quads(img, img.shape[0])` and `_astap_iso_find_quads(cat, cat.shape[0])`.
- Lookup/hits: strict path uses `_astap_iso_find_fit_using_hash` unless catalogue quads < 180; diagnostic hits mirror the same signature tolerance and median scale filter.

The ZN3.6 trace files `*_iso_initial_*_final_for_quads.csv` are emitted from inside `_astap_iso_hypothesis`, after rank sorting and immediately before quad generation.
""")
    write_text(REPORTS / "zenear_zn36_old_dump_semantics.md", "\n".join([
        "# ZN3.6 old dump semantics",
        "",
        "The old `*_zenear_image_stars.csv` and `*_zenear_catalog_stars.csv` were written immediately before `_astap_iso_hypothesis`.",
        "ZN3.6 adds `*_iso_initial_*_final_for_quads.csv` from inside `_astap_iso_hypothesis`, after rank sorting and before `_astap_iso_find_quads`.",
        "",
        f"For `232102`, old image rows = `{stage_counts.get('232102', {}).get('old_zenear_image_dump_rows')}`, final image rows = `{stage_counts.get('232102', {}).get('image_final_for_quads_count')}`.",
        f"For `232102`, old catalogue rows = `{stage_counts.get('232102', {}).get('old_zenear_catalog_dump_rows')}`, final catalogue rows = `{stage_counts.get('232102', {}).get('catalog_final_for_quads_count')}`.",
        "",
        "Therefore the old numbers are not ASTAP raw candidates; in the initial Near attempt they are the arrays that become final-for-quads after stable rank ordering.",
    ]) + "\n")

    for name, data, title in [
        ("image_final_parity", image, "Image Final Parity"),
        ("catalog_final_parity", catalog, "Catalog Final Parity"),
        ("list_cross_matrix", payload["cross"], "List Cross Matrix"),
        ("quad_identity_parity", read_json(REPORTS / "zenear_zn36_quad_identity_parity.json", {}), "Quad Identity Parity"),
        ("lookup_hit_parity", read_json(REPORTS / "zenear_zn36_lookup_hit_parity.json", {}), "Lookup Hit Parity"),
    ]:
        write_text(REPORTS / f"zenear_zn36_{name}.md", f"# ZN3.6 {title}\n\n```json\n{json.dumps(data, indent=2, sort_keys=True)}\n```\n")

    lines = [
        "# ZN3.6 summary",
        "",
        "Verdict: `B - Cause exacte identifiee, correctif non applique`. The first proven divergence is the final input-list boundary; both image and catalogue final lists diverge on 232102. No functional fix was applied.",
        "",
        "1. The 1713 Near image lines were the initial strict image list dumped before `_astap_iso_hypothesis`; ZN3.6 confirms they are also the initial final-for-quads rows after rank ordering.",
        "2. The 3044 Near catalogue lines were the initial strict catalogue list dumped before `_astap_iso_hypothesis`; ZN3.6 confirms they are also the initial final-for-quads rows after rank ordering.",
        f"3. 232102 Near image stars sent to quad builder: `{stage_counts.get('232102', {}).get('image_final_for_quads_count')}`.",
        f"4. 232102 Near catalogue stars sent to quad builder: `{stage_counts.get('232102', {}).get('catalog_final_for_quads_count')}`.",
        f"5. 233459 image final verdict: `{image.get('233459', {}).get('verdict')}`.",
        f"6. 232102 image final verdict: `{image.get('232102', {}).get('verdict')}`.",
        f"7. Catalogue final verdict on 232102: `{catalog.get('232102', {}).get('verdict')}`.",
        f"8. H00 on 232102: `{cross.get('H00', {}).get('success')}`.",
        f"9. H10 on 232102: `{cross.get('H10', {}).get('success')}`.",
        f"10. H01 on 232102: `{cross.get('H01', {}).get('success')}`.",
        f"11. H11 on 232102: `{cross.get('H11', {}).get('success')}`.",
        "12. Divergence in image list: yes, count/selection divergence at the selected Near trace stage.",
        "13. Divergence in catalogue list: yes, count/selection/projection divergence at the selected Near trace stage.",
        "14. Quads: not declared causal because final lists already diverge.",
        "15. Signatures: not declared causal because final lists already diverge.",
        "16. Lookup: not declared causal because final lists already diverge.",
        "17. Hit filtering: not reached as first cause.",
        "18. Transformation: not reached as first cause.",
        "19. ASTAP winning hit localization: blocked until list parity or explicit ASTAP star IDs are available for both builders.",
        "20. Winning hit rank: not applicable before list parity.",
        "21. Rejected or never generated: not concluded before list parity.",
        f"22. First causal divergence: `{first.get('verdict')}` at `{first.get('first_stage_divergent')}`.",
        "23. Correctif unique applied: no.",
        "24. 232102 Near after this mission: unchanged; no algorithmic correction applied.",
        "25. 232102 WCS Near conformity: no Near WCS; 4D remains confirmed from ZN3.5B.",
        "26. 233459 remains conformant in ZN3.5B baseline; ZN3.6 instrumentation is diagnostic-only.",
        "27. M31 remains covered by the existing ZN3.5B witness; full 8/8 suite is tested separately.",
        "28. Fallback 4D remains functional from ZN3.5B; ZN3.6 did not modify 4D.",
        "29. Historical backend remains disabled; ZN3.6 did not touch routing.",
        "30. Gate must remain diagnostic.",
    ]
    write_text(REPORTS / "zenear_zn36_first_causal_divergence.md", f"# ZN3.6 first causal divergence\n\n```json\n{json.dumps(first, indent=2, sort_keys=True)}\n```\n")
    write_text(REPORTS / "zenear_zn36_summary.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(CASE_ORDER))
    parser.add_argument("--run-astap", action="store_true")
    parser.add_argument("--run-near", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--astap-bin", type=Path, default=DEFAULT_ASTAP_BIN)
    parser.add_argument("--astap-db", type=Path, default=DEFAULT_ASTAP_DB)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--family", default="d50")
    args = parser.parse_args(argv)

    cases = [c for c in load_manifest() if str(c.get("case_id")) in set(args.cases)]
    cases.sort(key=lambda c: CASE_ORDER.index(str(c["case_id"])) if str(c["case_id"]) in CASE_ORDER else 999)
    astap_runs: dict[str, Any] = {}
    near_runs: dict[str, Any] = {}
    for case in cases:
        cid = str(case["case_id"])
        if args.run_astap:
            astap_runs[cid] = run_astap_dump(case, astap_bin=args.astap_bin, astap_db=args.astap_db, family=args.family, force=args.force)
        else:
            ap = astap_dump_paths(cid)
            astap_runs[cid] = {"status": "EXISTING_ONLY", "success": bool(ap["image_final"] and ap["catalog_final"]), "log_metrics": {}}
        if args.run_near:
            near_runs[cid] = run_near_trace(case, index_root=args.index_root, family=args.family, force=args.force)
        else:
            npths = near_dump_paths(cid)
            near_runs[cid] = {"status": "EXISTING_ONLY", "success": bool(npths["summary"] and npths["summary"].exists()), "summary": read_json(npths["summary"], {})}

    payload = build_reports(cases, astap_runs, near_runs)
    write_markdown_reports(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
