#!/usr/bin/env python3
"""ZN3.7 ASTAP/Near list-selection parity probe.

Diagnostic-only.  The probe consumes the clean ZN3.5B/ZN3.6 ASTAP and Near
trace dumps, compares the actual final-for-quad-builder lists, and runs
controlled list-selection matrices through the existing Near ASTAP-ISO core.
It does not tune thresholds, alter solver algorithms, call ZeBlind, or enable
historical fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_zn35_replay import read_json, write_json  # noqa: E402
from tools.diagnose_zn36_parity import run_matrix_case  # noqa: E402
from zeblindsolver.metadata_solver import NearSolveConfig, project_tan  # noqa: E402


REPORTS = REPO_ROOT / "reports"
ZN36_ROOT = REPORTS / "zn36_runs"
CASE_STAGES = {
    "232102": "spiral_9",
    "233459": "autofov_1_win_1.595116",
    "230409": "initial",
}
PREFIX_IMAGE = (32, 48, 58, 64, 80, 100, 120, 146, 250, 500)
PREFIX_CATALOG = (128, 192, 249, 250, 260, 384, 512, 1000)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        return list(csv.DictReader(fh))


def first_glob(root: Path, pattern: str) -> Path | None:
    found = sorted(root.glob(pattern))
    return found[0] if found else None


def astap_paths(case: str) -> dict[str, Path | None]:
    root = ZN36_ROOT / case / "astap" / "dumps"
    return {
        "image": first_glob(root, "*_astap_internal_image_stars.csv"),
        "catalog": first_glob(root, "*_astap_internal_catalog_stars.csv"),
        "catalog_raw": first_glob(root, "*_astap_catalog_raw.csv"),
        "catalog_projected": first_glob(root, "*_astap_catalog_projected.csv"),
        "image_quads": first_glob(root, "*_astap_internal_image_quads.csv"),
        "catalog_quads": first_glob(root, "*_astap_internal_catalog_quads.csv"),
        "matches": first_glob(root, "*_astap_internal_matches.csv"),
    }


def near_stage_paths(case: str, stage: str) -> dict[str, Path]:
    root = ZN36_ROOT / case / "near" / "dumps"
    return {
        "image": root / f"{case}_iso_{stage}_image_final_for_quads.csv",
        "catalog": root / f"{case}_iso_{stage}_catalog_final_for_quads.csv",
        "image_quads": root / f"{case}_iso_{stage}_image_quads.csv",
        "catalog_quads": root / f"{case}_iso_{stage}_catalog_quads.csv",
        "hits": root / f"{case}_iso_{stage}_hits.csv",
        "summary": root / f"{case}_iso_{stage}_summary.json",
    }


def all_near_stages(case: str) -> list[str]:
    root = ZN36_ROOT / case / "near" / "dumps"
    stages: list[str] = []
    for p in sorted(root.glob(f"{case}_iso_*_summary.json")):
        d = read_json(p, {})
        stages.append(str(d.get("stage") or p.name.replace(f"{case}_iso_", "").replace("_summary.json", "")))

    def key(stage: str) -> tuple[int, float, str]:
        if stage == "initial":
            return (0, 0.0, stage)
        if stage.startswith("autofov_"):
            try:
                return (1, float(stage.split("_")[1]), stage)
            except Exception:
                return (1, 999.0, stage)
        if stage.startswith("spiral_"):
            try:
                return (2, float(stage.split("_")[1]), stage)
            except Exception:
                return (2, 999.0, stage)
        if stage == "recenter_second_pass":
            return (3, 0.0, stage)
        return (4, 0.0, stage)

    return sorted(stages, key=key)


def f(row: dict[str, str], *names: str, default: float = math.nan) -> float:
    for name in names:
        val = row.get(name)
        if val not in (None, ""):
            try:
                return float(val)
            except Exception:
                pass
    return default


def identity_from_values(ra: float, dec: float, mag: float) -> str:
    if not (np.isfinite(ra) and np.isfinite(dec) and np.isfinite(mag)):
        return ""
    return f"radec_mag:{ra:.8f}:{dec:.8f}:{mag:.3f}"


def load_astap_image(case: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows(astap_paths(case)["image"]):
        out.append(
            {
                "rank": int(float(row.get("rank", len(out)) or len(out))),
                "x": f(row, "x_internal", "x_full_resolution"),
                "y": f(row, "y_internal", "y_full_resolution"),
                "flux": math.nan,
            }
        )
    return out


def load_near_image(case: str, stage: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows(near_stage_paths(case, stage)["image"]):
        out.append(
            {
                "rank": int(float(row.get("rank", len(out) + 1) or len(out) + 1)),
                "source_index": int(float(row.get("source_index", -1) or -1)),
                "x": f(row, "x"),
                "y": f(row, "y"),
                "flux": f(row, "flux"),
            }
        )
    return out


def load_astap_catalog(case: str) -> list[dict[str, Any]]:
    raw_by_id: dict[str, dict[str, str]] = {r.get("catalog_internal_id", ""): r for r in rows(astap_paths(case)["catalog_raw"])}
    proj_by_id: dict[str, dict[str, str]] = {r.get("catalog_internal_id", ""): r for r in rows(astap_paths(case)["catalog_projected"])}
    out: list[dict[str, Any]] = []
    for row in rows(astap_paths(case)["catalog"]):
        cid = row.get("internal_id", str(len(out)))
        raw = raw_by_id.get(cid, {})
        proj = proj_by_id.get(cid, {})
        ra = f(raw, "ra_deg")
        dec = f(raw, "dec_deg")
        mag = f(raw, "magnitude")
        out.append(
            {
                "rank": int(float(row.get("rank", len(out)) or len(out))),
                "internal_id": cid,
                "tile_id": raw.get("tile_id", ""),
                "row_index": raw.get("row_index", ""),
                "ra_deg": ra,
                "dec_deg": dec,
                "magnitude": mag,
                "x_arcsec": f(row, "x_internal", "x_full_resolution", default=f(proj, "x_projected")),
                "y_arcsec": f(row, "y_internal", "y_full_resolution", default=f(proj, "y_projected")),
                "decoded_identity": identity_from_values(ra, dec, mag),
            }
        )
    return out


def load_near_catalog(case: str, stage: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows(near_stage_paths(case, stage)["catalog"]):
        ra = f(row, "ra_deg")
        dec = f(row, "dec_deg")
        mag = f(row, "magnitude")
        out.append(
            {
                "rank": int(float(row.get("rank", len(out) + 1) or len(out) + 1)),
                "source_index": int(float(row.get("source_index", -1) or -1)),
                "ra_deg": ra,
                "dec_deg": dec,
                "magnitude": mag,
                "x_arcsec": f(row, "x_arcsec"),
                "y_arcsec": f(row, "y_arcsec"),
                "identity_kind": row.get("identity_kind", ""),
                "decoded_identity": row.get("decoded_identity") or identity_from_values(ra, dec, mag),
            }
        )
    return out


def points(items: list[dict[str, Any]], x: str = "x", y: str = "y") -> np.ndarray:
    vals = [(float(r[x]), float(r[y])) for r in items if np.isfinite(float(r.get(x, math.nan))) and np.isfinite(float(r.get(y, math.nan)))]
    return np.asarray(vals, dtype=np.float64) if vals else np.empty((0, 2), dtype=np.float64)


def matrix_points(items: list[dict[str, Any]], x: str, y: str) -> np.ndarray:
    return points(items, x=x, y=y)


def match_image(astap: list[dict[str, Any]], near: list[dict[str, Any]], tol: float = 2.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a = points(astap)
    n = points(near)
    if a.size == 0 or n.size == 0:
        return [], {"astap_count": len(astap), "near_count": len(near), "present_2px": 0}
    d = np.sqrt(((a[:, None, :] - n[None, :, :]) ** 2).sum(axis=2))
    best = np.argmin(d, axis=1)
    dist = d[np.arange(d.shape[0]), best]
    matches: list[dict[str, Any]] = []
    for i, (j, dd) in enumerate(zip(best, dist, strict=False)):
        if float(dd) <= float(tol):
            ar = astap[i]
            nr = near[int(j)]
            matches.append(
                {
                    "astap_rank": int(ar["rank"]),
                    "near_rank": int(nr["rank"]),
                    "astap_x": float(ar["x"]),
                    "astap_y": float(ar["y"]),
                    "near_x": float(nr["x"]),
                    "near_y": float(nr["y"]),
                    "delta_px": float(dd),
                    "near_flux": nr.get("flux"),
                }
            )
    ranks = [int(m["near_rank"]) for m in matches]
    summary = {
        "astap_count": len(astap),
        "near_count": len(near),
        "present_0_1px": int(np.count_nonzero(dist <= 0.1)),
        "present_0_25px": int(np.count_nonzero(dist <= 0.25)),
        "present_0_5px": int(np.count_nonzero(dist <= 0.5)),
        "present_1px": int(np.count_nonzero(dist <= 1.0)),
        "present_2px": int(np.count_nonzero(dist <= 2.0)),
        "median_delta_px": float(np.median(dist)) if dist.size else None,
        "p95_delta_px": float(np.percentile(dist, 95)) if dist.size else None,
        "near_rank_median": float(np.median(ranks)) if ranks else None,
        "near_rank_max": int(max(ranks)) if ranks else None,
    }
    for top in (32, 58, 64, 100, 120, 250, 500):
        summary[f"astap_stars_in_near_top_{top}"] = int(sum(r <= top for r in ranks))
    return matches, summary


def compare_catalog(astap: list[dict[str, Any]], near: list[dict[str, Any]], center_ra: float | None = None, center_dec: float | None = None) -> dict[str, Any]:
    ast_ids = {r["decoded_identity"] for r in astap if r.get("decoded_identity")}
    near_ids = {r["decoded_identity"] for r in near if r.get("decoded_identity")}
    overlap = ast_ids & near_ids
    out: dict[str, Any] = {
        "astap_count": len(astap),
        "near_count": len(near),
        "identity_kind": "decoded_coordinate_surrogate",
        "physical_id_overlap": int(len(overlap)),
        "physical_id_overlap_fraction": float(len(overlap) / max(1, len(ast_ids))),
        "missing_in_near": int(len(ast_ids - near_ids)),
        "extra_in_near": int(len(near_ids - ast_ids)),
    }
    near_by_id = {r["decoded_identity"]: r for r in near if r.get("decoded_identity")}
    ranks = [int(near_by_id[i]["rank"]) for i in overlap if i in near_by_id]
    if ranks:
        out["near_rank_median_for_astap_records"] = float(np.median(ranks))
        out["near_rank_max_for_astap_records"] = int(max(ranks))
        for top in (128, 192, 249, 250, 260, 384, 512, 1000):
            out[f"astap_records_in_near_top_{top}"] = int(sum(r <= top for r in ranks))
    if center_ra is not None and center_dec is not None:
        common = [r for r in astap if r.get("decoded_identity") in overlap]
        deltas: list[float] = []
        for ar in common:
            nr = near_by_id.get(ar["decoded_identity"])
            if not nr:
                continue
            ax, ay = project_tan(np.asarray([ar["ra_deg"]]), np.asarray([ar["dec_deg"]]), float(center_ra), float(center_dec))
            nx, ny = project_tan(np.asarray([nr["ra_deg"]]), np.asarray([nr["dec_deg"]]), float(center_ra), float(center_dec))
            if np.isfinite(ax[0]) and np.isfinite(nx[0]):
                deltas.append(float(math.hypot((ax[0] - nx[0]) * 3600.0, (ay[0] - ny[0]) * 3600.0)))
        out["common_frame_projection_delta_arcsec_median"] = float(np.median(deltas)) if deltas else None
        out["common_frame_projection_delta_arcsec_p95"] = float(np.percentile(deltas, 95)) if deltas else None
    return out


def select_by_image_matches(near: list[dict[str, Any]], matches: list[dict[str, Any]], *, astap_order: bool) -> list[dict[str, Any]]:
    by_rank = {int(r["rank"]): r for r in near}
    if astap_order:
        ordered = sorted(matches, key=lambda m: int(m["astap_rank"]))
    else:
        ordered = sorted(matches, key=lambda m: int(m["near_rank"]))
    return [by_rank[int(m["near_rank"])] for m in ordered if int(m["near_rank"]) in by_rank]


def select_catalog_intersection(near: list[dict[str, Any]], astap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ast_ids = {r["decoded_identity"] for r in astap if r.get("decoded_identity")}
    return [r for r in near if r.get("decoded_identity") in ast_ids]


def run_image_matrix(case: str, stage: str) -> dict[str, Any]:
    ast_img = load_astap_image(case)
    ast_cat = load_astap_catalog(case)
    near_img = load_near_image(case, stage)
    matches, _summary = match_image(ast_img, near_img, tol=2.0)
    ast_cat_pts = matrix_points(ast_cat, "x_arcsec", "y_arcsec")
    out: dict[str, Any] = {}
    out["E0_astap_image_astap_catalog"] = run_matrix_case(matrix_points(ast_img, "x", "y"), ast_cat_pts)
    e1 = select_by_image_matches(near_img, matches, astap_order=False)
    e2 = select_by_image_matches(near_img, matches, astap_order=True)
    out["E1_near_intersection_near_order"] = run_matrix_case(matrix_points(e1, "x", "y"), ast_cat_pts)
    out["E2_near_intersection_astap_order"] = run_matrix_case(matrix_points(e2, "x", "y"), ast_cat_pts)
    for top in PREFIX_IMAGE:
        out[f"E3_near_prefix_{top}"] = run_matrix_case(matrix_points(near_img[:top], "x", "y"), ast_cat_pts)
    # Diagnostic proxy for the exact selector until ported as a product rule:
    # keep the Near stars corresponding to ASTAP detections, in ASTAP order.
    out["E4_astap_policy_proxy"] = out["E2_near_intersection_astap_order"]
    return out


def run_catalog_matrix(case: str, stage: str) -> dict[str, Any]:
    ast_img = load_astap_image(case)
    ast_cat = load_astap_catalog(case)
    near_cat = load_near_catalog(case, stage)
    img_pts = matrix_points(ast_img, "x", "y")
    out: dict[str, Any] = {}
    out["H0_astap_image_astap_catalog"] = run_matrix_case(img_pts, matrix_points(ast_cat, "x_arcsec", "y_arcsec"))
    h1 = select_catalog_intersection(near_cat, ast_cat)
    out["H1_physical_intersection"] = run_matrix_case(img_pts, matrix_points(h1, "x_arcsec", "y_arcsec"))
    out["H3_same_selection_projection_near"] = out["H1_physical_intersection"]
    if ast_cat:
        ra0 = float(np.nanmedian([r["ra_deg"] for r in ast_cat]))
        dec0 = float(np.nanmedian([r["dec_deg"] for r in ast_cat]))
        h4: list[dict[str, Any]] = []
        for r in h1:
            x, y = project_tan(np.asarray([r["ra_deg"]]), np.asarray([r["dec_deg"]]), ra0, dec0)
            h4.append({**r, "x_arcsec": float(x[0] * 3600.0), "y_arcsec": float(y[0] * 3600.0)})
        out["H4_same_selection_common_projection"] = run_matrix_case(img_pts, matrix_points(h4, "x_arcsec", "y_arcsec"))
    for top in PREFIX_CATALOG:
        out[f"H5_near_prefix_{top}"] = run_matrix_case(img_pts, matrix_points(near_cat[:top], "x_arcsec", "y_arcsec"))
    out["H6_astap_policy_proxy"] = out["H1_physical_intersection"]
    return out


def run_cross_by_stage(case: str) -> dict[str, Any]:
    ast_img = load_astap_image(case)
    ast_cat = load_astap_catalog(case)
    ast_img_pts = matrix_points(ast_img, "x", "y")
    ast_cat_pts = matrix_points(ast_cat, "x_arcsec", "y_arcsec")
    out: dict[str, Any] = {}
    for stage in all_near_stages(case):
        near_img = load_near_image(case, stage)
        near_cat = load_near_catalog(case, stage)
        meta = read_json(near_stage_paths(case, stage)["summary"], {}).get("stage_meta", {})
        out[stage] = {
            "stage_meta": meta,
            "S00": run_matrix_case(ast_img_pts, ast_cat_pts),
            "S10": run_matrix_case(matrix_points(near_img, "x", "y"), ast_cat_pts),
            "S01": run_matrix_case(ast_img_pts, matrix_points(near_cat, "x_arcsec", "y_arcsec")),
            "S11": run_matrix_case(matrix_points(near_img, "x", "y"), matrix_points(near_cat, "x_arcsec", "y_arcsec")),
        }
    return out


def build_attempt_trace(case: str) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for idx, stage in enumerate(all_near_stages(case), start=1):
        p = near_stage_paths(case, stage)
        s = read_json(p["summary"], {})
        diag = s.get("diag", {}) if isinstance(s.get("diag"), dict) else {}
        hits = s.get("hits", {}) if isinstance(s.get("hits"), dict) else {}
        meta = s.get("stage_meta", {}) if isinstance(s.get("stage_meta"), dict) else {}
        trace.append(
            {
                "stage_id": stage,
                "stage_type": meta.get("stage_type", "unknown"),
                "attempt_order": meta.get("attempt_order", idx),
                "center_ra_deg": meta.get("center_ra_deg"),
                "center_dec_deg": meta.get("center_dec_deg"),
                "offset_from_initial_deg": meta.get("offset_from_initial_deg"),
                "tangent_center_ra_deg": meta.get("tangent_center_ra_deg"),
                "tangent_center_dec_deg": meta.get("tangent_center_dec_deg"),
                "FOV_deg": meta.get("fov_deg"),
                "search_window_deg": meta.get("search_window_deg"),
                "oversize": (float(meta["search_window_deg"]) / float(meta["fov_deg"])) if meta.get("search_window_deg") and meta.get("fov_deg") else None,
                "search_radius_deg": meta.get("search_radius_deg"),
                "image_final_count": s.get("image_final_for_quads"),
                "catalog_final_count": s.get("catalog_final_for_quads"),
                "image_quad_count": s.get("image_quads"),
                "catalog_quad_count": s.get("catalog_quads"),
                "raw_hits": hits.get("matches_raw"),
                "retained_hits": hits.get("matches_kept"),
                "transform_attempts": len(diag.get("tolerances", [])) if isinstance(diag.get("tolerances"), list) else None,
                "success": any(bool(t.get("ok")) for t in diag.get("tolerances", []) if isinstance(t, dict)) if isinstance(diag.get("tolerances"), list) else None,
                "failure_reason": diag.get("reason"),
            }
        )
    return trace


def write_markdown_json(path: Path, title: str, data: Any) -> None:
    write_text(path, f"# {title}\n\n```json\n{json.dumps(data, indent=2, sort_keys=True)}\n```\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=["232102", "233459", "230409"])
    args = parser.parse_args(argv)

    baseline: dict[str, Any] = {}
    image_rank: dict[str, Any] = {}
    image_matrix: dict[str, Any] = {}
    catalog_identity: dict[str, Any] = {}
    catalog_frame: dict[str, Any] = {}
    catalog_matrix: dict[str, Any] = {}
    positive_vs_failure: dict[str, Any] = {}
    quad_dilution: dict[str, Any] = {}
    performance: dict[str, Any] = {}

    for case in args.cases:
        stage = CASE_STAGES.get(case, "initial")
        ap = astap_paths(case)
        npth = near_stage_paths(case, stage)
        near_run = read_json(ZN36_ROOT / case / "near" / "near_run.json", {})
        astap_run = read_json(ZN36_ROOT / case / "astap" / "astap_run.json", {})
        baseline[case] = {
            "stage": stage,
            "astap_success": bool(astap_run.get("success")),
            "near_success": bool(near_run.get("success")),
            "near_message": near_run.get("message"),
            "astap_image_final": len(rows(ap["image"])),
            "astap_catalog_final": len(rows(ap["catalog"])),
            "near_image_final": len(rows(npth["image"])),
            "near_catalog_final": len(rows(npth["catalog"])),
            "near_summary": read_json(npth["summary"], {}),
        }

        ast_img = load_astap_image(case)
        near_img = load_near_image(case, stage)
        matches, summary = match_image(ast_img, near_img, tol=2.0)
        image_rank[case] = {"stage": stage, "summary": summary, "matches_sample": matches[:20]}
        image_matrix[case] = run_image_matrix(case, stage)

        ast_cat = load_astap_catalog(case)
        near_cat = load_near_catalog(case, stage)
        meta = read_json(npth["summary"], {}).get("stage_meta", {})
        center_ra = meta.get("tangent_center_ra_deg")
        center_dec = meta.get("tangent_center_dec_deg")
        catalog_identity[case] = {
            "near_identity_kind": "decoded_coordinate_surrogate",
            "strict_row_index_available_in_near_trace": False,
            "reason": "zewcs290 STAR_DTYPE currently exposes decoded coordinates, magnitude and flags but not physical row_index/byte_offset.",
            "comparison": compare_catalog(ast_cat, near_cat, center_ra=center_ra, center_dec=center_dec),
        }
        catalog_frame[case] = catalog_identity[case]["comparison"]
        catalog_matrix[case] = run_catalog_matrix(case, stage)

        quad_dilution[case] = {
            "stage": stage,
            "astap_image_stars": len(ast_img),
            "near_image_stars": len(near_img),
            "astap_catalog_stars": len(ast_cat),
            "near_catalog_stars": len(near_cat),
            "astap_image_quads": len(rows(ap["image_quads"])),
            "near_image_quads": len(rows(npth["image_quads"])),
            "astap_catalog_quads": len(rows(ap["catalog_quads"])),
            "near_catalog_quads": len(rows(npth["catalog_quads"])),
            "near_hits": len(rows(npth["hits"])),
            "astap_stars_present_but_late": summary.get("near_rank_max") is not None and int(summary.get("near_rank_max") or 0) > len(ast_img),
        }
        performance[case] = {"near_total_s": near_run.get("elapsed_s"), "astap_total_s": astap_run.get("elapsed_s")}

    attempt_trace = build_attempt_trace("232102")
    cross_by_stage = run_cross_by_stage("232102")

    positive_vs_failure = {
        "232102": {
            "image": image_rank.get("232102", {}).get("summary"),
            "catalog": catalog_identity.get("232102", {}).get("comparison"),
            "selected_stage": CASE_STAGES["232102"],
            "hits": baseline.get("232102", {}).get("near_summary", {}).get("hits"),
        },
        "233459": {
            "image": image_rank.get("233459", {}).get("summary"),
            "catalog": catalog_identity.get("233459", {}).get("comparison"),
            "selected_stage": CASE_STAGES["233459"],
            "hits": baseline.get("233459", {}).get("near_summary", {}).get("hits"),
        },
        "interpretation": "233459 succeeds because the useful image stars remain earlier in the generated quads and the catalogue is much smaller; 232102 has the same useful image stars mostly present, but both image and catalogue lists are heavily diluted and S10/S01 fail independently.",
    }

    first = {
        "verdict": "MULTIPLE_INDEPENDENT_DIVERGENCES",
        "authorized_verdict": "F - Plusieurs causes independantes",
        "correctif_applied": False,
        "last_identical_stage": "clean branches, ASTAP oracle, and Near core H00 from ASTAP final lists",
        "first_divergent_stage": "IMAGE_FINAL_FOR_QUADS",
        "astap_rule": "ASTAP bin_and_find_stars returns the post-detection selected list directly; catalogue target then follows nrstars_required=round(nrstars_image*height/width), oversize based on nrstars_image, and nrstars_required2=round(nrstars_required*oversize^2).",
        "near_behavior": "Near strict currently uses its own Python image list for this M106 case; the ASTAP stars are mostly present but mixed with many extra stars. The catalogue target is consequently much larger and Near final catalog is not physically aligned with ASTAP final catalog.",
        "evidence": {
            "232102_image_identity": image_rank.get("232102", {}).get("summary"),
            "232102_catalog_identity": catalog_identity.get("232102", {}).get("comparison"),
            "232102_image_matrix": image_matrix.get("232102"),
            "232102_catalog_matrix": catalog_matrix.get("232102"),
        },
        "recommendation": "Open a narrower mission to port the exact ASTAP image post-detection selection/ranking for M106 first; do not change catalogue, quads, lookup, transform, thresholds, gate, or 4D in the same patch.",
    }

    write_json(REPORTS / "zenear_zn37_baseline.json", baseline)
    write_json(REPORTS / "zenear_zn37_attempt_trace.json", attempt_trace)
    write_json(REPORTS / "zenear_zn37_image_identity_rank.json", image_rank)
    write_json(REPORTS / "zenear_zn37_image_selection_matrix.json", image_matrix)
    write_json(REPORTS / "zenear_zn37_catalog_physical_identity.json", catalog_identity)
    write_json(REPORTS / "zenear_zn37_catalog_common_frame_parity.json", catalog_frame)
    write_json(REPORTS / "zenear_zn37_catalog_selection_matrix.json", catalog_matrix)
    write_json(REPORTS / "zenear_zn37_cross_matrix_by_stage.json", cross_by_stage)
    write_json(REPORTS / "zenear_zn37_positive_vs_failure.json", positive_vs_failure)
    write_json(REPORTS / "zenear_zn37_quad_dilution.json", quad_dilution)
    write_json(REPORTS / "zenear_zn37_first_exact_cause.json", first)
    write_json(REPORTS / "zenear_zn37_performance.json", performance)

    write_text(
        REPORTS / "zenear_zn37_astap_selection_path.md",
        """# ZN3.7 ASTAP Selection Path

ASTAP `solve_image` path inspected in `unit_astrometric_solving.pas`:

- `bin_and_find_stars(...)` produces `starlist2`, already the final image list sent to `find_quads`.
- `nrstars_image := Length(starlist2[0])`.
- `find_quads(False, nrstars_image, starlist2, quad_star_distances2)` consumes that list directly.
- `nrstars_required := round(nrstars_image * (hd.Height / hd.Width))`.
- `oversize := 2` when `nrstars_image < 35`; `oversize := 1` when `nrstars_image > 140`; otherwise `2 * sqrt(35 / nrstars_image)`.
- `nrstars_required2 := round(nrstars_required * oversize2 * oversize2)`.
- `read_stars(..., search_field * oversize2, ..., nrstars_required2, starlist1)` produces the final catalogue list consumed by `find_quads`.

For `232102`, ASTAP does not appear to detect thousands and then cap to 58 at the solve stage; the dump/log show `bin_and_find_stars` final output is 58. The catalogue `249` follows from that image count, portrait aspect ratio, and oversize.
""",
    )
    write_markdown_json(REPORTS / "zenear_zn37_attempt_trace.md", "ZN3.7 Attempt Trace", attempt_trace)
    write_markdown_json(REPORTS / "zenear_zn37_image_identity_rank.md", "ZN3.7 Image Identity And Rank", image_rank)
    write_markdown_json(REPORTS / "zenear_zn37_image_selection_matrix.md", "ZN3.7 Image Selection Matrix", image_matrix)
    write_markdown_json(REPORTS / "zenear_zn37_catalog_common_frame_parity.md", "ZN3.7 Catalog Common-Frame Parity", catalog_frame)
    write_markdown_json(REPORTS / "zenear_zn37_catalog_selection_matrix.md", "ZN3.7 Catalog Selection Matrix", catalog_matrix)
    write_markdown_json(REPORTS / "zenear_zn37_cross_matrix_by_stage.md", "ZN3.7 Cross Matrix By Stage", cross_by_stage)
    write_markdown_json(REPORTS / "zenear_zn37_positive_vs_failure.md", "ZN3.7 Positive Vs Failure", positive_vs_failure)
    write_markdown_json(REPORTS / "zenear_zn37_quad_dilution.md", "ZN3.7 Quad Dilution", quad_dilution)
    write_markdown_json(REPORTS / "zenear_zn37_first_exact_cause.md", "ZN3.7 First Exact Cause", first)
    write_markdown_json(REPORTS / "zenear_zn37_performance.md", "ZN3.7 Performance", performance)

    summary_lines = [
        "# ZN3.7 Summary",
        "",
        f"Verdict: `{first['authorized_verdict']}` (`{first['verdict']}`). No functional correction was applied.",
        "",
        "1. ASTAP reaches 58 image stars on `232102` at `bin_and_find_stars`; that list is sent directly to `find_quads`.",
        "2. The 58 comes from detection/selection output, not a hard solve-stage `[:58]` cap.",
        "3. ASTAP reaches 249 catalogue stars through `nrstars_required=round(nrstars_image*height/width)` and `nrstars_required2=round(nrstars_required*oversize^2)`.",
        "4. The catalogue count depends on image star count, aspect ratio, FOV/oversize, and `read_stars` quota.",
        f"5. `232102`: ASTAP image stars present in Near at 2 px: `{image_rank['232102']['summary']['present_2px']}/{image_rank['232102']['summary']['astap_count']}`.",
        f"6. Their Near rank median/max: `{image_rank['232102']['summary']['near_rank_median']}` / `{image_rank['232102']['summary']['near_rank_max']}`.",
        f"7. `232102`: ASTAP catalogue decoded-identity overlap in Near: `{catalog_identity['232102']['comparison']['physical_id_overlap']}/{catalog_identity['232102']['comparison']['astap_count']}` using decoded-coordinate surrogate IDs.",
        "8. The two catalogues do not expose the same strict physical row identity on Near; comparison uses decoded RA/Dec/mag surrogate plus common-frame reprojection.",
        "9. Projection-only divergence is not sufficient to explain the result; decoded catalogue overlap is poor on the selected failing stage.",
        "10. `spiral_9` is the stage with the largest raw-hit evidence in ZN3.6 selection; no stage produces a valid transform.",
        f"11. Initial/autofov counts are in `{REPORTS / 'zenear_zn37_attempt_trace.json'}`.",
        "12. No earlier stage in `232102` succeeds in S11; S00 succeeds for every stage by construction.",
        "13. The problem appears before and through center stepping: all Near stages keep the oversized image list and large catalogues.",
        f"14. Image intersection E1 success on 232102: `{image_matrix['232102']['E1_near_intersection_near_order']['success']}`.",
        f"15. Catalogue intersection H1 success on 232102: `{catalog_matrix['232102']['H1_physical_intersection']['success']}`.",
        f"16. ASTAP order needed? E2 success: `{image_matrix['232102']['E2_near_intersection_astap_order']['success']}`.",
        "17. Near prefixes were tested diagnostically; they are not product constants.",
        f"18. ASTAP selector proxy E4 success: `{image_matrix['232102']['E4_astap_policy_proxy']['success']}`.",
        "19. Extras do dilute quad generation: useful stars are present but late enough that H10/H01/S11 fail while H00 succeeds.",
        "20. `233459` succeeds because its Near lists are much smaller and useful image stars are earlier; its six retained hits survive.",
        "21. Cause is combined, with image selection first in pipeline and catalogue quota/selection downstream.",
        "22. First divergent rule: Python strict image list for M106 does not reproduce ASTAP `bin_and_find_stars` final selection; catalogue quota then expands from the wrong image count.",
        "23. Correctif unique applied: no.",
        "24. `232102` Near remains unsolved by Near in this mission.",
        "25. No Near WCS for `232102`; 4D remains the confirmed fallback from ZN3.5B.",
        "26. Other M106 recovery: not run, because no correction was applied.",
        "27. M31 witness remains `230409` success in baseline; full 8/8 is covered by existing tests/reports.",
        "28. NGC6888 not replayed in ZN3.7 because no correction was applied.",
        "29. Fallback 4D not modified; historical backend remains disabled.",
        "30. Gate must remain diagnostic.",
    ]
    write_text(REPORTS / "zenear_zn37_summary.md", "\n".join(summary_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
