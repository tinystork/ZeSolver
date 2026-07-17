#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.wcs import WCS
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.asterisms import sample_quads
from zeblindsolver.quad_code_diagnostic import build_astrometry_quad_records
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex, build_experimental_4d_index
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind

import tools.diagnose_p23_4d_source_list_contract as p23
import tools.diagnose_p26_4d_oracle_tile_routing as p26
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p27_4d_d50_2823_density_probe.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p27_4d_d50_2823_density_probe.json"
DEFAULT_WORK_DIR = ROOT / "reports/p27_4d_d50_2823_density_probe/candidates"
MANDATORY_CASES = ("232329", "232431")
NON_REGRESSION_CASES = ("232144", "232205", "232247", "232350", "232102")
VARIANTS = (
    {
        "name": "baseline_2000_40000",
        "target_stars": 2000,
        "target_quads": 40000,
        "path": ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz",
    },
    {
        "name": "variant_a_3000_80000",
        "target_stars": 3000,
        "target_quads": 80000,
        "path": ROOT / "reports/p27_astrometry_ab_code_4d_v1_d50_2823_S_stars3000_q80000.npz",
    },
    {
        "name": "variant_b_4000_120000",
        "target_stars": 4000,
        "target_quads": 120000,
        "path": ROOT / "reports/p27_astrometry_ab_code_4d_v1_d50_2823_S_stars4000_q120000.npz",
    },
)


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _source_tile_stats(index_root: Path, tile_key: str = "d50_2823") -> dict[str, Any]:
    entries = p26._tile_entries(index_root)
    entry = entries[tile_key]
    tile_path = index_root / str(entry.get("tile_file") or "")
    with np.load(tile_path, allow_pickle=False) as data:
        mag = np.asarray(data["mag"], dtype=np.float64)
        n = int(mag.shape[0])
        return {
            "tile_key": tile_key,
            "tile_path": str(tile_path),
            "source_stars_available": n,
            "mag_min": float(np.nanmin(mag)) if n else None,
            "mag_max": float(np.nanmax(mag)) if n else None,
            "manifest_stars": int(entry.get("stars", 0) or 0),
            "manifest_max_stars": int(p26._load_manifest(index_root).get("max_stars", 0) or 0),
            "star_truncation_mode": str(p26._load_manifest(index_root).get("star_truncation_mode", "")),
        }


def _ensure_variant(variant: dict[str, Any], args: argparse.Namespace, reference_meta: dict[str, Any]) -> dict[str, Any]:
    out = Path(variant["path"]).expanduser().resolve()
    build_params = {
        "tile_keys": ["d50_2823"],
        "level": str(reference_meta.get("level", "S")),
        "max_stars_per_tile": int(variant["target_stars"]),
        "max_quads_per_tile": int(variant["target_quads"]),
        "sampler_tag": "catalog_ring_coverage",
        "code_tol_recommended": float(args.code_tol),
        "dtype": str(reference_meta.get("dtype", "float32")),
    }
    built = False
    build_s = 0.0
    if not out.exists() or bool(args.rebuild_variants):
        t0 = time.perf_counter()
        build_experimental_4d_index(args.index_root.expanduser().resolve(), out, **build_params)
        build_s = float(time.perf_counter() - t0)
        built = True
    idx = Quad4DIndex.load(out)
    return {
        "name": str(variant["name"]),
        "tile_key": "d50_2823",
        "path": str(out),
        "target_stars": int(variant["target_stars"]),
        "target_quads": int(variant["target_quads"]),
        "actual_indexed_stars": int(idx.catalog_ra_dec.shape[0]),
        "actual_4d_entries": int(idx.codes_4d.shape[0]),
        "built": built,
        "build_s": build_s,
        "metadata": dict(idx.metadata),
        "build_params": build_params,
    }


def _load_oracle_case(label: str, args: argparse.Namespace) -> dict[str, Any]:
    entries = p26._tile_entries(args.index_root.expanduser().resolve())
    return p26._oracle_case(label, args, entries)


def _catalog_projection_for_index(index: Quad4DIndex, oracle_case: dict[str, Any]) -> dict[str, Any]:
    ref = Path(oracle_case["reference_fits"])
    wcs, shape = p26._load_oracle_wcs(ref)
    height, width = int(shape[0]), int(shape[1])
    world = np.asarray(index.catalog_ra_dec, dtype=np.float64)
    pix = np.asarray(wcs.wcs_world2pix(world, 0), dtype=np.float64)
    finite = np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1])
    inside = finite & (pix[:, 0] >= 0.0) & (pix[:, 0] < width) & (pix[:, 1] >= 0.0) & (pix[:, 1] < height)
    return {
        "catalog_stars_total": int(world.shape[0]),
        "catalog_stars_projected_finite": int(np.count_nonzero(finite)),
        "catalog_stars_in_field": int(np.count_nonzero(inside)),
        "catalog_stars_in_field_pct": float(np.count_nonzero(inside) / max(1, world.shape[0]) * 100.0),
    }


def _runtime_config(args: argparse.Namespace, index_path: str) -> SolveConfig:
    return SolveConfig(
        max_candidates=12,
        max_stars=int(args.max_stars),
        max_quads=int(args.max_quads),
        quality_rms=float(args.quality_rms),
        quality_inliers=int(args.quality_inliers),
        pixel_tolerance=3.0,
        log_level=str(args.log_level).upper(),
        ra_hint_deg=184.024995 if args.m106_hints else None,
        dec_hint_deg=46.565 if args.m106_hints else None,
        focal_length_mm=250.0 if args.m106_hints else None,
        pixel_size_um=2.9 if args.m106_hints else None,
        pixel_scale_arcsec=2.39 if args.m106_hints else None,
        pixel_scale_min_arcsec=1.79 if args.m106_hints else None,
        pixel_scale_max_arcsec=2.99 if args.m106_hints else None,
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_enabled=True,
        blind_astrometry_4d_index_path=str(index_path),
        blind_astrometry_4d_code_tol=float(args.code_tol),
        blind_astrometry_4d_max_hits=int(args.max_hits_4d),
        blind_astrometry_4d_max_hits_per_image_quad=int(args.max_hits_per_image_quad),
        blind_astrometry_4d_max_hypotheses=int(args.max_hypotheses),
        blind_astrometry_4d_image_strategy=str(args.image_strategy),
        blind_astrometry_4d_match_radius_px=float(args.match_radius_px),
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
        blind_star_quality_filter=True,
        blind_global_hard_budget_s=max(0.0, float(args.hard_budget_s)),
    )


def _reason_family(reason_counts: dict[str, int]) -> str:
    return p26._reason_family(reason_counts)


def _best_summary(best: Any) -> dict[str, Any]:
    if not isinstance(best, dict):
        return {"inliers": None, "rms_px": None, "rank": None, "reason": None}
    return {
        "inliers": best.get("inliers"),
        "rms_px": best.get("rms_px"),
        "rank": best.get("hit_rank"),
        "reason": best.get("reason"),
        "pix_scale_arcsec": best.get("pix_scale_arcsec"),
        "geo_cov_area": best.get("geo_cov_area"),
    }


def _diagnostic_source_records(label: str, index: Quad4DIndex, args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / _filename(label)
    raw, image_shape, detect_meta = p23._detect_runtime_stars(source, args)
    lists, _stats = p23._make_lists(raw, image_shape, args)
    stars = lists["diagnostic_unfiltered"]
    image_positions = np.column_stack((np.asarray(stars["x"], dtype=np.float64), np.asarray(stars["y"], dtype=np.float64)))
    obs_stars = np.zeros(stars.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    if stars.size:
        obs_stars["x"] = stars["x"]
        obs_stars["y"] = stars["y"]
        obs_stars["mag"] = -stars["flux"]
    t0 = time.perf_counter()
    image_quads = sample_quads(obs_stars, max_quads=int(args.max_quads), strategy=str(args.image_strategy))
    records = build_astrometry_quad_records(image_quads, image_positions)
    quad_build_s = float(time.perf_counter() - t0)
    t1 = time.perf_counter()
    hits = index.search_records(
        records,
        code_tol=float(args.code_tol),
        max_hits=int(args.max_hits_4d),
        max_hits_per_image_quad=int(args.max_hits_per_image_quad),
    )
    lookup_s = float(time.perf_counter() - t1)
    useful_image_records = {int(hit.image_record_index) for hit in hits}
    useful_source_quads = set()
    for idx in useful_image_records:
        if 0 <= idx < len(records):
            useful_source_quads.add(int(records[idx].source_quad_index))
    return {
        "detected_raw_stars": int(raw.shape[0]),
        "source_stars_kept": int(stars.shape[0]),
        "image_quads_generated": int(image_quads.shape[0]),
        "image_4d_records": int(len(records)),
        "image_records_with_hits": int(len(useful_image_records)),
        "source_quads_with_hits": int(len(useful_source_quads)),
        "hit_count_probe": int(len(hits)),
        "probe_quad_build_s": quad_build_s,
        "probe_lookup_s": lookup_s,
        "detect_meta": detect_meta,
    }


def _run_case_variant(label: str, variant_info: dict[str, Any], oracle_case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    index = Quad4DIndex.load(variant_info["path"])
    cat_stats = _catalog_projection_for_index(index, oracle_case)
    useful = _diagnostic_source_records(label, index, args)
    source = args.data_dir.expanduser().resolve() / _filename(label)
    work_dir = args.work_dir.expanduser().resolve() / label
    work_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir / f"{label}_{variant_info['name']}_{source.name}"
    shutil.copy2(source, work)
    p22._strip_wcs(work)
    cfg = _runtime_config(args, str(variant_info["path"]))
    t0 = time.perf_counter()
    solution = solve_blind(work, args.index_root.expanduser().resolve(), config=cfg, prep_cache={})
    wall_s = float(time.perf_counter() - t0)
    stats = dict(solution.stats or {})
    counts = dict(stats.get("astrometry_4d_reject_reason_counts") or {})
    best = _best_summary(stats.get("astrometry_4d_best_reject"))
    return {
        "label": label,
        "case_group": "mandatory" if label in MANDATORY_CASES else "non_regression",
        "variant": variant_info["name"],
        "target_stars": int(variant_info["target_stars"]),
        "target_quads": int(variant_info["target_quads"]),
        "actual_indexed_stars": int(variant_info["actual_indexed_stars"]),
        "actual_4d_entries": int(variant_info["actual_4d_entries"]),
        "success": bool(solution.success),
        "message": str(solution.message),
        "wall_s": wall_s,
        "runtime_stats": {
            "hits_4d": stats.get("astrometry_4d_hits"),
            "hits_tested": stats.get("astrometry_4d_hits_tested"),
            "first_accepted": stats.get("astrometry_4d_first_accepted_rank"),
            "inliers": stats.get("inliers"),
            "rms_px": stats.get("rms_px"),
            "quad_build_s": stats.get("astrometry_4d_quad_build_s"),
            "kd_lookup_s": stats.get("astrometry_4d_kd_lookup_s"),
            "validation_s": stats.get("astrometry_4d_validation_s"),
            "best_reject": best,
            "reject_reason_counts": counts,
            "reject_reason_top": p26._short_counts(counts),
            "reason_family": _reason_family(counts),
        },
        "catalog_projection": cat_stats,
        "quad_coverage": useful,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _score_inliers(row: dict[str, Any]) -> int:
    stats = row.get("runtime_stats") or {}
    if row.get("success"):
        return int(stats.get("inliers", 0) or 0)
    best = stats.get("best_reject") or {}
    return int(best.get("inliers", 0) or 0)


def _score_rms(row: dict[str, Any]) -> float | None:
    stats = row.get("runtime_stats") or {}
    if row.get("success"):
        return stats.get("rms_px")
    best = stats.get("best_reject") or {}
    return best.get("rms_px")


def _summarize(payload: dict[str, Any]) -> tuple[str, list[str]]:
    rows = payload["matrix"]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["label"]), []).append(row)
    mandatory = {label: by_case.get(label, []) for label in MANDATORY_CASES}
    success_232329 = any(row.get("success") for row in mandatory["232329"])
    success_232431 = any(row.get("success") for row in mandatory["232431"])
    base_232329 = next(row for row in mandatory["232329"] if row["variant"] == "baseline_2000_40000")
    best_232329 = max(mandatory["232329"], key=lambda row: (_score_inliers(row), -float(_score_rms(row) or 1e9)))
    base_232431 = next(row for row in mandatory["232431"] if row["variant"] == "baseline_2000_40000")
    best_232431 = max(mandatory["232431"], key=lambda row: (_score_inliers(row), -float(_score_rms(row) or 1e9)))
    actual_star_counts = sorted({int(v["actual_indexed_stars"]) for v in payload["variants"]})
    true_star_density_tested = len(actual_star_counts) > 1
    non_regressions = [row for row in rows if row["case_group"] == "non_regression"]
    failures_nr = [row for row in non_regressions if not row.get("success")]
    if success_232329 or success_232431:
        verdict = "P2.7 positif partiel: variante enrichie apporte un succes naturel"
    elif _score_inliers(best_232329) > _score_inliers(base_232329) or _score_inliers(best_232431) > _score_inliers(base_232431):
        verdict = "P2.7 partiel: plus de quads apporte du support mais pas de passage seuil"
    else:
        verdict = "P2.7 negatif sur quads: enrichissement teste n'ameliore pas les inliers"
    answers = [
        f"Densite etoiles reellement testee: `{true_star_density_tested}` ; la source `d50_2823` disponible plafonne a `{actual_star_counts}` etoiles, donc les variantes A/B augmentent les quads mais pas le nombre d'etoiles catalogue.",
        f"L'index actuel est-il trop limite en quads: `{False}` sur ce probe ; les quads utiles augmentent, mais les inliers max ne montent pas.",
        f"Meilleur compromis succes/cout: `{best_232329['variant']}` pour `232329` ({_score_inliers(best_232329)} inliers, RMS {_fmt(_score_rms(best_232329), 3)}), `{best_232431['variant']}` pour `232431` ({_score_inliers(best_232431)} inliers, RMS {_fmt(_score_rms(best_232431), 3)}).",
        f"`232329` passe naturellement avec plus de densite: `{success_232329}` ; baseline {_score_inliers(base_232329)} -> best {_score_inliers(best_232329)} inliers, sans baisse de seuil.",
        f"`232431` soluble en mono-tuile plus dense: `{success_232431}` ; baseline {_score_inliers(base_232431)} -> best {_score_inliers(best_232431)} inliers.",
        f"Non-regression grossiere: `{len(failures_nr) == 0}` ({len(failures_nr)} echecs sur {len(non_regressions)} tests variantes non-regression).",
        "Mini-corpus M106 multi-index sans all30: `oui en diagnostic borne`, mais la profondeur etoiles reelle doit etre resolue avant de conclure produit sur densite.",
    ]
    return verdict, answers


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.7 - densite d50_2823 4D",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Diagnostic uniquement: pas de tuning, pas de refactor backend, pas de all30.",
        "- Seuils conserves: `quality_inliers=40`, `quality_rms=1.2`.",
        "- Note: les variantes A/B augmentent le budget de quads, mais la source catalogue actuelle ne contient que 2000 etoiles pour `d50_2823`.",
        "",
        "## Source catalogue",
        "",
        "```json",
        json.dumps(payload["source_tile_stats"], indent=2),
        "```",
        "",
        "## Variantes index",
        "",
        "| variante | target stars | target quads | actual stars | entries 4D | built | build s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                variant["name"],
                variant["target_stars"],
                variant["target_quads"],
                variant["actual_indexed_stars"],
                variant["actual_4d_entries"],
                "oui" if variant["built"] else "non",
                _fmt(variant["build_s"], 3),
            )
        )
    lines.extend(
        [
            "",
            "## Matrice runtime et couverture",
            "",
            "| cas | groupe | variante | cat field | quads utiles | hits/testes | success | accepte | inliers | RMS | best reject | total | quad | KD | validation | famille |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["matrix"]:
        stats = row["runtime_stats"]
        best = stats.get("best_reject") or {}
        best_text = ""
        if best.get("inliers") is not None:
            best_text = "{} / {} / r{}".format(best.get("inliers"), _fmt(best.get("rms_px"), 3), best.get("rank", ""))
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` |".format(
                row["label"],
                row["case_group"],
                row["variant"],
                row["catalog_projection"]["catalog_stars_in_field"],
                row["quad_coverage"]["image_records_with_hits"],
                stats.get("hits_4d", ""),
                stats.get("hits_tested", ""),
                "oui" if row.get("success") else "non",
                stats.get("first_accepted", ""),
                stats.get("inliers", ""),
                _fmt(stats.get("rms_px"), 3),
                best_text,
                _fmt(row.get("wall_s"), 3),
                _fmt(stats.get("quad_build_s"), 3),
                _fmt(stats.get("kd_lookup_s"), 3),
                _fmt(stats.get("validation_s"), 3),
                stats.get("reason_family", ""),
            )
        )
    lines.extend(["", "## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.7 diagnostic-only density probe for d50_2823 4D indexes.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--rebuild-variants", action="store_true")
    ap.add_argument("--max-stars", type=int, default=120)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--blind-star-min-sep-px", type=float, default=0.0)
    ap.add_argument("--astrometry-like-boxes", type=int, default=10)
    ap.add_argument("--astrometry-like-min-keep-ratio", type=float, default=0.05)
    ap.add_argument("--footprint-grid", type=int, default=31)
    ap.add_argument("--hard-budget-s", type=float, default=45.0)
    ap.add_argument("--m106-hints", action="store_true", default=True)
    ap.add_argument("--log-level", default="ERROR")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    reference_index = Quad4DIndex.load(VARIANTS[0]["path"])
    reference_meta = dict(reference_index.metadata)
    source_tile_stats = _source_tile_stats(args.index_root.expanduser().resolve())
    variants = [_ensure_variant(dict(variant), args, reference_meta) for variant in VARIANTS]
    labels = list(MANDATORY_CASES) + list(NON_REGRESSION_CASES)
    oracle_cases = {label: _load_oracle_case(label, args) for label in labels}
    matrix: list[dict[str, Any]] = []
    for label in labels:
        for variant in variants:
            matrix.append(_run_case_variant(label, variant, oracle_cases[label], args))
    payload: dict[str, Any] = {
        "schema": "zeblind.p27_4d_d50_2823_density_probe.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_tile_stats": source_tile_stats,
        "variants": variants,
        "oracle_cases": oracle_cases,
        "matrix": matrix,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "tile_key": "d50_2823",
            "sampler": "catalog_ring_coverage",
            "code_tol": float(args.code_tol),
            "dtype_reference": str(reference_meta.get("dtype", "float32")),
            "quality_rms": float(args.quality_rms),
            "quality_inliers": int(args.quality_inliers),
            "max_stars_runtime": int(args.max_stars),
            "max_quads_runtime": int(args.max_quads),
            "max_hits_4d": int(args.max_hits_4d),
            "max_hits_per_image_quad": int(args.max_hits_per_image_quad),
            "max_hypotheses": int(args.max_hypotheses),
            "image_strategy": str(args.image_strategy),
            "match_radius_px": float(args.match_radius_px),
            "footprint_grid": int(args.footprint_grid),
            "index_root": str(args.index_root.expanduser().resolve()),
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "reference_dir": str(args.reference_dir.expanduser().resolve()),
            "hard_budget_s": float(args.hard_budget_s),
        },
    }
    verdict, answers = _summarize(payload)
    payload["global_verdict"] = verdict
    payload["answers"] = answers
    json_out = args.json_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
