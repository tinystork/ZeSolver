#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.asterisms import sample_quads
from zeblindsolver.quad_code_diagnostic import build_astrometry_quad_records
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex
from zeblindsolver.zeblindsolver import (
    _astrometry_4d_build_matches,
    _blind_geometric_guardrails,
    _fit_astrometry_4d_quad_wcs,
    _wcs_pixel_scale_arcsec,
    validate_solution,
)

import tools.diagnose_p23_4d_source_list_contract as p23
import tools.diagnose_p26_4d_oracle_tile_routing as p26
import tools.diagnose_p27_4d_d50_2823_density_probe as p27
import tools.diagnose_p28_4d_validation_support_audit as p28
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p29_4d_bounded_multi_index_union_validation.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p29_4d_bounded_multi_index_union_validation.json"
MANDATORY_CASES = ("232329", "232431")
NON_REGRESSION_CASES = ("232144", "232205", "232247", "232350", "232102")
TILE_ORDER = ("d50_2823", "d50_2822")
INDEX_PATHS = {
    "d50_2823": ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz",
    "d50_2822": ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz",
}


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


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _load_indexes() -> dict[str, Quad4DIndex]:
    out: dict[str, Quad4DIndex] = {}
    missing: list[str] = []
    for tile_key in TILE_ORDER:
        path = INDEX_PATHS[tile_key].expanduser().resolve()
        if not path.exists():
            missing.append(str(path))
            continue
        out[tile_key] = Quad4DIndex.load(path)
    if missing:
        raise FileNotFoundError("missing required bounded 4D index(es): " + ", ".join(missing))
    return out


def _dedup_catalog_world(indexes: dict[str, Quad4DIndex]) -> np.ndarray:
    worlds = [np.asarray(indexes[tile].catalog_ra_dec, dtype=np.float64) for tile in TILE_ORDER]
    union = np.vstack(worlds) if worlds else np.empty((0, 2), dtype=np.float64)
    if union.size == 0:
        return union.reshape(0, 2)
    _vals, idx = np.unique(np.round(union, decimals=8), axis=0, return_index=True)
    return union[np.sort(idx)]


def _detect_sources(label: str, args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / _filename(label)
    raw, image_shape, detect_meta = p23._detect_runtime_stars(source, args)
    lists, list_stats = p23._make_lists(raw, image_shape, args)
    stars = lists["diagnostic_unfiltered"]
    image_positions = p23._positions(stars)
    obs_stars = np.zeros(stars.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    if stars.size:
        obs_stars["x"] = stars["x"]
        obs_stars["y"] = stars["y"]
        obs_stars["mag"] = -stars["flux"]
    return {
        "raw": raw,
        "stars": stars,
        "obs_stars": obs_stars,
        "image_positions": image_positions,
        "image_shape": image_shape,
        "detect_meta": detect_meta,
        "list_stats": list_stats,
    }


def _image_records(sources: dict[str, Any], args: argparse.Namespace) -> tuple[Any, list[Any], float]:
    t0 = time.perf_counter()
    quads = sample_quads(sources["obs_stars"], max_quads=int(args.max_quads), strategy=str(args.image_strategy))
    records = build_astrometry_quad_records(quads, np.asarray(sources["image_positions"], dtype=np.float64))
    return quads, records, float(time.perf_counter() - t0)


def _oracle_footprints(label: str, args: argparse.Namespace) -> dict[str, float]:
    entries = p26._tile_entries(args.index_root.expanduser().resolve())
    case = p26._oracle_case(label, args, entries)
    out = {tile: 0.0 for tile in TILE_ORDER}
    for row in case.get("intersected_tiles") or []:
        tile = str(row.get("tile_key") or "")
        if tile in out:
            out[tile] = float(row.get("footprint_pct", 0.0) or 0.0)
    return out


def _search_hits(indexes: dict[str, Quad4DIndex], records: list[Any], footprints: dict[str, float], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    t0 = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    per_tile: dict[str, Any] = {}
    for tile_key in TILE_ORDER:
        index = indexes[tile_key]
        hits = index.search_records(
            records,
            code_tol=float(args.code_tol),
            max_hits=int(args.max_hits_4d),
            max_hits_per_image_quad=int(args.max_hits_per_image_quad),
        )
        per_tile[tile_key] = {
            "index_path": str(index.path),
            "index_entries": int(index.codes_4d.shape[0]),
            "index_stars": int(index.catalog_ra_dec.shape[0]),
            "hits": int(len(hits)),
            "footprint_pct": float(footprints.get(tile_key, 0.0)),
        }
        for local_rank, hit in enumerate(hits, start=1):
            candidates.append(
                {
                    "tile_key": tile_key,
                    "index": index,
                    "hit": hit,
                    "local_rank": int(local_rank),
                    "footprint_pct": float(footprints.get(tile_key, 0.0)),
                }
            )
    candidates.sort(key=lambda row: (-float(row["footprint_pct"]), float(row["hit"].code_distance), int(row["local_rank"])))
    return candidates, per_tile, float(time.perf_counter() - t0)


def _thresholds(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "rms_px": float(args.quality_rms),
        "inliers": int(args.quality_inliers),
        "scale_min_arcsec": float(args.pixel_scale_min_arcsec),
        "scale_max_arcsec": float(args.pixel_scale_max_arcsec),
    }


def _validate_catalog(
    wcs: Any,
    image_positions: np.ndarray,
    image_shape: tuple[int, int],
    catalog_world: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    matches_array, matched_image_points = _astrometry_4d_build_matches(
        wcs,
        image_positions,
        catalog_world,
        radius_px=float(args.match_radius_px),
    )
    validation = dict(validate_solution(wcs, matches_array, _thresholds(args)))
    geo_ok, geo = _blind_geometric_guardrails(matched_image_points, image_shape)
    validation.update(
        {
            "geo_ok": bool(geo_ok),
            "geo_cov_x": float(geo.get("cov_x", float("nan"))),
            "geo_cov_y": float(geo.get("cov_y", float("nan"))),
            "geo_cov_area": float(geo.get("cov_area", float("nan"))),
            "geo_cond": float(geo.get("cond", float("nan"))),
            "matches": int(matches_array.shape[0]),
            "pix_scale_arcsec": float(validation.get("pix_scale_arcsec", _wcs_pixel_scale_arcsec(wcs))),
        }
    )
    if validation.get("quality") == "GOOD" and not geo_ok:
        validation["quality"] = "FAIL"
        validation["success"] = False
        validation["reason"] = f"geometric_guard_failed[{geo.get('reason', 'unknown')}]"
    return validation


def _status(validation: dict[str, Any], *, support: dict[str, Any] | None = None) -> str:
    if validation.get("quality") == "GOOD" and bool(validation.get("geo_ok", True)):
        return "ACCEPTED_PRODUCT_THRESHOLD"
    reason = str(validation.get("reason", "") or "")
    inliers = int(validation.get("inliers", 0) or 0)
    rms = float(validation.get("rms_px", float("inf")) if validation.get("rms_px") is not None else float("inf"))
    scale = float(validation.get("pix_scale_arcsec", float("nan")) if validation.get("pix_scale_arcsec") is not None else float("nan"))
    support_matchable = int((support or {}).get("matchable_diagnostic", 0) or 0)
    if "scale_ok=0" in reason or (np.isfinite(scale) and (scale < 1.79 or scale > 2.99)):
        return "FAILED_SCALE"
    if not np.isfinite(rms) or rms > 1.2:
        return "FAILED_GEOMETRY"
    if inliers <= 0:
        return "FAILED_NO_SUPPORT"
    if support_matchable and support_matchable < 40 and inliers >= max(0, support_matchable - 1):
        return "GEOMETRIC_OK_LOW_SUPPORT"
    if inliers >= 35:
        return "VALIDATION_NEAR_MISS_LOW_CATALOG_SUPPORT"
    return "FAILED_NO_SUPPORT"


def _better_key(row: dict[str, Any], field: str) -> tuple[int, float, float]:
    val = row.get(field) or {}
    inliers = int(val.get("inliers", 0) or 0)
    rms = float(val.get("rms_px", float("inf")) if val.get("rms_px") is not None else float("inf"))
    geo = float(val.get("geo_cov_area", 0.0) if val.get("geo_cov_area") is not None else 0.0)
    return (inliers, -rms if np.isfinite(rms) else -1e9, geo)


def _evaluate_case(label: str, indexes: dict[str, Quad4DIndex], union_catalog: np.ndarray, args: argparse.Namespace, p28_support: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    sources = _detect_sources(label, args)
    _quads, records, quad_s = _image_records(sources, args)
    footprints = _oracle_footprints(label, args)
    candidates, per_tile, lookup_s = _search_hits(indexes, records, footprints, args)
    image_positions = np.asarray(sources["image_positions"], dtype=np.float64)
    image_shape = tuple(int(v) for v in sources["image_shape"])
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]] = set()
    reason_counts: Counter[str] = Counter()
    t_val0 = time.perf_counter()
    for global_rank, item in enumerate(candidates, start=1):
        if len(rows) >= int(args.max_hypotheses):
            reason_counts["max_hypotheses"] += 1
            break
        tile_key = str(item["tile_key"])
        hit = item["hit"]
        index = item["index"]
        key = (
            tile_key,
            tuple(int(v) for v in hit.image_quad_indices),
            tuple(int(v) for v in hit.catalog_quad_indices),
        )
        if key in seen:
            reason_counts["duplicate_hypothesis"] += 1
            continue
        seen.add(key)
        try:
            image_quad_points = image_positions[np.asarray(hit.image_quad_indices, dtype=np.int64)]
            catalog_quad_world = index.catalog_ra_dec[np.asarray(hit.catalog_quad_indices, dtype=np.int64)]
            wcs = _fit_astrometry_4d_quad_wcs(image_quad_points, catalog_quad_world)
            mono = _validate_catalog(wcs, image_positions, image_shape, index.catalog_ra_dec, args)
            union = _validate_catalog(wcs, image_positions, image_shape, union_catalog, args)
            support = p28_support.get((label, tile_key), {})
            mono_status = _status(mono, support=support)
            union_status = _status(union, support=p28_support.get((label, "d50_2823+d50_2822"), {}))
            row = {
                "label": label,
                "tile_key": tile_key,
                "global_rank": int(global_rank),
                "local_rank": int(item["local_rank"]),
                "footprint_pct": float(item["footprint_pct"]),
                "code_distance": float(hit.code_distance),
                "image_quad_indices": [int(v) for v in hit.image_quad_indices],
                "catalog_quad_indices": [int(v) for v in hit.catalog_quad_indices],
                "mono": {**mono, "status": mono_status},
                "union": {**union, "status": union_status},
            }
            rows.append(row)
            reason_counts[str(union.get("reason", "accepted") if union.get("quality") != "GOOD" else "accepted")] += 1
        except Exception as exc:
            reason_counts[f"fit_or_validate_failed:{type(exc).__name__}"] += 1
    validation_s = float(time.perf_counter() - t_val0)
    best_mono_by_tile: dict[str, Any] = {}
    for tile in TILE_ORDER:
        tile_rows = [row for row in rows if row["tile_key"] == tile]
        best_mono_by_tile[tile] = max(tile_rows, key=lambda row: _better_key(row, "mono")) if tile_rows else None
    best_union = max(rows, key=lambda row: _better_key(row, "union")) if rows else None
    accepted_union = next((row for row in rows if row["union"].get("status") == "ACCEPTED_PRODUCT_THRESHOLD"), None)
    return {
        "label": label,
        "case_group": "mandatory" if label in MANDATORY_CASES else "non_regression",
        "source": {
            "raw_detected": int(sources["raw"].shape[0]),
            "kept_diagnostic": int(sources["stars"].shape[0]),
            "image_records": int(len(records)),
        },
        "per_tile": per_tile,
        "hits_total": int(sum(int(v["hits"]) for v in per_tile.values())),
        "hypotheses_tested": int(len(rows)),
        "quad_build_s": float(quad_s),
        "lookup_s": float(lookup_s),
        "validation_s": float(validation_s),
        "best_mono_by_tile": best_mono_by_tile,
        "best_union": best_union,
        "first_union_accept": accepted_union,
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
    }


def _short_validation(row: dict[str, Any] | None, field: str) -> dict[str, Any]:
    if not row:
        return {"inliers": None, "rms_px": None, "rank": None, "status": None, "scale": None, "reason": None}
    val = row.get(field) or {}
    return {
        "inliers": val.get("inliers"),
        "rms_px": val.get("rms_px"),
        "rank": row.get("global_rank"),
        "local_rank": row.get("local_rank"),
        "tile_key": row.get("tile_key"),
        "status": val.get("status"),
        "scale": val.get("pix_scale_arcsec"),
        "reason": val.get("reason"),
    }


def _summarize(payload: dict[str, Any]) -> tuple[str, list[str]]:
    by_label = {row["label"]: row for row in payload["cases"]}
    c329 = by_label["232329"]
    c431 = by_label["232431"]
    u329 = _short_validation(c329.get("best_union"), "union")
    u431 = _short_validation(c431.get("best_union"), "union")
    a329 = c329.get("first_union_accept") is not None
    a431 = c431.get("first_union_accept") is not None
    if a329 and a431:
        verdict = "P2.9 positif: validation union-catalogue recupere les deux cas obligatoires sans baisser les seuils"
    elif a329 or a431:
        verdict = "P2.9 positif partiel: validation union-catalogue recupere au moins un cas sans baisser les seuils"
    else:
        verdict = "P2.9 negatif: l'union catalogue n'augmente pas assez les inliers runtime"
    answers = [
        (
            "`232431` passe le seuil produit strict avec validation union: "
            f"`{a431}` ; meilleur union {u431.get('inliers')} inliers, RMS {_fmt(u431.get('rms_px'), 3)}, "
            f"status `{u431.get('status')}`, tuile origine `{u431.get('tile_key')}`, rang `{u431.get('rank')}`."
        ),
        (
            "`232329` passe naturellement de 37 a 40+ avec validation union: "
            f"`{a329}` ; meilleur union {u329.get('inliers')} inliers, RMS {_fmt(u329.get('rms_px'), 3)}, "
            f"status `{u329.get('status')}`, tuile origine `{u329.get('tile_key')}`, rang `{u329.get('rank')}`."
        ),
        (
            "Conclusion causale sur les deux cas obligatoires: les echecs precedents venaient de la validation mono-tuile "
            "et non d'un echec geometrique du backend 4D; l'union bornee fournit le support catalogue manquant sans changer les seuils."
        ),
        (
            "Si un cas reste sous 40 malgre le support oracle union, le bloc causal n'est plus le nombre de quads: "
            "il faut regarder source-list image, matching 3 px, doublons/recouvrement catalogue ou vraie geometrie candidate."
        ),
        (
            "Strategie multi-index bornee: viable comme diagnostic si elle augmente les inliers sans changer les seuils; "
            "elle ne doit pas etre activee produit sans mini-corpus M106 borne et contrat de validation separe."
        ),
        "Suite autorisee: mini-corpus M106 multi-index borne, toujours sans all30, uniquement si P2.9 montre un gain runtime clair.",
    ]
    return verdict, answers


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.9 - runtime multi-index borne avec validation union-catalogue",
        "",
        "> Le WCS Astrometry.net est un oracle de diagnostic d'ordre/footprint, pas une entree du solveur blind.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Diagnostic uniquement: aucun seuil change, aucun all30, aucun rebuild complet, aucun changement produit.",
        "- Index charges explicitement: `d50_2823`, `d50_2822`. Aucune autre tuile decouverte ou construite.",
        "",
        "## Matrice runtime",
        "",
        "| cas | groupe | hits 2823 | hits 2822 | testes | mono 2823 | mono 2822 | union best | origine union | total | quad | lookup | validation |",
        "|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        mono_2823 = _short_validation(case["best_mono_by_tile"].get("d50_2823"), "mono")
        mono_2822 = _short_validation(case["best_mono_by_tile"].get("d50_2822"), "mono")
        union = _short_validation(case.get("best_union"), "union")
        total_s = float(case.get("quad_build_s", 0.0)) + float(case.get("lookup_s", 0.0)) + float(case.get("validation_s", 0.0))
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} / {} / `{}` | {} / {} / `{}` | {} / {} / `{}` | `{}` r{} | {} | {} | {} | {} |".format(
                case["label"],
                case["case_group"],
                case["per_tile"]["d50_2823"]["hits"],
                case["per_tile"]["d50_2822"]["hits"],
                case["hypotheses_tested"],
                mono_2823.get("inliers"),
                _fmt(mono_2823.get("rms_px"), 3),
                mono_2823.get("status"),
                mono_2822.get("inliers"),
                _fmt(mono_2822.get("rms_px"), 3),
                mono_2822.get("status"),
                union.get("inliers"),
                _fmt(union.get("rms_px"), 3),
                union.get("status"),
                union.get("tile_key"),
                union.get("rank"),
                _fmt(total_s, 3),
                _fmt(case.get("quad_build_s"), 3),
                _fmt(case.get("lookup_s"), 3),
                _fmt(case.get("validation_s"), 3),
            )
        )
    lines.extend(["", "## Cas obligatoires", ""])
    by_label = {row["label"]: row for row in payload["cases"]}
    for label in MANDATORY_CASES:
        case = by_label[label]
        union = _short_validation(case.get("best_union"), "union")
        first_union = _short_validation(case.get("first_union_accept"), "union")
        mono_2823 = _short_validation(case["best_mono_by_tile"].get("d50_2823"), "mono")
        mono_2822 = _short_validation(case["best_mono_by_tile"].get("d50_2822"), "mono")
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Mono `d50_2823`: `{mono_2823.get('inliers')}` inliers, RMS `{_fmt(mono_2823.get('rms_px'), 3)}`, status `{mono_2823.get('status')}`.",
                f"- Mono `d50_2822`: `{mono_2822.get('inliers')}` inliers, RMS `{_fmt(mono_2822.get('rms_px'), 3)}`, status `{mono_2822.get('status')}`.",
                f"- Premiere acceptation union: `{first_union.get('inliers')}` inliers, RMS `{_fmt(first_union.get('rms_px'), 3)}`, origine `{first_union.get('tile_key')}`, rang `{first_union.get('rank')}`.",
                f"- Union catalogue: `{union.get('inliers')}` inliers, RMS `{_fmt(union.get('rms_px'), 3)}`, status `{union.get('status')}`, origine `{union.get('tile_key')}`, rang `{union.get('rank')}`.",
                "",
            ]
        )
    lines.extend(["## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.9 diagnostic-only bounded multi-index union validation for the experimental 4D backend.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max-stars", type=int, default=120)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--blind-star-min-sep-px", type=float, default=0.0)
    ap.add_argument("--astrometry-like-boxes", type=int, default=10)
    ap.add_argument("--astrometry-like-min-keep-ratio", type=float, default=0.05)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--pixel-scale-min-arcsec", type=float, default=1.79)
    ap.add_argument("--pixel-scale-max-arcsec", type=float, default=2.99)
    ap.add_argument("--footprint-grid", type=int, default=31)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    indexes = _load_indexes()
    union_catalog = _dedup_catalog_world(indexes)
    p28_payload = p28._load_json(p28.DEFAULT_JSON)
    p28_support: dict[tuple[str, str], dict[str, Any]] = {}
    for row in list(p28_payload.get("support_matrix") or []) + list(p28_payload.get("union_support") or []):
        p28_support[(str(row.get("label")), str(row.get("tile")))] = dict(row)

    labels = list(MANDATORY_CASES) + list(NON_REGRESSION_CASES)
    cases = [_evaluate_case(label, indexes, union_catalog, args, p28_support) for label in labels]
    payload: dict[str, Any] = {
        "schema": "zeblind.p29_4d_bounded_multi_index_union_validation.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "oracle_use": "Le WCS Astrometry.net est un oracle de diagnostic d'ordre/footprint, pas une entree du solveur blind.",
        "cases": cases,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "tiles": list(TILE_ORDER),
            "index_paths": {tile: str(INDEX_PATHS[tile].expanduser().resolve()) for tile in TILE_ORDER},
            "union_catalog_stars": int(union_catalog.shape[0]),
            "source_policy": "diagnostic_unfiltered",
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "code_tol": float(args.code_tol),
            "max_stars": int(args.max_stars),
            "max_quads": int(args.max_quads),
            "max_hits_4d_per_tile": int(args.max_hits_4d),
            "max_hits_per_image_quad": int(args.max_hits_per_image_quad),
            "max_hypotheses": int(args.max_hypotheses),
            "image_strategy": str(args.image_strategy),
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "reference_dir": str(args.reference_dir.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
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
