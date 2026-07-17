#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex

import tools.diagnose_p29_4d_bounded_multi_index_union_validation as p29
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p210_4d_m106_bounded_multi_index_corpus.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p210_4d_m106_bounded_multi_index_corpus.json"
MANDATORY_CASES = ("232329", "232431")
BASE_COMPARISON_CASES = ("232144", "232205", "232247", "232350", "232102")
EXTRA_MINI_CASES = ("232513", "232534", "232658")
TILE_ORDER = ("d50_2823", "d50_2822")
INDEX_PATHS = dict(p29.INDEX_PATHS)


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


def _load_indexes(tile_order: tuple[str, ...]) -> dict[str, Quad4DIndex]:
    out: dict[str, Quad4DIndex] = {}
    for tile in tile_order:
        path = INDEX_PATHS[tile].expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"missing explicit P2.10 index for {tile}: {path}")
        out[tile] = Quad4DIndex.load(path)
    return out


def _dedup_catalog(indexes: dict[str, Quad4DIndex], tile_order: tuple[str, ...]) -> np.ndarray:
    worlds = [np.asarray(indexes[tile].catalog_ra_dec, dtype=np.float64) for tile in tile_order]
    union = np.vstack(worlds) if worlds else np.empty((0, 2), dtype=np.float64)
    if union.size == 0:
        return union.reshape(0, 2)
    _vals, idx = np.unique(np.round(union, decimals=8), axis=0, return_index=True)
    return union[np.sort(idx)]


def _search_hits_explicit(indexes: dict[str, Quad4DIndex], records: list[Any], tile_order: tuple[str, ...], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    t0 = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    per_tile: dict[str, Any] = {}
    for tile_pos, tile_key in enumerate(tile_order):
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
            "tile_order": int(tile_pos),
        }
        for local_rank, hit in enumerate(hits, start=1):
            candidates.append(
                {
                    "tile_key": tile_key,
                    "index": index,
                    "hit": hit,
                    "local_rank": int(local_rank),
                    "tile_order": int(tile_pos),
                }
            )
    candidates.sort(key=lambda row: (int(row["tile_order"]), float(row["hit"].code_distance), int(row["local_rank"])))
    return candidates, per_tile, float(time.perf_counter() - t0)


def _accepted_key(row: dict[str, Any]) -> tuple[int, float, int, float, int]:
    val = row.get("union") or {}
    ok = 1 if val.get("status") == "ACCEPTED_PRODUCT_THRESHOLD" else 0
    rms = float(val.get("rms_px", float("inf")) if val.get("rms_px") is not None else float("inf"))
    inliers = int(val.get("inliers", 0) or 0)
    geo = float(val.get("geo_cov_area", 0.0) if val.get("geo_cov_area") is not None else 0.0)
    rank = int(row.get("global_rank", 10**9) or 10**9)
    return (ok, -rms if np.isfinite(rms) else -1e9, inliers, geo, -rank)


def _overall_key(row: dict[str, Any]) -> tuple[int, float, int, float, int]:
    val = row.get("union") or {}
    ok = 1 if val.get("status") == "ACCEPTED_PRODUCT_THRESHOLD" else 0
    inliers = int(val.get("inliers", 0) or 0)
    rms = float(val.get("rms_px", float("inf")) if val.get("rms_px") is not None else float("inf"))
    geo = float(val.get("geo_cov_area", 0.0) if val.get("geo_cov_area") is not None else 0.0)
    rank = int(row.get("global_rank", 10**9) or 10**9)
    return (ok, inliers, -rms if np.isfinite(rms) else -1e9, geo, -rank)


def _short(row: dict[str, Any] | None, field: str = "union") -> dict[str, Any]:
    if not row:
        return {"status": None, "inliers": None, "rms_px": None, "rank": None, "tile_key": None, "scale": None}
    val = row.get(field) or {}
    return {
        "status": val.get("status"),
        "inliers": val.get("inliers"),
        "rms_px": val.get("rms_px"),
        "rank": row.get("global_rank"),
        "local_rank": row.get("local_rank"),
        "tile_key": row.get("tile_key"),
        "scale": val.get("pix_scale_arcsec"),
        "geo_cov_area": val.get("geo_cov_area"),
        "reason": val.get("reason"),
    }


def _evaluate_case(label: str, indexes: dict[str, Quad4DIndex], union_catalog: np.ndarray, tile_order: tuple[str, ...], args: argparse.Namespace) -> dict[str, Any]:
    t_case0 = time.perf_counter()
    sources = p29._detect_sources(label, args)
    _quads, records, quad_s = p29._image_records(sources, args)
    candidates, per_tile, lookup_s = _search_hits_explicit(indexes, records, tile_order, args)
    image_positions = np.asarray(sources["image_positions"], dtype=np.float64)
    image_shape = tuple(int(v) for v in sources["image_shape"])
    rows: list[dict[str, Any]] = []
    accepts: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]] = set()
    reason_counts: Counter[str] = Counter()
    t_val0 = time.perf_counter()
    stop_reason = "candidate_exhausted"
    for global_rank, item in enumerate(candidates, start=1):
        if len(rows) >= int(args.max_hypotheses):
            stop_reason = "max_hypotheses"
            break
        if time.perf_counter() - t_case0 >= float(args.max_wall_s):
            stop_reason = "max_wall_s"
            break
        tile_key = str(item["tile_key"])
        hit = item["hit"]
        index = item["index"]
        key = (tile_key, tuple(int(v) for v in hit.image_quad_indices), tuple(int(v) for v in hit.catalog_quad_indices))
        if key in seen:
            reason_counts["duplicate_hypothesis"] += 1
            continue
        seen.add(key)
        try:
            image_quad_points = image_positions[np.asarray(hit.image_quad_indices, dtype=np.int64)]
            catalog_quad_world = index.catalog_ra_dec[np.asarray(hit.catalog_quad_indices, dtype=np.int64)]
            wcs = p29._fit_astrometry_4d_quad_wcs(image_quad_points, catalog_quad_world)
            mono = p29._validate_catalog(wcs, image_positions, image_shape, index.catalog_ra_dec, args)
            union = p29._validate_catalog(wcs, image_positions, image_shape, union_catalog, args)
            mono["status"] = p29._status(mono, support=None)
            union["status"] = p29._status(union, support=None)
            row = {
                "label": label,
                "tile_key": tile_key,
                "global_rank": int(global_rank),
                "local_rank": int(item["local_rank"]),
                "tile_order": int(item["tile_order"]),
                "code_distance": float(hit.code_distance),
                "mono": mono,
                "union": union,
            }
            rows.append(row)
            if union.get("status") == "ACCEPTED_PRODUCT_THRESHOLD":
                accepts.append(row)
                if len(accepts) >= int(args.max_accepts):
                    stop_reason = "max_accepts"
                    break
            reason_counts[str(union.get("reason", "accepted") if union.get("status") != "ACCEPTED_PRODUCT_THRESHOLD" else "accepted")] += 1
        except Exception as exc:
            reason_counts[f"fit_or_validate_failed:{type(exc).__name__}"] += 1
    validation_s = float(time.perf_counter() - t_val0)
    first_accept = accepts[0] if accepts else None
    best_within_budget = max(accepts, key=_accepted_key) if accepts else (max(rows, key=_overall_key) if rows else None)
    best_union_support = max(rows, key=_overall_key) if rows else None
    best_mono_by_tile: dict[str, Any] = {}
    for tile in tile_order:
        tile_rows = [row for row in rows if row["tile_key"] == tile]
        best_mono_by_tile[tile] = max(tile_rows, key=lambda row: p29._better_key(row, "mono")) if tile_rows else None
    return {
        "label": label,
        "case_group": "mandatory" if label in MANDATORY_CASES else ("comparison" if label in BASE_COMPARISON_CASES else "extra_mini"),
        "source": {
            "raw_detected": int(sources["raw"].shape[0]),
            "kept_diagnostic": int(sources["stars"].shape[0]),
            "image_records": int(len(records)),
        },
        "per_tile": per_tile,
        "hits_total": int(sum(int(v["hits"]) for v in per_tile.values())),
        "hypotheses_tested": int(len(rows)),
        "accepted_candidates": int(len(accepts)),
        "quad_build_s": float(quad_s),
        "lookup_s": float(lookup_s),
        "validation_s": float(validation_s),
        "wall_s": float(time.perf_counter() - t_case0),
        "stop_reason": stop_reason,
        "best_mono_by_tile": best_mono_by_tile,
        "first_accept": first_accept,
        "best_within_budget": best_within_budget,
        "best_union_support": best_union_support,
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
    }


def _case_labels(args: argparse.Namespace) -> list[str]:
    labels = list(MANDATORY_CASES) + list(BASE_COMPARISON_CASES)
    if bool(args.include_extra_mini):
        labels.extend(EXTRA_MINI_CASES)
    if args.cases.strip():
        labels = [part.strip() for part in args.cases.split(",") if part.strip()]
    out: list[str] = []
    for label in labels:
        if label not in out:
            out.append(label)
    return out


def _d50_2822_bad_control_labels(labels: list[str], args: argparse.Namespace) -> list[str]:
    bad: list[str] = []
    for label in labels:
        if label in MANDATORY_CASES:
            continue
        try:
            footprints = p29._oracle_footprints(label, args)
            if float(footprints.get("d50_2822", 0.0) or 0.0) <= float(args.bad_tile_max_footprint_pct):
                bad.append(label)
        except Exception:
            continue
    return bad


def _summarize(payload: dict[str, Any]) -> tuple[str, list[str]]:
    cases = payload["cases"]
    mandatory = [row for row in cases if row["case_group"] == "mandatory"]
    comparisons = [row for row in cases if row["case_group"] == "comparison"]
    extras = [row for row in cases if row["case_group"] == "extra_mini"]
    first_ok = [row for row in cases if _short(row.get("first_accept")).get("status") == "ACCEPTED_PRODUCT_THRESHOLD"]
    best_ok = [row for row in cases if _short(row.get("best_within_budget")).get("status") == "ACCEPTED_PRODUCT_THRESHOLD"]
    mandatory_ok = all(_short(row.get("first_accept")).get("status") == "ACCEPTED_PRODUCT_THRESHOLD" for row in mandatory)
    comparison_ok = all(_short(row.get("first_accept")).get("status") == "ACCEPTED_PRODUCT_THRESHOLD" for row in comparisons)
    bad_rows = payload.get("bad_tile_control") or []
    bad_accepts = [row for row in bad_rows if _short(row.get("first_accept")).get("status") == "ACCEPTED_PRODUCT_THRESHOLD"]
    regressions = [row for row in comparisons if _short(row.get("first_accept")).get("status") != "ACCEPTED_PRODUCT_THRESHOLD"]
    quality_gains = []
    for row in cases:
        fa = _short(row.get("first_accept"))
        bw = _short(row.get("best_within_budget"))
        if fa.get("status") == "ACCEPTED_PRODUCT_THRESHOLD" and bw.get("status") == "ACCEPTED_PRODUCT_THRESHOLD":
            if bw.get("rank") != fa.get("rank") and float(bw.get("rms_px") or 1e9) < float(fa.get("rms_px") or 1e9):
                quality_gains.append(row)
    if mandatory_ok and comparison_ok and not bad_accepts:
        verdict = "P2.10 positif: mini-corpus M106 borne valide le multi-index experimental sans faux positif evident"
    elif bad_accepts:
        verdict = "P2.10 stop: controle mauvaise liste produit une acceptation"
    else:
        verdict = "P2.10 partiel: regressions ou echecs a analyser avant elargissement"
    answers = [
        f"Mode multi-index borne resout le mini-corpus teste en first_accept: `{len(first_ok)}/{len(cases)}` ; en best_within_budget: `{len(best_ok)}/{len(cases)}`.",
        f"Cas obligatoires recuperes: `{mandatory_ok}` ; cas comparaison non-regresses: `{comparison_ok}` ({len(regressions)} regression).",
        f"`best_within_budget` apporte un gain de qualite RMS sur `{len(quality_gains)}/{len(cases)}` cas, notamment quand le premier accepte est correct mais pas optimal.",
        f"Controle mauvaise liste explicite `d50_2822` seule (footprint oracle <= {payload['params']['bad_tile_max_footprint_pct']}%): `{len(bad_accepts)}` acceptation(s) sur `{len(bad_rows)}` cas controles.",
        f"Cout median mesure: `{np.median([float(row['wall_s']) for row in cases]):.3f}s` total, `{np.median([float(row['validation_s']) for row in cases]):.3f}s` validation.",
        "Strategie propre comme option experimentale utilisateur: oui, avec liste courte explicite d'index, union-catalogue dedupliquee, budgets bornes, et OFF par defaut.",
        "Garanties avant promotion produit: mini-corpus M106 plus large mais borne, controle faux positifs plus dur, contrat d'acceptation separe, non-regression ancien backend/ZeNear.",
    ]
    if extras:
        answers.append(f"Extras M106 inclus sans all30: `{', '.join(row['label'] for row in extras)}`.")
    return verdict, answers


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.10 - mini-corpus M106 multi-index borne",
        "",
        "> Le WCS Astrometry.net reste un oracle offline de rapport; le runtime probe utilise seulement la liste explicite d'index 4D.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Diagnostic/experimental uniquement: aucun seuil change, aucun all30, aucun rebuild complet, aucun changement produit par defaut.",
        "- Interface experimentale visee: `quad_hash_schema=\"astrometry_ab_code_4d_v1\"`, `blind_astrometry_4d_index_paths=[...]`, `blind_astrometry_4d_validation_catalog_policy=\"union_candidate_tiles\"`, `blind_astrometry_4d_source_policy=\"diagnostic_unfiltered\"`.",
        "",
        "## Matrice corpus",
        "",
        "| cas | groupe | mono 2823 | mono 2822 | first_accept | best_within_budget | gain RMS | hits | testes | accepts | total | lookup | validation | stop |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["cases"]:
        mono23 = _short(row["best_mono_by_tile"].get("d50_2823"), "mono")
        mono22 = _short(row["best_mono_by_tile"].get("d50_2822"), "mono")
        fa = _short(row.get("first_accept"))
        bw = _short(row.get("best_within_budget"))
        gain = ""
        if fa.get("rms_px") is not None and bw.get("rms_px") is not None:
            gain = _fmt(float(fa["rms_px"]) - float(bw["rms_px"]), 3)
        lines.append(
            "| `{}` | `{}` | {} / {} / `{}` | {} / {} / `{}` | {} / {} / `{}` / `{}` r{} | {} / {} / `{}` / `{}` r{} | {} | {} | {} | {} | {} | {} | {} | `{}` |".format(
                row["label"],
                row["case_group"],
                mono23.get("inliers"), _fmt(mono23.get("rms_px"), 3), mono23.get("status"),
                mono22.get("inliers"), _fmt(mono22.get("rms_px"), 3), mono22.get("status"),
                fa.get("inliers"), _fmt(fa.get("rms_px"), 3), fa.get("status"), fa.get("tile_key"), fa.get("rank"),
                bw.get("inliers"), _fmt(bw.get("rms_px"), 3), bw.get("status"), bw.get("tile_key"), bw.get("rank"),
                gain,
                row["hits_total"],
                row["hypotheses_tested"],
                row["accepted_candidates"],
                _fmt(row["wall_s"], 3),
                _fmt(row["lookup_s"], 3),
                _fmt(row["validation_s"], 3),
                row["stop_reason"],
            )
        )
    lines.extend(["", "## Controle mauvaise liste", ""])
    lines.extend([
        "| cas | footprint d50_2822 | first_accept d50_2822 seule | best d50_2822 seule | hits | testes | stop |",
        "|---|---:|---|---|---:|---:|---|",
    ])
    for row in payload.get("bad_tile_control") or []:
        fa = _short(row.get("first_accept"))
        bw = _short(row.get("best_within_budget"))
        lines.append(
            "| `{}` | {} | {} / {} / `{}` | {} / {} / `{}` | {} | {} | `{}` |".format(
                row["label"],
                _fmt(row.get("bad_tile_footprint_pct"), 1),
                fa.get("inliers"), _fmt(fa.get("rms_px"), 3), fa.get("status"),
                bw.get("inliers"), _fmt(bw.get("rms_px"), 3), bw.get("status"),
                row["hits_total"],
                row["hypotheses_tested"],
                row["stop_reason"],
            )
        )
    lines.extend(["", "## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.10 mini-corpus M106 bounded multi-index 4D diagnostic.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--cases", default="")
    ap.add_argument("--include-extra-mini", type=int, choices=(0, 1), default=1)
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
    ap.add_argument("--max-wall-s", type=float, default=45.0)
    ap.add_argument("--max-accepts", type=int, default=64)
    ap.add_argument("--bad-tile-max-footprint-pct", type=float, default=40.0)
    ap.add_argument("--footprint-grid", type=int, default=31)
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--pixel-scale-min-arcsec", type=float, default=1.79)
    ap.add_argument("--pixel-scale-max-arcsec", type=float, default=2.99)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    tile_order = TILE_ORDER
    indexes = _load_indexes(tile_order)
    union_catalog = _dedup_catalog(indexes, tile_order)
    labels = _case_labels(args)
    cases = [_evaluate_case(label, indexes, union_catalog, tile_order, args) for label in labels]

    bad_labels = _d50_2822_bad_control_labels(labels, args)
    bad_indexes = {"d50_2822": indexes["d50_2822"]}
    bad_union = _dedup_catalog(bad_indexes, ("d50_2822",))
    bad_rows = []
    for label in bad_labels:
        row = _evaluate_case(label, bad_indexes, bad_union, ("d50_2822",), args)
        row["bad_tile_footprint_pct"] = float(p29._oracle_footprints(label, args).get("d50_2822", 0.0) or 0.0)
        bad_rows.append(row)

    payload: dict[str, Any] = {
        "schema": "zeblind.p210_4d_m106_bounded_multi_index_corpus.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_oracle_policy": "no WCS oracle is used by candidate generation or validation; WCS is offline report context only",
        "cases": cases,
        "bad_tile_control": bad_rows,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "index_paths": {tile: str(INDEX_PATHS[tile].expanduser().resolve()) for tile in tile_order},
            "blind_astrometry_4d_index_paths": [str(INDEX_PATHS[tile].expanduser().resolve()) for tile in tile_order],
            "blind_astrometry_4d_validation_catalog_policy": "union_candidate_tiles",
            "blind_astrometry_4d_accept_policies_compared": ["first_accept", "best_within_budget"],
            "blind_astrometry_4d_source_policy": "diagnostic_unfiltered",
            "union_catalog_stars": int(union_catalog.shape[0]),
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_hypotheses": int(args.max_hypotheses),
            "max_wall_s": float(args.max_wall_s),
            "max_accepts": int(args.max_accepts),
            "cases": labels,
            "bad_tile_control": "explicit d50_2822-only on non-mandatory cases",
            "bad_tile_max_footprint_pct": float(args.bad_tile_max_footprint_pct),
            "available_m106_local_common": 30,
            "all30_run": False,
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
