#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.diagnose_p23_4d_source_list_contract as p23
import tools.diagnose_p26_4d_oracle_tile_routing as p26
import tools.diagnose_p27_4d_d50_2823_density_probe as p27
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p28_4d_validation_support_audit.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p28_4d_validation_support_audit.json"
DEFAULT_P26_JSON = ROOT / "reports/zeblind_p26_4d_oracle_tile_routing.json"
DEFAULT_P27_JSON = ROOT / "reports/zeblind_p27_4d_d50_2823_density_probe.json"
MANDATORY_CASES = ("232329", "232431")
COMPARISON_CASES = ("232144", "232205", "232247", "232350", "232102")
PRIMARY_TILE = "d50_2823"
SECONDARY_TILE = "d50_2822"


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
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _runtime_rows(p26_payload: dict[str, Any], p27_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in p27_payload.get("matrix") or []:
        if str(row.get("variant")) == "baseline_2000_40000":
            rows[(str(row.get("label")), PRIMARY_TILE)] = dict(row)
    for row in p26_payload.get("solve_matrix") or []:
        key = (str(row.get("label")), str(row.get("tile_key")))
        if key[1] == SECONDARY_TILE:
            rows[key] = dict(row)
    return rows


def _runtime_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "success": False,
            "max_inliers": None,
            "rms_px": None,
            "best_rank": None,
            "reason": "runtime row unavailable",
            "geometric_ok": False,
            "hits_4d": None,
            "hits_tested": None,
        }
    stats = dict(row.get("runtime_stats") or {})
    if bool(row.get("success")):
        max_inliers = stats.get("inliers")
        rms_px = stats.get("rms_px")
        reason = "accepted"
        rank = stats.get("first_accepted")
    else:
        best = stats.get("best_reject") if isinstance(stats.get("best_reject"), dict) else {}
        max_inliers = best.get("inliers")
        rms_px = best.get("rms_px")
        rank = best.get("rank", best.get("hit_rank"))
        reason = best.get("reason")
    try:
        geometric_ok = bool(float(rms_px) <= 1.2) and (
            bool(row.get("success")) or "scale_ok=1" in str(reason or "")
        )
    except Exception:
        geometric_ok = False
    return {
        "success": bool(row.get("success")),
        "max_inliers": None if max_inliers is None else int(max_inliers),
        "rms_px": None if rms_px is None else float(rms_px),
        "best_rank": rank,
        "reason": reason,
        "geometric_ok": geometric_ok,
        "hits_4d": stats.get("hits_4d"),
        "hits_tested": stats.get("hits_tested"),
    }


def _reason_margin(reason: Any) -> int | None:
    m = re.search(r"inliers=(\d+),inliers_thr=(\d+)", str(reason or ""))
    if not m:
        return None
    return int(m.group(1)) - int(m.group(2))


def _catalog_world(index_root: Path, entries: dict[str, dict[str, Any]], tile_key: str) -> np.ndarray:
    entry = entries[tile_key]
    with np.load(index_root / str(entry.get("tile_file") or ""), allow_pickle=False) as data:
        return np.column_stack((np.asarray(data["ra_deg"], dtype=np.float64), np.asarray(data["dec_deg"], dtype=np.float64)))


def _unique_world(world: np.ndarray) -> np.ndarray:
    if world.size == 0:
        return world.reshape(0, 2)
    _vals, idx = np.unique(np.round(world, decimals=8), axis=0, return_index=True)
    return world[np.sort(idx)]


def _one_to_one_matches(catalog_xy: np.ndarray, image_xy: np.ndarray, radius_px: float) -> dict[str, Any]:
    if catalog_xy.size == 0 or image_xy.size == 0:
        return {"count": 0, "max_distance_px": None, "median_distance_px": None}
    tree = cKDTree(image_xy)
    pairs: list[tuple[float, int, int]] = []
    for cat_idx, xy in enumerate(catalog_xy):
        for img_idx in tree.query_ball_point(xy, float(radius_px)):
            dist = float(np.linalg.norm(image_xy[int(img_idx)] - xy))
            pairs.append((dist, int(cat_idx), int(img_idx)))
    pairs.sort()
    used_catalog: set[int] = set()
    used_image: set[int] = set()
    distances: list[float] = []
    for dist, cat_idx, img_idx in pairs:
        if cat_idx in used_catalog or img_idx in used_image:
            continue
        used_catalog.add(cat_idx)
        used_image.add(img_idx)
        distances.append(dist)
    return {
        "count": int(len(used_catalog)),
        "max_distance_px": float(max(distances)) if distances else None,
        "median_distance_px": float(np.median(distances)) if distances else None,
    }


def _detect_case_sources(label: str, args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / _filename(label)
    raw, image_shape, detect_meta = p23._detect_runtime_stars(source, args)
    lists, list_stats = p23._make_lists(raw, image_shape, args)
    return {
        "raw": raw,
        "diagnostic_unfiltered": lists["diagnostic_unfiltered"],
        "standard_runtime": lists["standard_runtime"],
        "image_shape": image_shape,
        "detect_meta": detect_meta,
        "list_stats": list_stats,
    }


def _support_for_world(label: str, tile_label: str, world: np.ndarray, args: argparse.Namespace, sources: dict[str, Any]) -> dict[str, Any]:
    ref = args.reference_dir.expanduser().resolve() / _filename(label)
    wcs, shape = p26._load_oracle_wcs(ref)
    height, width = int(shape[0]), int(shape[1])
    pix = np.asarray(wcs.wcs_world2pix(world, 0), dtype=np.float64)
    finite = np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1])
    inside = finite & (pix[:, 0] >= 0.0) & (pix[:, 0] < width) & (pix[:, 1] >= 0.0) & (pix[:, 1] < height)
    margin = float(args.edge_margin_px)
    near_field = finite & (pix[:, 0] >= -margin) & (pix[:, 0] < width + margin) & (pix[:, 1] >= -margin) & (pix[:, 1] < height + margin)
    catalog_xy = pix[inside]
    raw_xy = p23._positions(sources["raw"])
    kept_xy = p23._positions(sources["diagnostic_unfiltered"])
    standard_xy = p23._positions(sources["standard_runtime"])
    radii: dict[str, Any] = {}
    for radius in (2.0, float(args.match_radius_px), 5.0):
        key = f"r{radius:g}"
        radii[key] = {
            "raw": _one_to_one_matches(catalog_xy, raw_xy, radius),
            "diagnostic_unfiltered": _one_to_one_matches(catalog_xy, kept_xy, radius),
            "standard_runtime": _one_to_one_matches(catalog_xy, standard_xy, radius),
        }
    primary = radii[f"r{float(args.match_radius_px):g}"]
    return {
        "label": label,
        "tile": tile_label,
        "catalog_stars_total": int(world.shape[0]),
        "catalog_stars_in_field": int(np.count_nonzero(inside)),
        "catalog_stars_near_field": int(np.count_nonzero(near_field)),
        "raw_detected_image_stars": int(sources["raw"].shape[0]),
        "image_stars_kept_diagnostic": int(sources["diagnostic_unfiltered"].shape[0]),
        "image_stars_kept_standard": int(sources["standard_runtime"].shape[0]),
        "oracle_match_radius_px": float(args.match_radius_px),
        "matchable_raw": int(primary["raw"]["count"]),
        "matchable_diagnostic": int(primary["diagnostic_unfiltered"]["count"]),
        "matchable_standard": int(primary["standard_runtime"]["count"]),
        "match_radii": radii,
    }


def _support_row(
    label: str,
    tile_key: str,
    args: argparse.Namespace,
    entries: dict[str, dict[str, Any]],
    sources: dict[str, Any],
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    world = _catalog_world(args.index_root.expanduser().resolve(), entries, tile_key)
    support = _support_for_world(label, tile_key, world, args, sources)
    runtime_summary = _runtime_summary(runtime)
    max_inliers = runtime_summary["max_inliers"]
    support["runtime"] = runtime_summary
    support["ratios"] = {
        "inliers_per_catalog_field": None if max_inliers is None else float(max_inliers) / max(1, support["catalog_stars_in_field"]),
        "inliers_per_matchable_diagnostic": None if max_inliers is None else float(max_inliers) / max(1, support["matchable_diagnostic"]),
        "inliers_per_image_kept": None if max_inliers is None else float(max_inliers) / max(1, support["image_stars_kept_diagnostic"]),
    }
    support["classification"] = _classify_support(support, int(args.quality_inliers))
    return support


def _union_row(label: str, tiles: tuple[str, ...], args: argparse.Namespace, entries: dict[str, dict[str, Any]], sources: dict[str, Any]) -> dict[str, Any]:
    worlds = [_catalog_world(args.index_root.expanduser().resolve(), entries, tile) for tile in tiles]
    world = _unique_world(np.vstack(worlds))
    support = _support_for_world(label, "+".join(tiles), world, args, sources)
    support["classification"] = "support_only_union_not_runtime_tested"
    return support


def _classify_support(row: dict[str, Any], quality_inliers: int) -> str:
    runtime = row.get("runtime") or {}
    max_inliers = runtime.get("max_inliers")
    geometric_ok = bool(runtime.get("geometric_ok"))
    cat_near = int(row.get("catalog_stars_near_field", 0) or 0)
    matchable = int(row.get("matchable_diagnostic", 0) or 0)
    if max_inliers is not None and int(max_inliers) >= int(quality_inliers):
        return "accepted_product_threshold"
    if cat_near < int(quality_inliers):
        return "low_catalog_support_absolute_threshold_unreachable"
    if matchable < int(quality_inliers):
        if geometric_ok and max_inliers is not None and int(max_inliers) >= max(0, matchable - 1):
            return "geometric_ok_low_matchable_support"
        return "low_source_list_or_matchable_support"
    if geometric_ok:
        return "geometric_near_miss_validation_absolute"
    return "not_geometrically_established"


def _summarize(payload: dict[str, Any]) -> tuple[str, list[str]]:
    rows = {(row["label"], row["tile"]): row for row in payload["support_matrix"]}
    r329 = rows[("232329", PRIMARY_TILE)]
    r431 = rows[("232431", PRIMARY_TILE)]
    r431_2822 = rows.get(("232431", SECONDARY_TILE))
    union431 = next(row for row in payload["union_support"] if row["label"] == "232431")
    union329 = next(row for row in payload["union_support"] if row["label"] == "232329")
    rt329 = r329["runtime"]
    rt431 = r431["runtime"]
    verdict = "P2.8: validation absolue confond echec produit strict et solution geometrique a support limite"
    answers = [
        (
            "`232431` / `d50_2823`: seuil 40 non atteignable en mono-tuile avec le catalogue actuel "
            f"({r431['catalog_stars_in_field']} etoiles dans le champ, {r431['catalog_stars_near_field']} avec marge, "
            f"best reject {rt431['max_inliers']} inliers, RMS {_fmt(rt431['rms_px'], 3)}). "
            "A classer `GEOMETRIC_OK_LOW_SUPPORT` dans un futur statut experimental, sans acceptation produit."
        ),
        (
            "`232329` / `d50_2823`: near-miss geometrique, pas vrai echec de transform "
            f"(best reject {rt329['max_inliers']} inliers, RMS {_fmt(rt329['rms_px'], 3)}, "
            f"{r329['catalog_stars_in_field']} etoiles catalogue dans le champ, "
            f"{r329['matchable_raw']} matchables en detections brutes, {r329['matchable_diagnostic']} dans la source-list 4D). "
            "Sur les 3 inliers manquants, 2 sont visibles dans les detections brutes mais absents de la source-list 4D gardee; "
            "le dernier n'est pas disponible comme appariement oracle brut au rayon 3 px."
        ),
        (
            "`quality_inliers=40` reste correct comme seuil produit strict, mais il est trop absolu pour diagnostiquer "
            "des champs pauvres: il melange mauvais solve, support catalogue insuffisant et source-list insuffisante."
        ),
        (
            "Avant de juger le backend 4D sur ces deux cas, il faut soit une source catalogue plus profonde, "
            "soit un statut experimental distinct `GEOMETRIC_OK_LOW_SUPPORT` / `VALIDATION_NEAR_MISS_LOW_CATALOG_SUPPORT`."
        ),
        (
            "`232431` mono-tuile ne suffit pas (`d50_2823` best 31, `d50_2822` best "
            f"{(r431_2822 or {}).get('runtime', {}).get('max_inliers')}), mais l'union oracle "
            f"`d50_2823+d50_2822` expose {union431['catalog_stars_in_field']} etoiles champ / "
            f"{union431['matchable_diagnostic']} matchables. Une strategie multi-index bornee est donc plausible a auditer, pas a promouvoir."
        ),
        (
            "`232329` a aussi un support union confortable hors runtime "
            f"({union329['catalog_stars_in_field']} etoiles champ / {union329['matchable_diagnostic']} matchables), "
            "mais le bloc mono-tuile courant est surtout source-list/matchable support."
        ),
        "Suite: mini-corpus M106 multi-index borne possible en diagnostic, mais le contrat de validation doit distinguer support faible et acceptation produit stricte avant toute conclusion produit.",
    ]
    return verdict, answers


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.8 - audit validation par support catalogue reel",
        "",
        "> Le WCS Astrometry.net est un oracle de diagnostic de tuilage/support, pas une entree du solveur blind.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Diagnostic uniquement: aucun tuning, aucun changement de seuil, aucun refactor backend, aucun all30.",
        "- Seuils produit conserves pour l'analyse: `quality_inliers=40`, `quality_rms=1.2`.",
        "",
        "## Matrice support / validation",
        "",
        "| cas | tuile | cat champ | cat marge | raw match | 4D match | std match | img gardees | max inliers | RMS | ratios inl/cat/match/img | classification |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["support_matrix"]:
        rt = row["runtime"]
        ratios = row["ratios"]
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} | `{}` |".format(
                row["label"],
                row["tile"],
                row["catalog_stars_in_field"],
                row["catalog_stars_near_field"],
                row["matchable_raw"],
                row["matchable_diagnostic"],
                row["matchable_standard"],
                row["image_stars_kept_diagnostic"],
                rt.get("max_inliers", ""),
                _fmt(rt.get("rms_px"), 3),
                _fmt(ratios.get("inliers_per_catalog_field"), 2),
                _fmt(ratios.get("inliers_per_matchable_diagnostic"), 2),
                _fmt(ratios.get("inliers_per_image_kept"), 2),
                row["classification"],
            )
        )
    lines.extend(
        [
            "",
            "## Support union hors runtime",
            "",
            "| cas | tuiles | cat champ | cat marge | raw match | 4D match | img gardees | lecture |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["union_support"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | `{}` |".format(
                row["label"],
                row["tile"],
                row["catalog_stars_in_field"],
                row["catalog_stars_near_field"],
                row["matchable_raw"],
                row["matchable_diagnostic"],
                row["image_stars_kept_diagnostic"],
                row["classification"],
            )
        )
    lines.extend(["", "## Cas critiques", ""])
    rows = {(row["label"], row["tile"]): row for row in payload["support_matrix"]}
    for label in MANDATORY_CASES:
        row = rows[(label, PRIMARY_TILE)]
        rt = row["runtime"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Meilleur rejet `d50_2823`: `{rt.get('max_inliers')}` inliers, RMS `{_fmt(rt.get('rms_px'), 3)}`, rang `{rt.get('best_rank')}`.",
                f"- Support oracle `d50_2823`: `{row['catalog_stars_in_field']}` etoiles dans le champ, `{row['matchable_raw']}` matchables dans les detections brutes, `{row['matchable_diagnostic']}` matchables dans la source-list 4D.",
                f"- Marge inliers vs seuil: `{_reason_margin(rt.get('reason'))}`.",
                f"- Classification: `{row['classification']}`.",
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
    ap = argparse.ArgumentParser(description="P2.8 diagnostic-only validation support audit for the experimental 4D backend.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--p26-json", type=Path, default=DEFAULT_P26_JSON)
    ap.add_argument("--p27-json", type=Path, default=DEFAULT_P27_JSON)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max-stars", type=int, default=120)
    ap.add_argument("--blind-star-min-sep-px", type=float, default=0.0)
    ap.add_argument("--astrometry-like-boxes", type=int, default=10)
    ap.add_argument("--astrometry-like-min-keep-ratio", type=float, default=0.05)
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--edge-margin-px", type=float, default=3.0)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    p26_payload = _load_json(args.p26_json)
    p27_payload = _load_json(args.p27_json)
    runtime = _runtime_rows(p26_payload, p27_payload)
    entries = p26._tile_entries(args.index_root.expanduser().resolve())
    labels = list(MANDATORY_CASES) + list(COMPARISON_CASES)
    sources = {label: _detect_case_sources(label, args) for label in labels}

    support_matrix: list[dict[str, Any]] = []
    for label in labels:
        support_matrix.append(_support_row(label, PRIMARY_TILE, args, entries, sources[label], runtime.get((label, PRIMARY_TILE))))
        if label in MANDATORY_CASES:
            support_matrix.append(_support_row(label, SECONDARY_TILE, args, entries, sources[label], runtime.get((label, SECONDARY_TILE))))
    union_support = [
        _union_row(label, (PRIMARY_TILE, SECONDARY_TILE), args, entries, sources[label])
        for label in MANDATORY_CASES
    ]
    payload: dict[str, Any] = {
        "schema": "zeblind.p28_4d_validation_support_audit.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "oracle_use": "Le WCS Astrometry.net est un oracle de diagnostic de tuilage/support, pas une entree du solveur blind.",
        "support_matrix": support_matrix,
        "union_support": union_support,
        "params": {
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "edge_margin_px": float(args.edge_margin_px),
            "source_policy": "diagnostic_unfiltered",
            "primary_tile": PRIMARY_TILE,
            "secondary_tile": SECONDARY_TILE,
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "reference_dir": str(args.reference_dir.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
            "p26_json": str(args.p26_json.expanduser().resolve()),
            "p27_json": str(args.p27_json.expanduser().resolve()),
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
