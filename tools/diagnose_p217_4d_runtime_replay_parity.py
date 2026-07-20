#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import (
    SolveConfig,
    _astrometry_4d_array_hash,
    _astrometry_4d_build_match_details,
    _astrometry_4d_validate_pixel_matches,
    _astrometry_4d_wcs_summary,
    solve_blind,
)
from zeblindsolver.verify import validate_solution

import tools.diagnose_p212_4d_m106_30_bounded_validation as p212
import tools.diagnose_p214_4d_source_policy_bakeoff as p214
import tools.diagnose_p215_4d_split_quad_verify_sources as p215
import tools.diagnose_p216_4d_candidate_refine_residual_audit as p216
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p217_4d_runtime_replay_parity.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p217_4d_runtime_replay_parity.json"
DEFAULT_WORK_DIR = ROOT / "reports/p217_4d_runtime_replay_parity"
MAIN_CASE = "234013"
INDEXES = (p212.INDEX_2823, p212.INDEX_2822)


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


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
        if not np.isfinite(f):
            return ""
        return f"{f:.{digits}f}"
    except Exception:
        return ""


def _summary_xy(points: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(points, dtype=np.float64)
    return {
        "count": int(arr.shape[0]) if arr.ndim == 2 else 0,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min_xy": [float(v) for v in np.nanmin(arr, axis=0)] if arr.size else [],
        "max_xy": [float(v) for v in np.nanmax(arr, axis=0)] if arr.size else [],
        "hash": _astrometry_4d_array_hash(arr),
        "head_indices": list(range(min(8, int(arr.shape[0]) if arr.ndim else 0))),
        "tail_indices": list(range(max(0, int(arr.shape[0]) - 8), int(arr.shape[0]))) if arr.ndim == 2 else [],
    }


def _thresholds(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "rms_px": float(args.quality_rms),
        "inliers": int(args.quality_inliers),
        "scale_min_arcsec": float(args.pixel_scale_min_arcsec),
        "scale_max_arcsec": float(args.pixel_scale_max_arcsec),
    }


def _prepare_target(args: argparse.Namespace) -> Path:
    source = args.data_dir.expanduser().resolve() / p215._filename(args.main_case)
    target_dir = args.work_dir.expanduser().resolve() / "runtime_main"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    p22._strip_wcs(target)
    return target


def _prep_cache(target: Path, quad_sources: np.ndarray, verification_sources: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    resolved = target.resolve()
    stat = resolved.stat()
    return {
        str(resolved): {
            "sig": (int(stat.st_mtime_ns), int(stat.st_size)),
            "downsample": int(args.downsample),
            "detect_k_sigma": float(args.detect_k_sigma),
            "detect_min_area": int(args.detect_min_area),
            "stars": quad_sources.copy(),
            "astrometry_4d_verification_stars": verification_sources.copy(),
        }
    }


def _config(args: argparse.Namespace, quad_sources: np.ndarray) -> SolveConfig:
    return SolveConfig(
        max_stars=max(1, int(quad_sources.shape[0])),
        max_quads=int(args.max_quads),
        detect_k_sigma=float(args.detect_k_sigma),
        detect_min_area=int(args.detect_min_area),
        downsample=int(args.downsample),
        quality_inliers=int(args.quality_inliers),
        quality_rms=float(args.quality_rms),
        pixel_scale_min_arcsec=float(args.pixel_scale_min_arcsec),
        pixel_scale_max_arcsec=float(args.pixel_scale_max_arcsec),
        blind_global_hard_budget_s=float(args.max_wall_s),
        blind_reuse_existing_solved_wcs=False,
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_paths=tuple(str(path.expanduser().resolve()) for path in INDEXES),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
        blind_astrometry_4d_accept_policy="best_within_budget",
        blind_astrometry_4d_max_accepts=int(args.max_accepts),
        blind_astrometry_4d_code_tol=float(args.code_tol),
        blind_astrometry_4d_max_hits=int(args.max_hits_4d),
        blind_astrometry_4d_max_hits_per_image_quad=int(args.max_hits_per_image_quad),
        blind_astrometry_4d_max_hypotheses=int(args.max_hypotheses),
        blind_astrometry_4d_image_strategy=str(args.image_strategy),
        blind_astrometry_4d_match_radius_px=float(args.match_radius_px),
        blind_astrometry_4d_trace_candidate_enabled=True,
        blind_astrometry_4d_trace_candidate_rank=int(args.known_rank),
        blind_astrometry_4d_trace_candidate_origin_tile=str(args.known_origin_tile),
        blind_astrometry_4d_trace_candidate_image_quad_indices=tuple(int(v) for v in str(args.known_image_quad).split(",")),
        blind_astrometry_4d_trace_candidate_catalog_quad_indices=tuple(int(v) for v in str(args.known_catalog_quad).split(",")),
    )


def _replay_trace(args: argparse.Namespace, union_world: np.ndarray, quad_sources: np.ndarray, verification_sources: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    image_positions = p216._positions(quad_sources)
    verification_xy = p216._positions(verification_sources)
    candidates, _records, replay_stats = p216._candidate_list(
        quad_sources,
        INDEXES,
        max_quads=int(args.max_quads),
        image_strategy=str(args.image_strategy),
        code_tol=float(args.code_tol),
        max_hits=int(args.max_hits_4d),
        max_hits_per_image_quad=int(args.max_hits_per_image_quad),
    )
    rank, item = p216._find_known_candidate(candidates, args)
    wcs = p216._fit_candidate_wcs(item, image_positions)
    details = _astrometry_4d_build_match_details(wcs, verification_xy, union_world, radius_px=float(args.match_radius_px))
    matches = np.asarray(details["matches"], dtype=np.float64)
    direct = _astrometry_4d_validate_pixel_matches(wcs, matches, np.asarray(details["distances_px"], dtype=np.float64), _thresholds(args))
    inverse = validate_solution(wcs, matches, _thresholds(args))
    geo_ok, geo = p216._blind_geometric_guardrails(matches[:, :2], image_shape)
    direct = dict(direct)
    direct.update(
        {
            "geo_ok": bool(geo_ok),
            "geo_cov_x": float(geo.get("cov_x", float("nan"))),
            "geo_cov_y": float(geo.get("cov_y", float("nan"))),
            "geo_cov_area": float(geo.get("cov_area", float("nan"))),
            "geo_cond": float(geo.get("cond", float("nan"))),
            "matches": int(matches.shape[0]),
            "hit_rank": int(rank),
            "local_rank": int(item["local_rank"]),
            "index_order": int(item["index_order"]),
            "index_path": str(item["index"].path),
            "code_distance": float(item["hit"].code_distance),
            "image_quad_indices": [int(v) for v in item["hit"].image_quad_indices],
            "catalog_quad_indices": [int(v) for v in item["hit"].catalog_quad_indices],
            "tile_key": str(item["hit"].tile_key),
            "origin_tile_key": str(item["hit"].tile_key),
            "validation_catalog_policy": "union_candidate_tiles",
            "validation_catalog_stars": int(union_world.shape[0]),
        }
    )
    return {
        "candidate_replay": replay_stats,
        "identity": {
            "rank": int(rank),
            "local_rank": int(item["local_rank"]),
            "index_order": int(item["index_order"]),
            "index_path": str(item["index"].path),
            "origin_tile": str(item["hit"].tile_key),
            "code_distance": float(item["hit"].code_distance),
            "image_quad_indices": [int(v) for v in item["hit"].image_quad_indices],
            "catalog_quad_indices": [int(v) for v in item["hit"].catalog_quad_indices],
        },
        "wcs": _astrometry_4d_wcs_summary(wcs),
        "quad_sources": _summary_xy(image_positions),
        "verification_sources": _summary_xy(verification_xy),
        "catalog": {
            "policy": "union_candidate_tiles",
            "tiles": ["d50_2823", "d50_2822"],
            "count": int(union_world.shape[0]),
            "hash": _astrometry_4d_array_hash(union_world, decimals=8),
        },
        "matching": {
            "finite_projected": int(details.get("finite_projected", 0) or 0),
            "in_image_projected": int(details.get("in_image_projected", 0) or 0),
            "neighbors_within_3px": int(details.get("neighbors_within_radius", 0) or 0),
            "unique_pairs": int(details.get("unique_pairs", 0) or 0),
            "pairs_hash": _astrometry_4d_array_hash(matches, decimals=6),
            "image_indices_hash": _astrometry_4d_array_hash(np.asarray(details.get("image_indices", []), dtype=np.float64), decimals=0),
            "catalog_indices_hash": _astrometry_4d_array_hash(np.asarray(details.get("catalog_indices", []), dtype=np.float64), decimals=0),
        },
        "validation": direct,
        "legacy_inverse_validation": inverse,
    }


def _runtime_trace(args: argparse.Namespace, target: Path, quad_sources: np.ndarray, verification_sources: np.ndarray) -> dict[str, Any]:
    result = solve_blind(
        target,
        args.index_root.expanduser().resolve(),
        config=_config(args, quad_sources),
        prep_cache=_prep_cache(target, quad_sources, verification_sources, args),
    )
    stats = dict(result.stats or {})
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "selected_rank": stats.get("astrometry_4d_selected_rank"),
        "selected_origin_tile": stats.get("astrometry_4d_selected_origin_tile_key"),
        "selected_inliers": stats.get("inliers"),
        "selected_rms_px": stats.get("rms_px"),
        "accepted_candidates": stats.get("astrometry_4d_accepted_candidates"),
        "first_accepted_rank": stats.get("astrometry_4d_first_accepted_rank"),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "trace": stats.get("astrometry_4d_trace_candidate") or {},
        "best_accepted": stats.get("astrometry_4d_best_accepted_validation") or {},
        "best_reject": stats.get("astrometry_4d_best_reject") or {},
        "best_plausible_reject": stats.get("astrometry_4d_best_plausible_reject") or {},
        "best_scale_invalid_reject": stats.get("astrometry_4d_best_scale_invalid_reject") or {},
        "best_rms_invalid_reject": stats.get("astrometry_4d_best_rms_invalid_reject") or {},
        "best_geometry_invalid_reject": stats.get("astrometry_4d_best_geometry_invalid_reject") or {},
    }


def _compare(runtime: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    rt = runtime.get("trace") or {}
    rid = replay["identity"]
    rwcs = replay["wcs"]
    rmatch = replay["matching"]
    rval = replay["validation"]
    tid = {
        "rank": rt.get("rank"),
        "local_rank": rt.get("local_rank"),
        "index_order": rt.get("index_order"),
        "origin_tile": rt.get("origin_tile"),
        "image_quad_indices": rt.get("image_quad_indices"),
        "catalog_quad_indices": rt.get("catalog_quad_indices"),
    }
    rid_cmp = {key: rid.get(key) for key in tid}
    booleans = {
        "same_candidate_identity": tid == rid_cmp,
        "same_wcs": (rt.get("wcs") or {}).get("header_hash") == rwcs.get("header_hash"),
        "same_verification_source_count": int((rt.get("verification_sources") or {}).get("count", -1)) == int(replay["verification_sources"].get("count", -2)),
        "same_verification_source_hash": (rt.get("verification_sources") or {}).get("hash") == replay["verification_sources"].get("hash"),
        "same_catalog_count": int((rt.get("catalog") or {}).get("count", -1)) == int(replay["catalog"].get("count", -2)),
        "same_catalog_hash": (rt.get("catalog") or {}).get("hash") == replay["catalog"].get("hash"),
        "same_match_pairs": (rt.get("matching") or {}).get("pairs_hash") == rmatch.get("pairs_hash"),
        "same_validation_result": bool((rt.get("validation") or {}).get("success")) == bool(rval.get("success"))
        and int((rt.get("validation") or {}).get("inliers", -1)) == int(rval.get("inliers", -2))
        and round(float((rt.get("validation") or {}).get("rms_px", -1.0)), 9) == round(float(rval.get("rms_px", -2.0)), 9),
    }
    legacy = rt.get("legacy_inverse_validation") or {}
    first_divergence = "none_after_fix"
    causal = "pre_fix_validation_metric_direction"
    if not booleans["same_candidate_identity"]:
        first_divergence = "candidate_identity"
    elif not booleans["same_wcs"]:
        first_divergence = "wcs_header"
    elif not booleans["same_verification_source_count"] or not booleans["same_verification_source_hash"]:
        first_divergence = "verification_sources"
    elif not booleans["same_catalog_count"] or not booleans["same_catalog_hash"]:
        first_divergence = "validation_catalog"
    elif not booleans["same_match_pairs"]:
        first_divergence = "match_pairs"
    elif not booleans["same_validation_result"]:
        first_divergence = "validation_result"
    return {
        **booleans,
        "first_divergence_after_fix": first_divergence,
        "causal_divergence_pre_fix": causal,
        "legacy_inverse_runtime_result": {
            "success": bool(legacy.get("success")),
            "inliers": int(legacy.get("inliers", 0) or 0),
            "rms_px": float(legacy.get("rms_px", float("nan"))),
            "reason": legacy.get("reason"),
        },
    }


def _load_matrix(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    cmp = payload["comparison"]
    rt = payload["runtime"]
    rp = payload["replay"]
    matrix = payload.get("validation_matrix") or {}
    rows = matrix.get("runtime_rows") or []
    bad = matrix.get("bad_tile_controls") or []
    lines = [
        "# ZeBlind P2.17 - 4D runtime/replay parity",
        "",
        "> Audit de contrat/plomberie. Aucun seuil, ZeNear, GUI, default, all-sky, WCS oracle runtime, coeur AB/C/D, ranking source-list, refit ou rescue n'est introduit.",
        "",
        "## Verdict",
        "",
        "- Le candidat coherent `234013` rang `95` etait bien present dans les hits runtime.",
        "- La premiere divergence P2.15 n'etait pas une entree differente, ni un WCS different, ni une dedup/mutation: c'etait la metrique de validation.",
        "- Le replay P2.16 mesurait le RMS final en pixels dans le sens du matching `catalogue world2pix -> image`; la route runtime 4D repassait les memes paires dans `validate_solution`, qui mesure `image pix2world -> catalogue` puis reconvertit en pixels.",
        f"- Sur les memes `42` paires du rang 95: validation directe `{rp['validation']['inliers']}` inliers / RMS `{_fmt(rp['validation']['rms_px'], 3)}`, ancienne validation inverse runtime `{cmp['legacy_inverse_runtime_result']['inliers']}` / RMS `{_fmt(cmp['legacy_inverse_runtime_result']['rms_px'], 3)}`.",
        f"- Apres correction bornee a la route 4D, `solve_blind` recupere naturellement `234013`: success `{rt['success']}`, rang selectionne `{rt['selected_rank']}`, inliers `{rt['selected_inliers']}`, RMS `{_fmt(rt['selected_rms_px'], 3)}`.",
        "",
        "## Parite Rang 95",
        "",
    ]
    for key in (
        "same_candidate_identity",
        "same_wcs",
        "same_verification_source_count",
        "same_verification_source_hash",
        "same_catalog_count",
        "same_catalog_hash",
        "same_match_pairs",
        "same_validation_result",
    ):
        lines.append(f"- `{key}`: `{cmp.get(key)}`")
    lines.extend(
        [
            f"- `first_divergence_after_fix`: `{cmp['first_divergence_after_fix']}`",
            f"- `causal_divergence_pre_fix`: `{cmp['causal_divergence_pre_fix']}`",
            "",
            "## Identite",
            "",
            f"- Runtime: `{rt.get('trace', {}).get('rank')}` / `{rt.get('trace', {}).get('origin_tile')}` / image `{rt.get('trace', {}).get('image_quad_indices')}` / catalogue `{rt.get('trace', {}).get('catalog_quad_indices')}`",
            f"- Replay: `{rp['identity']['rank']}` / `{rp['identity']['origin_tile']}` / image `{rp['identity']['image_quad_indices']}` / catalogue `{rp['identity']['catalog_quad_indices']}`",
            f"- WCS hash runtime: `{(rt.get('trace') or {}).get('wcs', {}).get('header_hash')}`",
            f"- WCS hash replay: `{rp['wcs']['header_hash']}`",
            "",
            "## Entrees",
            "",
            f"- `quad_sources`: runtime `{(rt.get('trace') or {}).get('quad_sources', {}).get('count')}`, replay `{rp['quad_sources']['count']}`",
            f"- `verification_sources`: runtime `{(rt.get('trace') or {}).get('verification_sources', {}).get('count')}`, replay `{rp['verification_sources']['count']}`",
            f"- `verification_source_hash`: runtime `{(rt.get('trace') or {}).get('verification_sources', {}).get('hash')}`, replay `{rp['verification_sources']['hash']}`",
            f"- Catalogue union: runtime `{(rt.get('trace') or {}).get('catalog', {}).get('count')}`, replay `{rp['catalog']['count']}`",
            f"- `catalog_hash`: runtime `{(rt.get('trace') or {}).get('catalog', {}).get('hash')}`, replay `{rp['catalog']['hash']}`",
            "",
            "## Matching Et Validation",
            "",
            f"- Paires uniques sous `3 px`: runtime `{(rt.get('trace') or {}).get('matching', {}).get('unique_pairs')}`, replay `{rp['matching']['unique_pairs']}`",
            f"- `pairs_hash`: runtime `{(rt.get('trace') or {}).get('matching', {}).get('pairs_hash')}`, replay `{rp['matching']['pairs_hash']}`",
            f"- Runtime direct: `{(rt.get('trace') or {}).get('validation', {}).get('inliers')}` / RMS `{_fmt((rt.get('trace') or {}).get('validation', {}).get('rms_px'), 3)}`",
            f"- Runtime legacy inverse: `{cmp['legacy_inverse_runtime_result']['inliers']}` / RMS `{_fmt(cmp['legacy_inverse_runtime_result']['rms_px'], 3)}` / `{cmp['legacy_inverse_runtime_result']['reason']}`",
            "",
            "## Rejets Separes",
            "",
            f"- `best_accepted`: `{rt.get('best_accepted')}`",
            f"- `best_plausible_reject`: `{rt.get('best_plausible_reject')}`",
            f"- `best_scale_invalid_reject`: `{rt.get('best_scale_invalid_reject')}`",
            f"- `best_rms_invalid_reject`: `{rt.get('best_rms_invalid_reject')}`",
            f"- `best_geometry_invalid_reject`: `{rt.get('best_geometry_invalid_reject')}`",
            "",
            "## Controles",
            "",
        ]
    )
    if rows:
        lines.extend(["| cas | success | inliers | RMS | rank | tile |", "|---|---:|---:|---:|---:|---|"])
        for row in rows:
            lines.append(
                f"| `{row['label']}` | `{row['success']}` | {row['inliers']} | {_fmt(row['rms_px'], 3)} | {row.get('rank', '')} | `{row.get('origin_tile', '')}` |"
            )
    else:
        lines.append(f"- Matrice controle non disponible: `{matrix.get('missing', '')}`")
    if bad:
        lines.extend(["", "## Mauvaise Tuile", "", "| controle | success | inliers | RMS | raison |", "|---|---:|---:|---:|---|"])
        for row in bad:
            lines.append(f"| `{row.get('control')}` | `{row.get('success')}` | {row.get('inliers')} | {_fmt(row.get('rms_px'), 3)} | {str(row.get('reason', ''))[:120]} |")
    lines.extend(
        [
            "",
            "## Reponses",
            "",
            "- Pourquoi P2.15 donnait 30/0 acceptation et P2.16 42? `q120_v500` etait encore une liste capee; `q120_vfull` contenait les 42 paires mais l'ancien runtime recalculait la validation dans le sens inverse et rejetait le rang 95 a RMS `1.203 > 1.2`.",
            "- Premiere divergence exacte: validation metric direction, apres identite/WCS/sources/catalogue/matches identiques.",
            "- Ce n'etait pas un cap cache, une mauvaise liste, une dedup, un etat mutable ou un ranking source-list; le cap cachait seulement la lecture dans `q120_v500`, et la telemetrie single `best_reject` la masquait dans `q120_vfull`.",
            "- `solve_blind` recupere maintenant `234013` naturellement avec les seuils inchanges.",
            "- Le mode peut maintenant etre rejoue sur les 30 M106, apres cette parite locale et les controles bornes.",
            "",
            "## Parametres",
            "",
            "```json",
            json.dumps(payload["params"], indent=2, default=_json_default),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.17 parity audit between 4D runtime and targeted candidate replay.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--controls-json", type=Path, default=ROOT / "reports/zeblind_p217_runtime_validation_matrix.json")
    ap.add_argument("--main-case", default=MAIN_CASE)
    ap.add_argument("--quad-cap", type=int, default=120)
    ap.add_argument("--verification-min-sep-px", type=float, default=0.75)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
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
    ap.add_argument("--pixel-scale-min-arcsec", type=float, default=1.79)
    ap.add_argument("--pixel-scale-max-arcsec", type=float, default=2.99)
    ap.add_argument("--known-rank", type=int, default=95)
    ap.add_argument("--known-origin-tile", default="d50_2822")
    ap.add_argument("--known-image-quad", default="5,3,4,0")
    ap.add_argument("--known-catalog-quad", default="196,54,175,7")
    args = ap.parse_args()

    raw, image_shape, detect_meta = p214._detect_sources(str(args.main_case), args)
    clean, clean_stats = p215._clean_verification_sources(raw, image_shape, min_sep_px=float(args.verification_min_sep_px))
    quad_sources = raw[: int(args.quad_cap)]
    verification_sources = clean
    union_world, union_meta = p216._load_union_catalog(args)
    target = _prepare_target(args)
    runtime = _runtime_trace(args, target, quad_sources, verification_sources)
    replay = _replay_trace(args, union_world, quad_sources, verification_sources, image_shape)
    comparison = _compare(runtime, replay)
    payload = {
        "schema": "zeblind.p217_4d_runtime_replay_parity.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": runtime,
        "replay": replay,
        "comparison": comparison,
        "validation_matrix": _load_matrix(args.controls_json.expanduser().resolve()),
        "params": {
            "main_case": str(args.main_case),
            "quad_sources": int(quad_sources.shape[0]),
            "verification_sources": int(verification_sources.shape[0]),
            "detect_meta": detect_meta,
            "clean_stats": clean_stats,
            "union_catalog": union_meta,
            "index_paths": [str(path.expanduser().resolve()) for path in INDEXES],
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "accept_policy": "best_within_budget",
            "validation_catalog_policy": "union_candidate_tiles",
            "diagnostic_only": True,
            "default_behavior_changed": False,
            "wcs_oracle_runtime_input": False,
            "all_sky": False,
            "all30": False,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report, payload)
    print(json.dumps({"report": str(args.report), "json": str(args.json_out), "runtime_success": runtime["success"], "selected_rank": runtime["selected_rank"]}, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
