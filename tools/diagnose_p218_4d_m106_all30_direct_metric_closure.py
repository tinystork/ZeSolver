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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind

import tools.diagnose_p212_4d_m106_30_bounded_validation as p212
import tools.diagnose_p214_4d_source_policy_bakeoff as p214
import tools.diagnose_p215_4d_split_quad_verify_sources as p215
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p218_4d_m106_all30_direct_metric_closure.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p218_4d_m106_all30_direct_metric_closure.json"
DEFAULT_WORK_DIR = ROOT / "reports/p218_4d_m106_all30_direct_metric_closure"
INDEX_2823 = p212.INDEX_2823
INDEX_2822 = p212.INDEX_2822
LOW_FOOTPRINT_D50_2822_CONTROLS = ("232144", "232205", "232247", "232350", "232102")
WRONG_TILE_2823_CONTROLS = ("234013",)


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


def _percentile(values: list[float], pct: float) -> float | None:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, float(pct)))


def _median(values: list[float]) -> float | None:
    return _percentile(values, 50.0)


def _labels_from_data(data_dir: Path) -> list[str]:
    out: list[str] = []
    prefix = "Light_mosaic_M 106_20.0s_IRCUT_20250518-"
    for path in sorted(data_dir.glob(prefix + "*.fit")):
        stem = path.name[len(prefix):]
        if stem.endswith(".fit"):
            out.append(stem[:-4])
    return out


def _prepare_case(label: str, source_dir: Path, work_dir: Path, tag: str) -> Path:
    source = source_dir / p215._filename(label)
    if not source.exists():
        raise FileNotFoundError(f"missing M106 case {label}: {source}")
    target_dir = work_dir / tag / label
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


def _config(index_paths: tuple[Path, ...], args: argparse.Namespace, *, accept_policy: str | None = None) -> SolveConfig:
    return SolveConfig(
        max_stars=int(args.quad_sources),
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
        blind_astrometry_4d_index_paths=tuple(str(path.expanduser().resolve()) for path in index_paths),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
        blind_astrometry_4d_accept_policy=str(accept_policy or args.accept_policy),
        blind_astrometry_4d_max_accepts=int(args.max_accepts),
        blind_astrometry_4d_code_tol=float(args.code_tol),
        blind_astrometry_4d_max_hits=int(args.max_hits_4d),
        blind_astrometry_4d_max_hits_per_image_quad=int(args.max_hits_per_image_quad),
        blind_astrometry_4d_max_hypotheses=int(args.max_hypotheses),
        blind_astrometry_4d_image_strategy=str(args.image_strategy),
        blind_astrometry_4d_match_radius_px=float(args.match_radius_px),
    )


def _detect_sources(label: str, args: argparse.Namespace) -> tuple[np.ndarray, tuple[int, int], dict[str, Any], np.ndarray, dict[str, Any]]:
    raw, image_shape, detect_meta = p214._detect_sources(label, args)
    clean, clean_stats = p215._clean_verification_sources(raw, image_shape, min_sep_px=float(args.verification_min_sep_px))
    return raw, image_shape, detect_meta, clean, clean_stats


def _chosen_validation(stats: dict[str, Any], success: bool) -> dict[str, Any]:
    if success and isinstance(stats.get("astrometry_4d_best_accepted_validation"), dict):
        return dict(stats["astrometry_4d_best_accepted_validation"])
    if isinstance(stats.get("astrometry_4d_best_reject"), dict):
        return dict(stats["astrometry_4d_best_reject"])
    return {}


def _reject_summary(row: dict[str, Any]) -> dict[str, Any]:
    val = row.get("chosen_validation") or row.get("best_reject") or {}
    return {
        "scale": val.get("pix_scale_arcsec"),
        "rms": val.get("rms_px"),
        "coverage": {
            "cov_x": val.get("geo_cov_x"),
            "cov_y": val.get("geo_cov_y"),
            "cov_area": val.get("geo_cov_area"),
            "cond": val.get("geo_cond"),
        },
        "reason": val.get("reason") or row.get("message"),
        "inliers": val.get("inliers"),
        "rank": val.get("hit_rank") or row.get("rank"),
        "origin_tile": val.get("origin_tile_key") or row.get("origin_tile"),
    }


def _run_case(
    label: str,
    index_paths: tuple[Path, ...],
    args: argparse.Namespace,
    *,
    tag: str,
    accept_policy: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    target = _prepare_case(label, args.data_dir.expanduser().resolve(), args.work_dir.expanduser().resolve(), tag)
    raw, image_shape, detect_meta, clean, clean_stats = _detect_sources(label, args)
    quad_sources = raw[: int(args.quad_sources)]
    verification_sources = clean
    result = solve_blind(
        target,
        args.index_root.expanduser().resolve(),
        config=_config(index_paths, args, accept_policy=accept_policy),
        prep_cache=_prep_cache(target, quad_sources, verification_sources, args),
    )
    stats = dict(result.stats or {})
    chosen = _chosen_validation(stats, bool(result.success))
    best_accepted = stats.get("astrometry_4d_best_accepted_validation") if isinstance(stats.get("astrometry_4d_best_accepted_validation"), dict) else {}
    best_reject = stats.get("astrometry_4d_best_reject") if isinstance(stats.get("astrometry_4d_best_reject"), dict) else {}
    legacy_inliers = chosen.get("legacy_inverse_inliers")
    legacy_rms = chosen.get("legacy_inverse_rms_px")
    direct_inliers = int(chosen.get("inliers", stats.get("inliers", 0)) or 0)
    direct_rms = float(chosen.get("rms_px", stats.get("rms_px", float("nan"))))
    legacy_inliers_int = int(legacy_inliers) if legacy_inliers is not None else 0
    legacy_rms_float = float(legacy_rms) if legacy_rms is not None else float("nan")
    direct_success = bool(chosen.get("success", bool(result.success)))
    legacy_quality = str(chosen.get("legacy_inverse_quality", ""))
    legacy_success = legacy_quality == "GOOD"
    return {
        "label": label,
        "tag": tag,
        "success": bool(result.success),
        "message": str(result.message),
        "candidate": str(target),
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "raw_detected": int(raw.shape[0]),
        "quad_sources": int(quad_sources.shape[0]),
        "verification_sources": int(verification_sources.shape[0]),
        "clean_stats": clean_stats,
        "detect_meta": detect_meta,
        "rank": stats.get("astrometry_4d_selected_rank") or chosen.get("hit_rank"),
        "local_rank": stats.get("astrometry_4d_selected_local_rank") or chosen.get("local_rank"),
        "origin_tile": stats.get("astrometry_4d_selected_origin_tile_key") or chosen.get("origin_tile_key"),
        "inliers": direct_inliers,
        "rms_px": direct_rms,
        "residual_metric": chosen.get("residual_metric"),
        "legacy_inverse_inliers": legacy_inliers_int,
        "legacy_inverse_rms_px": legacy_rms_float,
        "legacy_inverse_quality": legacy_quality,
        "legacy_inverse_reason": chosen.get("legacy_inverse_reason"),
        "direct_minus_legacy_inliers": int(direct_inliers - legacy_inliers_int),
        "legacy_minus_direct_rms_px": float(legacy_rms_float - direct_rms) if np.isfinite(legacy_rms_float) and np.isfinite(direct_rms) else None,
        "direct_success": direct_success,
        "legacy_success": bool(legacy_success),
        "decision_diff_direct_vs_legacy": bool(direct_success != legacy_success),
        "hits": int(stats.get("astrometry_4d_hits", 0) or 0),
        "hypotheses_tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "accepted_candidates": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "quad_build_s": float(stats.get("astrometry_4d_quad_build_s", 0.0) or 0.0),
        "lookup_s": float(stats.get("astrometry_4d_kd_lookup_s", 0.0) or 0.0),
        "validation_s": float(stats.get("astrometry_4d_validation_s", 0.0) or 0.0),
        "solver_total_s": float(stats.get("astrometry_4d_total_s", 0.0) or 0.0),
        "wall_s": float(time.perf_counter() - t0),
        "max_accepts_hit": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0) >= int(args.max_accepts),
        "max_wall_hit": stats.get("astrometry_4d_stop_reason") == "cancelled",
        "chosen_validation": chosen,
        "best_accepted": dict(best_accepted),
        "best_reject": dict(best_reject),
        "best_plausible_reject": stats.get("astrometry_4d_best_plausible_reject") or {},
        "best_scale_invalid_reject": stats.get("astrometry_4d_best_scale_invalid_reject") or {},
        "best_rms_invalid_reject": stats.get("astrometry_4d_best_rms_invalid_reject") or {},
        "best_geometry_invalid_reject": stats.get("astrometry_4d_best_geometry_invalid_reject") or {},
        "reject_reason_counts": stats.get("astrometry_4d_reject_reason_counts") or {},
    }


def _make_incompatible_index(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    # Keep the 4D codes searchable but break the star-id -> RA/Dec contract.
    payload["catalog_ra_dec"] = np.asarray(payload["catalog_ra_dec"], dtype=np.float64)[::-1].copy()
    np.savez_compressed(target, **payload)
    return target


def _negative_controls(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in LOW_FOOTPRINT_D50_2822_CONTROLS:
        rows.append(_run_case(label, (INDEX_2822,), args, tag="neg_d50_2822_only"))
        rows[-1]["control"] = "d50_2822_only_low_footprint"
    for label in WRONG_TILE_2823_CONTROLS:
        rows.append(_run_case(label, (INDEX_2823,), args, tag="neg_d50_2823_only"))
        rows[-1]["control"] = "d50_2823_only_234013"

    missing = args.work_dir.expanduser().resolve() / "missing_indexes" / "missing_4d_index.npz"
    rows.append(_run_case("234013", (missing,), args, tag="neg_missing_index"))
    rows[-1]["control"] = "missing_index"

    incompatible = _make_incompatible_index(
        INDEX_2822.expanduser().resolve(),
        args.work_dir.expanduser().resolve() / "incompatible_indexes" / "d50_2822_reversed_catalog.npz",
    )
    rows.append(_run_case("234013", (INDEX_2823.expanduser().resolve(), incompatible), args, tag="neg_incompatible_catalog_order"))
    rows[-1]["control"] = "incompatible_shuffled_catalog"
    return rows


def _reversed_order_controls(labels: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        rows.append(_run_case(label, (INDEX_2822, INDEX_2823), args, tag="reversed_index_order"))
        rows[-1]["control"] = "reversed_index_order"
    return rows


def _first_accept_legacy_guard(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for label in ("234013",):
        rows.append(_run_case(label, (INDEX_2823, INDEX_2822), args, tag="first_accept_legacy_guard", accept_policy="first_accept"))
        rows[-1]["control"] = "first_accept_legacy_guard"
    return rows


def _metric_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    din = [float(row["direct_minus_legacy_inliers"]) for row in rows if row.get("legacy_inverse_quality")]
    drms = [float(row["legacy_minus_direct_rms_px"]) for row in rows if row.get("legacy_minus_direct_rms_px") is not None]
    worse_direct = [
        row for row in rows
        if row.get("legacy_minus_direct_rms_px") is not None and float(row["legacy_minus_direct_rms_px"]) < -0.05
    ]
    return {
        "delta_inliers_direct_minus_inverse": {
            "median": _median(din),
            "p95": _percentile(din, 95),
            "max": max(din) if din else None,
            "min": min(din) if din else None,
        },
        "delta_rms_inverse_minus_direct": {
            "median": _median(drms),
            "p95": _percentile(drms, 95),
            "max": max(drms) if drms else None,
            "min": min(drms) if drms else None,
        },
        "decision_diff_count": int(sum(1 for row in rows if bool(row.get("decision_diff_direct_vs_legacy")))),
        "decision_diff_labels": [row["label"] for row in rows if bool(row.get("decision_diff_direct_vs_legacy"))],
        "direct_worse_by_more_than_0p05_count": int(len(worse_direct)),
        "direct_worse_by_more_than_0p05_labels": [row["label"] for row in worse_direct],
    }


def _cost_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "median_total_s": _median([float(row["solver_total_s"]) for row in rows]),
        "p95_total_s": _percentile([float(row["solver_total_s"]) for row in rows], 95),
        "median_validation_s": _median([float(row["validation_s"]) for row in rows]),
        "p95_validation_s": _percentile([float(row["validation_s"]) for row in rows], 95),
        "median_hypotheses": _median([float(row["hypotheses_tested"]) for row in rows]),
        "p95_hypotheses": _percentile([float(row["hypotheses_tested"]) for row in rows], 95),
        "max_accepts_cases": [row["label"] for row in rows if bool(row.get("max_accepts_hit"))],
        "max_wall_cases": [row["label"] for row in rows if bool(row.get("max_wall_hit"))],
        "max_verification_sources": max((int(row["verification_sources"]) for row in rows), default=0),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    cost = payload["cost_stats"]
    metric = payload["metric_stats"]
    lines = [
        "# ZeBlind P2.18 - 4D M106 all30 direct metric closure",
        "",
        "> `solve_blind` uniquement. Aucun seuil, ZeNear, GUI, backend par defaut, all-sky, WCS oracle runtime, coeur AB/C/D, refit TAN/SIP ou rescue n'est modifie.",
        "",
        "## Verdict",
        "",
        f"- M106 all30: `{summary['all30_successes']}/{summary['all30_cases']}`.",
        f"- Faux positifs controles negatifs: `{summary['negative_accepts']}/{summary['negative_cases']}`.",
        f"- Ordre inverse `[d50_2822,d50_2823]`: `{summary['reversed_successes']}/{summary['reversed_cases']}`.",
        f"- Tests contrat: `{summary['tests']}`.",
        f"- Statut RC experimental: `{summary['experimental_release_candidate']}`.",
        "",
        "## All30",
        "",
        "| image | success | rank | tuile | inliers | RMS | legacy inliers | legacy RMS | hyp | accepts | stop | quad | lookup | validation | total |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["all30_rows"]:
        lines.append(
            f"| `{row['label']}` | `{row['success']}` | {row.get('rank')} | `{row.get('origin_tile')}` | {row['inliers']} | {_fmt(row['rms_px'], 3)} | {row['legacy_inverse_inliers']} | {_fmt(row['legacy_inverse_rms_px'], 3)} | {row['hypotheses_tested']} | {row['accepted_candidates']} | `{row['stop_reason']}` | {_fmt(row['quad_build_s'], 3)} | {_fmt(row['lookup_s'], 3)} | {_fmt(row['validation_s'], 3)} | {_fmt(row['solver_total_s'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Ecart Direct Legacy",
            "",
            f"- Delta inliers direct-inverse: `{metric['delta_inliers_direct_minus_inverse']}`",
            f"- Delta RMS inverse-direct: `{metric['delta_rms_inverse_minus_direct']}`",
            f"- Decisions differentes: `{metric['decision_diff_count']}` / `{metric['decision_diff_labels']}`",
            f"- Direct nettement pire (>0.05 px): `{metric['direct_worse_by_more_than_0p05_count']}` / `{metric['direct_worse_by_more_than_0p05_labels']}`",
            "",
            "## Cout",
            "",
            f"- Median total: `{_fmt(cost['median_total_s'], 3)}s`, p95 total: `{_fmt(cost['p95_total_s'], 3)}s`",
            f"- Median validation: `{_fmt(cost['median_validation_s'], 3)}s`, p95 validation: `{_fmt(cost['p95_validation_s'], 3)}s`",
            f"- Median hypotheses: `{_fmt(cost['median_hypotheses'], 0)}`, p95 hypotheses: `{_fmt(cost['p95_hypotheses'], 0)}`",
            f"- Cas touchant `max_accepts`: `{cost['max_accepts_cases']}`",
            f"- Cas touchant `max_wall_s`: `{cost['max_wall_cases']}`",
            f"- Max `verification_sources`: `{cost['max_verification_sources']}`",
            "",
            "## Controles Negatifs",
            "",
            "| controle | image | success | inliers | RMS | scale | coverage | raison |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["negative_controls"]:
        rej = _reject_summary(row)
        lines.append(
            f"| `{row.get('control')}` | `{row['label']}` | `{row['success']}` | {rej.get('inliers')} | {_fmt(rej.get('rms'), 3)} | {_fmt(rej.get('scale'), 3)} | {_fmt((rej.get('coverage') or {}).get('cov_area'), 3)} | {str(rej.get('reason'))[:140]} |"
        )
    lines.extend(
        [
            "",
            "## Ordre Inverse Et Policy",
            "",
            "| controle | image | success | rank | tuile | inliers | RMS | legacy quality |",
            "|---|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in payload["reversed_order_controls"] + payload["first_accept_controls"]:
        lines.append(
            f"| `{row.get('control')}` | `{row['label']}` | `{row['success']}` | {row.get('rank')} | `{row.get('origin_tile')}` | {row['inliers']} | {_fmt(row['rms_px'], 3)} | `{row.get('legacy_inverse_quality')}` |"
        )
    lines.extend(["", "## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2, default=_json_default), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.18 M106 all30 direct-metric closure for experimental 4D runtime.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--cases", default="auto")
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
    ap.add_argument("--pixel-scale-min-arcsec", type=float, default=1.79)
    ap.add_argument("--pixel-scale-max-arcsec", type=float, default=2.99)
    ap.add_argument("--accept-policy", default="best_within_budget")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    labels = _labels_from_data(data_dir) if str(args.cases).strip().lower() == "auto" else [p.strip() for p in str(args.cases).split(",") if p.strip()]
    index_paths = (INDEX_2823.expanduser().resolve(), INDEX_2822.expanduser().resolve())
    for path in index_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing explicit 4D index: {path}")

    all30_rows = []
    for idx, label in enumerate(labels, start=1):
        print(json.dumps({"event": "all30_start", "label": label, "index": idx, "count": len(labels)}), flush=True)
        row = _run_case(label, index_paths, args, tag="all30_q120_vfull")
        all30_rows.append(row)
        print(json.dumps({"event": "all30_done", "label": label, "success": row["success"], "inliers": row["inliers"], "rms": row["rms_px"], "total_s": row["solver_total_s"]}, default=_json_default), flush=True)

    negative = _negative_controls(args)
    reversed_rows = _reversed_order_controls(labels, args)
    first_accept_rows = _first_accept_legacy_guard(args)
    successes = [row for row in all30_rows if bool(row["success"])]
    negative_accepts = [row for row in negative if bool(row["success"]) and row.get("control") != "incompatible_shuffled_catalog"]
    incompatible_accepts = [row for row in negative if bool(row["success"]) and row.get("control") == "incompatible_shuffled_catalog"]
    reversed_successes = [row for row in reversed_rows if bool(row["success"])]
    metric_stats = _metric_stats(all30_rows)
    cost_stats = _cost_stats(all30_rows)
    tests_summary = "pytest -q tests/test_quad_code_diagnostic.py"
    rc = (
        len(successes) == len(all30_rows)
        and not negative_accepts
        and not incompatible_accepts
        and len(reversed_successes) == len(reversed_rows)
    )
    answers = [
        f"Le corpus M106 passe `{len(successes)}/{len(all30_rows)}` via `solve_blind` en `q120_vfull`.",
        f"Faux positifs observes sur mauvaises configurations: `{len(negative_accepts) + len(incompatible_accepts)}`.",
        f"Runtime/replay reste verrouille par les tests P2.17/P2.18; le rapport P2.18 conserve `residual_metric='catalog_world2pix_to_image_px'` et les champs `legacy_inverse_*`.",
        f"Decisions direct vs legacy differentes sur `{metric_stats['decision_diff_count']}` cas: `{metric_stats['decision_diff_labels']}`.",
        f"Cout `q120_vfull`: median total `{_fmt(cost_stats['median_total_s'], 3)}s`, p95 total `{_fmt(cost_stats['p95_total_s'], 3)}s`; median validation `{_fmt(cost_stats['median_validation_s'], 3)}s`.",
        "Baseline experimentale simple conservee si le cout reste dans cette enveloppe; aucune policy staged n'est implementee dans P2.18.",
        "Experimental release candidate: oui" if rc else "Experimental release candidate: non, au moins un critere P2.18 est rouge.",
        "Prochaine limite produit reelle apres M106: elargir la non-regression hors champ M106 et indices voisins avant promotion produit plus large.",
    ]
    payload = {
        "schema": "zeblind.p218_4d_m106_all30_direct_metric_closure.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "all30_rows": all30_rows,
        "negative_controls": negative,
        "reversed_order_controls": reversed_rows,
        "first_accept_controls": first_accept_rows,
        "metric_stats": metric_stats,
        "cost_stats": cost_stats,
        "summary": {
            "all30_cases": int(len(all30_rows)),
            "all30_successes": int(len(successes)),
            "negative_cases": int(len(negative)),
            "negative_accepts": int(len(negative_accepts) + len(incompatible_accepts)),
            "reversed_cases": int(len(reversed_rows)),
            "reversed_successes": int(len(reversed_successes)),
            "experimental_release_candidate": bool(rc),
            "tests": tests_summary,
        },
        "answers": answers,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "index_paths": [str(path) for path in index_paths],
            "quad_sources": int(args.quad_sources),
            "verification_sources": "full",
            "blind_astrometry_4d_validation_catalog_policy": "union_candidate_tiles",
            "blind_astrometry_4d_accept_policy": str(args.accept_policy),
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_quads": int(args.max_quads),
            "max_hypotheses": int(args.max_hypotheses),
            "max_accepts": int(args.max_accepts),
            "max_wall_s": float(args.max_wall_s),
            "cases": labels,
            "diagnostic_only": False,
            "default_behavior_changed": False,
            "wcs_oracle_runtime_input": False,
            "all_sky": False,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report, payload)
    print(json.dumps({"report": str(args.report), "json": str(args.json_out), "all30": f"{len(successes)}/{len(all30_rows)}", "rc": bool(rc)}, default=_json_default))
    return 0 if rc else 1


if __name__ == "__main__":
    raise SystemExit(main())
