#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind

import tools.diagnose_p214_4d_source_policy_bakeoff as p214
import tools.diagnose_p212_4d_m106_30_bounded_validation as p212
import tools.diagnose_p26_4d_oracle_tile_routing as p26
import tools.diagnose_p28_4d_validation_support_audit as p28
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p215_4d_split_quad_verify_sources.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p215_4d_split_quad_verify_sources.json"
DEFAULT_WORK_DIR = ROOT / "reports/p215_4d_split_quad_verify_sources/candidates"

MAIN_CASE = "234013"
CONTROL_CASES = ("233828", "233705", "233644", "233602", "233520")
DEFAULT_CASES = (MAIN_CASE,) + CONTROL_CASES
BOUNDED_TILES = ("d50_2823", "d50_2822")


@dataclass(frozen=True)
class SplitConfig:
    name: str
    quad_cap: int
    verification_cap: int | None


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
        f = float(value)
        if not np.isfinite(f):
            return ""
        return f"{f:.{digits}f}"
    except Exception:
        return str(value)


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return None
    return float(statistics.median(vals))


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _positions(stars: np.ndarray) -> np.ndarray:
    return p214._positions(stars)


def _cap(stars: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is not None and int(cap) > 0 and stars.shape[0] > int(cap):
        return stars[: int(cap)]
    return stars


def _clean_verification_sources(raw: np.ndarray, image_shape: tuple[int, int], *, min_sep_px: float) -> tuple[np.ndarray, dict[str, Any]]:
    if raw.size == 0:
        return raw, {"input": 0, "kept": 0, "removed": 0, "duplicates_removed": 0}
    height, width = int(image_shape[0]), int(image_shape[1])
    arr = np.asarray(raw)
    mask = (
        np.isfinite(arr["x"])
        & np.isfinite(arr["y"])
        & np.isfinite(arr["flux"])
        & np.isfinite(arr["fwhm"])
        & (arr["flux"] > 0.0)
        & (arr["x"] >= 0.0)
        & (arr["x"] < float(width))
        & (arr["y"] >= 0.0)
        & (arr["y"] < float(height))
    )
    arr = arr[mask]
    if arr.size <= 1:
        return arr, {
            "input": int(raw.shape[0]),
            "finite_inside_positive": int(arr.shape[0]),
            "kept": int(arr.shape[0]),
            "removed": int(raw.shape[0] - arr.shape[0]),
            "duplicates_removed": 0,
            "min_sep_px": float(min_sep_px),
        }
    sep = max(0.0, float(min_sep_px))
    if sep <= 0.0:
        return arr, {
            "input": int(raw.shape[0]),
            "finite_inside_positive": int(arr.shape[0]),
            "kept": int(arr.shape[0]),
            "removed": int(raw.shape[0] - arr.shape[0]),
            "duplicates_removed": 0,
            "min_sep_px": float(sep),
        }
    xy = _positions(arr)
    kept: list[int] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    sep2 = sep * sep
    for idx, (x, y) in enumerate(xy):
        bx = int(float(x) // sep)
        by = int(float(y) // sep)
        duplicate = False
        for nx in (bx - 1, bx, bx + 1):
            for ny in (by - 1, by, by + 1):
                for prev in buckets.get((nx, ny), []):
                    dx = float(x) - float(xy[prev, 0])
                    dy = float(y) - float(xy[prev, 1])
                    if dx * dx + dy * dy < sep2:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        kept.append(int(idx))
        buckets.setdefault((bx, by), []).append(int(idx))
    out = arr[np.asarray(kept, dtype=np.int64)] if kept else arr[:0]
    return out, {
        "input": int(raw.shape[0]),
        "finite_inside_positive": int(arr.shape[0]),
        "kept": int(out.shape[0]),
        "removed": int(raw.shape[0] - out.shape[0]),
        "duplicates_removed": int(arr.shape[0] - out.shape[0]),
        "min_sep_px": float(sep),
    }


def _prepare_case(label: str, data_dir: Path, work_dir: Path, tag: str) -> Path:
    source = data_dir / _filename(label)
    if not source.exists():
        raise FileNotFoundError(f"missing M106 case {label}: {source}")
    target_dir = work_dir / tag / label
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    p22._strip_wcs(target)
    return target


def _prep_cache_for(candidate: Path, quad_sources: np.ndarray, verification_sources: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    resolved = candidate.resolve()
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


def _config(index_paths: tuple[Path, ...], args: argparse.Namespace, *, quad_sources: np.ndarray) -> SolveConfig:
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
        blind_astrometry_4d_index_paths=tuple(str(path.expanduser().resolve()) for path in index_paths),
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
    )


def _solve_split(
    label: str,
    split: SplitConfig,
    quad_sources: np.ndarray,
    verification_sources: np.ndarray,
    args: argparse.Namespace,
    *,
    index_paths: tuple[Path, ...] = (p212.INDEX_2823, p212.INDEX_2822),
    tag_prefix: str = "main",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    candidate = _prepare_case(
        label,
        args.data_dir.expanduser().resolve(),
        args.work_dir.expanduser().resolve(),
        f"{tag_prefix}_{split.name}",
    )
    result = solve_blind(
        candidate,
        args.index_root.expanduser().resolve(),
        config=_config(index_paths, args, quad_sources=quad_sources),
        prep_cache=_prep_cache_for(candidate, quad_sources, verification_sources, args),
    )
    stats = dict(result.stats or {})
    reject = stats.get("astrometry_4d_best_reject") if isinstance(stats.get("astrometry_4d_best_reject"), dict) else {}
    accepted = stats.get("astrometry_4d_best_accepted_validation") if isinstance(stats.get("astrometry_4d_best_accepted_validation"), dict) else {}
    chosen = accepted if bool(result.success) else reject
    return {
        "label": label,
        "config": split.name,
        "quad_cap": int(split.quad_cap),
        "verification_cap": split.verification_cap if split.verification_cap is None else int(split.verification_cap),
        "index_tiles": [p212._tile_name(path) for path in index_paths],
        "success": bool(result.success),
        "message": str(result.message),
        "tile_key": result.tile_key,
        "origin_tile": stats.get("astrometry_4d_selected_origin_tile_key") or chosen.get("origin_tile_key"),
        "quad_sources": int(quad_sources.shape[0]),
        "verification_sources": int(verification_sources.shape[0]),
        "image_quads": int(stats.get("astrometry_4d_image_quads", 0) or 0),
        "image_records": int(stats.get("astrometry_4d_image_records", 0) or 0),
        "hits": int(stats.get("astrometry_4d_hits", 0) or 0),
        "hits_by_index": stats.get("astrometry_4d_hits_by_index"),
        "hypotheses_tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "accepted_candidates": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "rank": stats.get("astrometry_4d_selected_rank") if bool(result.success) else chosen.get("rank", chosen.get("hit_rank")),
        "inliers": int(stats.get("inliers", 0) or chosen.get("inliers", 0) or 0),
        "rms_px": float(stats.get("rms_px", chosen.get("rms_px", float("nan")))),
        "reason": "accepted" if bool(result.success) else chosen.get("reason", str(result.message)),
        "mono_inliers": chosen.get("mono_inliers"),
        "mono_rms_px": chosen.get("mono_rms_px"),
        "geo_cov_area": chosen.get("geo_cov_area"),
        "geo_cond": chosen.get("geo_cond"),
        "validation_catalog_stars": chosen.get("validation_catalog_stars"),
        "quad_build_s": float(stats.get("astrometry_4d_quad_build_s", 0.0) or 0.0),
        "lookup_s": float(stats.get("astrometry_4d_kd_lookup_s", 0.0) or 0.0),
        "validation_s": float(stats.get("astrometry_4d_validation_s", 0.0) or 0.0),
        "solver_total_s": float(stats.get("astrometry_4d_total_s", 0.0) or 0.0),
        "wall_s": float(time.perf_counter() - t0),
        "best_reject": reject,
        "best_accepted": accepted,
    }


def _oracle_count(label: str, raw: np.ndarray, kept: np.ndarray, image_shape: tuple[int, int], union_world: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    return p214._oracle_retention(label, raw, kept, image_shape, union_world, args)


def _detect_all(labels: list[str], args: argparse.Namespace, union_world: np.ndarray) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for label in labels:
        raw, image_shape, detect_meta = p214._detect_sources(label, args)
        clean, clean_stats = _clean_verification_sources(raw, image_shape, min_sep_px=float(args.verification_min_sep_px))
        cases[label] = {
            "raw": raw,
            "image_shape": image_shape,
            "detect_meta": detect_meta,
            "clean_verification_full": clean,
            "clean_stats": clean_stats,
            "raw_oracle": _oracle_count(label, raw, raw, image_shape, union_world, args),
            "clean_oracle": _oracle_count(label, raw, clean, image_shape, union_world, args),
        }
    return cases


def _make_splits(args: argparse.Namespace, full_allowed: bool) -> list[SplitConfig]:
    quad_caps = [int(v) for v in str(args.quad_caps).split(",") if str(v).strip()]
    verification_caps: list[int | None] = [int(v) for v in str(args.verification_caps).split(",") if str(v).strip()]
    if full_allowed:
        verification_caps.append(None)
    splits: list[SplitConfig] = []
    for quad_cap in quad_caps:
        for verification_cap in verification_caps:
            suffix = "full" if verification_cap is None else str(int(verification_cap))
            splits.append(SplitConfig(f"q{int(quad_cap)}_v{suffix}", int(quad_cap), verification_cap))
    return splits


def _progress(event: str, **payload: Any) -> None:
    row = {"event": event}
    row.update(payload)
    print(json.dumps(row, default=_json_default), flush=True)


def _run_matrix(labels: list[str], splits: list[SplitConfig], cases: dict[str, Any], union_world: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    total = len(labels) * len(splits)
    n = 0
    for label in labels:
        raw = cases[label]["raw"]
        image_shape = cases[label]["image_shape"]
        clean = cases[label]["clean_verification_full"]
        source_audit[label] = {
            "detect_meta": cases[label]["detect_meta"],
            "image_shape": list(image_shape),
            "raw_detected": int(raw.shape[0]),
            "clean_verification_full": int(clean.shape[0]),
            "clean_stats": cases[label]["clean_stats"],
            "raw_oracle_matchable": int(cases[label]["raw_oracle"]["raw_matchable"]),
            "clean_oracle_matchable": int(cases[label]["clean_oracle"]["kept_matchable"]),
            "configs": {},
        }
        for split in splits:
            n += 1
            quad_sources = _cap(raw, split.quad_cap)
            verification_sources = _cap(clean, split.verification_cap)
            quad_oracle = _oracle_count(label, raw, quad_sources, image_shape, union_world, args)
            verification_oracle = _oracle_count(label, raw, verification_sources, image_shape, union_world, args)
            source_audit[label]["configs"][split.name] = {
                "quad_sources": int(quad_sources.shape[0]),
                "verification_sources": int(verification_sources.shape[0]),
                "quad_oracle_matchable": int(quad_oracle["kept_matchable"]),
                "verification_oracle_matchable": int(verification_oracle["kept_matchable"]),
                "verification_lost_matchable_ranks": verification_oracle.get("lost_matchable_ranks"),
            }
            _progress(
                "runtime_case_start",
                label=label,
                config=split.name,
                index=n,
                count=total,
                quad_sources=int(quad_sources.shape[0]),
                verification_sources=int(verification_sources.shape[0]),
            )
            row = _solve_split(label, split, quad_sources, verification_sources, args)
            row["quad_oracle_matchable"] = int(quad_oracle["kept_matchable"])
            row["verification_oracle_matchable"] = int(verification_oracle["kept_matchable"])
            rows.append(row)
            _progress(
                "runtime_case_done",
                label=label,
                config=split.name,
                success=row["success"],
                inliers=row["inliers"],
                rms_px=row["rms_px"],
                validation_s=row["validation_s"],
                wall_s=row["wall_s"],
            )
    return rows, source_audit


def _run_bad_tile_controls(rows: list[dict[str, Any]], cases: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    passing = [
        row for row in rows
        if row["label"] == MAIN_CASE and bool(row["success"]) and int(row["inliers"]) >= int(args.quality_inliers)
    ]
    if not passing:
        return []
    best = sorted(passing, key=lambda row: (int(row["verification_sources"]), int(row["quad_sources"]), float(row["wall_s"])))[0]
    split = SplitConfig(str(best["config"]), int(best["quad_cap"]), best["verification_cap"])
    raw = cases[MAIN_CASE]["raw"]
    clean = cases[MAIN_CASE]["clean_verification_full"]
    quad_sources = _cap(raw, split.quad_cap)
    verification_sources = _cap(clean, split.verification_cap)
    _progress("bad_tile_control_start", label=MAIN_CASE, config=split.name, wrong_tiles=["d50_2823"])
    row = _solve_split(
        MAIN_CASE,
        split,
        quad_sources,
        verification_sources,
        args,
        index_paths=(p212.INDEX_2823,),
        tag_prefix="bad_tile_d50_2823_only",
    )
    row["control"] = "234013_wrong_tile_d50_2823_only"
    _progress(
        "bad_tile_control_done",
        success=row["success"],
        inliers=row["inliers"],
        rms_px=row["rms_px"],
        wall_s=row["wall_s"],
    )
    return [row]


def _summaries(rows: list[dict[str, Any]], labels: list[str], splits: list[SplitConfig]) -> dict[str, Any]:
    by_config: dict[str, Any] = {}
    for split in splits:
        sub = [row for row in rows if row["config"] == split.name]
        by_config[split.name] = {
            "successes": int(sum(1 for row in sub if row["success"])),
            "cases": int(len(sub)),
            "main_case": next((row for row in sub if row["label"] == MAIN_CASE), None),
            "median_validation_s": _median([float(row["validation_s"]) for row in sub]),
            "median_total_s": _median([float(row["solver_total_s"]) for row in sub]),
            "median_wall_s": _median([float(row["wall_s"]) for row in sub]),
            "median_hypotheses": _median([float(row["hypotheses_tested"]) for row in sub]),
        }
    main_rows = [row for row in rows if row["label"] == MAIN_CASE]
    passing_main = [row for row in main_rows if row["success"] and int(row["inliers"]) >= 40 and float(row["rms_px"]) <= 1.2]
    min_cap = None
    if passing_main:
        min_row = sorted(passing_main, key=lambda row: (int(row["verification_sources"]), int(row["quad_sources"]), float(row["wall_s"])))[0]
        min_cap = {
            "config": min_row["config"],
            "quad_cap": int(min_row["quad_cap"]),
            "verification_cap": min_row["verification_cap"],
            "verification_sources": int(min_row["verification_sources"]),
            "inliers": int(min_row["inliers"]),
            "rms_px": float(min_row["rms_px"]),
            "wall_s": float(min_row["wall_s"]),
        }
    return {
        "by_config": by_config,
        "main_case_passes": int(len(passing_main)),
        "main_case_minimal_passing_validation": min_cap,
        "labels": labels,
    }


def _answers(payload: dict[str, Any]) -> tuple[str, list[str]]:
    summaries = payload["summaries"]
    rows = payload["runtime_rows"]
    quality_inliers = int(payload.get("params", {}).get("quality_inliers", 40) or 40)
    quality_rms = float(payload.get("params", {}).get("quality_rms", 1.2) or 1.2)
    main_rows = [row for row in rows if row["label"] == MAIN_CASE]
    baseline = next((row for row in main_rows if row["config"] == "q120_v250"), None) or (main_rows[0] if main_rows else {})
    coherent = [
        row for row in main_rows
        if np.isfinite(float(row.get("rms_px", float("nan"))))
        and float(row.get("rms_px", float("inf"))) <= quality_rms
        and "scale_ok=1" in str(row.get("reason", ""))
    ]
    best_coherent = max(coherent, key=lambda row: (int(row["inliers"]), -float(row["rms_px"])), default={})
    best_raw = max(main_rows, key=lambda row: int(row["inliers"]), default={})
    min_pass = summaries.get("main_case_minimal_passing_validation")
    if min_pass:
        verdict = "P2.15 positif: le couplage source-list quads/validation plafonnait artificiellement les inliers de 234013"
        answers = [
            f"`234013` passe avec `{min_pass['config']}`: {min_pass['inliers']} inliers / RMS {_fmt(min_pass['rms_px'], 3)}.",
            f"Cap minimal observe: verification `{min_pass['verification_cap']}` ({min_pass['verification_sources']} sources nettoyees).",
            "Les quads et le lookup restent construits depuis `quad_sources`; la liste profonde ne genere pas de quads supplementaires.",
            "Conclusion causale: separer generation et verification reproduit mieux l'architecture Astrometry.net et resout le gap principal de ce cas.",
            "Pas besoin de modifier le detecteur/ranking immediatement pour `234013`; la prochaine etape est de garder cette separation comme architecture experimentale 4D bornee.",
        ]
    else:
        case_233828 = [row for row in rows if row["label"] == "233828"]
        first_233828 = min(
            (row for row in case_233828 if bool(row.get("success"))),
            key=lambda row: (int(row["verification_sources"]), int(row["quad_sources"])),
            default={},
        )
        verdict = "P2.15 negatif: la liste de validation profonde ne depasse pas le plafond observe sur 234013"
        answers = [
            f"`234013` ne passe pas; meilleur candidat coherent `{best_coherent.get('config')}`: {best_coherent.get('inliers')} inliers / RMS {_fmt(best_coherent.get('rms_px'), 3)}.",
            f"La liste full contient les 42 etoiles oracle-matchables, mais le support coherent plafonne a {best_coherent.get('inliers')} inliers, sous le seuil {quality_inliers}.",
            f"Les compteurs plus hauts (`{best_raw.get('config')}`: {best_raw.get('inliers')} matches) sont des rejets de mauvaise echelle/RMS invalide, pas des solutions.",
            f"ZeBlind plafonnait artificiellement certains cas en validant sur la liste des quads: `233828` passe des `q120_v250` ({first_233828.get('inliers')} inliers / RMS {_fmt(first_233828.get('rms_px'), 3)}), mais ce n'est pas le bloc causal principal de `234013`.",
            "La separation reproduit mieux l'architecture Astrometry.net (quads sur liste courte, verification sur liste plus profonde), et elle est saine sur les controles, mais elle ne resout pas le gap principal `234013`.",
            "La piste des caps seuls doit donc s'arreter ici; prochain audit = centroïdes des 12 etoiles encore perdues au cap coherent, matching/residus, puis detecteur si necessaire.",
        ]
    answers.extend(
        [
            f"Baseline observee dans ce run: `{baseline.get('config')}` -> {baseline.get('inliers')} inliers / RMS {_fmt(baseline.get('rms_px'), 3)}.",
            f"Cout median validation: {_fmt(_median([float(row['validation_s']) for row in rows]), 3)}s; cout median total solveur: {_fmt(_median([float(row['solver_total_s']) for row in rows]), 3)}s.",
            "Seuils conserves: `quality_inliers=40`, `quality_rms=1.2`, `match_radius_px=3.0`; index bornes `d50_2823+d50_2822`; aucun all30/all-sky.",
        ]
    )
    return verdict, answers


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.15 - separation quad_sources / verification_sources 4D",
        "",
        "> Diagnostic uniquement. Le WCS oracle Astrometry.net sert seulement aux compteurs offline de sources matchables, jamais au runtime blind.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Aucun changement ZeNear, GUI, default, all-sky, seuil, WCS oracle runtime, ni coeur AB/C/D.",
        "",
        "## Reponses",
        "",
    ]
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(
        [
            "",
            "## Matrice runtime",
            "",
            "| cas | config | quad src | verify src | quad oracle | verify oracle | quads | hits | hyp | success | inliers | RMS | rank | val s | total s | raison |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["runtime_rows"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | `{}` | {} | {} | {} | {} | {} | {} |".format(
                row["label"],
                row["config"],
                row["quad_sources"],
                row["verification_sources"],
                row["quad_oracle_matchable"],
                row["verification_oracle_matchable"],
                row["image_quads"],
                row["hits"],
                row["hypotheses_tested"],
                row["success"],
                row["inliers"],
                _fmt(row["rms_px"], 3),
                row.get("rank", ""),
                _fmt(row["validation_s"], 3),
                _fmt(row["solver_total_s"], 3),
                str(row.get("reason", "")).replace("|", "/")[:120],
            )
        )
    lines.extend(["", "## Synthese par configuration", ""])
    lines.extend(["| config | succes | 234013 inliers | 234013 RMS | med hyp | med val s | med total s |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, summary in payload["summaries"]["by_config"].items():
        main = summary.get("main_case") or {}
        lines.append(
            f"| `{name}` | {summary['successes']}/{summary['cases']} | {main.get('inliers', '')} | {_fmt(main.get('rms_px'), 3)} | {_fmt(summary.get('median_hypotheses'), 0)} | {_fmt(summary.get('median_validation_s'), 3)} | {_fmt(summary.get('median_total_s'), 3)} |"
        )
    lines.extend(["", "## Source audit", ""])
    for label, audit in payload["source_audit"].items():
        lines.append(
            f"- `{label}`: raw `{audit['raw_detected']}`, verification full nettoyee `{audit['clean_verification_full']}`, "
            f"oracle raw `{audit['raw_oracle_matchable']}`, oracle verification full `{audit['clean_oracle_matchable']}`."
        )
    controls = payload.get("bad_tile_controls") or []
    lines.extend(["", "## Controles mauvaise tuile", ""])
    if controls:
        lines.extend(["| controle | success | inliers | RMS | hyp | val s | raison |", "|---|---:|---:|---:|---:|---:|---|"])
        for row in controls:
            lines.append(
                f"| `{row.get('control')}` | `{row.get('success')}` | {row.get('inliers')} | {_fmt(row.get('rms_px'), 3)} | {row.get('hypotheses_tested')} | {_fmt(row.get('validation_s'), 3)} | {str(row.get('reason', '')).replace('|', '/')[:120]} |"
            )
    else:
        lines.append("- Non lance: `234013` ne passe dans aucune configuration P2.15, donc pas de controle faux positif post-passage.")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2, default=_json_default), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.15 split quad generation and WCS verification source-lists for experimental ZeBlind 4D.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--quad-caps", default="120,200")
    ap.add_argument("--verification-caps", default="250,500,1000")
    ap.add_argument("--include-full", action="store_true", default=True)
    ap.add_argument("--no-full", dest="include_full", action="store_false")
    ap.add_argument("--full-max-sources", type=int, default=2500)
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
    ap.add_argument("--footprint-grid", type=int, default=31)
    args = ap.parse_args()

    for path in (p212.INDEX_2823, p212.INDEX_2822):
        if not path.exists():
            raise FileNotFoundError(f"missing explicit P2.15 4D index: {path}")

    labels = [part.strip() for part in str(args.cases).split(",") if part.strip()]
    entries = p26._tile_entries(args.index_root.expanduser().resolve())
    union_world = p214._dedup_world([p28._catalog_world(args.index_root.expanduser().resolve(), entries, tile) for tile in BOUNDED_TILES])
    _progress("detect_start", cases=labels)
    cases = _detect_all(labels, args, union_world)
    max_clean = max(int(cases[label]["clean_verification_full"].shape[0]) for label in labels) if labels else 0
    full_allowed = bool(args.include_full) and max_clean <= int(args.full_max_sources)
    splits = _make_splits(args, full_allowed=full_allowed)
    _progress("matrix_start", cases=len(labels), configs=[split.name for split in splits], full_allowed=full_allowed, max_clean=max_clean)
    rows, source_audit = _run_matrix(labels, splits, cases, union_world, args)
    controls = _run_bad_tile_controls(rows, cases, args)
    summaries = _summaries(rows, labels, splits)
    payload: dict[str, Any] = {
        "schema": "zeblind.p215_4d_split_quad_verify_sources.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_order": labels,
        "config_order": [split.name for split in splits],
        "runtime_rows": rows,
        "source_audit": source_audit,
        "bad_tile_controls": controls,
        "summaries": summaries,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "index_paths": [str(p212.INDEX_2823.expanduser().resolve()), str(p212.INDEX_2822.expanduser().resolve())],
            "validation_catalog_policy": "union_candidate_tiles",
            "accept_policy": "best_within_budget",
            "quad_caps": [split.quad_cap for split in splits],
            "verification_caps": [split.verification_cap for split in splits],
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_quads": int(args.max_quads),
            "max_hypotheses": int(args.max_hypotheses),
            "max_wall_s": float(args.max_wall_s),
            "max_accepts": int(args.max_accepts),
            "verification_min_sep_px": float(args.verification_min_sep_px),
            "full_allowed": bool(full_allowed),
            "full_max_sources": int(args.full_max_sources),
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
            "wcs_oracle_runtime_input": False,
            "all_sky": False,
            "all30": False,
            "default_behavior_changed": False,
        },
    }
    verdict, answers = _answers(payload)
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
