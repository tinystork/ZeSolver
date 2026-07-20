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

import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p211_4d_experimental_runtime_hardening.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p211_4d_experimental_runtime_hardening.json"
DEFAULT_WORK_DIR = ROOT / "reports/p211_4d_experimental_runtime_hardening/candidates"
INDEX_2823 = ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"
INDEX_2822 = ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz"
CASES = ("232329", "232431", "232144", "232205", "232247", "232350", "232102", "232513", "232534", "232658")
BAD_TILE_CONTROL = ("232144", "232205", "232247", "232350", "232102")


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


def _config(index_paths: tuple[Path, ...], args: argparse.Namespace) -> SolveConfig:
    return SolveConfig(
        max_stars=int(args.max_stars),
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


def _prepare_case(label: str, source_dir: Path, work_dir: Path, tag: str) -> Path:
    source = source_dir / _filename(label)
    if not source.exists():
        raise FileNotFoundError(f"missing case FITS: {source}")
    target_dir = work_dir / tag / label
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    p22._strip_wcs(target)
    return target


def _run_case(label: str, index_paths: tuple[Path, ...], args: argparse.Namespace, *, tag: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    candidate = _prepare_case(label, args.data_dir.expanduser().resolve(), args.work_dir.expanduser().resolve(), tag)
    cfg = _config(index_paths, args)
    result = solve_blind(candidate, args.index_root.expanduser().resolve(), config=cfg)
    stats = dict(result.stats or {})
    return {
        "label": label,
        "tag": tag,
        "success": bool(result.success),
        "message": str(result.message),
        "tile_key": result.tile_key,
        "candidate": str(candidate),
        "inliers": int(stats.get("inliers", 0) or 0),
        "rms_px": float(stats.get("rms_px", float("nan"))),
        "rank": stats.get("astrometry_4d_selected_rank"),
        "local_rank": stats.get("astrometry_4d_selected_local_rank"),
        "origin_tile": stats.get("astrometry_4d_selected_origin_tile_key"),
        "hits": int(stats.get("astrometry_4d_hits", 0) or 0),
        "hits_by_index": stats.get("astrometry_4d_hits_by_index"),
        "hypotheses_tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "accepted_candidates": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "accept_policy": stats.get("astrometry_4d_accept_policy"),
        "validation_catalog_policy": stats.get("astrometry_4d_validation_catalog_policy"),
        "quad_build_s": float(stats.get("astrometry_4d_quad_build_s", 0.0) or 0.0),
        "lookup_s": float(stats.get("astrometry_4d_kd_lookup_s", 0.0) or 0.0),
        "validation_s": float(stats.get("astrometry_4d_validation_s", 0.0) or 0.0),
        "solver_total_s": float(stats.get("astrometry_4d_total_s", 0.0) or 0.0),
        "wall_s": float(time.perf_counter() - t0),
        "mono_validation": stats.get("astrometry_4d_selected_mono_validation"),
        "first_accepted_validation": stats.get("astrometry_4d_first_accepted_validation"),
        "best_accepted_validation": stats.get("astrometry_4d_best_accepted_validation"),
        "best_reject": stats.get("astrometry_4d_best_reject"),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.11 - hardening runtime 4D experimental multi-index",
        "",
        "> Le WCS Astrometry.net reste un oracle offline de diagnostic; ce replay appelle `solve_blind` avec une liste explicite d'index 4D et ne lit pas le WCS oracle dans le runtime blind.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Mini-corpus M106 via solveur principal: `{payload['summary']['successes']}/{payload['summary']['cases']}`.",
        f"- Controle mauvaise liste `d50_2822` seule: `{payload['summary']['bad_accepts']}/{payload['summary']['bad_cases']}` acceptation.",
        "- Aucun seuil produit change; backend 4D toujours OFF par defaut; ancien backend conserve.",
        "",
        "## Matrice solveur principal",
        "",
        "| cas | success | inliers | RMS | rang | tuile origine | hits | testes | accepts | stop | total | lookup | validation | mono ref |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["cases"]:
        mono = row.get("mono_validation") or {}
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | `{}` | {} | {} | {} | `{}` | {} | {} | {} | {} / {} |".format(
                row["label"],
                row["success"],
                row["inliers"],
                _fmt(row["rms_px"], 3),
                row["rank"],
                row["origin_tile"],
                row["hits"],
                row["hypotheses_tested"],
                row["accepted_candidates"],
                row["stop_reason"],
                _fmt(row["wall_s"], 3),
                _fmt(row["lookup_s"], 3),
                _fmt(row["validation_s"], 3),
                mono.get("inliers"),
                _fmt(mono.get("rms_px"), 3),
            )
        )
    lines.extend([
        "",
        "## Controle mauvaise liste",
        "",
        "| cas | success | inliers | RMS | hits | testes | stop | message |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in payload["bad_tile_control"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | `{}` | {} |".format(
                row["label"],
                row["success"],
                row["inliers"],
                _fmt(row["rms_px"], 3),
                row["hits"],
                row["hypotheses_tested"],
                row["stop_reason"],
                row["message"],
            )
        )
    lines.extend(["", "## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.11 replay P2.10 through solve_blind with bounded 4D multi-index runtime.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max-stars", type=int, default=120)
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
    args = ap.parse_args()

    labels = [part.strip() for part in str(args.cases).split(",") if part.strip()]
    index_paths = (INDEX_2823.expanduser().resolve(), INDEX_2822.expanduser().resolve())
    for path in index_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing P2.11 explicit index: {path}")

    cases = [_run_case(label, index_paths, args, tag="union_2823_2822") for label in labels]
    bad_rows = [_run_case(label, (INDEX_2822.expanduser().resolve(),), args, tag="bad_2822_only") for label in BAD_TILE_CONTROL]

    successes = [row for row in cases if bool(row["success"])]
    bad_accepts = [row for row in bad_rows if bool(row["success"])]
    verdict = (
        "P2.11 positif: le solveur principal reproduit P2.10 en multi-index 4D experimental"
        if len(successes) == len(cases) and not bad_accepts
        else "P2.11 stop: replay solveur principal incomplet ou controle mauvaise liste accepte"
    )
    answers = [
        f"Mode 4D multi-index depuis solveur principal: `{len(successes)}/{len(cases)}` succes.",
        "Comportement par defaut inchangé: activation toujours conditionnée par `quad_hash_schema=\"astrometry_ab_code_4d_v1\"`; les chemins d'index seuls ne basculent pas le backend.",
        "Fallback silencieux interdit: en schema 4D, liste absente/invalide/union vide renvoie une erreur 4D explicite.",
        "`union_candidate_tiles` est isolee au backend 4D et seulement quand la policy est demandee.",
        "`best_within_budget` est utilise avec `blind_astrometry_4d_max_accepts=64`, `max_hypotheses` borne et budget mur.",
        f"Controle mauvaise liste `d50_2822` seule: `{len(bad_accepts)}/{len(bad_rows)}` acceptation.",
        "Pret comme option experimentale utilisateur, toujours OFF par defaut, sous reserve de garder les controles faux positifs et mini-corpus bornes avant promotion produit.",
    ]
    payload: dict[str, Any] = {
        "schema": "zeblind.p211_4d_experimental_runtime_hardening.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_oracle_policy": "no Astrometry.net WCS oracle is used by solve_blind runtime",
        "global_verdict": verdict,
        "cases": cases,
        "bad_tile_control": bad_rows,
        "summary": {
            "cases": int(len(cases)),
            "successes": int(len(successes)),
            "bad_cases": int(len(bad_rows)),
            "bad_accepts": int(len(bad_accepts)),
        },
        "answers": answers,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "blind_astrometry_4d_index_paths": [str(path) for path in index_paths],
            "blind_astrometry_4d_validation_catalog_policy": "union_candidate_tiles",
            "blind_astrometry_4d_source_policy": "diagnostic_unfiltered",
            "blind_astrometry_4d_accept_policy": "best_within_budget",
            "blind_astrometry_4d_max_accepts": int(args.max_accepts),
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_hypotheses": int(args.max_hypotheses),
            "max_wall_s": float(args.max_wall_s),
            "cases": labels,
            "bad_tile_control": list(BAD_TILE_CONTROL),
            "all30_run": False,
        },
    }
    json_out = args.json_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0 if len(successes) == len(cases) and not bad_accepts else 1


if __name__ == "__main__":
    raise SystemExit(main())
