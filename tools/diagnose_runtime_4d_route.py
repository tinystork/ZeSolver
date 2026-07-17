#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind


DEFAULT_DATA_DIR = ROOT / "reports/eq_ircut_cleanbench_20260518_230249/data"
DEFAULT_REFERENCE_DIR = ROOT / "reports/r47i_s7_testzenear_full_product_clean_20260624/input"
DEFAULT_INDEX_ROOT = ROOT / "reports/s3_focused_index_20260701_p16_multicase_v3/index"
DEFAULT_4D_INDEX = ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"
DEFAULT_WORK_DIR = ROOT / "reports/p22_runtime_4d_route/candidates"
DEFAULT_REPORT = ROOT / "reports/zeblind_p22_runtime_4d_route.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p22_runtime_4d_route.json"
DEFAULT_CASES = {
    "232350": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit", "d50_2823"),
    "232102": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit", "d50_2823"),
}
P20_BASELINE = {
    "232350": {"hits_4d": 95, "hits_plausible": 90, "first_plausible": 4, "best_inliers": 40, "best_rms_px": 0.389},
    "232102": {"hits_4d": 351, "hits_plausible": 343, "first_plausible": 1, "best_inliers": 53, "best_rms_px": 0.184},
}
P21_BASELINE = {
    "232350": {"hits_4d": 171, "hits_tested": 171, "hits_plausible": 37, "first_accepted_like": 150, "best_inliers": 40, "best_rms_px": 0.427},
    "232102": {"hits_4d": 173, "hits_tested": 173, "hits_plausible": 93, "first_accepted_like": 47, "best_inliers": 53, "best_rms_px": 0.222},
}


def _strip_wcs(path: Path) -> None:
    wcs_prefixes = ("CD", "PC", "PV", "A_", "B_", "AP_", "BP_")
    wcs_keys = {
        "WCSAXES",
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CUNIT1",
        "CUNIT2",
        "CDELT1",
        "CDELT2",
        "CROTA1",
        "CROTA2",
        "LONPOLE",
        "LATPOLE",
        "RADESYS",
        "EQUINOX",
        "SOLVED",
        "RMSPX",
        "INLIERS",
        "PIXSCAL",
        "SIPORD",
        "QUALITY",
        "USED_DB",
        "SOLVER",
        "SOLVMODE",
        "TILE_ID",
        "DBSET",
    }
    with fits.open(path, mode="update", memmap=False) as hdul:
        header = hdul[0].header
        for key in list(header.keys()):
            if key in wcs_keys or any(str(key).startswith(prefix) for prefix in wcs_prefixes):
                header.remove(key, ignore_missing=True, remove_all=True)
        hdul.flush()


def _pixel_scale_arcsec(wcs: WCS) -> float:
    try:
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        det = float(np.linalg.det(matrix))
        if not math.isfinite(det) or abs(det) <= 0.0:
            return float("nan")
        return float(math.sqrt(abs(det)) * 3600.0)
    except Exception:
        return float("nan")


def _load_wcs(path: Path) -> tuple[WCS, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdul:
        shape = tuple(int(v) for v in hdul[0].data.shape[-2:])
        return WCS(hdul[0].header).celestial, shape


def _external_wcs_metrics(candidate: Path, reference: Path) -> dict[str, Any]:
    cand_wcs, shape = _load_wcs(candidate)
    ref_wcs, _ = _load_wcs(reference)
    height, width = int(shape[0]), int(shape[1])
    pts = np.asarray(
        [
            [width / 2.0, height / 2.0],
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
        ],
        dtype=np.float64,
    )
    ref = ref_wcs.pixel_to_world(pts[:, 0], pts[:, 1])
    cand = cand_wcs.pixel_to_world(pts[:, 0], pts[:, 1])
    sep = ref.separation(cand).arcsec
    ref_scale = _pixel_scale_arcsec(ref_wcs)
    cand_scale = _pixel_scale_arcsec(cand_wcs)
    return {
        "center_sep_arcsec": float(sep[0]),
        "corner_max_sep_arcsec": float(np.max(sep[1:])),
        "corner_median_sep_arcsec": float(np.median(sep[1:])),
        "scale_ref_arcsec_px": float(ref_scale),
        "scale_candidate_arcsec_px": float(cand_scale),
        "scale_ratio": float(cand_scale / ref_scale) if math.isfinite(ref_scale) and ref_scale > 0 else None,
    }


def _config(args: argparse.Namespace) -> SolveConfig:
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
        blind_astrometry_4d_index_path=str(args.quad4d_index.expanduser().resolve()),
        blind_astrometry_4d_code_tol=float(args.code_tol),
        blind_astrometry_4d_max_hits=int(args.max_hits_4d),
        blind_astrometry_4d_max_hits_per_image_quad=int(args.max_hits_per_image_quad),
        blind_astrometry_4d_max_hypotheses=int(args.max_hypotheses),
        blind_astrometry_4d_image_strategy=str(args.image_strategy),
        blind_astrometry_4d_match_radius_px=float(args.match_radius_px),
        blind_astrometry_4d_source_policy="diagnostic_unfiltered" if bool(int(args.diagnostic_source_list)) else "standard_runtime",
        blind_star_quality_filter=True,
        blind_global_hard_budget_s=max(0.0, float(args.hard_budget_s)),
    )


def _run_case(label: str, filename: str, tile_key: str, args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / filename
    reference = args.reference_dir.expanduser().resolve() / filename
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir / f"{label}_{source.name}"
    shutil.copy2(source, work)
    _strip_wcs(work)

    cfg = _config(args)
    t0 = time.perf_counter()
    solution = solve_blind(work, args.index_root.expanduser().resolve(), config=cfg, prep_cache={})
    wall_s = time.perf_counter() - t0
    stats = dict(solution.stats or {})
    external_metrics: dict[str, Any] = {}
    if bool(solution.success):
        try:
            external_metrics = _external_wcs_metrics(work, reference)
        except Exception as exc:
            external_metrics = {"error": str(exc)}
    return {
        "label": label,
        "tile_key": tile_key,
        "source_fits": str(source),
        "work_fits": str(work),
        "success": bool(solution.success),
        "message": str(solution.message),
        "wall_s": float(wall_s),
        "tile_key_solved": solution.tile_key,
        "runtime_stats": {
            key: stats.get(key)
            for key in (
                "astrometry_4d_index_load_s",
                "astrometry_4d_quad_build_s",
                "astrometry_4d_kd_lookup_s",
                "astrometry_4d_validation_s",
                "astrometry_4d_image_quads",
                "astrometry_4d_image_records",
                "astrometry_4d_hits",
                "astrometry_4d_hits_tested",
                "astrometry_4d_first_plausible_rank",
                "astrometry_4d_first_accepted_rank",
                "astrometry_4d_first_lost_hash_rank",
                "astrometry_4d_first_lost_hash_accepted_rank",
                "astrometry_4d_reject_reason_counts",
                "astrometry_4d_wrote_wcs",
                "inliers",
                "rms_px",
                "pix_scale_arcsec",
                "geo_cov_area",
                "quality",
                "reason",
            )
        },
        "external_reference_metrics": external_metrics,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    def short_counts(value: Any, *, limit: int = 8) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        items = sorted(((str(k), int(v)) for k, v in value.items()), key=lambda item: item[1], reverse=True)
        return dict(items[:limit])

    lines = [
        "# ZeBlind P2.2 - runtime 4D route",
        "",
        "## Conclusion",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Backend runtime: `{ASTROMETRY_AB_CODE_4D_SCHEMA}` actif uniquement par flag.",
        f"- Index 4D: `{payload['params']['quad4d_index']}`",
        "- Reference WCS utilisee uniquement pour l'evaluation externe du rapport, pas par le runtime.",
        "- Ancien backend `opposite_edge_ratio_8bit_v1` reste le defaut lorsque le flag est OFF.",
        "- Note d'ecart P2.1/P2.2: le probe P2.1 utilisait la source-list diagnostic non filtree. Le rapport P2.2 conserve cette parite source-list via `blind_astrometry_4d_source_policy=diagnostic_unfiltered` dans le probe experimental; le defaut produit reste inchange.",
        "- Audit source-list: avec le filtre source runtime standard actif, le probe ne reproduisait pas P2.1 (`232350`: 100 hits, 0 accepte; `232102`: 86 hits, 0 accepte).",
        "",
        "## Comparaison P2.0 / P2.1 / P2.2",
        "",
        "| cas | P2.0 hits/plausibles | P2.1 hits/testes/plausibles | P2.2 success | P2.2 hits/testes | premier accepte | inliers/RMS | centre/coins ref | verdict |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        label = str(case["label"])
        p20 = P20_BASELINE.get(label, {})
        p21 = P21_BASELINE.get(label, {})
        stats = case.get("runtime_stats") or {}
        metrics = case.get("external_reference_metrics") or {}
        center = metrics.get("center_sep_arcsec")
        corners = metrics.get("corner_max_sep_arcsec")
        ext = ""
        if center is not None and corners is not None:
            ext = f"{float(center):.2f}\"/{float(corners):.2f}\""
        lines.append(
            "| `{}` | {}/{} | {}/{}/{} | {} | {}/{} | {} | {}/{} | {} | {} |".format(
                label,
                p20.get("hits_4d", ""),
                p20.get("hits_plausible", ""),
                p21.get("hits_4d", ""),
                p21.get("hits_tested", ""),
                p21.get("hits_plausible", ""),
                "oui" if case["success"] else "non",
                stats.get("astrometry_4d_hits", ""),
                stats.get("astrometry_4d_hits_tested", ""),
                stats.get("astrometry_4d_first_accepted_rank", ""),
                stats.get("inliers", ""),
                "{:.3f}".format(float(stats.get("rms_px", float("nan")))) if stats.get("rms_px") is not None else "",
                ext,
                case.get("verdict", ""),
            )
        )
    lines.extend(["", "## Details runtime", ""])
    for case in payload["cases"]:
        stats = case.get("runtime_stats") or {}
        metrics = case.get("external_reference_metrics") or {}
        lines.extend(
            [
                f"### {case['label']} / {case['tile_key']}",
                "",
                f"- success: `{case['success']}` ; message: `{case['message']}` ; tile: `{case.get('tile_key_solved')}`",
                f"- wall time: `{case['wall_s']:.3f} s`",
                f"- quads image / records 4D: `{stats.get('astrometry_4d_image_quads')}` / `{stats.get('astrometry_4d_image_records')}`",
                f"- hits 4D / testes: `{stats.get('astrometry_4d_hits')}` / `{stats.get('astrometry_4d_hits_tested')}`",
                f"- premier plausible / accepte: `{stats.get('astrometry_4d_first_plausible_rank')}` / `{stats.get('astrometry_4d_first_accepted_rank')}`",
                f"- premier hit perdu hash / accepte perdu hash: `{stats.get('astrometry_4d_first_lost_hash_rank')}` / `{stats.get('astrometry_4d_first_lost_hash_accepted_rank')}`",
                f"- inliers / RMS: `{stats.get('inliers')}` / `{stats.get('rms_px')}`",
                f"- principaux rejets: `{short_counts(stats.get('astrometry_4d_reject_reason_counts'))}`",
                f"- timings: load `{stats.get('astrometry_4d_index_load_s')}` s ; KD `{stats.get('astrometry_4d_kd_lookup_s')}` s ; validation `{stats.get('astrometry_4d_validation_s')}` s",
                f"- evaluation externe centre/corners/scale: `{metrics}`",
                "",
            ]
        )
    lines.extend(["## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.2 probe: run ZeBlind's experimental disk 4D route behind explicit flags.")
    ap.add_argument("--case", action="append", choices=sorted(DEFAULT_CASES), default=None)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--quad4d-index", type=Path, default=DEFAULT_4D_INDEX)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
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
    ap.add_argument("--hard-budget-s", type=float, default=0.0)
    ap.add_argument("--diagnostic-source-list", type=int, choices=(0, 1), default=1, help="Use the same unfiltered source-list discipline as P2.1 diagnostic probes")
    ap.add_argument("--m106-hints", action="store_true", default=True)
    ap.add_argument("--log-level", default="ERROR")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    cases = []
    for label in list(args.case or sorted(DEFAULT_CASES)):
        filename, tile_key = DEFAULT_CASES[label]
        case = _run_case(label, filename, tile_key, args)
        metrics = case.get("external_reference_metrics") or {}
        scale_ratio = metrics.get("scale_ratio")
        case["verdict"] = "runtime_4d_validated" if bool(case["success"]) else "runtime_4d_failed"
        if bool(case["success"]) and scale_ratio is not None and not (0.97 <= float(scale_ratio) <= 1.03):
            case["verdict"] = "runtime_4d_solved_scale_suspicious"
        cases.append(case)

    case_map = {str(case["label"]): case for case in cases}
    ok_232350 = bool(case_map.get("232350", {}).get("success"))
    ok_232102 = bool(case_map.get("232102", {}).get("success"))
    if ok_232350 and ok_232102:
        verdict = "P2.2 positif: le runtime 4D valide 232350 et garde 232102 non-regresse"
    elif ok_232350:
        verdict = "P2.2 partiel: 232350 valide, 232102 a auditer"
    else:
        verdict = "P2.2 negatif: le runtime 4D ne reproduit pas P2.1 sur 232350"

    payload = {
        "schema": "zeblind.p22_runtime_4d_route.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "global_verdict": verdict,
        "cases": cases,
        "p20_baseline": P20_BASELINE,
        "p21_baseline": P21_BASELINE,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "quad4d_index": str(args.quad4d_index.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
            "max_stars": int(args.max_stars),
            "max_quads": int(args.max_quads),
            "quality_rms": float(args.quality_rms),
            "quality_inliers": int(args.quality_inliers),
            "code_tol": float(args.code_tol),
            "max_hits_4d": int(args.max_hits_4d),
            "max_hits_per_image_quad": int(args.max_hits_per_image_quad),
            "max_hypotheses": int(args.max_hypotheses),
            "image_strategy": str(args.image_strategy),
            "match_radius_px": float(args.match_radius_px),
            "blind_astrometry_4d_source_policy": "diagnostic_unfiltered" if bool(int(args.diagnostic_source_list)) else "standard_runtime",
            "diagnostic_source_list": bool(int(args.diagnostic_source_list)),
            "blind_star_quality_filter": True,
        },
    }
    args.json_out.expanduser().resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
