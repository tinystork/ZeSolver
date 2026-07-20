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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex, build_experimental_4d_index
from zeblindsolver.zeblindsolver import SolveConfig, _astrometry_4d_runtime_requested, solve_blind

import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p25_4d_experimental_product_slice.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p25_4d_experimental_product_slice.json"
DEFAULT_WORK_DIR = ROOT / "reports/p25_4d_experimental_product_slice/candidates"
DEFAULT_4D_INDEX_2823 = ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"
DEFAULT_4D_INDEX_2822 = ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz"
MANDATORY_2823 = {
    "232350": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit", "d50_2823"),
    "232102": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit", "d50_2823"),
}
NEIGHBOR_CANDIDATES = (
    "232144",
    "232205",
    "232247",
    "232329",
    "232431",
    "232513",
    "232534",
)


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_or_build_2822(args: argparse.Namespace, reference_index: Quad4DIndex) -> tuple[Quad4DIndex, dict[str, Any]]:
    out = args.quad4d_index_2822.expanduser().resolve()
    meta = dict(reference_index.metadata or {})
    build_params = {
        "tile_keys": ["d50_2822"],
        "level": str(meta.get("level", "S")),
        "max_stars_per_tile": int(meta.get("max_stars_per_tile", args.max_catalog_stars)),
        "max_quads_per_tile": int(meta.get("max_quads_per_tile", args.max_catalog_quads)),
        "sampler_tag": str(meta.get("sampler_tag", "catalog_ring_coverage")),
        "code_tol_recommended": float(meta.get("code_tol_recommended", args.code_tol)),
        "dtype": str(meta.get("dtype", "float32")),
    }
    build_meta: dict[str, Any] = {
        "path": str(out),
        "rebuilt": False,
        "params": build_params,
    }
    if out.exists() and not bool(args.rebuild_2822):
        return Quad4DIndex.load(out), build_meta

    t0 = time.perf_counter()
    build_experimental_4d_index(
        args.index_root.expanduser().resolve(),
        out,
        **build_params,
    )
    build_meta["rebuilt"] = True
    build_meta["wall_s"] = float(time.perf_counter() - t0)
    return Quad4DIndex.load(out), build_meta


def _runtime_config(args: argparse.Namespace, index_path: Path) -> SolveConfig:
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
        blind_astrometry_4d_index_path=str(index_path.expanduser().resolve()),
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


def _flag_off_check(args: argparse.Namespace) -> dict[str, Any]:
    cfg = SolveConfig(
        blind_astrometry_4d_index_enabled=True,
        blind_astrometry_4d_index_path=str(args.quad4d_index_2823.expanduser().resolve()),
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
    )
    default_cfg = SolveConfig()
    return {
        "default_schema": default_cfg.quad_hash_schema,
        "default_source_policy": default_cfg.blind_astrometry_4d_source_policy,
        "default_4d_requested": bool(_astrometry_4d_runtime_requested(default_cfg)),
        "flag_only_schema": cfg.quad_hash_schema,
        "flag_only_4d_requested": bool(_astrometry_4d_runtime_requested(cfg)),
        "legacy_backend": "opposite_edge_ratio_8bit_v1",
        "experimental_backend": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "source_policy_isolated": (not _astrometry_4d_runtime_requested(cfg)),
    }


def _run_runtime_case(
    label: str,
    filename: str,
    tile_key: str,
    index_path: Path,
    args: argparse.Namespace,
    *,
    group: str,
) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / filename
    reference = args.reference_dir.expanduser().resolve() / filename
    work_dir = args.work_dir.expanduser().resolve() / group
    work_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir / f"{label}_{tile_key}_{source.name}"
    if not source.exists():
        return {
            "label": label,
            "tile_key": tile_key,
            "group": group,
            "source_fits": str(source),
            "success": False,
            "message": "source FITS missing",
            "runtime_stats": {},
            "external_reference_metrics": {},
        }
    shutil.copy2(source, work)
    p22._strip_wcs(work)
    cfg = _runtime_config(args, index_path)
    t0 = time.perf_counter()
    solution = solve_blind(work, args.index_root.expanduser().resolve(), config=cfg, prep_cache={})
    wall_s = time.perf_counter() - t0
    stats = dict(solution.stats or {})
    external_metrics: dict[str, Any] = {}
    if bool(solution.success) and reference.exists():
        try:
            external_metrics = p22._external_wcs_metrics(work, reference)
        except Exception as exc:
            external_metrics = {"error": str(exc)}
    keep = (
        "quad_hash_schema",
        "astrometry_4d_runtime_enabled",
        "astrometry_4d_index_enabled",
        "blind_astrometry_4d_source_policy",
        "blind_star_quality_filter",
        "astrometry_4d_index_path",
        "astrometry_4d_index_entries",
        "astrometry_4d_index_star_count",
        "astrometry_4d_image_quads",
        "astrometry_4d_image_records",
        "astrometry_4d_quad_build_s",
        "astrometry_4d_kd_lookup_s",
        "astrometry_4d_validation_s",
        "astrometry_4d_hits",
        "astrometry_4d_hits_tested",
        "astrometry_4d_first_plausible_rank",
        "astrometry_4d_first_accepted_rank",
        "astrometry_4d_reject_reason_counts",
        "astrometry_4d_wrote_wcs",
        "inliers",
        "rms_px",
        "pix_scale_arcsec",
        "geo_cov_area",
        "quality",
        "reason",
    )
    return {
        "label": label,
        "tile_key": tile_key,
        "group": group,
        "source_fits": str(source),
        "work_fits": str(work),
        "success": bool(solution.success),
        "message": str(solution.message),
        "wall_s": float(wall_s),
        "tile_key_solved": solution.tile_key,
        "runtime_stats": {key: stats.get(key) for key in keep},
        "external_reference_metrics": external_metrics,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    def _fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    def _short_counts(value: Any, limit: int = 10) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        items = []
        for key, count in value.items():
            try:
                items.append((str(key), int(count)))
            except Exception:
                continue
        items.sort(key=lambda item: item[1], reverse=True)
        return dict(items[:limit])

    lines = [
        "# ZeBlind P2.5 - 4D experimental product slice",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Backend historique: `opposite_edge_ratio_8bit_v1`.",
        f"- Backend experimental: `{ASTROMETRY_AB_CODE_4D_SCHEMA}`.",
        "- Defaut produit inchange: schema historique + source policy `standard_runtime`.",
        "- Baseline experimentale 4D: source policy `diagnostic_unfiltered` uniquement avec le schema 4D.",
        "- Aucun all30, aucun rebuild complet, aucun changement de seuil, aucune promotion de `astrometry_like_candidate`.",
        "",
        "## Checks d'isolation",
        "",
        "```json",
        json.dumps(payload["flag_off_check"], indent=2),
        "```",
        "",
        "## Index 4D",
        "",
        f"- d50_2823: `{payload['indexes']['d50_2823']['path']}`",
        f"- d50_2822: `{payload['indexes']['d50_2822']['path']}`",
        f"- d50_2822 rebuild: `{payload['indexes']['d50_2822']['build']['rebuilt']}`",
        f"- d50_2822 entries/stars: `{payload['indexes']['d50_2822']['entries']}` / `{payload['indexes']['d50_2822']['stars']}`",
        "",
        "## Validation runtime",
        "",
        "| groupe | cas | tuile index | success | hits/testes | accepte | inliers | RMS | total | quad | KD | validation |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        stats = case.get("runtime_stats") or {}
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} |".format(
                case.get("group"),
                case.get("label"),
                case.get("tile_key"),
                "oui" if case.get("success") else "non",
                stats.get("astrometry_4d_hits", ""),
                stats.get("astrometry_4d_hits_tested", ""),
                stats.get("astrometry_4d_first_accepted_rank", ""),
                stats.get("inliers", ""),
                _fmt(stats.get("rms_px"), 3),
                _fmt(case.get("wall_s"), 3),
                _fmt(stats.get("astrometry_4d_quad_build_s"), 3),
                _fmt(stats.get("astrometry_4d_kd_lookup_s"), 3),
                _fmt(stats.get("astrometry_4d_validation_s"), 3),
            )
        )
    lines.extend(["", "## Reponses", ""])
    for item in payload["answers"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Cas detailles", ""])
    for case in payload["cases"]:
        stats = case.get("runtime_stats") or {}
        lines.extend(
            [
                f"### {case.get('group')} / {case.get('label')} / {case.get('tile_key')}",
                "",
                f"- message: `{case.get('message')}`",
                f"- tile solved: `{case.get('tile_key_solved')}`",
                f"- principaux rejets: `{_short_counts(stats.get('astrometry_4d_reject_reason_counts'))}`",
                f"- reference externe: `{case.get('external_reference_metrics')}`",
                "",
            ]
        )
    lines.extend(["## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _answers(payload: dict[str, Any]) -> list[str]:
    cases = payload["cases"]
    mandatory = [case for case in cases if case.get("group") == "d50_2823_mandatory"]
    d2823_ok = all(bool(case.get("success")) for case in mandatory)
    d2822 = [case for case in cases if case.get("group") == "d50_2822_neighbors"]
    d2822_success = [case for case in d2822 if bool(case.get("success"))]
    flag = payload["flag_off_check"]
    return [
        f"Mode 4D utilisable via interface claire: `{d2823_ok}` (schema 4D + index enabled + index path + source policy explicite).",
        f"Backend historique inchange flag OFF: `{bool(flag.get('default_4d_requested') is False and flag.get('flag_only_4d_requested') is False)}`.",
        f"`diagnostic_unfiltered` limite au backend 4D: `{bool(flag.get('source_policy_isolated'))}`.",
        f"`d50_2823` non-regresse sur les cas obligatoires: `{d2823_ok}`.",
        f"`d50_2822` couverture utile dans ce mini-perimetre: `{bool(d2822_success)}` ({len(d2822_success)}/{len(d2822)} succes).",
        "Mini-corpus plus large possible sans all30: `oui` si d50_2823 reste OK; garder d50_2822 comme extension ciblee et separer les echecs de couverture des echecs source-list.",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.5 product-slice probe for the experimental ZeBlind 4D backend.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--quad4d-index-2823", type=Path, default=DEFAULT_4D_INDEX_2823)
    ap.add_argument("--quad4d-index-2822", type=Path, default=DEFAULT_4D_INDEX_2822)
    ap.add_argument("--rebuild-2822", action="store_true")
    ap.add_argument("--neighbor-cases", default=",".join(NEIGHBOR_CANDIDATES[:5]))
    ap.add_argument("--neighbor-limit", type=int, default=5)
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
    ap.add_argument("--max-catalog-stars", type=int, default=2000)
    ap.add_argument("--max-catalog-quads", type=int, default=40000)
    ap.add_argument("--hard-budget-s", type=float, default=45.0)
    ap.add_argument("--m106-hints", action="store_true", default=True)
    ap.add_argument("--log-level", default="ERROR")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    index_2823 = Quad4DIndex.load(args.quad4d_index_2823.expanduser().resolve())
    index_2822, build_2822 = _load_or_build_2822(args, index_2823)

    cases: list[dict[str, Any]] = []
    for label, (filename, tile_key) in MANDATORY_2823.items():
        cases.append(
            _run_runtime_case(
                label,
                filename,
                tile_key,
                args.quad4d_index_2823,
                args,
                group="d50_2823_mandatory",
            )
        )

    neighbors = [part.strip() for part in str(args.neighbor_cases or "").split(",") if part.strip()]
    for label in neighbors[: max(0, int(args.neighbor_limit))]:
        cases.append(
            _run_runtime_case(
                label,
                _filename(label),
                "d50_2822",
                args.quad4d_index_2822,
                args,
                group="d50_2822_neighbors",
            )
        )

    payload: dict[str, Any] = {
        "schema": "zeblind.p25_4d_experimental_product_slice.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "flag_off_check": _flag_off_check(args),
        "indexes": {
            "d50_2823": {
                "path": str(args.quad4d_index_2823.expanduser().resolve()),
                "entries": int(index_2823.codes_4d.shape[0]),
                "stars": int(index_2823.catalog_ra_dec.shape[0]),
                "metadata": dict(index_2823.metadata),
            },
            "d50_2822": {
                "path": str(args.quad4d_index_2822.expanduser().resolve()),
                "entries": int(index_2822.codes_4d.shape[0]),
                "stars": int(index_2822.catalog_ra_dec.shape[0]),
                "metadata": dict(index_2822.metadata),
                "build": build_2822,
            },
        },
        "cases": cases,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "legacy_schema": "opposite_edge_ratio_8bit_v1",
            "blind_astrometry_4d_source_policy": "diagnostic_unfiltered",
            "blind_star_quality_filter": True,
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
            "neighbor_cases": neighbors[: max(0, int(args.neighbor_limit))],
            "hard_budget_s": float(args.hard_budget_s),
        },
    }
    payload["answers"] = _answers(payload)
    mandatory_ok = all(bool(case.get("success")) for case in cases if case.get("group") == "d50_2823_mandatory")
    flag_ok = bool(payload["flag_off_check"].get("default_4d_requested") is False and payload["flag_off_check"].get("flag_only_4d_requested") is False)
    if mandatory_ok and flag_ok:
        payload["global_verdict"] = "P2.5 positif: slice produit experimental 4D utilisable, flag OFF isole"
    elif not flag_ok:
        payload["global_verdict"] = "P2.5 stop: isolation flag OFF non conforme"
    else:
        payload["global_verdict"] = "P2.5 partiel: interface isolee mais validation d50_2823 incomplete"

    json_out = args.json_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": payload["global_verdict"], "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0 if mandatory_ok and flag_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
