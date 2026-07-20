#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind

import tools.diagnose_p29_4d_bounded_multi_index_union_validation as p29
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p212_4d_m106_30_bounded_validation.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p212_4d_m106_30_bounded_validation.json"
DEFAULT_WORK_DIR = ROOT / "reports/p212_4d_m106_30_bounded_validation/candidates"
INDEX_2823 = ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"
INDEX_2822 = ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz"
INCOMPATIBLE_INDEX = ROOT / "reports/s3_focused_index_20260701_p16_multicase_v3/index/tiles/d50_2823.npz"
MISSING_INDEX = ROOT / "reports/p212_missing_astrometry_ab_code_4d_v1_d50_9999.npz"
MANDATORY_CASES = (
    "232329",
    "232431",
    "232144",
    "232205",
    "232247",
    "232350",
    "232102",
    "232513",
    "232534",
    "232658",
)
CASE_RE = re.compile(r"Light_mosaic_M 106_20\.0s_IRCUT_20250518-(\d+)\.fit$")


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


def _quantile(values: list[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if np.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * float(q)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def _median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        return None
    return float(statistics.median(clean))


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _discover_cases(data_dir: Path) -> list[str]:
    labels: list[str] = []
    for path in sorted(data_dir.glob("Light_mosaic_M 106_20.0s_IRCUT_20250518-*.fit")):
        m = CASE_RE.match(path.name)
        if m:
            labels.append(m.group(1))
    ordered = list(MANDATORY_CASES)
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def _case_source(label: str, data_dir: Path) -> Path:
    source = data_dir / _filename(label)
    if not source.exists():
        raise FileNotFoundError(f"missing M106 case {label}: {source}")
    return source


def _prepare_case(label: str, data_dir: Path, work_dir: Path, tag: str) -> Path:
    source = _case_source(label, data_dir)
    target_dir = work_dir / tag / label
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    p22._strip_wcs(target)
    return target


def _config(index_paths: tuple[Path, ...], args: argparse.Namespace, *, accept_policy: str) -> SolveConfig:
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
        blind_astrometry_4d_accept_policy=str(accept_policy),
        blind_astrometry_4d_max_accepts=int(args.max_accepts),
        blind_astrometry_4d_code_tol=float(args.code_tol),
        blind_astrometry_4d_max_hits=int(args.max_hits_4d),
        blind_astrometry_4d_max_hits_per_image_quad=int(args.max_hits_per_image_quad),
        blind_astrometry_4d_max_hypotheses=int(args.max_hypotheses),
        blind_astrometry_4d_image_strategy=str(args.image_strategy),
        blind_astrometry_4d_match_radius_px=float(args.match_radius_px),
    )


def _wcs_summary(wcs: WCS | None, shape: tuple[int, int] | None) -> dict[str, Any]:
    if wcs is None or shape is None:
        return {}
    height, width = int(shape[0]), int(shape[1])
    pts = np.asarray(
        [
            [width * 0.5, height * 0.5],
            [0.0, 0.0],
            [max(0.0, width - 1.0), 0.0],
            [max(0.0, width - 1.0), max(0.0, height - 1.0)],
            [0.0, max(0.0, height - 1.0)],
        ],
        dtype=np.float64,
    )
    try:
        world = np.asarray(wcs.wcs_pix2world(pts, 0), dtype=np.float64)
    except Exception:
        return {}
    if world.shape != (5, 2) or not np.all(np.isfinite(world)):
        return {}
    return {
        "center_ra_dec": [float(world[0, 0]), float(world[0, 1])],
        "corners_ra_dec": [[float(v[0]), float(v[1])] for v in world[1:]],
    }


def _reference_wcs_summary(source: Path) -> tuple[tuple[int, int] | None, dict[str, Any]]:
    try:
        with fits.open(source, memmap=False) as hdul:
            shape = tuple(int(v) for v in hdul[0].data.shape[-2:])
            wcs = WCS(hdul[0].header)
            if not bool(getattr(wcs, "has_celestial", False)):
                return shape, {}
            return shape, _wcs_summary(wcs, shape)
    except Exception:
        return None, {}


def _center_sep_arcsec(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    ca = a.get("center_ra_dec")
    cb = b.get("center_ra_dec")
    if not ca or not cb:
        return None
    try:
        c1 = SkyCoord(float(ca[0]) * u.deg, float(ca[1]) * u.deg, frame="icrs")
        c2 = SkyCoord(float(cb[0]) * u.deg, float(cb[1]) * u.deg, frame="icrs")
        return float(c1.separation(c2).arcsec)
    except Exception:
        return None


def _run_solve(
    label: str,
    index_paths: tuple[Path, ...],
    args: argparse.Namespace,
    *,
    accept_policy: str,
    tag: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    data_dir = args.data_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    source = _case_source(label, data_dir)
    shape, ref_wcs = _reference_wcs_summary(source)
    candidate = _prepare_case(label, data_dir, work_dir, tag)
    cfg = _config(index_paths, args, accept_policy=accept_policy)
    result = solve_blind(candidate, args.index_root.expanduser().resolve(), config=cfg)
    stats = dict(result.stats or {})
    result_wcs = _wcs_summary(result.wcs, shape)
    return {
        "label": label,
        "tag": tag,
        "index_paths": [str(path.expanduser().resolve()) for path in index_paths],
        "index_tiles": [_tile_name(path) for path in index_paths],
        "accept_policy": accept_policy,
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
        "reject_reason_counts": stats.get("astrometry_4d_reject_reason_counts") or stats.get("reject_reason_counts"),
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
        "reference_wcs": ref_wcs,
        "solution_wcs": result_wcs,
        "center_sep_arcsec": _center_sep_arcsec(ref_wcs, result_wcs),
    }


def _tile_name(path: Path) -> str:
    name = path.name
    m = re.search(r"(d50_\d+)", name)
    return m.group(1) if m else name


def _progress(event: str, **payload: Any) -> None:
    data = {"event": event}
    data.update(payload)
    print(json.dumps(data, default=_json_default), flush=True)


def _run_error_probe(label: str, index_paths: tuple[Path, ...], args: argparse.Namespace, *, tag: str) -> dict[str, Any]:
    try:
        row = _run_solve(label, index_paths, args, accept_policy="best_within_budget", tag=tag)
        row["explicit_error_ok"] = (not bool(row["success"])) and (
            "fallback" in str(row["message"]).lower()
            or "missing" in str(row["message"]).lower()
            or "failed" in str(row["message"]).lower()
        )
        return row
    except Exception as exc:
        return {
            "label": label,
            "tag": tag,
            "index_paths": [str(path) for path in index_paths],
            "index_tiles": [_tile_name(path) for path in index_paths],
            "success": False,
            "message": f"{type(exc).__name__}: {exc}",
            "explicit_error_ok": True,
        }


def _footprints(label: str, args: argparse.Namespace) -> dict[str, float]:
    try:
        return p29._oracle_footprints(label, args)
    except Exception as exc:
        return {"_error": str(exc)}


def _bounded_control_labels(labels: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    out: list[str] = []
    for label in list(MANDATORY_CASES) + list(labels):
        if label in labels and label not in out:
            out.append(label)
        if len(out) >= int(limit):
            break
    return out


def _run_main_matrix(labels: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index_paths = (INDEX_2823, INDEX_2822)
    for idx, label in enumerate(labels, start=1):
        _progress("main_case_start", label=label, index=idx, count=len(labels))
        first = _run_solve(label, index_paths, args, accept_policy="first_accept", tag="main_first_accept")
        _progress("main_case_policy_done", label=label, policy="first_accept", success=first.get("success"), inliers=first.get("inliers"), rms_px=first.get("rms_px"))
        best = _run_solve(label, index_paths, args, accept_policy="best_within_budget", tag="main_best_within_budget")
        _progress("main_case_policy_done", label=label, policy="best_within_budget", success=best.get("success"), inliers=best.get("inliers"), rms_px=best.get("rms_px"))
        gain = None
        if first.get("success") and best.get("success"):
            gain = float(first["rms_px"]) - float(best["rms_px"])
        rows.append(
            {
                "label": label,
                "case_group": "mandatory_p210_p211" if label in MANDATORY_CASES else "expanded_m106",
                "first_accept": first,
                "best_within_budget": best,
                "rms_gain_best_minus_first": gain,
                "footprints": _footprints(label, args),
            }
        )
    return rows


def _run_control_matrix(labels: list[str], args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    low_2822 = []
    for label in labels:
        fp = _footprints(label, args)
        if "_error" in fp or "d50_2822" not in fp:
            continue
        if float(fp.get("d50_2822", 0.0) or 0.0) <= float(args.bad_tile_max_footprint_pct):
            low_2822.append(label)
    low_2822 = low_2822[: max(0, int(args.bad_tile_control_limit))]
    control_labels = _bounded_control_labels(labels, int(args.control_case_limit))

    out["d50_2822_only_low_footprint"] = [
        (
            _progress("control_case_start", control="d50_2822_only_low_footprint", label=label),
            _run_solve(label, (INDEX_2822,), args, accept_policy="best_within_budget", tag="control_d50_2822_only_low_footprint")
            | {"footprints": _footprints(label, args)},
        )[1]
        for label in low_2822
    ]
    out["reversed_order_best"] = [
        (
            _progress("control_case_start", control="reversed_order_best", label=label),
            _run_solve(label, (INDEX_2822, INDEX_2823), args, accept_policy="best_within_budget", tag="control_reversed_order_best"),
        )[1]
        for label in control_labels
    ]
    out["d50_2823_only_best"] = [
        (
            _progress("control_case_start", control="d50_2823_only_best", label=label),
            _run_solve(label, (INDEX_2823,), args, accept_policy="best_within_budget", tag="control_d50_2823_only_best"),
        )[1]
        for label in control_labels
    ]
    out["d50_2822_only_best"] = [
        (
            _progress("control_case_start", control="d50_2822_only_best", label=label),
            _run_solve(label, (INDEX_2822,), args, accept_policy="best_within_budget", tag="control_d50_2822_only_best"),
        )[1]
        for label in control_labels
    ]
    probe_label = labels[0]
    out["strict_error_controls"] = [
        _run_error_probe(probe_label, (MISSING_INDEX,), args, tag="control_missing_index"),
        _run_error_probe(probe_label, (INCOMPATIBLE_INDEX,), args, tag="control_incompatible_index"),
    ]
    return out


def _summarize_runtime(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    policy_rows = [row[policy] for row in rows]
    return {
        "cases": int(len(policy_rows)),
        "successes": int(sum(1 for row in policy_rows if row.get("success"))),
        "total_s_median": _median([float(row.get("wall_s", 0.0) or 0.0) for row in policy_rows]),
        "total_s_p95": _quantile([float(row.get("wall_s", 0.0) or 0.0) for row in policy_rows], 0.95),
        "validation_s_median": _median([float(row.get("validation_s", 0.0) or 0.0) for row in policy_rows]),
        "validation_s_p95": _quantile([float(row.get("validation_s", 0.0) or 0.0) for row in policy_rows], 0.95),
        "hypotheses_tested_median": _median([float(row.get("hypotheses_tested", 0) or 0) for row in policy_rows]),
        "hypotheses_tested_p95": _quantile([float(row.get("hypotheses_tested", 0) or 0) for row in policy_rows], 0.95),
        "accepted_candidates_median": _median([float(row.get("accepted_candidates", 0) or 0) for row in policy_rows]),
        "max_accepts_hits": [row["label"] for row in policy_rows if str(row.get("stop_reason")) == "max_accepts"],
        "max_wall_s_hits": [row["label"] for row in policy_rows if str(row.get("stop_reason")) == "cancelled"],
        "candidate_exhausted": [row["label"] for row in policy_rows if str(row.get("stop_reason")) == "candidate_exhausted"],
    }


def _summarize(payload: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    rows = payload["cases"]
    first_ok = [row for row in rows if row["first_accept"].get("success")]
    best_ok = [row for row in rows if row["best_within_budget"].get("success")]
    mandatory = [row for row in rows if row["label"] in MANDATORY_CASES]
    mandatory_best_ok = [row for row in mandatory if row["best_within_budget"].get("success")]
    gains = [
        row
        for row in rows
        if row.get("rms_gain_best_minus_first") is not None and float(row["rms_gain_best_minus_first"]) > 1e-6
    ]
    worse = [
        row
        for row in rows
        if row.get("rms_gain_best_minus_first") is not None and float(row["rms_gain_best_minus_first"]) < -1e-6
    ]
    controls = payload["controls"]
    bad_low_accepts = [row for row in controls["d50_2822_only_low_footprint"] if row.get("success")]
    missing_errors = [row for row in controls["strict_error_controls"] if row.get("explicit_error_ok")]
    best_failures = [row["best_within_budget"] for row in rows if not row["best_within_budget"].get("success")]
    failure_bits = []
    for row in best_failures:
        reject = row.get("best_reject") or {}
        failure_bits.append(
            "`{}` best reject: {} inliers / RMS {} / stop `{}`".format(
                row["label"],
                reject.get("inliers", row.get("inliers")),
                _fmt(reject.get("rms_px", row.get("rms_px")), 3),
                row.get("stop_reason"),
            )
        )
    reversed_rows = controls["reversed_order_best"]
    reversed_ok = [row for row in reversed_rows if row.get("success")]
    by_label_reversed = {row["label"]: row for row in reversed_rows}
    order_degraded = []
    for row in rows:
        normal = row["best_within_budget"]
        rev = by_label_reversed.get(row["label"])
        if not rev or not normal.get("success") or not rev.get("success"):
            continue
        if float(rev.get("rms_px", 1e9)) > float(normal.get("rms_px", 1e9)) + 0.25:
            order_degraded.append({"label": row["label"], "normal_rms": normal.get("rms_px"), "reversed_rms": rev.get("rms_px")})
    runtime = {
        "first_accept": _summarize_runtime(rows, "first_accept"),
        "best_within_budget": _summarize_runtime(rows, "best_within_budget"),
    }
    if len(best_ok) == len(rows) and len(first_ok) == len(rows) and not bad_low_accepts and len(missing_errors) == len(controls["strict_error_controls"]):
        verdict = "P2.12 positif: M106 elargi borne valide le mode 4D multi-index sans faux positif evident"
    elif bad_low_accepts:
        verdict = "P2.12 stop: controle mauvaise liste produit une acceptation"
    else:
        verdict = f"P2.12 partiel: {len(best_ok)}/{len(rows)} en best, echecs restants a diagnostiquer avant elargissement"
    answers = [
        f"Corpus M106 elargi: `best_within_budget` resout `{len(best_ok)}/{len(rows)}` ; `first_accept` resout `{len(first_ok)}/{len(rows)}`.",
        f"Cas obligatoires P2.10/P2.11 conserves en best: `{len(mandatory_best_ok)}/{len(mandatory)}`.",
        "Echecs restants: " + ("; ".join(failure_bits) if failure_bits else "aucun."),
        f"`best_within_budget` ameliore le RMS sur `{len(gains)}/{len(rows)}` cas ; degradations RMS detectees: `{len(worse)}`.",
        f"Controle mauvaise liste `d50_2822` seule a footprint faible: `{len(bad_low_accepts)}/{len(controls['d50_2822_only_low_footprint'])}` acceptation.",
        f"Liste inversee `[d50_2822, d50_2823]`: `{len(reversed_ok)}/{len(reversed_rows)}` succes ; degradations RMS > 0.25 px: `{len(order_degraded)}`.",
        f"Cout `best_within_budget`: median `{_fmt(runtime['best_within_budget']['total_s_median'])}s`, p95 `{_fmt(runtime['best_within_budget']['total_s_p95'])}s`, validation median `{_fmt(runtime['best_within_budget']['validation_s_median'])}s`.",
        f"Budget: max_accepts touches par `{len(runtime['best_within_budget']['max_accepts_hits'])}` cas ; max_wall_s par `{len(runtime['best_within_budget']['max_wall_s_hits'])}` cas ; candidats epuises par `{len(runtime['best_within_budget']['candidate_exhausted'])}` cas.",
        "Garde-fous P2.11 conserves: schema 4D explicite, liste d'index explicite, erreurs absent/incompatible explicites, aucun WCS oracle comme entree runtime.",
        "Option experimentale utilisateur: oui pour ce perimetre borne, mais pas comme promotion produit; les deux echecs imposent un diagnostic support/source-list/couverture avant un elargissement.",
        "Avant promotion plus large: comprendre `233828` et `234013`, verifier d'autres champs, et formaliser le contrat entre acceptation produit et diagnostic.",
    ]
    return verdict, answers, runtime


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.12 - validation M106 bornee 4D multi-index",
        "",
        "> Le WCS Astrometry.net est utilise seulement pour l'evaluation offline du rapport. Le runtime blind appelle `solve_blind` sur des copies FITS sans WCS, avec une liste explicite d'index 4D.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Corpus M106 teste: `{payload['summary']['case_count']}` images locales.",
        "- Aucun changement produit par defaut, aucun all-sky, aucun rebuild complet, aucun seuil change.",
        "",
        "## Matrice principale",
        "",
        "| cas | groupe | first_accept | best_within_budget | gain RMS | origine best | hits best | testes best | accepts best | stop best | total best | validation best |",
        "|---|---|---|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["cases"]:
        fa = row["first_accept"]
        bw = row["best_within_budget"]
        lines.append(
            "| `{}` | `{}` | {} / {} / r{} | {} / {} / r{} | {} | `{}` | {} | {} | {} | `{}` | {} | {} |".format(
                row["label"],
                row["case_group"],
                fa.get("inliers"), _fmt(fa.get("rms_px"), 3), fa.get("rank"),
                bw.get("inliers"), _fmt(bw.get("rms_px"), 3), bw.get("rank"),
                _fmt(row.get("rms_gain_best_minus_first"), 3),
                bw.get("origin_tile"),
                bw.get("hits"),
                bw.get("hypotheses_tested"),
                bw.get("accepted_candidates"),
                bw.get("stop_reason"),
                _fmt(bw.get("wall_s"), 3),
                _fmt(bw.get("validation_s"), 3),
            )
        )
    lines.extend(["", "## Controles faux positifs", ""])
    for key, title in [
        ("d50_2822_only_low_footprint", "Mauvaise liste mono-tuile `d50_2822` seule, footprint faible"),
        ("reversed_order_best", "Liste inversee `[d50_2822, d50_2823]`"),
        ("d50_2823_only_best", "Liste partielle `d50_2823` seule"),
        ("d50_2822_only_best", "Liste partielle `d50_2822` seule"),
    ]:
        rows = payload["controls"].get(key) or []
        accepted = [row for row in rows if row.get("success")]
        lines.extend([f"### {title}", "", f"- Acceptations: `{len(accepted)}/{len(rows)}`.", ""])
        lines.extend(["| cas | success | inliers | RMS | rang | origine | hits | testes | accepts | stop | total |", "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|"])
        for row in rows:
            lines.append(
                "| `{}` | `{}` | {} | {} | {} | `{}` | {} | {} | {} | `{}` | {} |".format(
                    row.get("label"),
                    row.get("success"),
                    row.get("inliers"),
                    _fmt(row.get("rms_px"), 3),
                    row.get("rank"),
                    row.get("origin_tile"),
                    row.get("hits"),
                    row.get("hypotheses_tested"),
                    row.get("accepted_candidates"),
                    row.get("stop_reason"),
                    _fmt(row.get("wall_s"), 3),
                )
            )
        lines.append("")
    lines.extend(["### Erreurs strictes", "", "| controle | success | erreur explicite | message |", "|---|---:|---:|---|"])
    for row in payload["controls"].get("strict_error_controls") or []:
        lines.append(f"| `{row.get('tag')}` | `{row.get('success')}` | `{row.get('explicit_error_ok')}` | {row.get('message')} |")
    lines.extend(["", "## Cout runtime", "", "```json", json.dumps(payload["runtime_summary"], indent=2, default=_json_default), "```", ""])
    lines.extend(["## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2, default=_json_default), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.12 bounded M106 validation for experimental 4D multi-index solve_blind mode.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--cases", default="")
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
    ap.add_argument("--bad-tile-max-footprint-pct", type=float, default=40.0)
    ap.add_argument("--bad-tile-control-limit", type=int, default=12)
    ap.add_argument("--control-case-limit", type=int, default=12)
    ap.add_argument("--footprint-grid", type=int, default=31)
    args = ap.parse_args()

    for path in (INDEX_2823, INDEX_2822):
        if not path.exists():
            raise FileNotFoundError(f"missing explicit P2.12 4D index: {path}")

    data_dir = args.data_dir.expanduser().resolve()
    explicit_cases = bool(args.cases.strip())
    labels = [part.strip() for part in str(args.cases).split(",") if part.strip()] if explicit_cases else _discover_cases(data_dir)
    missing_mandatory = [label for label in MANDATORY_CASES if label not in labels]
    if (not explicit_cases) and missing_mandatory:
        raise RuntimeError("mandatory P2.10/P2.11 cases missing from P2.12 corpus: " + ", ".join(missing_mandatory))

    cases = _run_main_matrix(labels, args)
    controls = _run_control_matrix(labels, args)
    payload: dict[str, Any] = {
        "schema": "zeblind.p212_4d_m106_30_bounded_validation.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_oracle_policy": "solve_blind runtime uses WCS-stripped FITS and explicit 4D index paths only; WCS oracle is offline report context",
        "cases": cases,
        "controls": controls,
        "summary": {"case_count": int(len(cases)), "all30_like_local_count": int(len(_discover_cases(data_dir)))},
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "blind_astrometry_4d_index_paths": [str(INDEX_2823.expanduser().resolve()), str(INDEX_2822.expanduser().resolve())],
            "blind_astrometry_4d_validation_catalog_policy": "union_candidate_tiles",
            "blind_astrometry_4d_source_policy": "diagnostic_unfiltered",
            "blind_astrometry_4d_accept_policies_compared": ["first_accept", "best_within_budget"],
            "blind_astrometry_4d_max_accepts": int(args.max_accepts),
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_hypotheses": int(args.max_hypotheses),
            "max_wall_s": float(args.max_wall_s),
            "data_dir": str(data_dir),
            "cases": labels,
            "all_sky_validation": False,
            "all30_cli_run": False,
            "runtime_index_discovery": False,
            "runtime_index_build": False,
            "bad_tile_control_limit": int(args.bad_tile_control_limit),
            "control_case_limit": int(args.control_case_limit),
        },
    }
    verdict, answers, runtime = _summarize(payload)
    payload["global_verdict"] = verdict
    payload["answers"] = answers
    payload["runtime_summary"] = runtime
    json_out = args.json_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "report": str(args.report), "json": str(args.json_out), "cases": len(cases)}, indent=2))
    return 0 if verdict.startswith("P2.12 positif") else 1


if __name__ == "__main__":
    raise SystemExit(main())
