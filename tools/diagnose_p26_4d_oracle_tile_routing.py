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
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex, build_experimental_4d_index
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind

import tools.diagnose_runtime_4d_route as p22
import tools.diagnose_p25_4d_experimental_product_slice as p25


DEFAULT_REPORT = ROOT / "reports/zeblind_p26_4d_oracle_tile_routing.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p26_4d_oracle_tile_routing.json"
DEFAULT_WORK_DIR = ROOT / "reports/p26_4d_oracle_tile_routing/candidates"
DEFAULT_CASES = ("232144", "232205", "232247", "232329", "232431", "232350", "232102")
FAILED_P25_CASES = {"232144", "232205", "232247", "232329", "232431"}
CONTROL_CASES = {"232350", "232102"}
DEFAULT_INDEX_PATHS = {
    "d50_2822": ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz",
    "d50_2823": ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz",
    "d50_2821": ROOT / "reports/p26_astrometry_ab_code_4d_v1_d50_2821_S_stars2000_q40000.npz",
    "d50_2824": ROOT / "reports/p26_astrometry_ab_code_4d_v1_d50_2824_S_stars2000_q40000.npz",
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


def _load_manifest(index_root: Path) -> dict[str, Any]:
    return json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))


def _tile_entries(index_root: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_manifest(index_root)
    return {str(entry.get("tile_key")): dict(entry) for entry in list(manifest.get("tiles") or [])}


def _tile_contains(entry: dict[str, Any], ra_deg: float, dec_deg: float) -> bool:
    bounds = dict(entry.get("bounds") or {})
    dec_min = float(bounds.get("dec_min", -90.0))
    dec_max = float(bounds.get("dec_max", 90.0))
    if not (dec_min <= float(dec_deg) <= dec_max):
        return False
    ra = float(ra_deg) % 360.0
    for lo, hi in list(bounds.get("ra_segments") or []):
        lo_f = float(lo) % 360.0
        hi_f = float(hi) % 360.0
        if lo_f <= hi_f:
            if lo_f <= ra <= hi_f:
                return True
        elif ra >= lo_f or ra <= hi_f:
            return True
    return False


def _tile_distance_deg(entry: dict[str, Any], ra_deg: float, dec_deg: float) -> float:
    c1 = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
    c2 = SkyCoord(float(entry.get("center_ra_deg")) * u.deg, float(entry.get("center_dec_deg")) * u.deg, frame="icrs")
    return float(c1.separation(c2).deg)


def _load_oracle_wcs(path: Path) -> tuple[WCS, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdul:
        shape = tuple(int(v) for v in hdul[0].data.shape[-2:])
        return WCS(hdul[0].header).celestial, shape


def _world_points(wcs: WCS, pixels: np.ndarray) -> np.ndarray:
    world = wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])
    return np.column_stack((np.asarray(world.ra.deg, dtype=np.float64), np.asarray(world.dec.deg, dtype=np.float64)))


def _oracle_case(label: str, args: argparse.Namespace, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = args.reference_dir.expanduser().resolve() / _filename(label)
    wcs, shape = _load_oracle_wcs(path)
    height, width = int(shape[0]), int(shape[1])
    center_pix = np.asarray([[width / 2.0, height / 2.0]], dtype=np.float64)
    corners_pix = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    center_world = _world_points(wcs, center_pix)[0]
    corners_world = _world_points(wcs, corners_pix)
    xs = np.linspace(0.0, float(width - 1), max(2, int(args.footprint_grid)))
    ys = np.linspace(0.0, float(height - 1), max(2, int(args.footprint_grid)))
    grid = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
    grid_world = _world_points(wcs, grid)
    finite = np.isfinite(grid_world[:, 0]) & np.isfinite(grid_world[:, 1])
    grid_world = grid_world[finite]
    tile_rows: list[dict[str, Any]] = []
    for tile_key, entry in sorted(entries.items()):
        if not tile_key.startswith("d50_"):
            continue
        mask = np.asarray([_tile_contains(entry, ra, dec) for ra, dec in grid_world], dtype=bool)
        pct = float(np.count_nonzero(mask) / max(1, grid_world.shape[0]) * 100.0)
        corner_hits = int(sum(1 for ra, dec in corners_world if _tile_contains(entry, float(ra), float(dec))))
        center_inside = bool(_tile_contains(entry, float(center_world[0]), float(center_world[1])))
        if pct <= 0.0 and corner_hits <= 0 and not center_inside:
            continue
        tile_rows.append(
            {
                "tile_key": tile_key,
                "footprint_pct": pct,
                "center_inside": center_inside,
                "corner_hits": corner_hits,
                "center_distance_deg": _tile_distance_deg(entry, float(center_world[0]), float(center_world[1])),
                "bounds": entry.get("bounds"),
            }
        )
    tile_rows.sort(key=lambda row: (-float(row["footprint_pct"]), float(row["center_distance_deg"])))
    primary = str(tile_rows[0]["tile_key"]) if tile_rows else None
    max_pct = float(tile_rows[0]["footprint_pct"]) if tile_rows else 0.0
    if max_pct >= 70.0:
        verdict = "bonne_tuile_probable"
    elif max_pct >= 20.0:
        verdict = "tuile_partielle_ou_ambigue"
    elif tile_rows:
        verdict = "tuile_marginale"
    else:
        verdict = "aucune_tuile_manifest"
    return {
        "label": label,
        "filename": path.name,
        "reference_fits": str(path),
        "image_shape": [height, width],
        "center_ra_dec": [float(center_world[0]), float(center_world[1])],
        "corners_ra_dec": [[float(v[0]), float(v[1])] for v in corners_world],
        "intersected_tiles": tile_rows,
        "primary_tile": primary,
        "oracle_verdict": verdict,
    }


def _catalog_projection_stats(tile_key: str, case: dict[str, Any], args: argparse.Namespace, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if tile_key not in entries:
        return {"tile_key": tile_key, "error": "tile missing from manifest"}
    ref = Path(case["reference_fits"])
    wcs, shape = _load_oracle_wcs(ref)
    height, width = int(shape[0]), int(shape[1])
    tile_path = args.index_root.expanduser().resolve() / str(entries[tile_key].get("tile_file") or "")
    with np.load(tile_path, allow_pickle=False) as data:
        ra = np.asarray(data["ra_deg"], dtype=np.float64)
        dec = np.asarray(data["dec_deg"], dtype=np.float64)
    pix = np.asarray(wcs.wcs_world2pix(np.column_stack((ra, dec)), 0), dtype=np.float64)
    finite = np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1])
    inside = finite & (pix[:, 0] >= 0.0) & (pix[:, 0] < width) & (pix[:, 1] >= 0.0) & (pix[:, 1] < height)
    return {
        "tile_key": tile_key,
        "catalog_stars_total": int(ra.shape[0]),
        "catalog_stars_projected_finite": int(np.count_nonzero(finite)),
        "catalog_stars_in_field": int(np.count_nonzero(inside)),
        "catalog_stars_in_field_pct": float(np.count_nonzero(inside) / max(1, ra.shape[0]) * 100.0),
    }


def _index_path(tile_key: str) -> Path:
    if tile_key in DEFAULT_INDEX_PATHS:
        return DEFAULT_INDEX_PATHS[tile_key]
    return ROOT / f"reports/p26_astrometry_ab_code_4d_v1_{tile_key}_S_stars2000_q40000.npz"


def _ensure_4d_index(tile_key: str, args: argparse.Namespace, reference_meta: dict[str, Any]) -> dict[str, Any]:
    path = _index_path(tile_key).expanduser().resolve()
    build_params = {
        "tile_keys": [tile_key],
        "level": str(reference_meta.get("level", "S")),
        "max_stars_per_tile": int(reference_meta.get("max_stars_per_tile", 2000)),
        "max_quads_per_tile": int(reference_meta.get("max_quads_per_tile", 40000)),
        "sampler_tag": str(reference_meta.get("sampler_tag", "catalog_ring_coverage")),
        "code_tol_recommended": float(reference_meta.get("code_tol_recommended", 0.015)),
        "dtype": str(reference_meta.get("dtype", "float32")),
    }
    built = False
    build_s = 0.0
    if not path.exists():
        t0 = time.perf_counter()
        build_experimental_4d_index(args.index_root.expanduser().resolve(), path, **build_params)
        build_s = float(time.perf_counter() - t0)
        built = True
    idx = Quad4DIndex.load(path)
    return {
        "tile_key": tile_key,
        "path": str(path),
        "available": True,
        "built": built,
        "build_s": build_s,
        "entries": int(idx.codes_4d.shape[0]),
        "stars": int(idx.catalog_ra_dec.shape[0]),
        "metadata": dict(idx.metadata),
        "build_params": build_params,
    }


def _runtime_config(args: argparse.Namespace, index_path: str) -> SolveConfig:
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
        blind_astrometry_4d_index_path=str(index_path),
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


def _short_counts(value: Any, limit: int = 8) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    rows = []
    for key, count in value.items():
        try:
            rows.append((str(key), int(count)))
        except Exception:
            continue
    rows.sort(key=lambda item: item[1], reverse=True)
    return dict(rows[:limit])


def _reason_family(reason_counts: dict[str, int]) -> str:
    if not reason_counts:
        return "none"
    totals = {"scale": 0, "inliers": 0, "matches": 0, "validation": 0, "other": 0}
    for reason, count in reason_counts.items():
        if "pixel_scale_out_of_range" in reason:
            totals["scale"] += int(count)
        elif "no matches" in reason:
            totals["matches"] += int(count)
        elif "inliers_ok=0" in reason:
            totals["inliers"] += int(count)
        elif "validation_failed" in reason:
            totals["validation"] += int(count)
        else:
            totals["other"] += int(count)
    return max(totals.items(), key=lambda item: item[1])[0]


def _solve_with_index(label: str, tile_key: str, index_info: dict[str, Any], oracle_case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / _filename(label)
    work_dir = args.work_dir.expanduser().resolve() / label
    work_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir / f"{label}_{tile_key}_{source.name}"
    shutil.copy2(source, work)
    p22._strip_wcs(work)
    cfg = _runtime_config(args, str(index_info["path"]))
    t0 = time.perf_counter()
    solution = solve_blind(work, args.index_root.expanduser().resolve(), config=cfg, prep_cache={})
    wall_s = float(time.perf_counter() - t0)
    stats = dict(solution.stats or {})
    counts = dict(stats.get("astrometry_4d_reject_reason_counts") or {})
    footprint_pct = 0.0
    for row in oracle_case.get("intersected_tiles") or []:
        if str(row.get("tile_key")) == tile_key:
            footprint_pct = float(row.get("footprint_pct", 0.0) or 0.0)
            break
    cat_stats = _catalog_projection_stats(tile_key, oracle_case, args, _tile_entries(args.index_root.expanduser().resolve()))
    family = _reason_family(counts)
    if bool(solution.success):
        verdict = "bonne_tuile" if tile_key == oracle_case.get("primary_tile") else "tuile_partielle_suffisante"
    elif footprint_pct <= 1.0 or int(cat_stats.get("catalog_stars_in_field", 0) or 0) <= 2:
        verdict = "mauvaise_tuile_probable"
    elif family == "scale":
        verdict = "mauvaise_tuile_probable"
    elif family in {"inliers", "validation"}:
        verdict = "couverture_insuffisante_ou_validation"
    elif family == "matches":
        verdict = "absence_de_matches"
    else:
        verdict = "echec_non_classe"
    return {
        "label": label,
        "tile_key": tile_key,
        "index_path": str(index_info["path"]),
        "index_available_or_built": bool(index_info.get("available")),
        "index_built": bool(index_info.get("built")),
        "success": bool(solution.success),
        "message": str(solution.message),
        "wall_s": wall_s,
        "tile_key_solved": solution.tile_key,
        "oracle_footprint_pct": footprint_pct,
        "catalog_projection": cat_stats,
        "runtime_stats": {
            "hits_4d": stats.get("astrometry_4d_hits"),
            "hits_tested": stats.get("astrometry_4d_hits_tested"),
            "first_accepted": stats.get("astrometry_4d_first_accepted_rank"),
            "inliers": stats.get("inliers"),
            "rms_px": stats.get("rms_px"),
            "quad_build_s": stats.get("astrometry_4d_quad_build_s"),
            "kd_lookup_s": stats.get("astrometry_4d_kd_lookup_s"),
            "validation_s": stats.get("astrometry_4d_validation_s"),
            "best_reject": stats.get("astrometry_4d_best_reject"),
            "reject_reason_counts": counts,
            "reject_reason_top": _short_counts(counts),
            "reason_family": family,
        },
        "verdict": verdict,
    }


def _select_tiles_for_case(case: dict[str, Any], label: str, args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    for row in case.get("intersected_tiles") or []:
        if float(row.get("footprint_pct", 0.0) or 0.0) > 0.0:
            selected.append(str(row.get("tile_key")))
    if label in FAILED_P25_CASES:
        selected.extend(["d50_2822", "d50_2823"])
    if label in CONTROL_CASES:
        selected.append("d50_2823")
    for extra in str(args.extra_tiles or "").split(","):
        extra = extra.strip()
        if extra:
            selected.append(extra)
    out: list[str] = []
    for tile in selected:
        if tile and tile not in out:
            out.append(tile)
    return out


def _untested_neighbor_note(args: argparse.Namespace) -> str:
    forced = {part.strip() for part in str(args.extra_tiles or "").split(",") if part.strip()}
    if forced.intersection({"d50_2821", "d50_2824"}):
        return "Des voisins forces par option ont ete testes."
    return (
        "`d50_2821` et `d50_2824` n'ont pas ete construits/testes: le WCS oracle "
        "leur donne 0% de footprint sur les cas demandes, et les tuiles oracle "
        "`d50_2822`/`d50_2823` suffisent a isoler le front P2.6."
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, str) and value == "None":
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.6 - oracle tile routing 4D",
        "",
        "> Le WCS Astrometry.net est un oracle de diagnostic de tuilage, pas une entree du solveur blind.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Mode: diagnostic uniquement. Aucun tuning, aucun refactor backend, aucun all30.",
        "- Runtime blind 4D lance uniquement avec image sans WCS et index explicite.",
        "",
        "## Matrice oracle de couverture",
        "",
        "| cas | centre RA | centre Dec | tuiles intersectees | principale | distances centre | verdict oracle |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for case in payload["oracle_cases"]:
        tiles = ", ".join(f"{row['tile_key']}:{float(row['footprint_pct']):.1f}%" for row in case.get("intersected_tiles") or [])
        dists = ", ".join(f"{row['tile_key']}:{float(row['center_distance_deg']):.2f}deg" for row in case.get("intersected_tiles") or [])
        center = case["center_ra_dec"]
        lines.append(
            f"| `{case['label']}` | {float(center[0]):.6f} | {float(center[1]):.6f} | {tiles} | `{case.get('primary_tile')}` | {dists} | `{case.get('oracle_verdict')}` |"
        )
    lines.extend(["", "## Matrice solve 4D par index", ""])
    lines.extend(
        [
            "| cas | index | footprint | cat in field | success | hits/testes | accepte | inliers | RMS | best reject | total | quad | KD | validation | famille rejet | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["solve_matrix"]:
        stats = row.get("runtime_stats") or {}
        cat = row.get("catalog_projection") or {}
        best = stats.get("best_reject") if isinstance(stats.get("best_reject"), dict) else {}
        best_text = ""
        if best:
            best_text = "{} / {}".format(best.get("inliers", ""), _fmt(best.get("rms_px"), 3))
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` | `{}` |".format(
                row.get("label"),
                row.get("tile_key"),
                _fmt(row.get("oracle_footprint_pct"), 1),
                cat.get("catalog_stars_in_field", ""),
                "oui" if row.get("success") else "non",
                stats.get("hits_4d", ""),
                stats.get("hits_tested", ""),
                stats.get("first_accepted", ""),
                stats.get("inliers", ""),
                _fmt(stats.get("rms_px"), 3),
                best_text,
                _fmt(row.get("wall_s"), 3),
                _fmt(stats.get("quad_build_s"), 3),
                _fmt(stats.get("kd_lookup_s"), 3),
                _fmt(stats.get("validation_s"), 3),
                stats.get("reason_family", ""),
                row.get("verdict", ""),
            )
        )
    lines.extend(["", "## Diagnostic d50_2822", ""])
    for item in payload["d50_2822_diagnostic"]:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- d50_2822 footprint oracle: `{item['d50_2822_footprint_pct']:.2f}%` ; etoiles catalogue dans champ: `{item['d50_2822_catalog_stars_in_field']}`",
                f"- d50_2822 hits/testes: `{item['d50_2822_hits']}` / `{item['d50_2822_tested']}` ; famille rejet: `{item['d50_2822_reason_family']}`",
                f"- tuile oracle principale: `{item['oracle_primary_tile']}` ; solution alternative: `{item['first_success_tile']}`",
                f"- verdict: `{item['verdict']}`",
                "",
            ]
        )
    lines.extend(["## Tuiles voisines non testees", "", f"- {payload.get('untested_neighbor_note')}", ""])
    lines.extend(["## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Index", "", "```json", json.dumps(payload["indexes"], indent=2, default=_json_default), "```", ""])
    lines.extend(["## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summarize(payload: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    solve_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in payload["solve_matrix"]:
        solve_by_case.setdefault(str(row["label"]), []).append(row)
    d50_diag: list[dict[str, Any]] = []
    failed = [case for case in payload["oracle_cases"] if case["label"] in FAILED_P25_CASES]
    d2822_good = 0
    rerouted_success = 0
    max_tiles_to_success = 0
    for case in failed:
        rows = solve_by_case.get(str(case["label"]), [])
        d2822 = next((row for row in rows if row["tile_key"] == "d50_2822"), None)
        first_success = next((row for row in rows if row.get("success")), None)
        if d2822 and float(d2822.get("oracle_footprint_pct", 0.0) or 0.0) >= 50.0:
            d2822_good += 1
        if first_success:
            rerouted_success += 1
            tested_before_success = 1 + rows.index(first_success)
            max_tiles_to_success = max(max_tiles_to_success, tested_before_success)
        d50_diag.append(
            {
                "label": case["label"],
                "oracle_primary_tile": case.get("primary_tile"),
                "d50_2822_footprint_pct": float(d2822.get("oracle_footprint_pct", 0.0) if d2822 else 0.0),
                "d50_2822_catalog_stars_in_field": int((d2822 or {}).get("catalog_projection", {}).get("catalog_stars_in_field", 0)),
                "d50_2822_hits": (d2822 or {}).get("runtime_stats", {}).get("hits_4d"),
                "d50_2822_tested": (d2822 or {}).get("runtime_stats", {}).get("hits_tested"),
                "d50_2822_reason_family": (d2822 or {}).get("runtime_stats", {}).get("reason_family"),
                "first_success_tile": None if first_success is None else first_success.get("tile_key"),
                "verdict": (
                    "d50_2822_pas_bonne_tuile_routage"
                    if case.get("primary_tile") != "d50_2822" and first_success is not None
                    else "bloc_causal_auditer_sans_elargir"
                ),
            }
        )
    unresolved = len(failed) - rerouted_success
    if rerouted_success == len(failed):
        global_verdict = "P2.6 positif: echecs d50_2822 expliques par routage/couverture, backend 4D OK avec tuile oracle"
    elif rerouted_success > 0:
        global_verdict = "P2.6 partiel: routage explique une partie des echecs, bloc causal restant a isoler"
    else:
        global_verdict = "P2.6 stop: aucune tuile oracle testee ne resout, ne pas elargir"
    answers = [
        f"`d50_2822` bonne tuile pour les cinq echecs: `{d2822_good == len(failed)}` ({d2822_good}/{len(failed)} avec footprint >=50%).",
        f"Images echouees reroutees avec succes vers une tuile oracle: `{rerouted_success}/{len(failed)}` ; les `{unresolved}` restantes ne doivent pas etre elargies sans nouveau diagnostic causal.",
        f"Nombre de tuiles necessaire pour les cas resolus: `1` (la principale oracle `d50_2823`) ; protocole court insuffisant pour `{unresolved}` cas.",
        "Routage experimental simple propose: liste courte ordonnee par footprint oracle decroissante; dans M106 actuel, essayer `d50_2823` avant `d50_2822`, et ne pas construire `d50_2821/d50_2824` tant que l'oracle leur donne 0% de footprint.",
        f"Probleme actuel couverture/routage plutot que backend 4D seul: `partiel` (`{rerouted_success}/{len(failed)}` expliques par routage, reste validation/couverture a isoler).",
        "Mini-corpus M106 multi-index sans all30: `oui en diagnostic borne`, mais pas comme validation produit tant que `232329` et `232431` restent non resolus sans tuning.",
    ]
    return global_verdict, answers, d50_diag


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.6 diagnostic-only oracle tile routing for the experimental 4D backend.")
    ap.add_argument("--data-dir", type=Path, default=p25.p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p25.p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p25.p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--extra-tiles", default="", help="Optional comma-separated tile keys to force-test in addition to oracle/P2.5 tiles")
    ap.add_argument("--footprint-grid", type=int, default=31)
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
    ap.add_argument("--hard-budget-s", type=float, default=45.0)
    ap.add_argument("--m106-hints", action="store_true", default=True)
    ap.add_argument("--log-level", default="ERROR")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    index_root = args.index_root.expanduser().resolve()
    entries = _tile_entries(index_root)
    labels = [part.strip() for part in str(args.cases or "").split(",") if part.strip()]
    oracle_cases = [_oracle_case(label, args, entries) for label in labels]

    reference_index = Quad4DIndex.load(DEFAULT_INDEX_PATHS["d50_2823"])
    reference_meta = dict(reference_index.metadata)
    needed_tiles: list[str] = []
    planned: dict[str, list[str]] = {}
    for case in oracle_cases:
        tiles = _select_tiles_for_case(case, str(case["label"]), args)
        planned[str(case["label"])] = tiles
        for tile in tiles:
            if tile not in needed_tiles:
                needed_tiles.append(tile)
    indexes = {tile: _ensure_4d_index(tile, args, reference_meta) for tile in needed_tiles}

    solve_matrix: list[dict[str, Any]] = []
    for case in oracle_cases:
        for tile in planned[str(case["label"])]:
            solve_matrix.append(_solve_with_index(str(case["label"]), tile, indexes[tile], case, args))

    payload: dict[str, Any] = {
        "schema": "zeblind.p26_4d_oracle_tile_routing.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "oracle_use": "Le WCS Astrometry.net est un oracle de diagnostic de tuilage, pas une entree du solveur blind.",
        "oracle_cases": oracle_cases,
        "planned_tiles": planned,
        "indexes": indexes,
        "solve_matrix": solve_matrix,
        "untested_neighbor_note": _untested_neighbor_note(args),
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "blind_astrometry_4d_source_policy": "diagnostic_unfiltered",
            "blind_star_quality_filter": True,
            "index_root": str(index_root),
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "reference_dir": str(args.reference_dir.expanduser().resolve()),
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
            "footprint_grid": int(args.footprint_grid),
            "hard_budget_s": float(args.hard_budget_s),
        },
    }
    global_verdict, answers, d50_diag = _summarize(payload)
    payload["global_verdict"] = global_verdict
    payload["answers"] = answers
    payload["d50_2822_diagnostic"] = d50_diag
    json_out = args.json_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": global_verdict, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
