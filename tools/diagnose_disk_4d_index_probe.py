#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.asterisms import sample_quads
from zeblindsolver.image_prep import build_pyramid, read_fits_as_luma, remove_background
from zeblindsolver.quad_code_diagnostic import build_astrometry_quad_records
from zeblindsolver.quad_index_4d import (
    ASTROMETRY_AB_CODE_4D_SCHEMA,
    Quad4DIndex,
    build_experimental_4d_index,
)
from zeblindsolver.star_detect import detect_stars


DEFAULT_DATA_DIR = ROOT / "reports/eq_ircut_cleanbench_20260518_230249/data"
DEFAULT_REFERENCE_DIR = ROOT / "reports/r47i_s7_testzenear_full_product_clean_20260624/input"
DEFAULT_INDEX_ROOT = ROOT / "reports/s3_focused_index_20260701_p16_multicase_v3/index"
DEFAULT_4D_INDEX = ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S.npz"
DEFAULT_REPORT = ROOT / "reports/zeblind_p21_disk_4d_index_probe.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p21_disk_4d_index_probe.json"
DEFAULT_CASES = {
    "232350": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit", "d50_2823"),
    "232102": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit", "d50_2823"),
}
P20_BASELINE = {
    "232350": {
        "hits_4d": 95,
        "hits_plausible": 90,
        "first_plausible": 4,
        "first_hash_exact_plausible": 8,
        "first_lost_hash_plausible": 4,
        "best_inliers": 40,
        "best_rms_px": 0.389,
    },
    "232102": {
        "hits_4d": 351,
        "hits_plausible": 343,
        "first_plausible": 1,
        "first_hash_exact_plausible": 1,
        "first_lost_hash_plausible": 6,
        "best_inliers": 53,
        "best_rms_px": 0.184,
    },
}


def _pixel_scale_arcsec(wcs: WCS) -> float:
    try:
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        det = float(np.linalg.det(matrix))
        if not math.isfinite(det) or det == 0.0:
            return float("nan")
        return float(math.sqrt(abs(det)) * 3600.0)
    except Exception:
        return float("nan")


def _load_reference_wcs(path: Path) -> tuple[WCS, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None or data.ndim < 2:
            raise RuntimeError(f"reference has no 2D image: {path}")
        return WCS(hdul[0].header).celestial, (int(data.shape[-2]), int(data.shape[-1]))


def _detect_image_stars(path: Path, *, max_stars: int, detect_k_sigma: float, detect_min_area: int) -> tuple[np.ndarray, dict[str, Any]]:
    image = read_fits_as_luma(path)
    work = remove_background(image, kernel_size=15)
    detection = build_pyramid(work)[-1]
    stars = detect_stars(
        detection,
        min_fwhm_px=1.5,
        max_fwhm_px=8.0,
        k_sigma=detect_k_sigma,
        min_area=detect_min_area,
        backend="cpu",
    )
    raw_count = int(stars.shape[0])
    if max_stars > 0 and stars.shape[0] > max_stars:
        stars = stars[:max_stars]
    return stars, {
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "raw_sources": raw_count,
        "kept_sources": int(stars.shape[0]),
    }


def _positions(stars: np.ndarray) -> np.ndarray:
    if stars.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack((stars["x"].astype(np.float64), stars["y"].astype(np.float64)))


def _fit_quad_wcs(image_points: np.ndarray, catalog_world: np.ndarray) -> WCS:
    coords = SkyCoord(ra=catalog_world[:, 0] * u.deg, dec=catalog_world[:, 1] * u.deg, frame="icrs")
    return fit_wcs_from_points((image_points[:, 0], image_points[:, 1]), coords, projection="TAN")


def _corner_metrics(candidate: WCS, reference: WCS, image_shape: tuple[int, int]) -> dict[str, Any]:
    height, width = int(image_shape[0]), int(image_shape[1])
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
    ref = reference.pixel_to_world(pts[:, 0], pts[:, 1])
    cand = candidate.pixel_to_world(pts[:, 0], pts[:, 1])
    sep = ref.separation(cand).arcsec
    ref_scale = _pixel_scale_arcsec(reference)
    cand_scale = _pixel_scale_arcsec(candidate)
    return {
        "center_sep_arcsec": float(sep[0]),
        "corner_max_sep_arcsec": float(np.max(sep[1:])),
        "corner_median_sep_arcsec": float(np.median(sep[1:])),
        "scale_ref_arcsec_px": float(ref_scale),
        "scale_candidate_arcsec_px": float(cand_scale),
        "scale_ratio": float(cand_scale / ref_scale) if math.isfinite(ref_scale) and ref_scale > 0 else None,
    }


def _evaluate_wcs(
    *,
    wcs: WCS,
    reference_wcs: WCS,
    image_shape: tuple[int, int],
    image_positions: np.ndarray,
    catalog_world: np.ndarray,
    inlier_radius_px: float,
) -> dict[str, Any]:
    projected = np.asarray(wcs.all_world2pix(catalog_world[:, 0], catalog_world[:, 1], 0), dtype=np.float64).T
    finite = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
    if not np.any(finite):
        inliers = 0
        rms = float("inf")
    else:
        tree = cKDTree(image_positions)
        distances, _idx = tree.query(projected[finite], k=1, distance_upper_bound=float(inlier_radius_px))
        good = np.isfinite(distances) & (distances <= float(inlier_radius_px))
        inliers = int(np.count_nonzero(good))
        rms = float(np.sqrt(np.mean(distances[good] ** 2))) if inliers else float("inf")
    out = {"inliers": inliers, "rms_px": rms}
    out.update(_corner_metrics(wcs, reference_wcs, image_shape))
    return out


def _reject_reason(metrics: dict[str, Any], args: argparse.Namespace) -> str:
    scale_ratio = metrics.get("scale_ratio")
    if int(metrics.get("inliers", 0)) < int(args.min_plausible_inliers):
        return "not_enough_inliers"
    if not math.isfinite(float(metrics.get("rms_px", float("inf")))) or float(metrics["rms_px"]) > float(args.max_plausible_rms_px):
        return "rms_too_high"
    if scale_ratio is None or not math.isfinite(float(scale_ratio)) or not (float(args.min_scale_ratio) <= float(scale_ratio) <= float(args.max_scale_ratio)):
        return "scale_ratio_outside_probe_bounds"
    if float(metrics.get("center_sep_arcsec", float("inf"))) > float(args.max_center_sep_arcsec):
        return "center_sep_too_high"
    if float(metrics.get("corner_max_sep_arcsec", float("inf"))) > float(args.max_corner_sep_arcsec):
        return "corner_sep_too_high"
    return "plausible"


def _build_if_needed(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    out = args.quad4d_index.expanduser().resolve()
    build_meta: dict[str, Any] = {"path": str(out), "rebuilt": False}
    if out.exists() and not bool(args.rebuild):
        return out, build_meta
    t0 = time.perf_counter()
    built = build_experimental_4d_index(
        args.index_root,
        out,
        tile_keys=[str(v) for v in args.tile],
        level=str(args.level),
        max_stars_per_tile=int(args.max_catalog_stars),
        max_quads_per_tile=int(args.max_catalog_quads),
        sampler_tag=str(args.catalog_strategy),
        code_tol_recommended=float(args.code_tol),
        dtype=str(args.code_dtype),
    )
    build_meta.update({"rebuilt": True, "wall_s": time.perf_counter() - t0})
    return built, build_meta


def _run_case(label: str, filename: str, tile_key: str, disk_index: Quad4DIndex, args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / filename
    reference = args.reference_dir.expanduser().resolve() / filename
    reference_wcs, _ref_shape = _load_reference_wcs(reference)
    image_stars, image_meta = _detect_image_stars(
        source,
        max_stars=int(args.max_image_stars),
        detect_k_sigma=float(args.detect_k_sigma),
        detect_min_area=int(args.detect_min_area),
    )
    image_positions = _positions(image_stars)
    image_quads = sample_quads(image_stars, max_quads=int(args.max_image_quads), strategy=str(args.image_strategy))
    image_records = build_astrometry_quad_records(image_quads, image_positions)
    t_lookup0 = time.perf_counter()
    hits = disk_index.search_records(
        image_records,
        code_tol=float(args.code_tol),
        max_hits=int(args.max_hits_4d),
        max_hits_per_image_quad=int(args.max_hits_per_image_quad),
    )
    lookup_s = time.perf_counter() - t_lookup0
    plausible = 0
    lost_pair_hits = 0
    lost_pair_plausible = 0
    reason_counts: Counter[str] = Counter()
    best: dict[str, Any] | None = None
    first_plausible: dict[str, Any] | None = None
    first_hash_pair_plausible: dict[str, Any] | None = None
    first_lost_pair_plausible: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = []
    t_hyp0 = time.perf_counter()
    tested = 0
    for hit_rank, hit in enumerate(hits, start=1):
        if tested >= int(args.max_hypotheses):
            break
        tested += 1
        image_record = image_records[hit.image_record_index]
        cat_hash = int(disk_index.ratio_hashes[hit.catalog_record_index]) if disk_index.ratio_hashes.shape[0] else -1
        img_hash = -1 if image_record.ratio_hash is None else int(image_record.ratio_hash)
        lost_by_pair_hash = img_hash != cat_hash
        if lost_by_pair_hash:
            lost_pair_hits += 1
        try:
            image_quad_points = image_positions[np.asarray(hit.image_quad_indices, dtype=np.int64)]
            catalog_quad_world = disk_index.catalog_ra_dec[np.asarray(hit.catalog_quad_indices, dtype=np.int64)]
            wcs = _fit_quad_wcs(image_quad_points, catalog_quad_world)
            metrics = _evaluate_wcs(
                wcs=wcs,
                reference_wcs=reference_wcs,
                image_shape=(int(image_meta["image_shape"][0]), int(image_meta["image_shape"][1])),
                image_positions=image_positions,
                catalog_world=disk_index.catalog_ra_dec,
                inlier_radius_px=float(args.inlier_radius_px),
            )
            reason = _reject_reason(metrics, args)
        except Exception as exc:
            metrics = {"error": str(exc), "inliers": 0, "rms_px": float("inf")}
            reason = "fit_failed"
        reason_counts[reason] += 1
        row = {
            "hit_rank": int(hit_rank),
            "image_record_index": int(hit.image_record_index),
            "catalog_record_index": int(hit.catalog_record_index),
            "code_distance": float(hit.code_distance),
            "lost_by_pair_hash": bool(lost_by_pair_hash),
            "image_ratio_hash": int(img_hash),
            "catalog_ratio_hash": int(cat_hash),
            "image_quad": list(hit.image_quad_indices),
            "catalog_quad": list(hit.catalog_quad_indices),
            "reason": reason,
            **metrics,
        }
        if reason == "plausible":
            plausible += 1
            if lost_by_pair_hash:
                lost_pair_plausible += 1
            if first_plausible is None:
                first_plausible = dict(row)
            if first_hash_pair_plausible is None and not lost_by_pair_hash:
                first_hash_pair_plausible = dict(row)
            if first_lost_pair_plausible is None and lost_by_pair_hash:
                first_lost_pair_plausible = dict(row)
        if len(examples) < int(args.max_examples) and (lost_by_pair_hash or reason == "plausible"):
            examples.append(dict(row))
        if best is None or (
            int(row.get("inliers", 0)),
            -float(row.get("rms_px", float("inf"))) if math.isfinite(float(row.get("rms_px", float("inf")))) else -1e9,
        ) > (
            int(best.get("inliers", 0)),
            -float(best.get("rms_px", float("inf"))) if math.isfinite(float(best.get("rms_px", float("inf")))) else -1e9,
        ):
            best = dict(row)
    hyp_s = time.perf_counter() - t_hyp0
    verdict = "no_plausible"
    if plausible > 0 and lost_pair_plausible > 0:
        if first_lost_pair_plausible and (
            not first_hash_pair_plausible
            or int(first_lost_pair_plausible["hit_rank"]) < int(first_hash_pair_plausible["hit_rank"])
        ):
            verdict = "gain_concret_lost_hash_before_hash_exact"
        else:
            verdict = "gain_partiel_lost_hash_plausible"
    elif plausible > 0:
        verdict = "non_regression_no_lost_hash_gain"
    return {
        "label": label,
        "filename": filename,
        "tile_key": tile_key,
        "source_fits": str(source),
        "reference_fits": str(reference),
        "image_meta": image_meta,
        "image_quads": int(image_quads.shape[0]),
        "image_4d_records": int(len(image_records)),
        "hits_4d": int(len(hits)),
        "hits_tested": int(tested),
        "hits_plausible": int(plausible),
        "hits_lost_by_pair_hash": int(lost_pair_hits),
        "plausible_lost_by_pair_hash": int(lost_pair_plausible),
        "reason_counts": dict(reason_counts),
        "first_plausible_hit": first_plausible,
        "first_hash_pair_plausible_hit": first_hash_pair_plausible,
        "first_lost_pair_plausible_hit": first_lost_pair_plausible,
        "best_hit": best,
        "examples": examples,
        "timing": {
            "kd_lookup_s": float(lookup_s),
            "hypothesis_validation_s": float(hyp_s),
        },
        "verdict": verdict,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.1 - disk 4D index probe",
        "",
        "## Conclusion",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Index disque: `{payload['disk_index']['path']}`",
        f"- Schema: `{payload['disk_index']['metadata'].get('schema')}`",
        f"- Entrees index: `{payload['disk_index']['metadata'].get('entry_count')}`",
        "- Scope: index disque experimental focalise ; ancien backend conserve par defaut.",
        "",
        "## Audit P2.0 -> P2.1",
        "",
        "- Une premiere variante disque limitee a `400` etoiles catalogue / `8000` quads etait partielle : `232350` ne produisait aucune hypothese plausible.",
        "- Cause auditee : les quads P2.0 utiles de `232350 / d50_2823` utilisent des etoiles de rang catalogue global superieur a `400` (`425`, `689`, `929`, `1543`, `1871` observes dans les exemples).",
        "- Le rapport courant utilise donc un index focalise `d50_2823` plus couvrant (`2000` etoiles, `40000` quads), toujours sans rebuild complet.",
        "",
        "## Comparaison P2.0 / P2.1 / hash exact",
        "",
        "| cas | P2.0 hits/plausibles | P2.1 hits/testes/plausibles | P2.1 perdus-hash plausibles | premier P2.1 plausible | premier P2.1 hash-exact | premier P2.1 perdu-hash | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        p20 = P20_BASELINE.get(str(case["label"]), {})
        def idx(row: dict[str, Any] | None) -> str:
            return "" if not row else str(row.get("hit_rank", ""))

        lines.append(
            "| `{label}` | {p20_hits}/{p20_plausible} | {hits}/{tested}/{plausible} | {lost_ok} | {first} | {first_hash} | {first_lost} | {verdict} |".format(
                label=case["label"],
                p20_hits=p20.get("hits_4d", ""),
                p20_plausible=p20.get("hits_plausible", ""),
                hits=case["hits_4d"],
                tested=case["hits_tested"],
                plausible=case["hits_plausible"],
                lost_ok=case["plausible_lost_by_pair_hash"],
                first=idx(case.get("first_plausible_hit")),
                first_hash=idx(case.get("first_hash_pair_plausible_hit")),
                first_lost=idx(case.get("first_lost_pair_plausible_hit")),
                verdict=case["verdict"],
            )
        )
    lines.extend(["", "## Details", ""])
    for case in payload["cases"]:
        lines.extend(
            [
                f"### {case['label']} / {case['tile_key']}",
                "",
                f"- quads image / records 4D image: `{case['image_quads']}` / `{case['image_4d_records']}`",
                f"- hits 4D / testes: `{case['hits_4d']}` / `{case['hits_tested']}`",
                f"- hypotheses plausibles: `{case['hits_plausible']}`",
                f"- plausibles perdus par hash exact pair-level: `{case['plausible_lost_by_pair_hash']}`",
                f"- raisons de rejet: `{case['reason_counts']}`",
                f"- temps KD lookup: `{case['timing']['kd_lookup_s']:.4f} s`",
                f"- temps hypotheses/validation: `{case['timing']['hypothesis_validation_s']:.4f} s`",
            ]
        )
        best = case.get("best_hit") or {}
        if best:
            lines.append(
                "- meilleur hit: rang `{}` ; inliers `{}` ; RMS `{:.3f}` px ; centre `{:.2f}\"` ; coins `{:.2f}\"` ; scale ratio `{}` ; lost_by_hash `{}`".format(
                    best.get("hit_rank"),
                    best.get("inliers"),
                    float(best.get("rms_px", float("inf"))),
                    float(best.get("center_sep_arcsec", float("inf"))),
                    float(best.get("corner_max_sep_arcsec", float("inf"))),
                    best.get("scale_ratio"),
                    best.get("lost_by_pair_hash"),
                )
            )
        examples = list(case.get("examples") or [])[:5]
        if examples:
            lines.extend(["", "Exemples:", ""])
            for item in examples:
                lines.append(
                    "- rang `{}` dist `{:.6f}` lost_hash `{}` reason `{}` inliers `{}` rms `{}` image_quad `{}` catalog_quad `{}`".format(
                        item.get("hit_rank"),
                        float(item.get("code_distance", float("nan"))),
                        item.get("lost_by_pair_hash"),
                        item.get("reason"),
                        item.get("inliers"),
                        item.get("rms_px"),
                        item.get("image_quad"),
                        item.get("catalog_quad"),
                    )
                )
        lines.append("")
    lines.extend(["## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.1 offline probe: build/load an experimental disk astrometry_ab_code_4d_v1 index and test WCS hypotheses.")
    ap.add_argument("--quad-hash-schema", default=ASTROMETRY_AB_CODE_4D_SCHEMA, choices=(ASTROMETRY_AB_CODE_4D_SCHEMA,))
    ap.add_argument("--case", action="append", choices=sorted(DEFAULT_CASES), default=None)
    ap.add_argument("--tile", action="append", default=["d50_2823"])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--quad4d-index", type=Path, default=DEFAULT_4D_INDEX)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--level", default="S")
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--code-dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--max-image-stars", type=int, default=120)
    ap.add_argument("--max-catalog-stars", type=int, default=400)
    ap.add_argument("--max-image-quads", type=int, default=2500)
    ap.add_argument("--max-catalog-quads", type=int, default=8000)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--catalog-strategy", default="catalog_ring_coverage")
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--max-examples", type=int, default=12)
    ap.add_argument("--inlier-radius-px", type=float, default=3.0)
    ap.add_argument("--min-plausible-inliers", type=int, default=8)
    ap.add_argument("--max-plausible-rms-px", type=float, default=2.0)
    ap.add_argument("--min-scale-ratio", type=float, default=0.97)
    ap.add_argument("--max-scale-ratio", type=float, default=1.03)
    ap.add_argument("--max-center-sep-arcsec", type=float, default=120.0)
    ap.add_argument("--max-corner-sep-arcsec", type=float, default=240.0)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    index_path, build_meta = _build_if_needed(args)
    t_load0 = time.perf_counter()
    disk_index = Quad4DIndex.load(index_path)
    load_s = time.perf_counter() - t_load0
    labels = list(args.case or sorted(DEFAULT_CASES))
    cases = []
    for label in labels:
        filename, tile_key = DEFAULT_CASES[label]
        cases.append(_run_case(label, filename, tile_key, disk_index, args))
    p2350 = next((case for case in cases if case["label"] == "232350"), None)
    p2102 = next((case for case in cases if case["label"] == "232102"), None)
    if p2350 and p2350.get("verdict") == "gain_concret_lost_hash_before_hash_exact" and p2102 and int(p2102.get("hits_plausible", 0)) > 0:
        global_verdict = "P2.1 positif: index disque 4D reproduit le gain concret de P2.0 et garde 232102 non-regresse"
    elif any(int(case.get("hits_plausible", 0)) > 0 for case in cases):
        global_verdict = "P2.1 partiel: index disque 4D produit des hypotheses plausibles mais ne reproduit pas completement P2.0"
    else:
        global_verdict = "P2.1 negatif: index disque 4D ne produit pas d'hypothese utile"
    payload = {
        "schema": "zeblind.p21_disk_4d_index_probe.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "global_verdict": global_verdict,
        "disk_index": {
            "path": str(index_path),
            "load_s": float(load_s),
            "build": build_meta,
            "metadata": disk_index.metadata,
        },
        "cases": cases,
        "p20_baseline": P20_BASELINE,
        "params": {
            "quad_hash_schema": str(args.quad_hash_schema),
            "code_tol": float(args.code_tol),
            "max_hits_4d": int(args.max_hits_4d),
            "max_hits_per_image_quad": int(args.max_hits_per_image_quad),
            "max_hypotheses": int(args.max_hypotheses),
            "max_image_stars": int(args.max_image_stars),
            "max_catalog_stars": int(args.max_catalog_stars),
            "max_image_quads": int(args.max_image_quads),
            "max_catalog_quads": int(args.max_catalog_quads),
            "image_strategy": str(args.image_strategy),
            "catalog_strategy": str(args.catalog_strategy),
            "inlier_radius_px": float(args.inlier_radius_px),
            "min_plausible_inliers": int(args.min_plausible_inliers),
            "max_plausible_rms_px": float(args.max_plausible_rms_px),
            "scale_ratio_bounds": [float(args.min_scale_ratio), float(args.max_scale_ratio)],
            "max_center_sep_arcsec": float(args.max_center_sep_arcsec),
            "max_corner_sep_arcsec": float(args.max_corner_sep_arcsec),
        },
    }
    report = args.report.expanduser().resolve()
    json_out = args.json_out.expanduser().resolve()
    _write_report(report, payload)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    print(f"wrote {json_out}")
    print(global_verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
