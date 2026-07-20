#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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

from zeblindsolver.asterisms import hash_quads, sample_quads
from zeblindsolver.image_prep import build_pyramid, read_fits_as_luma, remove_background
from zeblindsolver.quad_code_diagnostic import build_astrometry_quad_records, build_memory_quad_code_index
from zeblindsolver.star_detect import detect_stars


DEFAULT_DATA_DIR = ROOT / "reports/eq_ircut_cleanbench_20260518_230249/data"
DEFAULT_REFERENCE_DIR = ROOT / "reports/r47i_s7_testzenear_full_product_clean_20260624/input"
DEFAULT_INDEX_ROOT = ROOT / "reports/s3_focused_index_20260701_p16_multicase_v3/index"
DEFAULT_REPORT = ROOT / "reports/zeblind_p20_memory_4d_backend_probe.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p20_memory_4d_backend_probe.json"
DEFAULT_CASES = {
    "232350": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit", "d50_2823"),
    "232102": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit", "d50_2823"),
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
    raw = int(stars.shape[0])
    if max_stars > 0 and stars.shape[0] > max_stars:
        stars = stars[:max_stars]
    return stars, {
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "raw_sources": raw,
        "kept_sources": int(stars.shape[0]),
    }


def _positions(stars: np.ndarray) -> np.ndarray:
    if stars.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack((stars["x"].astype(np.float64), stars["y"].astype(np.float64)))


def _load_tile(index_root: Path, tile_key: str) -> dict[str, np.ndarray]:
    manifest = json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))
    for entry in list(manifest.get("tiles") or []):
        if str(entry.get("tile_key") or "") != str(tile_key):
            continue
        tile_path = index_root / str(entry.get("tile_file") or "")
        with np.load(tile_path, allow_pickle=False) as payload:
            out = {name: np.asarray(payload[name]) for name in payload.files}
        out["_tile_index"] = np.asarray([int(entry.get("tile_index", -1))], dtype=np.int32)
        return out
    raise KeyError(f"tile not found: {tile_key}")


def _catalog_for_case(
    *,
    tile: dict[str, np.ndarray],
    reference_wcs: WCS,
    image_shape: tuple[int, int],
    max_stars: int,
    margin_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ra = np.asarray(tile["ra_deg"], dtype=np.float64)
    dec = np.asarray(tile["dec_deg"], dtype=np.float64)
    projected = np.asarray(reference_wcs.all_world2pix(ra, dec, 0), dtype=np.float64).T
    height, width = int(image_shape[0]), int(image_shape[1])
    finite = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
    inside = (
        finite
        & (projected[:, 0] >= -float(margin_px))
        & (projected[:, 0] <= width + float(margin_px))
        & (projected[:, 1] >= -float(margin_px))
        & (projected[:, 1] <= height + float(margin_px))
    )
    idx = np.nonzero(inside)[0]
    mag = np.asarray(tile["mag"], dtype=np.float64) if "mag" in tile else np.arange(ra.shape[0], dtype=np.float64)
    idx = idx[np.argsort(mag[idx], kind="stable")]
    if max_stars > 0:
        idx = idx[:max_stars]
    stars = np.zeros(idx.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    stars["x"] = projected[idx, 0].astype(np.float32)
    stars["y"] = projected[idx, 1].astype(np.float32)
    stars["mag"] = mag[idx].astype(np.float32)
    world = np.column_stack((ra[idx], dec[idx])).astype(np.float64)
    return stars, world, {
        "projected_catalog_stars": int(np.count_nonzero(finite)),
        "inside_catalog_stars": int(np.count_nonzero(inside)),
        "kept_catalog_stars": int(stars.shape[0]),
    }


def _hash_lookup_rows(quads: np.ndarray, positions: np.ndarray) -> tuple[set[int], dict[int, int | None]]:
    hashed = hash_quads(quads, positions, return_source_indices=True)
    hash_set = {int(v) for v in hashed.hashes}
    by_source: dict[int, int | None] = {}
    if hashed.source_indices is not None:
        for source_idx, value in zip(hashed.source_indices, hashed.hashes):
            by_source[int(source_idx)] = int(value)
    return hash_set, by_source


def _wcs_corner_metrics(candidate: WCS, reference: WCS, image_shape: tuple[int, int]) -> dict[str, Any]:
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
    ref_world = reference.pixel_to_world(pts[:, 0], pts[:, 1])
    cand_world = candidate.pixel_to_world(pts[:, 0], pts[:, 1])
    sep = ref_world.separation(cand_world).arcsec
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
    if not np.any(finite) or image_positions.shape[0] == 0:
        inliers = 0
        rms = float("inf")
    else:
        tree = cKDTree(image_positions)
        distances, _indices = tree.query(projected[finite], k=1, distance_upper_bound=float(inlier_radius_px))
        good = np.isfinite(distances) & (distances <= float(inlier_radius_px))
        inliers = int(np.count_nonzero(good))
        rms = float(np.sqrt(np.mean(distances[good] ** 2))) if inliers > 0 else float("inf")
    out = {
        "inliers": inliers,
        "rms_px": rms,
    }
    out.update(_wcs_corner_metrics(wcs, reference_wcs, image_shape))
    return out


def _fit_quad_wcs(image_points: np.ndarray, catalog_world: np.ndarray) -> WCS:
    pixels = np.asarray(image_points, dtype=np.float64)
    world = np.asarray(catalog_world, dtype=np.float64)
    coords = SkyCoord(ra=world[:, 0] * u.deg, dec=world[:, 1] * u.deg, frame="icrs")
    return fit_wcs_from_points((pixels[:, 0], pixels[:, 1]), coords, projection="TAN")


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


def _run_case(label: str, filename: str, tile_key: str, args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / filename
    reference = args.reference_dir.expanduser().resolve() / filename
    reference_wcs, ref_shape = _load_reference_wcs(reference)
    image_stars, image_meta = _detect_image_stars(
        source,
        max_stars=int(args.max_image_stars),
        detect_k_sigma=float(args.detect_k_sigma),
        detect_min_area=int(args.detect_min_area),
    )
    image_positions = _positions(image_stars)
    tile = _load_tile(args.index_root.expanduser().resolve(), tile_key)
    catalog_stars, catalog_world, catalog_meta = _catalog_for_case(
        tile=tile,
        reference_wcs=reference_wcs,
        image_shape=(int(image_meta["image_shape"][0]), int(image_meta["image_shape"][1])),
        max_stars=int(args.max_catalog_stars),
        margin_px=float(args.catalog_margin_px),
    )
    catalog_positions = _positions(catalog_stars)
    image_quads = sample_quads(image_stars, max_quads=int(args.max_image_quads), strategy=str(args.image_strategy))
    catalog_quads = sample_quads(catalog_stars, max_quads=int(args.max_catalog_quads), strategy=str(args.catalog_strategy))
    image_records = build_astrometry_quad_records(image_quads, image_positions)
    index = build_memory_quad_code_index(catalog_quads, catalog_positions, tile_key=tile_key)
    catalog_hash_set, _catalog_hash_by_source = _hash_lookup_rows(catalog_quads, catalog_positions)

    hits = 0
    fitted = 0
    plausible = 0
    lost_pair_hits = 0
    lost_image_hash_hits = 0
    lost_pair_plausible = 0
    lost_image_hash_plausible = 0
    reason_counts: Counter[str] = Counter()
    best: dict[str, Any] | None = None
    first_plausible: dict[str, Any] | None = None
    first_hash_pair_plausible: dict[str, Any] | None = None
    first_lost_pair_plausible: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = []
    hit_index = 0
    for image_record_index, image_record in enumerate(image_records):
        neighbors = index.query(image_record.code, code_tol=float(args.code_tol))
        if not neighbors:
            continue
        image_hash_present = image_record.ratio_hash is not None and int(image_record.ratio_hash) in catalog_hash_set
        for catalog_record_index, distance in neighbors:
            if hits >= int(args.max_hits):
                break
            catalog_record = index.records[catalog_record_index]
            hits += 1
            hit_index += 1
            pair_hash_match = (
                image_record.ratio_hash is not None
                and catalog_record.ratio_hash is not None
                and int(image_record.ratio_hash) == int(catalog_record.ratio_hash)
            )
            lost_by_pair_hash = not pair_hash_match
            lost_by_image_hash = not bool(image_hash_present)
            if lost_by_pair_hash:
                lost_pair_hits += 1
            if lost_by_image_hash:
                lost_image_hash_hits += 1
            image_quad_points = image_positions[np.asarray(image_record.ordered_indices, dtype=np.int64)]
            catalog_quad_world = catalog_world[np.asarray(catalog_record.ordered_indices, dtype=np.int64)]
            try:
                wcs = _fit_quad_wcs(image_quad_points, catalog_quad_world)
                metrics = _evaluate_wcs(
                    wcs=wcs,
                    reference_wcs=reference_wcs,
                    image_shape=(int(image_meta["image_shape"][0]), int(image_meta["image_shape"][1])),
                    image_positions=image_positions,
                    catalog_world=catalog_world,
                    inlier_radius_px=float(args.inlier_radius_px),
                )
                fitted += 1
                reason = _reject_reason(metrics, args)
            except Exception as exc:
                metrics = {"error": str(exc), "inliers": 0, "rms_px": float("inf")}
                reason = "fit_failed"
            reason_counts[reason] += 1
            is_plausible = reason == "plausible"
            if is_plausible:
                plausible += 1
                if lost_by_pair_hash:
                    lost_pair_plausible += 1
                if lost_by_image_hash:
                    lost_image_hash_plausible += 1
                row = {
                    "hit_index": int(hit_index),
                    "image_record_index": int(image_record_index),
                    "catalog_record_index": int(catalog_record_index),
                    "code_distance": float(distance),
                    "lost_by_pair_hash": bool(lost_by_pair_hash),
                    "lost_by_image_hash": bool(lost_by_image_hash),
                    "image_ratio_hash": image_record.ratio_hash,
                    "catalog_ratio_hash": catalog_record.ratio_hash,
                    "image_quad": list(image_record.ordered_indices),
                    "catalog_quad": list(catalog_record.ordered_indices),
                    **metrics,
                }
                if first_plausible is None:
                    first_plausible = dict(row)
                if first_hash_pair_plausible is None and pair_hash_match:
                    first_hash_pair_plausible = dict(row)
                if first_lost_pair_plausible is None and lost_by_pair_hash:
                    first_lost_pair_plausible = dict(row)
            else:
                row = {
                    "hit_index": int(hit_index),
                    "image_record_index": int(image_record_index),
                    "catalog_record_index": int(catalog_record_index),
                    "code_distance": float(distance),
                    "lost_by_pair_hash": bool(lost_by_pair_hash),
                    "lost_by_image_hash": bool(lost_by_image_hash),
                    "reason": reason,
                    "image_ratio_hash": image_record.ratio_hash,
                    "catalog_ratio_hash": catalog_record.ratio_hash,
                    "image_quad": list(image_record.ordered_indices),
                    "catalog_quad": list(catalog_record.ordered_indices),
                    **metrics,
                }
            if len(examples) < int(args.max_examples) and (lost_by_pair_hash or is_plausible):
                examples.append(dict(row))
            if best is None or (
                int(metrics.get("inliers", 0)),
                -float(metrics.get("rms_px", float("inf"))) if math.isfinite(float(metrics.get("rms_px", float("inf")))) else -1e9,
            ) > (
                int(best.get("inliers", 0)),
                -float(best.get("rms_px", float("inf"))) if math.isfinite(float(best.get("rms_px", float("inf")))) else -1e9,
            ):
                best = dict(row)
        if hits >= int(args.max_hits):
            break
    return {
        "label": label,
        "filename": filename,
        "tile_key": tile_key,
        "source_fits": str(source),
        "reference_fits": str(reference),
        "image_meta": image_meta,
        "catalog_meta": catalog_meta,
        "image_quads": int(image_quads.shape[0]),
        "catalog_quads": int(catalog_quads.shape[0]),
        "image_4d_records": int(len(image_records)),
        "catalog_4d_records": int(len(index.records)),
        "hits_4d": int(hits),
        "hits_fitted": int(fitted),
        "hits_plausible": int(plausible),
        "hits_lost_by_pair_hash": int(lost_pair_hits),
        "hits_lost_by_image_hash": int(lost_image_hash_hits),
        "plausible_lost_by_pair_hash": int(lost_pair_plausible),
        "plausible_lost_by_image_hash": int(lost_image_hash_plausible),
        "reason_counts": dict(reason_counts),
        "best_hit": best,
        "first_plausible_hit": first_plausible,
        "first_hash_pair_plausible_hit": first_hash_pair_plausible,
        "first_lost_pair_plausible_hit": first_lost_pair_plausible,
        "examples": examples,
    }


def _case_verdict(case: dict[str, Any]) -> str:
    if int(case.get("hits_plausible", 0)) <= 0:
        return "range 4D produit des hits mais aucune hypothese WCS plausible"
    if int(case.get("plausible_lost_by_pair_hash", 0)) > 0:
        first_lost = case.get("first_lost_pair_plausible_hit") or {}
        first_hash = case.get("first_hash_pair_plausible_hit") or {}
        if not first_hash or int(first_lost.get("hit_index", 10**9)) < int(first_hash.get("hit_index", 10**9)):
            return "gain concret: candidat plausible perdu par hash trouve avant le premier candidat hash-exact"
        return "gain partiel: au moins un candidat plausible est perdu par hash exact"
    return "non-regression locale: candidats plausibles presents, mais pas de gain perdu-par-hash"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.0 - memory 4D backend probe",
        "",
        "## Conclusion",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Scope: probe offline en memoire uniquement ; aucun branchement runtime, aucun rebuild complet.",
        "",
        "## Synthese",
        "",
        "| cas | tile | hits 4D | plausibles | perdus par hash pair | plausibles perdus par hash | premier plausible | premier hash-exact plausible | premier perdu-hash plausible | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        def idx(row: dict[str, Any] | None) -> str:
            return "" if not row else str(row.get("hit_index", ""))

        lines.append(
            "| `{label}` | `{tile}` | {hits} | {plausible} | {lost} | {lost_ok} | {first} | {first_hash} | {first_lost} | {verdict} |".format(
                label=case["label"],
                tile=case["tile_key"],
                hits=case["hits_4d"],
                plausible=case["hits_plausible"],
                lost=case["hits_lost_by_pair_hash"],
                lost_ok=case["plausible_lost_by_pair_hash"],
                first=idx(case.get("first_plausible_hit")),
                first_hash=idx(case.get("first_hash_pair_plausible_hit")),
                first_lost=idx(case.get("first_lost_pair_plausible_hit")),
                verdict=case["verdict"],
            )
        )
    lines.extend(["", "## Details par cas", ""])
    for case in payload["cases"]:
        lines.extend(
            [
                f"### {case['label']} / {case['tile_key']}",
                "",
                f"- quads image/catalogue: `{case['image_quads']}` / `{case['catalog_quads']}`",
                f"- records 4D image/catalogue: `{case['image_4d_records']}` / `{case['catalog_4d_records']}`",
                f"- hits 4D produits: `{case['hits_4d']}`",
                f"- hypotheses fittees: `{case['hits_fitted']}`",
                f"- hypotheses plausibles: `{case['hits_plausible']}`",
                f"- hits absents du hash exact pair-level: `{case['hits_lost_by_pair_hash']}`",
                f"- plausibles absents du hash exact pair-level: `{case['plausible_lost_by_pair_hash']}`",
                f"- raisons de rejet: `{case['reason_counts']}`",
            ]
        )
        best = case.get("best_hit") or {}
        if best:
            lines.append(
                "- meilleur hit: index `{}` ; inliers `{}` ; RMS `{:.3f}` px ; centre `{:.2f}\"` ; corners `{:.2f}\"` ; scale ratio `{}` ; lost_by_pair_hash `{}`".format(
                    best.get("hit_index"),
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
                    "- hit `{}` dist `{:.6f}` lost_pair `{}` reason `{}` inliers `{}` rms `{}` image_quad `{}` catalog_quad `{}`".format(
                        item.get("hit_index"),
                        float(item.get("code_distance", float("nan"))),
                        item.get("lost_by_pair_hash"),
                        item.get("reason", "plausible" if item.get("inliers") else ""),
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
    ap = argparse.ArgumentParser(description="Offline P2.0 probe: in-memory Astrometry-like 4D quad backend -> WCS hypotheses.")
    ap.add_argument("--case", action="append", choices=sorted(DEFAULT_CASES), default=None)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-image-stars", type=int, default=120)
    ap.add_argument("--max-catalog-stars", type=int, default=400)
    ap.add_argument("--max-image-quads", type=int, default=2500)
    ap.add_argument("--max-catalog-quads", type=int, default=8000)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--catalog-strategy", default="catalog_ring_coverage")
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--catalog-margin-px", type=float, default=50.0)
    ap.add_argument("--max-hits", type=int, default=2000)
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

    labels = list(args.case or sorted(DEFAULT_CASES))
    cases: list[dict[str, Any]] = []
    for label in labels:
        filename, tile_key = DEFAULT_CASES[label]
        row = _run_case(label, filename, tile_key, args)
        row["verdict"] = _case_verdict(row)
        cases.append(row)

    any_gain = any(int(case.get("plausible_lost_by_pair_hash", 0)) > 0 for case in cases)
    any_plausible = any(int(case.get("hits_plausible", 0)) > 0 for case in cases)
    if any_gain:
        global_verdict = "P2.0 positif: au moins un hit range 4D perdu par hash produit une hypothese WCS plausible"
    elif any_plausible:
        global_verdict = "P2.0 partiel: range 4D produit des hypotheses plausibles, mais pas de gain perdu-par-hash"
    else:
        global_verdict = "P2.0 negatif: range 4D produit des hits mais aucune hypothese WCS utile"
    payload = {
        "schema": "zeblind.p20_memory_4d_backend_probe.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "global_verdict": global_verdict,
        "cases": cases,
        "params": {
            "code_tol": float(args.code_tol),
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
