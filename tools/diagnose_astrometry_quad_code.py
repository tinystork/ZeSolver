#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.asterisms import hash_quads, sample_quads
from zeblindsolver.image_prep import build_pyramid, read_fits_as_luma, remove_background
from zeblindsolver.quad_code_diagnostic import canonicalize_astrometry_longest_ab, compare_quad_codes
from zeblindsolver.star_detect import detect_stars


DEFAULT_SOURCE = ROOT / "reports/eq_ircut_cleanbench_20260518_230249/data/Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit"
DEFAULT_REFERENCE = ROOT / "reports/r47i_s7_testzenear_full_product_clean_20260624/input/Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit"
DEFAULT_INDEX_ROOT = ROOT / "reports/s3_focused_index_20260701_p16_multicase_v3/index"
DEFAULT_TILE = "d50_2823"
DEFAULT_REPORT = ROOT / "reports/zeblind_quad_code_diagnostic.md"


def _detect_image_stars(fits_path: Path, *, max_stars: int, detect_k_sigma: float, detect_min_area: int) -> tuple[np.ndarray, dict[str, Any]]:
    image = read_fits_as_luma(fits_path)
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
    meta = {
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "raw_sources": raw_count,
        "kept_sources": int(stars.shape[0]),
    }
    return stars, meta


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
    raise KeyError(f"tile {tile_key!r} not found in {index_root}")


def _catalog_stars_from_tile(
    *,
    tile: dict[str, np.ndarray],
    reference_fits: Path,
    image_shape: tuple[int, int],
    max_stars: int,
    margin_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    with fits.open(reference_fits, memmap=False) as hdul:
        wcs = WCS(hdul[0].header)
    if not wcs.has_celestial:
        raise RuntimeError(f"reference FITS has no celestial WCS: {reference_fits}")
    ra = np.asarray(tile["ra_deg"], dtype=np.float64)
    dec = np.asarray(tile["dec_deg"], dtype=np.float64)
    px = np.asarray(wcs.all_world2pix(ra, dec, 0), dtype=np.float64).T
    height, width = int(image_shape[0]), int(image_shape[1])
    finite = np.isfinite(px[:, 0]) & np.isfinite(px[:, 1])
    inside = (
        finite
        & (px[:, 0] >= -float(margin_px))
        & (px[:, 0] <= width + float(margin_px))
        & (px[:, 1] >= -float(margin_px))
        & (px[:, 1] <= height + float(margin_px))
    )
    idx = np.nonzero(inside)[0]
    mag = np.asarray(tile["mag"], dtype=np.float64) if "mag" in tile else np.arange(ra.shape[0], dtype=np.float64)
    idx = idx[np.argsort(mag[idx], kind="stable")]
    if max_stars > 0:
        idx = idx[:max_stars]
    stars = np.zeros(idx.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    stars["x"] = px[idx, 0].astype(np.float32)
    stars["y"] = px[idx, 1].astype(np.float32)
    stars["mag"] = mag[idx].astype(np.float32)
    return stars, {
        "projected_catalog_stars": int(np.count_nonzero(finite)),
        "inside_catalog_stars": int(np.count_nonzero(inside)),
        "kept_catalog_stars": int(stars.shape[0]),
    }


def _positions(stars: np.ndarray) -> np.ndarray:
    if stars.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack((stars["x"].astype(np.float64), stars["y"].astype(np.float64)))


def _astrometry_codes(quads: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    codes: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for source_idx, quad in enumerate(np.asarray(quads, dtype=np.int64)):
        if len(set(int(v) for v in quad)) < 4:
            continue
        result = canonicalize_astrometry_longest_ab(positions[quad])
        if result is None:
            continue
        codes.append(result.code)
        rows.append(
            {
                "source_quad_index": int(source_idx),
                "quad": [int(v) for v in quad],
                "astrometry_order": [int(v) for v in result.order],
                "code": [float(v) for v in result.code],
            }
        )
    if not codes:
        return np.zeros((0, 4), dtype=np.float64), rows
    return np.vstack(codes).astype(np.float64, copy=False), rows


def _hashes(quads: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    hashed = hash_quads(quads, positions, return_source_indices=True)
    rows: list[dict[str, Any]] = []
    source_indices = hashed.source_indices if hashed.source_indices is not None else np.arange(hashed.hashes.shape[0])
    for idx, hash_value in enumerate(hashed.hashes):
        rows.append(
            {
                "source_quad_index": int(source_indices[idx]),
                "quad": [int(v) for v in hashed.indices[idx]],
                "hash": int(hash_value),
            }
        )
    return hashed.hashes.astype(np.uint64, copy=False), rows


def _range_matches(image_codes: np.ndarray, catalog_codes: np.ndarray, *, code_tol: float) -> tuple[np.ndarray, np.ndarray]:
    if image_codes.shape[0] == 0 or catalog_codes.shape[0] == 0:
        return np.zeros(image_codes.shape[0], dtype=bool), np.full(image_codes.shape[0], -1, dtype=np.int64)
    tree = cKDTree(catalog_codes)
    neighbors = tree.query_ball_point(image_codes, r=float(code_tol))
    mask = np.asarray([len(row) > 0 for row in neighbors], dtype=bool)
    nearest = np.full(image_codes.shape[0], -1, dtype=np.int64)
    if np.any(mask):
        distances, indices = tree.query(image_codes[mask], k=1, distance_upper_bound=float(code_tol))
        nearest[np.nonzero(mask)[0]] = np.where(np.isfinite(distances), indices, -1).astype(np.int64)
    return mask, nearest


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# ZeBlind quad-code diagnostic",
        "",
        "## Conclusion",
        "",
        f"- Verdict: **{payload['conclusion']}**",
        f"- Cas: `{payload['source_fits']}`",
        f"- Reference WCS: `{payload['reference_fits']}`",
        f"- Index/tile: `{payload['index_root']}` / `{payload['tile_key']}`",
        "",
        "## Mesures",
        "",
        "| mesure | valeur |",
        "|---|---:|",
    ]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Exemples perdus par le hash actuel mais retrouves en range 4D", ""])
    examples = list(payload.get("lost_by_hash_examples") or [])
    if not examples:
        lines.append("- Aucun exemple dans cette enveloppe.")
    else:
        for item in examples:
            lines.append(
                "- image_quad `{img_quad}` -> catalog_quad `{cat_quad}`, dist4d `{dist:.6f}`, image_hash `{img_hash}`, catalog_hash `{cat_hash}`".format(
                    img_quad=item["image_quad"],
                    cat_quad=item["catalog_quad"],
                    dist=float(item["code_distance"]),
                    img_hash=item["image_hash"],
                    cat_hash=item["catalog_hash"],
                )
            )
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_fits = args.fits.expanduser().resolve()
    reference_fits = args.reference_fits.expanduser().resolve()
    index_root = args.index_root.expanduser().resolve()
    image_stars, image_meta = _detect_image_stars(
        source_fits,
        max_stars=int(args.max_image_stars),
        detect_k_sigma=float(args.detect_k_sigma),
        detect_min_area=int(args.detect_min_area),
    )
    tile = _load_tile(index_root, args.tile)
    catalog_stars, catalog_meta = _catalog_stars_from_tile(
        tile=tile,
        reference_fits=reference_fits,
        image_shape=(int(image_meta["image_shape"][0]), int(image_meta["image_shape"][1])),
        max_stars=int(args.max_catalog_stars),
        margin_px=float(args.catalog_margin_px),
    )
    image_positions = _positions(image_stars)
    catalog_positions = _positions(catalog_stars)
    image_quads = sample_quads(image_stars, max_quads=int(args.max_image_quads), strategy=str(args.image_strategy))
    catalog_quads = sample_quads(catalog_stars, max_quads=int(args.max_catalog_quads), strategy=str(args.catalog_strategy))

    image_hashes, image_hash_rows = _hashes(image_quads, image_positions)
    catalog_hashes, catalog_hash_rows = _hashes(catalog_quads, catalog_positions)
    catalog_hash_set = {int(v) for v in catalog_hashes}
    hash_match_mask = np.asarray([int(v) in catalog_hash_set for v in image_hashes], dtype=bool)

    image_codes, image_code_rows = _astrometry_codes(image_quads, image_positions)
    catalog_codes, catalog_code_rows = _astrometry_codes(catalog_quads, catalog_positions)
    range_mask, nearest_catalog = _range_matches(image_codes, catalog_codes, code_tol=float(args.code_tol))

    image_hash_by_source = {int(row["source_quad_index"]): int(row["hash"]) for row in image_hash_rows}
    catalog_hash_by_source = {int(row["source_quad_index"]): int(row["hash"]) for row in catalog_hash_rows}
    catalog_hash_counter = Counter(int(v) for v in catalog_hashes)
    range_without_hash_count = 0
    lost_examples: list[dict[str, Any]] = []
    image_source_to_hash_match = {
        int(row["source_quad_index"]): bool(int(row["hash"]) in catalog_hash_set)
        for row in image_hash_rows
    }
    for code_idx, ok in enumerate(range_mask):
        if not ok:
            continue
        source_idx = int(image_code_rows[code_idx]["source_quad_index"])
        if image_source_to_hash_match.get(source_idx, False):
            continue
        range_without_hash_count += 1
        if len(lost_examples) >= int(args.max_examples):
            continue
        cat_idx = int(nearest_catalog[code_idx])
        if cat_idx < 0:
            continue
        cat_source_idx = int(catalog_code_rows[cat_idx]["source_quad_index"])
        img_hash = image_hash_by_source.get(source_idx)
        cat_hash = catalog_hash_by_source.get(cat_source_idx)
        lost_examples.append(
            {
                "image_quad": image_code_rows[code_idx]["quad"],
                "catalog_quad": catalog_code_rows[cat_idx]["quad"],
                "image_hash": img_hash,
                "catalog_hash": cat_hash,
                "catalog_hash_frequency": int(catalog_hash_counter.get(int(img_hash or -1), 0)),
                "image_code": image_code_rows[code_idx]["code"],
                "catalog_code": catalog_code_rows[cat_idx]["code"],
                "code_distance": float(np.linalg.norm(image_codes[code_idx] - catalog_codes[cat_idx])),
                "image_compare": compare_quad_codes(image_positions[np.asarray(image_code_rows[code_idx]["quad"], dtype=np.int64)]),
                "catalog_compare": compare_quad_codes(catalog_positions[np.asarray(catalog_code_rows[cat_idx]["quad"], dtype=np.int64)]),
            }
        )

    hash_matches = int(np.count_nonzero(hash_match_mask))
    range_matches = int(np.count_nonzero(range_mask))
    if range_matches > hash_matches and lost_examples:
        conclusion = "divergence confirmee"
    elif range_matches > hash_matches:
        conclusion = "divergence partielle"
    else:
        conclusion = "divergence non confirmee dans cette enveloppe"

    rows = [
        ("sources image gardees", int(image_stars.shape[0])),
        ("sources catalogue gardees", int(catalog_stars.shape[0])),
        ("quads image generes", int(image_quads.shape[0])),
        ("quads catalogue generes", int(catalog_quads.shape[0])),
        ("hashes image valides", int(image_hashes.shape[0])),
        ("hashes catalogue valides", int(catalog_hashes.shape[0])),
        ("quads image matchant hash actuel", hash_matches),
        ("codes image 4D valides", int(image_codes.shape[0])),
        ("codes catalogue 4D valides", int(catalog_codes.shape[0])),
        ("quads image matchant range 4D", range_matches),
        ("range 4D sans hash exact", int(range_without_hash_count)),
    ]
    payload = {
        "schema": "zeblind.quad_code_diagnostic.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_fits": str(source_fits),
        "reference_fits": str(reference_fits),
        "index_root": str(index_root),
        "tile_key": str(args.tile),
        "image_meta": image_meta,
        "catalog_meta": catalog_meta,
        "rows": rows,
        "lost_by_hash_examples": lost_examples,
        "conclusion": conclusion,
        "params": {
            "max_image_stars": int(args.max_image_stars),
            "max_catalog_stars": int(args.max_catalog_stars),
            "max_image_quads": int(args.max_image_quads),
            "max_catalog_quads": int(args.max_catalog_quads),
            "image_strategy": str(args.image_strategy),
            "catalog_strategy": str(args.catalog_strategy),
            "code_tol": float(args.code_tol),
            "detect_k_sigma": float(args.detect_k_sigma),
            "detect_min_area": int(args.detect_min_area),
            "catalog_margin_px": float(args.catalog_margin_px),
        },
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline diagnostic comparing ZeBlind ratio hashes with Astrometry-like 4D AB/C/D range search.")
    ap.add_argument("--fits", type=Path, default=DEFAULT_SOURCE, help="Source FITS used for image star extraction.")
    ap.add_argument("--reference-fits", type=Path, default=DEFAULT_REFERENCE, help="FITS carrying the oracle WCS used to project the catalogue.")
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--tile", default=DEFAULT_TILE)
    ap.add_argument("--max-image-stars", type=int, default=120)
    ap.add_argument("--max-catalog-stars", type=int, default=400)
    ap.add_argument("--max-image-quads", type=int, default=2500)
    ap.add_argument("--max-catalog-quads", type=int, default=8000)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--catalog-strategy", default="catalog_ring_coverage")
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--catalog-margin-px", type=float, default=50.0)
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    payload = run(args)
    report = args.report.expanduser().resolve()
    _write_report(report, payload)
    if args.json_out:
        args.json_out.expanduser().resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    print(f"conclusion: {payload['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
