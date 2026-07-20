#!/usr/bin/env python3
"""Compare legacy-index and ASTAP-native ZeNear catalog providers.

This is an offline characterization tool. It reads catalogue/index/FITS inputs,
runs on temporary FITS copies when solving is requested, and never writes
provider artefacts back into ASTAP or legacy index directories.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zeblindsolver.metadata_solver import NearSolveConfig, solve_near
from zeblindsolver.near_catalog_provider import AstapNearCatalogProvider, LegacyIndexNearCatalogProvider


def _pixel_fingerprint(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            arr = np.ascontiguousarray(hdu.data)
            h.update(str(arr.dtype).encode("ascii"))
            h.update(str(tuple(arr.shape)).encode("ascii"))
            h.update(arr.tobytes())
    return h.hexdigest()


def _hdu_shapes(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with fits.open(path, memmap=False) as hdul:
        for idx, hdu in enumerate(hdul):
            data = hdu.data
            rows.append(
                {
                    "index": idx,
                    "name": str(getattr(hdu, "name", "")),
                    "has_data": data is not None,
                    "shape": list(data.shape) if data is not None else None,
                    "dtype": str(data.dtype) if data is not None else None,
                }
            )
    return rows


def _wcs_summary(path: Path) -> dict[str, Any]:
    try:
        with fits.open(path, memmap=False) as hdul:
            wcs = WCS(hdul[0].header, naxis=2, relax=True)
            if not bool(wcs.has_celestial):
                return {"has_celestial": False}
            data = hdul[0].data
            image_center_ra = None
            image_center_dec = None
            if data is not None and getattr(data, "ndim", 0) >= 2:
                height, width = data.shape[-2:]
                try:
                    image_center_ra, image_center_dec = wcs.all_pix2world([[float(width) / 2.0, float(height) / 2.0]], 0)[0]
                    image_center_ra = float(image_center_ra)
                    image_center_dec = float(image_center_dec)
                except Exception:
                    image_center_ra = None
                    image_center_dec = None
            scale = None
            rotation = None
            parity = None
            try:
                cd = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
                scale = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
                rotation = float(np.degrees(np.arctan2(cd[1, 0], cd[0, 0])))
                parity = int(1 if np.linalg.det(cd) >= 0 else -1)
            except Exception:
                pass
            return {
                "has_celestial": True,
                "center_ra_deg": float(wcs.wcs.crval[0]),
                "center_dec_deg": float(wcs.wcs.crval[1]),
                "image_center_ra_deg": image_center_ra,
                "image_center_dec_deg": image_center_dec,
                "scale_arcsec": scale,
                "rotation_deg": rotation,
                "parity": parity,
            }
    except Exception as exc:
        return {"has_celestial": False, "error": str(exc)}


def _parse_cone(text: str) -> tuple[str, float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) == 3:
        name = f"ra{parts[0]}_dec{parts[1]}"
        ra, dec, radius = parts
    elif len(parts) == 4:
        name, ra, dec, radius = parts
    else:
        raise argparse.ArgumentTypeError("cone must be RA,DEC,RADIUS or NAME,RA,DEC,RADIUS")
    return name, float(ra), float(dec), float(radius)


def _tile_summary(tile) -> dict[str, Any]:
    return {
        "tile_key": tile.tile_key,
        "family": tile.family,
        "tile_code": tile.tile_code,
        "center_ra_deg": tile.center_ra_deg,
        "center_dec_deg": tile.center_dec_deg,
    }


def _compare_cones(legacy, astap, cones: list[tuple[str, float, float, float]], limit: int, family: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = (family,) if family else None
    for name, ra, dec, radius in cones:
        legacy_tiles = legacy.select_tiles(ra, dec, radius, limit, families=families)
        astap_tiles = astap.select_tiles(ra, dec, radius, limit, families=families)
        legacy_by_key = {tile.tile_key: tile for tile in legacy_tiles}
        common: list[dict[str, Any]] = []
        for tile in astap_tiles:
            legacy_tile = legacy_by_key.get(tile.tile_key)
            if legacy_tile is None:
                continue
            astap_stars = astap.load_stars(tile)
            legacy_stars = legacy.load_stars(legacy_tile)
            comparable = astap_stars.size == legacy_stars.size
            common.append(
                {
                    "tile_key": tile.tile_key,
                    "astap_stars": astap_stars.size,
                    "legacy_stars": legacy_stars.size,
                    "ra_close": bool(comparable and np.allclose(astap_stars.ra_deg, legacy_stars.ra_deg, atol=2e-5)),
                    "dec_close": bool(comparable and np.allclose(astap_stars.dec_deg, legacy_stars.dec_deg, atol=2e-5)),
                    "mag_close": bool(comparable and np.allclose(astap_stars.mag, legacy_stars.mag, atol=0.06)),
                }
            )
        rows.append(
            {
                "case": name,
                "ra_deg": ra,
                "dec_deg": dec,
                "radius_deg": radius,
                "legacy_candidates": [_tile_summary(tile) for tile in legacy_tiles],
                "astap_candidates": [_tile_summary(tile) for tile in astap_tiles],
                "candidate_keys_match": [tile.tile_key for tile in legacy_tiles] == [tile.tile_key for tile in astap_tiles],
                "common_star_comparison": common,
            }
        )
    return rows


def _compare_solves(
    fits_paths: list[Path],
    *,
    index_root: Path,
    astap_provider,
    family: str | None,
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = NearSolveConfig(astap_iso_strict=True, family=family)
    solve_dir = out_dir / "solve_copies"
    solve_dir.mkdir(parents=True, exist_ok=True)
    for src in fits_paths:
        src = src.expanduser().resolve()
        if not src.exists():
            rows.append({"path": str(src), "skipped": "missing"})
            continue
        legacy_copy = solve_dir / f"{src.stem}_legacy{src.suffix}"
        astap_copy = solve_dir / f"{src.stem}_astap{src.suffix}"
        shutil.copy2(src, legacy_copy)
        shutil.copy2(src, astap_copy)
        source_pixels = _pixel_fingerprint(src)
        legacy_before_pixels = _pixel_fingerprint(legacy_copy)
        astap_before_pixels = _pixel_fingerprint(astap_copy)
        source_hdus = _hdu_shapes(src)
        t0 = time.perf_counter()
        legacy_result = solve_near(legacy_copy, index_root, config=cfg)
        legacy_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        astap_result = solve_near(astap_copy, None, config=cfg, catalog_provider=astap_provider)
        astap_s = time.perf_counter() - t0
        legacy_after_pixels = _pixel_fingerprint(legacy_copy)
        astap_after_pixels = _pixel_fingerprint(astap_copy)
        rows.append(
            {
                "path": str(src),
                "source_pixels_preserved": _pixel_fingerprint(src) == source_pixels,
                "legacy_copy_pixels_preserved": legacy_after_pixels == legacy_before_pixels,
                "astap_copy_pixels_preserved": astap_after_pixels == astap_before_pixels,
                "source_hdus": source_hdus,
                "legacy_hdus": _hdu_shapes(legacy_copy),
                "astap_hdus": _hdu_shapes(astap_copy),
                "legacy_success": bool(legacy_result.success),
                "astap_success": bool(astap_result.success),
                "legacy_message": legacy_result.message,
                "astap_message": astap_result.message,
                "legacy_tile": legacy_result.tile_key,
                "astap_tile": astap_result.tile_key,
                "legacy_inliers": (legacy_result.stats or {}).get("inliers"),
                "astap_inliers": (astap_result.stats or {}).get("inliers"),
                "legacy_rms_px": (legacy_result.stats or {}).get("rms_px"),
                "astap_rms_px": (astap_result.stats or {}).get("rms_px"),
                "legacy_wcs": _wcs_summary(legacy_copy),
                "astap_wcs": _wcs_summary(astap_copy),
                "legacy_elapsed_s": legacy_s,
                "astap_elapsed_s": astap_s,
            }
        )
    return rows


def _percentile(values: list[float], q: float) -> float | None:
    clean = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return None
    return float(np.percentile(clean, q))


def _solve_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if not row.get("skipped")]
    legacy_success = {row["path"] for row in completed if row.get("legacy_success")}
    astap_success = {row["path"] for row in completed if row.get("astap_success")}
    legacy_times = [float(row.get("legacy_elapsed_s", 0.0) or 0.0) for row in completed]
    astap_times = [float(row.get("astap_elapsed_s", 0.0) or 0.0) for row in completed]
    return {
        "total": len(completed),
        "legacy_success": len(legacy_success),
        "astap_success": len(astap_success),
        "success_intersection": len(legacy_success & astap_success),
        "astap_gained": sorted(astap_success - legacy_success),
        "astap_lost": sorted(legacy_success - astap_success),
        "legacy_elapsed_median_s": _percentile(legacy_times, 50),
        "legacy_elapsed_p95_s": _percentile(legacy_times, 95),
        "astap_elapsed_median_s": _percentile(astap_times, 50),
        "astap_elapsed_p95_s": _percentile(astap_times, 95),
        "source_pixels_preserved": all(bool(row.get("source_pixels_preserved")) for row in completed),
        "legacy_copy_pixels_preserved": all(bool(row.get("legacy_copy_pixels_preserved")) for row in completed),
        "astap_copy_pixels_preserved": all(bool(row.get("astap_copy_pixels_preserved")) for row in completed),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# ZeNear Catalog Provider Comparison", ""]
    lines.append("## Tile Candidates")
    for row in report["tile_candidate_parity"]:
        lines.append(f"- {row['case']}: match={row['candidate_keys_match']} legacy={len(row['legacy_candidates'])} astap={len(row['astap_candidates'])}")
    lines.append("")
    lines.append("## Solves")
    if report["solve_parity"]:
        summary = report.get("solve_summary", {})
        lines.append(
            f"- summary: total={summary.get('total')} legacy={summary.get('legacy_success')} "
            f"astap={summary.get('astap_success')} intersection={summary.get('success_intersection')}"
        )
        for row in report["solve_parity"]:
            lines.append(
                f"- {Path(row['path']).name}: legacy={row.get('legacy_success')} astap={row.get('astap_success')} "
                f"tiles={row.get('legacy_tile')}/{row.get('astap_tile')} "
                f"inliers={row.get('legacy_inliers')}/{row.get('astap_inliers')}"
            )
    else:
        lines.append("- not requested")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--astap-root", required=True, type=Path)
    parser.add_argument("--family", default="d50")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--cone", action="append", type=_parse_cone, default=[])
    parser.add_argument("--fits", action="append", type=Path, default=[])
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    cones = args.cone or [
        ("normal", 33.0, 12.0, 3.0),
        ("ra0", 359.5, -69.5, 3.0),
        ("high_dec", 120.0, 59.0, 3.0),
    ]
    legacy = LegacyIndexNearCatalogProvider(args.index_root)
    astap = AstapNearCatalogProvider(args.astap_root, families=(args.family,) if args.family else None)
    out_dir = args.out_json.expanduser().resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    solve_parity = _compare_solves(
        args.fits,
        index_root=args.index_root,
        astap_provider=astap,
        family=args.family,
        out_dir=out_dir,
    ) if args.fits else []
    report = {
        "index_root": str(args.index_root.expanduser().resolve()),
        "astap_root": str(args.astap_root.expanduser().resolve()),
        "family": args.family,
        "tile_candidate_parity": _compare_cones(legacy, astap, cones, args.limit, args.family),
        "solve_parity": solve_parity,
        "solve_summary": _solve_summary(solve_parity) if solve_parity else {},
    }
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_md:
        _write_markdown(args.out_md, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
