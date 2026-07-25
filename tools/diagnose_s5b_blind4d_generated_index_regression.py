#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap
from zeblindsolver.quad_index_4d import Quad4DIndex
from zeblindsolver.zeblindsolver import solve_blind

import importlib.util

_P1D3B_TOOL = ROOT / "tools" / "validate_direct_blind4d_runtime.py"
_SPEC = importlib.util.spec_from_file_location("validate_direct_blind4d_runtime", _P1D3B_TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_runtime = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _runtime
_SPEC.loader.exec_module(_runtime)


DEFAULT_CASES = (
    (
        "m106_233459",
        Path("/home/tristan/near_bench_cmp30/testzeblind/001_Light_mosaic_M 106_20.0s_IRCUT_20250518-233459_FAKE_HINT.fit"),
        "d50_2823",
    ),
    (
        "m31_230409",
        Path("/home/tristan/near_bench_cmp30/testzeblind/002_Light_M 31_11_30.0s_IRCUT_20250922-230409_FAKE_HINT.fit"),
        "d50_2602",
    ),
)
P1D3B_ORDER = ("d50_2823", "d50_2822", "d50_2644", "d50_2645", "d50_2602", "d50_2702")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _wcs_keys(header: fits.Header) -> list[str]:
    prefixes = ("CD", "PC", "PV", "A_", "B_", "AP_", "BP_")
    keys = {
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
    }
    return sorted(key for key in header if key in keys or key.startswith(prefixes))


def inspect_fits(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        data = np.asarray(hdul[0].data)
        return {
            "path": str(path),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "wcs_keys": _wcs_keys(header),
            "wcs_has_celestial": bool(WCS(header).has_celestial),
            "ra": header.get("RA"),
            "dec": header.get("DEC"),
            "objctra": header.get("OBJCTRA"),
            "objctdec": header.get("OBJCTDEC"),
            "focal_len": header.get("FOCALLEN"),
            "xpixsz": header.get("XPIXSZ"),
            "ypixsz": header.get("YPIXSZ"),
        }


def index_summary(path: Path) -> dict[str, Any]:
    index = Quad4DIndex.load(path)
    meta = dict(index.metadata)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "tile_count": int(len(index.tile_keys)),
        "tile_keys_head": list(index.tile_keys[:8]),
        "tile_keys_tail": list(index.tile_keys[-8:]),
        "star_count": int(index.catalog_ra_dec.shape[0]),
        "quad_count": int(index.codes_4d.shape[0]),
        "schema": meta.get("schema"),
        "version": meta.get("version"),
        "family": meta.get("source_family"),
        "level": meta.get("level"),
        "max_stars_per_tile": meta.get("max_stars_per_tile"),
        "max_quads_per_tile": meta.get("max_quads_per_tile"),
        "source_max_stars": (meta.get("build_parameters") or {}).get("source_max_stars"),
        "mag_cap": (meta.get("build_parameters") or {}).get("mag_cap"),
        "source_star_truncation_mode": (meta.get("build_parameters") or {}).get("source_star_truncation_mode"),
        "sampler_tag": meta.get("sampler_tag"),
        "code_tol_recommended": meta.get("code_tol_recommended"),
        "dtype": meta.get("dtype"),
        "source_fingerprint": meta.get("source_fingerprint"),
        "builder_version": meta.get("builder_version"),
    }


def build_target_index(astap_root: Path, out_path: Path, tile_keys: Iterable[str], *, dense: bool) -> dict[str, Any]:
    defaults = Astap4DBuildConfig()
    kwargs = {
        "family": "d50",
        "tile_keys": tuple(tile_keys),
        "mag_cap": defaults.mag_cap,
        "source_max_stars": defaults.source_max_stars,
        "source_star_truncation_mode": defaults.source_star_truncation_mode,
        "max_stars_per_tile": defaults.max_stars_per_tile,
        "max_quads_per_tile": defaults.max_quads_per_tile,
    }
    if dense:
        kwargs.update(
            {
                "mag_cap": 15.0,
                "source_max_stars": 2000,
                "max_stars_per_tile": 2000,
                "max_quads_per_tile": 40000,
            }
        )
    config = Astap4DBuildConfig(**kwargs)
    t0 = time.perf_counter()
    build_4d_index_from_astap(astap_root, out_path, config=config, overwrite=True)
    summary = index_summary(out_path)
    summary["build_s"] = float(time.perf_counter() - t0)
    summary["build_config"] = asdict(config)
    return summary


def run_solve(label: str, source: Path, index_paths: tuple[Path, ...], work_dir: Path) -> dict[str, Any]:
    target = work_dir / label / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    cfg = _runtime.build_runtime_config(index_paths, accept_policy="best_within_budget", policy=_runtime.RuntimePolicy())
    t0 = time.perf_counter()
    result = solve_blind(target, Path("/home/tristan/zesolver_index"), config=cfg)
    wall_s = time.perf_counter() - t0
    stats = dict(result.stats or {})
    return {
        "label": label,
        "source": str(source),
        "success": bool(result.success),
        "message": str(result.message),
        "tile": result.tile_key,
        "wall_s": float(wall_s),
        "hits": int(stats.get("astrometry_4d_hits", 0) or 0),
        "tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "accepted": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "first_plausible_rank": stats.get("astrometry_4d_first_plausible_rank"),
        "first_accepted_rank": stats.get("astrometry_4d_first_accepted_rank"),
        "selected_rank": stats.get("astrometry_4d_selected_rank"),
        "inliers": stats.get("inliers"),
        "rms_px": stats.get("rms_px"),
        "pix_scale_arcsec": stats.get("pix_scale_arcsec"),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "quad_sources": stats.get("astrometry_4d_quad_source_stars"),
        "verification_sources": stats.get("astrometry_4d_verification_source_stars"),
        "image_quads": stats.get("astrometry_4d_image_quads"),
        "image_records": stats.get("astrometry_4d_image_records"),
        "lookup_s": stats.get("astrometry_4d_kd_lookup_s"),
        "validation_s": stats.get("astrometry_4d_validation_s"),
        "total_s": stats.get("astrometry_4d_total_s"),
        "reject_reason_counts": stats.get("astrometry_4d_reject_reason_counts") or {},
        "best_reject": stats.get("astrometry_4d_best_reject") or {},
        "hits_by_index": {
            Path(key).name: {
                "hits": value.get("hits"),
                "tile_count": len(value.get("tile_keys") or ()),
                "tiles": value.get("tile_keys") if len(value.get("tile_keys") or ()) <= 12 else None,
            }
            for key, value in dict(stats.get("astrometry_4d_hits_by_index") or {}).items()
        },
    }


def hit_audit(index_path: Path, source: Path, work_dir: Path) -> dict[str, Any]:
    target = work_dir / "hit_audit" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    cfg = _runtime.build_runtime_config([index_path], accept_policy="best_within_budget", policy=_runtime.RuntimePolicy(search_budget_s=1.0))
    result = solve_blind(target, Path("/home/tristan/zesolver_index"), config=cfg)
    stats = dict(result.stats or {})
    # Reuse runtime stats for image-record counts; full pre-truncation is computed directly below.
    from zeblindsolver.quad_code_diagnostic import build_astrometry_quad_records
    from zeblindsolver.quad_sampling import sample_quads
    from zeblindsolver.star_detect import detect_stars

    with fits.open(target, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    stars = detect_stars(data, k_sigma=float(cfg.detect_k_sigma), min_area=int(cfg.detect_min_area))
    stars = stars[: int(cfg.max_stars)]
    positions = np.column_stack((stars["x"], stars["y"])).astype(np.float64, copy=False)
    quads = sample_quads(stars, max_quads=int(cfg.max_quads), strategy=str(cfg.blind_astrometry_4d_image_strategy))
    records = build_astrometry_quad_records(quads, positions)
    index = Quad4DIndex.load(index_path)
    tol = float(cfg.blind_astrometry_4d_code_tol)
    per_quad_counts: list[int] = []
    first2000_tiles: Counter[str] = Counter()
    first2000_per_quad: Counter[int] = Counter()
    total_before_truncation = 0
    first2000 = []
    for image_record_index, record in enumerate(records):
        ids = index.tree.query_ball_point(np.asarray(record.code, dtype=np.float64), r=tol) if index.tree is not None else []
        per_quad_counts.append(len(ids))
        total_before_truncation += len(ids)
        ranked = sorted(
            ((int(idx), float(np.linalg.norm(np.asarray(record.code, dtype=np.float64) - index.codes_4d[int(idx)]))) for idx in ids),
            key=lambda item: item[1],
        )
        for catalog_record_index, distance in ranked[: int(cfg.blind_astrometry_4d_max_hits_per_image_quad)]:
            tile_idx = int(index.tile_key_indices[catalog_record_index])
            tile = index.tile_keys[tile_idx] if 0 <= tile_idx < len(index.tile_keys) else ""
            first2000.append((image_record_index, catalog_record_index, float(distance), str(tile)))
            if len(first2000) >= int(cfg.blind_astrometry_4d_max_hits):
                break
        if len(first2000) >= int(cfg.blind_astrometry_4d_max_hits):
            break
    for image_record_index, _cat, _distance, tile in first2000:
        first2000_tiles[tile] += 1
        first2000_per_quad[image_record_index] += 1
    return {
        "source": str(source),
        "index": str(index_path),
        "runtime_hits": stats.get("astrometry_4d_hits"),
        "image_records": len(records),
        "total_hits_before_truncation": int(total_before_truncation),
        "first2000_tile_distribution_top20": first2000_tiles.most_common(20),
        "first2000_image_quad_distribution_top20": first2000_per_quad.most_common(20),
        "per_image_quad_hit_count_top20": sorted(per_quad_counts, reverse=True)[:20],
        "per_image_quad_hit_count_nonzero": int(sum(1 for value in per_quad_counts if value > 0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--astap-root", type=Path, default=Path("/opt/astap"))
    parser.add_argument("--library-index", type=Path, default=Path("/home/tristan/ZeSolverCatalog/new/indexes/blind4d/d50_4d.npz"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-hit-audit", action="store_true")
    args = parser.parse_args(argv)
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tile_keys = tuple(dict.fromkeys(case[2] for case in DEFAULT_CASES))

    variants: dict[str, dict[str, Any]] = {
        "A_p1d3b_repo_oracle": {
            "topology": "six_targeted_indexes",
            "paths": tuple((ROOT / "indexes" / "astrometry_4d" / f"{tile}_S_q40000.npz").resolve() for tile in P1D3B_ORDER),
        },
        "B_manager_current_all_sky": {
            "topology": "all_sky_monolith",
            "paths": (args.library_index.expanduser().resolve(),),
        },
    }
    current_path = work_dir / "indexes" / "C_current_targeted_400_8000.npz"
    dense_path = work_dir / "indexes" / "D_p1d3b_targeted_2000_40000.npz"
    if args.skip_build and current_path.exists() and dense_path.exists():
        variants["C_manager_current_targeted"] = {
            "topology": "two_tile_targeted_monolith",
            "paths": (current_path,),
        }
        variants["D_p1d3b_targeted"] = {
            "topology": "two_tile_targeted_monolith",
            "paths": (dense_path,),
        }
    elif not args.skip_build:
        variants["C_manager_current_targeted"] = {
            "topology": "two_tile_targeted_monolith",
            "paths": (current_path,),
            "build": build_target_index(args.astap_root, current_path, tile_keys, dense=False),
        }
        variants["D_p1d3b_targeted"] = {
            "topology": "two_tile_targeted_monolith",
            "paths": (dense_path,),
            "build": build_target_index(args.astap_root, dense_path, tile_keys, dense=True),
        }

    for variant in variants.values():
        variant["indexes"] = [index_summary(path) for path in variant["paths"]]

    solves: dict[str, list[dict[str, Any]]] = {}
    for variant_name, variant in variants.items():
        solves[variant_name] = []
        for label, source, _tile in DEFAULT_CASES:
            solves[variant_name].append(run_solve(label, source, tuple(variant["paths"]), work_dir / "solves" / variant_name))

    audits = {}
    if not args.skip_hit_audit:
        audits = {
            label: hit_audit(args.library_index.expanduser().resolve(), source, work_dir)
            for label, source, _tile in DEFAULT_CASES
        }
    report = {
        "cases": [{"label": label, "expected_tile": tile, "fits": inspect_fits(source)} for label, source, tile in DEFAULT_CASES],
        "variants": variants,
        "solves": solves,
        "hit_audits": audits,
    }
    _write_json(work_dir / "s5b_diagnostic_report.json", report)
    print(json.dumps({"work_dir": str(work_dir), "report": str(work_dir / "s5b_diagnostic_report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
