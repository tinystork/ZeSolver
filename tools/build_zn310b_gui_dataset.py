#!/usr/bin/env python3
"""Build the ZN3.10B hybrid GUI dataset.

The builder never edits the source directory.  It copies selected FITS files to
a timestamped sibling directory, strips stale WCS cards, creates CONTROL,
NOHINT and BADHINT variants with identical pixels, and writes the ZN3.10B
reports used by the manual GUI fallback test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.index_manifest_4d import load_4d_index_manifest  # noqa: E402
from zeblindsolver.quad_index_builder import load_manifest as load_near_manifest, select_tiles_in_cone  # noqa: E402
from tools.astap_zn2_build_and_compare import run_astap  # noqa: E402
from zesolver.blind4d_runtime import resolve_default_4d_manifest_path  # noqa: E402


SOURCE_DIR = Path("/home/tristan/near_bench_cmp30/thread4")
OUT_PARENT = Path("/home/tristan/near_bench_cmp30")
REPORTS = ROOT / "reports"
NEAR_INDEX_ROOT = Path("/home/tristan/zesolver_index")
ASTAP_BIN = ROOT / "ASTAP-main" / "command-line_version" / "astap_cli"
ASTAP_DB = Path("/opt/astap")
HINT_RA_KEYS = ("RA", "OBJCTRA", "OBJRA", "OBJ_RA", "TELRA", "CENTRA", "CRVAL1")
HINT_DEC_KEYS = ("DEC", "OBJCTDEC", "OBJDEC", "OBJ_DEC", "TELDEC", "CENTDEC", "CRVAL2")
NEUTRAL_HINT_KEYS = (*HINT_RA_KEYS, *HINT_DEC_KEYS, "OBJECT", "WCSNAME")
SOLVE_TRACE_KEYS = (
    "SOLVED",
    "QUALITY",
    "NEAR_VER",
    "RMSPX",
    "INLIERS",
    "REQINL",
    "TILE_ID",
    "SOLVMODE",
    "SOLVER",
    "SEED_SCALE",
    "SEED_ROT",
    "SEED_PAR",
    "PIXSCAL",
    "NEARTIME",
)
WCS_KEYS = {
    "WCSAXES",
    "WCSNAME",
    "CTYPE1",
    "CTYPE2",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
    "CDELT1",
    "CDELT2",
    "CROTA1",
    "CROTA2",
    "LONPOLE",
    "LATPOLE",
    "RADESYS",
    "RADECSYS",
    "EQUINOX",
    "A_ORDER",
    "B_ORDER",
    "AP_ORDER",
    "BP_ORDER",
}
WCS_PREFIXES = ("A_", "B_", "AP_", "BP_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_md(path: Path, title: str, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True, default=str)}\n```\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_digest(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data)
        h = hashlib.sha256(np.ascontiguousarray(arr).tobytes())
        return {
            "pixel_sha256": h.hexdigest(),
            "dtype": str(arr.dtype),
            "shape": [int(v) for v in arr.shape],
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "median": float(np.nanmedian(arr)) if arr.size else None,
            "decompressed_bytes_sha256": h.hexdigest(),
        }


def has_wcs(header: fits.Header) -> bool:
    try:
        return bool(WCS(header).has_celestial)
    except Exception:
        return False


def strip_solution_cards(header: fits.Header) -> list[str]:
    removed: list[str] = []
    for key in list(header.keys()):
        if key in WCS_KEYS or key in SOLVE_TRACE_KEYS or any(key.startswith(prefix) for prefix in WCS_PREFIXES):
            removed.append(key)
            del header[key]
    return removed


def remove_keys(header: fits.Header, keys: tuple[str, ...]) -> list[str]:
    removed: list[str] = []
    for key in keys:
        if key in header:
            removed.append(key)
            del header[key]
    return removed


def copy_variant(src: Path, dst: Path, *, variant: str, wrong_center: tuple[float, float] | None = None) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(src, memmap=False) as hdul:
        data = np.asarray(hdul[0].data).copy()
        header = hdul[0].header.copy()
    removed_solution = strip_solution_cards(header)
    removed_hints: list[str] = []
    if variant == "no_hints":
        removed_hints = remove_keys(header, NEUTRAL_HINT_KEYS)
        header["OBJECT"] = "ZN310B_NOHINT"
        del header["OBJECT"]
    elif variant == "wrong_hints":
        removed_hints = remove_keys(header, (*HINT_RA_KEYS, *HINT_DEC_KEYS))
        ra, dec = wrong_center if wrong_center is not None else (0.0, 0.0)
        for key in HINT_RA_KEYS:
            if key != "CRVAL1":
                header[key] = float(ra)
        for key in HINT_DEC_KEYS:
            if key != "CRVAL2":
                header[key] = float(dec)
        header["OBJECT"] = "ZN310B_BADHINT"
    fits.PrimaryHDU(data=data, header=header).writeto(dst, overwrite=True)
    written_header = fits.getheader(dst)
    if has_wcs(written_header):
        raise SystemExit(f"ZN3.10B contract failure: WCS survived in {dst}")
    if variant == "no_hints":
        forbidden = [key for key in (*HINT_RA_KEYS, *HINT_DEC_KEYS, "OBJECT") if key in written_header]
        if forbidden:
            raise SystemExit(f"ZN3.10B contract failure: no_hints keys survived in {dst}: {forbidden}")
    return {
        "path": str(dst),
        "variant": variant,
        "removed_solution_keys": removed_solution,
        "removed_hint_keys": removed_hints,
        "file_sha256": sha256_file(dst),
        **pixel_digest(dst),
        "wcs_present": has_wcs(fits.getheader(dst)),
    }


def object_group(path: Path, header: fits.Header) -> str:
    text = f"{path.name} {header.get('OBJECT', '')}".upper()
    if "M 106" in text or "M106" in text:
        return "M106"
    if "M 31" in text or "M31" in text:
        return "M31"
    if "NGC 6888" in text or "NGC6888" in text:
        return "NGC6888"
    if "NGC 3628" in text or "NGC3628" in text:
        return "NGC3628"
    return "other"


def center_from_oracle(oracle: dict[str, Any]) -> tuple[float, float] | None:
    wcs = oracle.get("wcs") or {}
    crval = wcs.get("crval")
    if isinstance(crval, list) and len(crval) >= 2:
        return float(crval[0]), float(crval[1])
    ini = (oracle.get("result") or {}).get("ini") or {}
    if ini.get("crval1") is not None and ini.get("crval2") is not None:
        return float(ini["crval1"]), float(ini["crval2"])
    return None


def wcs_payload_from_oracle(oracle: dict[str, Any]) -> dict[str, Any]:
    ini = (oracle.get("result") or {}).get("ini") or {}
    center = center_from_oracle(oracle)
    cd = None
    if all(ini.get(k) is not None for k in ("cd1_1", "cd1_2", "cd2_1", "cd2_2")):
        cd = [[float(ini["cd1_1"]), float(ini["cd1_2"])], [float(ini["cd2_1"]), float(ini["cd2_2"])]]
    scale = None
    if cd is not None:
        scale = float(math.sqrt(abs(np.linalg.det(np.asarray(cd, dtype=float)))) * 3600.0)
    return {
        "classification": "ASTAP_ORACLE_VALID",
        "center_ra_deg": center[0] if center else None,
        "center_dec_deg": center[1] if center else None,
        "scale_arcsec": scale,
        "rotation_deg": ini.get("crota1"),
        "cd": cd,
        "origin": "ZN3.9 ASTAP oracle matched by source SHA256",
        "raw": oracle,
    }


def wrong_hint(center: tuple[float, float]) -> tuple[float, float]:
    ra, dec = center
    wrong_ra = (float(ra) + 123.0) % 360.0
    wrong_dec = max(-70.0, min(70.0, -float(dec)))
    return wrong_ra, wrong_dec


def source_inventory(source_dir: Path, manifest_4d: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    z39_manifest = json.loads((REPORTS / "zenear_zn39_corpus_manifest.json").read_text(encoding="utf-8"))
    z39_oracles = json.loads((REPORTS / "zenear_zn39_astap_oracles.json").read_text(encoding="utf-8"))
    z39_near = json.loads((REPORTS / "zenear_zn39_near_replay.json").read_text(encoding="utf-8"))
    oracle_by_sha: dict[str, dict[str, Any]] = {}
    near_by_sha: dict[str, dict[str, Any]] = {}
    for row in z39_manifest:
        cid = row["case_id"]
        sha = row["SHA256"]
        if cid in z39_oracles:
            oracle_by_sha[sha] = z39_oracles[cid]
        if cid in z39_near:
            near_by_sha[sha] = z39_near[cid]
    near_manifest = load_near_manifest(NEAR_INDEX_ROOT)
    loaded_4d = load_4d_index_manifest(manifest_4d)
    covered_tiles = set(loaded_4d.tile_keys)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(source_dir.glob("*.fit")):
        sha = sha256_file(path)
        if sha in seen:
            continue
        seen.add(sha)
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header
            arr = np.asarray(hdul[0].data)
            group = object_group(path, header)
            oracle = oracle_by_sha.get(sha)
            oracle_origin = "ZN3.9_SHA_MATCH"
            if not (oracle and oracle.get("classification") == "ASTAP_ORACLE_VALID"):
                work = REPORTS / "zn310b_oracle_work" / path.stem
                clean = work / "clean" / path.name
                clean.parent.mkdir(parents=True, exist_ok=True)
                copy_variant(path, clean, variant="control_clean")
                out_base = work / "astap" / path.stem
                out_base.parent.mkdir(parents=True, exist_ok=True)
                rec = run_astap(str(ASTAP_BIN), clean, out_base, ASTAP_DB, "d50", ["-z", "2"])
                ini = rec.get("ini") or {}
                if rec.get("success") and ini.get("crval1") is not None and ini.get("crval2") is not None:
                    oracle = {
                        "classification": "ASTAP_ORACLE_VALID",
                        "result": rec,
                        "wcs": {"classification": "WCS_READABLE", "crval": [float(ini["crval1"]), float(ini["crval2"])]},
                    }
                    oracle_origin = "ZN310B_ASTAP_CLEAN_BRANCH"
                    oracle_by_sha[sha] = oracle
            center = center_from_oracle(oracle or {})
            tiles = []
            covered = False
            if center:
                idxs = select_tiles_in_cone(near_manifest, center[0], center[1], 2.0)
                tiles = [str((near_manifest.get("tiles") or [])[i].get("tile_key")) for i in idxs]
                covered = bool(set(tiles) & covered_tiles)
            rows.append(
                {
                    "source_path": str(path),
                    "SHA256": sha,
                    "dimensions": [int(header.get("NAXIS1", 0) or 0), int(header.get("NAXIS2", 0) or 0)],
                    "dtype": str(arr.dtype),
                    "OBJECT": header.get("OBJECT"),
                    "group": group,
                    "RA": header.get("RA"),
                    "DEC": header.get("DEC"),
                    "OBJCTRA": header.get("OBJCTRA"),
                    "OBJCTDEC": header.get("OBJCTDEC"),
                    "WCS_existing": has_wcs(header),
                    "FOCALLEN": header.get("FOCALLEN"),
                    "XPIXSZ": header.get("XPIXSZ"),
                    "YPIXSZ": header.get("YPIXSZ"),
                    "estimated_scale_arcsec": None,
                    "oracle_ASTAP_available": bool(oracle and oracle.get("classification") == "ASTAP_ORACLE_VALID"),
                    "oracle_origin": oracle_origin if oracle and oracle.get("classification") == "ASTAP_ORACLE_VALID" else None,
                    "near_ZN39_success": bool((near_by_sha.get(sha) or {}).get("success")),
                    "coverage_4d_offline": "COVERED" if covered else "NOT_COVERED",
                    "coverage_tiles": tiles,
                    "covered_4d_tiles": sorted(set(tiles) & covered_tiles),
                    **pixel_digest(path),
                }
            )
    return rows, oracle_by_sha


def select_sources(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [r for r in rows if r["oracle_ASTAP_available"] and r["coverage_4d_offline"] == "COVERED"]
    plan = [("M31", 3), ("M106", 3), ("NGC6888", 2), ("NGC3628", 1)]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group, n in plan:
        for row in [r for r in eligible if r["group"] == group][:n]:
            selected.append(row)
            seen.add(row["SHA256"])
    for row in eligible:
        if len(selected) >= count:
            break
        if row["SHA256"] not in seen:
            selected.append(row)
            seen.add(row["SHA256"])
    return selected[:count]


def make_output_dir(parent: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = parent / f"zn310b_gui_fallback4d_{stamp}"
    out = base
    i = 1
    while out.exists():
        i += 1
        out = parent / f"{base.name}_{i:02d}"
    for name in ("source_manifest", "oracle_sidecars", "control_clean", "no_hints", "wrong_hints", "gui_mixed", "logs", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    manifest_4d = Path(args.blind_4d_manifest or resolve_default_4d_manifest_path()).expanduser().resolve()
    source_rows, oracle_by_sha = source_inventory(args.source_dir, manifest_4d)
    selected = select_sources(source_rows, int(args.count))
    if len(selected) < 6:
        raise SystemExit(f"not enough covered sources selected: {len(selected)}")
    out = make_output_dir(args.output_parent)
    write_json(REPORTS / "zenear_zn310b_source_inventory.json", {"source_dir": str(args.source_dir), "selected_run_dir": str(out), "items": source_rows, "selected": selected})
    write_md(REPORTS / "zenear_zn310b_source_inventory.md", "ZN3.10B source inventory", {"selected_run_dir": str(out), "selected": selected})
    shutil.copy2(REPORTS / "zenear_zn310b_source_inventory.json", out / "source_manifest" / "source_inventory.json")

    pixel_integrity: dict[str, Any] = {}
    oracle_rows: dict[str, Any] = {}
    gui_entries: list[dict[str, Any]] = []
    variants: dict[str, dict[str, dict[str, Any]]] = {}
    for idx, row in enumerate(selected, start=1):
        src = Path(row["source_path"])
        sha = row["SHA256"]
        oracle = wcs_payload_from_oracle(oracle_by_sha[sha])
        center = (float(oracle["center_ra_deg"]), float(oracle["center_dec_deg"]))
        case_id = f"ZN310B_{idx:03d}_{row['group']}"
        sidecar = out / "oracle_sidecars" / f"{case_id}.json"
        oracle_rows[case_id] = {"case_id": case_id, "source_SHA256": sha, "source_path": str(src), **oracle, "sidecar": str(sidecar)}
        write_json(sidecar, oracle_rows[case_id])
        control = copy_variant(src, out / "control_clean" / f"ZN310B_CONTROL_{idx:03d}.fit", variant="control_clean")
        nohint = copy_variant(src, out / "no_hints" / f"ZN310B_NOHINT_{idx:03d}.fit", variant="no_hints")
        badhint = copy_variant(src, out / "wrong_hints" / f"ZN310B_BADHINT_{idx:03d}.fit", variant="wrong_hints", wrong_center=wrong_hint(center))
        variants[case_id] = {"control_clean": control, "no_hints": nohint, "wrong_hints": badhint}
        pix = {name: data["pixel_sha256"] for name, data in variants[case_id].items()}
        pixel_integrity[case_id] = {"case_id": case_id, "source_pixel_sha256": row["pixel_sha256"], "variants": variants[case_id], "pixels_identical": len(set(pix.values()) | {row["pixel_sha256"]}) == 1}

    # Fixed, readable GUI mix: 3 controls, 3 no-hints, 2 bad-hints.
    mix_plan = [("CONTROL", "control_clean", 1), ("CONTROL", "control_clean", 2), ("CONTROL", "control_clean", 3), ("NOHINT", "no_hints", 4), ("NOHINT", "no_hints", 5), ("NOHINT", "no_hints", 6), ("BADHINT", "wrong_hints", 7), ("BADHINT", "wrong_hints", 8)]
    for gui_i, (label, variant_key, selected_idx) in enumerate(mix_plan, start=1):
        case_id = list(variants.keys())[selected_idx - 1]
        src_variant = Path(variants[case_id][variant_key]["path"])
        gui_name = f"ZN310B_{label}_{gui_i:03d}.fit"
        gui_path = out / "gui_mixed" / gui_name
        shutil.copy2(src_variant, gui_path)
        expected_near = label == "CONTROL"
        gui_entries.append(
            {
                "gui_filename": gui_name,
                "gui_path": str(gui_path),
                "variant": label,
                "source_SHA256": oracle_rows[case_id]["source_SHA256"],
                "source_path": oracle_rows[case_id]["source_path"],
                "case_id": case_id,
                "oracle_sidecar": oracle_rows[case_id]["sidecar"],
                "expected_near": expected_near,
                "expected_blind4d": not expected_near,
                "expected_final_backend": "NEAR" if expected_near else "BLIND4D",
            }
        )

    if not all(row["pixels_identical"] for row in pixel_integrity.values()):
        raise SystemExit("pixel integrity failure")
    gui_manifest = {"run_dir": str(out), "gui_mixed": str(out / "gui_mixed"), "items": gui_entries}
    write_json(REPORTS / "zenear_zn310b_pixel_integrity.json", pixel_integrity)
    write_json(REPORTS / "zenear_zn310b_oracles.json", oracle_rows)
    write_json(REPORTS / "zenear_zn310b_gui_manifest.json", gui_manifest)
    write_md(REPORTS / "zenear_zn310b_gui_manifest.md", "ZN3.10B GUI manifest", gui_manifest)
    for name in ("zenear_zn310b_pixel_integrity.json", "zenear_zn310b_oracles.json", "zenear_zn310b_gui_manifest.json", "zenear_zn310b_gui_manifest.md"):
        shutil.copy2(REPORTS / name, out / "reports" / name)
    summary = {
        "run_dir": str(out),
        "selected_count": len(selected),
        "groups": {g: sum(1 for r in selected if r["group"] == g) for g in sorted({r["group"] for r in selected})},
        "all_pixels_identical": all(row["pixels_identical"] for row in pixel_integrity.values()),
        "all_sources_covered_4d": all(row["coverage_4d_offline"] == "COVERED" for row in selected),
        "gui_manifest": str(REPORTS / "zenear_zn310b_gui_manifest.json"),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    ap.add_argument("--output-parent", type=Path, default=OUT_PARENT)
    ap.add_argument("--blind-4d-manifest", type=Path, default=None)
    ap.add_argument("--count", type=int, default=8)
    args = ap.parse_args(argv)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
