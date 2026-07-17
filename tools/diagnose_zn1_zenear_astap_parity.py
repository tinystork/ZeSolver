#!/usr/bin/env python3
"""ZN1 diagnostic probe: ZeNear / ASTAP behavioral parity on the bounded M31 set.

This probe is diagnostic-only. It does not call ZeBlind and it does not change
ZeNear product thresholds. Runtime FITS copies keep instrumental hints (RA/DEC,
focal length, pixel size) and remove celestial WCS solution keys.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zeblindsolver.fits_utils import estimate_scale_and_fov
from zeblindsolver.metadata_solver import NearSolveConfig, solve_near


IMAGE_NAMES = [
    "Light_M 31_11_30.0s_IRCUT_20250922-230409.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230510.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230650.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230720.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230853.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231350.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231844.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231915.fit",
]

WCS_PREFIXES = (
    "CTYPE",
    "CRVAL",
    "CRPIX",
    "CD",
    "PC",
    "CDELT",
    "CROTA",
    "PV",
    "A_",
    "B_",
    "AP_",
    "BP_",
)
WCS_EXACT_KEYS = {
    "WCSAXES",
    "RADESYS",
    "EQUINOX",
    "LONPOLE",
    "LATPOLE",
    "SOLVED",
    "QUALITY",
    "RMSPX",
    "INLIERS",
    "REQINL",
    "TILE_ID",
    "SOLVMODE",
    "SOLVER",
    "NEAR_VER",
    "BLINDVER",
    "ASTAPVER",
    "PIXSCAL",
    "SEED_SCALE",
    "SEED_ROT",
    "SEED_PAR",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    t0 = time.perf_counter()
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "elapsed_s": time.perf_counter() - t0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": _text(exc.stdout),
            "stderr": _text(exc.stderr),
            "elapsed_s": time.perf_counter() - t0,
            "timeout": True,
        }


def parse_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "astropy"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", None))
        except Exception:
            versions[name] = None
    return versions


def git_info(repo: Path) -> dict[str, Any]:
    def one(args: list[str]) -> str:
        cp = subprocess.run(["git", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return (cp.stdout or cp.stderr).strip()

    return {
        "commit": one(["rev-parse", "HEAD"]),
        "branch": one(["branch", "--show-current"]),
        "status_short": one(["status", "--short"]),
    }


def astap_version(astap_bin: str) -> dict[str, Any]:
    cp = run_cmd([astap_bin, "-help"], timeout=10)
    text = (cp.get("stdout") or "") + (cp.get("stderr") or "")
    first = next((line for line in text.splitlines() if line.strip()), "")
    return {"binary": astap_bin, "help_first_line": first, "returncode": cp.get("returncode")}


def original_wcs_summary(path: Path) -> dict[str, Any]:
    h = fits.getheader(path)
    width = int(h.get("NAXIS1", 0) or 0)
    height = int(h.get("NAXIS2", 0) or 0)
    out: dict[str, Any] = {"has_celestial": False, "width": width, "height": height}
    try:
        w = WCS(h)
        out["has_celestial"] = bool(w.has_celestial)
        if w.has_celestial and width > 0 and height > 0:
            cx, cy = width / 2.0, height / 2.0
            ra_c, dec_c = w.pixel_to_world_values(cx, cy)
            corners = []
            for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
                ra, dec = w.pixel_to_world_values(float(x), float(y))
                corners.append({"x": x, "y": y, "ra_deg": float(ra), "dec_deg": float(dec)})
            out.update({"center_ra_deg": float(ra_c), "center_dec_deg": float(dec_c), "corners": corners})
            try:
                cd = np.asarray(w.pixel_scale_matrix, dtype=float)
                out["pixel_scale_arcsec"] = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
                out["cd_det"] = float(np.linalg.det(cd))
            except Exception:
                pass
    except Exception as exc:
        out["error"] = str(exc)
    return out


def strip_runtime_copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    removed: list[str] = []
    with fits.open(src, memmap=False) as hdul:
        h = hdul[0].header.copy()
        data = hdul[0].data
        for key in list(h.keys()):
            if key in WCS_EXACT_KEYS or any(key.startswith(prefix) for prefix in WCS_PREFIXES):
                removed.append(key)
                try:
                    del h[key]
                except Exception:
                    pass
        fits.writeto(dst, data, h, overwrite=True)
    return {"removed_keys": sorted(set(removed)), "runtime_sha256": sha256(dst)}


def hint_from_header(path: Path) -> dict[str, Any]:
    h = fits.getheader(path)
    width = int(h.get("NAXIS1", 0) or 0)
    height = int(h.get("NAXIS2", 0) or 0)
    ra_deg = float(h["RA"])
    dec_deg = float(h["DEC"])
    scale_arcsec, (fov_x, fov_y) = estimate_scale_and_fov(h, width, height)
    fov_deg = max(v for v in (fov_x, fov_y) if v is not None)
    return {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "ra_hours": ra_deg / 15.0,
        "spd_deg": dec_deg + 90.0,
        "scale_arcsec": scale_arcsec,
        "fov_deg": fov_deg,
        "width": width,
        "height": height,
        "focal_len": h.get("FOCALLEN") or h.get("FOCLEN") or h.get("FOCALLENGTH"),
        "xpixsz": h.get("XPIXSZ") or h.get("PIXSIZE1"),
        "ypixsz": h.get("YPIXSZ") or h.get("PIXSIZE2"),
    }


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


def parse_astap_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"(\d+)\s+stars,\s+(\d+)\s+quads selected in the image\.\s+(\d+)\s+database stars,\s+(\d+)\s+database quads", text)
    if m:
        out.update({
            "image_stars": int(m.group(1)),
            "image_quads": int(m.group(2)),
            "catalog_stars": int(m.group(3)),
            "catalog_quads": int(m.group(4)),
        })
    m = re.search(r"(\d+)\s+of\s+(\d+)\s+quads selected matching within\s+([0-9.]+)\s+tolerance", text)
    if m:
        out.update({"matched_quads": int(m.group(1)), "required_matches": int(m.group(2)), "quad_tolerance": float(m.group(3))})
    m = re.search(r"Solved in\s+([0-9.]+)\s+sec", text)
    if m:
        out["reported_solve_s"] = float(m.group(1))
    return out


def parse_astap_ini(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"exists": path.exists()}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "PLTSOLVD":
            out["success"] = (v.upper() == "T")
        elif k in {"CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CROTA1", "CROTA2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"}:
            try:
                out[k.lower()] = float(v)
            except Exception:
                out[k.lower()] = v
    return out


def run_astap_solve(runtime: Path, out_base: Path, hint: dict[str, Any], *, astap_bin: str, astap_db: Path, astap_family: str) -> dict[str, Any]:
    for ext in (".ini", ".wcs", ".log", ".csv"):
        try:
            out_base.with_suffix(ext).unlink()
        except FileNotFoundError:
            pass
    cmd = [
        astap_bin,
        "-f",
        str(runtime),
        "-ra",
        f"{float(hint['ra_hours']):.10f}",
        "-spd",
        f"{float(hint['spd_deg']):.10f}",
        "-fov",
        f"{float(hint['fov_deg']):.10f}",
        "-r",
        "3",
        "-d",
        str(astap_db),
        "-D",
        astap_family,
        "-wcs",
        "-log",
        "-o",
        str(out_base),
    ]
    cp = run_cmd(cmd, timeout=180)
    text = (cp.get("stdout") or "") + "\n" + (cp.get("stderr") or "")
    log_path = out_base.with_suffix(".log")
    if log_path.exists():
        text += "\n" + log_path.read_text(errors="ignore")
    ini = parse_astap_ini(out_base.with_suffix(".ini"))
    return {
        "success": bool(ini.get("success", False)),
        "elapsed_s": cp.get("elapsed_s"),
        "returncode": cp.get("returncode"),
        "cmd": cmd,
        "ini": ini,
        "log_metrics": parse_astap_text(text),
        "stdout_excerpt": (cp.get("stdout") or "")[:2000],
        "stderr_excerpt": (cp.get("stderr") or "")[:1000],
    }


def dump_astap_extract(runtime: Path, out_csv: Path, *, astap_bin: str) -> dict[str, Any]:
    tmp_csv = runtime.with_suffix(".csv")
    try:
        tmp_csv.unlink()
    except FileNotFoundError:
        pass
    cp = run_cmd([astap_bin, "-f", str(runtime), "-extract", "10"], timeout=180)
    result: dict[str, Any] = {"returncode": cp.get("returncode"), "elapsed_s": cp.get("elapsed_s"), "exists": tmp_csv.exists()}
    if not tmp_csv.exists():
        result["count"] = 0
        result["stdout_excerpt"] = (cp.get("stdout") or "")[:1000]
        result["stderr_excerpt"] = (cp.get("stderr") or "")[:1000]
        return result
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with tmp_csv.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("flux") or 0.0), reverse=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "x", "y", "flux", "hfd", "snr"])
        for i, row in enumerate(rows, start=1):
            writer.writerow([i, row.get("x"), row.get("y"), row.get("flux"), row.get("hfd"), row.get("snr")])
    result["count"] = len(rows)
    try:
        tmp_csv.unlink()
    except Exception:
        pass
    return result


def run_zenear(
    runtime: Path,
    index_root: Path,
    hint: dict[str, Any],
    *,
    variant: str,
    dump_dir: Path,
    label: str,
    rescue: bool = False,
    image_stars_csv: Path | None = None,
) -> dict[str, Any]:
    cfg = NearSolveConfig(
        family="d50",
        fov_override_deg=float(hint["fov_deg"]),
        detect_backend="cpu",
        detect_device=None,
        ransac_seed=0,
        astap_iso_strict=True,
        astap_hint_fastpath=False,
        log_level="INFO",
        diagnostic_dump_dir=str(dump_dir),
        diagnostic_dump_label=label,
        diagnostic_image_stars_csv=str(image_stars_csv) if image_stars_csv else None,
    )
    attempts = []
    t0 = time.perf_counter()
    sol = solve_near(runtime, index_root, config=cfg, cancel_check=lambda: False)
    attempts.append({"stage": "base", "success": sol.success, "message": sol.message, "stats": sol.stats, "tile_key": sol.tile_key})
    if rescue and not sol.success:
        cfg2 = NearSolveConfig(**{**asdict(cfg)})
        cfg2.search_margin = max(float(cfg.search_margin) * 1.6, float(cfg.search_margin) + 0.20)
        cfg2.max_tile_candidates = max(int(cfg.max_tile_candidates), 96)
        cfg2.quality_rms = max(float(cfg.quality_rms), 1.70)
        cfg2.diagnostic_dump_label = f"{label}_rescue"
        sol = solve_near(runtime, index_root, config=cfg2, cancel_check=lambda: False)
        attempts.append({"stage": "rescue", "success": sol.success, "message": sol.message, "stats": sol.stats, "tile_key": sol.tile_key})
    return {
        "variant": variant,
        "success": bool(sol.success),
        "message": sol.message,
        "stats": sol.stats,
        "tile_key": sol.tile_key,
        "attempts": attempts,
        "elapsed_s": time.perf_counter() - t0,
    }


def read_star_xy(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=float)
    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((float(row["x"]), float(row["y"])))
            except Exception:
                pass
    return np.asarray(rows, dtype=float)


def compare_image_lists(astap_csv: Path, zenear_csv: Path) -> dict[str, Any]:
    a = read_star_xy(astap_csv)
    z = read_star_xy(zenear_csv)
    out: dict[str, Any] = {"astap_count": int(a.shape[0]), "zenear_count": int(z.shape[0])}
    if a.size == 0 or z.size == 0:
        return out
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(z)
        for radius in (1.0, 2.0):
            d, _ = tree.query(a, k=1, distance_upper_bound=radius)
            out[f"overlap_{radius:.0f}px"] = int(np.isfinite(d).sum())
        for top in (50, 100, 200):
            aa = a[: min(top, len(a))]
            zz = z[: min(top, len(z))]
            tree_top = cKDTree(zz)
            d, _ = tree_top.query(aa, k=1, distance_upper_bound=2.0)
            out[f"top{top}_overlap_2px"] = int(np.isfinite(d).sum())
    except Exception as exc:
        out["comparison_error"] = str(exc)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_main_report(path: Path, baseline: dict[str, Any], reproduction: dict[str, Any], image_cmp: dict[str, Any]) -> None:
    lines = [
        "# ZN1 — Autopsie ZeNear / ASTAP",
        "",
        "## Résumé exécutif",
        "",
        "Baseline diagnostique créée sans modifier les seuils produit ZeNear ni ZeBlind. Les copies runtime retirent le WCS résolu et conservent les hints instrumentaux RA/DEC/focale/pixel.",
        "",
        "## Reproduction équitable",
        "",
    ]
    for variant in ("A0_zenear_native", "A1_zenear_rescue", "B0_astap_native"):
        runs = reproduction.get("runs", {}).get(variant, [])
        counts = []
        for run in runs:
            ok = sum(1 for item in run.get("images", []) if item.get("success"))
            counts.append(f"{ok}/8")
        lines.append(f"- `{variant}`: {', '.join(counts) if counts else 'non exécuté'}")
    lines += [
        "",
        "## Première divergence observée",
        "",
        "ASTAP est exécuté avec `-ra`, `-spd`, `-fov`, `-r 3`, `-d /opt/astap`, `-D d50`, `-wcs`, `-log` sur les mêmes copies runtime. ZeNear est exécuté via `solve_near` avec `family=d50`, `fov_override_deg`, `detect_backend=cpu`, `astap_iso_strict=True`, `ransac_seed=0`, sans fallback.",
        "",
        "Les dumps CSV ZeNear proviennent du point exact juste après sélection des étoiles image/catalogue. Les dumps ASTAP image utilisent `astap -extract 10`; la liste interne exacte du solveur ASTAP reste à instrumenter si une injection contrôlée est nécessaire.",
        "",
        "## Comparaison des étoiles image",
        "",
        "Voir `reports/zenear_zn1_image_star_comparison.json` et `reports/zn1_star_lists/`.",
        "",
        "## Matrices avancées",
        "",
        "Les matrices injection, signatures, transformation et classification sont initialisées mais restent `pending` tant que les listes internes ASTAP acceptées ne sont pas dumpées de façon instrumentée.",
        "",
        "## Questions obligatoires",
        "",
    ]
    q = reproduction.get("mandatory_answers", {})
    for key, value in q.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("/home/tristan/zemosaic/example/androtest"))
    ap.add_argument("--index-root", type=Path, default=Path("/home/tristan/zesolver_index"))
    ap.add_argument("--astap-bin", default="astap")
    ap.add_argument("--astap-db", type=Path, default=Path("/opt/astap"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    reports = (repo / args.reports_dir).resolve() if not args.reports_dir.is_absolute() else args.reports_dir
    runtime_root = reports / "zn1_runtime"
    oracle_root = reports / "zn1_oracle"
    star_root = reports / "zn1_star_lists"
    quad_root = reports / "zn1_quad_dumps"
    transform_root = reports / "zn1_transform_dumps"
    for d in (runtime_root, oracle_root, star_root, quad_root, transform_root):
        d.mkdir(parents=True, exist_ok=True)

    corpus = []
    for name in IMAGE_NAMES:
        src = args.source_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        stem = safe_stem(name)
        runtime = runtime_root / f"{stem}_runtime.fit"
        strip_info = strip_runtime_copy(src, runtime)
        hint = hint_from_header(runtime)
        oracle = original_wcs_summary(src)
        oracle_path = oracle_root / f"{stem}_oracle.json"
        write_json(oracle_path, {"source": str(src), "source_sha256": sha256(src), "oracle": oracle, "hint": hint})
        corpus.append({
            "name": name,
            "stem": stem,
            "source": str(src),
            "runtime": str(runtime),
            "source_sha256": sha256(src),
            "runtime_sha256": strip_info["runtime_sha256"],
            "removed_keys": strip_info["removed_keys"],
            "oracle_sidecar": str(oracle_path),
            "hint": hint,
        })

    baseline = {
        "git": git_info(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "packages": parse_package_versions(),
        "astap": astap_version(args.astap_bin),
        "astap_db": str(args.astap_db),
        "astap_family": "d50",
        "index_root": str(args.index_root),
        "corpus": corpus,
        "zenear_effective_config": asdict(NearSolveConfig(
            family="d50",
            detect_backend="cpu",
            ransac_seed=0,
            astap_iso_strict=True,
            astap_hint_fastpath=False,
        )),
    }
    write_json(reports / "zenear_zn1_baseline.json", baseline)

    reproduction: dict[str, Any] = {"runs": {"A0_zenear_native": [], "A1_zenear_rescue": [], "B0_astap_native": []}}
    image_cmp: dict[str, Any] = {}

    for run_idx in range(1, max(1, int(args.runs)) + 1):
        for variant in ("A0_zenear_native", "A1_zenear_rescue", "B0_astap_native"):
            run_record = {"run": run_idx, "images": []}
            for item in corpus:
                stem = item["stem"]
                src = Path(item["source"])
                runtime = runtime_root / f"{stem}_{variant}_run{run_idx}.fit"
                strip_runtime_copy(src, runtime)
                hint = hint_from_header(runtime)
                if variant.startswith("A"):
                    label = f"{stem}_{variant}_run{run_idx}"
                    res = run_zenear(
                        runtime,
                        args.index_root,
                        hint,
                        variant=variant,
                        dump_dir=star_root,
                        label=label,
                        rescue=(variant == "A1_zenear_rescue"),
                    )
                    run_record["images"].append({"name": item["name"], **res})
                else:
                    out_base = runtime_root / f"{stem}_{variant}_run{run_idx}_astap"
                    res = run_astap_solve(runtime, out_base, hint, astap_bin=args.astap_bin, astap_db=args.astap_db, astap_family="d50")
                    run_record["images"].append({"name": item["name"], **res})
                    if run_idx == 1:
                        astap_csv = star_root / f"{stem}_astap_image_stars.csv"
                        extract = dump_astap_extract(runtime, astap_csv, astap_bin=args.astap_bin)
                        z_csv = star_root / f"{stem}_A0_zenear_native_run1_zenear_image_stars.csv"
                        image_cmp[stem] = {
                            "astap_extract": extract,
                            "comparison_vs_zenear_A0_run1": compare_image_lists(astap_csv, z_csv),
                        }
            reproduction["runs"][variant].append(run_record)

    injection_matrix: dict[str, Any] = {
        "I0_baseline": {"source": "A0_zenear_native run1"},
        "I1_astap_image_stars_zenear_catalog": {"images": []},
    }
    for item in corpus:
        stem = item["stem"]
        src = Path(item["source"])
        runtime = runtime_root / f"{stem}_I1_astap_image_stars.fit"
        strip_runtime_copy(src, runtime)
        hint = hint_from_header(runtime)
        astap_csv = star_root / f"{stem}_astap_image_stars.csv"
        res = run_zenear(
            runtime,
            args.index_root,
            hint,
            variant="I1_astap_image_stars_zenear_catalog",
            dump_dir=star_root,
            label=f"{stem}_I1_astap_image_stars",
            rescue=False,
            image_stars_csv=astap_csv,
        )
        injection_matrix["I1_astap_image_stars_zenear_catalog"]["images"].append({"name": item["name"], **res})

    def success_counts(variant: str) -> list[str]:
        out = []
        for run in reproduction["runs"].get(variant, []):
            out.append(f"{sum(1 for item in run.get('images', []) if item.get('success'))}/8")
        return out

    reproduction["mandatory_answers"] = {
        "1_ASTAP_resout_copies_runtime_memes_hints": ", ".join(success_counts("B0_astap_native")),
        "2_ZeNear_echec_sequentiel_deterministe": ", ".join(success_counts("A0_zenear_native")),
        "3_etoiles_image_communes": "voir zenear_zn1_image_star_comparison.json",
        "4_classement_etoiles_communes": "comparaison top50/top100/top200 initialisee; correlation rang pending",
        "5_listes_catalogue_identiques": "pending instrumentation ASTAP interne",
        "8_ZeNear_reussit_avec_etoiles_ASTAP": f"{sum(1 for x in injection_matrix['I1_astap_image_stars_zenear_catalog']['images'] if x.get('success'))}/8 avec ASTAP -extract image stars",
        "17_message_similarity_transform_etage_reel": "premier etage observe: aucun motif ASTAP-ISO valide, iso_refs=0 dans near_debug",
        "18_rescue_actuel_bon_etage": "pending; A1 mesure si rescue courant change le resultat",
        "20_correction_ZN2": "pending apres injection controlee",
    }

    write_json(reports / "zenear_zn1_reproduction.json", reproduction)
    write_json(reports / "zenear_zn1_image_star_comparison.json", image_cmp)

    placeholders = {
        "zenear_zn1_catalog_star_comparison.json": {"status": "pending", "reason": "requires ASTAP internal catalog list dump"},
        "zenear_zn1_injection_matrix.json": injection_matrix,
        "zenear_zn1_quad_signature_parity.json": {"status": "pending", "reason": "requires accepted ASTAP motif trace"},
        "zenear_zn1_transform_parity.json": {"status": "pending", "reason": "requires accepted ASTAP correspondences"},
        "zenear_zn1_failure_classification.json": {"status": "pending", "classification": {item["stem"]: "unresolved" for item in corpus}},
    }
    for filename, obj in placeholders.items():
        write_json(reports / filename, obj)

    pipeline_map = reports / "zenear_zn1_astap_pipeline_map.md"
    pipeline_map.write_text(
        "\n".join([
            "# ZN1 ASTAP Pipeline Map",
            "",
            "- `ASTAP-main/command-line_version/astap_command_line.lpr`: CLI parsing (`-f`, `-r`, `-fov`, `-ra`, `-spd`, `-d`, `-D`, `-extract`, `-wcs`, `-log`).",
            "- `ASTAP-main/command-line_version/unit_command_line_solving.pas`: solve flow, image stars/quads, `find_fit`, `find_fit_using_hash`, offset/rotation and WCS update.",
            "- `ASTAP-main/command-line_version/unit_command_line_star_database.pas`: star database selection and tile database read.",
            "",
            "Detailed symbol-by-symbol mapping remains pending after baseline reproduction.",
        ]) + "\n",
        encoding="utf-8",
    )
    write_main_report(reports / "zenear_zn1_astap_parity_autopsy.md", baseline, reproduction, image_cmp)

    print(json.dumps({
        "baseline": str(reports / "zenear_zn1_baseline.json"),
        "reproduction": str(reports / "zenear_zn1_reproduction.json"),
        "summary": {
            "A0": success_counts("A0_zenear_native"),
            "A1": success_counts("A1_zenear_rescue"),
            "B0": success_counts("B0_astap_native"),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
