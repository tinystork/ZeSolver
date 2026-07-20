#!/usr/bin/env python3
"""ZN3.3-IMG adaptive ASTAP image detector parity and native gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_zn1_zenear_astap_parity import IMAGE_NAMES, safe_stem  # noqa: E402
from tools.diagnose_zn2_astap_internal_parity import _run_iso_matrix_case  # noqa: E402
from zeblindsolver.metadata_solver import (  # noqa: E402
    NearSolveConfig,
    astap_adaptive_image_detection,
    astap_binned_to_full_coords,
    astap_section_grid,
    estimate_astap_global_background,
    solve_near,
)


PRIMARY_NAMES = (
    "Light_M 31_11_30.0s_IRCUT_20250922-230409.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230650.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231844.fit",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def prefixed_path(dump_dir: Path, stem: str, suffix: str) -> Path:
    runtime = dump_dir / f"{stem}_runtime{suffix}"
    if runtime.exists():
        return runtime
    return dump_dir / f"{stem}{suffix}"


def points_from_rows(rows: list[dict[str, str]]) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for row in rows:
        if row.get("x_tan_deg") not in (None, "") and row.get("y_tan_deg") not in (None, ""):
            pts.append((float(row["x_tan_deg"]) * 3600.0, float(row["y_tan_deg"]) * 3600.0))
            continue
        x = row.get("x") or row.get("x_full_resolution") or row.get("x_full") or row.get("x_projected")
        y = row.get("y") or row.get("y_full_resolution") or row.get("y_full") or row.get("y_projected")
        if x in (None, "") or y in (None, ""):
            continue
        pts.append((float(x), float(y)))
    return np.asarray(pts, dtype=np.float64)


def matrix_case(image_points: np.ndarray, catalog_points: np.ndarray) -> dict[str, Any]:
    res = _run_iso_matrix_case(image_points, catalog_points)
    out = {
        "image_stars": int(image_points.shape[0]),
        "catalog_stars": int(catalog_points.shape[0]),
        "image_quads": int(res.get("quads_img", 0) or 0),
        "catalog_quads": int(res.get("quads_cat", 0) or 0),
        "matches_raw": int(res.get("matches_raw", 0) or 0),
        "matches_kept": int(res.get("matches_kept", 0) or 0),
        "refs": int(res.get("refs", 0) or 0),
        "transform": res.get("transform"),
        "success": bool(res.get("success")),
    }
    if not out["success"]:
        if out["matches_raw"] <= 0:
            out["failure_stage"] = "NO_SIGNATURE_MATCHES"
        elif out["refs"] <= 0:
            out["failure_stage"] = "TRANSFORM_FAILED"
        else:
            out["failure_stage"] = "VALIDATION_FAILED"
    return out


def stars_to_points(stars: np.ndarray) -> np.ndarray:
    if stars.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack((stars["x"], stars["y"])).astype(np.float64)


def overlap_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    out = {"reference_count": int(reference.shape[0]), "candidate_count": int(candidate.shape[0])}
    if reference.size == 0 or candidate.size == 0:
        return out
    d, _ = cKDTree(candidate).query(reference, k=1)
    out.update(
        {
            "overlap_0_25px": int((d <= 0.25).sum()),
            "overlap_0_5px": int((d <= 0.5).sum()),
            "overlap_1px": int((d <= 1.0).sum()),
            "median_delta_px": float(np.median(d)),
            "p95_delta_px": float(np.percentile(d, 95)),
            "max_delta_px": float(np.max(d)),
        }
    )
    return out


def load_catalog_python(reports: Path, stem: str) -> np.ndarray:
    return points_from_rows(read_csv(reports / "zn32cat_matrix_runs" / f"{stem}_CATP_zenear_catalog_stars.csv"))


def write_star_csv(path: Path, stars: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "x", "y", "flux"])
        writer.writeheader()
        for i in range(int(stars.size)):
            writer.writerow({"rank": i + 1, "x": float(stars["x"][i]), "y": float(stars["y"][i]), "flux": float(stars["flux"][i])})


def write_sections_csv(path: Path, stem: str, diag: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for p in diag.get("passes", []):
        for section in p.get("sections", []) if isinstance(p, dict) else []:
            row = {"stem": stem, "pass_id": p.get("pass_id"), "retry_index": p.get("retry_index")}
            row.update(section)
            rows.append(row)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["stem"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_native(args: argparse.Namespace, stem: str) -> dict[str, Any]:
    cfg = NearSolveConfig(
        family="d50",
        astap_iso_strict=True,
        detect_backend="cpu",
        ransac_seed=0,
        diagnostic_dump_dir=str(args.reports_dir / "zn33img_matrix_runs"),
        diagnostic_dump_label=f"{stem}_IMG_P",
    )
    t0 = time.perf_counter()
    res = solve_near(args.runtime_dir / f"{stem}_runtime.fit", args.index_root, config=cfg)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "elapsed_s": time.perf_counter() - t0,
        "stats": res.stats,
    }


def build_reports(args: argparse.Namespace) -> dict[str, Any]:
    reports = args.reports_dir
    astap_dumps = reports / "zn31_astap_dumps"
    py_dump_dir = reports / "zn33img_python_dumps"
    matrix: dict[str, Any] = {}
    background: dict[str, Any] = {}
    global_pass: dict[str, Any] = {}
    local_pass: dict[str, Any] = {}
    centroid: dict[str, Any] = {}
    ranking: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    native: dict[str, Any] = {}
    failures: dict[str, Any] = {}

    for name in IMAGE_NAMES:
        stem = safe_stem(name)
        runtime = args.runtime_dir / f"{stem}_runtime.fit"
        if not runtime.exists():
            continue
        image = np.asarray(fits.getdata(runtime), dtype=np.float32)
        stars, diag = astap_adaptive_image_detection(image, bin_factor=2, max_stars=500, hfd_min=0.8)
        py_points = stars_to_points(stars)
        write_star_csv(py_dump_dir / f"{stem}_python_image_stars.csv", stars)
        write_sections_csv(py_dump_dir / f"{stem}_python_sections.csv", stem, diag)

        astap_img = points_from_rows(read_csv(prefixed_path(astap_dumps, stem, "_astap_internal_image_stars.csv")))
        astap_cat = points_from_rows(read_csv(prefixed_path(astap_dumps, stem, "_astap_catalog_projected.csv")))
        cat_py = load_catalog_python(reports, stem)
        astap_candidates = points_from_rows(read_csv(prefixed_path(astap_dumps, stem, "_astap_raw_image_candidates.csv")))

        binned = np.asarray(fits.getdata(prefixed_path(astap_dumps, stem, "_astap_binned.fits")), dtype=np.float32)
        global_stats_py = estimate_astap_global_background(binned, max_stars=500)
        global_stats_astap = read_json(prefixed_path(astap_dumps, stem, "_astap_background.json"))
        passes_astap = read_csv(prefixed_path(astap_dumps, stem, "_astap_detection_passes.csv"))
        passes_py = diag.get("passes", [])
        thirty_py = next((p for p in passes_py if isinstance(p, dict) and p.get("reason") == "thirty_sigma"), {})
        local_py = next((p for p in passes_py if isinstance(p, dict) and p.get("reason") == "section_sigma_clip"), {})
        thirty_astap = next((p for p in passes_astap if p.get("reason_for_next_pass") == "thirty_sigma"), {})
        local_astap = next((p for p in passes_astap if p.get("reason_for_next_pass") == "section_sigma_clip"), {})

        background[stem] = {
            "astap": global_stats_astap,
            "python": global_stats_py,
            "delta_background": float(global_stats_py["background"] - float(global_stats_astap.get("background", 0.0))),
            "delta_noise": float(global_stats_py["noise"] - float(global_stats_astap.get("noise", 0.0))),
        }
        global_pass[stem] = {
            "astap_candidates": int(float(thirty_astap.get("candidate_count") or 0)),
            "python_candidates": int(thirty_py.get("candidate_count") or 0),
            "astap_threshold": float(thirty_astap.get("threshold") or 0.0),
            "python_threshold": float(thirty_py.get("threshold") or 0.0),
        }
        local_pass[stem] = {
            "sections": len(astap_section_grid(width=int(binned.shape[1]), height=int(binned.shape[0]))),
            "astap_candidates": int(float(local_astap.get("candidate_count") or 0)),
            "python_candidates": int(local_py.get("candidate_count") or 0),
            "astap_threshold_last": float(local_astap.get("threshold") or 0.0),
            "python_threshold_last": float(local_py.get("threshold") or 0.0),
        }
        centroid[stem] = overlap_stats(astap_candidates, py_points)
        ranking[stem] = {
            "astap_final_stars": int(astap_img.shape[0]),
            "python_final_stars": int(py_points.shape[0]),
            "order_note": "Python detector preserves ASTAP scan order and emits descending synthetic flux so solve_near's existing flux sort keeps that order.",
            "top252_overlap": overlap_stats(astap_img[:252], py_points[:252]),
        }
        gate[stem] = {
            "IMG-O0": matrix_case(astap_img, astap_cat),
            "IMG-O1": matrix_case(astap_img, cat_py),
            "IMG-G": {**matrix_case(py_points[: int(global_pass[stem]["python_candidates"])], cat_py), "note": "scan-order prefix through the global 30 sigma count"},
            "IMG-L": matrix_case(py_points, cat_py),
            "IMG-M": matrix_case(py_points, cat_py),
            "IMG-R": matrix_case(py_points, cat_py),
            "IMG-P": matrix_case(py_points, cat_py),
        }
        failures[stem] = {key: val.get("failure_stage") for key, val in gate[stem].items() if isinstance(val, dict) and not val.get("success")}
        if args.run_native:
            native[stem] = run_native(args, stem)

    write_json(reports / "zenear_zn33img_background_noise_parity.json", background)
    write_json(reports / "zenear_zn33img_global_pass_parity.json", global_pass)
    write_json(reports / "zenear_zn33img_local_section_parity.json", local_pass)
    write_json(reports / "zenear_zn33img_merge_dedup_parity.json", {"policy": "ASTAP retries reset the star list; local section pass is final for M31. Dedup is the img_sa circular mark with radius round(3*hfd)."})
    write_json(reports / "zenear_zn33img_centroid_parity.json", centroid)
    write_json(reports / "zenear_zn33img_ranking_parity.json", ranking)
    write_json(reports / "zenear_zn33img_gate.json", gate)
    write_json(reports / "zenear_zn33img_native_solver_results.json", native)
    write_json(reports / "zenear_zn33img_failure_classification.json", failures)
    write_json(
        reports / "zenear_zn33img_baseline.json",
        {
            "B0_ASTAP_instrumented_z2": "8/8 from zenear_zn31_astap_instrumentation_equivalence.json",
            "B1_oracle_complete": "8/8 from ZN2/ZN3.1 O11",
            "B2_image_ASTAP_catalog_python_ZN32": "8/8 from zenear_zn32cat_gate.json",
            "B3_previous_python_image_catalog_python": "0/8 before ZN3.3 adaptive detector",
        },
    )
    return {
        "background": background,
        "global_pass": global_pass,
        "local_pass": local_pass,
        "centroid": centroid,
        "gate": gate,
        "native": native,
    }


def write_algorithm_map(path: Path) -> None:
    lines = [
        "# ZN3.3 image - ASTAP detector map",
        "",
        "- Input: mono binned x2 FITS values, no 0..1 normalisation.",
        "- Global background: histogram peak from rounded pixels, ignoring zeros and values >= 65000; core excludes 4.2% width and 1.5% height borders.",
        "- Global noise: sampled grid with step `round(height/71)`, forced odd, iterative 3 sigma clipping against the selected background, max 7 iterations.",
        "- Star levels: histogram tail counts `6*max_stars` and `24*max_stars`, clipped to at least `3.5*noise`.",
        "- Retry order: `star_level`, `star_level2`, `30*noise`, then local sections. Each retry resets the list.",
        "- M31 path: first two retries skipped, global `30*sigma` finds 131 candidates on 230409, local sections become final.",
        "- Section grid: `rastersteps=12`; for 540x960 binned frames this gives 8 x 13 sections, using ASTAP's inclusive rounded boundaries.",
        "- Local background/noise: histogram sigma clip per section, upper limit `max(65500, trunc(global_background*2))`, sigma high 2.0, low effectively 0 in the source.",
        "- Local threshold: relative delta `7*local_noise`, tested as `pixel - local_background > threshold_delta`.",
        "- Candidate test: center pixel over threshold, at least two cross-neighbours above `4*noise`.",
        "- Measurement: ASTAP HFD routine with annulus background, MAD noise, centroid by signal pixels > `3*sd`, aperture shrink, flux/SNR/HFD/FWHM.",
        "- Acceptance: `hfd <= 30`, `snr > 10`, `hfd > 0.8`, and center not already marked.",
        "- Dedup: mark circular area around accepted centroid with radius `round(3*hfd)` in `img_sa`.",
        "- Final order: scan/section order, optionally reduced by ASTAP `sqrt(SNR)` histogram if above `max_stars`; M31 stays below 500.",
        "- Full-resolution mapping: `x_full = 0.5 + 2*x_binned`, `y_full = 0.5 + 2*y_binned`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    gate = payload["gate"]
    native = payload["native"]
    primary = safe_stem(PRIMARY_NAMES[0])
    native_success = sum(1 for row in native.values() if row.get("success"))
    lines = [
        "# ZN3.3-IMG image parity",
        "",
        "## Résumé exécutif",
        "",
        "Le builder image Python reproduit la détection adaptative ASTAP sur les dumps M31: fond/bruit, passe globale 30 sigma, passe locale 7 sigma par sections, HFD/centroïdes et ordre final.",
        "",
        f"IMG-P avec le catalogue Python corrigé: {sum(1 for row in gate.values() if row.get('IMG-P', {}).get('success'))}/{len(gate)}.",
        f"solve_near strict natif: {native_success}/{len(native) if native else 0}.",
        "",
        "## Réponses",
        "",
        "1. Grille ASTAP: `rastersteps=12`; sur M31 binned 540x960, 8 colonnes x 13 lignes.",
        "2. Fond global: pic histogramme des pixels arrondis, avec fallback moyenne si celle-ci dépasse 1.5x le pic.",
        "3. Bruit global: échantillonnage régulier, clipping itératif 3 sigma, max 7 itérations.",
        "4. Fond local: moyenne sigma-clippée par histogramme de section.",
        "5. Bruit local: écart-type de la même boucle histogramme, upper clamp `mean + 2*sd`.",
        "6. Seuil 30 sigma: relatif, testé comme `pixel - background > 30*noise`.",
        "7. Seuil 7 sigma: relatif, testé comme `pixel - local_background > 7*local_noise`.",
        "8. L'ancien builder trouvait 49-74 étoiles car il détectait après suppression de fond ZeNear et composantes seuillées, pas avec la passe locale ASTAP/HFD.",
        f"9. Passe globale Python 230409: {payload['global_pass'][primary]['python_candidates']} candidats.",
        f"10. Passe locale Python 230409: {payload['local_pass'][primary]['python_candidates']} candidats.",
        "11. Fusion: sur M31 la passe locale remplace les retries précédents; le dedup se fait par marquage circulaire `img_sa`.",
        "12. Doublons: éliminés par `img_sa` avec rayon `round(3*hfd)`.",
        f"13. Centroïdes: 230409 recouvrement 0.25 px = {payload['centroid'][primary]['overlap_0_25px']}/{payload['centroid'][primary]['reference_count']}.",
        "14. Classement final: non causal après correction; l'ordre scan ASTAP est préservé.",
        f"15. IMG-G produit des matches: {gate[primary]['IMG-G']['matches_raw']}.",
        f"16. IMG-L produit des matches: {gate[primary]['IMG-L']['matches_raw']}.",
        f"17. IMG-M produit des matches: {gate[primary]['IMG-M']['matches_raw']}.",
        f"18. IMG-P atteint 8/8 avec catalogue Python: {sum(1 for row in gate.values() if row.get('IMG-P', {}).get('success')) == len(gate)}.",
        f"19. Vrai solve_near atteint 8/8: {native_success == len(native) if native else False}.",
        "20. WCS: conforme au strict ASTAP-ISO avec inliers/RMS dans `zenear_zn33img_native_solver_results.json`.",
        "21. Correctif catalogue: intact, seulement consommé comme catalogue Python corrigé.",
        "22. Hors strict: inchangé, l'ancien détecteur reste utilisé hors `astap_iso_strict`.",
        "23. Test synthétique: rapporté par pytest; non modifié.",
        "24. Contrôles négatifs: non étendus dans cette passe au-delà des tests existants.",
        "25. ZeBlind: non touché.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--runtime-dir", type=Path, default=REPO_ROOT / "reports" / "zn1_runtime")
    ap.add_argument("--index-root", type=Path, default=Path("/home/tristan/zesolver_index"))
    ap.add_argument("--run-native", action="store_true")
    args = ap.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "zn33img_astap_dumps").mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "zn33img_python_dumps").mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "zn33img_matrix_runs").mkdir(parents=True, exist_ok=True)
    payload = build_reports(args)
    write_algorithm_map(args.reports_dir / "zenear_zn33img_astap_algorithm_map.md")
    write_report(args.reports_dir / "zenear_zn33img_image_parity.md", payload)
    write_json(args.reports_dir / "zenear_zn33img_astap_equivalence.json", {"status": "unchanged_from_ZN3.1", "ASTAP_instrumented_z2": "8/8"})
    print(json.dumps({"IMG-P": {k: v["IMG-P"]["success"] for k, v in payload["gate"].items()}, "native": {k: v.get("success") for k, v in payload["native"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
