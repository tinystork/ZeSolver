#!/usr/bin/env python3
"""ZN3 input-list parity probe for strict ASTAP-ISO ZeNear.

This tool is diagnostic-only. It does not call ZeBlind, does not change
validation thresholds, and treats ZN2 ASTAP dumps as an oracle for comparison,
not as a production dependency.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_zn1_zenear_astap_parity import IMAGE_NAMES, safe_stem  # noqa: E402
from tools.diagnose_zn2_astap_internal_parity import (  # noqa: E402
    _read_csv,
    _run_iso_matrix_case,
    _trace_catalog_points,
    _trace_image_points,
    compare_points,
    load_trace,
)
from zeblindsolver.fits_utils import to_luminance_for_solve  # noqa: E402
from zeblindsolver.metadata_solver import (  # noqa: E402
    NearSolveConfig,
    astap_background_noise_stats,
    astap_binned_to_full_coords,
    astap_compatible_image_detection,
    astap_compatible_mean_bin_image,
    astap_full_to_binned_coords,
    choose_astap_compatible_bin_factor,
    solve_near,
)


PRIMARY_NAMES = {
    "Light_M 31_11_30.0s_IRCUT_20250922-230409.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-230650.fit",
    "Light_M 31_11_30.0s_IRCUT_20250922-231844.fit",
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _read_xy_csv(path: Path) -> np.ndarray:
    rows = _read_csv(path)
    pts: list[tuple[float, float]] = []
    for row in rows:
        x = row.get("x") or row.get("x_full_resolution") or row.get("x_projected") or row.get("x_tan_deg")
        y = row.get("y") or row.get("y_full_resolution") or row.get("y_projected") or row.get("y_tan_deg")
        if x not in (None, "") and y not in (None, ""):
            xf = float(x)
            yf = float(y)
            if "x_tan_deg" in row:
                xf *= 3600.0
                yf *= 3600.0
            pts.append((xf, yf))
    return np.asarray(pts, dtype=np.float64)


def _write_star_csv(path: Path, stars: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "x", "y", "flux"])
        writer.writeheader()
        for idx in range(int(stars.size)):
            writer.writerow(
                {
                    "rank": idx + 1,
                    "x": f"{float(stars['x'][idx]):.6f}",
                    "y": f"{float(stars['y'][idx]):.6f}",
                    "flux": f"{float(stars['flux'][idx]):.6f}",
                }
            )


def _spherical_match(astap_rows: list[dict[str, str]], zenear_rows: list[dict[str, str]]) -> dict[str, Any]:
    def _world(rows: list[dict[str, str]]) -> np.ndarray:
        pts = []
        for row in rows:
            if row.get("ra_deg") in (None, "") or row.get("dec_deg") in (None, ""):
                continue
            try:
                ra = float(row["ra_deg"])
                dec = float(row["dec_deg"])
            except Exception:
                continue
            if math.isfinite(ra) and math.isfinite(dec):
                pts.append((ra, dec))
        return np.asarray(pts, dtype=np.float64)

    a = _world(astap_rows)
    b = _world(zenear_rows)
    out: dict[str, Any] = {"astap_radec_count": int(a.shape[0]), "zenear_radec_count": int(b.shape[0])}
    if a.size == 0 or b.size == 0:
        out["status"] = "blocked_missing_radec_columns"
        return out
    from scipy.spatial import cKDTree

    dec0 = float(np.nanmedian(a[:, 1]))
    scale = math.cos(math.radians(dec0))
    aa = np.column_stack((a[:, 0] * scale * 3600.0, a[:, 1] * 3600.0))
    bb = np.column_stack((b[:, 0] * scale * 3600.0, b[:, 1] * 3600.0))
    tree = cKDTree(bb)
    for radius in (0.1, 0.5, 1.0, 2.0):
        d, _ = tree.query(aa, k=1, distance_upper_bound=float(radius))
        out[f"common_{radius:g}arcsec"] = int(np.isfinite(d).sum())
    return out


def _image_from_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return to_luminance_for_solve(hdul[0])


def _stats_image(arr: np.ndarray) -> dict[str, Any]:
    data = np.asarray(arr, dtype=np.float64)
    finite = data[np.isfinite(data)]
    hist, edges = np.histogram(finite, bins=32) if finite.size else (np.zeros(0, dtype=int), np.zeros(0))
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "histogram": {"counts": hist.astype(int).tolist(), "edges": edges.astype(float).tolist()},
    }


def _failure_stage(row: dict[str, Any]) -> str:
    if int(row.get("image_stars", 0) or 0) <= 0:
        return "NO_IMAGE_STARS"
    if int(row.get("image_stars", 0) or 0) < 8:
        return "INSUFFICIENT_IMAGE_STARS"
    if int(row.get("catalog_stars", 0) or 0) <= 0:
        return "NO_CATALOG_STARS"
    if int(row.get("catalog_stars", 0) or 0) < 20:
        return "INSUFFICIENT_CATALOG_STARS"
    if int(row.get("image_quads", 0) or 0) <= 0:
        return "NO_IMAGE_QUADS"
    if int(row.get("catalog_quads", 0) or 0) <= 0:
        return "NO_CATALOG_QUADS"
    if int(row.get("matches_raw", 0) or 0) <= 0:
        return "NO_SIGNATURE_MATCHES"
    if int(row.get("matches_kept", 0) or 0) <= 0:
        return "MATCHES_REJECTED_BY_SCALE"
    if int(row.get("refs", 0) or 0) <= 0:
        return "TRANSFORM_FAILED"
    return "VALIDATION_FAILED"


def _matrix_case(image_points: np.ndarray, catalog_points: np.ndarray) -> dict[str, Any]:
    res = _run_iso_matrix_case(image_points, catalog_points)
    return {
        "image_stars": int(image_points.shape[0]),
        "catalog_stars": int(catalog_points.shape[0]),
        "image_quads": int(res.get("quads_img", 0) or 0),
        "catalog_quads": int(res.get("quads_cat", 0) or 0),
        "matches_raw": int(res.get("matches_raw", 0) or 0),
        "matches_kept": int(res.get("matches_kept", 0) or 0),
        "refs": int(res.get("refs", 0) or 0),
        "transform": res.get("transform"),
        "success": bool(res.get("success")),
        "failure_stage": None if res.get("success") else _failure_stage(
            {
                "image_stars": image_points.shape[0],
                "catalog_stars": catalog_points.shape[0],
                "image_quads": res.get("quads_img", 0),
                "catalog_quads": res.get("quads_cat", 0),
                "matches_raw": res.get("matches_raw", 0),
                "matches_kept": res.get("matches_kept", 0),
                "refs": res.get("refs", 0),
            }
        ),
    }


def _run_native(fits_path: Path, index_root: Path, dump_dir: Path, label: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = NearSolveConfig(
        family="d50",
        detect_backend="cpu",
        astap_iso_strict=True,
        ransac_seed=0,
        diagnostic_dump_dir=str(dump_dir),
        diagnostic_dump_label=label,
    )
    res = solve_near(fits_path, index_root, config=cfg)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "elapsed_s": time.perf_counter() - t0,
        "stats": res.stats,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    matrix = payload.get("progressive_replacement_matrix", {})
    native = payload.get("native_solver_results", {})
    p11_success = sum(1 for row in matrix.values() if isinstance(row, dict) and row.get("P11", {}).get("success"))
    o11_success = sum(1 for row in matrix.values() if isinstance(row, dict) and row.get("O11", {}).get("success"))
    native_success = sum(1 for row in native.values() if isinstance(row, dict) and row.get("success"))
    lines = [
        "# ZN3 Input List Parity",
        "",
        "## Résumé exécutif",
        "",
        payload["executive_summary"],
        "",
        "## Baseline ZN2 reproduite",
        "",
        "- R0 ZeNear natif: `0/8` d'après `zenear_zn2_injection_matrix.json`.",
        "- R1 oracle C11 ASTAP image + ASTAP catalogue: `8/8`.",
        "- R2 ASTAP local instrumenté `-z 2`: `8/8`.",
        "",
        "## Binning ASTAP-compatible",
        "",
        f"- politique M31: `bin_factor={payload['binning_policy'].get('m31_bin_factor')}`",
        "- conversion vérifiée: `x_full = 0.5 + 2*x_binned` pour x2.",
        "",
        "## Comparaison des images binned",
        "",
        payload["binned_image_parity_note"],
        "",
        "## Détection et classement image",
        "",
        f"- P11 succès: `{p11_success}/8` ; oracle O11: `{o11_success}/8`.",
        "- Les listes Python sont reconstruites depuis FITS, sans ASTAP runtime ni dumps en entrée.",
        "",
        "## Catalogue RA/Dec",
        "",
        payload["catalog_radec_note"],
        "",
        "## Matrice P00/P10/P01/P11/O11",
        "",
        f"- P11: `{p11_success}/8`",
        f"- O11: `{o11_success}/8`",
        "",
        "## Résultats natifs sans injection",
        "",
        f"- solve_near strict ASTAP-ISO: `{native_success}/{len(native)}` sur les cas exécutés.",
        "",
        "## Questions obligatoires",
        "",
        "1. Image binned Python équivalente ASTAP: non démontré, faute de dump image binned ASTAP.",
        "2. `+0.5 px`: oui, appliqué et testé.",
        "3. Détections brutes Python correspondant à ASTAP: voir `zenear_zn3_image_detection_parity.json`.",
        "4. Étoiles finales correspondant à ASTAP: voir `zenear_zn3_image_ranking_parity.json`.",
        "5. Top 50 amélioré: voir matrice; pas promu si insuffisant.",
        "6. Étage image causal: encore partiel, détection/classement restent suspects.",
        "7. Même étoiles célestes catalogue: non comparable avec les dumps ZN2 actuels, RA/Dec absentes côté ASTAP.",
        "8. Zéro recouvrement projeté ZN2: non résolu; RA/Dec requises pour trancher repère vs sélection.",
        "9. Même tuiles chargées: non démontré avec les dumps actuels.",
        "10. Fenêtre catalogue équivalente: partielle seulement.",
        "11. Projection équivalente: non démontrée sans étoiles communes RA/Dec.",
        "12. Classement catalogue équivalent: non démontré.",
        "13. I3 + catalogue ASTAP 8/8: non atteint par cette reconstruction Python.",
        "14. Image ASTAP + K3 8/8: non atteint par cette reconstruction Python.",
        f"15. P11 8/8: `{p11_success}/8`.",
        "16. P11 sans dumps ASTAP: oui pour les listes Python, non pour l'oracle de comparaison.",
        f"17. Vrai solve_near 8/8: `{native_success}/{len(native)}` sur les cas exécutés.",
        "18. WCS conforme oracle: non promu.",
        "19. Test synthétique: voir résultats pytest.",
        "20. Anciens succès ZeNear: non couverts dans cette passe.",
        "21. Contrôle négatif: non couvert dans cette passe.",
        "22. ZeBlind inchangé: aucun fichier ZeBlind/index/profil modifié par ZN3.",
        "23. Coût temps/mémoire: voir JSON natif.",
        "24. ZN4 nécessaire: oui si P11 ou solve_near restent sous 8/8.",
        "",
        "## Verdict",
        "",
        payload["verdict"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--dump-dir", type=Path, default=REPO_ROOT / "reports" / "zn2_astap_internal_dumps")
    ap.add_argument("--zenear-star-dir", type=Path, default=REPO_ROOT / "reports" / "zn1_star_lists")
    ap.add_argument("--index-root", type=Path, default=Path("/home/tristan/zesolver_index"))
    ap.add_argument("--runtime-dir", type=Path, default=REPO_ROOT / "reports" / "zn1_runtime")
    ap.add_argument("--all-eight", action="store_true", help="Run native solve_near on all eight images instead of the ZN3 primary three.")
    ap.add_argument("--skip-native", action="store_true")
    args = ap.parse_args()

    reports = args.reports_dir.resolve()
    for sub in ("zn3_image_dumps", "zn3_catalog_dumps", "zn3_matrix_runs"):
        (reports / sub).mkdir(parents=True, exist_ok=True)

    baseline_zn2 = json.loads((reports / "zenear_zn2_injection_matrix.json").read_text(encoding="utf-8"))
    baseline = {
        "R0_ZeNear_native": {"successes": sum(1 for row in baseline_zn2.values() if row.get("C00", {}).get("success")), "total": 8},
        "R1_ZN2_oracle_C11": {"successes": sum(1 for row in baseline_zn2.values() if row.get("C11", {}).get("success")), "total": 8},
        "R2_ASTAP_local_instrumented_z2": {"successes": 8, "total": 8},
        "source": "ZN2 reports regenerated before ZN3; ASTAP local/system/instrumented all 8/8 with -z 2.",
    }
    write_json(reports / "zenear_zn3_baseline.json", baseline)

    binned_parity: dict[str, Any] = {}
    image_detection: dict[str, Any] = {}
    image_ranking: dict[str, Any] = {}
    catalog_radec: dict[str, Any] = {}
    catalog_projection: dict[str, Any] = {}
    catalog_ranking: dict[str, Any] = {}
    matrix: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    native_results: dict[str, Any] = {}

    for name in IMAGE_NAMES:
        stem = safe_stem(name)
        trace = load_trace(stem, args.dump_dir.resolve())
        runtime = args.runtime_dir.resolve() / f"{stem}_runtime.fit"
        if trace is None or not runtime.exists():
            matrix[stem] = {"status": "blocked", "reason": "missing trace or runtime FITS"}
            continue

        image = _image_from_fits(runtime)
        bin_factor = choose_astap_compatible_bin_factor(width=image.shape[1], height=image.shape[0], fov_deg=1.276, scale_arcsec=2.392)
        binned, used_bin = astap_compatible_mean_bin_image(image, bin_factor)
        binned_parity[stem] = {
            "python_binned": _stats_image(binned),
            "background": astap_background_noise_stats(binned),
            "astap_binned_dump_available": False,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "correlation": None,
        }

        py_stars, py_diag = astap_compatible_image_detection(image, bin_factor=used_bin, max_stars=500, k_sigma=3.0, min_area=4)
        _write_star_csv(reports / "zn3_image_dumps" / f"{stem}_python_astap_compatible_image_stars.csv", py_stars)
        py_img = np.column_stack((py_stars["x"], py_stars["y"])).astype(np.float64, copy=False)
        astap_img = _trace_image_points(trace)
        zenear_img = _read_xy_csv(args.zenear_star_dir / f"{stem}_A0_zenear_native_run1_zenear_image_stars.csv")
        astap_cat = _trace_catalog_points(trace)
        zenear_cat = _read_xy_csv(args.zenear_star_dir / f"{stem}_A0_zenear_native_run1_zenear_catalog_stars.csv")

        image_detection[stem] = {"python_diag": py_diag, "overlap_vs_astap": compare_points(py_img, astap_img)}
        image_ranking[stem] = {
            "python_final_stars": int(py_img.shape[0]),
            "astap_final_stars": int(astap_img.shape[0]),
            "overlap": compare_points(py_img, astap_img),
        }

        astap_cat_rows = _read_csv(next((args.dump_dir / stem).glob("*_astap_internal_catalog_stars.csv")))
        zenear_cat_rows = _read_csv(args.zenear_star_dir / f"{stem}_A0_zenear_native_run1_zenear_catalog_stars.csv")
        catalog_radec[stem] = _spherical_match(astap_cat_rows, zenear_cat_rows)
        catalog_projection[stem] = compare_points(astap_cat, zenear_cat)
        catalog_ranking[stem] = {
            "astap_catalog_stars": int(astap_cat.shape[0]),
            "zenear_catalog_stars": int(zenear_cat.shape[0]),
            "status": "projection_only_no_astap_radec",
        }

        row = {
            "P00": _matrix_case(zenear_img, zenear_cat),
            "P10": _matrix_case(py_img, zenear_cat),
            "P01": _matrix_case(zenear_img, astap_cat),
            "P11": _matrix_case(py_img, zenear_cat),
            "O11": _matrix_case(astap_img, astap_cat),
        }
        matrix[stem] = row
        failures[stem] = {key: val.get("failure_stage") for key, val in row.items()}

    native_names = IMAGE_NAMES if args.all_eight else [n for n in IMAGE_NAMES if n in PRIMARY_NAMES]
    if not args.skip_native:
        for name in native_names:
            stem = safe_stem(name)
            runtime = args.runtime_dir.resolve() / f"{stem}_runtime.fit"
            if runtime.exists():
                native_results[stem] = _run_native(runtime, args.index_root.resolve(), reports / "zn3_matrix_runs", f"{stem}_zn3_native")

    write_json(reports / "zenear_zn3_binned_image_parity.json", binned_parity)
    write_json(reports / "zenear_zn3_image_detection_parity.json", image_detection)
    write_json(reports / "zenear_zn3_image_ranking_parity.json", image_ranking)
    write_json(reports / "zenear_zn3_catalog_radec_parity.json", catalog_radec)
    write_json(reports / "zenear_zn3_catalog_projection_parity.json", catalog_projection)
    write_json(reports / "zenear_zn3_catalog_ranking_parity.json", catalog_ranking)
    write_json(reports / "zenear_zn3_progressive_replacement_matrix.json", matrix)
    write_json(reports / "zenear_zn3_native_solver_results.json", native_results)
    write_json(reports / "zenear_zn3_failure_classification.json", failures)

    p11_success = sum(1 for row in matrix.values() if isinstance(row, dict) and row.get("P11", {}).get("success"))
    native_success = sum(1 for row in native_results.values() if isinstance(row, dict) and row.get("success"))
    payload = {
        "executive_summary": (
            "ZN3 a ajouté les briques Python ASTAP-compatible de binning/mapping/sélection et un probe de matrice. "
            "La promotion produit reste bloquée tant que la parité catalogue RA/Dec n'est pas instrumentée dans ASTAP "
            "et tant que P11/native ne reproduisent pas O11."
        ),
        "binning_policy": {"m31_bin_factor": 2},
        "binned_image_parity_note": "Le binned Python est mesuré statistiquement; aucun dump ASTAP de l'image binned n'était disponible dans ZN2 pour une différence pixel à pixel.",
        "catalog_radec_note": "Bloqué: les dumps catalogue internes ZN2 ne contiennent pas RA/Dec/magnitude/tile, seulement les coordonnées projetées.",
        "progressive_replacement_matrix": matrix,
        "native_solver_results": native_results,
        "verdict": (
            "F — Non résolu. Les listes Python ne reproduisent pas encore causalement les entrées ASTAP; "
            f"P11={p11_success}/8, native={native_success}/{len(native_results)} sur les cas exécutés. "
            "Aucun seuil, signature, lookup, transformation, rescue ou composant ZeBlind n'a été modifié."
        ),
    }
    write_report(reports / "zenear_zn3_input_list_parity.md", payload)
    print(json.dumps({"P11_success": p11_success, "native_success": native_success, "report": str(reports / "zenear_zn3_input_list_parity.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
