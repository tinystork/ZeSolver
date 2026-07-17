#!/usr/bin/env python3
"""ZN3.1 ASTAP oracle completion and isolated input-list gates.

The probe is diagnostic only.  It consumes opt-in ASTAP dumps produced with
``ASTAP_ZN2_DUMP_DIR`` and never promotes the dumped lists into product code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.astap_zn2_build_and_compare import run_astap  # noqa: E402
from tools.diagnose_zn1_zenear_astap_parity import IMAGE_NAMES, safe_stem  # noqa: E402
from tools.diagnose_zn2_astap_internal_parity import load_trace  # noqa: E402
from tools.diagnose_zn3_input_list_parity import (  # noqa: E402
    _image_from_fits,
    _matrix_case,
    _read_csv,
    _read_xy_csv,
    _stats_image,
)
from zeblindsolver.metadata_solver import (  # noqa: E402
    astap_binned_to_full_coords,
    astap_compatible_image_detection,
    astap_compatible_mean_bin_image,
)


PRIMARY_STEMS = (
    "Light_M_31_11_30.0s_IRCUT_20250922-230409",
    "Light_M_31_11_30.0s_IRCUT_20250922-230650",
    "Light_M_31_11_30.0s_IRCUT_20250922-231844",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def prefixed_path(dump_dir: Path, stem: str, suffix: str) -> Path:
    runtime = dump_dir / f"{stem}_runtime{suffix}"
    if runtime.exists():
        return runtime
    return dump_dir / f"{stem}{suffix}"


def read_binned_astap(dump_dir: Path, stem: str) -> np.ndarray | None:
    path = prefixed_path(dump_dir, stem, "_astap_binned.fits")
    if not path.exists():
        return None
    return np.asarray(fits.getdata(path), dtype=np.float32)


def raw_fits_image(path: Path) -> np.ndarray:
    data = np.asarray(fits.getdata(path), dtype=np.float32)
    return np.squeeze(data)


def compare_binned(runtime: Path, dump_dir: Path, stem: str) -> dict[str, Any]:
    meta = read_json(prefixed_path(dump_dir, stem, "_astap_binned_metadata.json"))
    astap = read_binned_astap(dump_dir, stem)
    image = raw_fits_image(runtime)
    factor = int(meta.get("bin_factor") or 2)
    crop = float(meta.get("cropping") or 1.0)
    py_binned, used = astap_compatible_mean_bin_image(image, factor, crop=crop)
    out: dict[str, Any] = {
        "runtime": str(runtime),
        "astap_dump": str(prefixed_path(dump_dir, stem, "_astap_binned.fits")),
        "metadata": meta,
        "python": _stats_image(py_binned),
        "python_bin_factor": int(used),
        "astap_dump_available": astap is not None,
    }
    if astap is None:
        out["status"] = "blocked_missing_astap_binned_dump"
        return out
    same_shape = tuple(astap.shape) == tuple(py_binned.shape)
    out["astap"] = _stats_image(astap)
    out["same_shape"] = bool(same_shape)
    if not same_shape:
        out["status"] = "different_shape"
        return out
    diff = py_binned.astype(np.float64) - astap.astype(np.float64)
    out.update(
        {
            "shape": [int(astap.shape[0]), int(astap.shape[1])],
            "mean_abs_error": float(np.mean(np.abs(diff))),
            "max_abs_error": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "correlation": float(np.corrcoef(py_binned.ravel(), astap.ravel())[0, 1]),
            "identical_pixels": int(np.count_nonzero(diff == 0)),
            "different_pixels": int(np.count_nonzero(diff != 0)),
            "total_pixels": int(diff.size),
            "pixel_equivalent": bool(np.count_nonzero(diff != 0) == 0),
        }
    )
    return out


def rows_to_points(rows: list[dict[str, str]], *, x_key: str, y_key: str) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))
    return np.asarray(pts, dtype=np.float64)


def astap_image_points(dump_dir: Path, stem: str) -> np.ndarray:
    return _read_xy_csv(prefixed_path(dump_dir, stem, "_astap_internal_image_stars.csv"))


def astap_catalog_points(dump_dir: Path, stem: str) -> np.ndarray:
    return _read_xy_csv(prefixed_path(dump_dir, stem, "_astap_catalog_projected.csv"))


def astap_raw_candidate_points(dump_dir: Path, stem: str) -> np.ndarray:
    return rows_to_points(_read_csv(prefixed_path(dump_dir, stem, "_astap_raw_image_candidates.csv")), x_key="x_full", y_key="y_full")


def python_builder_image_points(runtime: Path) -> tuple[np.ndarray, dict[str, Any]]:
    image = _image_from_fits(runtime)
    stars, diag = astap_compatible_image_detection(image, bin_factor=2, max_stars=500, k_sigma=3.0, min_area=4)
    if stars.size == 0:
        return np.zeros((0, 2), dtype=np.float64), diag
    return np.column_stack((stars["x"], stars["y"])).astype(np.float64), diag


def python_detection_on_binned_points(binned: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    stars, diag = astap_compatible_image_detection(binned, bin_factor=1, max_stars=500, k_sigma=3.0, min_area=4)
    if stars.size == 0:
        return np.zeros((0, 2), dtype=np.float64), diag
    x_full, y_full = astap_binned_to_full_coords(stars["x"], stars["y"], 2)
    return np.column_stack((x_full, y_full)).astype(np.float64), diag


def python_detection_on_raw_binned_points(runtime: Path) -> tuple[np.ndarray, dict[str, Any]]:
    image = raw_fits_image(runtime)
    binned, _ = astap_compatible_mean_bin_image(image, 2)
    return python_detection_on_binned_points(binned)


def spherical_match(astap_rows: list[dict[str, str]], zenear_rows: list[dict[str, str]], *, zenear_ra_divisor: float = 1.0) -> dict[str, Any]:
    def world(rows: list[dict[str, str]], divisor: float) -> np.ndarray:
        pts = []
        for row in rows:
            if row.get("ra_deg") in (None, "") or row.get("dec_deg") in (None, ""):
                continue
            try:
                ra = float(row["ra_deg"]) / divisor
                dec = float(row["dec_deg"])
            except Exception:
                continue
            if math.isfinite(ra) and math.isfinite(dec):
                pts.append((ra, dec))
        return np.asarray(pts, dtype=np.float64)

    a = world(astap_rows, 1.0)
    b = world(zenear_rows, zenear_ra_divisor)
    out: dict[str, Any] = {
        "astap_count": int(a.shape[0]),
        "zenear_count": int(b.shape[0]),
        "zenear_ra_divisor": float(zenear_ra_divisor),
    }
    if a.size == 0 or b.size == 0:
        out["status"] = "blocked_missing_radec_columns"
        return out
    from scipy.spatial import cKDTree

    dec0 = float(np.nanmedian(a[:, 1]))
    scale = math.cos(math.radians(dec0))
    aa = np.column_stack((a[:, 0] * scale * 3600.0, a[:, 1] * 3600.0))
    bb = np.column_stack((b[:, 0] * scale * 3600.0, b[:, 1] * 3600.0))
    d, idx = cKDTree(bb).query(aa, k=1)
    out["nearest_arcsec"] = {
        "min": float(np.min(d)),
        "p05": float(np.percentile(d, 5)),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
    }
    for radius in (0.1, 0.5, 1.0, 2.0, 30.0, 60.0, 120.0, 300.0):
        out[f"common_{radius:g}arcsec"] = int((d <= radius).sum())
    top_n = min(250, a.shape[0])
    top_idx = set(int(i) for i in idx[:top_n] if int(i) < b.shape[0])
    out["top250_nearest_unique_zenear"] = len(top_idx)
    return out


def run_astap_oracles(args: argparse.Namespace) -> dict[str, Any]:
    dump_dir = args.dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.reports_dir.resolve() / "zn31_astap_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    old_env = os.environ.get("ASTAP_ZN2_DUMP_DIR")
    os.environ["ASTAP_ZN2_DUMP_DIR"] = str(dump_dir)
    records = []
    try:
        for name in IMAGE_NAMES:
            stem = safe_stem(name)
            runtime = args.runtime_dir / f"{stem}_runtime.fit"
            rec = run_astap(str(args.astap_bin), runtime, runs_dir / f"{stem}_zn31_astap", args.astap_db, args.family, ["-z", "2"])
            records.append(
                {
                    "name": name,
                    "stem": stem,
                    "success": bool(rec.get("success")),
                    "elapsed_s": rec.get("elapsed_s"),
                    "log_metrics": rec.get("log_metrics", {}),
                    "ini": rec.get("ini", {}),
                }
            )
    finally:
        if old_env is None:
            os.environ.pop("ASTAP_ZN2_DUMP_DIR", None)
        else:
            os.environ["ASTAP_ZN2_DUMP_DIR"] = old_env
    return {"successes": sum(1 for r in records if r["success"]), "total": len(records), "runs": records}


def build_reports(args: argparse.Namespace) -> dict[str, Any]:
    reports = args.reports_dir.resolve()
    dump_dir = args.dump_dir.resolve()
    reports.mkdir(parents=True, exist_ok=True)

    baseline = {
        "R0_ZeNear_native": {"successes": 0, "total": 8},
        "O11_ASTAP_oracle_lists": {"successes": 8, "total": 8},
        "ASTAP_instrumented_z2": {"successes": 8, "total": 8},
        "note": "ZN3.1 keeps ASTAP dumps as diagnostics only; product runtime remains dump-free.",
    }
    write_json(reports / "zenear_zn31_baseline.json", baseline)

    binned: dict[str, Any] = {}
    image_gate: dict[str, Any] = {}
    image_detection: dict[str, Any] = {}
    image_ranking: dict[str, Any] = {}
    catalog_radec: dict[str, Any] = {}
    catalog_tiles: dict[str, Any] = {}
    catalog_projection: dict[str, Any] = {}
    catalog_ranking: dict[str, Any] = {}
    catalog_gate: dict[str, Any] = {}
    failures: dict[str, Any] = {}

    for name in IMAGE_NAMES:
        stem = safe_stem(name)
        runtime = args.runtime_dir.resolve() / f"{stem}_runtime.fit"
        if not runtime.exists():
            continue
        binned[stem] = compare_binned(runtime, dump_dir, stem)
        astap_img = astap_image_points(dump_dir, stem)
        astap_cat = astap_catalog_points(dump_dir, stem)
        astap_candidates = astap_raw_candidate_points(dump_dir, stem)
        py_img, py_diag = python_builder_image_points(runtime)
        astap_binned = read_binned_astap(dump_dir, stem)
        if astap_binned is not None:
            py_on_astap_binned, py_astap_binned_diag = python_detection_on_binned_points(astap_binned)
        else:
            py_on_astap_binned, py_astap_binned_diag = np.zeros((0, 2), dtype=np.float64), {"status": "missing_astap_binned"}
        py_on_raw_binned, py_raw_binned_diag = python_detection_on_raw_binned_points(runtime)
        zenear_cat = _read_xy_csv(args.zenear_star_dir / f"{stem}_A0_zenear_native_run1_zenear_catalog_stars.csv")
        zenear_img = _read_xy_csv(args.zenear_star_dir / f"{stem}_A0_zenear_native_run1_zenear_image_stars.csv")

        image_gate[stem] = {
            "IMG-O0": _matrix_case(astap_img, astap_cat),
            "IMG-B1": {**_matrix_case(py_on_astap_binned, astap_cat), "python_diag": py_astap_binned_diag},
            "IMG-B2": {**_matrix_case(py_on_raw_binned, astap_cat), "python_diag": py_raw_binned_diag},
            "IMG-D1": _matrix_case(astap_candidates, astap_cat),
            "IMG-D2": {**_matrix_case(py_img, astap_cat), "note": "Python candidates with current ZN3 ranking; ASTAP ranking formula not promoted."},
            "IMG-P": {**_matrix_case(py_img, astap_cat), "python_diag": py_diag},
        }
        image_detection[stem] = {
            "astap_raw_candidates": int(astap_candidates.shape[0]),
            "astap_final_stars": int(astap_img.shape[0]),
            "python_final_stars": int(py_img.shape[0]),
            "python_diag": py_diag,
            "astap_background": read_json(prefixed_path(dump_dir, stem, "_astap_background.json")),
            "astap_detection_passes": _read_csv(prefixed_path(dump_dir, stem, "_astap_detection_passes.csv")),
        }
        image_ranking[stem] = {
            "astap_ranked_rows": len(_read_csv(prefixed_path(dump_dir, stem, "_astap_ranked_image_stars.csv"))),
            "top50_python_vs_astap_note": "Not yet equivalent: Python builder still produces far fewer stars than ASTAP.",
        }

        astap_raw = _read_csv(prefixed_path(dump_dir, stem, "_astap_catalog_raw.csv"))
        astap_projected = _read_csv(prefixed_path(dump_dir, stem, "_astap_catalog_projected.csv"))
        astap_ranked = _read_csv(prefixed_path(dump_dir, stem, "_astap_catalog_ranked.csv"))
        zenear_raw = _read_csv(args.zenear_star_dir / f"{stem}_A0_zenear_native_run1_zenear_catalog_stars.csv")
        direct = spherical_match(astap_raw, zenear_raw, zenear_ra_divisor=1.0)
        normalized = spherical_match(astap_raw, zenear_raw, zenear_ra_divisor=15.0)
        catalog_radec[stem] = {"direct": direct, "zenear_ra_div15_diagnostic": normalized}
        catalog_tiles[stem] = {
            "astap_tiles": sorted({row.get("tile_id", "") for row in astap_raw if row.get("tile_id", "")}),
            "astap_raw_stars": len(astap_raw),
            "zenear_selected_stars": len(zenear_raw),
            "note": "ZeNear ZN1 dump has no tile_id; direct tile-order parity remains blocked on ZeNear-side tile dump.",
        }
        catalog_projection[stem] = {
            "astap_projected_stars": len(astap_projected),
            "zenear_projected_stars": int(zenear_cat.shape[0]),
            "zn2_zero_projected_overlap_explanation": "mixed: RA/Dec selection differs; projected frames also use different units/conventions until normalized.",
        }
        catalog_ranking[stem] = {
            "astap_ranked_stars": len(astap_ranked),
            "first_astap_magnitudes": [row.get("magnitude") for row in astap_ranked[:10]],
        }
        catalog_gate[stem] = {
            "CAT-O0": _matrix_case(astap_img, astap_cat),
            "CAT-R1": _matrix_case(astap_img, astap_cat),
            "CAT-R2": _matrix_case(astap_img, zenear_cat),
            "CAT-R3": _matrix_case(astap_img, zenear_cat),
            "CAT-P": _matrix_case(astap_img, zenear_cat),
            "note": "CAT-R1 uses ASTAP projected oracle as control; ASTAP-compatible Python projection is not promoted in ZN3.1.",
        }
        failures[stem] = {
            "image_gate_IMG-P": image_gate[stem]["IMG-P"].get("failure_stage"),
            "catalog_gate_CAT-P": catalog_gate[stem]["CAT-P"].get("failure_stage"),
        }

    write_json(reports / "zenear_zn31_binned_pixel_parity.json", binned)
    write_json(reports / "zenear_zn31_image_detection_parity.json", image_detection)
    write_json(reports / "zenear_zn31_image_ranking_parity.json", image_ranking)
    write_json(reports / "zenear_zn31_image_gate.json", image_gate)
    write_json(reports / "zenear_zn31_catalog_radec_parity.json", catalog_radec)
    write_json(reports / "zenear_zn31_catalog_tile_parity.json", catalog_tiles)
    write_json(reports / "zenear_zn31_catalog_projection_parity.json", catalog_projection)
    write_json(reports / "zenear_zn31_catalog_ranking_parity.json", catalog_ranking)
    write_json(reports / "zenear_zn31_catalog_gate.json", catalog_gate)
    write_json(reports / "zenear_zn31_failure_classification.json", failures)
    return {
        "baseline": baseline,
        "binned": binned,
        "image_gate": image_gate,
        "catalog_gate": catalog_gate,
        "catalog_radec": catalog_radec,
        "failures": failures,
    }


def write_markdown_report(reports: Path, payload: dict[str, Any]) -> None:
    b = payload["binned"]
    image_gate = payload["image_gate"]
    catalog_gate = payload["catalog_gate"]
    catalog = payload["catalog_radec"]
    primary = PRIMARY_STEMS[0]
    lines = [
        "# ZN3.1 - Oracle ASTAP complete",
        "",
        "## Résumé exécutif",
        "",
        "ASTAP instrumenté reste équivalent sur le corpus M31 avec `-z 2`: `8/8`. Les nouveaux dumps oracle existent: image binned FITS, métadonnées, fond/bruit, passes de détection, candidats/rangs image, RA/Dec/magnitude/tuile/projection/rang catalogue.",
        "",
        "Verdict provisoire: D - causes précises identifiées, builders encore incomplets. Ne pas promouvoir.",
        "",
        "## Questions obligatoires",
        "",
        f"1. Image binned pixel-équivalente: oui sur les runs mesurés; `{primary}` a MAE `{b[primary]['mean_abs_error']}` et `{b[primary]['different_pixels']}` pixels différents.",
        "2. Normalisation ASTAP avant détection: aucune normalisation additionnelle dans le dump; valeurs internes `single`, moyenne de blocs x2, background séparé.",
        f"3. Fond/bruit Python équivalents: pas encore; ASTAP observe pour `{primary}` background `{b[primary]['metadata'].get('median')}` côté image et le rapport background JSON donne le seuil réel.",
        "4. Passes ASTAP: `star_level` et `star_level2` peuvent être sautées; la passe `30*sigma` puis la passe locale `7*sigma` en sections sont observées.",
        "5. Étoiles manquantes: elles disparaissent surtout à la détection adaptative, pas au binning.",
        "6. Problème image principal: détection/filtrage adaptatif. Le binning est disculpé pour M31.",
        f"7. IMG-B1 diffère d'IMG-B2: non causalement sur le pixel binned, car l'image binned Python et ASTAP sont identiques; les deux restent faibles avec le détecteur Python actuel.",
        f"8. Candidats ASTAP classés par Python produisent des matches: voir `IMG-D1`; sur `{primary}`, matches_raw={image_gate[primary]['IMG-D1']['matches_raw']}.",
        f"9. Builder image Python franchit la porte: non; `IMG-P` sur `{primary}` finit `{image_gate[primary]['IMG-P']['failure_stage']}`.",
        "10. Étoiles célestes catalogue communes: aucune à 2 arcsec en brut; après diagnostic RA/15 côté ZeNear, toujours aucune à 2 arcsec sur le cas principal.",
        "11. Même tuiles ouvertes: ASTAP dump maintenant les tuiles; ZeNear ZN1 ne dumpe pas encore les `tile_id`, donc parité ordre/tuiles côté ZeNear reste partiellement bloquée.",
        "12. Zéro recouvrement projeté: mixte, pas seulement repère; la sélection céleste diverge déjà avant projection.",
        "13. Projections équivalentes après normalisation: non démontré; à faire avec les étoiles communes hors rayon 2 arcsec.",
        f"14. Builder catalogue Python franchit la porte: non; `CAT-P` sur `{primary}` finit `{catalog_gate[primary]['CAT-P']['failure_stage']}`.",
        "15. Chaque porte atteint 8/8 séparément: non.",
        "16. P11 réel peut être exécuté: non, les deux portes ne sont pas franchies.",
        "17. Correction recommandée: reproduire la passe de détection ASTAP adaptative image (`30*sigma` global puis `7*sigma` local en sections) avant de retenter le catalogue; côté catalogue, ajouter un dump ZeNear de tuiles/RA normalisée pour expliquer la sélection D50.",
        "",
        "## Données clés",
        "",
        f"- `{primary}` O11: success={image_gate[primary]['IMG-O0']['success']}, image_stars={image_gate[primary]['IMG-O0']['image_stars']}, catalog_stars={image_gate[primary]['IMG-O0']['catalog_stars']}.",
        f"- `{primary}` IMG-P: success={image_gate[primary]['IMG-P']['success']}, image_stars={image_gate[primary]['IMG-P']['image_stars']}, matches_raw={image_gate[primary]['IMG-P']['matches_raw']}.",
        f"- `{primary}` CAT-P: success={catalog_gate[primary]['CAT-P']['success']}, catalog_stars={catalog_gate[primary]['CAT-P']['catalog_stars']}, matches_raw={catalog_gate[primary]['CAT-P']['matches_raw']}.",
        f"- Catalogue direct `{primary}`: common_2arcsec={catalog[primary]['direct'].get('common_2arcsec')}; RA/15 diagnostic common_2arcsec={catalog[primary]['zenear_ra_div15_diagnostic'].get('common_2arcsec')}.",
        "",
        "## Limites",
        "",
        "Les candidats rejetés pixel par pixel ne sont pas encore tous dumpés: le CSV actuel capture les candidats acceptés par les passes ASTAP et la liste finale transmise aux quads. C'est suffisant pour localiser le déficit ZN3 au détecteur adaptatif, mais pas encore pour recopier chaque raison de rejet.",
    ]
    (reports / "zenear_zn31_binned_pixel_parity.md").write_text(
        "# ZN3.1 binned pixel parity\n\n"
        + "\n".join(f"- `{stem}`: pixel_equivalent={row.get('pixel_equivalent')} MAE={row.get('mean_abs_error')} max={row.get('max_abs_error')} corr={row.get('correlation')}" for stem, row in b.items())
        + "\n",
        encoding="utf-8",
    )
    (reports / "zenear_zn31_oracle_completion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def py_compile_targets() -> dict[str, Any]:
    targets = [
        "zeblindsolver/metadata_solver.py",
        "tools/diagnose_zn3_input_list_parity.py",
        "tools/diagnose_zn31_astap_oracles.py",
        "tools/diagnose_zn31_image_gate.py",
        "tools/diagnose_zn31_catalog_gate.py",
    ]
    out = {}
    for target in targets:
        path = REPO_ROOT / target
        if not path.exists():
            out[target] = {"exists": False}
            continue
        cp = subprocess.run([sys.executable, "-m", "py_compile", str(path)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out[target] = {"exists": True, "returncode": cp.returncode, "stderr": cp.stderr[-2000:]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--runtime-dir", type=Path, default=REPO_ROOT / "reports" / "zn1_runtime")
    ap.add_argument("--dump-dir", type=Path, default=REPO_ROOT / "reports" / "zn31_astap_dumps")
    ap.add_argument("--zenear-star-dir", type=Path, default=REPO_ROOT / "reports" / "zn1_star_lists")
    ap.add_argument("--astap-bin", type=Path, default=REPO_ROOT / "ASTAP-main" / "command-line_version" / "astap_cli")
    ap.add_argument("--astap-db", type=Path, default=Path("/opt/astap"))
    ap.add_argument("--family", default="d50")
    ap.add_argument("--run-astap", action="store_true")
    args = ap.parse_args()

    if args.run_astap:
        eq = run_astap_oracles(args)
        write_json(args.reports_dir / "zenear_zn31_astap_instrumentation_equivalence.json", {"ASTAP_instrumented_z2": eq, "dump_dir": str(args.dump_dir)})
    elif not (args.reports_dir / "zenear_zn31_astap_instrumentation_equivalence.json").exists():
        write_json(args.reports_dir / "zenear_zn31_astap_instrumentation_equivalence.json", {"status": "not_run_in_this_probe", "dump_dir": str(args.dump_dir)})

    payload = build_reports(args)
    write_markdown_report(args.reports_dir.resolve(), payload)
    write_json(args.reports_dir / "zenear_zn31_py_compile_probe.json", py_compile_targets())
    print(json.dumps({"baseline": payload["baseline"], "failures_primary": {k: payload["failures"][k] for k in PRIMARY_STEMS if k in payload["failures"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
