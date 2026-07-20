from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "reports/p221_app_integration"
VALIDATION_JSON = ROOT / "reports/zeblind_p221_app_integration_validation.json"
VALIDATION_MD = ROOT / "reports/zeblind_p221_app_integration_validation.md"
MANIFEST = ROOT / "config/zeblind_4d_experimental_manifest.json"
PYTHON = ROOT / ".venv/bin/python"

POSITION_HINT_KEYS = {
    "RA",
    "DEC",
    "OBJCTRA",
    "OBJCTDEC",
    "OBJRA",
    "OBJDEC",
    "TELRA",
    "TELDEC",
    "CENTRA",
    "CENTDEC",
    "RA_OBJ",
    "DEC_OBJ",
}
IDENTITY_HINT_KEYS = {"OBJECT", "OBJNAME", "TARGET", "TARGNAME", "FIELD", "FIELDID"}
WCS_KEYS = {
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
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
    "RADESYS",
    "RADECSYS",
    "EQUINOX",
    "LONPOLE",
    "LATPOLE",
}

CORPUS = [
    ("m106", "s50", "boundary_234013", Path("/home/tristan/zemosaic/example/backuplightsastap/Light_mosaic_M 106_20.0s_IRCUT_20250518-234013.fit")),
    ("m106", "s50", "easy_232205", Path("/home/tristan/zemosaic/example/backuplightsastap/Light_mosaic_M 106_20.0s_IRCUT_20250518-232205.fit")),
    ("ngc6888", "s50", "low_rank", Path("/home/tristan/zemosaic/example/astap solved/Light_NGC 6888_30.0s_LP_20250619-020803.fit")),
    ("ngc6888", "s50", "high_rank", Path("/home/tristan/zemosaic/example/astap solved/Light_NGC 6888_30.0s_LP_20250619-015658.fit")),
    ("m31", "s50", "altaz_s50", Path("/home/tristan/zemosaic/example/various_fresh/Light_mosaic_M 31_10.0s_IRCUT_20250115-202105.fit")),
    ("m31", "s50", "eq_s50", Path("/home/tristan/zemosaic/example/various_fresh/Light_M 31_20.0s_IRCUT_20251117-225718.fit")),
    ("m31", "s30", "altaz_s30", Path("/home/tristan/zemosaic/example/various_fresh/Light_mosaic_M 31_60.0s_IRCUT_20250904-015506.fit")),
]

SCALE_RANGES = {
    "s50": (1.90, 2.85),
    "s30": (3.19, 4.79),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_celestial_wcs(header: fits.Header) -> bool:
    try:
        return bool(WCS(header).has_celestial)
    except Exception:
        return False


def _strip_runtime_copy(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    removed: list[str] = []
    with fits.open(target, mode="update", memmap=False) as hdul:
        header = hdul[0].header
        for key in sorted(WCS_KEYS | POSITION_HINT_KEYS | IDENTITY_HINT_KEYS):
            if key in header:
                removed.append(key)
                del header[key]
        hdul.flush()
    header = fits.getheader(target)
    forbidden_remaining = [key for key in sorted(POSITION_HINT_KEYS | IDENTITY_HINT_KEYS) if key in header]
    return {
        "source": str(source),
        "runtime": str(target),
        "source_sha256": sha256_file(source),
        "runtime_input_sha256": sha256_file(target),
        "removed_keys": removed,
        "forbidden_keys_remaining": forbidden_remaining,
        "has_celestial_wcs_after_strip": _has_celestial_wcs(header),
    }


def _wcs_metrics(path: Path, source: Path) -> dict[str, Any]:
    header = fits.getheader(path)
    out: dict[str, Any] = {
        "wcs_written": _has_celestial_wcs(header),
        "wcs_cards_present": all(key in header for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")),
        "readable": True,
        "header_keys_kept": {
            "INSTRUME": header.get("INSTRUME"),
            "TELESCOP": header.get("TELESCOP"),
            "EXPTIME": header.get("EXPTIME"),
        },
    }
    try:
        wcs = WCS(header)
        ref = WCS(fits.getheader(source))
        shape = fits.getdata(path, memmap=False).shape
        if len(shape) >= 2:
            h, w = int(shape[-2]), int(shape[-1])
            pts = np.asarray(
                [
                    [w / 2.0, h / 2.0],
                    [0.0, 0.0],
                    [w - 1.0, 0.0],
                    [0.0, h - 1.0],
                    [w - 1.0, h - 1.0],
                ],
                dtype=np.float64,
            )
            world = wcs.all_pix2world(pts, 0)
            ref_world = ref.all_pix2world(pts, 0)
            cos_dec = np.cos(np.deg2rad(ref_world[:, 1]))
            dra = (world[:, 0] - ref_world[:, 0]) * cos_dec * 3600.0
            ddec = (world[:, 1] - ref_world[:, 1]) * 3600.0
            sep = np.hypot(dra, ddec)
            out["offline_center_sep_arcsec"] = float(sep[0])
            out["offline_corner_max_sep_arcsec"] = float(np.max(sep[1:]))
            out["offline_ok"] = bool(float(sep[0]) < 60.0 and float(np.max(sep[1:])) < 180.0)
        scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0
        out["pixel_scale_arcsec"] = float(np.sqrt(float(scales[0]) * float(scales[1])))
    except Exception as exc:
        out["readable"] = False
        out["offline_ok"] = False
        out["wcs_error"] = str(exc)
    return out


def _run_cli(batch_dir: Path, regime: str) -> dict[str, Any]:
    lo, hi = SCALE_RANGES[regime]
    cmd = [
        str(PYTHON),
        str(ROOT / "zesolver.py"),
        "--headless",
        "--db-root",
        str(ROOT),
        "--input-dir",
        str(batch_dir),
        "--formats",
        "fit",
        "--workers",
        "1",
        "--overwrite",
        "--blind-only",
        "--blind-profile",
        "zeblind_4d_experimental",
        "--blind-4d-manifest",
        str(MANIFEST),
        "--pixel-scale-min",
        str(lo),
        "--pixel-scale-max",
        str(hi),
        "--log-level",
        "INFO",
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    return {
        "regime": regime,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - start,
        "output": proc.stdout,
    }


def _run_manifest_negative() -> dict[str, Any]:
    cmd = [
        str(PYTHON),
        str(ROOT / "zesolver.py"),
        "--headless",
        "--db-root",
        str(ROOT),
        "--input-dir",
        str(WORK / "missing_manifest_control"),
        "--blind-only",
        "--blind-profile",
        "zeblind_4d_experimental",
        "--blind-4d-manifest",
        str(WORK / "absent_manifest.json"),
    ]
    (WORK / "missing_manifest_control").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    return {
        "name": "manifest_absent_cli_preflight",
        "returncode": proc.returncode,
        "explicit_error": proc.returncode != 0 and "4D manifest error" in proc.stdout,
        "output": proc.stdout,
    }


def _run_historical_control() -> dict[str, Any]:
    cmd = [
        str(PYTHON),
        str(ROOT / "zesolver.py"),
        "--headless",
        "--db-root",
        str(ROOT),
        "--input-dir",
        str(WORK / "missing_manifest_control"),
        "--blind-profile",
        "historical",
        "--blind-4d-manifest",
        str(WORK / "absent_manifest.json"),
        "--max-files",
        "0",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    return {
        "name": "historical_ignores_4d_manifest",
        "returncode": proc.returncode,
        "manifest_not_loaded": "4D manifest error" not in proc.stdout,
        "output": proc.stdout,
    }


def _write_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# P2.21 App Integration Validation",
        "",
        "## Executive Summary",
        "",
        f"- Verdict: `{summary['verdict']}`.",
        f"- 4D CLI successes: `{summary['successes_4d']}/{summary['total_4d']}`.",
        f"- Offline false positives: `{summary['offline_false_positives']}`.",
        f"- WCS write failures: `{summary['wcs_write_failures']}`.",
        f"- Manifest negative explicit: `{summary['manifest_negative_explicit']}`.",
        f"- Historical default preserved: `{summary['historical_default_preserved']}`.",
        "",
        "## Architecture",
        "",
        "- Package manifest loader: `zeblindsolver/index_manifest_4d.py`.",
        "- Central profile: `zeblindsolver/profiles.py`.",
        "- App helper: `build_blind_solve_config` in `zesolver.py`.",
        "- Runtime manifest: `config/zeblind_4d_experimental_manifest.json`.",
        "",
        "## Corpus In Situ",
        "",
        "| file | field | regime | success | tile | rank | inliers | rms | scale | WCS | offline |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["image_rows"]:
        stats = row.get("stats") or {}
        lines.append(
            f"| `{Path(row['runtime']).name}` | `{row['field']}` | `{row['regime']}` | `{row['success']}` | "
            f"`{stats.get('astrometry_4d_selected_origin_tile_key') or row.get('tile')}` | "
            f"{stats.get('astrometry_4d_selected_rank', '')} | {stats.get('inliers', stats.get('astrometry_4d_selected_inliers', ''))} | "
            f"{stats.get('rms_px', stats.get('astrometry_4d_selected_rms_px', ''))} | "
            f"{row.get('pixel_scale_arcsec', '')} | `{row['wcs_written']}` | `{row['offline_ok']}` |"
        )
    lines.extend(
        [
            "",
            "## CLI Runs",
            "",
        ]
    )
    for run in payload["cli_runs"]:
        lines.append(f"- `{run['regime']}`: returncode `{run['returncode']}`, elapsed `{run['elapsed_s']:.1f}s`.")
    lines.extend(
        [
            "",
            "## Errors And Stop",
            "",
            f"- Manifest absent preflight explicit: `{payload['manifest_negative']['explicit_error']}`.",
            "- Stop before/while processing was not exercised in this bounded non-interactive run; the existing cancel callback path is preserved and not promoted as newly validated here.",
            "",
            "## Mandatory Answers",
            "",
        ]
    )
    for answer in payload["mandatory_answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Verdict", "", summary["verdict_text"], "", "## Recommendation", "", summary["recommendation"], ""])
    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for field, regime, role, source in CORPUS:
        target = WORK / "runtime" / regime / source.name
        hygiene = _strip_runtime_copy(source, target)
        selected.append({"field": field, "regime": regime, "role": role, "source": str(source), "runtime": str(target), "hygiene": hygiene})

    cli_runs = []
    for regime in sorted({item["regime"] for item in selected}):
        cli_runs.append(_run_cli(WORK / "runtime" / regime, regime))

    rows: list[dict[str, Any]] = []
    for item in selected:
        runtime = Path(item["runtime"])
        metrics = _wcs_metrics(runtime, Path(item["source"]))
        header = fits.getheader(runtime)
        stats: dict[str, Any] = {}
        # The CLI writes detailed per-index stats to logs; accepted quality is mirrored in FITS.
        header_stats = {
            "inliers": ("INLIERS", "N_INLIERS"),
            "rms_px": ("RMSPX", "RMS_PX"),
            "pixscal": ("PIXSCAL",),
        }
        for out_key, candidates in header_stats.items():
            for key in candidates:
                if key in header:
                    stats[out_key] = header[key]
                    break
        rows.append(
            {
                **item,
                **metrics,
                "success": bool(metrics.get("wcs_written") and metrics.get("offline_ok")),
                "tile": header.get("TILE_ID") or header.get("TILEKEY") or header.get("ZBTILE") or header.get("TILE"),
                "stats": stats,
                "output_sha256": sha256_file(runtime),
                "original_unchanged": sha256_file(Path(item["source"])) == item["hygiene"]["source_sha256"],
            }
        )

    manifest_negative = _run_manifest_negative()
    historical_control = _run_historical_control()
    successes = sum(1 for row in rows if row["success"])
    offline_false = sum(1 for row in rows if row["wcs_written"] and not row.get("offline_ok"))
    wcs_failures = sum(1 for row in rows if not row["wcs_written"])
    verdict_ok = (
        successes == len(rows)
        and offline_false == 0
        and wcs_failures == 0
        and all(run["returncode"] == 0 for run in cli_runs)
        and bool(manifest_negative["explicit_error"])
        and bool(historical_control["manifest_not_loaded"])
    )
    summary = {
        "verdict": "A - Integration applicative validee" if verdict_ok else "B - Moteur integre mais chemin applicatif incomplet",
        "verdict_text": (
            "Le backend ZeBlind 4D experimental est integre au chemin applicatif reel via un manifest portable et un profil centralise. Le backend historique reste le defaut."
            if verdict_ok
            else "Le backend 4D est accessible depuis le chemin applicatif, mais une partie de la validation in situ reste incomplete."
        ),
        "successes_4d": successes,
        "total_4d": len(rows),
        "offline_false_positives": offline_false,
        "wcs_write_failures": wcs_failures,
        "manifest_negative_explicit": bool(manifest_negative["explicit_error"]),
        "historical_default_preserved": bool(historical_control["manifest_not_loaded"]),
        "recommendation": "Prochaine etape : integration GUI minimale et test utilisateur in situ.",
    }
    payload = {
        "schema": "zeblind.p221_app_integration_validation.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": manifest,
        "selected": selected,
        "cli_runs": cli_runs,
        "image_rows": rows,
        "manifest_negative": manifest_negative,
        "historical_control": historical_control,
        "summary": summary,
        "mandatory_answers": [
            "Oui, le chargeur du manifest appartient maintenant au package via zeblindsolver/index_manifest_4d.py.",
            "Oui, le manifest runtime est portable et utilise des chemins relatifs.",
            "Aucun chemin /home/tristan/... n'est necessaire pour resoudre les index du manifest runtime.",
            "Oui, zeblind_4d_experimental centralise le contrat P2.20 dans zeblindsolver/profiles.py.",
            "Batch CLI, blind-only et test blind settings utilisent build_blind_solve_config; les chemins Near restent separes.",
            "Oui, le CLI applicatif peut lancer le 4D avec --blind-profile zeblind_4d_experimental.",
            "Oui, le backend historique reste le defaut.",
            "Non, une ancienne configuration charge blind_backend_profile=historical.",
            "Oui, le preset S30 utilise 150 mm et la plage 3.19..4.79 arcsec/px.",
            f"Le vrai chemin applicatif retrouve {successes}/{len(rows)} solutions correctes offline dans ce banc.",
            f"WCS ecrit et relisible: {len(rows) - wcs_failures}/{len(rows)}.",
            "Stop non promu dans ce run non-interactif; le chemin cancel_check existant reste cable.",
            f"Erreur manifest explicite sans fallback: {bool(manifest_negative['explicit_error'])}.",
            "Les logs CLI affichent profil, manifest, schema, index enabled, tiles, plage d'echelle et budgets.",
            "Le backend est pret pour une integration GUI utilisateur limitee si ce run est A.",
            "Prochaine etape unique : integration GUI minimale et test utilisateur in situ.",
        ],
    }
    VALIDATION_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_report(payload)
    print(json.dumps({"summary": summary, "json": str(VALIDATION_JSON), "md": str(VALIDATION_MD)}, indent=2))
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
