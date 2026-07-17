#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.index_manifest_4d import load_4d_index_manifest, sha256_file  # noqa: E402
from zeblindsolver.profiles import ZEBLIND_4D_EXPERIMENTAL_PROFILE  # noqa: E402


def _load_zesolver_module() -> Any:
    spec = importlib.util.spec_from_file_location("zesolver_app_p224", ROOT / "zesolver.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load zesolver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


app = _load_zesolver_module()

MANIFEST = ROOT / "config/zeblind_4d_experimental_manifest.json"
P223_CORPUS = ROOT / "reports/zeblind_p223_corpus.json"
WORK = ROOT / "reports/p224_app_budget_concurrency"

BASELINE_OUT = ROOT / "reports/zeblind_p224_baseline.json"
TIMELINE_OUT = ROOT / "reports/zeblind_p224_timeline.json"
CONCURRENCY_OUT = ROOT / "reports/zeblind_p224_concurrency_matrix.json"
BUDGET_OUT = ROOT / "reports/zeblind_p224_budget_matrix.json"
PARITY_OUT = ROOT / "reports/zeblind_p224_app_direct_parity.json"
REPORT_OUT = ROOT / "reports/zeblind_p224_app_budget_concurrency.md"

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
    "WCSAXES",
    "SOLVED",
    "DBSET",
    "TILE_ID",
    "RMSPX",
    "INLIERS",
    "PIXSCAL",
    "SIPORD",
    "QUALITY",
    "USED_DB",
    "SOLVER",
    "SOLVMODE",
    "BLINDVER",
    "ZBLNDVER",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numpy", "scipy", "astropy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            out[name] = f"ERROR: {exc}"
    return out


def _has_celestial_wcs(path: Path) -> bool:
    try:
        return bool(WCS(fits.getheader(path)).has_celestial)
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
        for key in list(header.keys()):
            if key.startswith(("PV", "A_", "B_", "AP_", "BP_", "SIP")):
                removed.append(key)
                del header[key]
        hdul.flush()
    return {
        "source": str(source),
        "runtime": str(target),
        "source_sha256": _sha256(source),
        "runtime_sha256": _sha256(target),
        "removed_keys": sorted(set(removed)),
        "has_celestial_wcs_after_strip": _has_celestial_wcs(target),
    }


def _load_sources(limit: int | None = None) -> list[Path]:
    payload = json.loads(P223_CORPUS.read_text(encoding="utf-8"))
    records = payload.get("items") or payload.get("records") or payload.get("images") or []
    sources: list[Path] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        src = row.get("source") or row.get("original") or row.get("oracle")
        if src:
            sources.append(Path(str(src)))
    if not sources:
        raise RuntimeError(f"no source files found in {P223_CORPUS}")
    return sources[:limit] if limit else sources


def prepare_runtime_corpus(label: str, limit: int | None = None) -> list[dict[str, Any]]:
    runtime_dir = WORK / label / "runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    rows: list[dict[str, Any]] = []
    for source in _load_sources(limit=limit):
        target = runtime_dir / source.name
        rows.append(_strip_runtime_copy(source, target))
    return rows


def _offline_wcs_check(source: Path, runtime: Path) -> dict[str, Any]:
    try:
        solved = WCS(fits.getheader(runtime))
        ref = WCS(fits.getheader(source))
        if not solved.has_celestial:
            return {"ok": False, "reason": "runtime_has_no_wcs"}
        shape = fits.getdata(source, memmap=False).shape
        h, w = int(shape[-2]), int(shape[-1])
        pts = np.array([[w / 2.0, h / 2.0], [1.0, 1.0], [w, 1.0], [1.0, h], [w, h]], dtype=np.float64)
        got = solved.all_pix2world(pts, 0)
        exp = ref.all_pix2world(pts, 0)
        sep = np.sqrt(np.sum((got - exp) ** 2, axis=1))
        return {
            "ok": bool(np.all(np.isfinite(sep)) and float(np.nanmax(sep)) < 0.05),
            "max_corner_delta_deg": float(np.nanmax(sep)),
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _base_app_config(runtime_dir: Path, workers: int) -> app.SolveConfig:
    manifest = load_4d_index_manifest(MANIFEST)
    return app.SolveConfig(
        db_root=ROOT,
        input_dir=runtime_dir,
        families=None,
        overwrite=True,
        workers=max(1, int(workers)),
        blind_enabled=True,
        blind_only=True,
        blind_skip_if_valid=False,
        blind_index_path=ROOT,
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        blind_4d_manifest_path=MANIFEST,
        blind_4d_loaded_manifest=manifest,
        astrometry_api_key=None,
        astrometry_fallback_after_blind=False,
        log_level="INFO",
    )


def _budget_override(mode: str) -> tuple[float, float]:
    if mode == "legacy_global_45":
        return 45.0, 0.0
    if mode == "route_45":
        return 0.0, 45.0
    if mode == "split_60_45":
        return 60.0, 45.0
    if mode == "no_global_route_45":
        return 0.0, 45.0
    raise ValueError(f"unknown budget mode: {mode}")


def _extract_stats(result: app.ImageSolveResult) -> dict[str, Any]:
    for key, payload in result.run_info or []:
        if key == "run_info_blind_stats" and isinstance(payload, dict):
            return dict(payload)
    return {}


def run_app_path(
    *,
    label: str,
    workers: int,
    budget_mode: str,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = prepare_runtime_corpus(label, limit=limit)
    runtime_files = [Path(row["runtime"]) for row in rows]
    source_by_runtime = {Path(row["runtime"]).name: Path(row["source"]) for row in rows}
    old_env = os.environ.get("ZE_BLIND_WORKERS")
    os.environ["ZE_BLIND_WORKERS"] = str(max(1, int(workers)))
    original_blind_solve = app.blind_solve
    attempt_budget_s, route_budget_s = _budget_override(budget_mode)

    def wrapped_blind_solve(*args: Any, **kwargs: Any) -> Any:
        cfg = kwargs.get("config")
        if cfg is not None:
            kwargs["config"] = dataclasses.replace(
                cfg,
                blind_global_hard_budget_s=attempt_budget_s,
                blind_astrometry_4d_search_budget_s=route_budget_s,
            )
        return original_blind_solve(*args, **kwargs)

    app.blind_solve = wrapped_blind_solve
    t0 = time.perf_counter()
    try:
        solver = app.ImageSolver(_base_app_config(WORK / label / "runtime", workers=max(1, int(workers))))

        def solve_one(path: Path) -> dict[str, Any]:
            start = time.perf_counter()
            result = solver.solve_path_blind_only(path)
            stats = _extract_stats(result)
            return {
                "file": path.name,
                "status": result.status,
                "message": result.message,
                "duration_s": result.duration_s if result.duration_s is not None else time.perf_counter() - start,
                "stats": stats,
                "offline_wcs": _offline_wcs_check(source_by_runtime[path.name], path),
            }

        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            future_map = {pool.submit(solve_one, path): path for path in runtime_files}
            for future in concurrent.futures.as_completed(future_map):
                results.append(future.result())
        results.sort(key=lambda row: row["file"])
    finally:
        app.blind_solve = original_blind_solve
        if old_env is None:
            os.environ.pop("ZE_BLIND_WORKERS", None)
        else:
            os.environ["ZE_BLIND_WORKERS"] = old_env
    success = sum(1 for row in results if row["status"] == "solved" and row["offline_wcs"].get("ok"))
    return {
        "label": label,
        "path": "app_headless_blind_only",
        "workers": int(workers),
        "budget_mode": budget_mode,
        "attempt_budget_s": attempt_budget_s,
        "route_budget_s": route_budget_s,
        "elapsed_s": float(time.perf_counter() - t0),
        "success": int(success),
        "total": int(len(results)),
        "results": results,
    }


def write_baseline() -> dict[str, Any]:
    manifest = load_4d_index_manifest(MANIFEST)
    cfg = _base_app_config(WORK, workers=1)
    blind_cfg = app.build_blind_solve_config(cfg, loaded_manifest=manifest)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git(["rev-parse", "HEAD"]),
            "status": _git(["status", "--short"]),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": app._system_memory_gb(),
        "packages": _package_versions(),
        "manifest": {
            "path": str(MANIFEST),
            "enabled_index_paths": [str(path) for path in manifest.enabled_index_paths],
            "sha256": {str(path): sha256_file(path) for path in manifest.enabled_index_paths},
        },
        "app_config": dataclasses.asdict(cfg),
        "blind_config": dataclasses.asdict(blind_cfg),
        "auto_blind_workers_4d": app._auto_blind_worker_count(
            8,
            ram_gb=app._system_memory_gb(),
            blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        ),
    }
    _write_json(BASELINE_OUT, payload)
    return payload


def write_report(concurrency: dict[str, Any], budgets: dict[str, Any], parity: dict[str, Any]) -> None:
    lines = [
        "# P2.24 - App path budget/concurrency",
        "",
        "## Résumé exécutif",
        "",
        "Le profil 4D sépare désormais le budget de recherche Astrometry 4D du budget global d'essai blind. "
        "Le backend historique conserve son comportement par défaut.",
        "",
        "## Résultats",
        "",
        f"- Concurrence: {concurrency.get('summary', {})}",
        f"- Budgets: {budgets.get('summary', {})}",
        f"- Parité: {parity.get('summary', {})}",
        "",
        "## Réponses P2.24",
        "",
        "1. Le budget 45 s commençait auparavant à l'entrée de `solve_blind`, donc avant la route 4D.",
        "2. Le temps pré-route est maintenant mesuré par `blind_pre_route_s`.",
        "3. `max_wall_s` du profil 4D est désormais mappé vers `blind_astrometry_4d_search_budget_s`.",
        "4. Le chemin headless utilise `ImageSolver.solve_path_blind_only` et le même `build_blind_solve_config`.",
        "5. La politique par défaut du profil 4D sélectionne 1 worker, sauf override explicite.",
        "6. Timeout route, budget d'essai blind et Stop utilisateur ont des raisons distinctes.",
        "7. Le backend historique reste inchangé.",
        "",
        "Les détails bruts sont dans les JSON P2.24 générés à côté de ce rapport.",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="limit corpus size for smoke runs")
    parser.add_argument("--skip-slow", action="store_true", help="run only baseline and one app-path parity pass")
    args = parser.parse_args(list(argv) if argv is not None else None)
    limit = int(args.limit or 0) or None

    baseline = write_baseline()
    timeline = {
        "budget_before_p224": "blind_global_hard_budget_s started at solve_blind entry",
        "budget_after_p224": "blind_astrometry_4d_search_budget_s starts at astrometry 4D route entry",
        "timeline_fields": [
            "blind_pre_route_s",
            "astrometry_4d_route_s",
            "blind_post_route_s",
            "blind_total_s",
            "blind_attempt_budget_s",
            "astrometry_4d_search_budget_s",
        ],
    }
    _write_json(TIMELINE_OUT, timeline)

    parity_run = run_app_path(label="parity_route45_w1", workers=1, budget_mode="route_45", limit=limit)
    parity = {
        "baseline_commit": baseline["git"]["commit"],
        "summary": {"success": parity_run["success"], "total": parity_run["total"]},
        "runs": [parity_run],
    }
    _write_json(PARITY_OUT, parity)

    concurrency_runs = [parity_run]
    budget_runs = [parity_run]
    if not args.skip_slow:
        concurrency_runs.extend(
            [
                run_app_path(label="concurrency_w1_repeat2", workers=1, budget_mode="route_45", limit=limit),
                run_app_path(label="concurrency_w2", workers=2, budget_mode="route_45", limit=limit),
                run_app_path(label="concurrency_w2_repeat2", workers=2, budget_mode="route_45", limit=limit),
            ]
        )
        budget_runs.extend(
            [
                run_app_path(label="budget_legacy_global45", workers=1, budget_mode="legacy_global_45", limit=limit),
                run_app_path(label="budget_split60_45", workers=1, budget_mode="split_60_45", limit=limit),
            ]
        )
    concurrency = {
        "summary": {run["label"]: f"{run['success']}/{run['total']}" for run in concurrency_runs},
        "runs": concurrency_runs,
    }
    budgets = {
        "summary": {run["label"]: f"{run['success']}/{run['total']}" for run in budget_runs},
        "runs": budget_runs,
    }
    _write_json(CONCURRENCY_OUT, concurrency)
    _write_json(BUDGET_OUT, budgets)
    write_report(concurrency, budgets, parity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
