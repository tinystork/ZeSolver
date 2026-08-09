# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : GPL V3 (voir pyproject.toml / repository metadata)               ║
# ║                                                                                   ║
# ║ Remerciements amont :                                                             ║
# ║ - ASTAP, par Han Kleijn                                                           ║
# ║ - Astrometry.net, par Dustin Lang, David W. Hogg, Keir Mierle, et al.            ║
# ║                                                                                   ║
# ║ Description FR :                                                                  ║
# ║ Ce code sert à transformer des nuages de photons en solutions WCS et en images   ║
# ║ astronomiques exploitables. Merci de créditer les auteurs et projets amont lors   ║
# ║ de toute réutilisation.                                                           ║
# ║                                                                                   ║
# ║ EN Description:                                                                    ║
# ║ This code helps turn clouds of photons into usable WCS solutions and astronomical ║
# ║ imagery outputs. Please credit both project authors and upstream references when  ║
# ║ reusing this work.                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
# """

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _proc_square(x: int) -> int:
    return x * x


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return False, f"import failed: {exc}"
    ver = getattr(mod, "__version__", None)
    if ver:
        return True, f"ok (version {ver})"
    return True, "ok"


def _check_process_pool() -> tuple[bool, str]:
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            value = pool.submit(_proc_square, 7).result(timeout=10)
        if value != 49:
            return False, f"unexpected result {value}"
        return True, "ok (spawn)"
    except Exception as exc:
        return False, f"process pool failed: {exc}"


def _check_zesolver_help(python_exe: str, repo_root: Path) -> tuple[bool, str]:
    zesolver_py = repo_root / "zesolver/_app.py"
    if not zesolver_py.exists():
        return False, f"missing {zesolver_py}"
    try:
        proc = subprocess.run(
            [python_exe, str(zesolver_py), "--help"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return False, f"launch failed: {exc}"
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        msg = stderr[-1] if stderr else f"exit {proc.returncode}"
        return False, msg
    return True, "ok"


def _check_temp_paths_and_fits() -> tuple[bool, str]:
    try:
        import numpy as np
        from astropy.io import fits

        from zesolver.catalog_library.paths import default_cache_root, default_library_parent

        with tempfile.TemporaryDirectory(prefix="ZeSolver macOS éspace ") as tmp:
            home = Path(tmp) / "Utilisateur Test É"
            cache = default_cache_root(platform_name="Darwin", home=home)
            library_parent = default_library_parent(platform_name="Darwin", home=home)
            if cache != home / "Library" / "Caches" / "ZeSolver" / "catalogs":
                return False, f"unexpected cache path: {cache}"
            if library_parent != home / "ZeSolverCatalog" / "libraries":
                return False, f"unexpected library path: {library_parent}"
            cache.mkdir(parents=True, exist_ok=True)
            library_parent.mkdir(parents=True, exist_ok=True)
            fits_path = cache / "préflight sample's frame.fit"
            fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.uint16)).writeto(fits_path)
            with fits.open(fits_path, memmap=False) as hdul:
                shape = tuple(hdul[0].data.shape)
            if shape != (4, 4):
                return False, f"unexpected FITS shape: {shape}"
        return True, "ok"
    except Exception as exc:
        return False, f"path/FITS check failed: {exc}"


def _check_qt_offscreen() -> tuple[bool, str]:
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        widget = QtWidgets.QWidget()
        widget.setWindowTitle("ZeSolver macOS preflight")
        widget.show()
        app.processEvents()
        widget.close()
        app.processEvents()
        return True, "ok"
    except Exception as exc:
        return False, f"Qt offscreen failed: {exc}"


def _check_static_resources(repo_root: Path) -> tuple[bool, str]:
    icon_dir = repo_root / "icon"
    if not icon_dir.is_dir():
        return False, "icon directory missing"
    if not any((icon_dir / name).is_file() for name in ("ZSicon.icns", "ZSicon.png", "ZSicon.ico")):
        return False, "no usable application icon found"
    return True, "ok"


def _check_cupy_optional() -> tuple[bool, str]:
    try:
        importlib.import_module("cupy")
    except Exception:
        return True, "not installed (CPU fallback expected)"
    return True, "installed"


def main() -> int:
    parser = argparse.ArgumentParser(description="ZeSolver macOS compatibility preflight")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="Path to ZeSolver repository root")
    parser.add_argument("--strict-gui", action="store_true", help="Treat missing PySide6 as failure")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()

    print("== ZeSolver macOS preflight ==")
    print(f"platform: {platform.platform()}")
    print(f"python:   {sys.version.split()[0]} ({sys.executable})")
    print(f"repo:     {repo_root}")

    checks: list[tuple[str, bool, str, bool]] = []

    if sys.version_info < (3, 10):
        checks.append(("Python >= 3.10", False, "required", False))
    else:
        checks.append(("Python >= 3.10", True, "ok", False))

    for mod in ["numpy", "astropy", "scipy", "skimage", "astroalign", "threadpoolctl"]:
        ok, detail = _check_import(mod)
        checks.append((f"import {mod}", ok, detail, False))

    ok_gui, detail_gui = _check_import("PySide6")
    checks.append(("import PySide6 (GUI)", ok_gui, detail_gui, not args.strict_gui))

    ok_pool, detail_pool = _check_process_pool()
    checks.append(("multiprocessing ProcessPool spawn", ok_pool, detail_pool, False))

    ok_paths, detail_paths = _check_temp_paths_and_fits()
    checks.append(("macOS user/cache/temp paths + FITS", ok_paths, detail_paths, False))

    ok_static, detail_static = _check_static_resources(repo_root)
    checks.append(("static resources", ok_static, detail_static, False))

    ok_qt, detail_qt = _check_qt_offscreen()
    checks.append(("Qt offscreen widget", ok_qt, detail_qt, not args.strict_gui))

    ok_help, detail_help = _check_zesolver_help(sys.executable, repo_root)
    checks.append(("zesolver/_app.py --help", ok_help, detail_help, False))

    ok_cupy, detail_cupy = _check_cupy_optional()
    checks.append(("CuPy optional", ok_cupy, detail_cupy, True))

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        checks.append(("nvidia-smi", True, f"found at {nvidia}", True))
    else:
        checks.append(("nvidia-smi", True, "not found (normal on most macOS hosts)", True))

    failures = 0
    warnings = 0
    for label, ok, detail, optional in checks:
        if ok:
            print(f"[OK]   {label}: {detail}")
            continue
        if optional:
            warnings += 1
            print(f"[WARN] {label}: {detail}")
        else:
            failures += 1
            print(f"[FAIL] {label}: {detail}")

    print("-")
    print(f"summary: {len(checks) - failures - warnings} ok, {warnings} warning(s), {failures} failure(s)")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
