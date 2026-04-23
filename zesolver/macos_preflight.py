# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : MIT (voir pyproject.toml / repository metadata)               ║
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
import platform
import shutil
import subprocess
import sys
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
        with ProcessPoolExecutor(max_workers=1) as pool:
            value = pool.submit(_proc_square, 7).result(timeout=10)
        if value != 49:
            return False, f"unexpected result {value}"
        return True, "ok"
    except Exception as exc:
        return False, f"process pool failed: {exc}"


def _check_zesolver_help(python_exe: str, repo_root: Path) -> tuple[bool, str]:
    zesolver_py = repo_root / "zesolver.py"
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

    for mod in ["numpy", "astropy", "scipy", "skimage", "astroalign"]:
        ok, detail = _check_import(mod)
        checks.append((f"import {mod}", ok, detail, False))

    ok_gui, detail_gui = _check_import("PySide6")
    checks.append(("import PySide6 (GUI)", ok_gui, detail_gui, not args.strict_gui))

    ok_pool, detail_pool = _check_process_pool()
    checks.append(("multiprocessing ProcessPool", ok_pool, detail_pool, False))

    ok_help, detail_help = _check_zesolver_help(sys.executable, repo_root)
    checks.append(("zesolver.py --help", ok_help, detail_help, False))

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
