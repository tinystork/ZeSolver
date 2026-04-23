#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _pick_icon(icon_dir: Path) -> Path | None:
    if sys.platform == "darwin":
        preferred = ["ZSicon.icns", "ZSicon.png", "ZSicon.jpeg", "ZSicon.jpg", "ZSicon.ico"]
    elif sys.platform.startswith("win"):
        preferred = ["ZSicon.ico", "ZSicon.png", "ZSicon.jpeg", "ZSicon.jpg", "ZSicon.icns"]
    else:
        preferred = ["ZSicon.png", "ZSicon.ico", "ZSicon.jpeg", "ZSicon.jpg", "ZSicon.icns"]
    for name in preferred:
        candidate = icon_dir / name
        if candidate.is_file():
            return candidate
    return None


def _add_data_arg(src: Path, dest: str) -> str:
    sep = ";" if sys.platform.startswith("win") else ":"
    return f"{src}{sep}{dest}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ZeSolver GUI bundle with PyInstaller")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--onefile", action="store_true", help="Build onefile instead of onedir")
    parser.add_argument("--clean", action="store_true", help="Remove previous build/dist folders first")
    parser.add_argument("--name", default="ZeSolver")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    entry = repo_root / "zesolver.py"
    icon_dir = repo_root / "icon"

    if not entry.is_file():
        print(f"[FAIL] missing entry script: {entry}")
        return 2

    pyinstaller_bin = shutil.which("pyinstaller")
    if pyinstaller_bin:
        base_cmd = [pyinstaller_bin]
    else:
        base_cmd = [sys.executable, "-m", "PyInstaller"]

    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"

    if args.clean:
        shutil.rmtree(dist_dir, ignore_errors=True)
        shutil.rmtree(build_dir, ignore_errors=True)

    cmd: list[str] = [
        *base_cmd,
        "--noconfirm",
        "--windowed",
        "--name",
        args.name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--paths",
        str(repo_root),
    ]

    if args.onefile:
        cmd.append("--onefile")

    icon = _pick_icon(icon_dir)
    if icon is not None:
        cmd += ["--icon", str(icon)]

    # Bundle runtime icon assets for the in-app window icon loader.
    for name in ["ZSicon.ico", "ZSicon.icns", "ZSicon.png", "ZSicon.jpeg", "ZSicon.jpg"]:
        src = icon_dir / name
        if src.is_file():
            cmd += ["--add-data", _add_data_arg(src, "icon")]

    # Main entry
    cmd.append(str(entry))

    print("[INFO] Running:")
    print(" ".join(f'"{p}"' if " " in p else p for p in cmd))

    proc = subprocess.run(cmd, cwd=str(repo_root))
    if proc.returncode != 0:
        print(f"[FAIL] PyInstaller exited with code {proc.returncode}")
        return proc.returncode

    target = dist_dir / args.name
    if sys.platform == "darwin":
        app = target.with_suffix(".app")
        if app.exists():
            print(f"[OK] App bundle: {app}")
        else:
            print(f"[OK] Dist output: {target}")
    elif sys.platform.startswith("win"):
        exe = target / f"{args.name}.exe"
        if exe.exists():
            print(f"[OK] Windows executable: {exe}")
        else:
            print(f"[OK] Dist output: {target}")
    else:
        print(f"[OK] Dist output: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
