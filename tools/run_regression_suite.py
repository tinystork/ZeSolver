#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": round(time.perf_counter() - start, 3),
    }


def env_summary() -> dict[str, str | None]:
    keys = [
        "ZESOLVER_CORPUS_ROOT",
        "ZESOLVER_ZN310B_ROOT",
        "ZESOLVER_ASTAP_ROOT",
        "ZESOLVER_BLIND4D_MANIFEST",
        "ZESOLVER_LEGACY_INDEX_ROOT",
    ]
    return {key: os.environ.get(key) for key in keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ZeSolver regression baselines.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hermetic", action="store_true", help="run tests not requiring external corpus data")
    group.add_argument("--corpus", action="store_true", help="run corpus/external/slow tests")
    group.add_argument("--all", action="store_true", help="run the full pytest suite")
    parser.add_argument("--output-json", type=Path, help="write a machine-readable summary")
    args = parser.parse_args(argv)

    if args.hermetic:
        marker = "not external_catalog and not corpus and not slow"
    elif args.corpus:
        marker = "external_catalog or corpus or slow"
    else:
        marker = None

    print("ZeSolver regression runner")
    print(f"Python: {sys.executable}")
    print("External data:")
    for key, value in env_summary().items():
        print(f"  {key}={value or '<unset>'}")

    commands: list[list[str]] = []
    if marker:
        commands.append([sys.executable, "-m", "pytest", "-m", marker, "-q"])
    else:
        commands.append([sys.executable, "-m", "pytest", "-q"])
    commands.append(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "zeblindsolver",
            "zewcs290",
            "zesolver",
            "tools",
            "tests",
            "zesolver.py",
            "zewcscleaner.py",
            "zedatabase..py",
            "zeindexcheck.py",
        ]
    )

    results = [run(cmd) for cmd in commands]
    status = 0 if all(result["returncode"] == 0 for result in results) else 1
    payload = {
        "mode": "hermetic" if args.hermetic else "corpus" if args.corpus else "all",
        "env": env_summary(),
        "results": results,
        "status": "PASS" if status == 0 else "FAIL",
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
