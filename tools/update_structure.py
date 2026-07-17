#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "structure.txt"
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "ASTAP-main",
    "__pycache__",
    "astrometry-indexes",
    "astrometry-main",
    "astrometry-net-main",
    "indexes",
    "reports",
    "testrunlog",
    "logs",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".npz", ".fit", ".fits", ".fts", ".log", ".zip"}
EXCLUDED_NAMES = {".DS_Store"}


def main() -> int:
    lines = ["ZeSolver repository structure", "", "."]
    _walk(ROOT, prefix="", lines=lines)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated {OUTPUT.relative_to(ROOT)}")
    return 0


def _walk(path: Path, *, prefix: str, lines: list[str]) -> None:
    entries = [entry for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())) if _include(entry)]
    for index, entry in enumerate(entries):
        connector = "`-- " if index == len(entries) - 1 else "|-- "
        lines.append(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if index == len(entries) - 1 else "|   "
            _walk(entry, prefix=prefix + extension, lines=lines)


def _include(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return False
    if path.is_dir():
        return path.name not in EXCLUDED_DIRS and not path.name.startswith(".openclaw")
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name.endswith("~"):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
