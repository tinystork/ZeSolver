#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK_DIRS = (
    ROOT / "zesolver" / "core",
    ROOT / "zesolver" / "settings",
    ROOT / "zesolver" / "catalog_library",
)
FORBIDDEN_IMPORT_PREFIXES = ("PySide6",)
FORBIDDEN_EXACT_NAMES = {
    "ImageSolver",
    "BatchSolver",
    "SolveRunner",
    "QtCore",
    "QtGui",
    "QtWidgets",
}
FORBIDDEN_CALLS = {"spec_from_file_location", "module_from_spec"}


@dataclass(frozen=True, slots=True)
class Violation:
    path: Path
    line: int
    message: str


def main() -> int:
    violations: list[Violation] = []
    for base in CHECK_DIRS:
        for path in sorted(base.rglob("*.py")):
            violations.extend(check_file(path))
    if violations:
        for violation in violations:
            rel = violation.path.relative_to(ROOT)
            print(f"{rel}:{violation.line}: {violation.message}")
        return 1
    print("core boundary check: OK")
    return 0


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_check_import(path, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            violations.extend(_check_import(path, node.lineno, module))
            for alias in node.names:
                if alias.name in FORBIDDEN_EXACT_NAMES:
                    violations.append(Violation(path, node.lineno, f"forbidden symbol import: {alias.name}"))
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in FORBIDDEN_CALLS:
                violations.append(Violation(path, node.lineno, f"forbidden dynamic import helper: {call_name}"))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_EXACT_NAMES:
                violations.append(Violation(path, node.lineno, f"forbidden application/gui symbol: {node.id}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "zesolver.py" in node.value:
                violations.append(Violation(path, getattr(node, "lineno", 0), "forbidden root entrypoint path reference"))
    return violations


def _check_import(path: Path, line: int, module: str) -> list[Violation]:
    violations: list[Violation] = []
    if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES):
        violations.append(Violation(path, line, f"forbidden GUI import: {module}"))
    if module == "zesolver.py":
        violations.append(Violation(path, line, "forbidden root entrypoint import"))
    return violations


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
