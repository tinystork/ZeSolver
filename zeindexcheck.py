from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from zeblindsolver.quad_index_builder import validate_index

try:  # Optional GUI dependencies
    from PySide6 import QtCore, QtWidgets
except Exception:  # pragma: no cover - optional dependency
    QtCore = QtWidgets = None  # type: ignore[assignment]


def _scan_database_root(db_root: Path) -> Dict[str, Any]:
    """Perform lightweight checks on an ASTAP/HNSKY database folder."""
    result: Dict[str, Any] = {
        "exists": False,
        "families": [],
        "total_shards": 0,
        "samples": [],
    }
    if not db_root:
        result["message"] = "Database root not provided"
        return result
    db_root = db_root.expanduser().resolve()
    if not db_root.exists():
        result["message"] = f"{db_root} does not exist"
        return result
    if not db_root.is_dir():
        result["message"] = f"{db_root} is not a directory"
        return result
    result["exists"] = True
    families: Dict[str, int] = {}
    samples: list[str] = []
    for entry in db_root.rglob("*"):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in {".1476", ".290"}:
            continue
        result["total_shards"] += 1
        family = entry.parent.name
        families[family] = families.get(family, 0) + 1
        if len(samples) < 5:
            samples.append(str(entry.relative_to(db_root)))
    result["families"] = sorted(families.items())
    result["samples"] = samples
    if result["total_shards"] == 0:
        result["message"] = f"{db_root} contains no .1476/.290 shards"
    else:
        result["message"] = "Database scan completed"
    return result


def _format_index_report(report: Dict[str, Any]) -> str:
    parts = [
        f"Manifest OK: {report.get('manifest_ok')}",
        f"Tile count: {report.get('manifest_tile_count')}",
        f"Quads present: {', '.join(report.get('present_quads') or [])}",
        f"Quads missing: {', '.join(report.get('missing_quads') or [])}",
    ]
    if report.get("bad_empty_rings"):
        parts.append(f"Rings with empty tiles: {report['bad_empty_rings']}")
    if report.get("db_root_mismatch"):
        parts.append("Warning: DB root recorded in manifest differs from provided db_root")
    if report.get("tile_key_mismatch"):
        parts.append("Warning: tile key mismatch between manifest and DB scan")
    return "\n".join(parts)


def _format_database_report(report: Dict[str, Any]) -> str:
    if not report.get("exists"):
        return report.get("message", "Database path not found")
    parts = [
        report.get("message", "Database scan completed"),
        f"Families detected: {', '.join(f'{name} ({count})' for name, count in report.get('families') or [])}",
        f"Shard count: {report.get('total_shards', 0)}",
    ]
    samples = report.get("samples") or []
    if samples:
        parts.append("Sample shards:")
        parts.extend(f"  - {sample}" for sample in samples)
    return "\n".join(parts)


def run_cli(index_root: Path | None, db_root: Path | None) -> int:
    if index_root is None:
        print("Provide --index PATH (and optionally --db PATH)", file=sys.stderr)
        return 2
    report = validate_index(index_root, db_root=db_root if db_root else None)
    print("Index report:")
    print(_format_index_report(report))
    if db_root is not None:
        db_report = _scan_database_root(db_root)
        print("\nDatabase report:")
        print(_format_database_report(db_report))
    return 0


class IndexCheckWindow(QtWidgets.QWidget):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ZeIndexCheck")
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.index_edit = QtWidgets.QLineEdit()
        self.db_edit = QtWidgets.QLineEdit()
        browse_index = QtWidgets.QPushButton("Browse…")
        browse_db = QtWidgets.QPushButton("Browse…")

        def _browse(target: QtWidgets.QLineEdit) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
            if directory:
                target.setText(directory)

        browse_index.clicked.connect(lambda: _browse(self.index_edit))
        browse_db.clicked.connect(lambda: _browse(self.db_edit))

        index_row = QtWidgets.QHBoxLayout()
        index_row.addWidget(self.index_edit, 1)
        index_row.addWidget(browse_index)
        db_row = QtWidgets.QHBoxLayout()
        db_row.addWidget(self.db_edit, 1)
        db_row.addWidget(browse_db)

        form.addRow("Index root", index_row)
        form.addRow("Database root (optional)", db_row)
        layout.addLayout(form)

        self.run_button = QtWidgets.QPushButton("Run check")
        layout.addWidget(self.run_button)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, 1)

        self.run_button.clicked.connect(self._run_check)

    def _run_check(self) -> None:
        index_text = self.index_edit.text().strip()
        if not index_text:
            QtWidgets.QMessageBox.warning(self, "ZeIndexCheck", "Select an index root")
            return
        index_root = Path(index_text)
        db_text = self.db_edit.text().strip()
        db_root = Path(db_text) if db_text else None
        try:
            report = validate_index(index_root, db_root=db_root)
            lines = ["Index report:", _format_index_report(report)]
            if db_root is not None:
                db_report = _scan_database_root(db_root)
                lines.append("")
                lines.append("Database report:")
                lines.append(_format_database_report(db_report))
            self.output.setPlainText("\n".join(lines))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "ZeIndexCheck", str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check ZeSolver index/database integrity")
    parser.add_argument("--index", type=Path, help="Path to index root (manifest + hash tables)")
    parser.add_argument("--db", type=Path, help="Path to ASTAP/HNSKY database root")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode even if PySide6 is available")
    args = parser.parse_args(argv)
    if args.cli or QtWidgets is None or (args.index and args.db):
        return run_cli(args.index, args.db)
    # GUI mode
    app = QtWidgets.QApplication(sys.argv)
    window = IndexCheckWindow()
    if args.index:
        window.index_edit.setText(str(Path(args.index).expanduser()))
    if args.db:
        window.db_edit.setText(str(Path(args.db).expanduser()))
    window.resize(640, 360)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
