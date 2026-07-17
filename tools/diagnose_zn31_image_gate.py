#!/usr/bin/env python3
"""ZN3.1 image-gate runner.

This thin wrapper keeps the requested entry point while the shared oracle
parsers live in ``diagnose_zn31_astap_oracles.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from diagnose_zn31_astap_oracles import REPO_ROOT, build_reports


class Args:
    reports_dir = REPO_ROOT / "reports"
    runtime_dir = REPO_ROOT / "reports" / "zn1_runtime"
    dump_dir = REPO_ROOT / "reports" / "zn31_astap_dumps"
    zenear_star_dir = REPO_ROOT / "reports" / "zn1_star_lists"
    astap_bin = REPO_ROOT / "ASTAP-main" / "command-line_version" / "astap_cli"
    astap_db = Path("/opt/astap")
    family = "d50"


def main() -> int:
    payload = build_reports(Args())
    image_gate = payload["image_gate"]
    print(json.dumps({k: v for k, v in image_gate.items()}, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
