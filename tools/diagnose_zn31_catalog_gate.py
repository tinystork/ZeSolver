#!/usr/bin/env python3
"""ZN3.1 catalog-gate runner.

This wrapper regenerates the shared ZN3.1 reports and prints the catalog gate.
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
    catalog_gate = payload["catalog_gate"]
    print(json.dumps({k: v for k, v in catalog_gate.items()}, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
