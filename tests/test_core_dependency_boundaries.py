from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_dependency_boundary_guard_passes() -> None:
    result = subprocess.run(
        [sys.executable, "tools/check_core_boundaries.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "core boundary check: OK" in result.stdout
