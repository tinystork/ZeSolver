from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_public_main.sh"


def test_prepare_public_main_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)


def test_prepare_public_main_help_documents_publication_safety() -> None:
    proc = subprocess.run(
        [str(SCRIPT_PATH), "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "--dry-run" in proc.stdout
    assert "--main-worktree PATH" in proc.stdout
    assert "requires local test HEAD == origin/test" in proc.stdout
    assert "requires local main == origin/main" in proc.stdout
    assert "never commits" in proc.stdout
    assert "never pushes" in proc.stdout
    assert "never performs git merge or force push" in proc.stdout
