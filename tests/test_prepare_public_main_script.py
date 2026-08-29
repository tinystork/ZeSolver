from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_public_main.sh"
BUILDER_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_public_tree.py"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _has_python_bytecode(root: Path) -> bool:
    return any(
        path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        for path in root.rglob("*")
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_public_publication_fixture(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    source = tmp_path / "ZeSolver"
    main_worktree = tmp_path / "ZeSolver-main"

    _git(tmp_path, "init", "--bare", str(origin))
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "checkout", "-q", "-b", "test")
    _git(source, "remote", "add", "origin", str(origin))

    tools_dir = source / "tools"
    tools_dir.mkdir()
    shutil.copy2(SCRIPT_PATH, tools_dir / "prepare_public_main.sh")
    shutil.copy2(BUILDER_PATH, tools_dir / "build_public_tree.py")
    os.chmod(tools_dir / "prepare_public_main.sh", 0o755)

    _write(source / "README.md", "fixture\n")
    _write(source / "ZeSolver_Ballad.md", "English ballad\n")
    _write(source / "la_ballade_de_ZeSolver.md", "Ballade francaise\n")
    # Root compatibility shim
    _write(
        source / "zesolver.py",
        "from zesolver._app import main\n"
        "if __name__ == '__main__':\n"
        "    import sys\n"
        "    main(sys.argv[1:])\n",
    )
    # Package launcher (gui_scripts entry point)
    _write(
        source / "zesolver" / "_app.py",
        "def main(argv=None):\n"
        "    pass\n",
    )
    _write(source / "zesolver" / "__init__.py", "VALUE = 'zesolver'\n")
    _write(
        source / "zesolver" / "gpu_diagnostic.py",
        "import json\n"
        "print(json.dumps({'ok': True}))\n",
    )
    _write(source / "zesolver" / "api" / "__init__.py", "")
    _write(
        source / "zesolver" / "api" / "v1" / "__init__.py",
        'API_VERSION = "1.2"\n'
        'API_MAJOR = 1\n'
        '\n'
        'class _Info:\n'
        '    api_version = "1.2"\n'
        '    supported_capabilities = ("near_solve", "blind_solve", "wcs_write", "gpu", "cancel")\n'
        '\n'
        'class _Report:\n'
        '    api_version = "1.2"\n'
        '    operational = False\n'
        '\n'
        'def get_api_info():\n'
        '    return _Info()\n'
        '\n'
        'def readiness():\n'
        '    return _Report()\n'
    )

    _write(source / "zeblindsolver" / "__init__.py", "VALUE = 'zeblindsolver'\n")
    _write(source / "zewcs290" / "__init__.py", "VALUE = 'zewcs290'\n")
    _write(
        source / "packaging" / "public_manifest.txt",
        "\n".join(
            [
                "README.md",
                "ZeSolver_Ballad.md",
                "la_ballade_de_ZeSolver.md",
                "zesolver/_app.py",
                "zesolver.py",
                "zesolver/**",
                "zeblindsolver/**",
                "zewcs290/**",
            ]
        )
        + "\n",
    )

    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "user.name=Test",
        "commit",
        "-q",
        "-m",
        "fixture test",
    )
    _git(source, "push", "-u", "origin", "test")
    _git(source, "branch", "main")
    _git(source, "push", "-u", "origin", "main")

    subprocess.run(
        ["git", "clone", "--branch", "main", str(origin), str(main_worktree)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _write(main_worktree / "zesolver" / "__pycache__" / "old.cpython-313.pyc", "bytecode\n")
    _write(main_worktree / "zeblindsolver" / "legacy.pyo", "optimized bytecode\n")
    _git(main_worktree, "add", ".")
    _git(
        main_worktree,
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "user.name=Test",
        "commit",
        "-q",
        "-m",
        "bad bytecode publication",
    )
    _git(main_worktree, "push", "origin", "main")

    return source, main_worktree


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


def test_prepare_public_main_keeps_python_bytecode_out_of_public_tree(
    tmp_path: Path,
) -> None:
    source, main_worktree = _make_public_publication_fixture(tmp_path)

    proc = subprocess.run(
        [
            str(source / "tools" / "prepare_public_main.sh"),
            "--yes",
            "--main-worktree",
            str(main_worktree),
        ],
        check=True,
        cwd=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "MAIN CANDIDATE READY" in proc.stdout
    assert not _has_python_bytecode(main_worktree)
    assert main_worktree.joinpath("ZeSolver_Ballad.md").is_file()
    assert main_worktree.joinpath("la_ballade_de_ZeSolver.md").is_file()
    assert not main_worktree.joinpath("tools").exists()

    status = _git(main_worktree, "status", "--porcelain=v1").stdout
    assert "D  zeblindsolver/legacy.pyo" in status or " D zeblindsolver/legacy.pyo" in status
    assert (
        "D  zesolver/__pycache__/old.cpython-313.pyc" in status
        or " D zesolver/__pycache__/old.cpython-313.pyc" in status
    )
