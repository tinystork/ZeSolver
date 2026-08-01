#!/usr/bin/env python3
"""Build the minimal public ZeSolver tree from the canonical test branch."""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_MANIFEST = Path("packaging/public_manifest.txt")
REVISION_FILE = "ZESOLVER_SOURCE_REVISION"

FORBIDDEN_PARTS = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "reports",
    "tests",
    "tools",
}

FORBIDDEN_EXACT = {
    "AGENT.md",
    "memory.md",
    "followup.md",
    "structure.txt",
    "zedatabase..py",
    "install.ps1",
    "requirements.txt",
}

FORBIDDEN_PREFIXES = {
    "docs/architecture",
    "docs/stabilization",
    "packaging",
}

FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    ".bak",
}


@dataclass(frozen=True)
class GitState:
    branch: str
    sha: str
    origin_test: str | None
    clean: bool
    status: str


@dataclass(frozen=True)
class BuildReport:
    source: Path
    destination: Path
    manifest: Path
    source_branch: str
    source_sha: str
    source_origin_test: str | None
    generated_at_utc: str
    check_only: bool
    copied_files: tuple[str, ...]
    copied_bytes: int
    source_bytes: int
    forbidden_present: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "manifest": str(self.manifest),
            "source_branch": self.source_branch,
            "source_sha": self.source_sha,
            "source_origin_test": self.source_origin_test,
            "generated_at_utc": self.generated_at_utc,
            "check_only": self.check_only,
            "copied_file_count": len(self.copied_files),
            "copied_bytes": self.copied_bytes,
            "source_bytes": self.source_bytes,
            "forbidden_present": list(self.forbidden_present),
            "copied_files": list(self.copied_files),
        }


class PublicTreeError(RuntimeError):
    pass


def _run_git(source: Path, args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise PublicTreeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def read_git_state(source: Path) -> GitState:
    inside = _run_git(source, ["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        raise PublicTreeError(f"not a Git worktree: {source}")
    branch = _run_git(source, ["branch", "--show-current"])
    sha = _run_git(source, ["rev-parse", "HEAD"])
    origin_test = _run_git(source, ["rev-parse", "origin/test"], check=False) or None
    status = _run_git(source, ["status", "--porcelain=v1"])
    return GitState(branch=branch, sha=sha, origin_test=origin_test, clean=not status, status=status)


def assert_clean_test_source(source: Path) -> GitState:
    state = read_git_state(source)
    if state.branch != "test":
        raise PublicTreeError(f"public export must be built from branch test, got {state.branch!r}")
    if not state.clean:
        raise PublicTreeError("public export source must be clean; git status --short:\n" + state.status)
    return state


def _normalize_manifest_entry(entry: str) -> str:
    text = entry.strip()
    if not text or text.startswith("#"):
        return ""
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts:
        raise PublicTreeError(f"invalid manifest path: {entry!r}")
    return posix.as_posix()


def read_manifest(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise PublicTreeError(f"manifest not found: {path}")
    entries: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        entry = _normalize_manifest_entry(raw)
        if entry:
            entries.append(entry)
    if not entries:
        raise PublicTreeError(f"manifest is empty: {path}")
    return tuple(entries)


def _is_forbidden(rel: str) -> bool:
    posix = PurePosixPath(rel)
    parts = set(posix.parts)
    if parts & FORBIDDEN_PARTS:
        return True
    if rel in FORBIDDEN_EXACT or posix.name in FORBIDDEN_EXACT:
        return True
    if any(rel == prefix or rel.startswith(prefix + "/") for prefix in FORBIDDEN_PREFIXES):
        return True
    if any(rel.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        return True
    if posix.name.endswith(".egg-info") or ".egg-info" in parts:
        return True
    return False


def expand_manifest(source: Path, entries: Iterable[str]) -> tuple[str, ...]:
    selected: set[str] = set()
    missing: list[str] = []
    for entry in entries:
        if any(char in entry for char in "*?["):
            matches = []
            if entry.endswith("/**"):
                root_rel = entry[:-3].rstrip("/")
                root = source / root_rel
                if not root.exists():
                    missing.append(entry)
                    continue
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        rel = path.relative_to(source).as_posix()
                        if not _is_forbidden(rel):
                            matches.append(rel)
            else:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        rel = path.relative_to(source).as_posix()
                        if fnmatch.fnmatch(rel, entry) and not _is_forbidden(rel):
                            matches.append(rel)
            if not matches:
                missing.append(entry)
            selected.update(matches)
            continue

        path = source / entry
        if not path.is_file():
            missing.append(entry)
            continue
        if _is_forbidden(entry):
            raise PublicTreeError(f"manifest entry is forbidden in public tree: {entry}")
        selected.add(entry)
    if missing:
        raise PublicTreeError("manifest entries not found: " + ", ".join(missing))
    return tuple(sorted(selected))


def copy_files(source: Path, staging: Path, rel_paths: Iterable[str]) -> None:
    for rel in rel_paths:
        src = source / rel
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mode = src.stat().st_mode
        if mode & stat.S_IXUSR:
            dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def scan_forbidden(root: Path) -> tuple[str, ...]:
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if _is_forbidden(rel):
            found.append(rel)
    return tuple(found)


def tree_size(root: Path, *, skip_git: bool = True) -> int:
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if skip_git and ".git" in rel_parts:
            continue
        total += path.stat().st_size
    return total


def git_tracked_size(root: Path) -> int:
    output = _run_git(root, ["ls-files", "-z"])
    total = 0
    for raw in output.split("\0"):
        if not raw:
            continue
        path = root / raw
        if path.is_file():
            total += path.stat().st_size
    return total


def write_revision(staging: Path, state: GitState) -> str:
    generated_at = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    text = (
        f"branch={state.branch}\n"
        f"sha={state.sha}\n"
        f"origin_test={state.origin_test or ''}\n"
        f"generated_at_utc={generated_at}\n"
    )
    (staging / REVISION_FILE).write_text(text, encoding="utf-8")
    return generated_at


def replace_destination(staging: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(
            f".{destination.name}.old-{os.getpid()}-{int(_dt.datetime.now().timestamp())}"
        )
        if backup.exists():
            raise PublicTreeError(f"temporary backup path already exists: {backup}")
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_public_tree(
    *,
    source: Path,
    destination: Path,
    manifest: Path,
    check_only: bool,
) -> BuildReport:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    manifest = manifest if manifest.is_absolute() else source / manifest
    state = assert_clean_test_source(source)
    entries = read_manifest(manifest)
    selected = expand_manifest(source, entries)

    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=str(destination.parent)) as tmp:
        staging = Path(tmp) / destination.name
        staging.mkdir(parents=True)
        copy_files(source, staging, selected)
        generated_at = write_revision(staging, state)
        forbidden = scan_forbidden(staging)
        if forbidden:
            raise PublicTreeError("forbidden files in public tree: " + ", ".join(forbidden))
        copied_files = tuple(sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()))
        report = BuildReport(
            source=source,
            destination=destination,
            manifest=manifest,
            source_branch=state.branch,
            source_sha=state.sha,
            source_origin_test=state.origin_test,
            generated_at_utc=generated_at,
            check_only=check_only,
            copied_files=copied_files,
            copied_bytes=tree_size(staging, skip_git=False),
            source_bytes=git_tracked_size(source),
            forbidden_present=forbidden,
        )
        if not check_only:
            final_staging = destination.parent / f".{destination.name}.ready-{os.getpid()}"
            if final_staging.exists():
                raise PublicTreeError(f"temporary staging path already exists: {final_staging}")
            staging.rename(final_staging)
            replace_destination(final_staging, destination)
        return report


def write_report(path: Path, report: BuildReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd(), help="clean ZeSolver test worktree")
    parser.add_argument("--destination", type=Path, required=True, help="public tree destination")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="public manifest path")
    parser.add_argument("--check-only", action="store_true", help="validate and stage without replacing destination")
    parser.add_argument("--report", type=Path, help="write a JSON inventory report")
    args = parser.parse_args(argv)

    try:
        report = build_public_tree(
            source=args.source,
            destination=args.destination,
            manifest=args.manifest,
            check_only=args.check_only,
        )
    except PublicTreeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.report:
        write_report(args.report, report)
    print(
        f"public_tree files={len(report.copied_files)} "
        f"bytes={report.copied_bytes} "
        f"source_bytes={report.source_bytes} "
        f"destination={report.destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
