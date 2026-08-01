# RC-PUBLIC-1 - Minimal public main distribution

Date: 2026-08-02

## 1. Initial Git State

Commands recorded before modification:

```text
git status --short
<clean>

git branch -vv
  main bd3c57a [origin/main] Merge test into main
* test e04f129 [origin/test] Remove obsolete benchmark and handoff artifacts

git log --oneline --decorate -10
e04f129 (HEAD -> test, origin/test) Remove obsolete benchmark and handoff artifacts
02d63c9 Record Linux GPU provisioning validation
618ef00 Promote safe source-managed GPU provisioning
a7670df Add GPU provisioning install feedback
1bc6ccd Fix GPU provisioning thread lifecycle
4b73cc8 Add guided GPU diagnostic provisioning
c58f158 Record macOS CI validation status
caed8f2 Fix macOS CI portability failures
386d2b1 Fix Python 3.11 checksum compatibility
98a8de3 Improve macOS CI compile diagnostics

HEAD = e04f129dba6e48b44b69d772963116b5b71caaf3
origin/test = e04f129dba6e48b44b69d772963116b5b71caaf3
git diff --check = clean
```

The source branch was clean and aligned with `origin/test`.

## 2. Principle

`main` must not be produced by merging `test`.

The public branch is generated from a positive allowlist:

```text
packaging/public_manifest.txt
```

The generator is:

```text
tools/build_public_tree.py
```

The generator remains on `test` and is intentionally excluded from the generated public tree.

## 3. Runtime Audit

Files required for the public source distribution:

- `zesolver.py`: GUI and public batch entry point.
- `zesolver/**`: GUI, settings, CatalogLibrary, distribution install, engine selection, pipeline, GPU diagnostic/provisioning, macOS preflight, WCS cleanup integration.
- `zeblindsolver/**`: ZeBlind 4D runtime, index manifest loading, ZeNear metadata solver, index builders exposed by public CLI entry points.
- `zewcs290/**`: ASTAP/HNSKY `.290`/`.1476` catalogue reading and bundled `layouts.json`.
- `zewcscleaner.py`: standalone WCS cleanup helper.
- `zeindexcheck.py`: user-facing index validation helper.
- `config/zeblind_4d_experimental_manifest.json`: default compact Blind 4D manifest used by runtime fallback paths.
- `icon/ZSicon.*`: GUI icons and cross-platform fallback formats.
- `pyproject.toml`: install metadata and public entry points.
- `README.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE.md`, `legal/**`: public documentation, license and third-party data terms.
- `docs/zeblind_astrometry_4d_experimental.md`: public ZeBlind 4D operating note.

Excluded after audit:

- `tests/`: development validation only.
- `tools/`: internal diagnostics, benchmark and release-projection tooling only.
- `reports/`: local diagnostic outputs.
- `docs/stabilization/`: release engineering history, not end-user documentation.
- `docs/architecture/`: internal architecture notes.
- `.github/`: CI belongs to `test`, not generated public `main`.
- `packaging/`: development and release tooling; not needed to install or run the source tree.
- `AGENT.md`, `memory.md`, `followup.md`, `structure.txt`: local agent/project context.
- `tests/corpus/**` and oracle JSON files: development corpus only.
- `zedatabase..py`: legacy database explorer with extra GUI/Matplotlib assumptions, not required for ZeSolver install/runtime.
- `install.ps1` and `requirements.txt`: development install helpers; public source install is via `pyproject.toml`.

No file is kept only as a precaution; every included file is either runtime code, package data,
public metadata, icon/resource data, or essential user documentation.

## 4. Manifest Behavior

The manifest accepts relative paths and controlled recursive globs. The builder rejects:

- absolute paths;
- `..` traversal;
- missing manifest entries;
- forbidden paths in the output tree;
- dirty source worktrees;
- source branches other than `test`.

Generated output always includes:

```text
ZESOLVER_SOURCE_REVISION
```

with branch, SHA, `origin/test`, and generation date.

## 5. Publication Procedure

Recommended first-publication workflow:

```bash
cd /home/tristan/.openclaw/workspace/projects/ZeSolver
git status --short
git switch test
git pull --ff-only

python tools/build_public_tree.py \
  --source /home/tristan/.openclaw/workspace/projects/ZeSolver \
  --destination /tmp/zesolver-public-staging \
  --report /tmp/zesolver-public-staging-report.json

git worktree add ../ZeSolver-main main
rsync -a --delete --exclude='.git' /tmp/zesolver-public-staging/ ../ZeSolver-main/

cd ../ZeSolver-main
git status --short
git diff --stat
git diff --check
```

Do not run `git merge test` on `main`.

Do not commit or push `main` until Tristan explicitly approves the displayed diff.

When approved:

```bash
git add -A
git commit -m "Publish minimal public beta tree"
git push origin main
```

Tags and GitHub Releases should be created from `main` after this generated tree is accepted.

## 6. Public README Policy

`README.md` now states:

- `main` is a generated public distribution tree;
- development and pull requests target `test`;
- direct edits to `main` are not accepted;
- internal tests/tools/reports are intentionally absent from `main`.

## 7. Validation Results

Pre-commit targeted test:

```text
python -m pytest -q tests/test_public_tree_builder.py
3 passed
```

Post-commit public export:

```text
python tools/build_public_tree.py \
  --source /home/tristan/.openclaw/workspace/projects/ZeSolver \
  --destination /tmp/zesolver-public-staging \
  --report /tmp/zesolver-public-report.json

public_tree files=128 bytes=6040963 source_bytes=11537067
```

Size comparison:

- tracked `test` files: 11,537,067 bytes;
- generated public tree: 6,040,963 bytes;
- public tree is about 52.4% of tracked `test` by byte size.

Forbidden-path validation:

```text
tests False
tools False
reports False
packaging False
.github False
docs/stabilization False
docs/architecture False
AGENT.md False
memory.md False
followup.md False
structure.txt False
requirements.txt False
install.ps1 False
```

Export compile validation:

```text
python -m compileall zesolver zeblindsolver zewcs290
python -m py_compile zesolver.py zewcscleaner.py zeindexcheck.py
OK
```

Venv source-install validation from the exported tree:

```text
python -m venv /tmp/zesolver-public-venv
/tmp/zesolver-public-venv/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/zesolver-public-venv/bin/python -m pip install -e "/tmp/zesolver-public-staging[gui]"
OK
```

Smoke tests from the exported tree and venv:

```text
import zesolver, zeblindsolver, zewcs290
OK

zewcs290 layouts hnsky_1476 rings = 36
default Blind 4D manifest exists = true

python -m zesolver.gpu_diagnostic --json --show-install-plan
OK, CPU fallback non-blocking, SOURCE_MANAGED plan visible in venv

python zesolver.py --help
OK

Qt offscreen resource smoke:
QApplication + ThemeController + icon/ZSicon.png = OK

AST import scan for tests/tools/docs/packaging:
violations []
```

Main worktree preparation:

```text
git worktree add ../ZeSolver-main main
rsync -a --delete --exclude='.git' /tmp/zesolver-public-staging/ ../ZeSolver-main/
git -C ../ZeSolver-main diff --check
OK

git -C ../ZeSolver-main status --short | wc -l
411

git -C ../ZeSolver-main diff --stat
396 files changed, 6868 insertions(+), 114551 deletions(-)
```

No commit or push was performed on `main`.

## 8. Risks Residuals

- The public branch still exposes source installation only; packaged app production remains a
  separate release mission.
- `pyproject.toml` project naming remains historical (`zewcs290`) even though the public product
  name is ZeSolver. This was not changed in RC-PUBLIC-1 to avoid packaging churn.
- `requirements.txt` remains on `test` for development workflows but is not part of public `main`.
- The first actual `main` update remains pending explicit human approval after reviewing the
  generated worktree diff.

## 9. Final Hash

The exact containing commit hash is reported by `git log -1` after the final amend. A Git commit
cannot embed its own final SHA without changing that SHA.

## 10. Verdict

```text
RC_PUBLIC1_MINIMAL_MAIN_READY
MAIN_GENERATED_FROM_TEST
TEST_REMAINS_CANONICAL
PUBLIC_TREE_SMOKE_VALIDATED
MAIN_PUSH_PENDING_USER_APPROVAL
```
