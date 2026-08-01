# RC-MAC-1 - macOS Compatibility Audit and CI

Date: 2026-08-01
Branch: `test`

Status:

```text
MACOS_COMPATIBILITY_AUDIT_PASSED
MACOS_CI_READY_FOR_EXECUTION
MACOS_RUNTIME_VALIDATION_PENDING
```

This mission does not claim `MACOS_RUNTIME_VALIDATED`,
`MACOS_PRODUCTION_READY`, or `PRODUCTION_READY_FOR_MACOS`.

## 1. Initial Git State

Initial checks:

```text
git status --short
  clean
git branch -vv
  * test 5ec7fe6 [origin/test] Align Blind4D coverage warning semantics
git log --oneline --decorate -10
  5ec7fe6 (HEAD -> test, origin/test) Align Blind4D coverage warning semantics
  57caf7f agent mis a jour
  5858b51 Close startup wizard catalog activation transaction
  43fa8e6 Add system light and dark theme selector
  f42079f Add automatic catalog download resume controls
  a9a7c4c fix passage dossier de travail au gui durant le wizard
  2782f10 Fix ASTAP and CatalogLibrary resource composition
  d1c4a2e Add multisource parallel catalog downloads
  06cf1a2 validation wizard
  73f13d7 Add first-run wizard and align development dependencies
git rev-parse HEAD
  5ec7fe63c7e7041e42a14d3f4ce1d295333a3d2b
git rev-parse origin/test
  5ec7fe63c7e7041e42a14d3f4ce1d295333a3d2b
git diff --check
  clean
```

The local branch matched `origin/test` at mission start.

## 2. Scope

Covered:

- static macOS compatibility audit;
- platform/path handling;
- Finder opening adapter;
- PyInstaller resource-resolution paths;
- multiprocessing `spawn` compatibility;
- Qt offscreen startup;
- CatalogLibrary miniature fixture;
- downloader cache/resume behavior under Unicode paths;
- CPU fallback when CuPy/CUDA is absent;
- dedicated GitHub Actions macOS workflow.

Not covered:

- public `.app` package;
- DMG, Apple signing, notarization, or Developer ID;
- physical Mac validation;
- downloading the full official D50 library in CI;
- scientific solver algorithm changes.

## 3. Platforms Actually Tested

Local automated validation was executed on:

```text
Linux-6.12.96+deb13-amd64-x86_64-with-glibc2.41
Python 3.13.5
PySide6 6.11.1
NumPy 2.5.1
Astropy 8.0.1
SciPy 1.18.0
scikit-image 0.26.0
astroalign 2.6.2
threadpoolctl 3.6.0
```

macOS GitHub Actions has been configured but not executed from this local
session because the mission explicitly forbids pushing without a separate
request. The workflow prints its runtime macOS version and architecture during
execution.

## 4. GitHub Actions Runner

Workflow:

```text
.github/workflows/macos-ci.yml
```

Configured runner:

```text
runs-on: macos-latest
python-version: "3.11"
```

Rationale:

- `macos-latest` is the current official floating GitHub-hosted macOS runner
  label and avoids pinning an obsolete image;
- Python 3.11 is inside ZeSolver's declared `>=3.10` support range and remains
  conservative for PySide6, Astropy, SciPy, and scikit-image wheels;
- the workflow captures `python --version`, `uname -a`,
  `platform.platform()`, and `platform.machine()` so the real macOS release and
  CPU architecture are visible in each run.

Actual run URL, duration, architecture, skips, and warnings are pending until
the workflow is pushed and executed.

## 5. Platform Branch Audit

Searched:

```text
sys.platform
platform.system()
os.name
darwin
win32
linux
```

Findings:

- `zesolver.py` resolves icons through source, PyInstaller `_MEIPASS`, and
  platform-specific `.icns`, `.ico`, and `.png` fallbacks.
- `zesolver.py` handles macOS RSS units separately from Linux.
- process-mode defaults already reduce macOS `spawn` overhead by preferring
  threads for the legacy executor default on Darwin.
- `zesolver/catalog_library/paths.py` maps Darwin/mac/macOS to `macos`, stores
  cache data under `~/Library/Caches/ZeSolver/catalogs`, and stores installed
  libraries under `~/ZeSolverCatalog/libraries`.
- `packaging/pyinstaller/build.py` selects the `.icns` icon on Darwin and uses
  PyInstaller's correct path separator for `--add-data`.

No macOS path was found falling through to Windows-specific behavior.

## 6. File-System and Path Review

The user-facing catalogue storage paths are:

```text
macOS cache:    ~/Library/Caches/ZeSolver/catalogs
macOS library:  ~/ZeSolverCatalog/libraries
```

The audit covered paths with:

- spaces;
- accented characters;
- apostrophes;
- Unicode directory names.

Correction made:

- `validate_library_parent(..., platform_name="Darwin")` previously checked
  for a literal path component named `.app`; it did not reject paths under a
  real bundle component such as `ZeSolver.app`. The check now rejects any
  destination component ending in `.app`.

The downloader and extraction code use Python file APIs, `pathlib.Path`,
`zipfile`, hashlib/SHA-256 validation, `.part` files, and atomic promotion of
validated files. No dependency on `wget`, `curl`, `unzip`, or `sha256sum` was
required for the tested paths.

## 7. Finder and URL Opening

`open_in_file_manager()` now uses a testable adapter:

```text
Darwin -> ["open", path]
Linux  -> ["xdg-open", path]
Windows -> os.startfile(path)
```

The adapter passes paths as argument lists and does not require `shell=True`.
The new tests verify routing without opening Finder in CI.

## 8. Resources and PyInstaller

Resource resolution remains compatible with:

- source checkout;
- virtual environment;
- future PyInstaller `_MEIPASS` extraction root.

The preflight now checks that a usable application icon exists in `icon/`:

```text
ZSicon.icns
ZSicon.png
ZSicon.ico
```

The mission does not build the final macOS package.

## 9. Multiprocessing and Spawn

macOS uses `spawn` semantics for multiprocessing. The audit covered:

- `multiprocessing`;
- `ProcessPoolExecutor`;
- process cancellation tokens;
- functions passed to subprocesses;
- manager/event/dict serialization.

Corrections made:

- `ProcessCancellationController` now accepts an optional multiprocessing
  context and creates its manager from that context when provided.
- The cancellation-token test now explicitly uses
  `multiprocessing.get_context("spawn")`.
- `zesolver.macos_preflight` now validates a `ProcessPoolExecutor` with
  `mp_context=multiprocessing.get_context("spawn")`.

Representative spawn tests pass locally.

## 10. Qt and Threads

Qt review:

- GUI workers remain `QThread` based and communicate through Qt signals.
- The existing main-window guard still checks that direct widget updates happen
  on the application thread for queued append operations.
- Distribution, wizard, catalogue validation, and download work remain outside
  direct widget mutation from worker code.

Added checks:

- `zesolver.macos_preflight --strict-gui` creates and closes a Qt widget with
  `QT_QPA_PLATFORM=offscreen`;
- `tests/test_macos_compatibility.py` creates a QApplication-compatible widget,
  applies the theme controller, and closes it without leaving theme state
  polluted for later tests.

## 11. Dependencies

Required and compatible in the audited environment:

- NumPy;
- Astropy;
- SciPy;
- scikit-image;
- astroalign;
- rich;
- threadpoolctl for tests/dev tooling.

GUI dependency:

- PySide6, optional through the `gui` extra and required by the macOS CI.

Optional with fallback:

- CuPy/CUDA. The `gpu` extra is Linux-gated in `pyproject.toml`; CuPy absence is
  accepted by the macOS preflight and does not block CPU operation.

Potential platform risks left for packaging:

- BLAS/OpenMP wheel behavior on Apple Silicon versus Intel;
- PySide6 plugin collection in a final `.app`;
- PyInstaller hidden imports and Qt plugin paths in the signed bundle.

## 12. Apple Silicon and Intel

No repository code path was found that requires:

- CUDA on macOS;
- a Windows/Linux executable;
- x86-only binaries;
- hard-coded pointer sizes;
- endian-specific behavior.

The GitHub Actions workflow intentionally prints `platform.machine()` so the
actual runner architecture is recorded. Apple Silicon or Intel runtime support
must still be confirmed by the first macOS CI run and later by a physical Mac
test of the real user artifact.

## 13. Downloader and Archive Handling

The audit reviewed the distribution path used by the startup wizard:

- parallel component downloads;
- HTTP range resume;
- `.part` preservation;
- SHA-256 validation;
- extraction into a staging path;
- final promotion;
- cache reuse;
- cancellation and pause states.

Added coverage:

- a resumable download test writes into a cache path containing spaces,
  accented characters, and an apostrophe;
- the test verifies a `Range: bytes=<partial>-` request and final SHA-256
  promotion.

No full official library download is performed in CI.

## 14. CI Workflow

The new workflow performs:

1. checkout;
2. Python setup;
3. virtualenv creation;
4. controlled pip upgrade;
5. `pip install -e ".[dev,gui]"`;
6. platform information logging;
7. `compileall zesolver zeblindsolver tools`;
8. import smoke test;
9. `python -m zesolver.macos_preflight --strict-gui`;
10. `python zesolver.py --help`;
11. targeted macOS tests;
12. full CI-compatible pytest suite.

Environment:

```text
QT_QPA_PLATFORM=offscreen
MPLBACKEND=Agg
PYTHONUNBUFFERED=1
```

The workflow triggers on:

```text
pull_request
push to test
workflow_dispatch
```

It does not use secrets, upload releases, sign code, notarize, or download the
official multi-GB library.

## 15. Tests Added

New file:

```text
tests/test_macos_compatibility.py
```

Coverage:

- macOS cache/library paths;
- `.app` bundle destination rejection;
- Finder command routing without opening Finder;
- spawn worker;
- macOS preflight spawn and FITS path checks;
- Qt offscreen widget and theme controller startup/shutdown;
- CatalogLibrary miniature fixture under Unicode path;
- resumable downloader under Unicode/apostrophe cache path;
- CuPy absence as an optional condition.

Updated:

- `tests/test_process_cancellation_token.py` now uses a spawn context;
- `tests/test_zn310b_gui_fallback_dataset.py` now skips only when the external
  ZN310B corpus paths are absent, so GitHub runners do not fail on missing
  local-only data. When the external corpus is present, the integrity assertions
  still run and can fail.

## 16. Local Results

Compilation:

```text
.venv/bin/python -m compileall zesolver zeblindsolver tools
OK
```

macOS preflight, executed locally under Linux with macOS-path and spawn checks:

```text
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  .venv/bin/python -m zesolver.macos_preflight --strict-gui

summary: 15 ok, 0 warning(s), 0 failure(s)
```

Targeted macOS lot:

```text
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  .venv/bin/python -m pytest -q \
    tests/test_macos_compatibility.py \
    tests/test_catalog_library_paths.py \
    tests/test_process_cancellation_token.py \
    tests/test_gui_theme.py \
    tests/test_catalog_distribution.py \
    tests/test_catalog_distribution_multisource.py \
    tests/test_catalog_library_blind4d_integration.py

79 passed in 14.72s
```

Full CI-compatible local suite:

```text
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  .venv/bin/python -m pytest -q \
    --ignore tests/test_zn310b_gui_fallback_dataset.py

799 passed, 37 skipped, 17 warnings in 68.11s
```

Raw local full suite:

```text
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg .venv/bin/python -m pytest -q

2 failed, 813 passed, 37 skipped, 17 warnings in 58.92s
```

The two failures are local external-fixture integrity failures:

```text
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
```

They reproduce on a clean detached worktree at `origin/test`:

```text
git worktree add --detach /tmp/zesolver-origin-test-rcmac1 origin/test
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  /home/tristan/.openclaw/workspace/projects/ZeSolver/.venv/bin/python \
  -m pytest -q \
  tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha \
  tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs

2 failed
```

The failure data points to `/home/tristan/near_bench_cmp30/...`, not repository
code modified by RC-MAC-1.

Whitespace check:

```text
git diff --check
OK
```

## 17. GitHub Actions Results

Not executed yet from this local session.

Reason:

- pushing is explicitly forbidden without a separate request;
- no GitHub Actions run can exist until the commit is pushed or a workflow is
  manually dispatched on GitHub.

Therefore this report does not claim `MACOS_CI_VALIDATED`.

To validate CI after push:

```text
Workflow: macOS CI
Branch:   test
Runner:   macos-latest
Python:   3.11
```

Record after execution:

- run URL;
- commit hash;
- macOS version;
- architecture;
- duration;
- tests;
- skips;
- warnings;
- final status.

## 18. Skips and Limitations

CI-compatible suite intentionally excludes:

```text
tests/test_zn310b_gui_fallback_dataset.py
```

Reason:

- it references an absolute external corpus under `/home/tristan`;
- that corpus will not exist on a clean macOS GitHub runner;
- ignoring this file in CI does not hide local corpus mutation because the raw
  local suite still runs it when the corpus is present.

No generic `skip if darwin` was added.

Remaining limitations:

- no physical Mac run;
- no packaged `.app`;
- no signing or notarization;
- no Apple Silicon/Intel runtime statement until CI and community testing run;
- no full official D50 library download in CI.

## 19. Files Modified

```text
.github/workflows/macos-ci.yml
CHANGELOG.md
docs/stabilization/rc_mac1_compatibility_audit_and_ci_20260801.md
tests/test_macos_compatibility.py
tests/test_process_cancellation_token.py
tests/test_zn310b_gui_fallback_dataset.py
zesolver/cancellation.py
zesolver/catalog_library/paths.py
zesolver/macos_preflight.py
```

`AGENT.md` was not updated because the macOS CI has not yet actually succeeded.

## 20. Packaging Follow-Up

Future macOS packaging mission should cover:

- PyInstaller `.app` build;
- Qt plugin collection and verification;
- `.icns` validation in the packaged artifact;
- signing;
- notarization;
- DMG;
- physical Mac launch;
- real D50 library selection/install;
- user-visible documentation for experimental macOS support.

## 21. Final Commit

This report is included in the final RC-MAC-1 commit. The exact hash is
recorded in the final mission response after the commit is created.

## 22. Verdict

```text
MACOS_COMPATIBILITY_AUDIT_PASSED
MACOS_CI_READY_FOR_EXECUTION
MACOS_RUNTIME_VALIDATION_PENDING
```
