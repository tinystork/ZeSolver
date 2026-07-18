# P3A GUI Pipeline Integration Report

## Scope

P3A progressively wires the existing GUI local FITS path through:

```text
GuiSolveController -> EngineSelector -> PipelineGuiRunner -> BatchSolverPipeline -> SolverPipeline
```

Legacy routes remain available:

- `ImageSolver`;
- `BatchSolver`;
- `SolveRunner`;
- Astrometry.net web;
- raster sidecars;
- process/hybrid Near scheduling;
- historical fallback behavior.

## Baseline Before Modification

Environment used:

```text
ZESOLVER_CORPUS_ROOT=/home/tristan/zesolver-regression-corpus
ZESOLVER_ZN310B_ROOT=/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840
ZESOLVER_ASTAP_ROOT=/opt/astap
ZESOLVER_BLIND4D_MANIFEST=<repo>/config/zeblind_4d_experimental_manifest.json
ZESOLVER_LEGACY_INDEX_ROOT=/home/tristan/zesolver_index
```

Results:

```text
core boundary check: OK
hermetic: 383 passed, 1 skipped, 8 deselected, 1 warning
corpus: 6 passed, 2 skipped, 384 deselected, runner PASS
full pytest: 389 passed, 3 skipped, 37 warnings
compileall: OK
git diff --check: OK
```

Warning inventory before P3A:

| Warning | Count | Category | Status |
| ------- | ----: | -------- | ------ |
| `multiprocessing.popen_fork.DeprecationWarning` in `tests/test_failures.py`, `tests/test_quad_storage.py`, `tests/test_synthetic.py` | 36 | existing threading/fork deprecation | known before P3A |
| `astropy.io.fits.card.VerifyWarning` in `tests/test_p220_manifest_loader.py` | 1 | existing FITS warning | known before P3A |

## Implementation

Added:

- `zesolver/gui_pipeline/requests.py`;
- `settings_adapter.py`;
- `controller.py`;
- `pipeline_runner.py`;
- `legacy_runner.py`;
- `progress_adapter.py`;
- `result_adapter.py`;
- `lifecycle.py`;
- `shadow_validation.py`.

Changed:

- local GUI `SolveRunner` now runs `GuiSolveController`;
- `ZESOLVER_GUI_ENGINE=auto|pipeline|legacy` controls the initial route policy;
- local FITS `AUTO` can select `PIPELINE`;
- legacy remains injected from `zesolver.py` and wraps the existing `BatchSolver`;
- `AstrometryRunner` remains unchanged and separate.

## Final Automated Validation

P3A targeted:

```text
tests/test_gui_settings_adapter.py
tests/test_gui_engine_controller.py
tests/test_gui_progress_adapter.py
tests/test_gui_result_adapter.py
tests/test_gui_pipeline_runner.py
tests/test_gui_legacy_runner.py
tests/test_gui_cancellation.py
tests/test_gui_lifecycle.py
tests/test_gui_shadow_validation.py
tests/test_gui_controller_zn310b.py

17 passed in 217.49s
```

ZN3.10B through GUI controller:

```text
tests/test_gui_controller_zn310b.py
1 passed in 217.04s
selected_engine=PIPELINE
8 results
8 solved
0 failed
0 skipped
0 cancelled
CONTROL=3/3 Near
NOHINT=3/3 Blind 4D
BADHINT=2/2 Blind 4D
```

Additional routing/boundary smoke:

```text
GUI fast tests + engine selection + core dependency boundaries: 31 passed
core boundary check: OK
```

Regression:

```text
corpus: 7 passed, 2 skipped, 400 deselected, runner PASS
hermetic: 399 passed, 1 skipped, 9 deselected, 1 warning, runner PASS
full pytest: 406 passed, 3 skipped, 37 warnings
compileall: OK
git diff --check: OK
```

Remaining expected skips:

```text
g05 absent sous /opt/astap
S50 index or frame not configured
blind4d_p29_232329 non mappé
```

Warning inventory after P3A:

| Warning | Count | Category | P3A status |
| ------- | ----: | -------- | ---------- |
| `multiprocessing.popen_fork.DeprecationWarning` | 36 | existing threading/fork deprecation | unchanged |
| `astropy.io.fits.card.VerifyWarning` | 1 | existing FITS warning | unchanged |

No new warning category was introduced by P3A.

## Known Limits

- Manual GUI visual validation was not executed in this session.
- Raster and web routes intentionally remain legacy.
- Process/hybrid Near scheduling intentionally remains legacy.
- The GUI visual design and simplification are untouched.

## Decision

```text
READY_FOR_PIPELINE_DEFAULT_BUT_MANUAL_GUI_VALIDATION_PENDING
```
