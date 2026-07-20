# P2B-2C Core Decoupling & GUI Readiness Gate

Decision:

```text
READY_FOR_P3A_GUI_PIPELINE_INTEGRATION
```

## Scope

P2B-2C removed the inverse dependency from `zesolver/core` to the root
application entrypoint `zesolver.py`.

No solver algorithm, astrometric threshold, profile `v1`, catalog format,
manifest format, fixture, index, GUI route, or legacy rollback path was changed.

## Changes

Code:

```text
zesolver/solver_config/__init__.py
zesolver/solver_config/blind.py
zesolver/solver_config/compatibility.py
zesolver/engine_selection.py
tools/check_core_boundaries.py
tools/update_structure.py
```

Tests:

```text
tests/test_blind_config_builder_parity.py
tests/test_core_import_isolation.py
tests/test_core_dependency_boundaries.py
tests/test_core_without_entrypoint.py
tests/test_engine_selection.py
```

Docs:

```text
docs/architecture/core_dependency_boundary.md
docs/architecture/gui_pipeline_capability_matrix.md
docs/architecture/engine_selection.md
structure.txt
```

## Blind Config Builder

`build_blind_solve_config()` and `ensure_loaded_4d_manifest()` now live in the
package module:

```text
zesolver.solver_config.blind
```

The root `zesolver.py` imports and exposes the same functions for legacy tests
and tools. There is one implementation.

`ProductionBlindSolverPort` imports the shared builder directly and no longer
uses:

```text
importlib.util
spec_from_file_location
module_from_spec
zesolver_entrypoint_blind_port
repo_root / "zesolver.py"
```

The temporary compatibility input object is now explicit:

```text
BlindConfigInputs
build_blind_config_inputs()
```

## Boundary Guard

`tools/check_core_boundaries.py` checks:

```text
zesolver/core/
zesolver/settings/
zesolver/catalog_library/
```

It rejects root entrypoint dynamic loading, PySide6/Qt imports, and legacy
application symbols (`ImageSolver`, `BatchSolver`, `SolveRunner`) in the new
core/config/catalog boundary.

Result:

```text
core boundary check: OK
```

## Import Isolation

Verified in subprocess:

```text
from zesolver.core import SolverPipeline
from zesolver.core.blind_port import ProductionBlindSolverPort
```

The import succeeds while PySide6 imports are blocked. No module is loaded from
the root `zesolver.py`, and the synthetic packaging smoke test succeeds after
copying only package directories without the root entrypoint file.

## Engine Selection

Added a pure `AUTO` / `PIPELINE` / `LEGACY` policy in:

```text
zesolver.engine_selection
```

Initial AUTO policy:

```text
FITS local validated capabilities -> PIPELINE
raster / web / fallback web / process-hybrid batch / adaptive hints / unknown -> LEGACY
explicit PIPELINE unsupported -> supported=false, no silent fallback
explicit LEGACY -> LEGACY
READY_PARTIAL Blind 4D coverage -> warning, never all-sky promotion
```

The selector is not wired into the GUI yet.

## Validation

Baseline before modification:

```text
Hermetic: 362 passed, 1 skipped, 8 deselected, 1 warning, runner PASS
Corpus: 6 passed, 2 skipped, 363 deselected, runner PASS
Full pytest: 368 passed, 3 skipped, 37 warnings
compileall: OK
git diff --check: OK
```

Targeted P2B-2C:

```text
tests/test_blind_config_builder_parity.py
tests/test_core_import_isolation.py
tests/test_core_dependency_boundaries.py
tests/test_engine_selection.py
tests/test_core_without_entrypoint.py

21 passed in 9.00s
```

Production Blind and ZN3.10B with external data:

```text
tests/test_blind_production_port.py
tests/test_blind_port_config_parity.py
tests/test_blind_port_result_parity.py
tests/test_solver_pipeline_blind_production.py
tests/test_solver_pipeline_zn310b_production.py
tests/test_batch_pipeline_zn310b.py

13 passed in 302.49s
```

Final validation:

```text
tools/check_core_boundaries.py: OK

Corpus:
6 passed, 2 skipped, 384 deselected in 286.48s
runner PASS

Hermetic:
383 passed, 1 skipped, 8 deselected, 1 warning in 17.49s
runner PASS

Full pytest:
389 passed, 3 skipped, 37 warnings in 240.52s

compileall:
OK
```

Expected skips remain:

```text
g05 absent under /opt/astap
S50 index or frame not configured
blind4d_p29_232329 not mapped
```

## Risks

Remaining legacy-only capabilities are documented in
`docs/architecture/gui_pipeline_capability_matrix.md`:

```text
process/hybrid Near batch
io_concurrency enforcement
raster TIFF/PNG/JPEG sidecars
Astrometry.net web fallback
adaptive inter-image hints
GUI lifecycle/progress integration
```

These are not presented as pipeline-ready in AUTO mode.

## Decision

```text
READY_FOR_P3A_GUI_PIPELINE_INTEGRATION
```
