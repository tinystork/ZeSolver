# Core Dependency Boundary

P2B-2C audits the package-to-application dependency direction after the
SolverPipeline, ProductionBlindSolverPort, and BatchSolverPipeline additions.

Target graph:

```text
GUI / CLI
  |
  v
application adapters
  |
  v
SolverPipeline / BatchSolverPipeline
  |
  v
Near / Blind ports
  |
  v
solver engines
```

No module under `zesolver/core/`, `zesolver/settings/`, or
`zesolver/catalog_library/` may load the root entrypoint `zesolver.py`, import
PySide6, or import legacy application classes such as `ImageSolver` and
`BatchSolver`.

## Audit Table

| Module | Depends On | Type | Acceptable | Action |
| ------ | ---------- | ---- | ---------: | ------ |
| `zesolver/core/blind_port.py` before P2B-2C | dynamic `zesolver.py` loader for `build_blind_solve_config()` | LEGACY | no | Extract shared Blind config builder and import it directly. |
| `zesolver/core/blind_port.py` after P2B-2C | `zesolver.solver_config`, `zesolver.zeblindsolver`, `zeblindsolver.index_manifest_4d` | CORE / CONFIGURATION / ENGINE | yes | Keep as production port boundary. |
| `zesolver/core/pipeline.py` | P2A settings assembly, P1C catalog resources, Near/Blind ports | CORE / CONFIGURATION / CATALOG | yes | No action. |
| `zesolver/core/batch/runner.py` | `SolverPipeline`, batch models, cancellation/progress protocols | CORE | yes | No action. |
| `zesolver/settings/*` | product/runtime/profile/migration dataclasses | CONFIGURATION | yes | No action. |
| `zesolver/catalog_library/*` | manifest models and validation helpers | CATALOG | yes | No action. |
| `zesolver.py` | package modules plus GUI/CLI/legacy `ImageSolver` and `BatchSolver` | GUI / CLI / LEGACY | yes | Remains root application entrypoint and compatibility surface. |
| `tests/test_*` entrypoint loaders | dynamic `zesolver.py` loading for legacy parity tests | TOOLING | yes | Allowed outside package boundary. |
| `tools/diagnose_*` entrypoint loaders | dynamic `zesolver.py` loading for diagnostics | TOOLING | yes | Allowed outside package boundary. |

## Current Guardrail

`tools/check_core_boundaries.py` parses Python AST under:

```text
zesolver/core/
zesolver/settings/
zesolver/catalog_library/
```

It fails on:

```text
PySide6 imports
spec_from_file_location()
module_from_spec()
root entrypoint path references
ImageSolver
BatchSolver
SolveRunner
QtCore / QtGui / QtWidgets
```

This keeps the package importable without the root `zesolver.py` runtime
resource.
