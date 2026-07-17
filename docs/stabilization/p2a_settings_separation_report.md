# P2A Product Settings & Internal Profiles Separation Report

Decision: `READY_FOR_PROGRESSIVE_CORE_REFACTORING`

P2A adds a separation layer for product settings, runtime options, immutable
solver profiles and explicit developer overrides. The GUI is not redesigned and
legacy settings are not removed.

## Implemented

Created:

- `docs/architecture/settings_inventory.md`
- `docs/architecture/settings_public_surface.md`
- `docs/architecture/solver_profiles.md`
- `docs/architecture/settings_migration_v2.md`
- `zesolver/settings/__init__.py`
- `zesolver/settings/product.py`
- `zesolver/settings/runtime.py`
- `zesolver/settings/profiles.py`
- `zesolver/settings/migration.py`
- `zesolver/settings/assembly.py`
- `tests/test_product_settings.py`
- `tests/test_solver_profiles.py`
- `tests/test_settings_migration_v2.py`
- `tests/test_configuration_assembly.py`
- `tests/test_developer_overrides.py`

## Boundaries

- `ProductSettings` contains user choices and profile ids only.
- `RuntimeOptions` contains execution-local state and is not persisted.
- `zenear-v1`, `zeblind4d-v1`, and `pipeline-v1` freeze the current baseline
  values.
- `DeveloperOverrides` has no effect unless `enabled=True`.
- `SettingsMigrationResult` reports migrated, ignored, deprecated fields and
  warnings without writing over old settings.

## Validation

Targeted P2A tests:

```text
.venv/bin/python -m pytest \
  tests/test_product_settings.py \
  tests/test_solver_profiles.py \
  tests/test_settings_migration_v2.py \
  tests/test_configuration_assembly.py \
  tests/test_developer_overrides.py \
  -q

15 passed
```

Post-implementation hermetic baseline:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
320 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK
```

Final P2A validation:

```text
.venv/bin/python -m pytest \
  tests/test_product_settings.py \
  tests/test_solver_profiles.py \
  tests/test_settings_migration_v2.py \
  tests/test_configuration_assembly.py \
  tests/test_developer_overrides.py \
  -q

15 passed

.venv/bin/python tools/run_regression_suite.py --hermetic
320 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

.venv/bin/python -m pytest -q
320 passed, 7 skipped, 1 warning

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK

git diff --check
OK
```

External corpus data were not configured in this shell:

```text
ZESOLVER_CORPUS_ROOT=<unset>
ZESOLVER_ZN310B_ROOT=<unset>
ZESOLVER_BLIND4D_MANIFEST=<unset>
```

Clean snapshot:

```text
temporary snapshot worktree: /tmp/zesolver-p2a-snapshot-Ic1PoO/worktree
temporary snapshot commit: 9cb97430f88b2ec0be366b99da686def884e5a74
git status --short: clean

tools/run_regression_suite.py --hermetic:
310 passed, 11 skipped, 6 deselected, 1 warning
runner status PASS

compileall:
OK
```

The snapshot has additional expected skips because bundled NPZ indexes, ASTAP
source checkout, ZN dump artifacts and selected FITS/report fixtures are not
tracked in the clean snapshot.

## Limits

- The GUI still feeds the old compatibility layer.
- The new settings v2 format is defined and testable but not yet written as the
  default on-disk format.
- Direct legacy catalogue/index paths remain readable for rollback.
- Developer tools are classified but not moved out of the GUI yet.

## Decision

```text
READY_FOR_PROGRESSIVE_CORE_REFACTORING
```

The project can start P2B because catalogue resources are separated from solver
settings, product settings no longer need to expose internal thresholds, v1
profiles are immutable, and developer overrides are explicit.
