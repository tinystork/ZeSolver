# P2B-0 Git Freeze Audit

Date: 2026-07-17

This audit was performed before any P2B code extraction. It classifies the
current P0-P2A working tree so the local milestone commit can freeze source,
tests, tools and documentation without forcing external catalogue artifacts into
Git.

## Commands Executed

```text
git status --short --branch
git diff --stat
git diff --check
git ls-files --others --exclude-standard
git check-ignore -v \
 indexes/astrometry_4d/*.npz \
 reports/ \
 tests/corpus/ \
 tools/run_regression_suite.py
```

## Branch and Hygiene

```text
## test...origin/test
```

`git diff --check` produced no output.

Tracked diff stat:

```text
 .gitignore                    | 57 ++++++++++++++++++++++++++++--
 pyproject.toml                |  6 ++++
 tests/test_catalog290.py      |  9 +++++
 tests/test_metadata_solver.py | 69 +++++++++++++++++++++++++++++-------
 zeblindsolver/downloads.py    |  6 +++-
 zesolver.py                   | 81 ++++++++++++++++++++++++++++++++++++++-----
 zesolver/settings_store.py    |  2 ++
 7 files changed, 205 insertions(+), 25 deletions(-)
```

## Tracked Modified Files

These files are part of the P0-P2A stabilization milestone and should be
included in the local freeze commit:

- `.gitignore`: makes test, corpus manifest, selected reports and diagnostic
  tools followable while keeping heavy local artifacts ignored.
- `pyproject.toml`: declares pytest markers for external catalogue, corpus,
  slow and Blind 4D tests.
- `tests/test_catalog290.py`: marks external ASTAP/HNSKY tests and skips
  explicitly when the local database is absent.
- `tests/test_metadata_solver.py`: stabilizes synthetic strict ASTAP-ISO tests
  with deterministic detector stubs.
- `zeblindsolver/downloads.py`: fixes fake downloader completion behavior used
  by tests.
- `zesolver.py`: P1C CatalogLibrary integration boundary and CLI/GUI resource
  application.
- `zesolver/settings_store.py`: non-destructive persistent field for
  `catalog_library_path`.

## Newly Created Followable Files

The following untracked groups are source, test, tool, schema or documentation
files required by P0-P2A and should be included in the freeze commit:

- `docs/architecture/`: P1/P2 architecture inventories, catalog library design,
  integration map, settings inventory and profile documentation.
- `docs/stabilization/`: P0-P2A stabilization reports and reproducibility
  notes.
- `tests/corpus/`: versioned corpus manifest and compact oracle metadata only;
  no FITS data are stored there.
- `tests/catalog_library_fixtures.py`
- `tests/catalog_resource_helpers.py`
- `tests/corpus_loader.py`
- `tests/test_catalog_library_*.py`
- `tests/test_catalog_resource_resolution.py`
- `tests/test_configuration_assembly.py`
- `tests/test_developer_overrides.py`
- `tests/test_product_settings.py`
- `tests/test_regression_*.py`
- `tests/test_settings_migration_v2.py`
- `tests/test_solver_profiles.py`
- `tests/test_zn*.py`
- `tools/audit_catalog_library.py`
- `tools/run_regression_suite.py`
- `tools/build_zn310b_gui_dataset.py`
- `tools/diagnose_*.py`
- `tools/astap_zn2_build_and_compare.py`
- `zesolver/catalog_library/`
- `zesolver/catalog_resources.py`
- `zesolver/settings/`

## Selected Report Oracles

Several compact JSON/Markdown reports under `reports/` are followable through
explicit `.gitignore` exceptions and are consumed by the P0-P2A tests or
documentation. They belong to the milestone when they are small oracle or audit
inputs, not raw benchmark output.

Examples currently followable:

- `reports/zenear_zn39_corpus_manifest.json`
- `reports/zenear_zn39_final_matrix.json`
- `reports/zenear_zn39_near_matrix.json`
- `reports/zenear_zn310b_gui_manifest.json`
- `reports/zenear_zn310b_pixel_integrity.json`

## Voluntarily Ignored Artifacts

The ignore audit confirms:

```text
.gitignore:80:indexes/astrometry_4d/*.npz indexes/astrometry_4d/d50_2602_S_q40000.npz
.gitignore:80:indexes/astrometry_4d/*.npz indexes/astrometry_4d/d50_2644_S_q40000.npz
.gitignore:80:indexes/astrometry_4d/*.npz indexes/astrometry_4d/d50_2645_S_q40000.npz
.gitignore:80:indexes/astrometry_4d/*.npz indexes/astrometry_4d/d50_2702_S_q40000.npz
.gitignore:80:indexes/astrometry_4d/*.npz indexes/astrometry_4d/d50_2822_S_q40000.npz
.gitignore:80:indexes/astrometry_4d/*.npz indexes/astrometry_4d/d50_2823_S_q40000.npz
.gitignore:22:reports/** reports/
.gitignore:76:!tests/corpus/** tests/corpus/
.gitignore:69:!tools/run_regression_suite.py tools/run_regression_suite.py
```

The six 4D NPZ files are external runtime artifacts. They must not be forced
into Git just to make a clean checkout behave like the local working tree.

`reports/**` remains ignored by default. Only explicitly unignored compact
oracle/report files should be tracked.

## External Data

External data expected by the current corpus and catalogue tests:

- FITS corpus under `ZESOLVER_CORPUS_ROOT`.
- ZN3.10B generated dataset under `ZESOLVER_ZN310B_ROOT`.
- ASTAP/HNSKY catalogue root under `ZESOLVER_ASTAP_ROOT` or legacy `db_root`.
- ZeBlind 4D strict manifest under `ZESOLVER_BLIND4D_MANIFEST`.
- Legacy index root under `ZESOLVER_LEGACY_INDEX_ROOT` when diagnostic tests
  need it.

These paths should be configured locally or through shell variables, not stored
as personal absolute paths in versioned files.

## Temporary or Local-Only Items

No temporary P2B files were created before this audit. Existing local cache,
database, ASTAP, NPZ and broad report directories remain governed by
`.gitignore`.

## Milestone Inclusion Decision

Include in the P0-P2A milestone:

- all tracked modifications listed above;
- architecture and stabilization documentation needed to explain P0-P2A;
- test fixtures, regression tests and corpus manifest/oracles;
- catalog library/resource/settings implementation modules;
- diagnostic and regression tools explicitly made followable;
- selected compact report oracles that existing tests reference.

Do not include:

- FITS source images;
- ASTAP/HNSKY databases;
- ZeBlind 4D NPZ indexes;
- broad benchmark logs or ad-hoc report output;
- virtual environments or caches.

## Audit Decision

The working tree content is suitable for a local P0-P2A freeze commit after the
baseline is rerun. The remaining dirty state is expected milestone work, not an
unrelated change set.
