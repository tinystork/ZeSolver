# P1D-1A ZeNear ASTAP-native Provider Report

Decision: `READY_FOR_P1D1B_ASTAP_PRODUCT_SWITCH`

P1D-1A introduces a ZeNear catalogue-provider boundary and proves that the
Near algorithm can run either from the historical index provider or from an
ASTAP-native provider. The product default is not switched in this mission.

## 1. Objectif

P1D-1A creates the technical seam required by P1D-1B:

- keep the current `index_root/manifest.json` path available as explicit
  rollback/oracle;
- add an ASTAP-native provider that selects tiles and loads stars directly from
  ASTAP/HNSKY shards;
- keep projection, star selection, matching, RANSAC, WCS fit, validation and
  acceptance code shared and unchanged.

## 2. Etat Git initial

Initial state before P1D-1A work:

```text
## test...origin/test
 M AGENT.md
 M zesolver.py
 M zesolver/gui_pipeline/legacy_runner.py
 M zesolver/gui_pipeline/pipeline_runner.py
?? docs/architecture/gui_live_state_flow.md
?? docs/stabilization/original_stabilization_roadmap_20260716.md
?? docs/stabilization/p1d0_astap_single_library_gap_audit.md
?? docs/stabilization/p3av3_gui_live_state_report.md
?? docs/stabilization/p3av3_manual_gui_state_test.md
?? tests/p3av3_helpers.py
?? tests/test_gui_progress_*.py
?? tests/test_gui_wcs_*.py
```

The working tree already contained P3A-V3 and P1D-0 changes. P1D-1A preserved
them and did not revert unrelated files.

## 3. Architecture retenue

New engine-local module:

```text
zeblindsolver/near_catalog_provider.py
```

It defines:

- `NearCatalogTile`
- `NearTileBounds`
- `NearCatalogStars`
- `NearCatalogProvider`
- `NearCatalogProviderError`
- `LegacyIndexNearCatalogProvider`
- `AstapNearCatalogProvider`

The package `zeblindsolver` still does not import `zesolver`,
`zesolver.catalog_library`, `zesolver.catalog_resources`, PySide6, GUI modules,
or Blind 4D runtime resources from the provider.

## 4. Contrat du provider

Provider contract:

```text
kind
families
select_tiles(ra_deg, dec_deg, radius_deg, limit, families=None)
load_stars(tile)
telemetry()
```

`load_stars()` returns `ra_deg`, `dec_deg`, `mag` arrays:

- RA and DEC are degrees;
- magnitude is the source catalogue magnitude;
- row order is provider-defined and deterministic;
- finite filtering remains in the shared Near solver;
- provider never sees FITS data, RANSAC, WCS, GUI, or Blind 4D.

## 5. Fichiers modifies

Code:

- `.gitignore`
- `zeblindsolver/near_catalog_provider.py`
- `zeblindsolver/metadata_solver.py`
- `zesolver/zeblindsolver.py`
- `zesolver/catalog_resources.py`

Tests/tools:

- `tests/near_catalog_provider_helpers.py`
- `tests/test_near_catalog_provider_astap.py`
- `tests/test_near_catalog_provider_legacy.py`
- `tests/test_near_catalog_provider_parity.py`
- `tests/test_near_catalog_provider_boundaries.py`
- `tools/compare_zenear_catalog_providers.py`

Documentation:

- `docs/stabilization/p1d1a_zenear_astap_provider_report.md`

## 6. Comportement legacy preserve

When `catalog_provider` is absent:

```text
solve_near(input_fits, index_root, config=...)
```

still loads `index_root/manifest.json` and uses the historical candidate model.
This is the product-compatible default and remains the path used by current GUI
and pipeline calls.

`LegacyIndexNearCatalogProvider` wraps the historical manifest/tile path for
explicit comparison. It reads `tiles/*.npz` first and can use the historical
ASTAP fallback from `manifest["db_root"]` if the tile NPZ is unavailable. That
fallback is read-only in the provider and does not persist a replacement NPZ.

## 7. Comportement ASTAP-native

`AstapNearCatalogProvider(db_root, families=...)`:

- uses `astap_db_reader.iter_tiles()`;
- selects tiles directly from ASTAP tile geometry;
- loads raw stars with `astap_db_reader.load_tile_stars()`;
- requires explicit `db_root` and optional explicit families;
- has no `index_root`;
- reads no historical `manifest.json`;
- reads no `tiles/*.npz`;
- writes no cache or derived artefact;
- raises explicit provider errors for absent/mismatched families or unreadable
  tiles.

## 8. Preuve d'absence d'ecriture

Automated tests verify:

- ASTAP provider does not create `*.npz`;
- file mtimes under the synthetic ASTAP root are unchanged after loads;
- cache returns defensive copies, so callers cannot mutate cached arrays;
- legacy ASTAP fallback does not recreate a missing historical NPZ.

External solve comparison used copies under `/tmp`; original FITS and catalogue
directories were not modified.

## 9. Parite des tuiles

Synthetic provider tests cover:

- normal regions;
- RA 0/360 crossing;
- polar region;
- declination bounds;
- deterministic distance ordering;
- `max_tile_candidates`;
- missing family rejection.

External characterization with:

```bash
.venv/bin/python tools/compare_zenear_catalog_providers.py \
  --index-root /home/tristan/zesolver_index \
  --astap-root /opt/astap \
  --family d50 \
  --out-json /tmp/p1d1a_provider_compare.json \
  --out-md /tmp/p1d1a_provider_compare.md
```

Results:

| Case | Legacy | ASTAP-native | Keys | Note |
|---|---:|---:|---|---|
| normal | 4 | 4 | match | same tile order |
| RA 0 | 6 | 6 | match | same tile order |
| high DEC | 6 | 6 | same set, order differs | center coordinates differ slightly between historical NPZ-derived manifest and ASTAP layout geometry |

The high-declination order difference is explained by the old manifest storing
slightly different centers from the ASTAP layout-derived centers. The selected
tile set is the same. No acceptance threshold or solver algorithm was changed.

## 10. Parite des etoiles

Synthetic parity test:

- builds a minimal ASTAP `.1476` tile;
- builds a matching legacy NPZ tile;
- compares RA/DEC/mag arrays with explicit tolerances;
- proves ASTAP-native serves the same stars as the strict ASTAP source path.

External characterization shows historical NPZ tiles under
`/home/tristan/zesolver_index/tiles` are often capped at `2000` stars while
ASTAP raw tiles under `/opt/astap` contain full tile populations, often tens of
thousands of stars. Therefore old NPZ byte parity is not the scientific oracle
for strict ASTAP-ISO. The provider target is the raw ASTAP star stream already
used by the strict path.

## 11. Parite des resultats

Synthetic solver parity:

- legacy provider path and ASTAP-native provider path both solve the same
  synthetic frame on separate FITS copies;
- same success status;
- same tile key;
- same inlier count;
- shared algorithm body after provider catalogue assembly.

External short solve parity:

```text
FITS: /home/tristan/near_bench100_input/069_Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit
legacy: success=True, tile=d50_2823, inliers=39, RMS=0.1855110930 px
ASTAP-native: success=True, tile=d50_2823, inliers=50, RMS=0.1875417555 px
```

Both runs used temporary copies created by
`tools/compare_zenear_catalog_providers.py`.

## 12. Tests cibles

Executed:

```text
.venv/bin/python -m pytest \
 tests/test_metadata_solver.py \
 tests/test_catalog290.py \
 tests/test_catalog_library_adapters.py \
 tests/test_catalog_resource_resolution.py \
 tests/test_catalog_library_near_integration.py \
 tests/test_catalog_library_pipeline_integration.py \
 tests/test_near_catalog_provider_astap.py \
 tests/test_near_catalog_provider_legacy.py \
 tests/test_near_catalog_provider_parity.py \
 tests/test_near_catalog_provider_boundaries.py \
 -q
```

Result:

```text
31 passed, 2 skipped
```

Skips:

```text
tests/test_catalog290.py: external ASTAP/HNSKY test database not found: ./database
```

The real external ASTAP root `/opt/astap` was used by the explicit comparison
tool, not by those hermetic tests.

## 13. Barrieres generales

Executed:

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
436 passed, 1 skipped, 9 deselected, 44 warnings
runner status PASS

.venv/bin/python -m pytest -q
436 passed, 10 skipped, 44 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py
OK

git diff --check
OK
```

## 14. Corpus externe

The canonical corpus environment variables were not set:

```text
ZESOLVER_CORPUS_ROOT=
ZESOLVER_ASTAP_ROOT=
ZESOLVER_LEGACY_INDEX_ROOT=
```

Available local data used for characterization:

```text
/opt/astap
/home/tristan/zesolver_index
/home/tristan/near_bench100_input
```

Only one real FITS solve comparison was run to keep P1D-1A scoped. A broader
P1D-1B gate should run the same tool across the canonical Near corpus.

## 15. Warnings

No new warning category has been observed in targeted tests. Known project
warning categories are still expected in broad suites:

- `multiprocessing.popen_fork.DeprecationWarning`
- `Astropy VerifyWarning`

## 16. Limites

- Product default is not switched; GUI and pipeline still use the historical
  `index_root` path unless a caller explicitly passes a provider.
- High-declination tile ordering can differ where historical manifest centers
  differ from ASTAP layout-derived centers.
- Historical NPZ star arrays are often capped and should not be used as byte
  oracle for ASTAP-native strict stars.
- External solve parity was demonstrated on one local M106 frame, not the full
  canonical corpus.

## 17. Etat Git final

Final relevant state:

```text
## test...origin/test
 M .gitignore
 M AGENT.md
 M zeblindsolver/metadata_solver.py
 M zesolver.py
 M zesolver/catalog_resources.py
 M zesolver/gui_pipeline/legacy_runner.py
 M zesolver/gui_pipeline/pipeline_runner.py
 M zesolver/zeblindsolver.py
?? docs/stabilization/p1d1a_zenear_astap_provider_report.md
?? tests/near_catalog_provider_helpers.py
?? tests/test_near_catalog_provider_astap.py
?? tests/test_near_catalog_provider_boundaries.py
?? tests/test_near_catalog_provider_legacy.py
?? tests/test_near_catalog_provider_parity.py
?? tools/compare_zenear_catalog_providers.py
?? zeblindsolver/near_catalog_provider.py
```

Existing P3A-V3 and P1D-0 files remain in the working tree and are not listed
again here in full.

## 18. Prochaine etape unique

P1D-1B: switch the product Near runtime to build and pass
`AstapNearCatalogProvider` from resolved `CatalogLibrary` resources, behind an
explicit rollback setting selecting `LegacyIndexNearCatalogProvider`.

## 19. Decision de gate

`READY_FOR_P1D1B_ASTAP_PRODUCT_SWITCH`
