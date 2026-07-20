# P1D-5B - Manual GUI Validation

Date: 2026-07-19

This validation used a real Qt graphical session through the active
Wayland/X11 environment:

```text
WAYLAND_DISPLAY = wayland-0
DISPLAY = :1
QT_QPA_PLATFORM = unset
```

`QT_QPA_PLATFORM=offscreen` was not used.

Temporary working root:

```text
/tmp/p1d5b_real_gui_olbt2pxv
```

Product resources were read-only:

```text
/home/tristan/ZeSolverCatalog
/opt/astap
/home/tristan/zesolver_index
config/zeblind_4d_experimental_manifest.json
indexes/astrometry_4d
```

## Case A - Normal CatalogLibrary Path

Selected in the visible normal field:

```text
Bibliotheque ZeSolver = /home/tristan/ZeSolverCatalog
```

The GUI reported:

```text
READY_PARTIAL
catalog_source = library
catalog_library_id = adopted-existing
Near ASTAP = D50, 1476 tiles
Blind 4D = 6 / 1476 tiles
all_sky = false
```

In Easy mode, the historical fields were not visible:

```text
Historical ASTAP source visible = false
Historical Near index visible = false
External Blind 4D manifest visible = false
Compatibility group visible = false
Compatibility group checked by default = false
```

Five copied M106 FITS files were solved from the GUI queue:

```text
5 / 5 SOLVED
terminal_count = 1
near_catalog_provider = astap_native
near_catalog_mode_effective = astap-native
blind4d_catalog_source = catalog_library_view
blind4d_catalog_mode_effective = library-view
blind4d_external_fallback_used = false
legacy_blind4d_manifest_used = absent
```

Preflight timing example from the run:

```text
catalog_library_open_s = 0.2115
catalog_resource_resolution_s = 0.4589
near_runtime_resolution_s = 0.3181
blind4d_runtime_resolution_s = 0.2598
catalog_preflight_total_s = 1.2511
```

## Case B - Reported Confusion

Opened the advanced compatibility surface and entered:

```text
Index Near historique = /home/tristan/ZeSolverCatalog
```

The typed validator returned:

```text
CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX
```

French user message:

```text
Ce dossier est une Bibliotheque ZeSolver, pas un index Near historique.
```

No index construction was launched and no legacy run was started.

## Case C - Explicit Rollback

Configured in the advanced surface:

```text
Base ASTAP historique = /opt/astap
Index Near historique = /home/tristan/zesolver_index
Manifeste Blind 4D externe = config/zeblind_4d_experimental_manifest.json
Near = legacy-index
Blind 4D = external-manifest
```

The GUI showed:

```text
ROLLBACK HISTORIQUE ACTIF - Near : legacy-index; Blind 4D : external-manifest
```

The generated run configuration kept:

```text
near_catalog_mode = legacy-index
blind4d_catalog_mode = external-manifest
```

## Case D - Restore Automatic Mode

Clicked:

```text
Retablir le mode automatique recommande
```

Result:

```text
near_catalog_mode = auto
blind4d_catalog_mode = auto
```

The historical paths remained present:

```text
Base ASTAP historique = /opt/astap
Index Near historique = /home/tristan/zesolver_index
Manifeste Blind 4D externe = config/zeblind_4d_experimental_manifest.json
```

## Case E - Stop and Relaunch

Started a CatalogLibrary run, requested Stop, then relaunched one copied FITS.

Result:

```text
stop_called = true
terminal_count_after_stop = 1
relaunch = 1 / 1 SOLVED
terminal_count_after_relaunch = 1
```

## Qt Warnings

The real GUI opening did not produce:

```text
QLayout::addChildLayout
already has a parent
```

No global Qt warning suppression was used.

## Integrity

Only copied FITS files were solved.  Their pixel arrays were unchanged:

```text
pixel_hash_pairs_ok = 5 / 5
```

Current product-resource fingerprints after validation:

```text
/opt/astap: files=1487 size=977092518
/home/tristan/zesolver_index: files=1480 size=371695452
config/zeblind_4d_experimental_manifest.json:
  1847e075b25650ee00664bb6db23f80307f4d89caa548aca4c0d2c09e69e79a5
/home/tristan/ZeSolverCatalog/catalog.json:
  d666a62aad8bc619be79bd764b40bd447d336fe919c036a515c545b0016d3c0f
```

The six product Blind 4D NPZ hashes remained:

```text
d50_2602: 3ab3a747d2005ac6523a5dc62ef82fc19374fc31df27c8757307b36ade556693
d50_2644: 577bb69cbe23063a718f177d58a8e1b1367f4ccc19a0c2dab09bfbad26c7c9e7
d50_2645: b963ebe462c98556b98520584e515b77b5ab87bd08de70de52ec9d871c91b2df
d50_2702: 15d82411ab1213505660be448a9290b9129e2bc3243cff960dfa79b6475d0fd6
d50_2822: 04ecc3ea867307e64cbc8a8bf00cdca847b590b822289edce991e96aa9db1967
d50_2823: 63ede21d82d4bb885ad10b73ececff2750e49bd914c2625f321ed16a3a1529e5
```

