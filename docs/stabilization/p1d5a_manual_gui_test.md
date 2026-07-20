# P1D-5A — Manual GUI Validation

Date: 2026-07-19

This validation used a real Qt graphical session through the current Wayland/X11
environment.  `QT_QPA_PLATFORM=offscreen` was not used.

Temporary working root:

```text
/tmp/p1d5a_gui_library_DerPSR
```

Product resources were read-only:

```text
/opt/astap
/home/tristan/zesolver_index
config/zeblind_4d_experimental_manifest.json
indexes/astrometry_4d
```

## Library

A temporary library was adopted from the current installation:

```text
library/catalog.json
status = READY_PARTIAL
source ASTAP = D50, 1476 shards
Blind 4D = 6 indexes / 1476 tiles
all_sky = false
```

The GUI settings file was also temporary:

```text
/tmp/p1d5a_gui_library_DerPSR/gui_settings.json
```

It persisted:

```text
catalog_library_path = /tmp/p1d5a_gui_library_DerPSR/library
db_root = null
index_root = null
blind_4d_manifest_path = /tmp/p1d5a_gui_invalid_blind4d_manifest_DO_NOT_USE.json
near_catalog_mode = auto
blind4d_catalog_mode = auto
```

The invalid manifest sentinel proves that normal library mode did not consult
the historical external manifest field.

## Case A — CatalogLibrary

Steps:

1. Open GUI.
2. Select the temporary library root.
3. Verify the library.
4. Save settings.
5. Close and reopen the GUI.
6. Confirm the path is restored.
7. Solve five copied M106 FITS files in `auto`.

Result:

```text
5/5 SOLVED
progress = 500 / 500
terminal_count = 1
messages = []
```

Run telemetry:

```text
catalog_source = library
catalog_library_id = adopted-existing
catalog_library_status = READY_PARTIAL
near_catalog_mode_effective = astap-native
near_catalog_provider = astap_native
near_catalog_source = library
blind4d_catalog_mode_effective = library-view
blind4d_catalog_source = catalog_library_view
blind4d_external_fallback_used = false
blind4d_covered_tiles = 6
blind4d_total_tiles = 1476
blind4d_all_sky = false
```

Preflight timings from the run:

```text
catalog_library_open_s = 0.2522
catalog_resource_resolution_s = 0.4530
near_runtime_resolution_s = 0.3288
blind4d_runtime_resolution_s = 0.2595
catalog_preflight_total_s = 1.2961
```

## Case B — Explicit Rollback

Configuration:

```text
near_catalog_mode = legacy-index
blind4d_catalog_mode = external-manifest
db_root = /opt/astap
index_root = /home/tristan/zesolver_index
blind_4d_manifest_path = config/zeblind_4d_experimental_manifest.json
```

Result:

```text
1/1 SOLVED
terminal_count = 1
near_catalog_mode_effective = legacy-index
near_catalog_provider = legacy_index
blind4d_catalog_mode_effective = external-manifest
blind4d_catalog_source = external_manifest
```

The rollback was explicit and visible in the log.

## Case C — Invalid Library

Selected path:

```text
/tmp/p1d5a_gui_invalid_library_missing_catalog
```

Result:

```text
status = MISSING - catalog.json not found
message = CATALOG_LIBRARY_MANIFEST_MISSING
worker_started = false
```

No legacy fallback occurred.

## Case D — Stop and Relaunch

Steps:

1. Start a run on five copied M106 FITS files.
2. Request Stop.
3. Wait for cancellation.
4. Relaunch on the same copied set.

Result:

```text
stop_called = true
STOP_RUNNER_RECEIVED
STOP_CONTROLLER_RECEIVED
first run terminal_count = 1
second run = 5/5 SOLVED
second run terminal_count = 1
```

## Integrity

Pixel hashes for copied FITS files were compared against the originals:

```text
pixel_hash_pairs_ok = 10
```

Current read-only resource snapshot after validation:

```text
/opt/astap: files=1487 size=977092518
/home/tristan/zesolver_index: files=1480 size=371695452
indexes/astrometry_4d: files=6 size=6345747
```

Strict manifest SHA256:

```text
1847e075b25650ee00664bb6db23f80307f4d89caa548aca4c0d2c09e69e79a5
```

The six product NPZ SHA256 values remained:

```text
d50_2602: 3ab3a747d2005ac6523a5dc62ef82fc19374fc31df27c8757307b36ade556693
d50_2644: 577bb69cbe23063a718f177d58a8e1b1367f4ccc19a0c2dab09bfbad26c7c9e7
d50_2645: b963ebe462c98556b98520584e515b77b5ab87bd08de70de52ec9d871c91b2df
d50_2702: 15d82411ab1213505660be448a9290b9129e2bc3243cff960dfa79b6475d0fd6
d50_2822: 04ecc3ea867307e64cbc8a8bf00cdca847b590b822289edce991e96aa9db1967
d50_2823: 63ede21d82d4bb885ad10b73ececff2750e49bd914c2625f321ed16a3a1529e5
```

Only temporary copied FITS received WCS updates.

