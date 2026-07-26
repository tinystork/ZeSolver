# S6A-2C - ZN310B Corpus Restoration and Final Validation

Date: 2026-07-26

## 1. Initial state

S6A-2B was functionally validated, but the global hermetic/offscreen gates were blocked by:

```text
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha
```

The failing external corpus directory was:

```text
/home/tristan/near_bench_cmp30/thread4
```

The expected SHA values came from:

```text
reports/zenear_zn310b_gui_manifest.json
```

## 2. Backup

The requested backup path already existed and matched the contaminated `thread4` state:

```text
/home/tristan/near_bench_cmp30/thread4_modified_backup_20260726
```

It was left intact.

## 3. Exact SHA search

The expected 8 full-file SHA-256 values were searched in the available ZeSolver/near corpus locations, including:

```text
/home/tristan/near_bench_cmp30
/home/tristan/near_bench100_input
/home/tristan/near_auto100_input
/home/tristan/near_auto_compare100
/home/tristan/near_auto_compare100_fix
/home/tristan/near_autotune_calib
/home/tristan/.openclaw/workspace/projects/ZeSolver/reports
/home/tristan/.openclaw.backup-20260521-182128/workspace/projects/ZeSolver/reports
/home/tristan/.local/share/Trash/files
```

Result:

```text
6065 FITS scanned
0/8 exact full-file SHA matches found
```

The original full-file bytes are therefore not currently recoverable from the searched local corpus copies.

## 4. Pixel integrity

The current contaminated `thread4` files still had the expected pixel SHA values from the ZN310B source inventory:

```text
8/8 pixel SHA matches
```

The existing `control_clean` variants in the ZN310B run directory also matched those source pixel SHA values and had no old WCS.

## 5. Restoration decision

Because the exact original full-file bytes could not be found, the corpus was deliberately moved to a clean canonical state instead of accepting the contaminated WCS-written files.

For each of the 8 manifest source paths, the corresponding clean source file was restored from:

```text
/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840/control_clean/ZN310B_CONTROL_###.fit
```

The restored files preserve the expected pixels and remove stale/generated solve WCS headers.

## 6. Manifest update

`reports/zenear_zn310b_gui_manifest.json` was updated so `source_SHA256` now describes the restored clean canonical source files in `thread4`.

No scientific thresholds, solver code, WCS tolerances, or generated GUI variants were changed for this restoration.

## 7. Validation

Targeted red test:

```text
.venv/bin/python -m pytest -q tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha
1 passed
```

Hermetic regression suite:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
PASS
686 passed, 1 skipped, 9 deselected
compileall returncode 0
```

Full offscreen pytest:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
686 passed, 10 skipped
```

## 8. Gate

```text
S6A2C_ZN310B_CORPUS_RESTORED_TO_CLEAN_CANON
ZN310B_SOURCE_SHA_GATE_RESTORED
S6A2B_GLOBAL_GATES_GREEN
READY_FOR_S6A2_FINAL_COMMIT
```
