# ZeSolver Regression Corpus

The regression corpus manifest is versioned, but the FITS images, ASTAP databases,
and 4D indexes are external data. Tests must never search personal directories
silently.

## Environment Variables

- `ZESOLVER_CORPUS_ROOT`: root for general external FITS corpus entries.
- `ZESOLVER_ZN310B_ROOT`: root of a generated ZN3.10B dataset directory containing
  `control_clean/`, `no_hints/`, `wrong_hints/`, and `gui_mixed/`.
- `ZESOLVER_ASTAP_ROOT`: optional ASTAP/HNSKY database root for external catalog tests.
- `ZESOLVER_BLIND4D_MANIFEST`: path to the ZeBlind 4D manifest to use for corpus
  blind tests.
- `ZESOLVER_LEGACY_INDEX_ROOT`: optional legacy index root, only for legacy
  diagnostic tests.

Resolution order for corpus data is:

1. explicit runner argument;
2. environment variable named by the manifest case;
3. local non-versioned configuration, when a future runner supports it;
4. explicit pytest skip with the missing variable or file named.

No test should infer `/home/tristan/near_bench_cmp30/`, `/home/tristan/zesolver_index/`,
or any other personal path.

## Commands

Hermetic baseline:

```bash
.venv/bin/python -m pytest -m "not external_catalog and not corpus and not slow" -q
```

Corpus baseline, when data are configured:

```bash
.venv/bin/python -m pytest -m "external_catalog or corpus or slow" -q
```
