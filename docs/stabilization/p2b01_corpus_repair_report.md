# P2B-0.1 Corpus Baseline Repair Report

Date: 2026-07-17

Decision: `NOT_READY_FOR_CORE_EXTRACTION`

## Scope

This short corrective phase attempted to repair the P2B-0 external corpus gate
without changing solvers, thresholds, profiles, catalogues, indexes, FITS
originals, or corpus oracle SHA values.

The target was the only P2B-0 failing corpus case:

```text
near_m106_232102
expected SHA256:
3edf2580268376dc51409591fd9f91452c8f1e3c2817d3fe29b77ae2d7980821
```

## Located Copy

Exact SHA match found:

```text
<repo>/reports/zn34_work/zn34_090_027_Light_mosaic_M_106_20.0s_IRCUT_20250518-232102_027_Light_mosaic_M_106_20.0s_IRCUT_20250518-232102.fit
```

Verified hash:

```text
3edf2580268376dc51409591fd9f91452c8f1e3c2817d3fe29b77ae2d7980821
```

Source file stat before inspection:

```text
size=4155840
mtime=1776098297
```

Source file stat after inspection:

```text
size=4155840
mtime=1776098297
```

The inspection did not modify the file contents or modification time.

## FITS Inspection

Inspection with Astropy:

```text
fits_readable=True
image_present=True
image_shape=(1920, 1080)
has_celestial_wcs=True
has_SOLVED=True
has_SOLVER=True
```

This copy is byte-identical to the manifest SHA, but it is not a clean input
fixture. It already contains a celestial WCS and ZeSolver solve cards.

Because SHA256 identifies the full byte content, any other copy with the same
SHA would have the same residual WCS/SOLVED/SOLVER content. A broader read-only
search for this SHA found the same path before being stopped as redundant.

## Corpus Root Repair

No repaired corpus root was created.

The requested root:

```text
/home/tristan/zesolver-regression-corpus/
```

was not populated because the only located SHA-exact file failed the required
cleanliness contract:

- absence of celestial WCS: failed;
- absence of `SOLVED`: failed;
- absence of `SOLVER`: failed.

No contaminated FITS was copied. No oracle SHA was changed.

## Runner Status

The corpus runner was not rerun with a repaired root because the repair preflight
failed before any copy step.

Last P2B-0 corpus runner status remains:

```text
1 failed, 3 passed, 2 skipped, 321 deselected
runner status FAIL
```

The failure remains:

```text
tests/test_regression_near.py::test_zenear_corpus_files_resolve_or_skip_explicitly
near_m106_232102 SHA mismatch under the previously configured corpus root
```

Expected skips from P2B-0 remain:

- ASTAP/HNSKY `g05` absent under `/opt/astap`;
- P29 Blind 4D source FITS path not mapped in `tests/corpus/manifest.json`.

## ZN3.10B Status

ZN3.10B automated pipeline validation from P2B-0 remains valid:

```text
8 solved, 0 skipped, 0 failed
CONTROL_NEAR_CORRECT=3
NOHINT_4D_CORRECT=3
BADHINT_4D_CORRECT=2
historical_blind_called=0
astrometry_web_called=0
verdict=PASS
```

No ZN3.10B files were modified during P2B-0.1.

## Conclusion

```text
NOT_READY_FOR_CORE_EXTRACTION
```

The SHA expected by the current oracle exists, but it points to a FITS file that
already contains a WCS and ZeSolver solve metadata. Under the P2B-0.1 rules, that
file cannot be used to repair the external corpus. P2B-1 must remain blocked
until a genuinely clean input fixture is restored or the corpus oracle is
updated through an explicit baseline decision.
