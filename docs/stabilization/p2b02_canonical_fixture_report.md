# P2B-0.2 Canonical Near Fixture Derivation Report

Date: 2026-07-17

Decision: `READY_FOR_CORE_EXTRACTION`

## Scope

This corrective phase repaired the external Near corpus baseline without
changing solvers, thresholds, profiles, catalogues, indexes, or original FITS
files.

The primary target was:

```text
near_m106_232102
```

The historical manifest SHA pointed to a solved FITS parent. A clean canonical
fixture was therefore derived by removing solver WCS cards from a copy while
preserving pixels and legitimate acquisition metadata.

During corpus root reconstruction, the second Near corpus fixture
`near_m106_233459` was found to have the same latent issue under the configured
external root. No copy matching its previous manifest SHA was found under
`/home/tristan`, so it was also canonically derived and recorded explicitly in
the manifest. This avoids replacing the original `232102` failure with a hidden
`233459` skip or a contaminated copy.

## Local Corpus Root

Dedicated local root:

```text
/home/tristan/zesolver-regression-corpus/
```

Populated relative paths:

```text
near_bench100_input/027_Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit
near_bench100_input/001_Light_mosaic_M 106_20.0s_IRCUT_20250518-233459.fit
```

No FITS source under the repository or the historical working folders was
modified. Parent file size and mtime for `232102` remained:

```text
size=4155840
mtime=1776098297
```

## Transformation

Transformation:

```text
strip_solver_wcs_cards_v1
```

Determinism check:

```text
232102: two independent outputs had identical SHA256
233459: two independent outputs had identical SHA256
```

The transform removes WCS solution cards across all HDUs, including:

```text
MJDREF
WCSAXES
CRPIX1
CRPIX2
CUNIT1
CUNIT2
CTYPE1
CTYPE2
CRVAL1
CRVAL2
LONPOLE
LATPOLE
RADESYS
CD1_1
CD1_2
CD2_1
CD2_2
```

It also removes ZeSolver result cards:

```text
SOLVED
SOLVMODE
SOLVER
```

`CHECKSUM` and `DATASUM` are removed if present. No timestamp, date-dependent
card, or `HISTORY` card is added.

## Fixture `near_m106_232102`

Parent path:

```text
<repo>/reports/zn34_work/zn34_090_027_Light_mosaic_M_106_20.0s_IRCUT_20250518-232102_027_Light_mosaic_M_106_20.0s_IRCUT_20250518-232102.fit
```

Parent SHA256:

```text
3edf2580268376dc51409591fd9f91452c8f1e3c2817d3fe29b77ae2d7980821
```

Canonical SHA256:

```text
d40f655f40475928d921755a6dc1c46290f03b487e417de81c53eecef229355f
```

Pixel hash before and after:

```text
51641f1803ca639940c1684d4a0eda140e7939446b5a46dff4fe1672f3a46abd
```

Image data:

```text
shape=(1920, 1080)
dtype=uint16
```

Pre-transform state:

```text
has_celestial_wcs=true
solver_cards=SOLVED,SOLVMODE,SOLVER
```

Post-transform state:

```text
has_celestial_wcs=false
solver_cards=<none>
wcs_cards=<none>
checksum_cards=<none>
pixels_unchanged=true
```

Retained input metadata included:

```text
RA=184.91667
DEC=47.162222
OBJECT=M 106
FOCALLEN=250
XPIXSZ=2.90000009536743
YPIXSZ=2.90000009536743
DATE-OBS=2025-05-19T03:20:36.811822
EXPTIME=20.0
EXPOSURE=20.0
FILTER=IRCUT
INSTRUME=Seestar S50
TELESCOP=S50_ebbd8627
BAYERPAT=GRBG
GAIN=80
FOCUSPOS=1694
```

## Fixture `near_m106_233459`

Parent path:

```text
<home>/near_bench100_input/001_Light_mosaic_M 106_20.0s_IRCUT_20250518-233459.fit
```

Previous manifest SHA256:

```text
6eaae759a5b33542c5a8df763c6e26d4f7d7000f8c3970ca7280b9c57ade46e6
```

No file with that SHA was found under `/home/tristan`.

Parent SHA256:

```text
b0953a0a88999487f4071bf3785551ad28464e1f266297c23b252409a0ed68e0
```

Canonical SHA256:

```text
284ea1bc73ed341cdca92edc54106367bb779df580139a32579d7a4811ec606a
```

Pixel hash before and after:

```text
88b096ff1f978ae6d96e6f754f05052182ae01523b0f4a49b4ed58dfce833ba8
```

Post-transform state:

```text
has_celestial_wcs=false
solver_cards=<none>
wcs_cards=<none>
checksum_cards=<none>
pixels_unchanged=true
```

## Manifest Update

`tests/corpus/manifest.json` now records canonical SHA values and explicit
provenance:

```json
{
  "transformation": "strip_solver_wcs_cards_v1",
  "pixels_unchanged": true
}
```

For `232102`, `parent_sha256` is the historical manifest SHA. For `233459`, the
previous manifest SHA is preserved as `previous_manifest_sha256` and the solved
parent actually available locally is recorded as `parent_sha256`.

## Hermetic Transform Test

Added:

```text
tests/test_corpus_fixture_derivation.py
```

Validation:

```text
.venv/bin/python -m pytest tests/test_corpus_fixture_derivation.py -q
1 passed
```

The test verifies:

- forbidden WCS cards are removed from multiple HDUs;
- `SOLVED`, `SOLVER`, `SOLVMODE`, `CHECKSUM`, and `DATASUM` are removed;
- pixels are unchanged;
- Near acquisition metadata is preserved;
- two derivations from the same parent are byte-identical.

## Real ZeNear Check

Command path:

```text
copy canonical 232102 to a temporary file
run near_solve(..., index_root=<home>/zesolver_index, family=d50)
```

Result:

```text
success=true
wrote_wcs=true
message=near solution found
inliers=39
rms_px=0.1855110930303141
pixel_scale_arcsec=2.3717138060454057
source_pixels_unchanged=true
work_pixels_unchanged=true
```

The image-center world coordinate matches the oracle:

```text
center_pixel_world=(184.62777087563808, 47.298584781528334)
center_pixel_separation=8.35443928886129e-09 arcsec
```

Note: `CRVAL` itself is not the image center for this accepted WCS because the
solution has `CRPIX1=-53.813439055377`. The center check therefore uses the
world coordinate at the image center, matching the solver's recorded
`strict_acceptance.center_ra_deg` and `center_dec_deg`.

## Corpus Runner

Environment:

```text
ZESOLVER_CORPUS_ROOT=/home/tristan/zesolver-regression-corpus
ZESOLVER_ZN310B_ROOT=/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840
ZESOLVER_ASTAP_ROOT=/opt/astap
ZESOLVER_BLIND4D_MANIFEST=<repo>/config/zeblind_4d_experimental_manifest.json
ZESOLVER_LEGACY_INDEX_ROOT=/home/tristan/zesolver_index
```

Command:

```bash
.venv/bin/python tools/run_regression_suite.py --corpus
```

Result:

```text
4 passed, 2 skipped, 322 deselected
compileall: OK
runner status: PASS
```

Expected skips:

```text
tests/test_catalog290.py: external ASTAP/HNSKY family 'g05' not found under /opt/astap
tests/test_regression_blind4d.py: blind4d source FITS paths not mapped yet: blind4d_p29_232329
```

## ZN3.10B

The ZN3.10B root used in P2B-0 remains configured and present:

```text
<home>/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840
```

The corpus runner resolves the configured ZN3.10B paths with that root. The last
full automated P2B-0 validation remains:

```text
8 solved, 0 skipped, 0 failed
CONTROL_NEAR_CORRECT=3
NOHINT_4D_CORRECT=3
BADHINT_4D_CORRECT=2
historical_blind_called=0
astrometry_web_called=0
verdict=PASS
```

No ZN3.10B files were modified during P2B-0.2.

## Full Validation

```text
.venv/bin/python tools/run_regression_suite.py --corpus
4 passed, 2 skipped, 322 deselected; compileall OK; runner PASS

.venv/bin/python tools/run_regression_suite.py --hermetic
321 passed, 1 skipped, 6 deselected, 1 warning; compileall OK; runner PASS

.venv/bin/python -m pytest -q
325 passed, 3 skipped, 1 warning

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK

git diff --check
OK
```

## Conclusion

```text
READY_FOR_CORE_EXTRACTION
```
