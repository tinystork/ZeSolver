# ZeBlind 4D Experimental Profile

`zeblind_4d_experimental` is an explicitly enabled ZeBlind profile for the
Astrometry-like 4D backend:

```text
astrometry_ab_code_4d_v1
```

The normal local runtime chain is now ZeNear -> ZeBlind 4D. The 4D profile is
not an all-sky solver and does not discover or download indexes automatically.
The legacy backend is a diagnostic/development backend only and is disabled in
the normal runtime chain.

## Status

Validated diagnostic scope:

- M106: 30/30.
- NGC6888: 10/10.
- M31: 9/9.
- S50 and S30 scale regimes.
- Alt-Az and EQ examples.
- P2.20 mixed pool: 15/15 with the same six indexes for every image.
- Offline false positives: 0.
- Wrong-region accepts: 0.

C11 is not validated yet.

## Runtime Manifest

The package loader is:

```python
from zeblindsolver.index_manifest_4d import load_4d_index_manifest
```

The application manifest is:

```text
config/zeblind_4d_experimental_manifest.json
```

It uses relative index paths into:

```text
indexes/astrometry_4d/
```

The local `.npz` files are intentionally not source-controlled. The manifest
contains SHA-256 checksums and is rejected if any enabled index is absent,
corrupt, incompatible, duplicated, or metadata-inconsistent.

The current six tile slices are:

```text
d50_2823
d50_2822
d50_2644
d50_2645
d50_2602
d50_2702
```

No image name, target name, object header, RA/Dec centre, or image-to-index
mapping is stored in the manifest.

## Profile Contract

The central profile is available through:

```python
from zeblindsolver.profiles import get_solver_profile

profile = get_solver_profile("zeblind_4d_experimental")
```

Validated parameters:

```text
quad_sources = 120
verification_sources = full
validation_catalog_policy = union_candidate_tiles
accept_policy = best_within_budget

quality_inliers = 40
quality_rms = 1.2
match_radius_px = 3.0

max_quads = 2500
max_hypotheses = 2000
max_accepts = 64
max_wall_s = 45.0
```

The final metric is direct pixel residuals:

```text
catalogue world2pix -> image
```

`legacy_inverse_*` remains telemetry only.

## CLI Usage

Default local 4D chain:

```bash
python zesolver.py \
  --headless \
  --db-root /path/to/catalogues \
  --input-dir /path/to/fits \
  --blind-profile zeblind_4d_experimental \
  --blind-4d-manifest config/zeblind_4d_experimental_manifest.json
```

With explicit scale bounds:

```bash
python zesolver.py \
  --headless \
  --db-root /path/to/catalogues \
  --input-dir /path/to/fits \
  --blind-profile zeblind_4d_experimental \
  --blind-4d-manifest config/zeblind_4d_experimental_manifest.json \
  --pixel-scale-min 1.90 \
  --pixel-scale-max 2.85 \
  --overwrite
```

For S30 use the fixed experimental scale range:

```text
3.19 .. 4.79 arcsec/px
```

For S50 use:

```text
1.90 .. 2.85 arcsec/px
```

Do not derive these ranges from the WCS of the image being solved.

## Tester ZeBlind 4D In The GUI

The GUI exposes the profile in a deliberately small form.

Easy and Wizard modes:

1. Open ZeSolver.
2. Enable `Activer le blind solver` / `Enable blind solver`.
3. Enable `Utiliser ZeBlind 4D` / `Use ZeBlind 4D`.
4. Verify that the installed indexes are present when prompted.
5. Use a bounded, coherent pixel-scale range:
   - S50: `1.90 .. 2.85 arcsec/px`;
   - S30: `3.19 .. 4.79 arcsec/px`.
6. For first tests, run on copies of the FITS files.

Expert mode:

1. Select `Profil ZeBlind` / `ZeBlind profile`.
2. Choose `ZeBlind 4D` / `ZeBlind 4D`.
3. Keep the default manifest or choose a JSON manifest explicitly.
4. Press `Vérifier` / `Verify`.
5. Start the run only after the manifest reports the expected index count and
   tile list.

ZeBlind 4D coverage is limited to the installed indexes. It is not an all-sky
solver yet. The legacy diagnostic backend is not part of the normal Easy/Wizard
runtime chain.

## Operational Limits

- Only six compact tile indexes are currently covered.
- No all-sky index is bundled.
- No automatic index download or build is performed.
- The runtime cost is higher than specialized two-index probes.
- A corrupt manifest stops the run before solving.
- No fallback to the legacy diagnostic backend is performed after a 4D manifest or
  4D solve failure.
- The GUI exposes only the profile, manifest path, and manifest verification.
  Internal budgets and thresholds remain hidden.

## Legacy Diagnostic Backend

The legacy backend remains available only for explicit development diagnostics:

```bash
--blind-profile historical
```

It is not selected automatically and is not the default normal runtime chain.
