# Engine Selection Policy

P2B-2C introduces a pure selection policy in `zesolver.engine_selection`.
It is not wired into the GUI yet.

Modes:

```text
AUTO
PIPELINE
LEGACY
```

## AUTO

AUTO selects `PIPELINE` only for capabilities validated by the new core:

```text
local FITS
single FITS Near
single FITS Blind 4D
single FITS Near then Blind
batch FITS with thread scheduler
CatalogLibrary resources
partial Blind 4D coverage reported as partial
```

AUTO selects `LEGACY` with a reason for:

```text
raster TIFF/PNG/JPEG
Astrometry.net web backend
fallback web after Blind
process or hybrid batch scheduling
raster WCS sidecars
adaptive inter-image hints
unknown capabilities
unknown file types
```

## PIPELINE

Explicit `PIPELINE` mode is strict. If the request needs a capability not
supported by SolverPipeline, the selector returns:

```text
selected_mode=PIPELINE
supported=false
reason=pipeline_unsupported: ...
```

There is no silent fallback to legacy in explicit Pipeline mode.

## LEGACY

Explicit `LEGACY` mode always selects the legacy path. This preserves rollback
while P3A wires the GUI progressively.

## Blind 4D Coverage

Partial 4D coverage is represented as a warning:

```text
blind4d_coverage_partial_not_all_sky
```

The selector does not promote partial coverage to all-sky capability.
