# GUI Pipeline Capability Matrix

This matrix records the real P2B-2C capability boundary. It is intentionally
conservative: capabilities validated only by deterministic doubles are not
presented as production-ready.

| Capability | Pipeline | Legacy | Route AUTO initiale | Risk | Future Phase |
| ---------- | -------: | -----: | ------------------- | ---- | ------------ |
| FITS local Near | yes | yes | PIPELINE | Low; production Near port is covered. | P3A |
| FITS local Blind 4D | yes | yes | PIPELINE | Medium; real 4D port is additive and legacy remains rollback. | P3A |
| FITS Near then Blind | yes | yes | PIPELINE | Medium; ZN3.10B parity is the guardrail. | P3A |
| batch FITS workers=1 | yes | yes | PIPELINE | Medium; new batch is additive. | P3A |
| batch FITS threads >1 | yes | yes | PIPELINE | Medium; thread scheduler exists, process scheduler remains legacy. | P3A |
| batch Near process | no | yes | LEGACY | Process worker initialization remains monolithic. | P2B-3 |
| batch hybrid CPU/GPU | no | yes | LEGACY | GPU/process heuristics remain legacy. | P2B-3 |
| io_concurrency | modeled | yes | LEGACY when required | Modeled but not enforced by new scheduler. | P2B-3 |
| cancellation before run | yes | yes | PIPELINE | Covered by new batch tests. | P3A |
| cancellation during Near | partial | yes | LEGACY when strict parity required | Depends on engine cancellation behavior. | P2B-3 |
| cancellation between Near and Blind | yes | yes | PIPELINE | Covered by batch phase routing. | P3A |
| cancellation during Blind | partial | yes | LEGACY when strict parity required | Blind engine cancellation is best-effort. | P2B-3 |
| TIFF | no | yes | LEGACY | Raster sidecars remain legacy. | P3/P4 |
| PNG | no | yes | LEGACY | Raster sidecars remain legacy. | P3/P4 |
| JPEG | no | yes | LEGACY | Raster sidecars remain legacy. | P3/P4 |
| sidecars WCS raster | no | yes | LEGACY | Not implemented in SolverPipeline. | P3/P4 |
| Astrometry.net web | no | yes | LEGACY | Fallback web policy unchanged. | Later |
| fallback web after Blind | no | yes | LEGACY | Must not silently happen in Pipeline mode. | Later |
| hints FITS | yes | yes | PIPELINE | Metadata extraction remains preflight/pipeline compatible. | P3A |
| hints user | yes | yes | PIPELINE | ProductSettings carries user hints. | P3A |
| hints adaptatifs entre images | no | yes | LEGACY | Still in `ImageSolver`. | P2B-3 |
| warm start Near | no | yes | LEGACY | Still legacy. | P2B-3 |
| CatalogLibrary | yes | compat | PIPELINE | READY_PARTIAL remains explicit. | P3A |
| couverture Blind partielle | yes | yes | PIPELINE | Must never be promoted all-sky. | P3A |
| WCS existant | yes | yes | PIPELINE | Preflight rejects overwrite-forbidden case. | P3A |
| écriture dans copie | yes | yes | PIPELINE | `wcs_io.py` owns final write. | P3A |
| écriture dans source | yes | yes | PIPELINE | Controlled by request/output mode. | P3A |
| progression GUI | no | yes | LEGACY | No GUI rewire in P2B-2C. | P3A |
| fermeture du GUI pendant run | no | yes | LEGACY | Qt lifecycle remains legacy. | P3A |

AUTO policy for P3A should use the new pipeline for validated FITS-local cases
and legacy for raster, web fallback, process/hybrid batch, adaptive hints, and
unknown capabilities.
