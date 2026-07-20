# Settings Public Surface

P2A classifies the current settings surface without removing GUI controls.

| Réglage | Décision | Raison |
| ------- | -------- | ------ |
| Catalogue library | `KEEP_VISIBLE` | Core user choice replacing raw catalogue/index paths. |
| Input/output selection | `KEEP_VISIBLE` | Main workflow. |
| WCS overwrite/preserve | `KEEP_VISIBLE` | Safety-critical product choice. |
| Workers / Auto | `KEEP_VISIBLE` | Legitimate performance control. |
| GPU mode auto/cpu/cuda | `KEEP_VISIBLE` | Hardware choice, not solve threshold. |
| Web fallback | `KEEP_VISIBLE` | External-service consent. |
| Language/log level | `KEEP_VISIBLE` | Product preference. |
| FOV/focal/pixel/scale hints | `MOVE_TO_ADVANCED` | Useful user hints, can bias solving. |
| Downsample | `MOVE_TO_ADVANCED` | User-visible but can affect results. |
| Direct `db_root` | `DEPRECATE_LATER` | Legacy rollback path until CatalogLibrary GUI exists. |
| Direct `index_root` | `MOVE_TO_DEVELOPER_TOOL` | Index build/diagnostic surface. |
| Direct 4D manifest path | `DEPRECATE_LATER` | Temporary compatibility until library manifests are complete. |
| Family selection | `HIDE_INTERNAL` | Profile/library should choose families. |
| Near thresholds and RANSAC caps | `HIDE_INTERNAL` | `zenear-v1`. |
| Blind candidates/quads/inliers/RMS | `HIDE_INTERNAL` | `zeblind4d-v1`. |
| Bucket caps/vote percentile | `MOVE_TO_DEVELOPER_TOOL` | Developer override only. |
| Index build parameters | `MOVE_TO_DEVELOPER_TOOL` | Can alter generated artifacts. |
| Benchmark fields | `MOVE_TO_DEVELOPER_TOOL` | Not normal product workflow. |
| Historical blind profile | `MOVE_TO_DEVELOPER_TOOL` | Diagnostic compatibility only. |

No GUI control is removed in P2A.
