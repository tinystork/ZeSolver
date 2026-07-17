# Settings Inventory

P2A-1 inventory before moving settings. No setting is removed or migrated by
this document.

## Sources Inspected

- `zesolver.SolveConfig`
- `zeblindsolver.metadata_solver.NearSolveConfig`
- `zeblindsolver.zeblindsolver.SolveConfig`
- `zesolver.settings_store.PersistentSettings`
- GUI builders in `zesolver.py`
- CLI arguments in `zesolver.py`
- current `zeblindsolver.profiles`
- environment variables used by the app and tests
- index builders, benchmark and diagnostic tools

## Inventory Table

| Paramètre | Défini dans | Consommé par | Défaut | Persistant | Catégorie cible | Risque | Profil actuel |
| --------- | ----------- | ------------ | ------ | ---------: | --------------- | ------ | ------------- |
| `catalog_library_path` | `SolveConfig`, `PersistentSettings`, CLI | P1C catalog resource resolver | `None` | Yes | PRODUCT | Low | `pipeline-v1` resource input |
| `db_root` | `SolveConfig`, `PersistentSettings`, CLI/GUI legacy | `CatalogDB`, index builder, compatibility | required/`None` | Yes | DEPRECATED | Medium: raw source path | legacy compatibility |
| `index_root` / `blind_index_path` | `PersistentSettings`, `SolveConfig`, CLI/GUI legacy | historical index build/diagnostics, Astrometry fallback metadata | `None` | Yes | DEPRECATED | Medium: raw derived-index path | legacy diagnostic |
| `blind_4d_manifest_path` | `SolveConfig`, `PersistentSettings`, CLI/GUI legacy | strict 4D manifest loader | `None` | Yes | DEPRECATED | Medium: raw internal manifest path | `zeblind4d-v1` resource input until library complete |
| `blind_4d_loaded_manifest` | `SolveConfig` runtime | `ensure_loaded_4d_manifest`, `build_blind_solve_config` | `None` | No | RUNTIME | Low | derived runtime cache |
| `families`, `solver_family`, `dev_family_selection`, `db_family_cache` | `SolveConfig`, GUI, `PersistentSettings` | `CatalogDB`, family dropdown/dev selection | `None`/Auto | Yes | DEPRECATED | Medium: exposes catalogue internals | library/profile should decide |
| `input_dir` | `SolveConfig`, CLI/GUI | file collection, batch runner | required | No | RUNTIME | Low | request/runtime |
| `formats`, `solver_formats` | `SolveConfig`, CLI/GUI | file collection | FITS+raster | Yes | PRODUCT | Low | product setting |
| `max_files`, `solver_max_files` | `SolveConfig`, CLI/GUI | file collection | `None` / `0` | Yes | RUNTIME | Low | runtime/debug |
| `overwrite`, `solver_overwrite` | `SolveConfig`, GUI/CLI | WCS writing policy | `True` | Yes | PRODUCT | High: file writes | product setting |
| `workers`, `solver_workers` | `SolveConfig`, GUI/CLI | batch worker strategy | app auto / `1` | Yes | PRODUCT | Medium: perf/stability | runtime resolves |
| `io_concurrency` | `SolveConfig`, `PersistentSettings`, GUI | I/O semaphore | `0` auto | Yes | RUNTIME | Medium: perf/stability | runtime option |
| `gc_interval` | `SolveConfig`, CLI | batch GC hook | `0` | No | DEVELOPER | Low | developer override |
| `log_level`, `benchmark_log_level` | `SolveConfig`, `PersistentSettings`, CLI/GUI | logging | `INFO` | Yes | PRODUCT | Low | product setting |
| `interface_mode` | `PersistentSettings`, GUI | GUI visibility | `easy` | Yes | PRODUCT | Low | product setting |
| `solver_backend` | `PersistentSettings`, GUI | local vs Astrometry path | `local` | Yes | PRODUCT | Medium: external service | product setting |
| `astrometry_api_url` | `SolveConfig`, `PersistentSettings`, GUI | Astrometry.net client | `https://nova.astrometry.net/api` | Yes | PRODUCT | Medium: external endpoint | product setting |
| `astrometry_api_key` | `SolveConfig`, `PersistentSettings`, env | Astrometry.net client | `None` | Yes | PRODUCT | High: secret/external action | product setting, handle carefully |
| `astrometry_parallel_jobs` | `SolveConfig`, `PersistentSettings` | web fallback batch | `2` | Yes | PRODUCT | Medium | product setting |
| `astrometry_timeout_s` | `SolveConfig`, `PersistentSettings` | web fallback timeout | `600` | Yes | PRODUCT | Low | product setting |
| `astrometry_use_hints` | `SolveConfig`, `PersistentSettings` | web fallback hints | `True` | Yes | PRODUCT | Low | product setting |
| `astrometry_fallback_after_blind`, `astrometry_fallback_local` | `SolveConfig`, `PersistentSettings` | pipeline fallback | `True` | Yes | PRODUCT | Medium: chain behavior | `pipeline-v1` + product toggle |
| `blind_enabled`, `solver_blind_enabled` | `SolveConfig`, GUI/CLI | pipeline Near->Blind handoff | `True` | Yes | PRODUCT | Medium | `pipeline-v1` + product toggle |
| `blind_only` | `SolveConfig`, CLI | skips Near/CatalogDB | `False` | No | RUNTIME | Medium | runtime option |
| `blind_skip_if_valid` | `SolveConfig` | blind WCS skip policy | `True` | No | RUNTIME | Medium | pipeline behavior |
| `blind_backend_profile` | `SolveConfig`, `PersistentSettings`, CLI/GUI | profile selection | `zeblind_4d_experimental` | Yes | PRODUCT | Medium: profile id only | profile reference |
| `fov_deg`, `solver_fov_deg` | `SolveConfig`, GUI/CLI | Near search radius/hints | `1.5` | Yes | PRODUCT | Low | product hint |
| `downsample`, `solver_downsample` | `SolveConfig`, GUI/CLI | image loading and solvers | `1` | Yes | PRODUCT | Medium: can change solve result | product advanced |
| `mag_limit` | `SolveConfig`, CLI | catalogue filtering legacy | `None` | No | DEPRECATED | Medium | legacy/dev |
| `cache_size`, `solver_cache_size`, `tile_cache_size`, `near_tile_cache_size` | app/near/blind configs, settings | catalogue/tile caches | `12`/`128` | Yes | RUNTIME | Low/medium perf | runtime/profile |
| `hint_ra_deg`, `hint_dec_deg`, `hint_radius_deg` and `solver_hint_*` | `SolveConfig`, GUI/CLI | Near/Blind/Web hints | `None` | Yes | PRODUCT | Medium: can bias solve | product hint |
| `hint_focal_mm`, `hint_pixel_um`, `hint_resolution_*` and `solver_hint_*` | `SolveConfig`, GUI/CLI | scale/FOV hints | `None` | Yes | PRODUCT | Medium | product hint |
| `last_preset_id`, `last_fov_*` | `PersistentSettings`, GUI presets | GUI FOV helper | `None`/`0` | Yes | PRODUCT | Low | product helper |
| `near_max_tile_candidates` | app/near config, settings, CLI | Near tile selection cap | `48` | Yes | NEAR_PROFILE | High: search coverage | `zenear-v1` |
| `near_tile_cache_size` | app/near config, settings, CLI | Near tile cache | `128` | Yes | RUNTIME | Low | runtime/profile |
| `near_detect_backend`, `near_detect_device` | app/near config, settings, CLI | Near detector device | `auto`/`None` | Yes | PRODUCT | Medium: hardware | product/runtime |
| `near_detect_k_sigma`, `near_detect_min_area`, `near_detect_max_labels` | app/near config, settings, CLI | Near star detection | `4.5`, `8`, `1200` app | Yes | NEAR_PROFILE | High: solve behavior | `zenear-v1` |
| `near_detect_gpu_slots` | app/near config, settings | Near GPU guard | `1` | Yes | RUNTIME | Medium perf/stability | runtime option |
| `near_warm_start` | app config, settings | sequential Near seed | `True` | Yes | NEAR_PROFILE | Medium | `pipeline-v1`/`zenear-v1` |
| `near_quality_inliers`, `near_quality_rms`, `near_pixel_tolerance` | app/near config, settings | Near acceptance/validation | `60`, `1.0`, `3.0` | Yes | NEAR_PROFILE | High | `zenear-v1` |
| `near_ransac_trials`, `near_ransac_seed` | app/near config, settings | Near RANSAC | `1200`, auto | Yes | NEAR_PROFILE | High | `zenear-v1` with seed runtime override |
| `near_max_img_stars`, `near_max_cat_stars` | app/near config, settings | Near candidate lists | `800`, `2000` | Yes | NEAR_PROFILE | High | `zenear-v1` |
| `near_try_parity_flip`, `near_search_margin` | app/near config, settings | Near matching/search | `True`, `1.2` app | Yes | NEAR_PROFILE | High | `zenear-v1` |
| `near_astap_iso_strict` | app/near config, settings/CLI compat | Near strict path | `True` forced | Yes | DEPRECATED | Medium: legacy key | fixed in `zenear-v1` |
| `near_defer_blind_fallback` | app config/settings/CLI | batch handoff policy | `False` | Yes | PIPELINE_PROFILE | High: known rollback | `pipeline-v1` |
| `near_allow_second_rescue`, `near_astap_hint_fastpath`, `near_second_pass_refine_in_fastpath`, `near_astap_hint_radius_deg` | app config/settings | Near rescue/hint experiments | mostly `False` | Yes/No | NEAR_PROFILE | High | `zenear-v1` fixed |
| `search_radius_scale`, `search_radius_attempts`, `max_search_radius_deg` | app config/settings/CLI | Near search radius retries | `1.8`, `3`, `None` | Yes | NEAR_PROFILE | High | `zenear-v1` |
| `max_catalog_stars`, `max_image_stars`, `max_alignment_stars` | app config | legacy/app matching limits | `2000`, `200`, `60` | No | NEAR_PROFILE | Medium | `zenear-v1` legacy bridge |
| `blind_max_stars`, `blind_max_quads`, `blind_max_candidates` | app config/settings/CLI | Blind base config before profile | `500`, `8000`, `10` | Yes | BLIND_PROFILE | High | `zeblind4d-v1` |
| `blind_pixel_tolerance`, `blind_quality_inliers`, `blind_quality_rms` | app config/settings/CLI | Blind validation | `2.5`, `40`, `1.2` | Yes | BLIND_PROFILE | High | `zeblind4d-v1` |
| `blind_fast_mode` | app config/settings | Blind mode | `True` | Yes | BLIND_PROFILE | Medium | `zeblind4d-v1` |
| `blind_index_scale_overlap_prefilter_*` | settings/app blind config | index scale prefilter | disabled, `0.05`, `0.95` | Yes | BLIND_PROFILE | High false-negative risk | off in `zeblind4d-v1` |
| `dev_bucket_limit_override`, `dev_vote_percentile`, `dev_bucket_cap_*` | app config/settings/CLI | Blind bucket/vote internals | app `0/40/0`; settings caps nonzero | Yes | DEVELOPER | High | developer override only |
| `dev_collect_matches_vectorized_experimental` | app config/CLI | Blind collect optimization | `False` | No | DEVELOPER | Medium | developer override |
| `dev_detect_k_sigma`, `dev_detect_min_area` | app config/settings/CLI | Blind detector dev override | `3.0`, `5` | Yes | DEVELOPER | High | developer override |
| `dev_hash_quads_*`, `mag_cap`, `max_stars`, `max_quads_per_tile`, `quad_storage`, `tile_compression` | settings/index builder | index construction | builder defaults | Yes | DEVELOPER | High: rebuild/index formats | developer tool |
| `benchmark_*` | `PersistentSettings`, GUI benchmark tab | benchmark tools | mixed | Yes | DEVELOPER | Low/medium | developer tool |
| `sample_fits` | settings GUI tools | settings test runners | `None` | Yes | DEVELOPER | Low | developer tool |
| `schema_version` | settings file | settings migration | `9` | Yes | PRODUCT | Medium migration | settings v1 legacy |
| `settings_schema_version` | new P2A format | future settings v2 | `2` | Yes | PRODUCT | Medium migration | settings v2 |
| `ProductSettings.profiles.*` | new P2A format | assembly layer | v1 ids | Yes | PRODUCT | Low | profile references only |
| `ZE_BLIND_WORKERS` | environment | blind worker override | unset | No | RUNTIME | Medium perf/stability | runtime override |
| `ZE_MEM_SNAPSHOT_INTERVAL_S` | environment | memory telemetry | `2` | No | RUNTIME | Low | runtime diagnostic |
| `ZE_NEAR_PARALLEL_MODE`, `ZE_NEAR_PROCESS_POOL` | environment | Near phase strategy | `auto`/enabled | No | RUNTIME | Medium | runtime diagnostic |
| `ZE_TILE_CACHE_SIZE` | environment | low-level blind tile cache | `128` | No | RUNTIME | Low | runtime override |
| `ZESOLVER_ASTAP_ROOT`, `ZESOLVER_BLIND4D_MANIFEST`, `ZEBLIND_4D_MANIFEST`, `ZESOLVER_LEGACY_INDEX_ROOT` | environment | P1C resource discovery/tests | unset | No | RUNTIME | Medium | controlled discovery |
| `ZESOLVER_CORPUS_ROOT`, `ZESOLVER_ZN310B_ROOT` | environment | regression corpus | unset | No | DEVELOPER | Low | tests only |
| `ASTROMETRY_API_KEY` | environment | web fallback | unset | No | PRODUCT | High secret/external | product runtime |
| `BlindSolveConfig.blind_astrometry_4d_*` | blind config/profile | 4D route internals | many | No direct app persistence except subset | BLIND_PROFILE | High | `zeblind4d-v1` |
| `BlindSolveConfig.blind_astrometry_*`, `blind_hypothesis_*`, `blind_verify_*`, `blind_scale_*`, `blind_pair*`, `blind_collect_*`, `verify_logodds_*`, `depth_ladder_*` | blind config/diagnostic CLI | historical/experimental blind internals | many | Mostly no | BLIND_PROFILE or DEVELOPER | High | freeze current effective subset in `zeblind4d-v1`; keep rest internal |
| `NearSolveConfig.diagnostic_*` | Near config/tools | diagnostic dumps | `None`/`False` | No | DEVELOPER | Low/medium | developer tool |

## Category Decisions

Allowed categories used above:

- `PRODUCT`: legitimate user-facing choice or profile identifier.
- `RUNTIME`: execution-local choice or resolved runtime state; not persisted
  automatically in the new format.
- `NEAR_PROFILE`: internal Near behavior frozen by profile id.
- `BLIND_PROFILE`: internal Blind behavior frozen by profile id.
- `PIPELINE_PROFILE`: internal orchestration behavior frozen by profile id.
- `DEVELOPER`: diagnostic/build/benchmark/internal override.
- `DEPRECATED`: legacy compatibility surface kept for rollback/migration.
- `UNKNOWN`: none currently assigned after this pass.

## P2A Boundary Decision

Move forward with:

- `ProductSettings`: user choices and profile ids only.
- `RuntimeOptions`: input/run-local execution state.
- `NearSolverProfile`, `BlindSolverProfile`, `PipelineProfile`: immutable v1
  baselines.
- `DeveloperOverrides`: explicit opt-in map, never loaded silently from old user
  settings.

No GUI control is removed in P2A. Existing persisted fields remain readable.
