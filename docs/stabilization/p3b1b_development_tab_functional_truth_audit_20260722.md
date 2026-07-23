# P3B-1B - Audit de verite fonctionnelle de l'onglet Developpement

Date: 2026-07-22

## Objectif

Auditer les controles de l'onglet **Developpement** sans les deplacer ni les supprimer, et separer les reglages actifs du parcours produit actuel des reglages historiques, doublons et actions de maintenance.

Parcours produit de reference:

```text
CatalogLibrary -> ZeNear ASTAP-native -> ZeBlind 4D -> Astrometry.net optionnel
```

## Etat initial

Commandes executees avant modification:

```text
git status --short --branch
## test...origin/test
 M AGENT.md
 M zesolver.py
?? tests/test_gui_benchmark_removed.py

git diff --check
AGENT.md:3: trailing whitespace.
AGENT.md:4: trailing whitespace.
AGENT.md:5: trailing whitespace.
AGENT.md:6: trailing whitespace.
AGENT.md:7: trailing whitespace.
```

`zesolver.py` et `tests/test_gui_benchmark_removed.py` etaient deja modifies par la mission P3B-1A. Ces changements n'ont pas ete revertis.

## Fichiers inspectes

- `AGENT.md`
- `zesolver.py`
- `zesolver/settings_store.py`
- `zesolver/gui_profiles.py`
- `zesolver/gui_settings_sections.py`
- `zesolver/gui_pipeline/`
- `zesolver/settings/`
- `zesolver/solver_config/`
- `zesolver/core/pipeline.py`
- `zesolver/core/blind_port.py`
- `zesolver/core/blind_models.py`
- `zesolver/catalog_resources.py`
- `zeblindsolver/`
- `tools/`
- `tests/`

Recherches initiales effectuees avec les motifs demandes:

```text
dev_
bucket
vote
cap_S
cap_M
cap_L
detect_sigma
detect_min_area
family_auto
family
hash
rebuild
worker
threads
cache
```

## Inventaire fonctionnel

| Element GUI | Widget | Cle traduction | Cle PersistentSettings | Config intermediaire | Product 4D | Legacy | Effet reel | Classification | Recommandation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Limite paires/quads | `dev_bucket_spin` | `dev_bucket_label` | `dev_bucket_limit_override` | `SolveConfig.dev_bucket_limit_override`; `BlindConfigInputs.dev_bucket_limit_override`; `BlindSolveConfig.bucket_limit_override` | Non | Oui | Consommee par `zeblindsolver.solve_blind()` pour limiter les buckets/quads dans le backend historique. Non injectee par la route produit PIPELINE. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique. |
| Percentile votes | `dev_vote_spin` | `dev_vote_label` | `dev_vote_percentile` | `SolveConfig.dev_vote_percentile`; `BlindSolveConfig.vote_percentile` | Non | Oui | Influence la selection par votes dans le backend blind historique. Non injecte par la route produit PIPELINE. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique. |
| Cap niveau S | `dev_cap_s_spin` | `dev_bucket_cap_s_label` | `dev_bucket_cap_S` | `SolveConfig.dev_bucket_cap_S`; `BlindSolveConfig.bucket_cap_S` | Non | Oui | Applique un cap de bucket S via `set_bucket_cap_overrides()` dans le backend historique. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique ou outils diagnostic S/M/L. |
| Cap niveau M | `dev_cap_m_spin` | `dev_bucket_cap_m_label` | `dev_bucket_cap_M` | `SolveConfig.dev_bucket_cap_M`; `BlindSolveConfig.bucket_cap_M` | Non | Oui | Applique un cap de bucket M via `set_bucket_cap_overrides()` dans le backend historique. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique ou outils diagnostic S/M/L. |
| Cap niveau L | `dev_cap_l_spin` | `dev_bucket_cap_l_label` | `dev_bucket_cap_L` | `SolveConfig.dev_bucket_cap_L`; `BlindSolveConfig.bucket_cap_L` | Non | Oui | Applique un cap de bucket L via `set_bucket_cap_overrides()` dans le backend historique. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique ou outils diagnostic S/M/L. |
| Sigma detection developpement | `dev_sigma_spin` | `dev_detect_sigma_label` | `dev_detect_k_sigma` | `SolveConfig.dev_detect_k_sigma`; `BlindSolveConfig.detect_k_sigma` | Non | Oui, blind historique | Influence `zeblindsolver.star_detect.detect_stars()` via la config blind historique. ZeNear produit utilise `near_detect_k_sigma` depuis l'onglet Solveur rapide/profil, pas ce champ. | `ACTIVE_LEGACY_ONLY` | Renommer/deplacer en detection blind historique si conserve. |
| Surface minimale developpement | `dev_area_spin` | `dev_detect_area_label` | `dev_detect_min_area` | `SolveConfig.dev_detect_min_area`; `BlindSolveConfig.detect_min_area` | Non | Oui, blind historique | Influence `detect_stars(min_area=...)` dans le blind historique. ZeNear produit utilise `near_detect_min_area`, pas ce champ. | `ACTIVE_LEGACY_ONLY` | Renommer/deplacer en detection blind historique si conserve. |
| Downsample | `downsample_spin` | `downsample_label` | `solver_downsample` | `SolveConfig.downsample`; `ProductSettings.downsample`; `BlindSolveConfig.downsample` | Oui | Oui | Transmis a `ProductSettings` puis aux configs produit; egalement consomme par le builder blind historique. | `ACTIVE_PRODUCT` | Garder dans Reglages avances ou Solveur, hors bloc legacy. |
| Cache catalogue | `cache_spin` | `cache_label` | `solver_cache_size` | `SolveConfig.cache_size` | Non | Oui | Passe a `CatalogDB(..., cache_size=...)` dans `ImageSolver`. Le parcours produit utilise plutot `near_tile_cache_size` depuis Performance. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique; ne pas confondre avec cache near produit. |
| Threads solveur | `dev_workers_combo` | `dev_workers_label` | `solver_workers` | `_dev_workers_choice`; `SolveConfig.workers`; `ProductSettings.workers`; `RuntimeOptions.worker_count_resolved` | Oui | Oui | Synchronise avec `workers_spin` du panneau Solveur et utilise au lancement via `_effective_workers_for_run()`. | `DUPLICATE` | Garder une seule presentation, idealement Performance. |
| Familles Auto | `dev_family_auto_check` | `dev_family_auto` | `dev_family_auto` | `SolveConfig.families` seulement si le combo familles principal reste en Auto | Non avec CatalogLibrary | Oui | En mode CatalogLibrary, `resolve_catalog_resources()` renvoie les familles du manifest et `apply_catalog_resources_to_config()` ecrase `SolveConfig.families`. En legacy, limite `CatalogDB` aux familles choisies. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique. |
| Familles manuelles | `_dev_family_checks` dans `dev_family_box` | `dev_family_group`, `dev_family_hint`, `dev_family_missing`, `dev_family_none_error` | `dev_family_selection` | `SolveConfig.families` en fallback legacy | Non avec CatalogLibrary | Oui | Meme comportement que l'option Auto: utile uniquement pour les anciens index/bases lorsque le parcours produit n'utilise pas CatalogLibrary. | `ACTIVE_LEGACY_ONLY` | Deplacer vers Compatibilite historique. |
| Reconstruction hashes S/M/L | `dev_hash_buttons["S/M/L"]` | `dev_hash_group`, `dev_hash_button`, `dev_hash_started`, `dev_hash_done`, `dev_hash_failed`, `dev_hash_busy` | n/a | `IndexBuilder(quads_only=True, levels=(level,))` | n/a | Maintenance historique | Declenche une reconstruction de quads/hash via `IndexBuilder`; ne construit pas `SolveConfig` et ne lance pas un solve. | `MAINTENANCE_ACTION` | Deplacer vers Outils avances. |
| Quads par reconstruction S/M/L | `dev_hash_quads_spin["S/M/L"]` | `dev_hash_value_hint` | `dev_hash_quads_S/M/L` | `IndexBuilder(level_quads=...)` | n/a | Maintenance historique | Parametre uniquement l'action de reconstruction S/M/L; aucun effet sur un run normal. | `MAINTENANCE_ACTION` | Deplacer avec les boutons de reconstruction. |
| Bouton Enregistrer developpement | `dev_save_btn` | `settings_save_btn` | plusieurs cles du tab | `PersistentSettings` | Indirect | Indirect | Persiste les valeurs du tab; ne declenche aucune branche solveur. | `PERSISTED_BUT_NO_EFFECT` | Remplacer par une sauvegarde globale ou autosave lors du regroupement. |

## Preuves principales

### Route GUI vers settings

`_build_dev_tab()` cree les widgets du tab Developpement. `_save_dev_settings()` et `_build_config()` recopient leurs valeurs vers `PersistentSettings`.

`_build_config()` cree ensuite `SolveConfig` avec:

- `dev_bucket_limit_override`
- `dev_vote_percentile`
- `dev_bucket_cap_S/M/L`
- `dev_detect_k_sigma`
- `dev_detect_min_area`
- `downsample`
- `cache_size`
- `workers`
- `families`

### Route produit PIPELINE

La route GUI produit passe par `build_gui_solve_request_from_legacy_config()` puis `build_product_settings()`.

Fait demontre: `ProductSettings` ne possede pas les champs `dev_bucket_*`, `dev_vote_*`, `dev_detect_*` ou `dev_family_*`.

`build_solver_configuration()` ne copie des `DeveloperOverrides` que si un canal explicite `DeveloperOverrides(enabled=True, values=...)` est fourni. Le GUI principal ne fournit pas ce canal.

Consequence: un champ developpement peut etre sauvegarde puis recopie dans `SolveConfig`, mais ne pas atteindre la configuration effective du runtime ZeBlind 4D produit.

### Route ZeNear produit

`ExistingNearSolverPort` construit `NearSolveConfig` avec:

- `near_detect_k_sigma`
- `near_detect_min_area`
- `near_tile_cache_size`
- `near_max_tile_candidates`

Ces valeurs viennent des profils/onglets Near/Performance, pas des widgets `dev_sigma_spin` et `dev_area_spin`.

### Route Blind 4D produit

`ProductionBlindSolverPort.build_config()` appelle `build_blind_config_inputs()` sur `configuration.legacy_solve_config_values`.

Fait demontre par test: sans `DeveloperOverrides`, `legacy_solve_config_values` ne contient aucune cle `dev_*`, et le `BlindSolveConfig` produit garde:

- `bucket_limit_override = 0`
- `vote_percentile = 40`
- `bucket_cap_S/M/L = 0`
- `detect_k_sigma = 3.0`
- `detect_min_area = 5`

Le champ `downsample`, lui, passe bien par `ProductSettings.downsample` et atteint le blind 4D produit.

### Route legacy

`ImageSolver._run_blind()` utilise `build_blind_solve_config(self.config, ...)`.

`build_blind_solve_config()` mappe directement les champs `SolveConfig.dev_*` vers `BlindSolveConfig`, puis `zeblindsolver.solve_blind()` les consomme dans le backend historique.

Fait demontre par test: une config legacy avec `dev_bucket_limit_override=777`, `dev_vote_percentile=61`, `dev_bucket_cap_S/M/L=111/222/333`, `dev_detect_k_sigma=4.8` et `dev_detect_min_area=9` produit bien un `BlindSolveConfig` avec ces valeurs.

### Familles et CatalogLibrary

`_build_config()` applique la selection manuelle `dev_family_selection` seulement si le combo familles principal reste en Auto.

Ensuite `resolve_catalog_resources_for_config()` privilegie `catalog_library=config.catalog_library_path`. Quand une `CatalogLibrary` est valide, `_resources_from_library()` renvoie les familles du manifest; `apply_catalog_resources_to_config()` ecrase `SolveConfig.families` par `resources.near.families`.

Conclusion: les familles manuelles de Developpement sont utiles au rollback legacy, mais pas comme source de verite du parcours produit CatalogLibrary.

### Threads

`dev_workers_combo` et `workers_spin` modifient la meme valeur logique:

- `_on_dev_workers_changed()` met a jour `_dev_workers_choice` et `solver_workers`, puis synchronise `workers_spin`.
- `_on_workers_spin_changed()` met a jour `_dev_workers_choice`, resynchronise le combo et met a jour `solver_workers`.
- `_build_config()` utilise `_effective_workers_for_run()`.

Conclusion: fonctionnel, mais duplicate visuel.

### Reconstruction hashes

`_rebuild_hash_level()` verifie `db_root` et `index_root`, lit `dev_hash_quads_spin[level]`, construit `IndexBuilder(quads_only=True, levels=(level_key,), level_quads=...)`, branche les callbacks et demarre le worker.

Il ne construit pas de `SolveConfig`, ne lance pas `ImageSolver`, et ne fait pas partie d'un run normal.

## Tests ajoutes

Fichier ajoute:

```text
tests/test_gui_development_tab_audit.py
```

Tests:

- `test_development_dev_fields_are_not_product_settings`
- `test_product_blind4d_runtime_does_not_receive_gui_dev_overrides_without_developer_channel`
- `test_legacy_blind_builder_consumes_development_bucket_vote_cap_and_detection_fields`
- `test_development_workers_control_is_synchronized_with_solver_workers_spin`
- `test_hash_rebuild_controls_start_quads_only_indexbuilder_not_a_solve_run`

Resultat cible initial:

```text
.venv/bin/python -m pytest tests/test_gui_development_tab_audit.py -q
5 passed
```

## Zones incertaines

Aucun controle visible de l'onglet Developpement n'est classe obsolescent sans preuve.

Zone a traiter prudemment en P3B-1C: le bouton `dev_save_btn` n'est pas un reglage solveur, mais il persiste des valeurs dont certaines restent actives en legacy ou produit (`downsample`, `workers`). Il ne doit donc pas etre supprime avant d'avoir remplace le flux de sauvegarde.

## Organisation cible recommandee

Organisation proposee a partir de l'audit:

```text
Reglages avances
├── Downsample
├── Detection Near reellement active (depuis l'onglet Solveur rapide)
└── Restauration des valeurs recommandees

Performance
├── Threads
├── Concurrence I/O
├── Cache Near produit
├── GPU / CPU detection
└── Warm start

Compatibilite historique
├── Familles manuelles
├── Cache catalogue legacy
├── Sigma/surface detection blind historique
├── Parametres buckets / votes / caps S-M-L
└── Backend historique

Outils avances
├── Reconstruction hashes S/M/L
├── Construction / reparation d'index
├── Index checker
├── Explorateur de catalogues
└── Benchmark
```

## Prochaine mission unique recommandee

P3B-1C: deplacer hors du parcours principal les controles `ACTIVE_LEGACY_ONLY` et `MAINTENANCE_ACTION` identifies ici, en conservant `Downsample` et en dedoublonnant `Threads` vers Performance.

## Validation

### Tests cibles et GUI

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_development_tab_audit.py tests/test_gui_* -q
60 passed, 1 skipped, 3 warnings
```

Skip:

- `tests/test_gui_controller_zn310b.py`: `ZESOLVER_ZN310B_ROOT` non defini.

Warnings:

- `multiprocessing.popen_fork`: warning deprecation `fork()` dans processus multi-thread, deja observe sur les tests GUI/legacy.

### Barrieres generales

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK
```

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
status: PASS
580 passed, 1 skipped, 9 deselected, 58 warnings
```

Skip hermetique:

- `tests/test_real_s50.py`: S50 index/frame non configure.

```text
.venv/bin/python -m pytest -q
580 passed, 10 skipped, 58 warnings
```

Skips complets:

- `tests/test_batch_pipeline_zn310b.py`: `ZESOLVER_ZN310B_ROOT` non defini.
- `tests/test_catalog290.py`: base ASTAP/HNSKY externe absente.
- `tests/test_gui_controller_zn310b.py`: `ZESOLVER_ZN310B_ROOT` non defini.
- `tests/test_real_s50.py`: S50 index/frame non configure.
- `tests/test_regression_blind4d.py`: `ZESOLVER_BLIND4D_MANIFEST` non defini.
- `tests/test_regression_blind4d.py`: mapping FITS `blind4d_p29_232329` absent.
- `tests/test_regression_near.py`: `ZESOLVER_CORPUS_ROOT` non defini.
- `tests/test_regression_pipeline.py`: `ZESOLVER_ZN310B_ROOT` non defini.
- `tests/test_solver_pipeline_zn310b_production.py`: `ZESOLVER_ZN310B_ROOT` non defini.

```text
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
PASS
```

Warnings complets:

- `datetime.utcnow()` deprecation dans `zeblindsolver/db_convert.py`.
- `multiprocessing.popen_fork` deprecation dans tests multiprocessing/legacy.
- `astropy.io.fits.card.VerifyWarning` sur une carte longue tronquee.

### Inspection graphique reelle

Inspection effectuee en session graphique reelle, plateforme Qt `wayland`, sans `QT_QPA_PLATFORM=offscreen`.

Resultat:

- FR Expert: onglets visibles `Solveur`, `Base de donnees`, `Developpement`, `Reglages`, `Performance`, `Parametres fast solver`, `Astrometry.net`.
- EN Expert: onglets visibles `Solver`, `Database`, `Development`, `Settings`, `Performance`, `Fast solver settings`, `Astrometry.net`.
- EN Easy: onglets visibles `Solver`, `Settings`.
- Toggle Easy -> Expert: l'onglet `Development` redevient visible.
- Tous les onglets Expert visibles ont ete ouverts une fois.
- L'onglet Developpement est masque en Easy mais ses widgets restent construits, notamment `dev_bucket_spin`, `dev_vote_spin`, `dev_cap_s/m/l_spin`, `dev_sigma_spin`, `dev_area_spin`, `downsample_spin`, `cache_spin`, `dev_workers_combo`, `dev_family_group`, `dev_hash_group`, `dev_save_btn`.
- Boutons et spins de hashes detectes pour `S`, `M`, `L`.
- Fermeture normale confirmee.
- Aucune erreur terminal observee pendant l'inspection.

### Etat Git final attendu

```text
git diff --check
PASS

git status --short --branch
## test...origin/test
 M AGENT.md
 M zesolver.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_development_tab_audit.py
```

Note: `zesolver.py` et `tests/test_gui_benchmark_removed.py` proviennent de P3B-1A et etaient deja presents dans l'etat initial.

## Gate

Les controles de l'onglet Developpement ont ete inventories, traces et classes. Les tests et barrieres demandees sont verts, avec uniquement des skips externes/configuration locale documentes.

```text
READY_FOR_P3B1C_NEXT_GUI_SIMPLIFICATION
```
