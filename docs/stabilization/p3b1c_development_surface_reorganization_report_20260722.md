# P3B-1C - Reorganisation de la surface Developpement

Date: 2026-07-22

## Objectif

Appliquer l'audit P3B-1B en retirant l'onglet **Developpement** du GUI principal, sans perte de capacite et sans changement astrometrique.

Rapport de reference:

```text
docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
```

## Etat Git initial

```text
git status --short --branch
## test...origin/test
 M AGENT.md
 M zesolver.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_development_tab_audit.py

git diff --check
PASS
```

Les changements P3B-1A/P3B-1B etaient deja presents et n'ont pas ete revertis.

## Inventaire avant migration

| Controle actuel | Classification P3B-1B | Nouvelle destination | Widget conserve/remplace | Cle persistee |
| --- | --- | --- | --- | --- |
| `downsample_spin` | `ACTIVE_PRODUCT` | Onglet Solveur | Conserve, reconstruit dans `options_box` | `solver_downsample` |
| `dev_workers_combo` | `DUPLICATE` | Supprime; `workers_spin` deplace dans Performance | Supprime | `solver_workers` |
| `dev_sigma_spin` | `ACTIVE_LEGACY_ONLY` | Reglages -> Compatibilite historique | Deplace | `dev_detect_k_sigma` |
| `dev_area_spin` | `ACTIVE_LEGACY_ONLY` | Reglages -> Compatibilite historique | Deplace | `dev_detect_min_area` |
| `dev_bucket_spin` | `ACTIVE_LEGACY_ONLY` | Reglages -> Compatibilite historique | Deplace | `dev_bucket_limit_override` |
| `dev_vote_spin` | `ACTIVE_LEGACY_ONLY` | Reglages -> Compatibilite historique | Deplace | `dev_vote_percentile` |
| `dev_cap_s/m/l_spin` | `ACTIVE_LEGACY_ONLY` | Reglages -> Compatibilite historique | Deplaces | `dev_bucket_cap_S/M/L` |
| familles Auto/manuelles | `ACTIVE_LEGACY_ONLY` | Reglages -> Compatibilite historique | Deplacees | `dev_family_auto`, `dev_family_selection` |
| `cache_spin` | `ACTIVE_LEGACY_ONLY` | Reglages -> Compatibilite historique | Deplace | `solver_cache_size` |
| hashes S/M/L | `MAINTENANCE_ACTION` | Menu Outils avances -> Maintenance des index historiques | Deplaces dans dialogue dedie | `dev_hash_quads_S/M/L` |
| `dev_save_btn` | persistance seule | Remplace par `settings_save_btn` et bouton du dialogue maintenance | Supprime | Plusieurs cles |

## Fichiers modifies

- `zesolver.py`
- `tests/test_gui_development_tab_audit.py`
- `tests/test_gui_development_surface_reorganized.py`
- `docs/stabilization/p3b1c_development_surface_reorganization_report_20260722.md`
- `AGENT.md` sera mis a jour uniquement apres validation.

## Comportement avant/apres

Avant:

- onglet **Developpement** visible en Expert;
- Downsample, threads, legacy blind, familles et hashes melanges dans le meme formulaire;
- deux widgets permettaient de regler les threads;
- actions hash S/M/L presentes dans une surface de reglages de run.

Apres:

- aucun onglet **Developpement/Development** n'est ajoute a la barre d'onglets;
- `Downsample` est dans l'onglet Solveur;
- `Threads` est presente une seule fois dans Performance;
- les reglages historiques sont dans `Reglages -> Compatibilite historique et diagnostic`;
- les actions hash S/M/L sont dans `Outils avances -> Maintenance des index historiques`;
- la fenetre hash ne construit ses boutons qu'a l'ouverture et ne lance aucun worker automatiquement;
- les surfaces legacy/outils avances sont masquees en Easy.

## Persistance retenue

Le bouton global **Sauvegarder** de l'onglet `Reglages` remplace `dev_save_btn` pour les champs de compatibilite.

`_read_settings_from_ui()` persiste maintenant explicitement:

- `solver_downsample`;
- `solver_workers`;
- `solver_cache_size`;
- `dev_bucket_limit_override`;
- `dev_vote_percentile`;
- `dev_bucket_cap_S/M/L`;
- `dev_detect_k_sigma`;
- `dev_detect_min_area`;
- `dev_family_auto`;
- `dev_family_selection`;
- `dev_hash_quads_S/M/L`.

La fenetre de maintenance possede aussi un bouton de sauvegarde qui appelle le meme mecanisme.

## Widgets supprimes

- `dev_tab`;
- `dev_scroll`;
- `_build_dev_tab()`;
- `dev_workers_combo`;
- `dev_save_btn`;
- les callbacks de synchronisation de `dev_workers_combo`.

## Widgets deplaces

- `downsample_spin` -> Solveur;
- `workers_spin` -> Performance;
- `cache_spin` -> Compatibilite historique;
- `dev_bucket_spin`, `dev_vote_spin`, `dev_cap_s/m/l_spin`, `dev_sigma_spin`, `dev_area_spin` -> Compatibilite historique;
- `dev_family_group`, `dev_family_auto_check`, `dev_family_box` -> Compatibilite historique;
- `dev_hash_buttons`, `dev_hash_quads_spin` -> dialogue Maintenance des index historiques.

## Callbacks, workers et imports

- `_index_worker` est conserve: il sert deja a la construction/reconstruction d'index.
- `_rebuild_hash_level()` reste le point de depart hash S/M/L et construit toujours `IndexBuilder(quads_only=True, levels=(level_key,))`.
- `_rebuild_hash_level()` relit/sauvegarde les settings courants avant de lancer le worker.
- Aucun `SolveConfig`, `ImageSolver` ou run solveur n'est construit dans la branche hash.
- L'ancien import Benchmark reste retire par P3B-1A.
- `closeEvent()` conserve l'arret de `_index_worker`, toujours necessaire pour maintenance/index.

## Tests ajoutes ou adaptes

Ajoute:

```text
tests/test_gui_development_surface_reorganized.py
```

Adapte:

```text
tests/test_gui_development_tab_audit.py
```

Couverture:

- absence onglet Developpement FR/EN;
- absence anciens attributs `dev_tab/dev_scroll/dev_workers_combo/dev_save_btn`;
- Downsample charge, sauvegarde et restaure;
- Threads unique dans Performance, Auto preserve;
- legacy visible Expert, cachee Easy;
- avertissement ZeBlind 4D visible;
- hash S/M/L absent avant ouverture du dialogue;
- ouverture dialogue sans worker;
- valeurs hash sauvegardees;
- relance de fenetre avec valeurs conservees.

## Validation automatisee

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_development_tab_audit.py tests/test_gui_development_surface_reorganized.py -q
7 passed
```

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_development_tab_audit.py tests/test_gui_* -q
62 passed, 1 skipped, 3 warnings
```

Skip:

- `tests/test_gui_controller_zn310b.py`: `ZESOLVER_ZN310B_ROOT` non defini.

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK
```

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
PASS
582 passed, 1 skipped, 9 deselected, 58 warnings
```

Skip hermetique:

- `tests/test_real_s50.py`: S50 index/frame non configure.

```text
.venv/bin/python -m pytest -q
582 passed, 10 skipped, 58 warnings
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

git diff --check
PASS
```

## Validation graphique reelle

Inspection effectuee en session graphique reelle Qt `wayland`, sans `QT_QPA_PLATFORM=offscreen`, avec settings temporaires.

Expert FR:

- onglets visibles: `Solveur`, `Base de donnees`, `Reglages`, `Performance`, `Parametres fast solver`, `Astrometry.net`;
- aucun onglet `Developpement`;
- aucun attribut `dev_tab`, `dev_scroll`, `dev_workers_combo`, `dev_save_btn`;
- Downsample visible dans Solveur;
- Threads present dans Performance;
- compatibilite historique visible dans Reglages;
- avertissement legacy visible: `Ces parametres n'affectent pas ZeBlind 4D.`;
- outil hash accessible par le menu;
- aucun worker demarre a l'ouverture du dialogue;
- tous les onglets Expert visibles ouverts successivement.

Expert EN:

- onglets visibles: `Solver`, `Database`, `Settings`, `Performance`, `Fast solver settings`, `Astrometry.net`;
- aucun onglet `Development`;
- avertissement legacy visible: `These settings do not affect ZeBlind 4D.`

Easy FR/EN:

- FR: `Solveur`, `Reglages`;
- EN: `Solver`, `Settings`;
- compatibilite historique cachee;
- menu Outils avances cache;
- fermeture propre.

Persistance graphique:

- Downsample modifie `2 -> 4`;
- Threads modifies `2 -> Auto(0)`;
- cache legacy modifie `14 -> 23`;
- sigma legacy modifie `4.2 -> 5.6`;
- hash S modifie `1111 -> 4444`;
- sauvegarde;
- fermeture/recreation de fenetre;
- valeurs relues conformes.

## Tests non executes

Smoke solve non execute: aucune ressource catalogue locale convenable n'est configuree.

Constat:

```text
ZESOLVER_* unset
ASTROMETRY_* unset
pas de dossier database exploitable
seulement deux FITS sous reports/
```

Lancer un solve sur ces FITS sans catalogues/index valides n'aurait pas prouve la non-regression runtime.

Stop puis relance non execute en graphique reel pour la meme raison catalogue.

## Warnings

Warnings deja presents:

- `multiprocessing.popen_fork`: deprecation warning dans tests multiprocessing/legacy;
- `datetime.utcnow()`: deprecation warning dans `zeblindsolver/db_convert.py`;
- `astropy.io.fits.card.VerifyWarning`: carte FITS longue tronquee dans un test.

Aucune nouvelle categorie de warning identifiee.

## Etat Git final

```text
git status --short --branch
## test...origin/test
 M AGENT.md
 M zesolver.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? docs/stabilization/p3b1c_development_surface_reorganization_report_20260722.md
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_development_surface_reorganized.py
?? tests/test_gui_development_tab_audit.py
```

## Limites

- Les anciens noms d'attributs `dev_*` restent utilises pour plusieurs widgets de compatibilite afin de conserver le mapping et les tests d'audit sans migration de format.
- Les capacites legacy restent disponibles et visibles en Expert, pas en Easy.
- La maintenance hash reste attachee au meme `_index_worker`, ce qui preserve la fermeture propre.

## Prochaine mission unique

P3B-1D: simplifier l'onglet `Reglages` en separant mieux Bibliotheque ZeSolver, Compatibilite historique et Maintenance catalogue, sans modifier la resolution.

## Gate

```text
READY_FOR_P3B1D_NEXT_GUI_SIMPLIFICATION
```
