# S4 ASTAP family handling report - 2026-07-23

## 1. Objectif

Accepter proprement toute combinaison valide de familles ASTAP reellement installees, avec le cas minimal D50 seule, sans traiter les familles absentes comme des erreurs et sans modifier les algorithmes astrometriques.

## 2. Etat Git initial

```text
## test...origin/test
 M AGENT.md
 M zesolver.py
 M zesolver/catalog_library/__init__.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? docs/stabilization/p3b1c_development_surface_reorganization_report_20260722.md
?? docs/stabilization/p3b1d_catalog_library_manager_report_20260722.md
?? docs/stabilization/s3_async_wcs_cleaning_report_20260723.md
?? tests/test_catalog_library_management_service.py
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_catalog_library_manager.py
?? tests/test_gui_development_surface_reorganized.py
?? tests/test_gui_development_tab_audit.py
?? tests/test_gui_wcs_cleanup_worker.py
?? zesolver/catalog_library/management.py
?? zesolver/gui_wcs_cleanup.py
```

`git diff --check` global etait deja en echec sur `AGENT.md` lignes 3-7, espaces finaux preexistants. Les changements S3 ont ete conserves.

## 3. Comportement utilisateur reproduit

Par inspection et tests synthetiques: D50 seule etait detectee par le service, mais certains parcours ouvraient encore `CatalogDB(root)` sans liste explicite de familles. `CatalogDB` inspectait alors toutes les familles connues et journalisait `family X: no tiles matched ...` pour D05/D20/D80/V50/G05, alors que ces absences sont normales.

## 4. Parcours de detection identifie

```text
racine ASTAP
-> CatalogLibraryManagementService.detect_astap_families()
-> LibraryManagerWindow._refresh_detected_families()
-> LibraryManagerWindow._selected_families()
-> LibraryCreateOptions.families
-> CatalogLibraryManagementService._selected_families()
-> _prepare_sources()
-> _build_blind4d_indexes()
-> build_4d_index_from_astap()
-> CatalogLibraryAdoptionPlan.reference_existing()
-> CatalogLibrary.validate()
-> build_blind4d_manifest_view()
```

## 5. Cause racine

Deux causes se combinaient:

- `zeblindsolver.astap_db_reader.iter_tiles()` et `load_tile_stars()` ouvraient `CatalogDB` sans filtre de famille, ce qui declenchait des scans/logs pour toutes les familles theoriques.
- En mode reference, `CatalogLibraryAdoptionPlan.reference_existing()` rescannait les racines ASTAP sans filtre et pouvait reinclure des familles presentes mais non selectionnees dans `catalog.json`.

## 6. Fichiers modifies

- `zewcs290/catalog290.py`
- `zeblindsolver/astap_db_reader.py`
- `zesolver/catalog_library/adoption.py`
- `zesolver/catalog_library/discovery.py`
- `zesolver/catalog_library/management.py`
- `zesolver.py`
- `tests/test_catalog_library_management_service.py`
- `tests/test_catalog_library_discovery.py`
- `tests/test_gui_catalog_library_manager.py`
- `docs/stabilization/s4_astap_family_handling_report_20260723.md`

## 7. Contrat de detection

La detection est stable, canonique et insensible a la casse des noms de fichiers (`D50_*.1476` et `d50_*.1476` -> `d50`). Elle regarde la racine et ses sous-dossiers directs, ignore les fichiers sans rapport et ne dedoublonne que par famille canonique.

## 8. Contrat de selection

La selection est normalisee en minuscules, dedoublonnee et figee avant le worker. Une famille explicitement selectionnee mais absente declenche une erreur claire avant construction et avant publication.

## 9. Comportement mode standard

Le standard selectionne toutes les familles detectees, et seulement elles. D50 seule produit `families=("d50",)`, progression Blind 4D sur une famille, et manifeste avec `direct-d50` uniquement.

## 10. Comportement mode personnalise

Le personnalise respecte les cases cochees. Si `d20,d50` sont detectees mais seule `d50` est selectionnee, Near, Blind 4D et manifestes restent limites a `d50`.

## 11. Cas D50 seul

Valide par test synthetique:

```text
detection: d50
standard selection: d50
near manifest: d50
blind4d requested indexes: d50
blind4d produced indexes: direct-d50
publication: OK
```

Les logs ne contiennent plus `no tiles matched` pour les familles absentes pendant le build D50 seul.

## 12. Distinction absent / incomplet / corrompu

- Absente non selectionnee: ignoree, pas d'erreur, pas de warning utilisateur.
- Selectionnee mais absente: `ASTAP_FAMILY_MISSING` avec famille, racine inspectee et action.
- Racine vide: `ASTAP_NO_FAMILIES_DETECTED` avant construction.
- Presente mais invalide/incomplete: les validations existantes de lecture, source, integrite, coverage et build restent actives; elles n'ont pas ete affaiblies.

## 13. Comportement avant/apres

Avant: D50 seule pouvait produire des logs repetitifs sur D05/D20/D80/V50/G05 absentes, et un sous-ensemble personnalise pouvait etre reintegre par le manifeste Near en mode reference.

Apres: tous les parcours critiques recoivent la selection effective; aucune famille absente ou non selectionnee n'est ouverte, construite ou publiee.

## 14. Tests cibles

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
 tests/test_catalog_library_management_service.py \
 tests/test_gui_catalog_library_manager.py \
 tests/test_catalog_library_discovery.py \
 tests/test_catalog_library_validation.py \
 -q

37 passed
```

Couverture ajoutee: racine vide, D50 seule, plusieurs familles, casse des noms, fichiers ignores, ordre stable, racine inexistante, standard, personnalise, selection absente, publication D50 seule, manifestes D50 uniquement, progression sur une famille, GUI standard/custom/empty.

## 15. Suite hermetique

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
status: PASS
609 passed, 1 skipped, 9 deselected, 59 warnings
```

## 16. Suite complete

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
609 passed, 10 skipped, 59 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
PASS
```

Skips complets: env/corpus externes non configures (`ZESOLVER_ZN310B_ROOT`, base ASTAP externe, S50, Blind4D manifest, corpus regression).

## 17. Validation locale reelle ou instructions

Validation sur vraie racine D50 non executee: aucune tuile `d50_*.1476`/`D50_*.1476` n'a ete trouvee localement dans les chemins recherches sous `/home/tristan` a profondeur raisonnable.

Instructions:

1. ouvrir le gestionnaire de bibliotheques;
2. choisir la vraie racine ASTAP contenant D50 seule;
3. verifier `Familles detectees: D50`;
4. lancer en mode standard;
5. verifier que le journal ne liste pas D05/D20/D80/V50/G05 comme erreurs;
6. verifier que la progression Blind 4D compte une famille;
7. verifier que `catalog.json` et `strict_4d_manifest.json` ne mentionnent que `d50`;
8. selectionner la bibliotheque produite dans le GUI.

## 18. Warnings

Warnings inchanges: `datetime.utcnow()` dans `db_convert.py`/`zewcscleaner.py`, warnings multiprocessing fork multi-thread, VerifyWarning Astropy. Aucun warning S4 bloquant.

## 19. Tests non executes

Validation reelle sur base D50 complete non executee faute de racine locale accessible dans cette session.

## 20. Etat Git final

```text
## test...origin/test
 M AGENT.md
 M tests/test_catalog_library_discovery.py
 M zeblindsolver/astap_db_reader.py
 M zesolver.py
 M zesolver/catalog_library/__init__.py
 M zesolver/catalog_library/adoption.py
 M zesolver/catalog_library/discovery.py
 M zewcs290/catalog290.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? docs/stabilization/p3b1c_development_surface_reorganization_report_20260722.md
?? docs/stabilization/p3b1d_catalog_library_manager_report_20260722.md
?? docs/stabilization/s3_async_wcs_cleaning_report_20260723.md
?? tests/test_catalog_library_management_service.py
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_catalog_library_manager.py
?? tests/test_gui_development_surface_reorganized.py
?? tests/test_gui_development_tab_audit.py
?? tests/test_gui_wcs_cleanup_worker.py
?? zesolver/catalog_library/management.py
?? zesolver/gui_wcs_cleanup.py
```

`git diff --check -- <fichiers S4>` passe. `git diff --check` global echoue encore uniquement sur `AGENT.md` lignes 3-7, preexistant.

## 21. Problemes distincts decouverts

Aucune correction n'a ete faite sur `BLIND4D_LIBRARY_VIEW_INVALID`. La validation de vue Blind 4D reste hors perimetre S4 et appartient a la mission suivante.

## 22. Une seule prochaine etape

Valider puis corriger la logique de validation de la vue Blind 4D sur bibliotheque D50 complete.

## 23. Decision de gate

Les criteres automatises S4 sont satisfaits. La validation reelle D50 complete reste a faire par Tristan sur sa racine locale, mais aucun blocage code/test ne subsiste pour passer a la mission suivante.

READY_FOR_S5_BLIND4D_LIBRARY_VIEW_VALIDATION
