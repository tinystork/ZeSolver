# S5 - Blind 4D Library View Validation

## 1. Objectif

Corriger la validation de la vue Blind 4D d'une Bibliotheque ZeSolver afin qu'une couverture complete coherente (`FULL` + `all_sky=true`) soit acceptee, sans accepter les couvertures vides, partielles mensongeres ou contradictoires.

## 2. Etat Git initial

Etat initial S5:

```text
## test...origin/test
```

`git diff --check` initial: OK, aucune anomalie detectee.

Les travaux S3/S4 etaient deja conserves dans la branche au debut de S5.

## 3. Reproduction

Reproduction synthetique avant correction avec une bibliotheque D50 minimale equivalente:

```text
report.status= READY_FULL
capabilities.blind4d= True
capabilities.all_sky_blind4d= True
coverage.status= FULL
coverage.all_sky= True
covered/total= 1 1
view.errors= ['BLIND4D_VIEW_COVERAGE_INCONSISTENT']
```

La reproduction demontre que la vue rejetait une couverture complete coherente.

## 4. Parcours de donnees

Parcours trace:

```text
strict_4d_manifest.json / catalog.json
-> CatalogLibrary.open()
-> CatalogLibrary.validate()
-> build_blind4d_manifest_view()
-> _ordered_indexes()
-> _entry_from_index()
-> _view_coverage()
-> CatalogLibraryManagementService._validate_result()
-> resolve_catalog_resources()
-> resolve_blind4d_runtime(mode=auto/library-view)
```

## 5. Cause racine

Dans `zesolver/catalog_library/blind4d_view.py`, `_build_view()` executait:

```python
if coverage.all_sky or coverage.status is CoverageStatus.FULL:
    errors.append(BLIND4D_VIEW_COVERAGE_INCONSISTENT)
```

La condition etait inversee: elle traitait comme incoherente la situation valide attendue pour une couverture complete.

## 6. Contrat de couverture

Objets concernes:

- `CatalogCoverage.status`: statut calcule/declare de couverture (`FULL`, `PARTIAL`, `MISSING`, `UNKNOWN`, `CORRUPT`, `INCOMPATIBLE`).
- `CatalogCoverage.all_sky`: invariant explicite indiquant une couverture globale.
- `tile_keys`, `covered_tiles`, `total_tiles`, `fraction`: support compact de la couverture effective.

Contrat retenu:

- `status == FULL` implique `all_sky == true`.
- `all_sky == true` implique `status == FULL`.
- une couverture `FULL` ou `PARTIAL` doit couvrir au moins une tuile.
- si `total_tiles` est connu, `FULL` exige `covered_tiles >= total_tiles`.
- une couverture partielle coherente reste partielle et ne devient pas `READY_FULL`.
- les index eux-memes restent soumis aux validations de chemin, checksum, schema, source tiles et runtime order.

## 7. Table de verite

| Cas | Attendu | Resultat S5 |
| --- | --- | --- |
| `FULL`, `all_sky=true`, index valide | Vue valide, `READY_FULL` | Accepte |
| `PARTIAL`, `all_sky=false`, tuiles non vides | Vue valide, `READY_PARTIAL` | Accepte |
| `FULL`, `all_sky=false` | `BLIND4D_VIEW_COVERAGE_INCONSISTENT` | Rejete |
| `PARTIAL`, `all_sky=true` | `BLIND4D_VIEW_COVERAGE_INCONSISTENT` | Rejete |
| `PARTIAL`, aucune tuile couverte | Vue invalide | Rejete |
| `FULL`, compteur partiel connu | Vue invalide | Rejete |
| index manquant/checksum/order/duplication | Erreurs specifiques existantes | Preserve |

## 8. Fichiers modifies

- `zesolver/catalog_library/blind4d_view.py`
- `tests/test_catalog_blind4d_manifest_view.py`
- `docs/stabilization/s5_blind4d_library_view_validation_report_20260723.md`

## 9. Correction appliquee

La condition inversee a ete remplacee par `_coverage_inconsistency_reasons()`, appelee:

- sur chaque index ordonne;
- sur la couverture fusionnee de la vue.

La telemetrie de vue expose maintenant:

```text
entry_count
coverage_status
covered_tiles
total_tiles
all_sky
validation
error_codes
```

## 10. Validations non affaiblies

La correction ne modifie pas:

- les algorithmes ZeNear ou ZeBlind 4D;
- le format strict du manifest 4D;
- le chargement `Quad4DIndex`;
- les checks de checksum;
- le runtime order;
- le rejet des index manquants/corrompus;
- le rejet des tuiles dupliquees;
- la validation generale `CatalogLibrary.validate()`.

Elle ajoute au contraire des rejets explicites pour couverture vide ou compteur `FULL` incomplet.

## 11. Avant / apres

Avant:

```text
FULL + all_sky=true -> BLIND4D_VIEW_COVERAGE_INCONSISTENT
```

Apres:

```text
FULL + all_sky=true -> vue valide
PARTIAL + all_sky=false -> vue valide partielle
contradictions -> BLIND4D_VIEW_COVERAGE_INCONSISTENT
```

## 12. Statut des bibliotheques completes

Validation reelle sur `/home/tristan/ZeSolverCatalog/new`:

```text
status= READY_FULL
issues= []
sources= d50, 1476 tuiles, /opt/astap
derived_indexes= direct-d50, 1476 source_tiles, FULL, all_sky=True
view_errors= []
view_coverage= FULL True 1476 1476 1.0
runtime_available= True
runtime_mode= library-view
runtime_indexes= direct-d50
```

## 13. Statut des bibliotheques partielles

Les tests existants et S5 confirment que les vues partielles coherentes restent `READY_PARTIAL`, avec `all_sky=false`, sans promotion artificielle en `READY_FULL`.

## 14. Cas incoherents rejetes

Tests ajoutes:

- `FULL` sans `all_sky`;
- `PARTIAL` avec `all_sky`;
- `FULL` avec `covered_tiles < total_tiles`;
- `PARTIAL` sans tuile couverte.

## 15. Publication atomique

La bibliotheque finale reelle est relisible depuis sa destination publiee:

```text
/home/tristan/ZeSolverCatalog/new/catalog.json
/home/tristan/ZeSolverCatalog/new/indexes/blind4d/d50_4d.npz
/home/tristan/ZeSolverCatalog/new/indexes/blind4d/strict_4d_manifest.json
```

Le controle JSON ne montre aucune reference a un staging temporaire.

## 16. Runtime `library-view`

`resolve_catalog_resources()` puis `resolve_blind4d_runtime(mode="auto")` retournent:

```text
available=True
source=catalog_library_view
effective_mode=library-view
index_count=1
index_ids=('direct-d50',)
error_code=None
```

## 17. Tests cibles

Commandes executees:

```text
.venv/bin/python -m pytest tests/test_catalog_blind4d_manifest_view.py -q
17 passed

.venv/bin/python -m pytest \
 tests/test_catalog_library_blind4d_integration.py \
 tests/test_catalog_library_validation.py \
 tests/test_catalog_library_status.py \
 tests/test_catalog_resource_resolution.py \
 tests/test_catalog_library_pipeline_integration.py \
 tests/test_catalog_library_management_service.py \
 tests/test_catalog_blind4d_manifest_view.py \
 tests/test_blind4d_runtime_source_policy.py \
 -q
69 passed
```

## 18. Suite hermetique

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
PASS
615 passed, 1 skipped, 9 deselected, 59 warnings
```

Skip hermetique: `tests/test_real_s50.py`, index/frame S50 non configure.

## 19. Suite complete

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
615 passed, 10 skipped, 59 warnings
```

Skips explicites: corpus/env externes non configures (`ZESOLVER_ZN310B_ROOT`, `ZESOLVER_CORPUS_ROOT`, `ZESOLVER_BLIND4D_MANIFEST`, base ASTAP de test historique absente, S50 non configure).

## 20. Validation reelle D50

Racine ASTAP reelle:

```text
/opt/astap
1476 fichiers d50_*.1476
familles detectees: d50 uniquement
```

Bibliotheque reelle:

```text
/home/tristan/ZeSolverCatalog/new
status READY_FULL
direct-d50
1476 tuiles
coverage FULL
all_sky true
runtime library-view disponible
```

La reconstruction complete n'a pas ete relancee: les artefacts D50 reels existaient deja, leur integrite et leur vue finale ont ete revalides sans edition manuelle.

## 21. Smoke test solve

ZeNear reel sur copie temporaire:

```text
FITS: /home/tristan/near_bench100_input/069_Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit
near_provider_kind=astap_native
success=true
message=near solution found
```

Smoke Blind 4D reel sur copie temporaire:

```text
blind4d_catalog_source=catalog_library_view
blind4d_catalog_mode_effective=library-view
blind4d_index_count=1
blind4d_index_ids=[direct-d50]
blind4d_external_fallback_used=false
status=UNSOLVED
message=astrometry 4D runtime failed: no validated hypothesis
```

Le smoke Blind 4D confirme l'utilisation effective de la bibliotheque publiee. L'echec de resolution de cette image dans le budget produit est distinct de S5 et ne touche pas la validation de vue.

## 22. Telemetrie

La vue reelle produit:

```text
entry_count=1
coverage_status=FULL
covered_tiles=1476
total_tiles=1476
all_sky=True
validation=valid
error_codes=()
```

## 23. Warnings

Warnings de tests observes:

- `datetime.utcnow()` deprecation dans `zeblindsolver/db_convert.py` et `zewcscleaner.py`;
- warning multiprocessing `fork()` dans des tests legacy;
- warning Astropy FITS card truncation.

Aucun warning n'est introduit par S5.

## 24. Tests non executes

Aucun test requis n'a ete saute volontairement.

Les skips de la suite complete dependent de variables d'environnement/corpus externes non configures et sont rapportes par pytest.

## 25. Etat Git final

Etat apres implementation et validation:

```text
## test...origin/test
 M tests/test_catalog_blind4d_manifest_view.py
 M zesolver/catalog_library/blind4d_view.py
?? docs/stabilization/s5_blind4d_library_view_validation_report_20260723.md
```

Checks finaux:

```text
git diff --check -- zesolver/catalog_library/blind4d_view.py tests/test_catalog_blind4d_manifest_view.py
OK

git diff --check
OK
```

## 26. Limites restantes

Le smoke Blind 4D reel charge correctement `library-view` mais ne valide pas d'hypothese sur l'image M106 choisie dans le budget produit. C'est un comportement solveur/runtime a investiguer hors S5 si Tristan veut qualifier ce cas precis.

## 27. Prochaine etape

Passer a S6: stabilisation runtime batch et memoire, en conservant la bibliotheque D50 `READY_FULL` comme reference de validation.

## 28. Decision de gate

```text
READY_FOR_S6_BATCH_RUNTIME_AND_MEMORY_STABILIZATION
```
