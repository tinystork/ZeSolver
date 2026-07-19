# P1D-3A — Builder Blind 4D direct depuis ASTAP

## 1. Objectif

Créer un builder moteur capable de produire un payload Blind 4D directement
depuis les shards ASTAP/HNSKY, sans lire l'ancien `index_root/manifest.json`,
les `tiles/*.npz` historiques ni les `hash_tables`, et sans modifier le runtime
solveur.

## 2. État Git initial et HEAD

Commandes initiales:

```text
git status --short --branch
git rev-parse HEAD
git diff --check
```

État initial:

```text
## test...origin/test
```

avec des modifications et ajouts non commités issus de P1D-2A/P1D-2B. HEAD:

```text
de1bc8e064d813037750e11ab0bbb6ce9a3a673f
```

`git diff --check` était propre. La branche avait été précédemment annoncée
`ahead 1`; aucun commit existant n'a été réécrit, écrasé ou modifié.

`AGENT.md` indiquait bien:

- P1D-2A terminé;
- P1D-2B terminé;
- P1D-3 à préparer;
- P3B en pause.

## 3. Comportement historique caractérisé

`zeblindsolver/db_convert.py`:

- lit les shards via `iter_tiles()` et `load_tile_stars()`;
- calcule un centre TAN cartésien;
- bascule sur le centre de layout si aucune projection TAN n'est finie;
- filtre les projections non finies;
- applique `mag_cap`;
- applique `max_stars` avec `native_prefix` ou `brightest_mag`;
- écrit `tiles/*.npz` et `manifest.json`.

`zeblindsolver/quad_index_4d.py`:

- lit `manifest.json`;
- relit `tiles/*.npz`;
- trie stablement les étoiles par magnitude;
- applique `max_stars_per_tile`;
- appelle `sample_quads()`;
- appelle `build_astrometry_quad_records()`;
- écrit le payload NPZ runtime.

## 4. Architecture directe

Nouveau module:

```text
zeblindsolver/astap_4d_builder.py
```

API principale:

```python
build_4d_index_from_astap(db_root, out_path, *, config)
```

Le module ne dépend ni de `zesolver`, ni de `CatalogLibrary`, ni du GUI. Il lit
uniquement la racine ASTAP et écrit uniquement le `out_path` explicite.

## 5. Contrat de matérialisation

Ajout de:

- `Quad4DSourceTile`;
- `AstapTileMaterializationConfig`;
- `materialize_astap_tile_for_4d()`.

La matérialisation expose les identifiants de tuile, centre TAN, RA/DEC,
magnitude, coordonnées TAN, indices source équivalents à `sweep_rank`, compteurs
avant/après filtres et statut de fallback layout.

## 6. Configuration du builder

Ajout de `Astap4DBuildConfig`, dataclass immuable couvrant:

- famille;
- tuiles;
- niveau;
- `mag_cap`;
- limite source;
- politique de troncature source;
- politique de centre TAN;
- limites 4D;
- sampler;
- tolérance recommandée;
- dtype;
- projection;
- schéma et version des quads;
- version builder.

Paramètres réels utilisés pour la validation externe:

```text
family=d50
mag_cap=15.0
source_max_stars=2000
source_star_truncation_mode=native_prefix
max_stars_per_tile=2000
max_quads_per_tile=40000
sampler_tag=catalog_ring_coverage
dtype=float32
```

## 7. Coeur partagé

Ajout de `build_4d_index_from_payload_tiles()` dans `quad_index_4d.py`.

Le wrapper historique `build_experimental_4d_index()` charge maintenant les
NPZ historiques puis appelle ce coeur partagé. Le builder direct matérialise les
tuiles ASTAP puis appelle le même coeur.

`sample_quads()` et `build_astrometry_quad_records()` n'ont pas été modifiés.

## 8. Payload produit

Le payload conserve:

- `codes_4d`;
- `quad_star_indices`;
- `source_quad_indices`;
- `tile_key_indices`;
- `ratio_hashes`;
- `tile_keys`;
- `catalog_ra_dec`;
- `catalog_xy`;
- `metadata`.

La metadata directe indique `source_catalog=astap_raw` et ajoute
`source_family`, `source_tiles`, `source_fingerprint`, `build_parameters`,
`provenance_fingerprint` et `builder_version`.

## 9. Provenance et empreintes

Ajout de `scientific_payload_fingerprint()`.

L'empreinte couvre les tableaux, shapes, dtypes, bytes et métadonnées
scientifiques canoniques. Elle exclut `generated_at` et les chemins machine
locaux (`source_index_root`, `source_db_root`, chemins temporaires).

## 10. Déterminisme

Test hermétique:

```text
deux builds directs identiques -> même empreinte scientifique
```

Un changement scientifique dans la source ASTAP est détecté par changement
d'empreinte et divergence de tableaux.

## 11. Comparaison par tuile

Validation externe dans:

```text
/tmp/p1d3a_direct_astap_4d_full_20260719_071731
```

Tuiles:

```text
d50_2602
d50_2644
d50_2645
d50_2702
d50_2822
d50_2823
```

Résultat contre le chemin historique actuel reconstruit depuis
`/home/tristan/zesolver_index`:

```text
6/6 exact
```

## 12. Comparaison des tableaux

Pour les six tuiles:

```text
historical_vs_direct exact=True
```

Les tableaux suivants sont identiques:

- `codes_4d`;
- `quad_star_indices`;
- `source_quad_indices`;
- `tile_key_indices`;
- `ratio_hashes`;
- `catalog_ra_dec`;
- `catalog_xy`;
- `tile_keys`.

## 13. Divergences

Les NPZ stricts existants sous `indexes/astrometry_4d` divergent du direct.
Premières divergences:

| Tuile | Premier écart | Max abs |
|---|---:|---:|
| `d50_2602` | `codes_4d[200,0]` | `1.7236246764659882` |
| `d50_2644` | `codes_4d[0,0]` | `1.7644349038600922` |
| `d50_2645` | `codes_4d[240,0]` | `1.7168328762054443` |
| `d50_2702` | `codes_4d[160,0]` | `1.7266706824302673` |
| `d50_2822` | `codes_4d[420,0]` | `1.7120437622070312` |
| `d50_2823` | `codes_4d[580,0]` | `1.748843938112259` |

Contrôle complémentaire:

```text
existing NPZ vs rebuild depuis leur source_index_root déclaré -> divergent
```

Conclusion: les artefacts stricts actuels ont une provenance partielle/ancienne
génération non reproductible exactement avec le builder historique actuel. La
divergence est mesurée et localisée; elle ne provient pas d'un écart entre le
builder direct et le chemin historique actuel.

## 14. Paramètres historiques connus et inconnus

Connus:

- `mag_cap=15.0`;
- `max_stars=2000`;
- `star_truncation_mode=native_prefix`;
- `max_stars_per_tile=2000`;
- `max_quads_per_tile=40000`;
- `sampler_tag=catalog_ring_coverage`;
- `dtype=float32`;
- schéma `astrometry_ab_code_4d_v1`.

Inconnus ou partiels pour les NPZ stricts existants:

- version exacte de code ayant produit les NPZ;
- fingerprint source complet;
- preuve cryptographique des `tiles/*.npz` historiques au moment de génération;
- commit builder.

Statut de comparaison des artefacts existants:

```text
FUNCTIONALLY_EQUIVALENT_WITH_PARTIAL_PROVENANCE
```

à valider en runtime corpus dans P1D-3B.

## 15. Smoke runtime

Un manifeste strict temporaire a été créé:

```text
/tmp/p1d3a_direct_astap_4d_full_20260719_071731/direct_strict_4d_manifest.json
```

Résultats:

- `load_4d_index_manifest()` charge 6 entrées;
- `Quad4DIndex.load()` charge chaque NPZ direct;
- les tuiles chargées sont les six attendues;
- une recherche synthétique par code retourne des hits sur chaque index.

## 16. Intégrité externe

Ressources utilisées en lecture seule:

- `/opt/astap`;
- `/home/tristan/zesolver_index`;
- `config/zeblind_4d_experimental_manifest.json`;
- `indexes/astrometry_4d`.

Contrôle avant/après:

| Ressource | Fichiers | Taille totale | Somme mtimes |
|---|---:|---:|---:|
| `/opt/astap` | `1487` | `977092518` | `2496662849353000000000` |
| `/home/tristan/zesolver_index` | `1480` | `371695452` | `2633056069620268559568` |
| `indexes/astrometry_4d` | `6` | `6345747` | `10702894514220325179` |

SHA256 des six NPZ existants inchangés:

- `d50_2602`: `3ab3a747d2005ac6523a5dc62ef82fc19374fc31df27c8757307b36ade556693`;
- `d50_2644`: `577bb69cbe23063a718f177d58a8e1b1367f4ccc19a0c2dab09bfbad26c7c9e7`;
- `d50_2645`: `b963ebe462c98556b98520584e515b77b5ab87bd08de70de52ec9d871c91b2df`;
- `d50_2702`: `15d82411ab1213505660be448a9290b9129e2bc3243cff960dfa79b6475d0fd6`;
- `d50_2822`: `04ecc3ea867307e64cbc8a8bf00cdca847b590b822289edce991e96aa9db1967`;
- `d50_2823`: `63ede21d82d4bb885ad10b73ececff2750e49bd914c2625f321ed16a3a1529e5`.

## 17. Performances

Temps directs externes:

| Tuile | Direct ASTAP |
|---|---:|
| `d50_2602` | `45.890 s` |
| `d50_2644` | `47.250 s` |
| `d50_2645` | `47.679 s` |
| `d50_2702` | `32.532 s` |
| `d50_2822` | `32.040 s` |
| `d50_2823` | `31.727 s` |

Temps historiques `tiles/*.npz -> 4D`:

| Tuile | Historique |
|---|---:|
| `d50_2602` | `30.851 s` |
| `d50_2644` | `31.379 s` |
| `d50_2645` | `31.663 s` |
| `d50_2702` | `31.529 s` |
| `d50_2822` | `32.785 s` |
| `d50_2823` | `47.804 s` |

La performance n'a pas été optimisée en P1D-3A. Le coût dominant reste
l'échantillonnage et l'assemblage des 40k quads.

## 18. Tests ciblés

Commande adaptée:

```text
.venv/bin/python -m pytest \
 tests/test_astap_4d_tile_materialization.py \
 tests/test_astap_4d_direct_builder.py \
 tests/test_astap_4d_builder_determinism.py \
 tests/test_astap_4d_builder_parity.py \
 tests/test_astap_4d_builder_boundaries.py \
 tests/test_astap_4d_builder_cli.py \
 tests/test_quad_code_diagnostic.py \
 tests/test_p221_app_integration.py \
 tests/test_p224_budget_concurrency.py \
 tests/test_catalog_library_provenance.py \
 -q
```

Résultat:

```text
62 passed, 2 warnings
```

Les deux warnings viennent de `datetime.utcnow()` dans le convertisseur
historique, exercé par les tests de parité.

## 19. Barrières générales

Résultats:

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
513 passed, 1 skipped, 9 deselected, 58 warnings
status: PASS

.venv/bin/python -m pytest -q
513 passed, 10 skipped, 58 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

## 20. Warnings

Pas de nouvelle catégorie de warning runtime. Les warnings observés sont:

- `datetime.utcnow()` historique dans `db_convert.py`;
- warnings multiprocessing/fork déjà présents dans les tests existants;
- un `VerifyWarning` Astropy déjà présent.

## 21. Limites

P1D-3A ne bascule aucun produit vers les nouveaux index directs.

Les NPZ stricts actuels ne sont pas reproductibles exactement par le wrapper
historique actuel depuis leurs métadonnées déclarées. P1D-3B devra donc comparer
le runtime sur corpus avec un manifeste strict temporaire direct, sans utiliser
la seule égalité aux artefacts existants comme preuve.

Aucun solve corpus 30 FITS n'a été présenté comme validation P1D-3A.

## 22. État Git final

État final:

```text
## test...origin/test
 M .gitignore
 M AGENT.md
 M docs/architecture/catalog_library_implementation.md
 M docs/architecture/catalog_manifest_example.json
 M docs/architecture/catalog_manifest_schema.json
 M zeblindsolver/__init__.py
 M zeblindsolver/quad_index_4d.py
 M zesolver/catalog_library/__init__.py
 M zesolver/catalog_library/coverage.py
 M zesolver/catalog_library/manifest.py
 M zesolver/catalog_library/models.py
 M zesolver/catalog_library/validation.py
 M zesolver/catalog_resources.py
?? docs/architecture/catalog_atomic_adoption.md
?? docs/architecture/catalog_provenance_and_adoption.md
?? docs/architecture/direct_astap_blind4d_builder.md
?? docs/stabilization/p1d2a_catalog_provenance_plan_report.md
?? docs/stabilization/p1d2b_atomic_catalog_adoption_report.md
?? docs/stabilization/p1d3a_direct_astap_blind4d_builder_report.md
?? tests/test_astap_4d_builder_boundaries.py
?? tests/test_astap_4d_builder_cli.py
?? tests/test_astap_4d_builder_determinism.py
?? tests/test_astap_4d_builder_parity.py
?? tests/test_astap_4d_direct_builder.py
?? tests/test_astap_4d_tile_materialization.py
?? tests/test_catalog_library_adopted_runtime.py
?? tests/test_catalog_library_adoption_cli.py
?? tests/test_catalog_library_adoption_conflicts.py
?? tests/test_catalog_library_adoption_idempotence.py
?? tests/test_catalog_library_adoption_plan.py
?? tests/test_catalog_library_adoption_rollback.py
?? tests/test_catalog_library_atomic_writer.py
?? tests/test_catalog_library_fingerprints.py
?? tests/test_catalog_library_paths.py
?? tests/test_catalog_library_provenance.py
?? tests/test_catalog_library_repair_actions.py
?? tools/adopt_catalog_library.py
?? tools/compare_blind4d_builders.py
?? zeblindsolver/astap_4d_builder.py
?? zesolver/catalog_library/adoption.py
?? zesolver/catalog_library/atomic_adoption.py
```

Les modifications P1D-2A/P1D-2B ont été conservées. Aucun commit ni push n'a
été effectué.

## 23. Prochaine étape

Une seule prochaine étape:

```text
P1D-3B — validation runtime Blind 4D directe sur corpus avec manifeste strict temporaire.
```

## 24. Décision de gate

Critères:

- builder direct sans manifeste historique ni `tiles/*.npz`;
- sources ASTAP inchangées;
- index existants inchangés;
- payload compatible `Quad4DIndex.load()`;
- sampling et codes partagés et inchangés;
- empreinte scientifique directe déterministe;
- six tuiles construites;
- divergences mesurées et qualifiées;
- manifeste strict temporaire chargeable;
- tests et barrières verts;
- aucune ressource produit basculée.

Décision:

```text
READY_FOR_P1D3B_DIRECT_BLIND4D_RUNTIME_VALIDATION
```
