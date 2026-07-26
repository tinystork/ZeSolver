# S6A-0/S6A-1 - ZeNear shared catalog runtime report - 2026-07-26

## 1. Etat Git initial

Avant modification:

```text
git status --short --branch
## test...origin/test

git diff --check
<aucune sortie>
```

Aucun commit, aucun push. D50, ZeBlind 4D, shards fixed32, WCS et seuils scientifiques non modifies.

## 2. Fichiers et architecture inspectes

Fichiers inspectes:

```text
zesolver/gui_pipeline/pipeline_runner.py
zesolver/core/batch/runner.py
zesolver/core/pipeline.py
zesolver/catalog_resources.py
zesolver/resource_telemetry.py
zeblindsolver/near_catalog_provider.py
zeblindsolver/astap_db_reader.py
zeblindsolver/metadata_solver.py
zewcs290/catalog290.py
tests/test_s5c_batch_resource_lifecycle.py
tests/test_near_catalog_provider_astap.py
```

## 3. Cycle de vie Near avant modification

Chemin reel confirme:

```text
PipelineGuiRunner
 -> BatchSolverPipeline
 -> ThreadPool near
 -> SolverPipeline local au thread
 -> ExistingNearSolverPort
 -> ExistingNearSolverPort.solve(image)
 -> resolve_near_catalog_runtime()
 -> build_near_catalog_provider()
 -> AstapNearCatalogProvider()
 -> iter_astap_tiles()
 -> CatalogDB via astap_db_reader._TileCache
 -> provider cache payload local
```

Le pipeline etait bien worker-local, mais le port Near resolvait le runtime catalogue a chaque image.

## 4. Cause exacte des constructions multiples

`ExistingNearSolverPort.solve()` appelait `resolve_near_catalog_runtime()` pour chaque `SolveRequest`. Donc, meme avec un `SolverPipeline` conserve par worker, chaque image reconstruisait au moins:

```text
NearCatalogRuntime
AstapNearCatalogProvider
tuple d'inventaire NearCatalogTile
cache payload provider
```

`CatalogDB` etait parfois recupere depuis le cache global de `astap_db_reader`, donc le log "catalog ready" ne signifiait pas toujours une relecture physique complete disque. Mais l'inventaire provider et le cache provider etaient bien recrees.

## 5. Relog, reconstruction et lecture disque

Avant:

```text
relog: messages preparation/catalog ready repetes via provider/runtime
reconstruction: runtime/provider/inventaire/cache par image
lecture disque: partiellement amortie par CatalogDB global et page cache OS
payload provider: copies sur put et copies sur hit
```

Apres:

```text
NearBatchRuntime par batch
NearCatalogRuntime cree une seule fois au premier acquire
AstapNearCatalogProvider cree une seule fois
CatalogDB possede par le provider batch
inventaire tuple partage
payload cache partage borne
payloads retournes en tableaux read-only
```

## 6. Identites Python relevees

Test rouge initial avant correction:

```text
6 images / 6 workers
provider ids observes = 6 ids distincts
assertion len(set(provider_ids)) == 1 echouee
```

Mesure apres correction sur 30 FITS:

```text
runtime_id unique: 139799349496176
provider_id unique: 139799379470592
inventory_id unique: 1004804912
payload_cache_id unique: 139799379472608
workers: 6
solver_pipeline_constructor_count: 6
near_port_constructor_count: 1
```

## 7. Allocations dupliquees

Supprime ou bornee:

```text
provider ASTAP par image -> 1 par batch
inventaire 1476 tiles par image -> 1 tuple par batch
cache payload par image -> 1 cache partage par batch
copies payload sur cache hit -> remplacees par payloads immutable shared
```

Restent locales par image:

```text
image FITS
detection etoiles image
fenetre RA/DEC
liste locale de tuiles candidates
projections locales
hypotheses/RANSAC/WCS/resultat
```

## 8. Comportement initial des caches

`zeblindsolver.astap_db_reader._TileCache` gardait des `CatalogDB` par `(root, families)`, sans verrou. `_MemoryStarCache` n'etait pas thread-safe et retournait des copies. Les misses concurrents sur la meme tuile pouvaient declencher plusieurs chargements applicatifs du meme payload.

## 9. Architecture retenue

Architecture cible implementee:

```text
PipelineGuiRunner
 -> NearBatchRuntime
    -> NearCatalogRuntime
    -> AstapNearCatalogProvider
       -> CatalogDB possede par le provider
       -> inventaire immutable tuple
       -> _MemoryStarCache partage, borne, thread-safe
 -> ExistingNearSolverPort(shared NearBatchRuntime)
 -> SolverPipeline worker-local
```

`NearBatchRuntime` est paresseux: l'objet appartient au batch des le lancement, mais la resolution lourde n'arrive qu'au premier `acquire()` du port Near reel. Cela preserve les tests et chemins qui monkeypatchent le port Near sans utiliser le runtime.

## 10. Proprietaire du runtime

Le proprietaire est le `PipelineGuiRunner` du run courant. Il cree `NearBatchRuntime`, injecte un `ExistingNearSolverPort(shared_runtime)` dans les pipelines worker-local, puis ferme le runtime avant de retourner le resume GUI. Le `finally` conserve une fermeture de secours.

## 11. Methode d'injection

`ExistingNearSolverPort` accepte maintenant un runtime optionnel:

```text
None -> comportement legacy: resolve_near_catalog_runtime() dans solve()
NearBatchRuntime -> acquire() puis reuse du runtime batch
NearCatalogRuntime -> injection directe possible en test
```

Les signatures existantes restent compatibles.

## 12. Separation etat global / etat image

Partage:

```text
chemin catalogue
familles selectionnees
inventaire tuiles
geometrie tuiles
CatalogDB provider-owned
cache payload immutable
```

Local:

```text
pixels FITS
etoiles detectees
selection locale
tableaux projetes
tri candidates
RANSAC
WCS
resultat
```

## 13. Thread safety

`_MemoryStarCache` utilise un verrou court pour l'OrderedDict/LRU et un single-flight par cle. Le chargement disque n'est pas couvert par un verrou global: seuls les autres threads demandant la meme tuile attendent cette cle. Les tuiles differentes restent paralleles.

`CatalogDB` reste protege par ses verrous internes; son cache interne est vide a la fermeture du provider.

## 14. Immutabilite des tableaux

Les payloads stockes et retournes par le cache sont copies une fois puis marques:

```python
array.flags.writeable = False
```

Un test verifie qu'une mutation leve `ValueError`. `near_solve` consomme ces tableaux en lecture seule et construit ses propres tableaux locaux pour les projections.

## 15. Politique du cache

Cache provider:

```text
capacite: near_tile_cache_size, defaut 128
politique: LRU deterministe
hits/misses/evictions instrumentes
close(): clear du cache provider et du cache CatalogDB
```

Mesure reelle 30 FITS:

```text
payload_cache_hits=108
payload_cache_misses=12
payload_cache_evictions=0
payload_physical_loads=12
payload_duplicate_loads=19
```

`physical_loads` signifie "chargements applicatifs au niveau du provider", pas garantie de lecture physique disque hors page cache OS.

## 16. Misses concurrents

Test hermetique: 4 threads demandent la meme tuile simultanement.

Resultat:

```text
load_count=1
payload_cache_misses=1
payload_physical_loads=1
payload_duplicate_loads>=1
arrays partagees et read-only
```

## 17. Fermeture et annulation

Fermeture testee:

```text
run normal -> near_catalog_runtime_closed=1
exception worker -> near_catalog_runtime_closed=1
deux runs successifs -> fermeture a chaque run
```

La fermeture intervient apres fin de l'executor batch, donc aucun worker n'utilise encore le provider.

## 18. Deuxieme batch

Deux batches successifs dans le meme processus creent deux `NearBatchRuntime` distincts. Pas de singleton application implicite pour le chemin produit. Les objets runtime/provider/cache du premier batch sont fermes avant le second.

## 19. Changement de bibliotheque

Test hermetique avec bibliotheque A puis B:

```text
provider A db_root = library A / sources / astap / d50
provider B db_root = library B / sources / astap / d50
provider ids distincts tant que gardes vivants
near_runtime_resolution_count=1 par batch
```

La cle d'identite `NearBatchRuntime.identity` contient:

```text
chemin canonique bibliotheque
source resources
familles near
mode near effectif demande
legacy_index_root
blind_only
```

## 20. Telemetrie

Compteurs ajoutes:

```text
near_catalog_runtime_created
near_catalog_runtime_reused
near_catalog_runtime_closed
near_catalog_inventory_load_count
near_catalog_provider_created
near_catalog_provider_reused
near_catalog_db_created
near_catalog_db_reused
near_catalog_payload_cache_hits
near_catalog_payload_cache_misses
near_catalog_payload_cache_evictions
near_catalog_payload_physical_loads
near_catalog_payload_duplicate_loads
```

Evenements bornes:

```text
near_batch_runtime_created
near_batch_runtime_ready
near_catalog_runtime_created
near_catalog_runtime_reused
near_catalog_runtime_closed
```

## 21. Tests ajoutes ou modifies

Ajoute:

```text
tests/test_s6a1_near_shared_catalog_runtime.py
```

Couvre:

```text
runtime/provider partages par batch
resolve_near_catalog_runtime non rappele par image
cache single-flight
cache hits/evictions
payloads read-only
fermeture normale
fermeture apres exception worker
deux batches meme processus
changement de bibliotheque
```

Modifie:

```text
tests/test_near_catalog_provider_astap.py
```

Le contrat cache devient "payload immutable shared", pas "copie mutable par hit".

## 22. Parite scientifique

Run reel Near-only mesure sur `/home/tristan/near_auto100_input`, 30 FITS, S50/catalogue production, 6 workers, Blind neutralise uniquement pour garder la mesure focalisee:

```text
30/30 SOLVED via ZeNear
premier resultat: 11.67 s
total: 68.10 s
```

Aucun seuil, parametre scientifique, WCS ou algorithme Near/Blind n'a ete modifie.

## 23. Mesures avant/apres

Avant correction, test rouge:

```text
6 images / 6 workers
provider ids distincts: 6
cause: runtime/provider resolus dans ExistingNearSolverPort.solve(image)
```

Apres correction, mesure cycle de vie pur:

```text
commande: .venv/bin/python tools/measure_s6a1_near_runtime.py /home/tristan/near_auto100_input --catalog-library /home/tristan/ZeSolverCatalog/new --workers 6 --max-files 30 --stub-near
duration=2.302 s
first_result=1.000 s
near_runtime_resolution_count=1
near_catalog_runtime_created=1
near_catalog_provider_created=1
near_catalog_db_created=1
near_catalog_inventory_load_count=1
solver_pipeline_constructor_count=6
worker_thread_count=6
```

Apres correction, mesure Near reelle:

```text
commande: .venv/bin/python tools/measure_s6a1_near_runtime.py /home/tristan/near_auto100_input --catalog-library /home/tristan/ZeSolverCatalog/new --workers 6 --max-files 30 --stub-blind
30/30 SOLVED
duration=68.095 s
first_result=11.670 s
near_runtime_resolution_count=1
near_catalog_runtime_created=1
near_catalog_provider_created=1
near_catalog_db_created=1
near_catalog_inventory_load_count=1
payload_physical_loads=12
payload_cache_hits=108
payload_cache_misses=12
payload_cache_evictions=0
```

## 24. Memoire RSS

Mesure Near reelle 30 FITS:

```text
RSS batch_start=281720 KiB
RSS before_near=281720 KiB
RSS after_preflight=281716 KiB
RSS after_near=704420 KiB
RSS batch_end=704420 KiB
RSS after_diagnostic_gc=704420 KiB
```

Le RSS inclut le solve Near reel et les allocations NumPy/astropy; le cache provider a charge 12 payloads applicatifs. La mission ne conclut pas a une reduction OS page-cache, seulement a la suppression des constructions provider/runtime/inventory par image.

## 25. Limites restantes

```text
pas de travail GPU dans S6A-1
pas de run 100/200 FITS complet dans cette mission
le helper astap_db_reader conserve son cache global pour compatibilite hors chemin produit
la mesure de lecture disque reste applicative, pas kernel/page-cache
les messages de phase "Preparation de ZeNear" peuvent encore etre emis par worker, mais le gros catalogue n'est plus reconstruit par worker/image
```

## 26. Fichiers modifies

```text
.gitignore
zeblindsolver/astap_db_reader.py
zeblindsolver/near_catalog_provider.py
zewcs290/catalog290.py
zesolver/catalog_resources.py
zesolver/core/pipeline.py
zesolver/gui_pipeline/pipeline_runner.py
zesolver/resource_telemetry.py
tests/test_near_catalog_provider_astap.py
tests/test_s6a1_near_shared_catalog_runtime.py
tools/measure_s6a1_near_runtime.py
docs/stabilization/s6a1_zenear_shared_catalog_runtime_report_20260726.md
```

## 27. Barrieres executees

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python -m pytest -q tests/test_s5c_batch_resource_lifecycle.py tests/test_s6a1_near_shared_catalog_runtime.py tests/test_near_catalog_provider_astap.py tests/test_solver_pipeline_near_provider.py tests/test_near_catalog_runtime_policy.py tests/test_near_catalog_provider_boundaries.py
25 passed

.venv/bin/python -m pytest -q tests/test_s6a1_near_shared_catalog_runtime.py tests/test_near_catalog_provider_astap.py tests/test_solver_pipeline_near_provider.py tests/test_near_catalog_runtime_policy.py tests/test_near_catalog_provider_boundaries.py tests/test_s5f_blind_progressive_results.py
28 passed

.venv/bin/python tools/run_regression_suite.py --hermetic
PASS, 649 passed, 1 skipped, 9 deselected

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
649 passed, 10 skipped

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

## 28. Etat Git final

```text
## test...origin/test
 M .gitignore
 M tests/test_near_catalog_provider_astap.py
 M zeblindsolver/astap_db_reader.py
 M zeblindsolver/near_catalog_provider.py
 M zesolver/catalog_resources.py
 M zesolver/core/pipeline.py
 M zesolver/gui_pipeline/pipeline_runner.py
 M zesolver/resource_telemetry.py
 M zewcs290/catalog290.py
?? tests/test_s6a1_near_shared_catalog_runtime.py
?? tools/measure_s6a1_near_runtime.py
?? docs/stabilization/s6a1_zenear_shared_catalog_runtime_report_20260726.md
```

## 29. Decision de gate

Le runtime catalogue Near est mutualise par batch, observable, ferme proprement, cache payload partage/thread-safe, sans verrou global sur le solve et sans regression des barrieres.

```text
READY_FOR_S6A2_ZENEAR_GPU_RUNTIME
```
