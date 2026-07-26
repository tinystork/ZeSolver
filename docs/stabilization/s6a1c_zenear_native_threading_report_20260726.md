# S6A-1C - Qualification des pools natifs BLAS/OpenMP et de la concurrence ZeNear CPU

Date: 2026-07-26  
Depot: `/home/tristan/.openclaw/workspace/projects/ZeSolver`

## 1. Etat Git initial

Commandes executees avant les changements S6A-1C:

```bash
git status --short --branch
git diff --check
git diff
```

Etat observe:

```text
## test...origin/test
 M .gitignore
 M tests/test_near_catalog_provider_astap.py
 M zeblindsolver/astap_db_reader.py
 M zeblindsolver/metadata_solver.py
 M zeblindsolver/near_catalog_provider.py
 M zesolver/catalog_resources.py
 M zesolver/core/batch/models.py
 M zesolver/core/batch/runner.py
 M zesolver/core/pipeline.py
 M zesolver/gui_pipeline/pipeline_runner.py
 M zesolver/gui_pipeline/requests.py
 M zesolver/gui_pipeline/settings_adapter.py
 M zesolver/resource_telemetry.py
 M zesolver/settings/assembly.py
 M zewcs290/catalog290.py
?? docs/stabilization/s6a1_zenear_shared_catalog_runtime_report_20260726.md
?? docs/stabilization/s6a1b_zenear_scheduler_continuity_report_20260726.md
?? tests/test_s6a1_near_shared_catalog_runtime.py
?? tests/test_s6a1b_near_scheduler_continuity.py
?? tools/measure_s6a1_near_runtime.py
?? tools/measure_s6a1b_near_scheduling.py
```

`git diff --check` etait propre. Le diff courant a ete inspecte: il contenait les travaux S6A-1/S6A-1B existants, notamment le runtime Near partage par batch, la telemetrie scheduler, les tests associes et les outils de mesure precedents.

## 2. Version de threadpoolctl

`threadpoolctl` est disponible dans la venv, version verifiee:

```text
threadpoolctl 3.6.0
```

## 3. Commande d'installation utilisee

Commande fournie par Tristan avant mission:

```bash
.venv/bin/python -m pip install "threadpoolctl>=3.6,<4"
```

La mission n'ajoute pas `threadpoolctl` aux dependances runtime principales.

## 4. Resultats complets de threadpool_info()

Captures realisees par `tools/measure_s6a1c_native_threading.py`:

```text
cold: []
after_numpy: 1 pool OpenBLAS NumPy
after_scipy: 1 pool OpenBLAS NumPy
after_skimage: 1 pool OpenBLAS NumPy
after_zesolver_imports: 2 pools OpenBLAS, NumPy + SciPy
after_near_runtime_resource_resolution: 2 pools OpenBLAS, NumPy + SciPy
during_limit: pools limites a 1 ou 2 selon configuration
after_restore: pools restaures a 8 threads
```

## 5. Bibliotheques natives chargees

Aucun runtime OpenMP compatible `threadpoolctl` n'a ete observe. Les pools charges sont de type:

```text
user_api=blas
internal_api=openblas
threading_layer=pthreads
architecture=Haswell
```

## 6. Chemins et versions OpenBLAS

Deux bibliotheques natives OpenBLAS distinctes sont chargees apres les imports ZeSolver:

```text
NumPy:
.venv/lib/python3.13/site-packages/numpy.libs/libscipy_openblas64_-017048f4.so
version=0.3.33.112.0

SciPy:
.venv/lib/python3.13/site-packages/scipy.libs/libscipy_openblas-5f890258.so
version=0.3.31.dev
```

Conclusion: NumPy et SciPy n'utilisent pas le meme fichier natif OpenBLAS dans ce processus.

## 7. Nombre de threads natifs initial

Par defaut:

```text
OpenBLAS NumPy: 8 threads
OpenBLAS SciPy: 8 threads
```

Variables d'environnement inspectees:

```text
OMP_NUM_THREADS=None
OPENBLAS_NUM_THREADS=None
MKL_NUM_THREADS=None
NUMEXPR_NUM_THREADS=None
VECLIB_MAXIMUM_THREADS=None
```

## 8. Preuve de limitation a 1

Sous:

```python
threadpool_limits(limits=1)
```

les deux pools OpenBLAS declares par `threadpool_info()` passent a:

```text
num_threads=1
```

Le contexte est applique autour du batch Near complet, pas par image ni par worker.

## 9. Preuve de limitation a 2

Sous:

```python
threadpool_limits(limits=2)
```

les deux pools OpenBLAS declares par `threadpool_info()` passent a:

```text
num_threads=2
```

## 10. Preuve de restauration

Apres sortie normale, exception synthetique et Stop synthetique, les pools reviennent a:

```text
OpenBLAS NumPy: 8
OpenBLAS SciPy: 8
```

Les tests automatises couvrent ces trois cas.

## 11. Architecture de l'outil

Nouvel outil:

```text
tools/measure_s6a1c_native_threading.py
```

Il utilise un controleur qui lance un sous-processus Python frais par run. Le sous-processus execute le chemin produit:

```text
PipelineGuiRunner
BatchSolverPipeline
NearBatchRuntime
ExistingNearSolverPort partage
SolverPipeline worker-local
ThreadPoolExecutor reel
```

Blind est desactive, le backend Near est force CPU, et le stagger reste a 0.

Options principales:

```text
--input-dir PATH
--workers N
--native-threads default|1|2
--native-user-api all|blas|openmp
--max-files N
--repeat N
--max-loadavg1 FLOAT
--json-output PATH
--csv-output PATH
--trace-output PATH
```

## 12. Controle de charge systeme

Chaque run enregistre:

```text
loadavg 1/5/15
CPU logiques
estimation CPU physiques
memoire RSS
nombre de threads process
resource.getrusage()
```

La garde `--max-loadavg1` rejette les runs dont le loadavg 1 minute depasse le seuil.

## 13. Runs rejetes par la garde de charge

Dans la matrice 18 FITS stricte `--max-loadavg1 2.0`, les runs suivants ont ete rejetes:

```text
5 workers / native_threads=2
6 workers / default
6 workers / native_threads=1
6 workers / native_threads=2
```

Cause: `loadavg1=2.380 > 2.0`. Ces runs ne sont pas inclus dans les medianes comparatives strictes.

Une matrice complementaire 5/6 workers avec `--max-loadavg1 4.0` a ete executee pour ne pas laisser la qualification 6 workers vide.

## 14. Matrice 18 FITS

Corpus: 18 FITS issus de `/home/tristan/near_bench100_input`, meme ordre, profil S50, D50 complet, Near CPU, Blind desactive.

Resultats valides stricts:

| Configuration | Duree | Debit | RSS peak | Threads peak | Resultat |
| --- | ---: | ---: | ---: | ---: | --- |
| 4/default | 33.673 s | 32.073 img/min | 695136 KiB | 20 | 18/18 SOLVED |
| 4/native1 | 33.490 s | 32.249 img/min | 687876 KiB | 20 | 18/18 SOLVED |
| 4/native2 | 33.451 s | 32.287 img/min | 694468 KiB | 20 | 18/18 SOLVED |
| 5/default | 34.523 s | 31.283 img/min | 776272 KiB | 21 | 18/18 SOLVED |
| 5/native1 | 35.313 s | 30.583 img/min | 770688 KiB | 21 | 18/18 SOLVED |

Complement 5/6 workers avec seuil de charge elargi:

| Configuration | Duree | Debit | RSS peak | Threads peak | Resultat |
| --- | ---: | ---: | ---: | ---: | --- |
| 5/default | 35.863 s | 30.115 img/min | 753840 KiB | 21 | 18/18 SOLVED |
| 5/native1 | 36.168 s | 29.861 img/min | 850440 KiB | 21 | 18/18 SOLVED |
| 5/native2 | 35.887 s | 30.094 img/min | 792020 KiB | 21 | 18/18 SOLVED |
| 6/default | 35.385 s | 30.521 img/min | 889636 KiB | 22 | 18/18 SOLVED |
| 6/native1 | 35.718 s | 30.237 img/min | 889884 KiB | 22 | 18/18 SOLVED |
| 6/native2 | 35.797 s | 30.170 img/min | 876260 KiB | 22 | 18/18 SOLVED |

La matrice 18 FITS ne demontre pas de gain net des limites natives. Elle a servi a choisir les finalistes 100 FITS.

## 15. Repetitions valides

Matrice 18 FITS:

```text
1 repetition valide par configuration retenue
4 runs rejetes par la garde de charge stricte
```

Matrice 100 FITS:

```text
2 repetitions valides par finaliste
0 rejet
```

## 16. Metriques de threads du processus

Sur 100 FITS:

```text
4/default: threads peak 20
5/default: threads peak 21
6/default: threads peak 22
6/native1: threads peak 22
6/native2: threads peak 22
```

La limitation `threadpoolctl` modifie `num_threads` declare par OpenBLAS, mais ne reduit pas le nombre de threads OS observes. Interpretation probable: les workers natifs OpenBLAS restent alloues ou parques, tandis que la taille d'equipe active est limitee.

## 17. Context switches

L'outil conserve par run:

```text
ru_nvcsw_delta
ru_nivcsw_delta
ru_utime_delta_s
ru_stime_delta_s
```

Aucune signature claire de sursouscription productive ou destructive n'a emerge: les limites 1/2 ne reduisent pas le pic de threads et n'apportent pas un gain de debit superieur au seuil de 3%.

## 18. Temps CPU

Les deltas `resource.getrusage()` sont enregistres dans les JSON. Les donnees restent coherentes avec un batch CPU-bound, mais ne justifient pas une politique produit de limitation native dans cette mission.

## 19. Debit

Finalistes 100 FITS:

| Configuration | Mediane duree | Debit median | Ecart vs 6/default |
| --- | ---: | ---: | ---: |
| 6/default | 120.431 s | 49.823 img/min | baseline |
| 6/native1 | 120.315 s | 49.870 img/min | +0.10% |
| 6/native2 | 119.630 s | 50.156 img/min | +0.67% |
| 5/default | 120.283 s | 49.883 img/min | +0.12% |
| 4/default | 147.862 s | 40.593 img/min | -18.6% |

Les gains `native1`, `native2` et `5/default` sont inferieurs au seuil de decision de 3%.

## 20. Temps au premier resultat

Le premier resultat 100 FITS 6/default arrive autour de 15 s. Les variations entre finalistes restent secondaires par rapport au temps total; aucun reglage natif n'a montre un avantage de demarrage robuste.

## 21. Durees de detection

Exemple 100 FITS, 6 workers, native1:

```text
median_detect_duration_ms=4235.632
p95_detect_duration_ms=7984.127
```

Les durees de detection restent la zone chaude principale, mais la limitation OpenBLAS n'a pas produit de reduction significative et reproductible du temps total.

## 22. Occupation des workers

Exemple 100 FITS, 6 workers, native1:

```text
worker_threads_unique=6
solver_pipelines_unique=6
average_tasks_active=5.951
time_at_full_occupancy_s=110.197
time_at_workers_minus_one_or_more_s=111.727
median_worker_handoff_gap_ms=0.048
p95_worker_handoff_gap_ms=0.062
```

La continuite scheduler S6A-1B reste confirmee pendant S6A-1C.

## 23. Memoire RSS

Sur 100 FITS:

```text
4/default: RSS peak max 814692 KiB
5/default: RSS peak max 930120 KiB
6/default: RSS peak max 1131000 KiB
6/native1: RSS peak max 1060844 KiB
6/native2: RSS peak max 1013456 KiB
```

La limitation native reduit le RSS peak observe dans ces runs 6 workers, mais le gain memoire ne s'accompagne pas d'un gain de debit significatif. `5/default` donne un debit equivalent a 6/default sur cette machine avec moins de RSS et un thread process de moins, mais cela ne suffit pas a changer la politique Auto globale.

## 24. Selection des finalistes

Finalistes retenus apres 18 FITS:

```text
6/default: baseline produit
6/native1: limitation maximale
6/native2: limitation moderee
5/default: equivalent S6A-1B avec pression memoire moindre possible
4/default: temoin sous-parallelise
```

## 25. Matrice 100 FITS

Corpus:

```text
/home/tristan/near_bench100_input
100 FITS
profil S50
D50 complet
Near CPU
Blind desactive
processus frais par run
page cache OS potentiellement chaud
```

Deux repetitions valides par finaliste. Tous les runs ont termine:

```text
100/100 SOLVED
0 FAILED
0 CANCELLED
```

## 26. Comparaison scientifique

Les signatures resultats ont ete comparees entre la baseline 6/default et tous les finalistes:

```text
status
backend
inliers
rms_px arrondi a 1e-6
pixel_scale_arcsec arrondi a 1e-6
wcs_written
```

Resultat:

```text
0 difference
0 fichier manquant
0 fichier supplementaire
```

Les entetes WCS complets CRVAL/CD n'ont pas ete re-hashes dans l'outil S6A-1C; la parite retenue ici est la parite des sorties ZeSolver capturees par le pipeline et l'absence de regression de statut.

## 27. Resultats Stop

Test hermetique:

```text
threadpool_limits(limits=1)
PipelineGuiRunner
Stop declenche depuis result_callback
NearBatchRuntime ferme exactement une fois
resultats finaux presents
threadpool_info restaure apres sortie
```

Le test passe.

## 28. Comportement apres exception

Test hermetique:

```text
with native_thread_context("1", "all"):
    raise RuntimeError("synthetic")
```

Apres exception, `threadpool_info()` retrouve les valeurs initiales.

## 29. Meilleur reglage observe

Le meilleur temps median brut sur 100 FITS est:

```text
6 workers / native_threads=2
119.630 s
50.156 images/min
```

Mais l'ecart face a 6/default est seulement:

```text
~0.67%
```

Ce gain est inferieur au seuil minimal de 3%.

## 30. Gain median

Gains face a 6/default:

```text
6/native1: ~0.10%
6/native2: ~0.67%
5/default: ~0.12%
4/default: regression ~18.6%
```

## 31. Stabilite du gain

Les deux repetitions 100 FITS ne montrent pas de gain reproductible superieur au bruit. `native2` est legerement devant, mais trop faiblement pour justifier une integration produit.

## 32. Decision sur la sursouscription

Sursouscription possible en theorie:

```text
6 workers Python
2 pools OpenBLAS
8 threads declares par pool
```

Sursouscription non demontree comme probleme de debit dans ces mesures:

```text
pas de gain >= 3% avec limits=1
pas de gain >= 3% avec limits=2
pas de reduction du nombre de threads process observe
parite scientifique conservee
Stop conserve
```

## 33. Decision sur le nombre de workers

Ne pas modifier la politique Auto des workers dans S6A-1C.

Observation utile:

```text
5/default ~= 6/default en debit sur cette machine
5/default utilise moins de RSS peak dans ces runs
```

Ce signal merite d'etre conserve comme information de tuning, mais il reste mono-machine et sous le seuil de 3%.

## 34. Recommandation ou non d'integration produit

Recommandation:

```text
ne pas integrer threadpoolctl au chemin produit maintenant
ne pas changer le nombre de workers par defaut
conserver l'outil S6A-1C comme diagnostic reproductible
```

Une mission produit separee serait justifiee seulement si un corpus plus large ou une autre machine montre un gain stable >= 5%, ou une reduction memoire critique sans perte de debit.

## 35. Decision concernant les dependances

`threadpoolctl` reste une dependance d'outil/diagnostic dans la venv locale. Aucun ajout a `requirements.txt` ou aux dependances runtime principales.

Si une politique native devient produit plus tard, options recommandees:

```text
extra optionnel performance
ou dependance runtime principale explicite
```

## 36. Fichiers modifies

Modifications S6A-1C ciblees:

```text
.gitignore
tools/measure_s6a1c_native_threading.py
tests/test_s6a1c_native_threading.py
docs/stabilization/s6a1c_zenear_native_threading_report_20260726.md
```

Les autres fichiers modifies/non suivis visibles dans `git status` appartiennent aux etapes S6A-1/S6A-1B et au travail precedent non committe.

## 37. Barrieres executees

Commandes executees:

```bash
.venv/bin/python tools/check_core_boundaries.py
```

Resultat:

```text
core boundary check: OK
```

Tests cibles:

```bash
.venv/bin/python -m pytest -q \
 tests/test_s6a1_near_shared_catalog_runtime.py \
 tests/test_s6a1b_near_scheduler_continuity.py \
 tests/test_s6a1c_native_threading.py \
 tests/test_near_catalog_provider_astap.py \
 tests/test_batch_pipeline_scheduling.py \
 tests/test_batch_pipeline_concurrency.py \
 tests/test_batch_pipeline_cancellation.py
```

Resultat:

```text
26 passed
```

Suite hermetique:

```bash
.venv/bin/python tools/run_regression_suite.py --hermetic
```

Resultat:

```text
PASS
661 passed, 1 skipped, 9 deselected
```

Suite complete offscreen:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

Resultat:

```text
661 passed, 10 skipped
```

Compilation:

```bash
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
```

Resultat:

```text
OK
```

`git diff --check` final:

```text
OK
```

## 38. Etat Git final

Etat final verifie:

```text
## test...origin/test
 M .gitignore
 M tests/test_near_catalog_provider_astap.py
 M zeblindsolver/astap_db_reader.py
 M zeblindsolver/metadata_solver.py
 M zeblindsolver/near_catalog_provider.py
 M zesolver/catalog_resources.py
 M zesolver/core/batch/models.py
 M zesolver/core/batch/runner.py
 M zesolver/core/pipeline.py
 M zesolver/gui_pipeline/pipeline_runner.py
 M zesolver/gui_pipeline/requests.py
 M zesolver/gui_pipeline/settings_adapter.py
 M zesolver/resource_telemetry.py
 M zesolver/settings/assembly.py
 M zewcs290/catalog290.py
?? docs/stabilization/s6a1_zenear_shared_catalog_runtime_report_20260726.md
?? docs/stabilization/s6a1b_zenear_scheduler_continuity_report_20260726.md
?? docs/stabilization/s6a1c_zenear_native_threading_report_20260726.md
?? tests/test_s6a1_near_shared_catalog_runtime.py
?? tests/test_s6a1b_near_scheduler_continuity.py
?? tests/test_s6a1c_native_threading.py
?? tools/measure_s6a1_near_runtime.py
?? tools/measure_s6a1b_near_scheduling.py
?? tools/measure_s6a1c_native_threading.py
```

Aucun commit, aucun push. Les changements S6A precedents restent presents et n'ont pas ete restaures.

## 39. Gate final

Decision:

```text
S6A1C_NATIVE_THREADING_QUALIFIED
NO_NATIVE_POOL_LIMIT_REQUIRED
READY_FOR_S6A2_ZENEAR_GPU_RUNTIME
```

Raison:

```text
threadpoolctl fonctionne et restaure correctement
deux pools OpenBLAS distincts sont bien identifies
les limites 1/2 sont applicables autour du batch complet
aucun gain de debit significatif n'est demontre sur 100 FITS
la parite scientifique est conservee
Stop reste correct
aucune politique produit ne doit etre changee a ce stade
```
