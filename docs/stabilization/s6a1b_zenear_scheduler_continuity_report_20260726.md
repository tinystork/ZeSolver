# S6A-1B - ZeNear CPU scheduler continuity

Date: 2026-07-26

Gate:

```text
S6A1B_SCHEDULER_CONTINUITY_CONFIRMED
READY_FOR_S6A2_ZENEAR_GPU_RUNTIME
```

## 1. Etat Git initial

Commandes demandees executees avant modification:

```text
git status --short --branch
git diff --check
git diff
```

Etat initial observe:

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
?? docs/stabilization/s6a1_zenear_shared_catalog_runtime_report_20260726.md
?? tests/test_s6a1_near_shared_catalog_runtime.py
?? tools/measure_s6a1_near_runtime.py
```

`git diff --check` etait propre. Aucun commit, aucun push.

## 2. Fichiers inspectes

Inspectes pour le scheduler et le chemin Near:

- `zesolver/core/batch/runner.py`
- `zesolver/core/pipeline.py`
- `zesolver/gui_pipeline/pipeline_runner.py`
- `zesolver/resource_telemetry.py`
- `zeblindsolver/metadata_solver.py`
- `zeblindsolver/star_detect.py`
- `zeblindsolver/near_catalog_provider.py`
- `zesolver/settings/assembly.py`

## 3. Architecture reelle du scheduler

La phase Near utilise un `ThreadPoolExecutor(max_workers=workers)` unique par phase. Les pipelines sont worker-local via `threading.local()`. Le port Near et le `NearBatchRuntime` sont partages par le batch depuis S6A-1.

## 4. Politique de soumission des taches

Le code courant soumet toutes les taches des le depart:

```python
future_map[pool.submit(_task, item)] = idx
for future in concurrent.futures.as_completed(future_map):
    ...
```

Il n'y a pas de soumission par blocs de six.

## 5. Preuve absence de traitement par blocs

Test hermetique ajoute:

```text
tests/test_s6a1b_near_scheduler_continuity.py
```

Le test heterogene prouve que le thread terminant la tache courte demarre une tache suivante avant la fin des cinq taches plus lentes de la premiere vague.

Observation reelle 18 FITS, 6 workers: le log alterne bien des `succeeded` et de nouveaux `starting` pendant la phase Near. Les six premieres images demarrent ensemble, puis les workers reprennent la file.

## 6. Persistance des threads

Mesure reelle 18 FITS, 6 workers:

```text
worker_threads_unique=6
tasks_per_thread=3 par thread
max_tasks_active=6
average_tasks_active=5.913
```

Les threads persistent pendant toute la phase.

## 7. Persistance des pipelines

Mesure reelle 18 FITS, 6 workers:

```text
solver_pipelines_unique=6
near_port_ids_unique=1
near_batch_runtime_ids_unique=1
runtime_ids_unique=1
provider_ids_unique=1
```

Un pipeline est cree par thread, pas par image.

## 8. Role du port Near partage

`ExistingNearSolverPort` reste unique dans le chemin GUI produit. Il acquiert le `NearBatchRuntime` a chaque image, mais l'acquisition ne couvre pas le solve: elle ne fait que verifier l'etat ferme, resoudre le runtime au premier appel, puis retourner le runtime deja pret.

## 9. Instrumentation ajoutee

La telemetry batch enregistre maintenant, par phase:

```text
future_submitted_at
task_started_at
near_solve_started_at
near_detect_started_at
near_detect_finished_at
near_solve_finished_at
task_finished_at
result_emitted_at
thread_ident
thread_name
pipeline_id
near_port_id
near_batch_runtime_id
runtime_id
provider_id
```

Les details sont stockes dans `telemetry["scheduler"]["near"]`; le journal principal ne recoit pas une ligne par image.

## 10. Definition worker_handoff_gap_ms

`worker_handoff_gap_ms` est mesure comme le temps entre la fin d'une tache par un thread et le debut de la tache suivante par le meme thread, uniquement lorsqu'il restait des taches non commencees dans la file du pool.

## 11. Metriques extraites du log utilisateur

Le log utilisateur 200 fichiers montrait des groupes visuels de resultats. Le code inspecte et les mesures S6A-1B montrent que ces groupes ne viennent pas d'une barriere explicite, mais de durees de detection/solve proches pour les images d'une meme vague.

## 12. Correction des evenements de telemetry

`near_catalog_runtime_reused` reste un compteur, mais n'est plus enregistre comme evenement complet a chaque image. Les evenements prioritaires restent observables meme si la liste bornee atteint 128 entrees:

```text
near_batch_runtime_created
near_catalog_runtime_created
near_batch_runtime_ready
near_catalog_runtime_closed
batch_complete
```

Test ajoute avec 140 acquisitions: `near_catalog_runtime_closed` reste visible et `near_catalog_runtime_reused` ne remplit plus la trace.

## 13. Clarification du compteur single-flight

Le compteur historique:

```text
near_catalog_payload_duplicate_loads
```

est conserve pour compatibilite, mais un alias explicite a ete ajoute:

```text
near_catalog_payload_singleflight_waiters
```

Il designe les threads qui attendent un chargement single-flight deja engage. Ce ne sont pas des lectures physiques multiples.

Definitions retenues:

- cache miss logique: premiere demande absente du cache;
- chargement applicatif reel: appel du loader par le thread proprietaire;
- single-flight waiter: thread coalesce sur une cle en cours de chargement;
- lecture physique disque: non mesuree directement, car le page cache OS peut servir les donnees.

## 14. Logs "Preparation de ZeNear"

`PipelineGuiRunner` n'emet plus `Preparation de ZeNear` une fois par pipeline worker-local. Le message est emis une seule fois au niveau de la phase Near du batch.

## 15. Machine de benchmark

Machine observee:

```text
Intel Core i7-8550U
4 coeurs physiques / 8 logiques
```

La charge systeme etait tres variable pendant les mesures:

```text
loadavg smoke: ~73 / 79 / 69
loadavg 6w initial: ~58 / 69 / 68
loadavg 4w: ~25 / 52 / 62
loadavg 5w: ~11 / 43 / 58
loadavg 6w comparable: ~6 / 35 / 54
```

Les comparaisons de debit sont donc indicatives, pas definitives.

## 16. CPU physique et logique

```text
CPU(s)=8
Thread(s) per core=2
Core(s) per socket=4
Socket(s)=1
```

## 17. Environnement BLAS/OpenMP

Variables observees:

```text
OMP_NUM_THREADS=<unset>
OPENBLAS_NUM_THREADS=<unset>
MKL_NUM_THREADS=<unset>
NUMEXPR_NUM_THREADS=<unset>
VECLIB_MAXIMUM_THREADS=<unset>
```

`threadpoolctl` n'est pas installe dans la venv, donc la variante F "pools numeriques limites a 1" n'a pas pu etre qualifiee proprement sans ajouter de dependance.

`numpy.show_config()` et `scipy.show_config()` indiquent OpenBLAS:

```text
NumPy: scipy-openblas 0.3.33, MAX_THREADS=64
SciPy: scipy-openblas 0.3.31, MAX_THREADS=64
```

Risque de sursouscription possible, non demontre dans cette mission.

## 18. Corpus reel utilise

Corpus trouve:

```text
/home/tristan/near_bench100_input
```

Il contient 100 FITS de plusieurs cibles: M106, M31, NGC6888, NGC3628.

Le smoke test 3 FITS a pris ~52 s sous tres forte charge. La matrice complete 100 fichiers x variantes aurait dure plusieurs heures dans l'etat machine observe. La matrice reelle executee a donc utilise les 18 premiers FITS du corpus, avec outil et sortie JSON conserves dans `reports/`.

## 19. Conditions cold/warm

Les runs ont ete faits dans des processus frais separes, mais la page cache OS et la charge systeme ont evolue fortement. Les resultats ne doivent pas etre traites comme une etude thermique stable.

## 20. Matrice 4/5/6 workers

Sous-corpus: 18 FITS, backend Near CPU, Blind desactive.

```text
4 workers, stagger 0: duration=87.04s, throughput=12.41 img/min, first_result=28.67s
5 workers, stagger 0: duration=40.08s, throughput=26.95 img/min, first_result=9.37s
6 workers, stagger 0 initial: duration=161.22s, throughput=6.70 img/min, loadavg tres eleve
6 workers, stagger 0 comparable: duration=40.07s, throughput=26.96 img/min, first_result=11.86s
```

La mesure 6w initiale est conservee comme preuve de l'impact de charge exterieure, pas comme comparaison workers.

## 21. Variantes stagger

```text
6 workers, stagger 100 ms: duration=38.31s, throughput=28.19 img/min
6 workers, stagger 200 ms: duration=37.77s, throughput=28.60 img/min
```

Le gain apparent vs 6w/0 comparable est ~4.4% puis ~5.7%, mais il est confondu avec une baisse continue de charge systeme. Decision: ne pas integrer le stagger dans le produit. L'option reste experimentale dans l'outil et dans `BatchSolveRequest`, par defaut a `0`.

## 22. Variante pools numeriques limites

Non executee: `threadpoolctl` indisponible. Aucune modification produit n'a ete faite sur les pools natifs.

## 23. Temps total

Voir sections 20 et 21.

## 24. Debit images/minute

Meilleure mesure courte observee:

```text
6 workers + stagger 200 ms: 28.60 img/min
```

Mais decision produit basee sur robustesse:

```text
6 workers + stagger 0 ms: 26.96 img/min
```

Scheduler actuel conserve.

## 25. Temps au premier resultat

```text
4w0: 28.67s
5w0: 9.37s
6w0 comparable: 11.86s
6w100: 10.16s
6w200: 10.13s
```

## 26. Duree mediane et p95

Extrait scheduler:

```text
4w0 median_task=14.03s p95=34.87s
5w0 median_task=10.52s p95=12.26s
6w0 comparable median_task=13.40s p95=14.11s
6w100 median_task=12.34s p95=13.95s
6w200 median_task=12.33s p95=14.17s
```

## 27. Duree de detection

```text
4w0 median_detect=9.64s p95=25.66s
5w0 median_detect=8.04s p95=10.67s
6w0 comparable median_detect=10.22s p95=11.26s
6w100 median_detect=9.34s p95=10.43s
6w200 median_detect=9.12s p95=10.06s
```

## 28. Occupation moyenne

```text
4w0 average_active=3.896/4
5w0 average_active=4.633/5
6w0 comparable average_active=5.913/6
6w100 average_active=5.814/6
6w200 average_active=5.782/6
```

## 29. Temps pleine occupation

```text
4w0 full=83.21s
5w0 full=32.72s
6w0 comparable full=38.11s
6w100 full=33.53s
6w200 full=33.13s
```

## 30. Handoff median et p95

```text
4w0 median=0.145 ms p95=5.150 ms
5w0 median=0.078 ms p95=0.101 ms
6w0 comparable median=0.068 ms p95=0.103 ms
6w100 median=0.070 ms p95=0.100 ms
6w200 median=0.063 ms p95=0.077 ms
```

Conclusion: pas d'idle scheduler significatif lorsque la queue est non vide.

## 31. Intervalles entre resultats

Exemples:

```text
6w0 comparable median interval=196.8 ms, p90=5402.6 ms
6w100 median interval=298.5 ms, p90=5342.3 ms
6w200 median interval=312.6 ms, p90=5185.6 ms
```

Les resultats restent visuellement groupes, mais le scheduler travaille en continu.

## 32. Memoire RSS

Extraits:

```text
4w0 rss_end=343516 KiB, peak=508212 KiB
5w0 rss_end=587320 KiB, peak=796156 KiB
6w0 comparable rss_end=668396 KiB, peak=861932 KiB
6w100 rss_end=558940 KiB, peak=866440 KiB
6w200 rss_end=651864 KiB, peak=827044 KiB
```

Pas de fuite manifeste observee dans ces runs courts; l'augmentation avec workers suit le nombre de pipelines et d'images en cours.

## 33. Charge CPU

La charge systeme exterieure etait le principal biais de mesure. La machine est un 4C/8T mobile; 5 et 6 workers se tiennent sur le run court quand la charge redevient raisonnable.

## 34. Parite scientifique

Toutes les configurations executees sur le sous-corpus 18 FITS donnent:

```text
18/18 SOLVED via ZeNear
```

Aucun seuil scientifique n'a ete modifie. Les modifications touchent le transport, la telemetry, le mode CPU expose depuis `ProductSettings.gpu_mode`, et le stagger experimental desactive par defaut.

## 35. Test Stop

Test hermetique ajoute:

```text
test_s6a1b_stop_during_startup_stagger_is_not_blocked_by_sleep
```

Il verifie que le delai experimental initial est annulable. Un defaut adjacent a ete corrige: si Stop arrive apres un resultat Near resolu, les requetes non finales sont backfill en `CANCELLED` au lieu de pouvoir disparaitre du resultat final.

## 36. Conclusion contention

Pas de contention scheduler prouvee. Les handoff gaps sont sub-millisecondes a quelques millisecondes selon les runs. La contention eventuelle est cote calcul CPU/detection et charge systeme, pas cote pool/refill.

## 37. Decision scheduler

Conserver le scheduler produit actuel:

```text
ThreadPoolExecutor unique
toutes les taches soumises
threads persistants
pipelines worker-local
as_completed
```

## 38. Decision stagger

Ne pas activer en produit. Le gain observe est marginal/biaise et le seuil de decision reproductible n'est pas atteint. Le stagger reste disponible uniquement pour mesure experimentale, valeur par defaut `0`.

## 39. Decision pools numeriques

Pas de changement produit. `threadpoolctl` absent; OpenBLAS MAX_THREADS=64 indique une piste a qualifier plus tard si une vraie sursouscription est reproduite.

## 40. Fichiers modifies

Nouveaux:

- `tests/test_s6a1b_near_scheduler_continuity.py`
- `tools/measure_s6a1b_near_scheduling.py`
- `docs/stabilization/s6a1b_zenear_scheduler_continuity_report_20260726.md`

Modifies pendant S6A-1B:

- `.gitignore`
- `zeblindsolver/metadata_solver.py`
- `zeblindsolver/near_catalog_provider.py`
- `zesolver/core/batch/models.py`
- `zesolver/core/batch/runner.py`
- `zesolver/gui_pipeline/pipeline_runner.py`
- `zesolver/gui_pipeline/requests.py`
- `zesolver/gui_pipeline/settings_adapter.py`
- `zesolver/resource_telemetry.py`
- `zesolver/settings/assembly.py`
- `tests/test_s6a1_near_shared_catalog_runtime.py`

Les autres fichiers modifies etaient deja dans le worktree S6A-1.

## 41. Barrieres executees

```text
.venv/bin/python tools/check_core_boundaries.py
=> core boundary check: OK

.venv/bin/python -m pytest -q \
 tests/test_s6a1_near_shared_catalog_runtime.py \
 tests/test_s6a1b_near_scheduler_continuity.py \
 tests/test_near_catalog_provider_astap.py \
 tests/test_batch_pipeline_scheduling.py \
 tests/test_batch_pipeline_concurrency.py \
 tests/test_batch_pipeline_cancellation.py \
 tests/test_s5f_blind_progressive_results.py
=> 28 passed

.venv/bin/python tools/run_regression_suite.py --hermetic
=> PASS, 655 passed, 1 skipped, 9 deselected

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
=> 655 passed, 10 skipped

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
=> OK

git diff --check
=> OK
```

## 42. Etat Git final

Etat final exact:

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

## 43. Gate final

Reponses directes:

```text
Les threads sont-ils recrees tous les six fichiers ? Non.
Les pipelines sont-ils reconstruits par image ? Non.
Existe-t-il une barriere apres six taches ? Non.
Toutes les taches sont-elles soumises des le debut ? Oui.
worker_handoff_gap median/p95 ? 6w0 comparable: 0.068 ms / 0.103 ms.
Concurrence active moyenne ? 6w0 comparable: 5.913/6.
4, 5 ou 6 workers meilleur debit ? 5 et 6 equivalents sur run comparable; 6 legerement meilleur, non decisif.
Stagger gain reel ? Non qualifie; gain apparent biaise, pas integre.
Resultats seulement plus reguliers visuellement ? Les vagues sont visuelles/calcul, pas scheduler.
Sursouscription BLAS/OpenMP ? Risque possible, non mesure faute threadpoolctl.
Evenements fermeture observables ? Oui, test >128 acquisitions.
Compteur duplicate_loads clarifie ? Oui, alias singleflight_waiters ajoute.
Modification conservee ? Telemetry/tests/outils/log Near une fois; scheduler produit inchange.
Barrieres ? Toutes executees et passees.
Gate ? S6A1B_SCHEDULER_CONTINUITY_CONFIRMED / READY_FOR_S6A2_ZENEAR_GPU_RUNTIME.
```
