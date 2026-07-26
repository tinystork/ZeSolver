# S6A-2B - ZeNear GPU Telemetry and Stop Hardening

Date: 2026-07-26

## 1. Etat Git initial

Etat inspecte avant modification S6A-2B:

```text
git status --short --branch
## test...origin/test
 M .gitignore
 M zeblindsolver/metadata_solver.py
 M zesolver.py
 M zesolver/core/batch/runner.py
 M zesolver/core/pipeline.py
 M zesolver/gui_pipeline/settings_adapter.py
 M zesolver/resource_telemetry.py
 M zesolver/settings/assembly.py
 M zesolver/settings/product.py
?? docs/stabilization/s6a2_zenear_gpu_runtime_report_20260726.md
?? tests/test_s6a2_zenear_gpu_detection.py
?? tests/test_s6a2_zenear_gpu_fallback.py
?? tests/test_s6a2_zenear_gpu_parity.py
?? tests/test_s6a2_zenear_gpu_runtime.py
?? tools/capture_s6a2_zenear_cpu_oracle.py
?? tools/measure_s6a2_zenear_gpu_detection.py

git diff --check
OK
```

`git diff` a ete inspecte. Le worktree contenait deja les changements S6A-2 non commit avant cette finition. Aucun commit et aucun push effectue.

## 2. Fichiers inspectes

Inspectes pour S6A-2B:

```text
zeblindsolver/metadata_solver.py
zesolver/resource_telemetry.py
zesolver/core/pipeline.py
zesolver/core/batch/runner.py
zesolver/gui_pipeline/pipeline_runner.py
zesolver/gui_pipeline/requests.py
zesolver/gui_pipeline/settings_adapter.py
zesolver/settings/assembly.py
zesolver/settings/product.py
zesolver.py
docs/stabilization/s6a2_zenear_gpu_runtime_report_20260726.md
zesolver.log
```

## 3. Analyse du log utilisateur

Log manuel inspecte: `zesolver.log`.

Lignes clefs observees:

```text
GUI_RUN_BEGIN run_id=2
STOP_UI_CLICKED              2026-07-26 10:13:57.600
STOP_RUNNER_RECEIVED         2026-07-26 10:13:57.712
STOP_CONTROLLER_RECEIVED     2026-07-26 10:13:57.780
near detect start            2026-07-26 10:13:58.857
near detect start            2026-07-26 10:13:58.883
near detect start            2026-07-26 10:13:58.897
GUI_RUN_BEGIN run_id=3
```

Le log contenait aussi de nombreuses lignes INFO par image:

```text
near detect backend used: requested=auto selected=cuda used=cuda device=0 gpu_slots=1
```

## 4. Preuve Auto -> CUDA

Le log manuel confirme:

```text
requested=auto
selected=cuda
used=cuda
device=0
gpu_slots=1
```

Verification reelle post-patch en detection seule sur 30 FITS:

```text
backend_used=['cuda']
count=30
fallbacks=0
vram_peak=55312384 bytes
cupy_pool_reserved_peak=12443136 bytes
```

Verification produit post-patch sur copies temporaires:

```text
.venv/bin/python zesolver.py --headless --input-dir /tmp/zesolver_s6a2b_auto30_20260726 --catalog-library /home/tristan/ZeSolverCatalog/catalog.json --workers 6 --near-detect-backend auto --near-detect-device 0 --near-detect-gpu-slots 1 --no-blind --overwrite --max-files 30 --log-level INFO
Done in 60.5s - 30 solved, 0 skipped, 0 failed
VRAM runtime log: used=53.0MiB, process=46.0MiB
```

## 5. Chronologie Stop observee

Avant S6A-2B, des taches deja entrees dans la resolution Near pouvaient atteindre `near detect start` environ 1.1 a 1.3 s apres `STOP_UI_CLICKED`.

Cause auditee: dans le pipeline GUI moderne, le pipeline testait bien l'annulation avant Near, mais `ExistingNearSolverPort.solve()` ne transmettait pas le token a `near_solve`. Les taches deja engagees dans `solve_near` ne voyaient donc pas le Stop avant la detection.

## 6. Checkpoints existants avant modification

Existants:

```text
SolverPipeline.solve(): avant Near
SolverPipeline.solve(): entre Near et Blind
solve_near(): debut de fonction
solve_near(): apres resolution provider/index
solve_near(): avant selection/chargement de chaque tuile catalogue
solve_near(): avant detection
solve_near(): avant ecriture WCS
BatchSolverPipeline: token batch entre futures
GUI worker: STOP_UI_CLICKED -> STOP_RUNNER_RECEIVED -> STOP_CONTROLLER_RECEIVED
```

Le checkpoint `solve_near(): avant detection` n'etait pas utile dans le pipeline GUI moderne tant que le token n'etait pas propage.

## 7. Checkpoints ajoutes

Ajoutes ou durcis:

```text
ExistingNearSolverPort -> near_solve(cancel_check=...)
detect_stars_astap_strict(cancel_check=...)
_gpu_detection_slot(): attente annulee avant acquisition
_gpu_detection_slot(): re-check juste apres acquisition
detect_stars_astap_strict(): re-check apres preparation GPU, avant continuation CPU ASTAP-ISO
SolverPipeline: exception "cancelled" conservee en SolveStatus.CANCELLED
BatchResourceTelemetry: horodatage monotone cancel_requested_at
```

`near detect start` et `near detect backend used` passent en DEBUG pour ne plus noyer le log INFO.

## 8. Definition d'une section GPU

Pour S6A-2B, une section GPU est l'intervalle protege par le semaphore CUDA autour de la preparation GPU stricte ASTAP-ISO:

```text
acquisition slot GPU
gpu_section_started_at
H2D
calcul CuPy
D2H
gpu_section_finished_at
release slot GPU
```

La suite CPU ASTAP-ISO apres retour NumPy n'est pas une section GPU.

## 9. Telemetrie interne

`BatchResourceTelemetry` agregue maintenant:

```text
backend requested/selected/used
images_cuda/images_cpu
fallbacks et fallback_reasons
gpu_errors/gpu_oom
gpu_disabled_for_batch/reason
detect_duration_ms
gpu_slot_wait_ms
transfer_h2d_ms_total
gpu_compute_ms_total
transfer_d2h_ms_total
vram_peak_bytes
cupy_pool_used/reserved peak
detections_started_after_cancel
gpu_sections_started_after_cancel
gpu_sections_finished_after_cancel
samples bornes
```

La telemetrie reste best-effort: une erreur d'enregistrement ne change pas le resultat scientifique.

## 10. Politique des niveaux de log

INFO:

```text
ZeNear detection requested: ...
ZeNear detection active: ...
ZeNear detection summary: ...
```

WARNING:

```text
ZeNear CUDA fallback: reason=... continuing_on=cpu gpu_disabled_for_batch=...
```

DEBUG:

```text
near detect start
near detect backend used: requested=... selected=... used=...
timings detailles par image
```

## 11. Resume INFO de debut

`PipelineGuiRunner` emet au debut:

```text
ZeNear detection requested: backend=auto gpu_slots=1
```

La premiere utilisation effective emet ensuite:

```text
ZeNear detection active: requested=auto selected=cuda used=cuda device=0
```

## 12. Resume INFO de fin

`BatchSolverPipeline._finish()` emet une synthese compacte:

```text
ZeNear detection summary: requested=auto cuda_images=... cpu_images=... fallbacks=... gpu_errors=... gpu_oom=... device=... detect_median_ms=... detect_p95_ms=... gpu_slot_wait_p95_ms=... vram_peak=... terminal=completed|cancelled|failed
```

Cette synthese est testee pour `completed`, `cancelled` et `failed` via payload sidecar/summary hermetiques.

## 13. Sidecar JSON

Ajout d'un sidecar par run GUI:

```text
zesolver_run_YYYYMMDD_HHMMSS.log
zesolver_run_YYYYMMDD_HHMMSS.telemetry.json
```

Le sidecar contient:

```text
schema=zesolver.run_telemetry.v1
run
input
near_detection
cancellation
```

## 14. Schema JSON

Schema minimal versionne:

```json
{
  "schema": "zesolver.run_telemetry.v1",
  "run": {},
  "input": {},
  "near_detection": {},
  "cancellation": {}
}
```

Les valeurs inconnues restent `null` ou absentes via les agregats, jamais inventees.

## 15. Ecriture atomique

`write_run_telemetry_sidecar()` ecrit:

```text
destination.telemetry.json.tmp
flush
fsync best-effort
os.replace()
```

Une erreur d'ecriture est loggee en warning et n'altere pas le statut du solve.

## 16. Politique de volumetrie

Le sidecar est compact:

```text
compteurs
sommes
min/median/p95/max
raisons agregees
samples first_16_last_16
sample_count
samples_truncated
sample_policy
```

Pas de liste complete par image par defaut sur les tres grands batchs.

## 17. Indicateur GUI

Ajout d'une ligne discrète dans la barre basse:

```text
ZeNear : préparation...
ZeNear : Auto -> CUDA - GPU 0
ZeNear : CUDA - GPU 0
ZeNear : Auto -> CPU
ZeNear : Auto -> CPU - fallback CUDA
```

L'indicateur lit `used_last`/`device_last` depuis la telemetrie, donc il suit le backend effectif et pas seulement la demande.

## 18. Reset entre runs

Reset effectue au demarrage d'un run GUI:

```text
near_backend_label = "ZeNear : préparation..."
_last_summary reset par nouveau worker
run_id nouveau via GUI_RUN_BEGIN
```

Reset effectue au niveau batch moderne:

```text
reset_zenear_gpu_runtime_state()
nouveau BatchResourceTelemetry
nouveau contexte active_batch_telemetry
```

Tests hermetiques: etat GPU disabled/fallback reinitialise et CUDA reutilisable apres reset.

## 19. Acquisition annulable du semaphore

`_gpu_detection_slot()` n'utilise plus une acquisition bloquante longue:

```text
while not acquired:
    if cancel_check(): raise RuntimeError("cancelled_waiting_for_gpu_slot")
    acquired = sem.acquire(timeout=0.05)
if cancel_check(): raise RuntimeError("cancelled_after_gpu_slot_acquired")
```

Le `finally` libere le slot exactement une fois si acquis.

## 20. Comportement kernel deja lance

Un kernel deja lance n'est pas tue brutalement. Le contrat applique:

```text
laisser finir la preparation GPU deja entree
release semaphore
verifier Stop juste apres retour GPU
ne pas lancer la continuation CPU ASTAP-ISO si Stop est present
retourner CANCELLED au pipeline
```

## 21. Metriques apres Stop

Telemetrie monotone:

```text
cancel_requested_at
detect_started_at
gpu_slot_wait_started_at
gpu_section_started_at
gpu_section_finished_at
detect_finished_at
```

Compteurs:

```text
detections_started_after_cancel
gpu_sections_started_after_cancel
gpu_sections_finished_after_cancel
```

Gate hermetique attendu pour S6A-2B:

```text
gpu_sections_started_after_cancel = 0
```

## 22. Test completed

Couvert par:

```text
tests/test_s6a2b_gpu_telemetry.py
tests/test_s6a2b_gui_backend_indicator.py
```

Le sidecar `completed` est construit avec schema versionne et nom de base partage avec le log.

## 23. Test cancelled

Couvert par:

```text
tests/test_s6a2b_gpu_stop_checkpoint.py
tests/test_s6a2b_gpu_telemetry.py
tests/test_batch_pipeline_cancellation.py
```

Les annulations avant detection et autour du slot GPU restent `CANCELLED`, pas `FAILED` ni fallback CUDA.

## 24. Test failed

Couvert par payload sidecar `failed` et par la synthese terminale `terminal=failed` en telemetrie.

## 25. Test relance apres Stop

Hermetique:

```text
reset_zenear_gpu_runtime_state()
CUDA probe en echec -> gpu_disabled_for_batch
reset
CUDA probe OK -> backend_used=cuda
```

Manuel fourni:

```text
STOP sur run_id=2
GUI_RUN_BEGIN run_id=3
Auto -> CUDA reutilise dans la meme session
```

## 26. Test reel CUDA

Machine: TINYDEBIAN.

CUDA:

```text
GPU: NVIDIA GeForce MX150
driver: 550.163.01
CuPy: 14.1.1
runtime: 12090
driverGetVersion: 12040
device_count: 1
mem: 2050621440 free / 2091253760 total
```

Detection seule Auto/30:

```text
backend_used=['cuda']
count=30
fallbacks=0
detect_total median=1.5416s p95=1.8090s total=42.9137s
H2D total=0.1233s
GPU compute total=0.0637s
D2H total=0.0257s
vram_peak=55312384 bytes
```

Produit Auto/30 sur copies temporaires:

```text
30 solved, 0 skipped, 0 failed
VRAM stable: used=53.0MiB, process=46.0MiB
```

## 27. gpu_sections_started_after_cancel

Tests hermetiques:

```text
Stop avant attente slot -> aucune section GPU
Stop juste apres acquisition -> aucune fonction CUDA lancee, semaphore libere
compteur synthetique detecte correctement une section demarree apres cancel
```

Le log manuel pre-patch ne permettait pas de distinguer section GPU deja entree et detection commencee apres Stop; S6A-2B ajoute les timestamps necessaires.

## 28. Parite scientifique

Aucune modification scientifique volontaire:

```text
aucun seuil modifie
near_astap_iso_strict conserve
etiquetage/centroides/tri ASTAP-ISO inchanges
```

Non-regression executee:

```text
S6A-2 targeted tests: OK
Produit Auto/30 sur copies temporaires: 30/30 solved
```

La mission ne rouvre pas la parite CPU/GPU deja qualifiee par S6A-2.

## 29. Fichiers modifies

Fichiers S6A-2B principaux:

```text
zeblindsolver/metadata_solver.py
zesolver/resource_telemetry.py
zesolver/core/pipeline.py
zesolver/core/batch/runner.py
zesolver/gui_pipeline/pipeline_runner.py
zesolver.py
tests/test_s6a2b_gpu_telemetry.py
tests/test_s6a2b_gpu_stop_checkpoint.py
tests/test_s6a2b_gui_backend_indicator.py
docs/stabilization/s6a2b_gpu_telemetry_stop_report_20260726.md
reports/s6a2b_detection_auto30_20260726.json
```

Le worktree contient aussi les fichiers S6A-2 preexistants non commit.

## 30. Barrieres executees

Passees:

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python -m pytest -q tests/test_s6a2b_gpu_telemetry.py tests/test_s6a2b_gpu_stop_checkpoint.py tests/test_s6a2b_gui_backend_indicator.py
9 passed

.venv/bin/python -m pytest -q [tests cibles S6A-1/S6A-2/S6A-2B/batch]
47 passed

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

Detection seule CUDA reelle Auto/30
30 CUDA, 0 fallback

Produit CUDA reel Auto/30 sur copies temporaires
30 solved, 0 skipped, 0 failed
```

Executees mais non passees pour cause externe locale:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
685 passed, 1 skipped, 9 deselected, 1 failed

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
685 passed, 10 skipped, 1 failed
```

Echec commun:

```text
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha
```

Cause identifiee: les 8 FITS externes de `/home/tristan/near_bench_cmp30/thread4` referencés par `reports/zenear_zn310b_gui_manifest.json` ne correspondent plus aux `source_SHA256` attendus. Exemple:

```text
ZN310B_001_M31
expected c24fe1c2fee406bd9d98f617998061aba6e4cdfbcd2ec6dff8b640be17dcfd5b
actual   56c449c0a432e8dfb73e41628bc5cb8bf1f656490369d9d9ef828af5fb9a7990
```

Cet ecart n'apparait pas dans `git status` et concerne des donnees externes au repository.

## 31. Etat Git final

Etat final apres travaux:

```text
git diff --check
OK

git status --short --branch
## test...origin/test
 M .gitignore
 M zeblindsolver/metadata_solver.py
 M zesolver.py
 M zesolver/core/batch/runner.py
 M zesolver/core/pipeline.py
 M zesolver/gui_pipeline/pipeline_runner.py
 M zesolver/gui_pipeline/settings_adapter.py
 M zesolver/resource_telemetry.py
 M zesolver/settings/assembly.py
 M zesolver/settings/product.py
?? docs/stabilization/s6a2_zenear_gpu_runtime_report_20260726.md
?? docs/stabilization/s6a2b_gpu_telemetry_stop_report_20260726.md
?? tests/test_s6a2_zenear_gpu_detection.py
?? tests/test_s6a2_zenear_gpu_fallback.py
?? tests/test_s6a2_zenear_gpu_parity.py
?? tests/test_s6a2_zenear_gpu_runtime.py
?? tests/test_s6a2b_gpu_stop_checkpoint.py
?? tests/test_s6a2b_gpu_telemetry.py
?? tests/test_s6a2b_gui_backend_indicator.py
?? tools/capture_s6a2_zenear_cpu_oracle.py
?? tools/measure_s6a2_zenear_gpu_detection.py
```

Le worktree reste non commit comme demande.

## 32. Gate final

Gate fonctionnel S6A-2B:

```text
S6A2B_GPU_OBSERVABILITY_QUALIFIED
S6A2B_STOP_CHECKPOINTS_HARDENED
GPU_SECTIONS_STARTED_AFTER_CANCEL_ZERO
GUI_BACKEND_VISIBILITY_CONFIRMED
RUN_TELEMETRY_SIDECAR_CONFIRMED
```

Gate release global:

```text
NOT_READY_FOR_S6A2_FINAL_COMMIT
```

Raison: les barrieres globales `run_regression_suite.py --hermetic` et `pytest -q offscreen` sont executees mais restent rouges sur le hash externe ZN310B local. Le code S6A-2B cible est vert; la validation finale necessite de restaurer ou regenerer le corpus ZN310B externe attendu.
