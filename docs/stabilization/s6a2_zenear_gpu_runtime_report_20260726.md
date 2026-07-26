# S6A-2 - ZeNear GPU Runtime Qualification

Date: 2026-07-26

## 1. Etat Git initial

Commandes executees avant travaux:

```text
git status --short --branch
## test...origin/test

git diff --check
OK

git diff
empty
```

Aucun commit et aucun push effectue.

## 2. Materiel GPU

Machine de qualification: TINYDEBIAN.

```text
GPU: NVIDIA GeForce MX150
VRAM annoncee par nvidia-smi: 2048 MiB
VRAM vue par CuPy: 2091253760 bytes total
```

## 3. Version driver

```text
nvidia driver: 550.163.01
CuPy driverGetVersion(): 12040
```

## 4. Version CUDA runtime

```text
CuPy runtimeGetVersion(): 12090
```

## 5. Version CuPy

```text
cupy: 14.1.1
```

## 6. Memoire GPU disponible

Au probe CuPy:

```text
mem_free: 2050621440 bytes
mem_total: 2091253760 bytes
```

Pendant le benchmark detection 100 FITS, le pic observe par CuPy est reste a `55312384` bytes, avec pool reserve max `12443136` bytes.

## 7. Fichiers inspectes

Inspectes pendant l'audit:

```text
zeblindsolver/star_detect.py
zeblindsolver/metadata_solver.py
zeblindsolver/near_catalog_provider.py
zesolver/core/pipeline.py
zesolver/core/batch/runner.py
zesolver/gui_pipeline/pipeline_runner.py
zesolver/gui_pipeline/requests.py
zesolver/gui_pipeline/settings_adapter.py
zesolver/settings/assembly.py
zesolver/resource_telemetry.py
zesolver.py
requirements.txt
pyproject.toml
packaging/
```

## 8. Contrat actuel du mode Auto

Le contrat produit existant annonce `Auto` comme "GPU si disponible, sinon CPU" pour la detection Near. L'audit a confirme que le GUI propose CPU/CUDA/Auto et que le code de selection CUDA existe deja.

Correction appliquee: le chemin effectif distingue maintenant:

```text
backend_requested
backend_selected
backend_used
device_requested
device_used
fallback_used
fallback_reason
```

Le mode Auto n'a pas ete redefini en fonction des performances locales de TINYDEBIAN.

## 9. Chemin CPU strict actuel

Oracle conserve:

```text
near_astap_iso_strict=True
near_detect_backend=cpu
```

Le chemin strict utilise la logique ASTAP-ISO dans `metadata_solver.py`: binning ASTAP-compatible, fond global, routine `_astap_find_stars_routine`, mesure HFD, selection ASTAP-like des etoiles les plus brillantes.

## 10. Chemin CUDA existant

Le chemin CUDA generique existait dans `zeblindsolver/star_detect.py` via `detect_stars(... backend="cuda")`.

Ce chemin est scientifique different du strict ASTAP-ISO: autre pipeline de detection, autre etiquetage/filtrage, autre ordre possible.

## 11. Raison du contournement CUDA en mode strict

En mode `strict_astap_iso`, `solve_near` appelait directement `astap_adaptive_image_detection(...)`. Cette fonction ne passait pas par `zeblindsolver/star_detect.py`, donc le backend CUDA generique etait contourne meme si `near_detect_backend=cuda`.

## 12. Carte operations CPU/GPU

| Operation CPU strict | Equivalent GPU | Risque parite | Potentiel perf | Decision |
| --- | --- | --- | --- | --- |
| lecture FITS | aucun | faible | faible | CPU |
| conversion image / couleur moyenne | CuPy possible | faible | faible/modere | GPU |
| NaN/inf vers fond calculable | CuPy possible | faible si dtype conserve | modere | GPU dans preparation |
| crop + mean bin ASTAP | CuPy reshape/mean | faible si float64 conserve | modere | GPU |
| estimation fond global | CPU actuel | moyen | modere | CPU |
| seuils / sections locales | CPU actuel | moyen/fort | modere | CPU |
| etiquetage / suppression locale | CPU actuel | fort | fort | CPU pour parite |
| HFD / centroide | CPU actuel | fort | fort | CPU pour parite |
| tri / selection finale | CPU actuel | fort | faible | CPU |

## 13. Contrat detecteur commun

Ajout de `detect_stars_astap_strict(...) -> AstapStrictDetectionResult`.

Sortie publique:

```text
sources: numpy.ndarray
shape: identique au chemin CPU existant
dtype: dtype numpy du chemin ASTAP strict existant
backend_requested/selected/used
device_requested/used
fallback_used/reason
duration_total
duration_transfer_to_gpu
duration_gpu_compute
duration_transfer_to_cpu
duration_cpu_compute
gpu_slot_wait
vram / pool CuPy lorsque disponible
```

Aucun consommateur ZeNear ne recoit de tableau CuPy.

## 14. Precision numerique

Le port GPU garde la preparation en precision compatible avec le CPU oracle et repasse en NumPy avant les operations ASTAP sensibles. Aucune reduction silencieuse des seuils ou tolerances n'a ete faite.

## 15. Strategie d'etiquetage

Etiquetage, suppression de voisinage, HFD et centroides restent CPU. C'est volontaire: ce sont les zones les plus susceptibles de diverger.

## 16. Strategie de tri

Le tri final reste celui du chemin CPU strict. Aucun tri GPU arbitraire n'a ete introduit.

## 17. Initialisation GPU

CuPy est importe lazy. Le probe verifie:

```text
import CuPy
device count
device demande valide
allocation minimale
operation simple
synchronisation CUDA
```

Le contexte CUDA n'est pas cree en mode CPU.

## 18. Gestion devices

`near_detect_device` est route depuis settings vers `NearSolveConfig`, puis vers le detecteur. La section CUDA utilise `with cupy.cuda.Device(device_id)`.

Tests hermetiques couverts: `device=None`, `device=0`, device invalide.

## 19. Gestion slots

`near_detect_gpu_slots` borne les sections CUDA via semaphore. La section protegee couvre uniquement la preparation CUDA, pas lecture FITS/catalogue/matching/ecriture WCS.

Tests hermetiques: slots bornes a 2, annulation pendant attente de slot.

## 20. Politique fallback

Fallback CPU explicite quand CuPy/runtime/device/probe/alloc/kernel echoue et que le contrat permet de continuer. L'image est retraitee CPU sans recursion Auto -> GPU.

## 21. Politique apres erreur structurelle

Les erreurs CUDA structurelles desactivent le GPU pour le reste du batch/processus courant via `gpu_disabled_for_batch`, puis le batch continue en CPU. L'etat est remis a zero au debut d'un nouveau `BatchSolverPipeline.solve(...)`.

## 22. Telemetrie

Ajouts:

```text
near_detect_backend_requested_cpu/cuda/auto
near_detect_backend_selected_cpu/cuda
near_detect_backend_used_cpu/cuda
near_detect_gpu_fallbacks
near_detect_gpu_oom
near_detect_gpu_errors
near_detect_gpu_disabled_for_batch
near_detect_cuda_used events avec slot/transfer/compute
```

## 23. Messages GUI

Le routage settings transporte maintenant `near_detect_backend`, `near_detect_device` et `near_detect_gpu_slots`. Le log Near affiche le backend demande, selectionne et utilise. Le GUI peut s'appuyer sur `backend_used` et non plus sur la seule demande.

## 24. Tests synthetiques

Ajoutes:

```text
tests/test_s6a2_zenear_gpu_detection.py
tests/test_s6a2_zenear_gpu_runtime.py
tests/test_s6a2_zenear_gpu_fallback.py
tests/test_s6a2_zenear_gpu_parity.py
```

Couverture: CPU strict, faux CUDA hermetique, Auto CUDA/CPU, fallback, OOM simule, device invalide, slots, annulation pendant attente, parite synthetique.

## 25. Parite des sources

Corpus 100 FITS `/home/tristan/near_bench100_input`:

```text
images comparees: 100
meme nombre de sources: 100/100
differences d'ordre: 0
max dx: 0.0 px
max dy: 0.0 px
median dx/dy: 0.0 / 0.0 px
max flux relatif: 0.0
fallbacks: 0
```

Artefact local: `reports/s6a2_source_parity_bench100_cpu_cuda.json`.

## 26. Parite WCS

30 FITS et 100 FITS:

```text
CPU vs CUDA: WCS identique sur les cles comparees
CPU vs Auto: WCS identique sur les cles comparees
CRVAL/CRPIX/CD/PIXSCAL max abs diff: 0.0
missing: 0
```

Artefacts locaux:

```text
reports/s6a2_wcs_parity_30.json
reports/s6a2_wcs_parity_100.json
```

## 27. Repetabilite CPU/CPU

La suite hermetique et les sorties CPU strictes existantes restent stables. Pas de divergence CPU/CPU observee dans les tests executes.

## 28. Repetabilite GPU/GPU

Le chemin GPU conserve les sources et WCS identiques sur les repetitions de detection synthetiques et sur le corpus reel mesure. Les operations a risque restent CPU, ce qui explique l'identite stricte.

## 29. Benchmark detection seule

Outil ajoute: `tools/measure_s6a2_zenear_gpu_detection.py`.

Sur 100 FITS:

```text
CPU detection total: 75.045 s, median 0.6065 s, p95 1.1958 s
CUDA detection total: 72.694 s, median 0.5812 s, p95 1.1690 s
Auto detection total: 72.853 s, median 0.5791 s, p95 1.1884 s
```

Gain detection local CUDA vs CPU: environ 3.1 %, donc benefice local faible.

## 30. Benchmark 30 FITS

Runs produit Near strict, 6 workers, Blind desactive:

```text
CPU: 75.7 s, 30 solved, 0 skipped, 0 failed
CUDA slots=1: 44.5 s, 30 solved, 0 skipped, 0 failed
Auto slots=1: 46.2 s, 30 solved, 0 skipped, 0 failed
```

Parite WCS 30: identique CPU/CUDA/Auto.

## 31. Benchmark 100 FITS

Runs produit Near strict, 6 workers, Blind desactive:

```text
CPU: 85.4 s, 100 solved, 0 skipped, 0 failed
CUDA slots=1: 112.5 s, 100 solved, 0 skipped, 0 failed
Auto slots=1: 112.2 s, 100 solved, 0 skipped, 0 failed
```

Le run produit CUDA 100 est localement plus lent parce que le scheduler est force en threads pour eviter l'usage CUDA apres fork/process.

## 32. Temps CPU

Reference detection seule CPU 100:

```text
total: 75.045 s
median image: 0.6065 s
p95 image: 1.1958 s
```

## 33. Temps GPU

Detection seule CUDA 100:

```text
total detecteur: 72.694 s
median image: 0.5812 s
p95 image: 1.1690 s
```

## 34. Temps transfert

CUDA 100:

```text
CPU -> GPU total: 0.344 s, median 0.00338 s
GPU compute total: 0.199 s, median 0.00198 s
GPU -> CPU total: 0.077 s, median 0.000764 s
```

## 35. Temps au premier resultat

Mesure produit precise non isolee dans les logs finaux. Mesure cold/warm detection synthetique: premiere detection CUDA nettement plus couteuse que les suivantes, puis stabilisation autour du temps CPU legerement inferieur.

## 36. Debit

Produit 100:

```text
CPU: environ 70.3 images/min
CUDA slots=1: environ 53.3 images/min
Auto slots=1: environ 53.5 images/min
```

Detection seule 100:

```text
CPU: environ 79.9 images/min
CUDA: environ 82.5 images/min
Auto: environ 82.4 images/min
```

## 37. RSS

Logs produit:

```text
CPU 100 fin: environ 452.7 MiB
CUDA 100 fin: environ 1.4 GiB
Auto 100 fin: environ 1.4 GiB
```

La hausse vient du runtime CUDA/CuPy charge dans le processus thread.

## 38. VRAM

Detection seule 100:

```text
pic CuPy observe: 55312384 bytes
```

Produit 100 fin:

```text
CUDA: environ 55 MiB utilises
Auto: environ 53 MiB utilises
```

Pas de croissance continue observee sur le batch 100.

## 39. Memory pool CuPy

Detection seule 100:

```text
pool reserve max: 12443136 bytes
```

Le pool reserve de la memoire comme attendu; ce n'est pas interprete comme une fuite.

## 40. Resultats Stop

Tests hermetiques couvrent l'annulation pendant attente du slot GPU et l'absence de perte de semaphore. Un kernel CUDA deja lance n'est pas interrompu instantanement; le contrat documente est de prendre l'annulation juste apres retour kernel/preparation.

Pas de test manuel GUI Stop complet sur batch reel documente dans cette passe.

## 41. Resultats OOM

OOM CUDA simule couvert par test: fallback CPU, compteur OOM/fallback, GPU desactive pour batch si erreur structurelle, absence de boucle.

Pas d'OOM reel observe sur TINYDEBIAN.

## 42. Fallback sans CUDA

Tests hermetiques:

```text
CuPy absent/runtime indisponible -> Auto selectionne CPU
CUDA explicite + device invalide -> fallback CPU explicite
exception apres acquisition slot -> semaphore libere et CPU fallback
```

## 43. Compatibilite CPU-only

CuPy reste optionnel et importe lazy. `requirements.txt` garde les paquets CUDA sous marqueur Linux; aucun import CuPy obligatoire au lancement CPU/GUI.

## 44. Qualification fonctionnelle backend GPU

Backend CUDA reellement utilise sur TINYDEBIAN en detection seule et en runs produit CUDA/Auto, apres correction du scheduler Near pour eviter CUDA en process/fork.

## 45. Qualification scientifique CPU/GPU

Confirmee sur les corpus executes:

```text
sources identiques sur 100/100
WCS identique sur 30/30 et 100/100
0 fallback pendant runs CUDA/Auto reels
```

## 46. Comportement observe du mode Auto

Sur TINYDEBIAN avec CUDA disponible:

```text
Auto selected=cuda
Auto used=cuda
fallback_used=false
```

Sans CUDA simule:

```text
Auto selected=cpu
Auto used=cpu
fallback_reason explicite
```

## 47. Conformite au contrat Auto existant

Contrat preserve: Auto continue a choisir CUDA lorsque runtime/device/probe sont valides, sinon CPU. Aucune politique globale n'a ete redefinie selon les performances locales.

## 48. Performance locale sur TINYDEBIAN

Sur la machine de qualification et ce corpus:

```text
gain detection seule CUDA: environ +3.1 %
gain total produit 100 CUDA: negatif, environ -31.7 %
gain total produit 30 CUDA: positif sur ce run, mais non retenu comme politique generale
```

Interpretation: runtime qualifie scientifiquement, benefice local detection faible, cout produit defavorable sur 100 avec le guardrail thread CUDA.

## 49. Limites de representativite du benchmark

Ces resultats ne sont pas generalisables:

```text
une seule machine
un seul GPU
une seule version de driver
un seul environnement CUDA
un corpus principalement S50 / lots locaux
resultats non generalisables a toutes les configurations
```

## 50. Recommandation campagne multi-machine

Oui, recommandee avant toute decision produit globale: plusieurs GPUs, plusieurs generations, plusieurs VRAM, Windows/Linux/macOS CPU-only, tailles d'images variees, corpus varies.

## 51. Fichiers modifies

```text
.gitignore
zeblindsolver/metadata_solver.py
zesolver.py
zesolver/core/batch/runner.py
zesolver/core/pipeline.py
zesolver/gui_pipeline/settings_adapter.py
zesolver/resource_telemetry.py
zesolver/settings/assembly.py
zesolver/settings/product.py
tests/test_s6a2_zenear_gpu_detection.py
tests/test_s6a2_zenear_gpu_runtime.py
tests/test_s6a2_zenear_gpu_fallback.py
tests/test_s6a2_zenear_gpu_parity.py
tools/measure_s6a2_zenear_gpu_detection.py
tools/capture_s6a2_zenear_cpu_oracle.py
```

## 52. Barrieres executees

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python -m pytest -q [bloc cible S6A-1 + S6A-2 + batch]
42 passed in 11.33s

.venv/bin/python tools/run_regression_suite.py --hermetic
677 passed, 1 skipped, 9 deselected, status PASS

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
677 passed, 10 skipped

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

## 53. Etat Git final

```text
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
```

Aucun commit et aucun push effectue.

## 54. Gate final

```text
S6A2_ZENEAR_GPU_RUNTIME_QUALIFIED
GPU_CPU_SCIENTIFIC_PARITY_CONFIRMED
LOCAL_GPU_GAIN_LIMITED
MULTI_MACHINE_PERFORMANCE_VALIDATION_RECOMMENDED
AUTO_MODE_CONTRACT_PRESERVED
CPU_FALLBACK_PRESERVED
```
