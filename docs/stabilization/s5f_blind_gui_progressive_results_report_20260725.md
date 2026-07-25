# S5F - Blind GUI Progressive Results

Date: 2026-07-25

Gate: `READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION`

## 1. Reproduction du buffering Blind

Le diagnostic utilisateur est confirme par le code et par un test rouge ajoute avant correction:

```text
tests/test_s5f_blind_progressive_results.py::test_s5f_blind_result_is_emitted_while_later_blind_work_continues
```

Avant correction, `blind-A` terminait, `blind-B` restait bloque, et `progress_sink` n'avait encore recu aucun resultat:

```text
assert emitted == [("blind-A", 1)]
E AssertionError: assert [] == [('blind-A', 1)]
```

## 2. Chronologie du run 200 fichiers

Run utilisateur:

```text
Near: 100 SOLVED, emis progressivement
Blind: 88 SOLVED, 12 FAILED
Blind first success: 22:17:56
Blind final solve: 23:37:34
GUI Blind emissions: 23:37:34.793 -> 23:37:35.321
```

La rafale finale venait du buffering batch, pas du solveur Blind.

## 3. Cause exacte

Dans `zesolver/core/batch/runner.py`, Near appelait `_run_phase(..., on_result=_near_completed)`.

Blind appelait `_run_phase()` sans callback, puis emettait les resultats seulement apres le retour complet de toutes les futures:

```python
blind_phase = self._run_phase("blind", requests_by_index=unresolved, batch_request=batch_request)
for idx in tuple(unresolved):
    result = blind_phase.get(idx)
    self._emit(...)
```

`_run_phase()` ne revient qu'apres la fin de la phase, donc le GUI ne pouvait rien recevoir pendant Blind.

## 4. Test rouge avant correction

Le test rouge utilise des `threading.Event`, pas des sleeps fragiles:

```text
blind-A termine
blind-B reste non termine
attendu: result_callback(A) deja appele
observe avant patch: aucun callback
```

## 5. Architecture de callback choisie

Ajout d'un helper unique `_record_terminal()`:

```text
enregistre final[idx]
ajoute emitted
marque emitted_indices
appelle _emit()
ignore tout doublon
```

Ajout de `_blind_completed(index, result)` passe a:

```python
self._run_phase(
    "blind",
    requests_by_index=unresolved,
    batch_request=batch_request,
    on_result=_blind_completed,
)
```

## 6. Comportement Near inchange

Near conserve le meme callback et le meme routage:

```text
Near SOLVED -> terminal GUI immediat
Near unresolved -> garde pour fallback Blind, sans affichage Near-failed premature
Near CANCELLED -> terminal GUI immediat
```

## 7. Comportement Blind avant/apres

Avant:

```text
Blind result ready
attente fin de tous les Blind
emission en rafale finale
```

Apres:

```text
Blind result ready
_blind_completed()
progress_sink
PipelineGuiRunner.on_progress
result_callback
signal Qt / ligne GUI
phase Blind continue
```

## 8. Progression et compteurs

Tests automatises:

```text
2 Near SOLVED + 3 Blind -> completed progresse 1,2,3,4,5
```

Validation reelle 1 normal + 3 FAKE_HINT:

```text
Near callback: +9.426s
Blind callbacks: +55.175s, +94.524s, +125.548s
```

Il n'y a plus de rafale terminale.

## 9. preserve_order

Contrat preserve:

```text
callbacks GUI = ordre reel de terminaison
BatchSolveResult.results = ordre d'entree quand preserve_order=True
```

Test automatise couvert dans `test_s5f_callbacks_use_completion_order_but_final_result_preserves_input_order`.

## 10. Stop

Stop pendant Blind:

```text
premier resultat conserve
restants backfill CANCELLED
aucun doublon
```

## 11. stop_on_error

Apres un echec Blind:

```text
echec emis immediatement
pending futures annulees
restants synthetiques coherents
aucun doublon
```

## 12. Absence de doublons

Le nouvel ensemble `emitted_indices` borne l'emission terminale a une seule fois par index.

`PipelineGuiRunner.emitted_paths` continue de proteger la couche GUI et le backfill final.

## 13. Validation GUI reelle

Protocole sur copies temporaires sans WCS:

```text
profil S50: focal=250 mm, pixel=2.90 um, scale=2.39"/px
bibliotheque: /home/tristan/ZeSolverCatalog/new
1 FITS normal
3 FITS FAKE_HINT connus
workers Near=6
worker Blind=1
```

Resultats:

```text
normal M106 233459 -> SOLVED NEAR, inliers=53, rms=0.3005
M106 234013 FAKE_HINT -> SOLVED BLIND4D, inliers=49, rms=1.1257, callback +55.175s
M31 230409 FAKE_HINT -> SOLVED BLIND4D, inliers=57, rms=0.6538, callback +94.524s
M106 233459 FAKE_HINT -> SOLVED BLIND4D, inliers=60, rms=0.5749, callback +125.548s
WCS written: 4/4
```

Telemetry:

```text
blind_result_ready -> blind_result_emitted lag ~= 0.00019..0.00029s
```

## 14. Verification du warning de couverture

Le run 4 fichiers a revele un residu S5E:

```text
resources before fix:
blind4d_index_count=48
blind4d_all_sky=False
warnings=['blind4d_coverage_not_all_sky']

runtime final:
blind4d_index_count=47
blind4d_covered_tiles=1476
blind4d_total_tiles=1476
blind4d_all_sky=True
```

Correction:

```text
_resources_from_library() publie maintenant la vue Blind4D runtime finale
monolithe compatibility exclu quand fixed32 complet est valide
warnings calcules apres cette vue finale
```

Validation post-correctif:

```text
resources: index_count=47, coverage=1476/1476, all_sky=True, warnings=[]
selection warnings=()
Near result warnings=()
Blind result warnings=()
```

## 15. Validation manuelle S5E

Nouveau processus:

```text
last_preset_id=seestar_s50
focal_mm=250.0
pixel_um=2.9
resolution=1080x1920
catalog_library_path=/home/tristan/ZeSolverCatalog/new
```

Le store utilisateur n'avait pas encore de cache de verification actif. Une verification FULL structuree a ete enregistree via les APIs S5E, puis relue:

```text
message=Vérifiée — cache valide
cache_reused=True
payload_hash_count=0
blind4d_index_count=47
covered_tiles=1476
total_tiles=1476
all_sky=True
```

## 16. Tests cibles

Ajouts:

```text
tests/test_s5f_blind_progressive_results.py
tests/test_catalog_library_blind4d_product_switch.py::test_s5f_catalog_resources_publish_final_library_view_coverage_without_compat_warning
```

Couverture:

```text
Blind progressif minimal
succès + échec
batch mixte
ordre final preserve_order
Stop pendant Blind
stop_on_error
PipelineGuiRunner
Qt offscreen event loop
telemetrie blind_result_ready/emitted
warning all-sky absent avec monolithe compatibility present
```

## 17. Barrieres

```text
.venv/bin/python tools/check_core_boundaries.py
-> core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
-> PASS, 644 passed, 1 skipped, 9 deselected

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
-> 644 passed, 10 skipped

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
-> OK

git diff --check
-> OK
```

## 18. Fichiers modifies

S5F:

```text
zesolver/core/batch/runner.py
zesolver/catalog_resources.py
tests/test_s5f_blind_progressive_results.py
tests/test_catalog_library_blind4d_product_switch.py
docs/stabilization/s5f_blind_gui_progressive_results_report_20260725.md
```

Changements S5E deja presents au depart et conserves:

```text
AGENT.md
zesolver.py
zesolver/gui_pipeline/settings_adapter.py
zesolver/gui_settings_sections.py
zesolver/settings_store.py
zesolver/catalog_library/verification_cache.py
tests/test_s5e_settings_catalog_persistence.py
docs/stabilization/s5e_settings_catalog_verification_persistence_report_20260725.md
```

## 19. Etat Git final

```text
## test...origin/test
 M AGENT.md
 M tests/test_catalog_library_blind4d_product_switch.py
 M zesolver.py
 M zesolver/catalog_resources.py
 M zesolver/core/batch/runner.py
 M zesolver/gui_pipeline/settings_adapter.py
 M zesolver/gui_settings_sections.py
 M zesolver/settings_store.py
?? docs/stabilization/s5e_settings_catalog_verification_persistence_report_20260725.md
?? docs/stabilization/s5f_blind_gui_progressive_results_report_20260725.md
?? tests/test_s5e_settings_catalog_persistence.py
?? tests/test_s5f_blind_progressive_results.py
?? zesolver/catalog_library/verification_cache.py
```

## 20. Prochaine etape

```text
P3B-1E — integration de distribution officielle des Bibliotheques ZeSolver
```

## 21. Decision de gate

```text
READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
```

