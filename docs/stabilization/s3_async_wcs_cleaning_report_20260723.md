# S3 async WCS cleaning report - 2026-07-23

## 1. Objectif

Rendre le nettoyage WCS integre du mode simple non bloquant pour le thread Qt principal, sans modifier `zewcscleaner.process_fits()` ni la semantique de nettoyage FITS.

## 2. Etat Git initial

Commande initiale:

```text
git status --short --branch
## test...origin/test
 M AGENT.md
 M zesolver.py
 M zesolver/catalog_library/__init__.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? docs/stabilization/p3b1c_development_surface_reorganization_report_20260722.md
?? docs/stabilization/p3b1d_catalog_library_manager_report_20260722.md
?? tests/test_catalog_library_management_service.py
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_catalog_library_manager.py
?? tests/test_gui_development_surface_reorganized.py
?? tests/test_gui_development_tab_audit.py
?? zesolver/catalog_library/management.py
```

`git diff --check` etait deja en echec sur `AGENT.md` lignes 3-7, espaces finaux preexistants.

## 3. Reproduction

Par inspection du parcours existant:

1. point d'entree: `_start_solving()` appelle `_run_simple_mode_wcs_cleaning(self._pending_files)`;
2. thread: appel direct depuis le slot GUI du bouton Resoudre, donc thread Qt principal;
3. boucle: `_run_simple_mode_wcs_cleaning()` itere les FITS cibles;
4. appel long: chaque fichier appelle directement `zewcscleaner.process_fits(...)`;
5. mises a jour GUI: `files_view.setUpdatesEnabled`, `_apply_item_status`, `QMessageBox`, `_log`;
6. progression: pas de vraie progression dediee, seulement rafraichissement ponctuel de lignes;
7. erreur: premiere exception fichier affiche une warning box et abort le demarrage;
8. fermeture: aucune annulation possible pendant cette boucle synchrone;
9. Stop: inoperant pendant cette etape, car le run solver n'est pas encore cree et l'event loop est occupee;
10. demarrage solveur: seulement apres retour `True` de la boucle synchrone.

## 4. Cause racine

Les operations FITS longues etaient executees dans le thread Qt principal. L'appel ponctuel a `QApplication.processEvents(...ExcludeUserInputEvents)` ne permettait pas une interface interactive ni un Stop coherent.

## 5. Architecture choisie

Ajout d'un worker Qt dedie `WcsCleanupRunner`, sous-classe de `QThread`, dans `zesolver/gui_wcs_cleanup.py`.

Flux:

```text
thread Qt principal -> WcsCleanupRunner QThread -> process_fits sequentiel
                    <- signaux structures: started/progress/result/error/completed/cancelled
thread Qt principal -> widgets, journal, statuts WCS, puis reprise solveur si succes
```

## 6. Fichiers modifies

- `zesolver/gui_wcs_cleanup.py`: nouveau contrat immuable, evenements structures, runner Qt.
- `zesolver.py`: orchestration GUI async, Stop/fermeture, progression et reprise solveur apres succes.
- `tests/test_gui_wcs_cleanup_worker.py`: tests worker/concurrence/FITS reel.
- `docs/stabilization/s3_async_wcs_cleaning_report_20260723.md`: ce rapport.

## 7. Comportement avant/apres

Avant: boucle FITS synchrone dans `_run_simple_mode_wcs_cleaning()`, GUI bloquee, Stop inutile.

Apres: `_run_simple_mode_wcs_cleaning()` demarre un worker et retourne immediatement `started`; les widgets sont mis a jour par slots Qt dans le thread principal; le solveur reprend uniquement apres `completed`.

## 8. Contrat du worker

`WcsCleanupConfig` est une dataclass gelee contenant: liste ordonnee de FITS, `dry_run`, `backup`, `only_if_wcs`, `all_hdus`.

Pour chaque fichier: controle annulation avant fichier, appel `process_fits()`, emission `WcsCleanupFileResult`, puis `WcsCleanupProgress`.

## 9. Cycle de vie Qt

Le GUI garde `_wcs_cleanup_active_id` pour ignorer les callbacks tardifs. A la fin, le thread emet un seul terminal, puis `finished` declenche `deleteLater()`, libere `_wcs_cleanup_worker`, et remet le GUI disponible ou relance le solveur.

## 10. Politique d'annulation

Annulation cooperative: Stop pose l'event d'annulation; le fichier courant finit; aucun nouveau FITS ne demarre; terminal `cancelled`; retour a l'etat pret.

## 11. Gestion des erreurs

Politique conservee: une exception fichier arrete toute l'etape et empeche le solveur. L'erreur journalisee contient chemin, operation `process_fits`, message et statut final.

## 12. Tests cibles

Commandes executees:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_wcs_cleanup_worker.py tests/test_gui_wcs_cleanup_refresh.py -q
6 passed, 1 warning

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_*wcs* tests/test_*clean* -q
7 passed, 1 warning
```

Couverture ajoutee: multi-fichiers, ordre, progression fichier par fichier, resultat structure, terminal unique, exception fichier, annulation entre fichiers, thread termine, `process_fits` hors thread principal, slot GUI dans thread principal, event Qt traite pendant nettoyage.

## 13. Controle FITS et pixels

Test reel local sur FITS synthetique avec `PRIMARY` + extension:

- pixels identiques;
- dimensions/HDU conserves;
- WCS PRIMARY retire;
- WCS extension conserve avec `all_hdus=False`;
- fichier relisible par Astropy;
- `.bak` cree et valide avec WCS original.

## 14. Suite hermetique

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
status: PASS
601 passed, 1 skipped, 9 deselected, 59 warnings
```

## 15. Suite complete

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
601 passed, 10 skipped, 59 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
PASS
```

Skips complets: corpus/env externes non configures (`ZESOLVER_ZN310B_ROOT`, base ASTAP locale, S50, Blind4D manifest, corpus regression).

## 16. Validation graphique reelle ou instructions

Validation graphique reelle non effectuee dans cette session. `QT_QPA_PLATFORM=offscreen` ne vaut pas validation graphique reelle.

Instructions pour Tristan:

1. lancer le GUI dans une vraie session desktop;
2. charger le lot significatif des 100 FITS;
3. activer mode simple + nettoyage WCS avant run;
4. lancer et deplacer/redimensionner la fenetre pendant le nettoyage;
5. changer d'onglet;
6. verifier progression fichier par fichier et compteur restant;
7. verifier que les statuts WCS changent apres nettoyage;
8. cliquer Stop au milieu: le fichier courant doit finir, aucun suivant ne doit demarrer;
9. relancer ensuite un run;
10. fermer pendant un nettoyage et verifier absence de crash ou warning `QThread: Destroyed while thread is still running`.

## 17. Warnings

Warnings de tests existants: `datetime.utcnow()` dans `zewcscleaner.py`/`db_convert.py`, warnings multiprocessing fork multi-thread, un VerifyWarning Astropy. Aucun nouveau warning bloquant.

## 18. Tests non executes

Validation graphique reelle non executee faute de session graphique utilisateur et lot reel de 100 FITS accessible/selectionne pendant cette intervention.

## 19. Etat Git final

```text
## test...origin/test
 M AGENT.md
 M zesolver.py
 M zesolver/catalog_library/__init__.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? docs/stabilization/p3b1c_development_surface_reorganization_report_20260722.md
?? docs/stabilization/p3b1d_catalog_library_manager_report_20260722.md
?? tests/test_catalog_library_management_service.py
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_catalog_library_manager.py
?? tests/test_gui_development_surface_reorganized.py
?? tests/test_gui_development_tab_audit.py
?? tests/test_gui_wcs_cleanup_worker.py
?? zesolver/catalog_library/management.py
?? zesolver/gui_wcs_cleanup.py
```

`git diff --check -- zesolver.py zesolver/gui_wcs_cleanup.py tests/test_gui_wcs_cleanup_worker.py` passe.

`git diff --check` global echoue encore uniquement sur `AGENT.md` lignes 3-7, etat deja present avant intervention.

## 20. Limites restantes

Le test graphique reel doit etre fait sur le lot utilisateur. L'annulation reste volontairement cooperative entre deux FITS, pas pendant une ecriture.

## 21. Prochaine etape recommandee

Faire la validation graphique reelle sur le lot des 100 FITS ayant reproduit le gel.

## 22. Decision de gate

Les tests automatises et barrieres generales sont verts, mais la validation graphique reelle demandee n'a pas ete effectuee.

Validation graphique réelle effectuée par Tristan le 23 juillet 2026 sur le lot utilisateur ; absence de gel confirmée ; gate promu à READY_FOR_S4_ASTAP_FAMILY_HANDLING
