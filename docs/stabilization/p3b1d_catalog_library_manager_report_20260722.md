# P3B-1D - Gestionnaire de Bibliotheques ZeSolver

Date: 2026-07-22

## Objectif

Ajouter un gestionnaire local de Bibliotheques ZeSolver sans modifier les
solveurs, profils, seuils, pixels, WCS, routage ou resultats astrometriques.

Les trois intentions utilisateur implementees:

1. installer une bibliotheque prete a l'emploi depuis un paquet local;
2. creer une bibliotheque depuis une base ASTAP existante;
3. verifier ou reparer une bibliotheque existante.

## 1. Etat Git initial

Etat au debut de P3B-1D:

```text
## test...origin/test
 M AGENT.md
 M zesolver.py
?? docs/stabilization/p3b1b_development_tab_functional_truth_audit_20260722.md
?? docs/stabilization/p3b1c_development_surface_reorganization_report_20260722.md
?? tests/test_gui_benchmark_removed.py
?? tests/test_gui_development_surface_reorganized.py
?? tests/test_gui_development_tab_audit.py
```

`git diff --check`: PASS.

Les changements P3B-1A/P3B-1B/P3B-1C etaient deja presents et n'ont pas ete
revertis.

## 2. API existantes reutilisees

- `CatalogLibrary.open()` et `CatalogLibrary.validate()`;
- `CatalogLibraryAdoptionPlan.reference_existing()`;
- `CatalogLibraryAdoptionWriter.commit()`;
- `build_blind4d_manifest_view()`;
- `build_4d_index_from_astap()`;
- `load_4d_index_manifest()`;
- `sha256_file()`;
- `iter_tiles()` et `FAMILY_SPECS` pour detecter les familles ASTAP.

## 3. Doublons evites

Le GUI ne reconstruit pas de logique catalogue.

La nouvelle fenetre appelle:

```text
LibraryManagerWindow
-> CatalogLibraryManagementService
-> adoption / writer / validation / builder 4D existants
-> CatalogLibrary
```

Le service est place dans `zesolver/catalog_library/management.py` et ne depend
pas de PySide6.

## 4. Architecture du service

Nouveau service:

```text
CatalogLibraryManagementService
```

Fonctions principales:

- `detect_astap_families(root)`;
- `create_from_astap(options)`;
- `install_package(options)`;
- `analyze_library(root)`;
- `repair_library(plan)`.

Le service expose des dataclasses simples:

- `LibraryCreateOptions`;
- `LibraryInstallOptions`;
- `LibraryOperationResult`;
- `LibraryAnalysisResult`;
- `LibraryRepairPlan`;
- `LibraryManagementProgress`.

## 5. Architecture GUI

Nouveaux composants dans `zesolver.py`:

- `CatalogLibraryManagementWorker`;
- `LibraryManagerWindow`;
- action menu `catalog_library_manager_action`;
- bouton `settings_catalog_library_manage_btn`.

Les deux acces ouvrent la meme instance:

```text
Reglages -> Bibliotheque ZeSolver -> Gerer les bibliotheques...
Outils avances -> Gestionnaire de Bibliotheques ZeSolver...
```

Aucun worker n'est cree a l'ouverture simple du gestionnaire.

## 6. Parcours Installer

P3B-1D implemente l'installation depuis un paquet local:

- dossier package;
- archive zip;
- archive tar.

Le paquet doit contenir une metadata:

```text
zesolver-library-package.json
```

ou:

```text
library_package.json
```

La bibliotheque est attendue dans `library/` ou directement a la racine si
`catalog.json` y est present.

## 7. Parcours Creer

Le parcours standard demande uniquement:

- source ASTAP;
- destination;
- politique de stockage: reference ou copie.

Les familles sont detectees automatiquement depuis la racine ou ses
sous-dossiers immediats.

Le mode Personnalise reste replie par defaut et expose seulement des valeurs
supportees par le builder direct:

- magnitude max;
- etoiles max;
- quads max;
- workers.

Les reglages historiques S/M/L ne sont pas reutilises pour construire la
Bibliotheque ZeSolver 4D.

## 8. Parcours Verifier/Reparer

`analyze_library()` affiche des diagnostics lisibles:

- `catalog.json`;
- sources ASTAP;
- ZeNear;
- Blind 4D;
- couverture globale Blind 4D;
- integrite.

Le plan de reparation minimal reconstruit Blind 4D uniquement lorsqu'une source
Near ASTAP existe et que Blind 4D manque.

## 9. Format du paquet local

Champs metadata requis:

- `library_id`;
- `version`;
- `format_version`;
- `astap_families`;
- `near_coverage`;
- `blind4d_coverage`;
- `all_sky_blind4d`;
- `installed_size_bytes`;
- `sha256`;
- `provenance`;
- `astap_credit`;
- `astrometry_credit`;
- `license`;
- `generated_at`.

Les SHA-256 sont verifies par chemin relatif a la racine de bibliotheque.

## 10. Strategie d'installation atomique

Flux:

```text
package local
-> extraction/copie dans staging
-> verification metadata
-> controle espace disque
-> verification SHA-256
-> validation CatalogLibrary
-> validation vue Blind 4D si disponible
-> publication vers destination
-> selection automatique dans le GUI
```

En cas d'erreur, le staging est nettoye et la destination finale n'est pas
presentee comme valide.

## 11. Strategie de creation atomique

Flux:

```text
source ASTAP
-> validation source/destination
-> staging <destination>.partial-*
-> reference ou copie non destructive des bases ASTAP
-> build Blind 4D direct ASTAP
-> strict_4d_manifest.json
-> publication du staging
-> adoption CatalogLibrary
-> validation CatalogLibrary
-> validation vue Blind 4D
```

La source ASTAP n'est jamais deplacee.

## 12. Annulation

Le worker GUI appelle le service avec `cancel_callback`.

Le builder direct ASTAP recoit ce callback via `build_4d_index_from_astap`.

En cas d'annulation:

- le staging est supprime;
- aucun `catalog.json` final valide n'est cree;
- la fenetre ne laisse pas un thread actif a la fermeture.

## 13. Securite d'extraction

Le service refuse:

- chemins absolus;
- `..`;
- symlinks;
- hardlinks tar;
- chemins mal formes;
- chemins SHA absolus ou sortant de la bibliotheque.

## 14. Espace disque

`installed_size_bytes` est compare a l'espace libre du parent de destination.

Le test simule un disque plein via injection de `disk_usage`.

## 15. Persistance et selection automatique

Apres succes:

1. le gestionnaire emet `librarySelected(path)`;
2. la fenetre principale remplit `settings_catalog_library_edit`;
3. `_validate_catalog_library_from_gui()` relance la validation;
4. `PersistentSettings.catalog_library_path` est sauvegarde.

La relance GUI avec les settings sauvegardes restaure le chemin.

## 16. Couverture Blind 4D affichee

Le statut existant de Bibliotheque ZeSolver reste utilise:

- Near ASTAP disponible;
- Blind 4D disponible;
- couverture globale Blind 4D oui/non;
- couverture partielle explicite.

Le gestionnaire n'affiche jamais une couverture complete par simple presence
d'une famille ASTAP.

## 17. Fichiers modifies

Ajoutes:

- `zesolver/catalog_library/management.py`;
- `tests/test_catalog_library_management_service.py`;
- `tests/test_gui_catalog_library_manager.py`;
- `docs/stabilization/p3b1d_catalog_library_manager_report_20260722.md`.

Modifies:

- `zesolver/catalog_library/__init__.py`;
- `zesolver.py`;
- `AGENT.md` apres validation.

Fichiers deja modifies avant P3B-1D et conserves:

- `zesolver.py`;
- `AGENT.md`;
- rapports/tests P3B-1A/P3B-1B/P3B-1C non suivis.

## 18. Tests ajoutes

```text
tests/test_catalog_library_management_service.py
```

Couvre:

- detection directe et sous-dossiers;
- creation reference;
- creation copie;
- rejet source/destination qui se chevauchent;
- staging nettoye apres erreur;
- staging nettoye apres annulation;
- package dossier valide;
- package zip valide;
- SHA invalide;
- archive avec traversee de chemin;
- metadata incomplete;
- espace disque insuffisant;
- analyse bibliotheque complete;
- analyse Near-only;
- reparation Near-only vers Blind 4D.

```text
tests/test_gui_catalog_library_manager.py
```

Couvre:

- ouverture depuis Reglages;
- ouverture depuis Outils avances;
- meme instance de gestionnaire;
- trois intentions visibles;
- aucun worker au simple affichage;
- detection familles via service injecte;
- Personnalise replie par defaut;
- progression;
- analyse/reparation GUI;
- selection automatique;
- persistance;
- FR/EN;
- Easy/Expert;
- fermeture pendant worker simule.

## 19. Resultats automatises

Tests cibles:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_catalog_library_management_service.py \
  tests/test_gui_catalog_library_manager.py \
  -q
14 passed
```

GUI cible:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_catalog_library_management_service.py \
  tests/test_gui_catalog_library_manager.py \
  tests/test_gui_* \
  -q
76 passed, 1 skipped, 3 warnings
```

Barriere core:

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK
```

Regression hermetique:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
PASS
596 passed, 1 skipped, 9 deselected, 58 warnings
```

Pytest complet:

```text
.venv/bin/python -m pytest -q
596 passed, 10 skipped, 58 warnings
```

Compile:

```text
.venv/bin/python -m compileall -q \
  zeblindsolver zewcs290 zesolver tools tests \
  zesolver.py zewcscleaner.py zeindexcheck.py
PASS
```

Whitespace:

```text
git diff --check
PASS
```

## 20. Validation graphique reelle

Validation effectuee avec Qt `wayland`, pas `offscreen`.

Fixture:

- source ASTAP synthetique: 1 tuile `d50_1501.1476`;
- destination temporaire `/tmp/p3b1d-real-gui-*`;
- dossier nettoye apres validation.

Resultats:

- ouverture depuis Reglages OK;
- ouverture depuis Outils avances OK;
- meme gestionnaire pour les deux acces;
- trois intentions visibles;
- aucun worker au simple affichage;
- creation Standard en mode reference OK;
- progression observee;
- `catalog.json` cree;
- strict manifest Blind 4D cree;
- selection automatique dans le GUI principal OK;
- sauvegarde settings OK;
- relance d'une deuxieme fenetre avec chemin persiste OK;
- analyse de la bibliotheque OK;
- bibliotheque Near-only synthetique analysee;
- reparation Near-only -> Blind 4D OK;
- FR puis EN OK;
- Easy masque Outils avances et garde Reglages visible;
- Expert reaffiche Outils avances;
- fermeture pendant worker simule: `request_cancel()` + `wait()` appeles.

Synthese JSON observee:

```json
{
  "analyze_rows": 6,
  "close_active_clean": true,
  "create_exists": true,
  "custom_folded": true,
  "easy_settings_visible": true,
  "easy_tools_hidden": true,
  "en_title": "ZeSolver Library Manager",
  "expert_tools_visible": true,
  "families_detected": ["d50"],
  "opened_from_settings": true,
  "relaunch_persisted": true,
  "repair_created_4d": true,
  "repair_plan_enabled": true,
  "same_from_menu": true,
  "saved_after_create": true,
  "selected_after_create": true,
  "worker_on_open": false
}
```

## 21. Tests non executes

Non execute:

- telechargement distant officiel: hors scope P3B-1D, aucune URL officielle;
- construction complete d'une base ASTAP utilisateur: volontairement evitee;
- smoke solve FITS PIPELINE/AUTO/Stop: ressources externes catalogue/corpus non
  configurees (`ZESOLVER_*` unset).

Skips externes du pytest complet:

- `ZESOLVER_ZN310B_ROOT` non defini;
- base ASTAP/HNSKY externe `database/` absente;
- S50 index/frame non configure;
- `ZESOLVER_BLIND4D_MANIFEST` non defini;
- mapping FITS `blind4d_p29_232329` absent;
- `ZESOLVER_CORPUS_ROOT` non defini.

## 22. Warnings

Warnings connus et non introduits fonctionnellement par P3B-1D:

- `datetime.utcnow()` dans `zeblindsolver/db_convert.py`;
- `multiprocessing fork` depuis un process multi-thread;
- `astropy.io.fits.card.VerifyWarning` sur carte longue tronquee.

## 23. Etat Git final

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
?? zesolver/catalog_library/management.py
```

Aucun commit, aucun push.

## 24. Limites

- Pas de catalogue distant officiel;
- pas de telechargement reseau;
- pas de format public final de distribution au-dela du paquet local teste;
- la construction Standard groupe les tuiles detectees par famille avec le
  builder direct existant;
- la reparation automatique couvre le cas minimal Near-only -> Blind 4D.

## 25. Prochaine mission unique

P3B-1E: integrer la distribution officielle des bibliotheques lorsque les
paquets, URLs, versions, tailles et empreintes SHA-256 auront ete publies.

## 26. Decision de gate

```text
READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
```
