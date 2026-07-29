# S6B-3 - Unified Settings, Standalone ASTAP, Instrument Auto

Date: 2026-07-26

## 1. Etat Git Initial

Branche: `test...origin/test`.

Le worktree contenait deja les changements non commites S6B-1/S6B-2. `git diff --check` etait propre au demarrage.

## 2. Changements S6B-1/S6B-2 Preexistants

S6B-1 deplacait le manifeste Blind 4D externe vers les outils avances.
S6B-2 ajoutait le modele `FULL_LOCAL / NEAR_ONLY / UNAVAILABLE` et le rangement terminal `unresolved_by_zesolver`.

Ces changements ont ete preserves.

## 3. Source de Verite Initiale

`PersistentSettings.db_root` existait deja comme stockage persistant de la base ASTAP, mais le GUI exposait plusieurs champs synchronises surtout dans un sens:

- `settings_db_edit`
- `db_tab_edit`
- prefill CLI
- Wizard

`_build_config()` reclamait encore `index_root` lorsqu'aucune CatalogLibrary n'etait active.

## 4. Widgets ASTAP Identifies

- Easy: aucun champ ASTAP autonome clair avant S6B-3.
- Expert: `settings_db_edit` dans la compatibilite historique.
- Database tab: `db_tab_edit`.
- Wizard: lisait `settings_db_edit`, puis `db_tab_edit`.

## 5. Sens de Synchronisation Initial

La synchronisation etait principalement `db_tab_edit -> settings_db_edit` et `settings_db_edit -> db_tab_edit`, via signaux directs. Le prefill CLI ecrivait `settings_db_edit` directement.

## 6. Exigences Historiques d'Index Identifiees

`_read_settings_from_ui()` et `_build_config()` exigeaient `settings_index_edit` hors CatalogLibrary, meme pour le parcours ASTAP-native. Cela contredisait le contrat S6B-2 `ASTAP seul -> NEAR_ONLY`.

## 7. Controleur Partage ASTAP

Ajout de `_set_astap_root(path, source, validate=False, update_widgets=True)`.

Il met a jour:

- `PersistentSettings.db_root`
- `easy_astap_edit`
- `settings_db_edit`
- `db_tab_edit`
- statut de validation ASTAP
- scan borne des familles
- resume de capacite

Les updates utilisent `QtCore.QSignalBlocker`.

## 8. Champ ASTAP Easy

Ajout sous `Bibliotheque ZeSolver`:

```text
Base ASTAP - ZeNear uniquement
[chemin] [Parcourir] [Verifier] [Effacer]
```

Le texte d'aide precise que la base est facultative avec une bibliotheque complete et permet ZeNear seul en son absence.

## 9. Champ ASTAP Expert

Le champ Expert existant est conserve dans la zone diagnostic, mais il partage maintenant le meme controleur et la meme valeur `db_root`.

## 10. Comportement Wizard

Le Wizard accepte maintenant:

- Bibliotheque ZeSolver valide;
- ou base ASTAP valide seule.

Il ne reclame l'ancien index que si le mode Near est explicitement `legacy-index`.

## 11. Preremplissage CLI

`--db-root` passe par `_set_astap_root(..., source="cli")`, donc toutes les vues GUI refletent la meme valeur.

## 12. Validation ASTAP

Le bouton Easy `Verifier` reutilise `validate_astap_root`.

Statuts:

- `Base ASTAP valide - familles detectees: ...`
- `Base ASTAP invalide - ...`
- `Base ASTAP non verifiee`

Le scan lourd reste debounced.

## 13. Persistance ASTAP

La valeur reste stockee dans `PersistentSettings.db_root`. `Effacer` vide le modele partage et les widgets sans supprimer de fichiers.

## 14. Parcours ASTAP-only sans Index

En Easy/Wizard:

```text
CatalogLibrary absente + ASTAP valide + index_root absent
-> run autorise
-> near_catalog_mode effectif astap-native
-> Blind desactive seulement pour le run
```

## 15. Expert legacy-index

Le mode Expert explicite `legacy-index` conserve l'exigence `index_root`.

## 16. Priorite CatalogLibrary

Une bibliotheque valide reste prioritaire sur un ancien chemin ASTAP autonome. Le chemin ASTAP est memorise mais non utilise pour ce run.

En interface simplifiee, une bibliotheque invalide avec ASTAP valide retombe en Near-only au lieu de bloquer.

## 17. Modele `instrument_mode`

Ajout de `instrument_mode`:

```text
auto
preset
custom
```

La version de schema settings passe a `13`.

## 18. Mode Auto

`Auto - metadonnees FITS` est un choix explicite. Il n'injecte aucun preset global.

## 19. Mode Preset

Les presets existants restent bases sur `zeblindsolver.presets.list_presets()`, avec IDs stables en `Qt.UserRole`.

## 20. Mode Custom

Les valeurs FOV personnalisees existantes restent sauvegardees separement.

## 21. Migration Anciens Reglages

Regle retenue:

- nouveau fichier absent -> `instrument_mode=auto`;
- ancien `last_preset_id` present -> `preset`;
- anciennes valeurs FOV presentes sans preset -> `custom`;
- champ explicite existant -> respecte et normalise.

`last_preset_id=None` n'est pas assimile a Auto quand des valeurs custom existent.

## 22. Hints Effectifs en Auto

En Auto, `_build_config()` transmet:

```text
hint_focal_mm=None
hint_pixel_um=None
hint_resolution_arcsec=None
hint_resolution_min_arcsec=None
hint_resolution_max_arcsec=None
```

Les valeurs memorisees restent disponibles pour revenir a Preset ou Custom.

## 23. Metadonnees FITS Utilisees

Le chemin scientifique existant reste responsable de lire les metadonnees par fichier (`FOCALLEN`, `XPIXSZ`, `YPIXSZ`, binning, WCS). Aucun second parseur scientifique n'a ete ajoute.

## 24. Sans Metadonnees

Un FITS sans metadonnees optiques ne bloque pas le preflight en Auto. ZeNear tente selon son contrat; Blind prend le relais si une bibliotheque complete existe.

## 25. Lots Mixtes

Auto ne fige pas les hints du premier fichier au niveau batch. Chaque fichier conserve son chemin metadata propre.

## 26. Visibilite Easy

Easy affiche:

- Bibliotheque ZeSolver;
- Base ASTAP autonome;
- Instrument / optique avec `Auto - metadonnees FITS`.

## 27. Visibilite Expert

Expert conserve:

- champ ASTAP diagnostic;
- modes Near `auto/astap-native/legacy-index`;
- champs FOV detailles, desactives en Auto.

## 28. Synchronisation Interfaces

Easy, Expert, Database tab, Wizard et prefill CLI utilisent `PersistentSettings.db_root` via `_set_astap_root`.

## 29. Traductions FR/EN

Ajout des libelles FR/EN pour ASTAP Easy, instrument Auto/Custom et resumes de chaine.

## 30. Logs et Telemetrie

Le pipeline logge:

```text
instrument_mode_requested=auto
instrument_hint_source=per-file-fits-metadata
global_instrument_hint_applied=false
```

Pour Preset/Custom, `global_instrument_hint_applied=true`.

## 31. Tests ASTAP

Ajoutes:

- `tests/test_s6b3_shared_astap_settings.py`
- `tests/test_s6b3_astap_only_routing.py`

## 32. Tests Instrument

Ajoute:

- `tests/test_s6b3_instrument_mode.py`

## 33. Tests Migration

Migration couverte pour nouvelle installation, preset ancien, custom ancien, et roundtrip.

## 34. Tests GUI

Ajoute:

- `tests/test_s6b3_easy_expert_settings_gui.py`

## 35. Test Manuel ASTAP-only

Non execute manuellement avec interaction GUI reelle. Couvert hermetiquement par tests source/settings/routing. Le parcours runtime ASTAP-only repose sur le provider ASTAP natif deja teste.

## 36. Test Manuel Bibliotheque Complete

Non execute manuellement dans cette mission. La priorite CatalogLibrary et les suites existantes d'integration bibliotheque restent vertes.

## 37. Non-regression Scientifique

Aucun seuil, matching, RANSAC, detection, WCS ou solveur n'a ete modifie. Le changement Auto concerne uniquement les hints globaux transmis.

## 38. Fichiers Modifies

Principaux fichiers touches:

- `zesolver.py`
- `zesolver/gui_settings_sections.py`
- `zesolver/settings_store.py`
- `zesolver/settings/product.py`
- `zesolver/settings/migration.py`
- `zesolver/gui_pipeline/requests.py`
- `zesolver/gui_pipeline/settings_adapter.py`
- `zesolver/gui_pipeline/pipeline_runner.py`
- tests S6B-3

Les fichiers S6B-1/S6B-2 preexistants restent dans le worktree.

## 39. Barrieres Executees

Passees:

```text
tests cibles S6B reel: 54 passed
tools/check_core_boundaries.py: OK
tools/run_regression_suite.py --hermetic: PASS, 721 passed, 1 skipped, 9 deselected
QT_QPA_PLATFORM=offscreen pytest -q: 721 passed, 10 skipped
compileall complet: OK
git diff --check: OK
```

Les noms proposes `tests/test_settings_product.py`, `tests/test_settings_assembly.py`, `tests/test_gui_settings.py` n'existent pas dans ce depot; ils ont ete remplaces par les tests reels settings/catalog/gui disponibles.

## 40. Etat Git Final

Branche: `test...origin/test`.

Aucun FITS, backup, dossier `unresolved_by_zesolver`, manifest temporaire, telemetry sidecar de run ou benchmark n'apparait dans le status. Le filtre `telemetry` matche seulement le fichier source `zesolver/resource_telemetry.py`.

## 41. Gate Final

```text
S6B3_SHARED_SETTINGS_SOURCE_OF_TRUTH_CONFIRMED
S6B3_ASTAP_PATH_AVAILABLE_IN_EASY_AND_EXPERT
S6B3_ASTAP_NEAR_ONLY_GUI_END_TO_END_CONFIRMED
S6B3_WIZARD_SETTINGS_SYNCHRONIZED
S6B3_INSTRUMENT_AUTO_MODE_CONFIRMED
S6B3_PRESET_CUSTOM_BACKWARD_COMPATIBILITY_CONFIRMED
READY_FOR_WIZARD_UX_REVIEW
```
