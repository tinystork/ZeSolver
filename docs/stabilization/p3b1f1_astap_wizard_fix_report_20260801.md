# P3B-1F1 - Correctif ASTAP du wizard relance

Date: 2026-08-01
Branche: `test`

## Probleme reproduit

Le parcours ASTAP du nouvel assistant pouvait refuser `Terminer` avec:

`Validez la base ASTAP avant de terminer.`

Le symptome etait local au wizard. Le meme chemin ASTAP restait utilisable via
l'interface avancee, donc le correctif n'a pas modifie ZeNear, ZeBlind, les
catalogues, ni les algorithmes.

## Cause

Le wizard conservait seulement un booleen global `_operation_completed`.
Il ne liait pas le succes de validation a:

- l'operation exacte (`astap`, bibliotheque existante, installation officielle);
- le chemin exact visible dans le champ;
- la valeur saisie au moment du lancement du worker.

Ce modele pouvait refuser un ASTAP valide dans certains parcours relances, ou
au contraire laisser survivre un ancien succes apres modification du champ.

## Correctif

`zesolver/gui_startup_wizard.py` maintient maintenant un etat explicite:

- `_validated_astap_path`;
- `_validated_library_path`;
- `_completed_operation`;
- `_completed_operation_signature`;
- `_active_operation_signature`.

Le bouton `Parcourir` et la saisie manuelle passent par les memes `QLineEdit`.
Tout changement de texte invalide le succes precedent si le chemin ne correspond
plus au chemin valide.

Le clic sur `Terminer` relit le choix courant et le chemin courant. Pour ASTAP,
il accepte uniquement si le chemin visible est exactement le chemin valide. La
selection ASTAP n'est emise vers la fenetre principale qu'au clic `Terminer`,
ce qui evite de modifier les anciens reglages apres une validation suivie d'une
fermeture ou d'un `Annuler`.

## Persistance

Le chemin ASTAP autonome est applique par le handler existant de la fenetre
principale apres le signal `astapSelected`. Le wizard n'ecrit donc pas un
nouveau `db_root` pendant la simple validation du worker.

En cas d'annulation ou de fermeture sans terminer:

- l'ancien `db_root` reste intact;
- l'ancien `index_root` reste intact;
- l'ancienne bibliotheque selectionnee reste intacte.

En cas de fin reussie ASTAP:

- la selection Bibliotheque ZeSolver est nettoyee;
- `db_root` prend le chemin ASTAP valide;
- `near_catalog_mode` passe par le chemin ASTAP native/Near-only existant;
- le wizard est marque termine.

## Legacy

Le correctif ne reactive pas le dialogue legacy `Nouvelles bases detectees`.
Ce dialogue reste garde par `should_allow_legacy_family_prompt` et ne doit pas
interrompre le parcours produit normal.

## Tests ajoutes

`tests/test_startup_wizard.py` couvre maintenant:

- selection ASTAP par `Parcourir`;
- saisie manuelle;
- validation reussie puis `Terminer`;
- changement de chemin apres validation;
- chemin invalide;
- relance depuis le menu avec le meme constructeur de wizard;
- persistance uniquement apres succes final;
- absence de modification des anciens reglages apres `Annuler`;
- parcours ASTAP Near-only sans Bibliotheque ZeSolver;
- suppression du prompt legacy en mode produit normal.

## Validation Linux

Commandes executees:

```bash
.venv/bin/python -m pytest -q tests/test_startup_wizard.py
.venv/bin/python -m pytest -q tests/test_startup_wizard.py tests/test_settings_persistence.py tests/test_gui_catalog_library_control.py tests/test_gui_catalog_resource_type_validation.py tests/test_gui_catalog_library_manager.py
```

Resultats:

- `24 passed` pour le fichier wizard cible;
- `37 passed` pour le paquet wizard/settings/catalog GUI.

## Validation Windows demandee

Sur la machine Windows:

1. relancer le wizard depuis `Interface`;
2. choisir `Utiliser une base ASTAP existante - ZeNear uniquement`;
3. selectionner la racine ASTAP valide avec `Parcourir`;
4. lancer la validation du parcours ASTAP;
5. cliquer `Terminer`;
6. verifier que le wizard se ferme sans message `Validez la base ASTAP avant de terminer.`;
7. verifier que le mode Near-only apparait;
8. lancer un petit solve;
9. redemarrer ZeSolver et verifier que le chemin ASTAP est conserve;
10. revenir ensuite a la Bibliotheque ZeSolver existante sans telechargement.

## Limites

La validation Windows manuelle reste a faire sur l'installation qui reproduisait
le defaut. Le correctif a ete valide sous Linux avec Qt offscreen et faux chemins
de test, sans base ASTAP Windows reelle.

Statut: `READY_FOR_P3B1F1_WINDOWS_VALIDATION`
