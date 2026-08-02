AGENT.md — État courant ZeSolver

Projet : ZeSolverÉcosystème : ZeMosaic / ZeSeestarStackerAuteur principal : Tinystork — Tristan NauleauMise à jour : 1er août 2026Phase active : Release Candidate Acceptance

READY_FOR_RELEASE_CANDIDATE_ACCEPTANCE
NOT_YET_PRODUCTION_READY

Portée

Ce fichier s’applique à tout le dépôt, sauf instruction plus précise dans unsous-répertoire.

Il décrit uniquement l’état courant, les invariants et la prochaine étape. Lesrapports détaillés de mission restent dans docs/stabilization/ et font foi pourl’historique technique.

Ne pas rouvrir un chantier fermé sans régression reproduite et documentée.

État validé

Les chantiers d’architecture, de stabilisation du pipeline et d’intégration desbibliothèques sont fermés.

Éléments produit désormais intégrés :

CatalogLibrary et gestionnaire de bibliothèques ;

distribution officielle multi-source avec reprise ;

installation officielle, paquet local et bibliothèque existante ;

D50 fixed32 en 47 shards avec recherche progressive ;

wizard relançable et activation transactionnelle ;

modes produit auto/auto après adoption d’une bibliothèque ;

thème système / clair / sombre ;

pipeline batch Near puis Blind lazy ;

persistance des réglages et vérification de bibliothèque.

Dernière validation automatisée connue :

801 passed, 36 skipped, 17 warnings
compileall zesolver zeblindsolver tools : OK
git diff --check : OK

Validation macOS CI connue :

Audit statique macOS : passed
CI macOS automatisée : passed sur macos-26-arm64 / arm64, Python 3.11.9
Validation runtime sur Mac physique : pending
Paquet public macOS : non construit

Dernière validation Linux réelle connue :

Bibliothèque : /home/tristan/ZeSolverCatalog/new
Statut : READY_FULL
Near : astap-native depuis la bibliothèque
Blind 4D : library-view
Index Blind 4D : 47
Tuiles couvertes : 1476 / 1476
Couverture ciel complet : true
Fallback manifeste externe : false
Batch Near : succès
Batch Blind : succès

Contrat produit courant

Bibliothèque ZeSolver complète

Après sélection ou installation réussie d’une bibliothèque READY_FULL :

catalog_library_path=<racine contenant catalog.json>
near_catalog_mode=auto
blind4d_catalog_mode=auto

Le runtime doit résoudre :

ZeNear -> ressources ASTAP de la Bibliothèque ZeSolver
ZeBlind 4D -> library-view

État Blind 4D attendu :

blind4d_catalog_mode_effective=library-view
blind4d_index_count=47
blind4d_covered_tiles=1476
blind4d_total_tiles=1476
blind4d_all_sky=true
blind4d_external_fallback_used=false

Le monolithe direct-d50 reste un artefact de compatibilité. Les 47 shardsfixed32 valides sont le chemin runtime normal.

Wizard

Les parcours suivants doivent converger vers la même activation produit :

bibliothèque existante ;

installation officielle ;

paquet local.

Le wizard ne peut être marqué terminé que si la transaction principale aréellement réussi. Une erreur ne doit jamais être suivie d’un succès silencieuxou d’une seconde sauvegarde contradictoire.

Parcours ASTAP seul

Sans bibliothèque complète :

near_catalog_mode=astap-native
blind4d_catalog_mode=auto

Compatibilité avancée

Les modes historiques restent disponibles uniquement sur demande explicite :

near_catalog_mode=legacy-index
blind4d_catalog_mode=external-manifest

Un ancien chemin de manifeste externe peut rester mémorisé, mais ne doit être nivalidé ni utilisé lorsque blind4d_catalog_mode=auto.

Invariants techniques

Ne pas modifier les algorithmes ZeNear ou ZeBlind sans mission explicite.

Ne pas relâcher les critères scientifiques (quality_inliers, quality_rms,match_radius, code_tol) pendant une mission GUI, packaging ou documentation.

Conserver le routage Near puis Blind lazy.

Ne pas reconstruire D50 ni les 47 shards fixed32 sans mission dédiée.

Ne pas supprimer le monolithe ni les shards existants sans plan de migration.

Ne pas accepter un manifeste invalide pour masquer une erreur de configuration.

Ne pas réactiver automatiquement un rollback historique au démarrage.

Ne pas bloquer le thread Qt avec des hashes complets, des chargements NPZlourds ou une opération réseau longue.

Préserver l’annulation et la reprise des opérations longues quand leur contratle prévoit.

Règles Git obligatoires

Ne jamais supposer que la branche locale est synchronisée avec le remote.

Avant toute mission :

git status --short
git branch -vv
git rev-parse HEAD
git rev-parse origin/test
git log --oneline --decorate -10
git diff --check

Instantané connu après P3B-1F : la branche locale test était en avance de deuxcommits sur origin/test :

5858b51 Close startup wizard catalog activation transaction
43fa8e6 Add system light and dark theme selector

Toujours revérifier cet état : ce n’est pas un invariant.

Sans demande explicite de Tristan :

ne pas pousser ;

ne pas merger vers main ;

ne pas créer de tag ou de Release ;

ne pas réécrire l’historique ;

ne pas utiliser push --force, reset --hard ou un rebase destructif ;

ne pas écraser des changements non commités existants.

Si l’état Git ne correspond pas littéralement à une mission, le consigner dansle rapport au lieu de modifier l’historique.

Publication `test` -> `main`

Règle de référence :

docs/maintenance/main_publication_workflow.md

Le dépôt GitHub reste unique, mais les deux branches ont des rôles strictement séparés.

`test` est la seule branche de développement et la source de vérité. Elle conserve le code complet, les tests, les outils, les rapports, la documentation technique et les scripts de publication.

`main` est une distribution publique générée. Elle contient uniquement le runtime ZeSolver, les ressources nécessaires, les métadonnées d’installation, la documentation utilisateur essentielle et les mentions légales.

Ne jamais développer directement sur `main`.

Ne jamais corriger directement sur `main`.

Ne jamais faire `git merge test` depuis `main`.

Toute contribution, même minuscule, cible d’abord `test`.

Toute publication vers `main` passe par :

1. correction et validation sur `test` ;
2. commit et push de `test` ;
3. génération via `tools/build_public_tree.py` ;
4. synchronisation vers `/home/tristan/.openclaw/workspace/projects/ZeSolver-main` ;
5. contrôle du diff ;
6. commit et push de `main` uniquement après validation explicite de Tristan.

Le contenu public est défini par :

packaging/public_manifest.txt

Le générateur officiel est :

tools/build_public_tree.py

Le script d'orchestration sûr pour préparer une candidate `main` locale est :

tools/prepare_public_main.sh

Usage minimal :

tools/prepare_public_main.sh --dry-run

Puis, après validation du dry-run :

tools/prepare_public_main.sh

Ce script vérifie les deux worktrees, génère et valide la projection publique,
demande confirmation avant de synchroniser vers ZeSolver-main, puis affiche les
commandes manuelles de revue, commit et push. Il ne commit jamais, ne pousse
jamais et ne merge jamais `test` dans `main`.

Ne jamais publier depuis un `test` sale ou non poussé.

Ne jamais utiliser `push --force` sur `main`.

Phase active — Release Candidate Acceptance

Aucun nouveau chantier architectural ne doit être ouvert avant fermeture du gateRelease Candidate.

Gate Windows — bloquant

Valider le paquet réellement destiné aux utilisateurs, depuis un profil vierge :

installation ou extraction propre ;

premier lancement et wizard ;

installation ou sélection de la bibliothèque ;

fermeture puis redémarrage ;

statut READY_FULL ;

batch Near réussi ;

fallback Blind 4D réussi ;

reprise d’un téléchargement interrompu ;

thème système / clair / sombre persistant ;

chemins contenant des espaces ;

Stop et annulation propres ;

aucune traceback ni console Python inattendue.

Audit macOS — requis

Sans machine macOS physique, le verdict autorisé est :

MACOS_COMPATIBILITY_AUDIT_PASSED

et non :

MACOS_RUNTIME_VALIDATED

L’audit doit couvrir : build PyInstaller, icône .icns, plugins Qt, chemins,permissions, comportement multiprocessing avec spawn et absence de dépendanceLinux ou Windows implicite.

Tant qu’aucun essai humain n’a été fait sur Mac, la documentation publique doitindiquer que macOS est expérimental ou non validé sur machine physique.

Documentation et publication

Avant promotion vers main :

mettre à jour README.md et CHANGELOG.md ;

vérifier crédits et licences ;

documenter installation, premier lancement, bibliothèque et dépannage ;

construire et inspecter les artefacts publics ;

exécuter les tests finaux sur le commit exact à promouvoir ;

rédiger le rapport Release Candidate Acceptance.

Le site ZeSoftware peut être développé en parallèle, mais ses liens publicsdoivent pointer vers des Releases et artefacts stables.

Points de contrôle résiduels

À vérifier pendant le gate final, sans lancer une refonte générale :

absence de contradiction entre un avertissement précoce de couverture Blind 4Dpartielle et la télémétrie finale 1476 / 1476, all_sky=true ;

RC-GPU-1 ferme le bruit principal : en l’absence permanente de CuPy, ZeNear
désactive CUDA une seule fois pour le batch et continue directement sur CPU. La
gestion GPU reste optionnelle, guidée seulement en environnement source
explicitement modifiable, et le packaging GPU frozen reste à définir.

cohérence des caches FAST quand le chemin canonique change ;

maintien de FULL comme route explicite pour les hashes complets.

Critères de promotion vers main

La promotion test -> main est autorisée seulement après :

branche test poussée et état Git partagé ;

gate Windows réussi sur l’artefact final ;

audit macOS terminé et limites documentées ;

suite globale verte ;

compileall vert ;

git diff --check propre ;

README et changelog à jour ;

contenu du paquet vérifié ;

aucun fichier sensible, local ou temporaire inclus ;

rapport Release Candidate Acceptance rédigé.

Verdict attendu avant promotion :

RELEASE_CANDIDATE_ACCEPTANCE_PASSED
READY_TO_PROMOTE_TEST_TO_MAIN

Verdict autorisé après promotion, tag et validation des artefacts publiés :

PRODUCTION_READY_FOR_PUBLIC_BETA

Ne pas annoncer PRODUCTION_READY avant la fermeture effective de ces gates.

Référence immédiate

Rapport P3B-1F :

docs/stabilization/p3b1f_startup_wizard_catalog_activation_report_20260801.md

Commit P3B-1F :

5858b51884518c41fd2773fae99a9bd37eaa66bf
Close startup wizard catalog activation transaction
