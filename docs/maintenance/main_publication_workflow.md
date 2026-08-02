# ZeSolver - Règle de publication `test` -> `main`

## Statut

Ce document est la règle de référence pour Tristan et Junior.

Emplacement sur la branche `test` :

```text
docs/maintenance/main_publication_workflow.md
```

Il est aussi mentionné dans `AGENT.md`.

## 1. Principe général

Le dépôt GitHub reste unique :

```text
tinystork/ZeSolver
```

Il contient deux branches aux rôles strictement séparés.

### `test`

`test` est la seule branche de développement et la source de vérité.

Elle contient le code complet, les tests, les outils, les rapports, la documentation technique et les scripts de publication.

Toute correction, évolution ou contribution doit être faite sur `test`.

### `main`

`main` est une distribution publique générée.

Elle contient uniquement le code nécessaire au fonctionnement, les ressources runtime, les métadonnées d’installation, la documentation utilisateur essentielle et les mentions légales.

Elle ne contient pas les tests, outils internes, rapports ou documents de travail.

`main` est la branche par défaut visible par les utilisateurs.

## 2. Règles impératives

1. Ne jamais développer directement sur `main`.
2. Ne jamais corriger un fichier directement sur `main`.
3. Ne jamais fusionner `test` dans `main` avec `git merge`.
4. Toute contribution tierce doit cibler `test`.
5. Toute modification publique doit d’abord être réalisée, testée et commitée sur `test`.
6. `main` doit toujours être régénérée depuis un commit propre de `test`.
7. Aucun `force push` sur `main`.
8. Le contenu public est défini par `packaging/public_manifest.txt`.
9. Le générateur officiel est `tools/build_public_tree.py`.

## 3. Organisation locale

```text
/home/tristan/.openclaw/workspace/projects/ZeSolver
```

- branche : `test`
- usage : développement et validation

```text
/home/tristan/.openclaw/workspace/projects/ZeSolver-main
```

- branche : `main`
- usage : publication publique

Pour passer de `test` à `main`, on change de dossier, pas de branche :

```bash
cd /home/tristan/.openclaw/workspace/projects/ZeSolver-main
```

Pour revenir au développement :

```bash
cd /home/tristan/.openclaw/workspace/projects/ZeSolver
```

## 4. Préconditions avant publication

Depuis `test` :

```bash
cd /home/tristan/.openclaw/workspace/projects/ZeSolver
git branch --show-current
git status -sb
git fetch origin
git pull --ff-only origin test
```

Conditions obligatoires :

```text
branche courante = test
arbre de travail propre
HEAD local = origin/test
tests terminés
aucun commit local non poussé
```

## 5. Génération de la projection publique

Contrôle sans remplacement définitif :

```bash
python tools/build_public_tree.py --source . --destination /tmp/zesolver-public-check --check-only --report /tmp/zesolver-public-check-report.json
```

Génération réelle :

```bash
python tools/build_public_tree.py --source . --destination /tmp/zesolver-public --report /tmp/zesolver-public-report.json
```

Le générateur :

- exige une branche `test` propre ;
- lit l’allowlist ;
- copie uniquement les fichiers autorisés ;
- refuse les chemins interdits ;
- écrit `ZESOLVER_SOURCE_REVISION` ;
- peut produire un rapport JSON ;
- remplace atomiquement la destination.

## 6. Synchronisation vers `main`

```bash
rsync -a --delete --exclude='.git' /tmp/zesolver-public/ /home/tristan/.openclaw/workspace/projects/ZeSolver-main/
```

Les nombreuses suppressions sont normales : elles retirent les tests, rapports, outils et documents internes.

## 7. Contrôles dans `main`

```bash
cd /home/tristan/.openclaw/workspace/projects/ZeSolver-main

git branch --show-current
git status -sb
git diff --check
git diff --stat
```

La branche doit être `main`.

Vérifier l’absence des éléments internes :

```bash
for path in tests tools reports docs/stabilization docs/architecture packaging .github AGENT.md structure.txt
do
 if [ -e "$path" ]; then
 echo "ERREUR - ENCORE PRÉSENT : $path"
 fi
done
```

Cette commande ne doit rien afficher.

Vérifier la provenance :

```bash
cat ZESOLVER_SOURCE_REVISION
```

Le fichier doit indiquer :

```text
branch=test
sha=<SHA du commit test publié>
origin_test=<même SHA>
generated_at_utc=<date UTC>
```

Vérifier l’absence de divergence :

```bash
git fetch origin
git rev-list --left-right --count main...origin/main
```

Résultat attendu :

```text
0 0
```

## 8. Publication

Après validation explicite de Tristan :

```bash
git add -A
git diff --cached --check
git diff --cached --stat
```

Puis :

```bash
SOURCE_SHA="$(awk -F= '$1=="sha"{print substr($2,1,7)}' ZESOLVER_SOURCE_REVISION)"

git commit -m "Publish minimal ZeSolver distribution from test ${SOURCE_SHA}"
git push origin main
```

Contrôle final :

```bash
git status -sb
git log -1 --oneline
```

Résultat attendu :

```text
## main...origin/main
```

## 9. Ce que voit un utilisateur

Le dépôt public reste :

```text
https://github.com/tinystork/ZeSolver
```

Comme `main` est la branche par défaut :

- un visiteur arrive automatiquement sur `main` ;
- `Code -> Download ZIP` télécharge la projection publique ;
- un clone classique se positionne automatiquement sur `main`.

```bash
git clone https://github.com/tinystork/ZeSolver.git
```

Clone limité à `main` :

```bash
git clone --branch main --single-branch https://github.com/tinystork/ZeSolver.git
```

## 10. Politique de contribution

Toute pull request doit cibler `test`.

Une pull request vers `main` doit être refusée ou redirigée vers `test`.

Même une correction minuscule suit ce chemin :

```text
corriger sur test
-> tester
-> commit et push sur test
-> régénérer la projection
-> contrôler main
-> commit et push sur main
```

## 11. Interdictions

Ne jamais exécuter depuis `main` :

```bash
git merge test
```

Ne jamais exécuter :

```bash
git push --force origin main
```

Ne jamais modifier manuellement `ZeSolver-main` comme branche de développement.

Ne jamais publier depuis un `test` sale ou non poussé.

## 12. Récupération avant commit

Pour abandonner une projection non commitée dans `main` :

```bash
cd /home/tristan/.openclaw/workspace/projects/ZeSolver-main
git reset --hard HEAD
git clean -fd
```

Attention : cela supprime toutes les modifications non commitées de ce worktree.

Après une publication erronée déjà poussée, ne pas réécrire l’historique. Corriger sur `test`, régénérer et publier un nouveau commit correctif sur `main`.

## 13. Résumé

```text
Un seul dépôt GitHub : tinystork/ZeSolver

ZeSolver/
└── test
    └── développement complet et source de vérité

ZeSolver-main/
└── main
    └── distribution publique générée

Jamais de merge test -> main.
Jamais de développement direct sur main.
Toujours générer main depuis un commit propre et poussé de test.
```
