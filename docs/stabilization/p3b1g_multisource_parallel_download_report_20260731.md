# P3B-1G - Telechargement multi-source et parallele des catalogues

Date: 2026-08-01
Branche: `test`

## Architecture retenue

La logique reste dans `zesolver/catalog_library/distribution.py`, hors Qt.
Le wizard continue d'appeler simplement `CatalogDistributionService.install_distribution(plan)`.

Nouveaux concepts:

- `DistributionSource`: source de transport possible;
- `DistributionSourceCandidate`: URL effective pour un asset;
- `DistributionDownloadPolicy`: ordre des sources et limite de concurrence;
- `ParallelDistributionDownloader`: orchestration parallele par composant;
- `SourceHealthState`: sante temporaire par run;
- `_DistributionProgressAggregator`: progression globale serialisee sous verrou.

Les composants distincts sont telecharges avec un `ThreadPoolExecutor`. Les
assets identiques ne sont planifies qu'une seule fois.

## Miroirs dormants

La politique par defaut contient exactement:

- `mirror-1`: `enabled=false`, URL vide;
- `mirror-2`: `enabled=false`, URL vide;
- `github-release`: `enabled=true`, source canonique.

Une source desactivee ou sans URL ne produit aucun candidat et ne provoque
aucune requete reseau.

## Compatibilite manifeste v1

Le manifeste `zesolver.catalog_distribution.v1` reste inchangé. Aucun champ
miroir n'est requis. Les anciennes releases GitHub continuent donc de fournir:

- manifeste JSON;
- assets nommes;
- tailles;
- SHA-256.

Les miroirs futurs pourront etre actives par configuration de service
(`DistributionDownloadPolicy`) avec `base_url` ou `url_template`, sans modifier
le moteur.

## Ordre et basculement

Ordre effectif quand les miroirs sont actifs:

1. `mirror-1`
2. `mirror-2`
3. `github-release`

Dans l'etat initial dormant, l'ordre effectif est seulement:

1. `github-release`

Pour un composant, une seule source est essayee a la fois. En cas d'echec
reseau, HTTP, taille, SHA ou reprise incompatible, le composant bascule vers la
source suivante. Si toutes les sources echouent, l'erreur
`DISTRIBUTION_SOURCE_UNAVAILABLE` contient les sources essayees et leurs causes.

## Parallélisme

La valeur par defaut est `max_parallel_downloads=3`, bornee de 1 a 4. Le
parallele s'applique uniquement entre composants distincts. L'assemblage,
l'extraction, la validation et l'activation atomique restent ordonnes apres la
reussite de tous les telechargements requis.

## Progression

L'agregateur de progression maintient:

- octets globaux actuels et attendus;
- composants actifs;
- composants termines;
- source courante par composant;
- octets telecharges, repris et reutilises depuis cache;
- debit global approximatif;
- duree ecoulee;
- estimation restante quand calculable;
- evenements de basculement de source.

Les callbacks de progression sont serialises sous verrou afin que Qt ne recoive
pas d'appels concurrents.

## Reprise inter-source

Le cache final reste indexe par:

`library_id / version / asset`

Il n'est pas indexe par hebergeur. Un `.part` peut donc etre repris depuis une
autre source si elle fournit le meme composant. Les metadonnees `ETag` et
`Last-Modified` ne sont comparees strictement que lorsqu'elles proviennent de
la meme source. L'autorite finale reste toujours la taille attendue et le
SHA-256 du manifeste.

Un fichier partiel pollue par une taille, un SHA ou un `Content-Range` invalide
est supprime avant de tenter la source suivante.

## Annulation

Une annulation:

- active un stop global;
- est observee par tous les downloaders;
- empeche l'assemblage;
- conserve les `.part` reprenables;
- attend la terminaison des workers actifs;
- ne persiste aucun chemin final.

## Securite et integrite

Conserves:

- taille obligatoire;
- SHA-256 obligatoire;
- extraction ZIP protegee;
- detection de collisions;
- staging separe;
- controle d'espace disque;
- validation `CatalogLibrary`;
- activation atomique par le service de management;
- persistance seulement apres succes complet.

Les miroirs ne sont qu'un transport alternatif. Ils ne deviennent jamais une
autorite d'integrite.

## Telemetrie

Evenements structures ajoutes aux logs:

- `DISTRIBUTION_RUN_BEGIN`
- `DISTRIBUTION_COMPONENT_BEGIN`
- `DISTRIBUTION_COMPONENT_END`
- `DISTRIBUTION_SOURCE_SWITCH`
- `DISTRIBUTION_ASSEMBLY_BEGIN`
- `DISTRIBUTION_ASSEMBLY_END`
- `DISTRIBUTION_VALIDATION_BEGIN`
- `DISTRIBUTION_VALIDATION_END`
- `DISTRIBUTION_RUN_END`

Le resume final contient notamment `library_id`, `version`, destination, cache,
sources utilisees, nombre de composants, octets telecharges/repris/reutilises,
durees de decouverte, telechargement, SHA-256, assemblage, validation et total,
debit moyen global, concurrence maximale observee et statut final.

Les erreurs loggees masquent les URLs completes pour eviter les URLs signees.

## Tests executes

Commandes:

```bash
.venv/bin/python -m pytest -q tests/test_catalog_distribution_multisource.py
.venv/bin/python -m pytest -q tests/test_catalog_distribution.py tests/test_catalog_distribution_multisource.py
.venv/bin/python -m pytest -q tests/test_catalog_distribution.py tests/test_catalog_distribution_multisource.py tests/test_catalog_library_paths.py tests/test_catalog_library_management_service.py tests/test_startup_wizard.py
.venv/bin/python -m pytest -q --ignore=tests/test_catalog_blind4d_manifest_view_cli.py
```

Resultats:

- `23 passed` pour le fichier multi-source;
- `40 passed` pour distribution existante + multi-source;
- `92 passed` pour le lot cible demande;
- `769 passed, 29 skipped` pour le lot elargi avec l'ignore demande.

Note Qt: les tests QWizard du startup wizard restent executes quand
`tests/test_startup_wizard.py` est appele explicitement (`25 passed`). Dans le
lot global, ils sont skips pour eviter un abort natif PySide apres d'autres
tests GUI historiques dans le meme process.

## Preuve de concurrence synthetique

Le backend de test `_FakeBackend` mesure:

- `max_active`: nombre maximal de streams actifs simultanes;
- `duplicate_assets`: detection d'un meme asset actif en double;
- requetes par host et par asset;
- headers `Range`.

Les tests prouvent:

- chevauchement reel avec `max_active >= 2`;
- limite respectee avec `max_active <= 2` quand `max_parallel_downloads=2`;
- aucun double telechargement simultane du meme asset;
- progression globale monotone.

## Resultats de performance synthetiques

Avec trois composants simules et un delai de lecture par chunk, le downloader
atteint une concurrence observee superieure a 1. Le gain reel dependra des
tailles d'assets, du debit Windows et de GitHub Releases. Les miroirs restent
dormants pendant cette mission.

## Limites restantes

- Pas de telechargement reel multi-Go effectue ici.
- Pas d'activation de `tinystork.free.fr`, Cloudflare R2, Hugging Face ou autre.
- La comparaison de performance reelle reste a faire apres le telechargement
  Windows de reference.

## Activation future d'un miroir

Exemple cote service, sans changement du moteur:

```python
policy = DistributionDownloadPolicy(
    sources=(
        DistributionSource("mirror-1", enabled=True, base_url="https://example.invalid/zesolver-catalogs"),
        DistributionSource("mirror-2", enabled=False),
        DistributionSource("github-release", enabled=True, canonical=True),
    ),
    max_parallel_downloads=3,
)
service = CatalogDistributionService(download_policy=policy)
```

Le miroir doit exposer les memes noms d'assets que la release GitHub et les
memes octets. La validation reste faite par taille et SHA-256 du manifeste.

Statut: `READY_FOR_P3B1G_DOWNLOAD_VALIDATION`
