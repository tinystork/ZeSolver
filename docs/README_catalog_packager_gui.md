# ZeSolver Catalogue Packager GUI

Outil graphique PySide6 pour convertir une Bibliothèque ZeSolver de travail utilisant des références externes en une copie portable et publiable.

## Sécurités

- ne modifie jamais la bibliothèque source ;
- ne modifie jamais `/opt/astap` ;
- construit dans un répertoire temporaire puis publie atomiquement ;
- exclut le monolithe `d50_4d.npz` et `.source_mmap_cache` ;
- réécrit les chemins runtime en chemins relatifs ;
- retire les chemins personnels de la provenance Blind4D ;
- valide la copie avec le `CatalogLibrary` de la codebase courante ;
- calcule les SHA-256 de tous les fichiers installables ;
- refuse la publication si un chemin absolu subsiste dans les manifestes portables.

## Installation dans le dépôt

Copier le script dans `tools/` :

```bash
cd ~/.openclaw/workspace/projects/ZeSolver
mkdir -p tools
cp /chemin/vers/zesolver_catalog_packager_gui.py tools/
chmod +x tools/zesolver_catalog_packager_gui.py
```

PySide6 est déjà prévu par les dépendances GUI de ZeSolver. Au besoin :

```bash
source .venv/bin/activate
pip install PySide6
```

## Lancement

Depuis la racine de ZeSolver et dans son environnement virtuel :

```bash
source .venv/bin/activate
python tools/zesolver_catalog_packager_gui.py
```

Valeurs préremplies pour la machine de développement :

- bibliothèque : `/home/tristan/ZeSolverCatalog/new` ;
- ASTAP D50 : `/opt/astap` ;
- sortie : `~/ZeSolverCatalog/releases`.

## Sortie

Chaque construction crée un nouveau dossier horodaté, par exemple :

```text
zesolver-d50-v1.0.0-20260727-123456/
├── assets/
│   ├── zesolver-d50-near-v1.0.0.zip
│   ├── zesolver-d50-blind4d-fixed32-v1.0.0.zip
│   ├── zesolver-d50-metadata-v1.0.0.zip
│   ├── zesolver-d50-distribution-v1.0.0.json
│   └── SHA256SUMS
├── portable-package/
│   ├── zesolver-library-package.json
│   ├── NOTICE.md
│   ├── legal/
│   └── library/
│       ├── catalog.json
│       ├── sources/astap-d50/
│       └── indexes/blind4d-fixed32/
└── reports/
    └── packaging-report.json
```

Les ZIP sont créés sans recomprimer les gros fichiers scientifiques : cela évite une très longue compression et garantit que chaque gros asset reste nettement sous la limite GitHub de 2 Gio.

Le dossier `portable-package/` est directement compatible avec `CatalogLibraryManagementService.install_package()` pour un essai local. Les trois ZIP constituent le format segmenté destiné au futur téléchargement P3B-1E : ils doivent être extraits dans une même racine de paquet.

## Vérification sans GUI

Une analyse seule est disponible en ligne de commande :

```bash
python tools/zesolver_catalog_packager_gui.py --analyze \
  --library /home/tristan/ZeSolverCatalog/new \
  --astap /opt/astap
```
