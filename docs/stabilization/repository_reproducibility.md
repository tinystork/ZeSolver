# P0B - Repository Reproducibility Audit

Date: 2026-07-17

## Ignored File Audit

Command used:

```bash
git check-ignore -v tests/test_batch_blind_fallback.py \
  tests/test_p221_app_integration.py \
  tests/test_p222_gui_integration.py \
  tools/build_zn310b_gui_dataset.py \
  reports/ \
  reports/zenear_zn310b_gui_manifest.json \
  reports/zenear_zn310b_gui_result.json
```

Initial cause: `.gitignore` ignored entire `tests/`, `tools/`, and `reports/` trees. This made the P0A green baseline depend on local ignored files.

| Fichier | Regle `.gitignore` | Role | Doit etre suivi | Action |
| --- | --- | --- | --- | --- |
| `tests/test_batch_blind_fallback.py` | ancienne regle `.gitignore:25:tests/`; maintenant `!tests/*.py` | test indispensable de fallback Near/Blind | oui | rendre les tests Python suivis |
| `tests/test_p221_app_integration.py` | ancienne regle `.gitignore:25:tests/`; maintenant `!tests/*.py` | test indispensable settings/app 4D | oui | rendre les tests Python suivis |
| `tests/test_p222_gui_integration.py` | ancienne regle `.gitignore:25:tests/`; maintenant `!tests/*.py` | test indispensable GUI/settings 4D | oui | rendre les tests Python suivis |
| `tools/build_zn310b_gui_dataset.py` | ancienne regle `.gitignore:24:tools/`; maintenant exception ciblee | generateur reproductible du dataset ZN3.10B | oui | exception ciblee |
| `tools/diagnose_*.py` | ancienne regle `.gitignore:24:tools/`; maintenant `!tools/diagnose_*.py` | fermeture d'import des tests ZN | oui | suivre ces scripts source legers, sans suivre tous les outils |
| `reports/` | `reports/**` | dossier de rapports generes | non globalement | conserver ignore global |
| rapports ZN selectionnes | exceptions `!reports/zenear_*.json` / `.md` | petits oracles historiques lus par les tests existants | oui, selection seulement | exceptions fichier par fichier |
| `reports/zenear_zn310b_gui_result.json` | `reports/**` | resultat genere du run GUI manuel | non | reste ignore |

## Gitignore Decision

Tracked or trackable:

- test sources under `tests/*.py`;
- `tests/corpus/` manifest, README, and small oracle JSON files;
- source tools required by tests and dataset reproduction;
- selected small report oracles already consumed by existing tests.

Still ignored:

- FITS corpus copies;
- ASTAP databases;
- 4D NPZ indexes;
- generated GUI run result logs;
- unselected benchmark/report output;
- caches and temporary data.

The selected report exceptions are intentionally narrow. They do not force-add the whole `reports/` tree.

## Clean Checkout Verification

Because the mission is in-progress and no branch commit was created, a temporary detached commit was created from a temporary Git index, without moving the branch:

```bash
tmp_index=$(mktemp /tmp/zesolver-index.XXXXXX)
export GIT_INDEX_FILE="$tmp_index"
git read-tree HEAD
git add -A
TREE=$(git write-tree)
COMMIT=$(printf 'temporary P0B reproducibility snapshot final\n' | git commit-tree "$TREE" -p HEAD)
unset GIT_INDEX_FILE
rm -f "$tmp_index"
git worktree add --detach /tmp/zesolver-clean-baseline-p0b "$COMMIT"
```

Temporary commit tested:

```text
c765ca6d16b8d0414d7182df24bc9a53d4c14241
```

Clean environment setup:

```bash
python -m venv /tmp/zesolver-clean-baseline-p0b/.venv-clean
/tmp/zesolver-clean-baseline-p0b/.venv-clean/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/zesolver-clean-baseline-p0b/.venv-clean/bin/python -m pip install numpy astropy scipy scikit-image astroalign rich pytest PySide6
```

Editable install check:

```bash
/tmp/zesolver-clean-baseline-p0b/.venv-clean/bin/python -m pip install -e '.[dev,gui]'
```

Result: failed before tests because packaging has no explicit package discovery and setuptools sees multiple top-level packages (`icon`, `config`, `reports`, `zewcs290`, `zesolver`, `packaging`, `zeblindsolver`). This is a P4 packaging issue, not a P0B solver regression.

Clean baseline command:

```bash
/tmp/zesolver-clean-baseline-p0b/.venv-clean/bin/python tools/run_regression_suite.py --hermetic
```

Result:

```text
248 passed, 1 skipped, 6 deselected, 1 warning
compileall OK
runner status PASS
```

## Remaining Reproducibility Limits

- The project is not yet installable with `pip install -e .`; P4 must add explicit package discovery or a source layout.
- Corpus tests skip until external data variables are configured.
- The selected `reports/` files are historical oracle inputs. They should eventually be replaced by smaller normalized files under `tests/corpus/oracles/`, but P0B preserves existing tests without mass-report tracking.
