# Legacy Process Cancellation Map

Mission: P3A-V1 responsive Stop and multiprocess cancellation hardening.

## Current Route

```text
Stop click
-> ZeSolverWindow._stop_solving()
-> SolveRunner.request_cancel()
-> GuiSolveController.cancel()
-> LegacyGuiRunner.cancel()
-> BatchSolver.run()
-> ProcessPoolExecutor
-> _near_worker_init(..., process_cancel_token)
-> _near_worker_solve()
-> ImageSolver.solve_path()
-> executor shutdown / worker termination
```

## Mechanism

Before P3A-V1, `SolveRunner.request_cancel()` only set the QThread-local
`threading.Event`. The selected `LegacyGuiRunner` owned a second event, so the
legacy runner did not reliably receive Stop through `GuiSolveController`.
Process workers also initialized `ImageSolver` with `set_cancel_event(None)`,
so already-running Near tasks could not see the parent token.

P3A-V1 now forwards Stop to the active controller runner and passes a
`ProcessCancellationToken` through the process-pool initializer.

| Etape | Processus/thread | Token visible | Peut être interrompue | Risque actuel | Correction |
| ----- | ---------------- | ------------: | --------------------: | ------------- | ---------- |
| GUI event loop | Qt main | QThread event | oui | Stop accuse tardivement si seul le worker reagit | UI passe immediatement en `Arrêt en cours...`, Stop desactive |
| QThread / worker Qt | `SolveRunner` | `threading.Event` + controller runner | oui | event local non propage au runner selectionne | `request_cancel()` appelle `GuiSolveController.cancel()` |
| GuiSolveController | QThread, appele depuis Qt main pour cancel | runner actif | oui | runner interne inaccessible avant | controller conserve `_runner` et relaie `cancel()` |
| LegacyGuiRunner | QThread | `threading.Event` interne | oui | son event pouvait rester unset | Stop le set via controller |
| BatchSolver | QThread | `CompositeCancellationToken` | oui | token parent non process-safe | compose event Qt + token IPC |
| ProcessPoolExecutor | process parents/enfants | token IPC via initializer | partiellement | `shutdown(wait=False)` ne stoppe pas les actifs | shutdown borne + cancel pending + terminate fallback |
| future en attente | parent | n/a | oui | pouvait disparaitre du resultat | `future.cancel()` + resultat `cancelled` |
| future active | worker process | `ProcessCancellationToken` | cooperative, puis force | pouvait tourner plusieurs dizaines de secondes | checkpoints existants voient le token; force apres grace |
| initialisation worker | worker process | token IPC | oui apres init | token absent | `_near_worker_init(config, token)` |
| lecture FITS | worker process | token avant/apres | cooperative | annulation vue seulement sous threads | token IPC visible dans `ImageSolver` |
| chargement catalogue | worker process | token avant appels couteux Near | cooperative | worker process aveugle au Stop | token passe comme `cancel_check` |
| detection d'etoiles | worker process | token avant/apres | cooperative | latence selon detection native | checkpoints existants conserves |
| selection de tuiles | worker process | token dans `near_solve` | cooperative | pas visible en process | `cancel_check` process-safe |
| construction catalogue local | worker process | token dans `near_solve` | cooperative | pas visible en process | `cancel_check` process-safe |
| RANSAC / validation | worker process | token dans `near_solve` | cooperative periodique | longues boucles possibles | `cancel_check` process-safe |
| ecriture WCS | worker process | token + etat `wcs_write` | avant/apres, pas pendant | kill dangereux pendant update FITS | section critique protegee; pas de kill pendant `wcs_write` |
| retour du resultat | parent | n/a | n/a | annulation forcee convertie en failed | exception de future pendant Stop => `cancelled` |
| shutdown executor | parent | token IPC + states | oui | executor ferme reutilise / futures apres shutdown | references locales fermees, nouveau run cree un nouveau pool |

## Token Type

- GUI/QThread: `threading.Event`, wrapped by `ThreadCancellationToken`.
- Process workers: `multiprocessing.Manager().Event`, wrapped by
  `ProcessCancellationToken`.
- Legacy batch: `CompositeCancellationToken`.

The process token is a Manager proxy so it is serializable for `spawn` and does
not depend on Linux `fork` inheritance.
