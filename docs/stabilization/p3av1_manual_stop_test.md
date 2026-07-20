# P3A-V1 Manual Stop Test

Status: not executed visually in this session.

Reason: this Codex run is headless. The real Qt desktop checklist still needs a
manual GUI pass, which is exactly the expected next gate.

## Automated Substitute Executed

Synthetic legacy process tests covered:

- 6-worker/process-style route with more files than workers;
- cooperative active workers seeing the shared process token;
- pending futures returned as `cancelled`;
- non-cooperative active workers terminated after a bounded grace period;
- no missing or duplicate result entries;
- restart with a fresh executor after Stop;
- close-style cancellation through the same process shutdown path;
- WCS critical-section protection in forced shutdown.

Observed targeted result:

```text
9 passed in 8.87s
```

Warnings observed in this targeted lot:

```text
multiprocessing.popen_fork.DeprecationWarning
```

This warning category existed before P3A-V1.

## Manual GUI Checklist To Re-run

Use a normal desktop session:

```bash
ZESOLVER_GUI_ENGINE=auto .venv/bin/python zesolver.py
```

Scenario:

1. select 30 FITS;
2. use 6 workers;
3. keep AUTO routed to LEGACY with process Near;
4. start the run;
5. wait for several active workers;
6. click Stop;
7. verify immediate `Arrêt en cours...`;
8. verify no new submissions;
9. verify bounded stop;
10. verify no child processes remain;
11. start a second run;
12. close the window during that run;
13. verify clean close.

Fields to record after the visual run:

| Metric | Value |
| ------ | ----- |
| latence clic -> accuse GUI | pending manual |
| latence clic -> fin | pending manual |
| workers actifs au clic | pending manual |
| futures annulees | pending manual |
| workers termines cooperativement | pending manual |
| workers termines de force | pending manual |
| processus restants | pending manual |
| resultat relance | pending manual |
| resultat fermeture | pending manual |
