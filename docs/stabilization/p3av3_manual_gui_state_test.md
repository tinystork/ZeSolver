# P3A-V3 Manual GUI State Test

Status: not passed in this session.

Graphical environment was available:

```text
DISPLAY=:1
WAYLAND_DISPLAY=wayland-0
XDG_SESSION_TYPE=wayland
```

Temporary corpus root:

```text
/tmp/zesolver_p3av3_gui_CNAvxn
```

Planned tests:

| Test | Scope | Result |
| ---- | ----- | ------ |
| TEST_A_WCS_CLEANUP_REFRESH | WCS cleanup row refresh | blocked by harness scan |
| TEST_B_LEGACY_LIVE_GUI | legacy live progress | blocked by harness scan |
| TEST_C_PIPELINE_LIVE_GUI | pipeline live progress | blocked by harness scan |
| TEST_D_PROGRESS_STOP_RESTART | stop then restart progress | blocked by harness scan |
| TEST_E_LARGE_BATCH_LIVE_PROGRESS | large batch first progress | not executed after scan blockage |

Harness logs:

```text
/tmp/p3av3_manual_runner.out
/tmp/p3av3_manual_runner2.out
/tmp/p3av3_manual_runner3.out
/tmp/p3av3_manual_runner4.out
```

Observed blockage:

```text
automatic scan emitted tens of thousands of "fichier(s) detecte(s)" lines
before the scenario completed
```

No manual validation result is invented. The automated tests cover the corrected
state transitions and live-callback behavior, but the manual Wayland gate remains
open for P3A-V3.
