# P3A GUI Manual Test Instructions

Use a normal graphical desktop session. Do not run this as a headless-only
substitute.

1. Start the GUI:

   ```bash
   ZESOLVER_GUI_ENGINE=auto .venv/bin/python zesolver.py
   ```

2. Verify there is no launch error.
3. Select a FITS input folder.
4. Start a run in `AUTO`.
5. Confirm the log includes `Engine selection: requested=auto selected=pipeline`.
6. Observe progress and per-file results.
7. Run a small FITS batch.
8. Press Stop during a run and verify the UI remains responsive.
9. Start another run after Stop.
10. Close the window during an active run and verify shutdown completes.
11. Test a raster input (`TIFF`, `PNG`, or `JPEG`) in `AUTO` and confirm the log
    selects legacy with a raster reason.
12. Test a raster with:

    ```bash
    ZESOLVER_GUI_ENGINE=pipeline .venv/bin/python zesolver.py
    ```

    Confirm the request is rejected clearly and does not silently fallback.

13. Test Astrometry.net web mode and confirm it stays on the legacy route.
14. Verify source pixels are unchanged except for explicitly expected WCS writes.

Expected rollback command:

```bash
ZESOLVER_GUI_ENGINE=legacy .venv/bin/python zesolver.py
```
