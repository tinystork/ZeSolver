#!/usr/bin/env python3
"""ZeSolver compatibility shim — delegates to the installed package.

When running from a source checkout (Python path resolves the local file),
this shim delegates directly to ``zesolver._app.main``.  When installed
via the ZeSolver wheel, ``gui_scripts:zesolver`` maps to the same
``zesolver._app:main`` entry point and this file is never used.
"""

from zesolver._app import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
