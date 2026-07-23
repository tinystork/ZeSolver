from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest


def test_main_gui_builds_without_benchmark_surface_or_engine_import() -> None:
    pytest.importorskip("PySide6")

    script = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        from pathlib import Path
        import sys

        sys.modules.pop("tools.benchmark_solver", None)

        from PySide6 import QtWidgets
        from zesolver.settings_store import PersistentSettings

        spec = importlib.util.spec_from_file_location("zesolver_app", Path("zesolver.py"))
        assert spec is not None and spec.loader is not None
        zesolver_app = importlib.util.module_from_spec(spec)
        sys.modules["zesolver_app"] = zesolver_app
        spec.loader.exec_module(zesolver_app)

        captured = {}

        def _labels(window):
            return [window.tabs.tabText(i) for i in range(window.tabs.count())]

        def _visible_labels(window):
            return [
                window.tabs.tabText(i)
                for i in range(window.tabs.count())
                if window.tabs.isTabVisible(i)
            ]

        def fake_exec(self):
            self.processEvents()
            windows = [w for w in self.topLevelWidgets() if w.__class__.__name__ == "ZeSolverWindow"]
            assert len(windows) == 1, [w.__class__.__name__ for w in self.topLevelWidgets()]
            window = windows[0]
            captured["labels_fr_expert"] = _labels(window)
            captured["visible_fr_expert"] = _visible_labels(window)
            captured["has_benchmark_attrs"] = any(
                hasattr(window, name)
                for name in (
                    "benchmark_tab",
                    "benchmark_scroll",
                    "_benchmark_worker",
                    "bench_inputs_edit",
                    "bench_run_btn",
                )
            )
            captured["has_benchmark_methods"] = any(
                hasattr(window, name)
                for name in (
                    "_build_benchmark_tab",
                    "_on_benchmark_run_clicked",
                    "_on_benchmark_stop_clicked",
                    "_on_benchmark_finished",
                )
            )
            window._language = "en"
            window._apply_language()
            captured["labels_en_expert"] = _labels(window)
            window._interface_mode = "easy"
            window._apply_interface_mode()
            captured["visible_en_easy"] = _visible_labels(window)
            window._interface_mode = "expert"
            window._apply_interface_mode()
            captured["visible_en_expert_after_toggle"] = _visible_labels(window)
            captured["benchmark_module_loaded"] = "tools.benchmark_solver" in sys.modules
            window.close()
            self.processEvents()
            captured["closed"] = not window.isVisible()
            return 0

        zesolver_app.load_persistent_settings = lambda: PersistentSettings(interface_mode="expert")
        zesolver_app.save_persistent_settings = lambda settings: None
        QtWidgets.QApplication.exec = fake_exec

        args = zesolver_app.build_arg_parser().parse_args(["--gui"])
        code = zesolver_app.launch_gui(args)
        captured["return_code"] = code
        print(json.dumps(captured, sort_keys=True), flush=True)
        os._exit(0)
        """
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.fspath(os.path.dirname(os.path.dirname(__file__))),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["return_code"] == 0
    assert payload["closed"]
    assert not payload["has_benchmark_attrs"]
    assert not payload["has_benchmark_methods"]
    assert not payload["benchmark_module_loaded"]
    for key, labels in payload.items():
        if key.startswith(("labels_", "visible_")):
            assert "Benchmark" not in labels
