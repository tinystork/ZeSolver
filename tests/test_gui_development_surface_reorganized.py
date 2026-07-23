from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest


def test_development_surface_reorganized_and_persisted() -> None:
    pytest.importorskip("PySide6")

    script = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        from pathlib import Path
        import sys
        import tempfile
        from types import SimpleNamespace

        from PySide6 import QtWidgets
        from zesolver.settings_store import PersistentSettings

        spec = importlib.util.spec_from_file_location("zesolver_app_p3b1c", Path("zesolver.py"))
        assert spec is not None and spec.loader is not None
        zesolver_app = importlib.util.module_from_spec(spec)
        sys.modules["zesolver_app_p3b1c"] = zesolver_app
        spec.loader.exec_module(zesolver_app)

        root = Path(tempfile.mkdtemp(prefix="p3b1c-gui-"))
        db_root = root / "database"
        index_root = root / "index"
        db_root.mkdir()
        index_root.mkdir()

        ok = SimpleNamespace(ok=True)
        zesolver_app.validate_astap_root = lambda _path: ok
        zesolver_app.validate_legacy_near_index_root = lambda _path: ok
        zesolver_app.validate_blind4d_manifest_file = lambda _path: ok
        saved = []

        initial = PersistentSettings(
            interface_mode="expert",
            db_root=str(db_root),
            index_root=str(index_root),
            solver_downsample=2,
            solver_workers=3,
            solver_cache_size=13,
            dev_bucket_limit_override=222,
            dev_vote_percentile=55,
            dev_bucket_cap_S=1234,
            dev_bucket_cap_M=2345,
            dev_bucket_cap_L=3456,
            dev_detect_k_sigma=4.4,
            dev_detect_min_area=8,
            dev_hash_quads_S=1111,
            dev_hash_quads_M=2222,
            dev_hash_quads_L=3333,
            dev_family_auto=True,
        )

        def _labels(window):
            return [window.tabs.tabText(i) for i in range(window.tabs.count())]

        def _visible_labels(window):
            return [window.tabs.tabText(i) for i in range(window.tabs.count()) if window.tabs.isTabVisible(i)]

        def _save(settings):
            saved.append(settings)

        def fake_exec(self):
            self.processEvents()
            windows = [w for w in self.topLevelWidgets() if w.__class__.__name__ == "ZeSolverWindow"]
            assert len(windows) == 1
            window = windows[0]
            captured = {}
            captured["labels_fr_expert"] = _labels(window)
            captured["visible_fr_expert"] = _visible_labels(window)
            captured["has_dev_tab_attrs"] = any(hasattr(window, name) for name in ("dev_tab", "dev_scroll", "dev_workers_combo", "dev_save_btn"))
            captured["has_dev_builder"] = hasattr(window, "_build_dev_tab")
            captured["downsample_value"] = window.downsample_spin.value()
            captured["workers_value"] = window.workers_spin.value()
            captured["workers_in_performance"] = window.performance_tab.isAncestorOf(window.workers_spin)
            captured["legacy_cache_value"] = window.cache_spin.value()
            captured["legacy_scope_text"] = window.legacy_blind_scope_label.text()
            window.catalog_compat_group.setChecked(True)
            self.processEvents()
            captured["legacy_visible_expert"] = (not window.catalog_compat_group.isHidden()) and (not window.legacy_blind_group.isHidden())
            captured["hash_widgets_before_open"] = hasattr(window, "dev_hash_buttons")
            window._open_historical_index_maintenance_dialog()
            self.processEvents()
            captured["hash_dialog_visible"] = window._hash_maintenance_dialog.isVisible()
            captured["hash_buttons"] = sorted(window.dev_hash_buttons.keys())
            captured["hash_worker_after_open"] = window._index_worker is not None

            window.downsample_spin.setValue(4)
            window.workers_spin.setValue(0)
            window.cache_spin.setValue(22)
            window.dev_sigma_spin.setValue(5.5)
            window.dev_hash_quads_spin["S"].setValue(4444)
            window._on_save_settings_clicked()
            latest = saved[-1]
            captured["saved"] = {
                "downsample": latest.solver_downsample,
                "workers": latest.solver_workers,
                "cache": latest.solver_cache_size,
                "sigma": latest.dev_detect_k_sigma,
                "hash_s": latest.dev_hash_quads_S,
            }

            window._language = "en"
            window._apply_language()
            captured["labels_en_expert"] = _labels(window)
            window._interface_mode = "easy"
            window._apply_interface_mode()
            self.processEvents()
            captured["visible_en_easy"] = _visible_labels(window)
            captured["legacy_visible_easy"] = not window.catalog_compat_group.isHidden()
            captured["tools_visible_easy"] = window.tools_menu.menuAction().isVisible()

            second = window.__class__(latest)
            second.show()
            self.processEvents()
            captured["reopened"] = {
                "downsample": second.downsample_spin.value(),
                "workers": second.workers_spin.value(),
                "cache": second.cache_spin.value(),
                "sigma": second.dev_sigma_spin.value(),
            }
            second.close()
            window.close()
            self.processEvents()
            captured["closed"] = not window.isVisible() and not second.isVisible()
            print(json.dumps(captured, sort_keys=True), flush=True)
            return 0

        zesolver_app.load_persistent_settings = lambda: initial
        zesolver_app.save_persistent_settings = _save
        QtWidgets.QApplication.exec = fake_exec

        args = zesolver_app.build_arg_parser().parse_args(["--gui"])
        code = zesolver_app.launch_gui(args)
        assert code == 0
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

    assert "Développement" not in payload["labels_fr_expert"]
    assert "Development" not in payload["labels_en_expert"]
    assert not payload["has_dev_tab_attrs"]
    assert not payload["has_dev_builder"]
    assert payload["downsample_value"] == 2
    assert payload["workers_value"] == 3
    assert payload["workers_in_performance"]
    assert payload["legacy_cache_value"] == 13
    assert "ZeBlind 4D" in payload["legacy_scope_text"]
    assert payload["legacy_visible_expert"]
    assert payload["hash_widgets_before_open"] is False
    assert payload["hash_dialog_visible"]
    assert payload["hash_buttons"] == ["L", "M", "S"]
    assert payload["hash_worker_after_open"] is False
    assert payload["saved"] == {
        "downsample": 4,
        "workers": 0,
        "cache": 22,
        "sigma": 5.5,
        "hash_s": 4444,
    }
    assert payload["visible_en_easy"] == ["Solver", "Settings"]
    assert payload["legacy_visible_easy"] is False
    assert payload["tools_visible_easy"] is False
    assert payload["reopened"] == {
        "downsample": 4,
        "workers": 0,
        "cache": 22,
        "sigma": 5.5,
    }
    assert payload["closed"]
