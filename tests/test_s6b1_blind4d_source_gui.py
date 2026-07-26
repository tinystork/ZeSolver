from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest


def test_blind4d_external_manifest_controls_live_only_in_advanced_source_selector() -> None:
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

        spec = importlib.util.spec_from_file_location("zesolver_app_s6b1", Path("zesolver.py"))
        assert spec is not None and spec.loader is not None
        zesolver_app = importlib.util.module_from_spec(spec)
        sys.modules["zesolver_app_s6b1"] = zesolver_app
        spec.loader.exec_module(zesolver_app)

        root = Path(tempfile.mkdtemp(prefix="s6b1-gui-"))
        db_root = root / "database"
        index_root = root / "index"
        db_root.mkdir()
        index_root.mkdir()
        manifest = root / "blind4d_manifest.json"
        new_manifest = root / "diagnostic_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        new_manifest.write_text("{}", encoding="utf-8")

        ok = SimpleNamespace(ok=True)
        zesolver_app.validate_astap_root = lambda _path: ok
        zesolver_app.validate_legacy_near_index_root = lambda _path: ok
        zesolver_app.validate_blind4d_manifest_file = lambda _path: ok

        initial = PersistentSettings(
            interface_mode="expert",
            db_root=str(db_root),
            index_root=str(index_root),
            blind_backend_profile=zesolver_app.ZEBLIND_4D_EXPERIMENTAL_PROFILE,
            blind4d_catalog_mode="library-view",
            blind_4d_manifest_path=str(manifest),
        )

        def _hidden(window):
            return {
                "label": window.settings_blind_4d_manifest_label.isHidden(),
                "row": window.settings_blind_4d_manifest_row.isHidden(),
                "edit": window.settings_blind_4d_manifest_edit.isHidden(),
                "browse": window.settings_blind_4d_manifest_browse_btn.isHidden(),
                "verify": window.settings_blind_4d_manifest_verify_btn.isHidden(),
                "status": window.settings_blind_4d_manifest_status_label.isHidden(),
            }

        def fake_exec(self):
            self.processEvents()
            windows = [w for w in self.topLevelWidgets() if w.__class__.__name__ == "ZeSolverWindow"]
            assert len(windows) == 1
            window = windows[0]
            window.catalog_compat_group.setChecked(True)
            self.processEvents()

            combo = window.blind4d_catalog_mode_combo
            captured = {}
            captured["standard_attrs_absent"] = {
                name: not hasattr(window, name)
                for name in (
                    "blind_4d_manifest_label",
                    "blind_4d_manifest_edit",
                    "blind_4d_manifest_browse_btn",
                    "blind_4d_manifest_verify_btn",
                )
            }
            captured["status_not_in_solver_tab"] = not window.solver_scroll.isAncestorOf(window.settings_blind_4d_manifest_status_label)
            captured["source_label_fr"] = window.blind4d_catalog_mode_label.text()
            captured["help_fr"] = window.blind4d_source_help_label.text()
            captured["combo_values"] = [combo.itemData(i) for i in range(combo.count())]
            captured["combo_labels_fr"] = [combo.itemText(i) for i in range(combo.count())]
            captured["legacy_library_view_maps_to_auto"] = combo.currentData() == "auto" and window._current_blind4d_catalog_mode_from_ui() == "auto"
            captured["hidden_auto"] = _hidden(window)

            combo.setCurrentIndex(combo.findData("external-manifest"))
            self.processEvents()
            captured["shown_external"] = {key: not value for key, value in _hidden(window).items()}
            captured["path_loaded_in_external"] = window.settings_blind_4d_manifest_edit.text()
            captured["external_settings"] = {
                "mode": window._read_settings_from_ui().blind4d_catalog_mode,
                "manifest": str(window._read_settings_from_ui().blind_4d_manifest_path),
            }

            fake_loaded = SimpleNamespace(entries=[object()], tile_keys=["d50_2823"], manifest_path=manifest)
            window._set_manifest_status("valid", manifest=fake_loaded)
            captured["status_valid_before_change"] = window.settings_blind_4d_manifest_status_label.text()
            window.settings_blind_4d_manifest_edit.setText(str(new_manifest))
            self.processEvents()
            captured["status_after_path_change"] = window.settings_blind_4d_manifest_status_label.text()

            combo.setCurrentIndex(combo.findData("auto"))
            self.processEvents()
            captured["hidden_after_return_auto"] = _hidden(window)
            captured["auto_settings"] = {
                "mode": window._read_settings_from_ui().blind4d_catalog_mode,
                "manifest": str(window._read_settings_from_ui().blind_4d_manifest_path),
            }
            combo.setCurrentIndex(combo.findData("external-manifest"))
            self.processEvents()
            captured["path_persisted_after_auto_external"] = window.settings_blind_4d_manifest_edit.text()

            window._language = "en"
            window._apply_language()
            self.processEvents()
            captured["source_label_en"] = window.blind4d_catalog_mode_label.text()
            captured["combo_labels_en"] = [combo.itemText(i) for i in range(combo.count())]

            window.close()
            self.processEvents()
            print(json.dumps(captured, sort_keys=True), flush=True)
            return 0

        zesolver_app.load_persistent_settings = lambda: initial
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

    assert all(payload["standard_attrs_absent"].values())
    assert payload["status_not_in_solver_tab"]
    assert payload["source_label_fr"] == "Source des index Blind 4D"
    assert "bibliothèque ZeSolver active" in payload["help_fr"]
    assert payload["combo_values"] == ["auto", "external-manifest"]
    assert payload["combo_labels_fr"] == ["Auto — bibliothèque active", "Manifeste externe"]
    assert payload["legacy_library_view_maps_to_auto"]
    assert all(payload["hidden_auto"].values())
    assert all(payload["shown_external"].values())
    assert payload["external_settings"]["mode"] == "external-manifest"
    assert payload["external_settings"]["manifest"].endswith("blind4d_manifest.json")
    assert "index" in payload["status_valid_before_change"].lower()
    assert "Non vérifié" in payload["status_after_path_change"]
    assert all(payload["hidden_after_return_auto"].values())
    assert payload["auto_settings"]["mode"] == "auto"
    assert payload["auto_settings"]["manifest"].endswith("diagnostic_manifest.json")
    assert payload["path_persisted_after_auto_external"].endswith("diagnostic_manifest.json")
    assert payload["source_label_en"] == "Blind 4D index source"
    assert payload["combo_labels_en"] == ["Auto — active library", "External manifest"]
