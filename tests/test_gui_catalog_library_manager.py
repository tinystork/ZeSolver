from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_catalog_library_manager_gui_entrypoints_and_selection() -> None:
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

        from PySide6 import QtCore, QtWidgets
        from zesolver.catalog_resources import SolverCatalogResources
        from zesolver.catalog_library import CatalogStatus
        from zesolver.catalog_library.models import CatalogCoverage, CoverageStatus, NearCatalogDescriptor
        from zesolver.settings_store import PersistentSettings

        spec = importlib.util.spec_from_file_location("zesolver_app_p3b1d", Path("zesolver.py"))
        assert spec is not None and spec.loader is not None
        appmod = importlib.util.module_from_spec(spec)
        sys.modules["zesolver_app_p3b1d"] = appmod
        spec.loader.exec_module(appmod)

        root = Path(tempfile.mkdtemp(prefix="p3b1d-gui-"))
        library = root / "managed-library"
        (library / "sources" / "astap" / "d50").mkdir(parents=True)
        (library / "sources" / "astap" / "d50" / "d50_2823.1476").write_bytes(b"fixture-tile")
        (library / "indexes" / "blind4d").mkdir(parents=True)
        index_path = library / "indexes" / "blind4d" / "d50_2823.npz"
        index_path.write_bytes(b"fixture-index")
        import hashlib
        index_sha = hashlib.sha256(b"fixture-index").hexdigest()
        manifest = {
            "schema_version": 1,
            "library_id": "managed-library",
            "created_at": "2026-07-22T00:00:00Z",
            "created_by": "test",
            "minimum_zesolver_version": None,
            "status": "READY_PARTIAL",
            "sources": [{
                "id": "astap-d50",
                "kind": "astap_hnsky",
                "family": "d50",
                "format": "1476-5",
                "path": {"kind": "relative", "value": "sources/astap/d50"},
                "tile_count": 1,
                "layout": "hnsky_1476",
                "coverage": {"status": "FULL", "all_sky": True, "families": ["d50"], "tile_keys": [], "dec_min_deg": -90.0, "dec_max_deg": 90.0, "ra_segments_deg": [[0.0, 360.0]]},
                "integrity": {"files": []},
                "status": "PRESENT",
            }],
            "derived_indexes": [{
                "id": "blind4d-d50-2823",
                "engine": "blind4d",
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "algorithm_version": "astrometry_ab_code_4d_v1",
                "path": {"kind": "relative", "value": "indexes/blind4d/d50_2823.npz"},
                "manifest_path": None,
                "source_ids": ["astap-d50"],
                "source_tiles": ["d50_2823"],
                "coverage": {"status": "PARTIAL", "all_sky": False, "families": ["d50"], "tile_keys": ["d50_2823"]},
                "integrity": {"files": [{"path": "indexes/blind4d/d50_2823.npz", "sha256": index_sha, "size_bytes": len(b"fixture-index")}]},
                "status": "PRESENT",
            }],
            "coverage": {"status": "PARTIAL", "all_sky": False, "families": ["d50"], "tile_keys": ["d50_2823"]},
            "integrity": {"checksum_algorithm": "sha256"},
            "provenance": {"notes": "gui test"},
        }
        (library / "catalog.json").write_text(json.dumps(manifest), encoding="utf-8")
        saved = []

        ok = SimpleNamespace(ok=True)
        appmod.validate_catalog_library_root = lambda _path: ok
        appmod.validate_astap_root = lambda _path: ok
        appmod.validate_legacy_near_index_root = lambda _path: ok
        appmod.validate_blind4d_manifest_file = lambda _path: ok

        def fake_resolve_catalog_resources(**_kwargs):
            coverage = CatalogCoverage(status=CoverageStatus.PARTIAL, all_sky=False, families=("d50",), covered_tiles=1, total_tiles=1476)
            near = NearCatalogDescriptor(root=library, families=("d50",), formats=("1476-5",), coverage=coverage, external_reference=False)
            return SolverCatalogResources(
                library_path=library,
                library_status=CatalogStatus.READY_PARTIAL,
                near=near,
                blind4d_indexes=(),
                blind4d_runtime_paths=(),
                blind4d_manifest_path=None,
                legacy_index_root=None,
                source="library",
                warnings=(),
                catalog_library_id="managed-library",
                coverage=coverage,
                all_sky_blind4d=False,
            )

        appmod.resolve_catalog_resources = fake_resolve_catalog_resources
        appmod.save_persistent_settings = lambda settings: saved.append(settings)

        initial = PersistentSettings(
            interface_mode="expert",
            catalog_library_path=None,
            db_root=str(root / "legacy-db"),
            index_root=str(root / "legacy-index"),
        )
        (root / "legacy-db").mkdir()
        (root / "legacy-index").mkdir()

        class FakeService:
            def detect_astap_families(self, _path):
                return [
                    SimpleNamespace(family="d50", shard_count=1, size_bytes=123, root=Path(_path)),
                    SimpleNamespace(family="d20", shard_count=2, size_bytes=456, root=Path(_path) / "d20"),
                ]

            def analyze_library(self, path):
                item = SimpleNamespace(element="catalog.json", state="ready", detail="schema v1")
                plan = SimpleNamespace(library_root=Path(path), actions=("build_missing_blind4d",), items=(item,))
                return SimpleNamespace(library_root=Path(path), status="NEAR_ONLY", items=(item,), repair_plan=plan)

        def fake_exec(self):
            self.processEvents()
            window = [w for w in self.topLevelWidgets() if w.__class__.__name__ == "ZeSolverWindow"][0]
            captured = {}
            captured["settings_button_text_fr"] = window.settings_catalog_library_manage_btn.text()
            window.settings_catalog_library_manage_btn.click()
            self.processEvents()
            manager = window._catalog_library_manager_dialog
            captured["manager_opened_from_settings"] = manager is not None and manager.isVisible()
            captured["worker_on_open"] = manager._worker is not None
            captured["home_buttons_fr"] = [
                manager.home_install_btn.text(),
                manager.home_create_btn.text(),
                manager.home_repair_btn.text(),
            ]
            manager._service_factory = FakeService
            manager.create_source_edit.setText(str(root / "astap"))
            manager._show_create()
            self.processEvents()
            captured["families"] = [manager.family_list.item(i).data(QtCore.Qt.UserRole) for i in range(manager.family_list.count())]
            for row in range(manager.family_list.count()):
                item = manager.family_list.item(row)
                if item.data(QtCore.Qt.UserRole) == "d20":
                    item.setCheckState(QtCore.Qt.Unchecked)
            manager.standard_radio.setChecked(True)
            captured["standard_selection"] = list(manager._selected_families())
            manager.custom_radio.setChecked(True)
            captured["custom_selection"] = list(manager._selected_families())
            class EmptyService:
                def detect_astap_families(self, _path):
                    return []
            manager._service_factory = EmptyService
            manager.create_source_edit.setText(str(root / "empty-astap"))
            manager._refresh_detected_families()
            captured["empty_create_enabled"] = manager.create_btn.isEnabled()
            captured["empty_message"] = manager.family_list.item(0).text()
            manager._service_factory = FakeService
            manager.create_source_edit.setText(str(root / "astap"))
            manager._refresh_detected_families()
            captured["advanced_visible_initial"] = manager.advanced_box.isVisible()
            manager.custom_radio.setChecked(True)
            captured["advanced_still_folded_custom"] = manager.advanced_box.isVisible()
            manager.advanced_toggle.setChecked(True)
            captured["advanced_visible_after_toggle"] = manager.advanced_box.isVisible()

            first_id = id(manager)
            window.catalog_library_manager_action.trigger()
            self.processEvents()
            captured["same_manager_from_menu"] = first_id == id(window._catalog_library_manager_dialog)
            captured["tools_visible_expert"] = window.tools_menu.menuAction().isVisible()

            manager.repair_library_edit.setText(str(library))
            manager._show_repair()
            manager._analyze_library()
            captured["repair_enabled"] = manager.repair_btn.isEnabled()
            captured["diagnostic_rows"] = manager.diagnostic_table.rowCount()
            manager._on_progress(SimpleNamespace(stage="build_blind4d", message="Building", overall_current=2, overall_total=4, family="d50"))
            captured["progress_value"] = manager.progress.value()

            manager._on_finished(True, SimpleNamespace(library_root=library), "")
            self.processEvents()
            captured["selected_path"] = window.settings_catalog_library_edit.text()
            captured["saved_catalog_path"] = saved[-1].catalog_library_path if saved else None

            window._language = "en"
            window._apply_language()
            captured["manager_title_en"] = manager.windowTitle()
            captured["settings_button_text_en"] = window.settings_catalog_library_manage_btn.text()
            captured["home_buttons_en"] = [
                manager.home_install_btn.text(),
                manager.home_create_btn.text(),
                manager.home_repair_btn.text(),
            ]

            window._interface_mode = "easy"
            window._apply_interface_mode()
            self.processEvents()
            captured["tools_visible_easy"] = window.tools_menu.menuAction().isVisible()
            captured["settings_visible_easy"] = window.tabs.isTabVisible(window.tabs.indexOf(window.settings_scroll))

            class FakeWorker:
                def __init__(self):
                    self.cancelled = False
                    self.waited = False
                def isRunning(self):
                    return True
                def request_cancel(self):
                    self.cancelled = True
                def wait(self, _ms):
                    self.waited = True
                    return True

            fake_worker = FakeWorker()
            manager._worker = fake_worker
            manager.close()
            self.processEvents()
            captured["close_cancelled_worker"] = fake_worker.cancelled and fake_worker.waited

            window.close()
            print(json.dumps(captured, sort_keys=True), flush=True)
            return 0

        appmod.load_persistent_settings = lambda: initial
        QtWidgets.QApplication.exec = fake_exec
        args = appmod.build_arg_parser().parse_args(["--gui"])
        code = appmod.launch_gui(args)
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

    assert payload["settings_button_text_fr"] == "Gérer les bibliothèques…"
    assert payload["manager_opened_from_settings"]
    assert payload["worker_on_open"] is False
    assert payload["home_buttons_fr"] == [
        "Installer une bibliothèque prête à l’emploi",
        "Créer depuis une base ASTAP existante",
        "Vérifier ou réparer une bibliothèque",
    ]
    assert payload["families"] == ["d20", "d50"] or payload["families"] == ["d50", "d20"]
    assert sorted(payload["standard_selection"]) == ["d20", "d50"]
    assert payload["custom_selection"] == ["d50"]
    assert payload["empty_create_enabled"] is False
    assert "ASTAP" in payload["empty_message"]
    assert payload["advanced_visible_initial"] is False
    assert payload["advanced_still_folded_custom"] is False
    assert payload["advanced_visible_after_toggle"] is True
    assert payload["same_manager_from_menu"] is True
    assert payload["tools_visible_expert"] is True
    assert payload["repair_enabled"] is True
    assert payload["diagnostic_rows"] == 1
    assert payload["progress_value"] == 50
    assert payload["selected_path"] == str(Path(payload["saved_catalog_path"]))
    assert payload["manager_title_en"] == "ZeSolver Library Manager"
    assert payload["settings_button_text_en"] == "Manage libraries…"
    assert payload["home_buttons_en"] == [
        "Install a ready-to-use library",
        "Create from an existing ASTAP database",
        "Verify or repair a library",
    ]
    assert payload["tools_visible_easy"] is False
    assert payload["settings_visible_easy"] is True
    assert payload["close_cancelled_worker"] is True
