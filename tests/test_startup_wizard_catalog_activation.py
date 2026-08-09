from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_startup_wizard_existing_library_activation_ignores_stale_external_manifest(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    script = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        from dataclasses import replace
        from pathlib import Path
        from types import SimpleNamespace

        from PySide6 import QtWidgets
        from zesolver.catalog_library import CatalogStatus
        from zesolver.catalog_library.models import CatalogCoverage, CoverageStatus, NearCatalogDescriptor
        from zesolver.catalog_library.verification_cache import CatalogVerificationFingerprint
        from zesolver.catalog_resources import SolverCatalogResources
        from zesolver.gui_startup_wizard import StartupAstapProbe, StartupCatalogProbe, StartupWizardDecision
        from zesolver.settings_store import PersistentSettings

        spec = importlib.util.spec_from_file_location("zesolver_app_p3b1f_activation", Path("zesolver/_app.py"))
        appmod = importlib.util.module_from_spec(spec)
        sys.modules["zesolver_app_p3b1f_activation"] = appmod
        spec.loader.exec_module(appmod)

        root = Path({str(tmp_path)!r})
        library = root / "ready-full-library"
        library.mkdir()
        bad_manifest = root / "old-invalid-manifest.json"
        bad_manifest.write_text("{{}}", encoding="utf-8")
        saved = []
        blind_manifest_validations = []

        class FakeLibrary:
            root = library
            def validate(self):
                return SimpleNamespace(status=CatalogStatus.READY_FULL, issues=())

        def fake_resources(**_kwargs):
            coverage = CatalogCoverage(
                status=CoverageStatus.FULL,
                all_sky=True,
                families=("d50",),
                covered_tiles=1476,
                total_tiles=1476,
            )
            near = NearCatalogDescriptor(
                root=library / "sources" / "astap" / "d50",
                families=("d50",),
                formats=("1476-5",),
                coverage=coverage,
                external_reference=False,
            )
            return SolverCatalogResources(
                library_path=library,
                library_status=CatalogStatus.READY_FULL,
                near=near,
                blind4d_indexes=tuple(object() for _ in range(47)),
                blind4d_runtime_paths=tuple(library / f"idx-{{idx}}.npz" for idx in range(47)),
                blind4d_manifest_path=library / "runtime" / "blind4d_manifest.json",
                legacy_index_root=None,
                source="library",
                warnings=(),
                catalog_library_id="ready-full-library",
                coverage=coverage,
                all_sky_blind4d=True,
            )

        def fake_fingerprint(_path):
            return CatalogVerificationFingerprint(
                canonical_library_path=str(library),
                library_id="ready-full-library",
                catalog_manifest_fingerprint="catalog",
                blind4d_view_fingerprint="blind4d",
                runtime_order=tuple(f"idx-{{idx}}" for idx in range(47)),
                blind4d_index_count=47,
                covered_tiles=1476,
                total_tiles=1476,
                all_sky=True,
                fingerprint="fingerprint",
                inspected_file_count=1,
            )

        appmod.validate_catalog_library_root = lambda _path: SimpleNamespace(ok=True)
        appmod.validate_astap_root = lambda _path: SimpleNamespace(ok=True)
        appmod.validate_legacy_near_index_root = lambda _path: SimpleNamespace(ok=True)
        appmod.validate_blind4d_manifest_file = lambda path: blind_manifest_validations.append(str(path)) or SimpleNamespace(ok=False, message="manifest_schema_invalid")
        appmod.CatalogLibrary = SimpleNamespace(open=lambda _path: FakeLibrary())
        appmod.resolve_catalog_resources = fake_resources
        appmod.build_lightweight_catalog_fingerprint = fake_fingerprint

        initial = PersistentSettings(
            interface_mode="expert",
            catalog_library_path=str(library),
            near_catalog_mode="legacy-index",
            blind4d_catalog_mode="external-manifest",
            blind_4d_manifest_path=str(bad_manifest),
            db_root=str(root / "legacy-db"),
            index_root=str(root / "legacy-index"),
            ui_theme="dark",
        )
        (root / "legacy-db").mkdir()
        (root / "legacy-index").mkdir()
        appmod.load_persistent_settings = lambda: initial
        appmod.save_persistent_settings = lambda settings: saved.append(replace(settings))
        appmod.args = appmod.build_arg_parser().parse_args(["--gui"])

        def fake_exec(self):
            self.processEvents()
            window = [w for w in self.topLevelWidgets() if w.__class__.__name__ == "ZeSolverWindow"][0]
            decision = StartupWizardDecision(
                True,
                "fresh",
                StartupCatalogProbe("none"),
                StartupAstapProbe("none"),
                "test",
                False,
                0,
            )
            window._open_startup_wizard(manual=False, decision=decision)
            dialog = window._startup_wizard_dialog
            dialog.existing_radio.setChecked(True)
            dialog.existing_library_edit.setText(str(library))
            dialog._active_operation_signature = dialog._operation_signature("existing_library")
            dialog._on_finished(True, {{"path": str(library)}}, "", "existing_library")
            dialog.accept()
            self.processEvents()
            assert not dialog.isVisible()
            assert blind_manifest_validations == []
            assert saved
            accepted = saved[-1]
            assert accepted.catalog_library_path == str(library)
            assert accepted.near_catalog_mode == "auto"
            assert accepted.blind4d_catalog_mode == "auto"
            assert accepted.blind_4d_manifest_path == str(bad_manifest)
            assert accepted.startup_wizard_completed is True
            assert accepted.ui_theme == "dark"
            assert window._settings.catalog_library_path == str(library)
            assert window._settings.near_catalog_mode == "auto"
            assert window._settings.blind4d_catalog_mode == "auto"
            for source in ("official", "local_package"):
                window._settings.near_catalog_mode = "legacy-index"
                window._settings.blind4d_catalog_mode = "external-manifest"
                window._set_combo_current_data(window.near_catalog_mode_combo, "legacy-index", "legacy-index")
                window._set_combo_current_data(window.blind4d_catalog_mode_combo, "external-manifest", "external-manifest")
                result = window._complete_startup_wizard_transaction(
                    appmod.StartupWizardCompletionRequest(
                        source=source,
                        catalog_library_path=str(library),
                        blind_enabled=True,
                    )
                )
                assert result.ok, result.error
                assert window._settings.catalog_library_path == str(library)
                assert window._settings.near_catalog_mode == "auto"
                assert window._settings.blind4d_catalog_mode == "auto"
            assert blind_manifest_validations == []
            window.close()
            return 0

        QtWidgets.QApplication.exec = fake_exec
        raise SystemExit(appmod.launch_gui(appmod.args))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_startup_wizard_completion_failure_keeps_dialog_open_without_save() -> None:
    pytest.importorskip("PySide6")
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    script = textwrap.dedent(
        """
        from pathlib import Path
        import sys

        from PySide6 import QtWidgets
        from zesolver.gui_startup_wizard import (
            StartupAstapProbe,
            StartupCatalogProbe,
            StartupWizardCompletionResult,
            StartupWizardDecision,
            ZeSolverStartupWizard,
        )
        from zesolver.settings_store import PersistentSettings

        app = QtWidgets.QApplication([])
        messages = []
        QtWidgets.QMessageBox.warning = lambda _parent, _title, text: messages.append(str(text))
        settings = PersistentSettings(catalog_library_path="/old")
        saved = []
        dialog = ZeSolverStartupWizard(
            settings=settings,
            decision=StartupWizardDecision(
                True,
                "fresh",
                StartupCatalogProbe("none"),
                StartupAstapProbe("none"),
                "test",
                False,
                0,
            ),
            save_settings=saved.append,
            completion_handler=lambda _request: StartupWizardCompletionResult(False, "activation failed"),
        )
        dialog.existing_radio.setChecked(True)
        dialog.existing_library_edit.setText("/invalid")
        dialog._active_operation_signature = dialog._operation_signature("existing_library")
        dialog._on_finished(True, {"path": "/invalid"}, "", "existing_library")
        dialog.show()
        app.processEvents()
        dialog.accept()
        app.processEvents()
        assert dialog.isVisible()
        assert saved == []
        assert messages == ["activation failed"]
        assert settings.startup_wizard_completed is False
        dialog.close()
        raise SystemExit(0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
