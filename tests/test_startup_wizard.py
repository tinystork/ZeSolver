from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from zesolver.gui_startup_wizard import (
    STARTUP_WIZARD_VERSION,
    StartupAstapProbe,
    StartupCatalogProbe,
    StartupWizardDecision,
    clear_invalid_catalog_selection,
    decide_startup_wizard,
    mark_startup_wizard_completed,
    should_allow_legacy_family_prompt,
    startup_default_paths,
)
from zesolver.settings_store import PersistentSettings, load_persistent_settings, save_persistent_settings


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")
_QT_APP = None


def _catalog(state: str):
    return lambda _path: StartupCatalogProbe(state, Path("/catalog") if state != "none" else None)


def _astap(state: str):
    return lambda _path: StartupAstapProbe(state, Path("/astap") if state != "none" else None)


@pytest.fixture()
def qt_widgets(monkeypatch: pytest.MonkeyPatch):
    global _QT_APP
    if not any(str(arg).replace("\\", "/").endswith("tests/test_startup_wizard.py") for arg in sys.argv):
        pytest.skip("startup wizard Qt widget tests run in an explicit isolated test invocation")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    _QT_APP = QtWidgets.QApplication.instance() or _QT_APP or QtWidgets.QApplication([])
    return QtWidgets, _QT_APP


def _fresh_decision() -> StartupWizardDecision:
    return StartupWizardDecision(
        True,
        "fresh",
        StartupCatalogProbe("none"),
        StartupAstapProbe("none"),
        "test",
        False,
        0,
    )


def _wizard(qt_widgets, settings: PersistentSettings | None = None, decision: StartupWizardDecision | None = None):
    from zesolver.gui_startup_wizard import ZeSolverStartupWizard

    saved: list[PersistentSettings] = []
    dialog = ZeSolverStartupWizard(
        settings=settings or PersistentSettings(),
        decision=decision or _fresh_decision(),
        save_settings=saved.append,
    )
    return dialog, saved


def _capture_information(monkeypatch: pytest.MonkeyPatch, qt_widgets) -> list[str]:
    QtWidgets, _app = qt_widgets
    messages: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(str(text)),
    )
    return messages


def test_fresh_profile_schedules_startup_wizard_without_probe_work() -> None:
    def fail_catalog(_path):
        raise AssertionError("catalog probe should not run for a fresh profile")

    def fail_astap(_path):
        raise AssertionError("astap probe should not run for a fresh profile")

    decision = decide_startup_wizard(PersistentSettings(), catalog_probe=fail_catalog, astap_probe=fail_astap)

    assert decision.should_show
    assert decision.mode == "fresh"
    assert decision.reason == "fresh profile without configured catalog"


def test_source_schedules_wizard_after_window_show() -> None:
    assert "window.show()\n    window.schedule_startup_wizard_if_needed()" in SOURCE
    assert "QtCore.QTimer.singleShot(max(0, int(delay_ms)), _open_if_needed)" in SOURCE


def test_source_suppresses_legacy_rebuild_prompt_before_message_box() -> None:
    guard = SOURCE.index("if not should_allow_legacy_family_prompt")
    prompt = SOURCE.index("self._text(\"db_family_prompt_title\")")
    assert guard < prompt


def test_ready_full_completed_does_not_force_download() -> None:
    settings = PersistentSettings(
        catalog_library_path="/catalog",
        startup_wizard_completed=True,
        startup_wizard_version=STARTUP_WIZARD_VERSION,
    )

    decision = decide_startup_wizard(settings, catalog_probe=_catalog("ready_full"))

    assert not decision.should_show
    assert decision.mode == "ready"
    assert decision.has_ready_full_library


def test_ready_full_not_completed_can_finish_or_modify_without_download_requirement() -> None:
    settings = PersistentSettings(catalog_library_path="/catalog")

    decision = decide_startup_wizard(settings, catalog_probe=_catalog("ready_full"))

    assert decision.should_show
    assert decision.mode == "ready"
    assert decision.reason == "catalog library is ready but wizard not completed"


def test_invalid_library_goes_to_repair_path_even_if_completed() -> None:
    settings = PersistentSettings(
        catalog_library_path="/missing",
        startup_wizard_completed=True,
        startup_wizard_version=STARTUP_WIZARD_VERSION,
    )

    decision = decide_startup_wizard(settings, catalog_probe=_catalog("missing"))

    assert decision.should_show
    assert decision.requires_repair


def test_astap_only_is_near_only_offer() -> None:
    settings = PersistentSettings(db_root="/opt/astap")

    decision = decide_startup_wizard(settings, astap_probe=_astap("valid"))

    assert decision.should_show
    assert decision.has_astap_near_only
    assert decision.mode == "astap_near_only"


def test_configure_later_marks_completed_without_invalid_catalog_path() -> None:
    settings = PersistentSettings(catalog_library_path="/broken")

    clear_invalid_catalog_selection(settings)
    mark_startup_wizard_completed(settings)

    assert settings.catalog_library_path is None
    assert settings.catalog_library_verification is None
    assert settings.startup_wizard_completed is True
    assert settings.startup_wizard_version == STARTUP_WIZARD_VERSION


def test_completed_empty_profile_after_configure_later_does_not_relaunch() -> None:
    settings = PersistentSettings(
        startup_wizard_completed=True,
        startup_wizard_version=STARTUP_WIZARD_VERSION,
    )

    decision = decide_startup_wizard(settings)

    assert not decision.should_show
    assert decision.mode == "later"


def test_menu_action_relaunches_startup_wizard() -> None:
    assert "self.interface_wizard_action.triggered.connect(self._run_startup_wizard_from_menu)" in SOURCE
    assert "def _run_startup_wizard_from_menu(self) -> None:\n            self._open_startup_wizard(manual=True)" in SOURCE
    manual_branch = SOURCE.index("if manual:")
    constructor = SOURCE.index("dialog = ZeSolverStartupWizard", manual_branch)
    assert "decision = decide_startup_wizard(self._settings)" in SOURCE[manual_branch:constructor]


def test_legacy_prompt_requires_explicit_legacy_index_mode() -> None:
    assert not should_allow_legacy_family_prompt(PersistentSettings(near_catalog_mode="auto"))
    assert should_allow_legacy_family_prompt(PersistentSettings(near_catalog_mode="legacy-index"))
    ready = decide_startup_wizard(
        PersistentSettings(catalog_library_path="/catalog", near_catalog_mode="legacy-index"),
        catalog_probe=_catalog("ready_full"),
    )
    assert not should_allow_legacy_family_prompt(PersistentSettings(near_catalog_mode="legacy-index"), ready)


def test_cross_platform_default_destinations_are_available() -> None:
    linux = startup_default_paths(platform_name="Linux", env={"HOME": "/home/alice"})
    windows = startup_default_paths(platform_name="Windows", env={"USERPROFILE": r"C:\Users\Alice"})
    mac = startup_default_paths(platform_name="Darwin", env={"HOME": "/Users/alice"})

    assert linux["library_parent"] == Path("/home/alice/ZeSolverCatalog/libraries")
    assert str(windows["library_parent"]).replace("/", "\\").endswith(r"ZeSolverCatalog\libraries")
    assert mac["cache_root"] == Path("/Users/alice/Library/Caches/ZeSolver/catalogs")


def test_settings_roundtrip_preserves_existing_fields_and_wizard_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.settings_store as store

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "_resolve_settings_path", lambda: settings_file)

    original = PersistentSettings(
        catalog_library_path="/catalog",
        db_root="/astap",
        index_root="/legacy-index",
        solver_search_scale=1.5,
        startup_wizard_completed=True,
        startup_wizard_version=STARTUP_WIZARD_VERSION,
    )
    save_persistent_settings(original)

    loaded = load_persistent_settings()

    assert loaded.catalog_library_path == "/catalog"
    assert loaded.db_root == "/astap"
    assert loaded.index_root == "/legacy-index"
    assert loaded.solver_search_scale == pytest.approx(1.5)
    assert loaded.startup_wizard_completed is True
    assert loaded.startup_wizard_version == STARTUP_WIZARD_VERSION


def test_worker_progress_and_cancellation_without_network() -> None:
    pytest.importorskip("PySide6")
    from PySide6 import QtCore
    from zesolver.catalog_library.distribution import DistributionCancelled
    from zesolver.gui_startup_wizard import StartupCatalogWorker

    class FakeService:
        def __init__(self, *, progress_callback=None, cancel_callback=None, **_kwargs):
            self.progress_callback = progress_callback
            self.cancel_callback = cancel_callback

        def fetch_latest_distribution(self):
            return object(), SimpleNamespace(library_id="fixture", version="1.0")

        def build_install_plan(self, release, manifest, *, parent=None, destination=None):
            del release, destination
            return SimpleNamespace(manifest=manifest, cache_dir=Path("/cache"), components=(), destination=Path(parent or "/tmp/lib"))

        def build_storage_plan(self, _plan):
            return SimpleNamespace(requirements=())

        def install_distribution(self, _plan):
            if self.progress_callback is not None:
                self.progress_callback(SimpleNamespace(stage="download", message="Downloading", overall_current=1, overall_total=2))
            if self.cancel_callback and self.cancel_callback():
                raise DistributionCancelled()
            return SimpleNamespace(library_result=SimpleNamespace(library_root=Path("/installed/library")))

    progress = []
    finished = []
    worker = StartupCatalogWorker("official_install", {"install_parent": "/tmp"}, distribution_factory=FakeService)
    worker.progress.connect(progress.append, QtCore.Qt.DirectConnection)
    worker.finished.connect(lambda ok, result, error, op: finished.append((ok, result, error, op)), QtCore.Qt.DirectConnection)
    worker.request_cancel()
    worker.run()

    assert progress
    assert finished and finished[0][0] is False
    assert finished[0][3] == "official_install"


def test_successful_install_path_is_forwarded_to_main_window_handler() -> None:
    selected = SOURCE.index("dialog.librarySelected.connect(self._on_startup_wizard_library_selected)")
    handler = SOURCE.index("def _on_startup_wizard_library_selected")
    apply_existing = SOURCE.index("self._on_catalog_library_manager_selected(value)")
    assert selected < handler < apply_existing
    assert "self._update_simplified_capability_summary()" in SOURCE[handler : handler + 1200]


def test_astap_browse_selection_updates_visible_and_internal_path(qt_widgets, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    QtWidgets, _app = qt_widgets
    astap = tmp_path / "astap"
    astap.mkdir()
    dialog, _saved = _wizard(qt_widgets)
    dialog.astap_radio.setChecked(True)

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *_args: str(astap))
    dialog.astap_browse.click()

    assert Path(dialog.astap_edit.text()) == astap
    assert dialog._current_astap_path() == str(astap)
    assert not dialog._is_current_astap_validated()


def test_astap_manual_entry_validation_then_finish_persists_near_only(qt_widgets, tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    settings = PersistentSettings(catalog_library_path=None, db_root=None)
    dialog, saved = _wizard(qt_widgets, settings=settings)
    selected: list[str] = []
    dialog.astapSelected.connect(lambda path: (selected.append(path), setattr(settings, "db_root", path), setattr(settings, "near_catalog_mode", "astap-native")))
    dialog.astap_radio.setChecked(True)
    dialog.astap_edit.setText(str(astap))

    dialog._on_finished(True, {"path": str(astap), "families": ("d50",)}, "", "astap")

    assert dialog._is_current_astap_validated()
    assert settings.db_root is None

    dialog.accept()

    assert selected == [str(astap)]
    assert settings.db_root == str(astap)
    assert settings.near_catalog_mode == "astap-native"
    assert saved and saved[-1].startup_wizard_completed is True


def test_astap_start_operation_validates_current_visible_field(qt_widgets, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import zesolver.gui_startup_wizard as wizard_module

    astap = tmp_path / "astap"
    astap.mkdir()
    captured: dict[str, object] = {}

    class FakeSignal:
        def __init__(self) -> None:
            self._slots = []

        def connect(self, slot) -> None:
            self._slots.append(slot)

        def emit(self, *args) -> None:
            for slot in list(self._slots):
                slot(*args)

    class FakeWorker:
        def __init__(self, operation, payload, **_kwargs) -> None:
            self.operation = operation
            self.payload = payload
            self.progress = FakeSignal()
            self.discovered = FakeSignal()
            self.finished = FakeSignal()

        def isRunning(self) -> bool:
            return False

        def start(self) -> None:
            captured["operation"] = self.operation
            captured["payload"] = dict(self.payload)
            self.finished.emit(True, {"path": self.payload["path"], "families": ("d50",)}, "", self.operation)

    monkeypatch.setattr(wizard_module, "StartupCatalogWorker", FakeWorker)
    settings = PersistentSettings(db_root="/old/astap")
    dialog, saved = _wizard(qt_widgets, settings=settings)
    selected: list[str] = []
    dialog.astapSelected.connect(selected.append)
    dialog.astap_radio.setChecked(True)
    dialog.astap_edit.setText(str(astap))

    dialog._start_operation()
    dialog.accept()

    assert captured == {"operation": "astap", "payload": {"path": str(astap)}}
    assert selected == [str(astap)]
    assert saved and saved[-1].startup_wizard_completed is True


def test_astap_validation_success_is_tied_to_exact_path(qt_widgets, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    messages = _capture_information(monkeypatch, qt_widgets)
    first = tmp_path / "astap-a"
    second = tmp_path / "astap-b"
    first.mkdir()
    second.mkdir()
    dialog, saved = _wizard(qt_widgets)
    selected: list[str] = []
    dialog.astapSelected.connect(selected.append)
    dialog.astap_radio.setChecked(True)
    dialog.astap_edit.setText(str(first))
    dialog._on_finished(True, {"path": str(first), "families": ("d50",)}, "", "astap")

    dialog.astap_edit.setText(str(second))
    dialog.accept()

    assert not dialog._is_current_astap_validated()
    assert selected == []
    assert saved == []
    assert messages == ["Validez la base ASTAP avant de terminer."]


def test_astap_invalid_or_unvalidated_path_cannot_finish(qt_widgets, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    messages = _capture_information(monkeypatch, qt_widgets)
    dialog, saved = _wizard(qt_widgets)
    selected: list[str] = []
    dialog.astapSelected.connect(selected.append)
    dialog.astap_radio.setChecked(True)
    dialog.astap_edit.setText(str(tmp_path / "missing-astap"))
    dialog._on_finished(False, None, "ASTAP_SOURCE_MISSING", "astap")

    dialog.accept()

    assert selected == []
    assert saved == []
    assert messages == ["Validez la base ASTAP avant de terminer."]


def test_astap_path_is_not_persisted_until_finish(qt_widgets, tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    settings = PersistentSettings(db_root="/old/astap", index_root="/old/index")
    dialog, saved = _wizard(qt_widgets, settings=settings)
    dialog.astap_radio.setChecked(True)
    dialog.astap_edit.setText(str(astap))

    dialog._on_finished(True, {"path": str(astap), "families": ("d50",)}, "", "astap")

    assert settings.db_root == "/old/astap"
    assert settings.index_root == "/old/index"
    assert saved == []


def test_astap_cancel_after_validation_preserves_existing_settings(qt_widgets, tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    settings = PersistentSettings(db_root="/old/astap", index_root="/old/index", catalog_library_path="/old/library")
    dialog, saved = _wizard(qt_widgets, settings=settings)
    dialog.astap_radio.setChecked(True)
    dialog.astap_edit.setText(str(astap))
    dialog._on_finished(True, {"path": str(astap), "families": ("d50",)}, "", "astap")

    dialog.reject()

    assert settings.db_root == "/old/astap"
    assert settings.index_root == "/old/index"
    assert settings.catalog_library_path == "/old/library"
    assert saved == []


def test_astap_near_only_path_preserves_existing_library_on_successful_finish(qt_widgets, tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    settings = PersistentSettings(catalog_library_path="/old/library", db_root="/old/astap")
    dialog, saved = _wizard(qt_widgets, settings=settings)

    def apply_astap(path: str) -> None:
        settings.db_root = path
        settings.near_catalog_mode = "astap-native"
        settings.blind4d_catalog_mode = "auto"

    dialog.astapSelected.connect(apply_astap)
    dialog.astap_radio.setChecked(True)
    dialog.astap_edit.setText(str(astap))
    dialog._on_finished(True, {"path": str(astap), "families": ("d50",)}, "", "astap")
    dialog.accept()

    assert settings.catalog_library_path == "/old/library"
    assert settings.db_root == str(astap)
    assert settings.near_catalog_mode == "astap-native"
    assert settings.blind4d_catalog_mode == "auto"
    assert saved and saved[-1].startup_wizard_completed is True


def test_startup_wizard_astap_handler_does_not_clear_catalog_library() -> None:
    handler = SOURCE.index("def _on_startup_wizard_astap_selected")
    body = SOURCE[handler : SOURCE.index("def _on_startup_wizard_completed", handler)]

    assert "self._clear_catalog_library_selection()" not in body
    assert 'self._settings.near_catalog_mode = "astap-native"' in body
    assert 'self._settings.blind4d_catalog_mode = "auto"' in body


def test_auto_blind_runtime_does_not_reuse_stale_external_manifest_path() -> None:
    resolver = SOURCE.index("def _resolve_blind4d_runtime")
    body = SOURCE[resolver : SOURCE.index("    @staticmethod", resolver)]

    assert "requested = Blind4DCatalogMode.normalize(mode)" in body
    assert "if requested is Blind4DCatalogMode.EXTERNAL_MANIFEST" in body
    assert "else self.catalog_resources.blind4d_manifest_path" in body


def test_initial_valid_astap_decision_can_finish_without_download(qt_widgets, tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    settings = PersistentSettings(db_root=str(astap))
    decision = StartupWizardDecision(
        True,
        "astap_near_only",
        StartupCatalogProbe("none"),
        StartupAstapProbe("valid", astap),
        "test",
        False,
        0,
    )
    dialog, saved = _wizard(qt_widgets, settings=settings, decision=decision)
    selected: list[str] = []
    dialog.astapSelected.connect(selected.append)

    dialog.accept()

    assert selected == [str(astap)]
    assert saved and saved[-1].startup_wizard_completed is True


def test_startup_wizard_tests_cover_legacy_prompt_suppression() -> None:
    assert "Nouvelles bases détectées" in SOURCE
    assert "Legacy index rebuild prompt suppressed; active catalog mode is not explicit legacy-index." in SOURCE
