from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zesolver.gui_startup_wizard import (
    STARTUP_WIZARD_VERSION,
    StartupAstapProbe,
    StartupCatalogProbe,
    clear_invalid_catalog_selection,
    decide_startup_wizard,
    mark_startup_wizard_completed,
    should_allow_legacy_family_prompt,
    startup_default_paths,
)
from zesolver.settings_store import PersistentSettings, load_persistent_settings, save_persistent_settings


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")


def _catalog(state: str):
    return lambda _path: StartupCatalogProbe(state, Path("/catalog") if state != "none" else None)


def _astap(state: str):
    return lambda _path: StartupAstapProbe(state, Path("/astap") if state != "none" else None)


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
