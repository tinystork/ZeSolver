"""First-run startup wizard for ZeSolver.

The decision layer in this module is GUI-free so tests can exercise startup
policy without constructing a QApplication.  The Qt dialog below only presents
those decisions and delegates catalog work to existing catalog services.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .catalog_library import (
    CatalogDistributionService,
    CatalogLibrary,
    CatalogLibraryError,
    CatalogLibraryManagementCancelled,
    CatalogLibraryManagementError,
    CatalogLibraryManagementService,
    CatalogStatus,
    DistributionCancelled,
    DistributionError,
    LibraryInstallOptions,
    build_storage_plan,
    default_cache_root,
    default_library_parent,
    format_bytes_binary,
    resolve_library_destination,
    validate_library_parent,
)
from .catalog_resources import resolve_catalog_resources


STARTUP_WIZARD_VERSION = 1

CatalogState = Literal["none", "missing", "invalid", "ready_full", "ready_partial", "near_only"]
AstapState = Literal["none", "missing", "invalid", "valid"]
StartupWizardMode = Literal["fresh", "ready", "repair", "astap_near_only", "later"]
CatalogSourceChoice = Literal["official", "existing_library", "astap", "local_package", "later"]


@dataclass(frozen=True, slots=True)
class StartupCatalogProbe:
    state: CatalogState
    path: Path | None = None
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.state in {"ready_full", "ready_partial", "near_only"}


@dataclass(frozen=True, slots=True)
class StartupAstapProbe:
    state: AstapState
    path: Path | None = None
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.state == "valid"


@dataclass(frozen=True, slots=True)
class StartupWizardDecision:
    should_show: bool
    mode: StartupWizardMode
    catalog: StartupCatalogProbe
    astap: StartupAstapProbe
    reason: str
    completed: bool
    version: int

    @property
    def requires_repair(self) -> bool:
        return self.mode == "repair"

    @property
    def has_ready_full_library(self) -> bool:
        return self.catalog.state == "ready_full"

    @property
    def has_astap_near_only(self) -> bool:
        return self.mode == "astap_near_only"


def default_catalog_probe(path: str | Path | None) -> StartupCatalogProbe:
    if not path:
        return StartupCatalogProbe("none")
    root = Path(path).expanduser()
    if not root.exists():
        return StartupCatalogProbe("missing", root, f"missing: {root}")
    if not root.is_dir():
        return StartupCatalogProbe("invalid", root, f"not a directory: {root}")
    catalog_json = root / "catalog.json"
    if not catalog_json.is_file():
        return StartupCatalogProbe("invalid", root, "catalog.json missing")
    try:
        library = CatalogLibrary.open(root)
        report = library.validate()
    except Exception as exc:
        return StartupCatalogProbe("invalid", root, str(exc))
    if report.status is CatalogStatus.READY_FULL:
        return StartupCatalogProbe("ready_full", root, report.status.value)
    if report.status in {CatalogStatus.READY_PARTIAL, CatalogStatus.BLIND4D_ONLY}:
        return StartupCatalogProbe("ready_partial", root, report.status.value)
    if report.status in {CatalogStatus.NEAR_ONLY, CatalogStatus.SOURCE_ONLY}:
        return StartupCatalogProbe("near_only", root, report.status.value)
    return StartupCatalogProbe("invalid", root, report.status.value)


def default_astap_probe(path: str | Path | None) -> StartupAstapProbe:
    if not path:
        return StartupAstapProbe("none")
    root = Path(path).expanduser()
    if not root.exists():
        return StartupAstapProbe("missing", root, f"missing: {root}")
    if not root.is_dir():
        return StartupAstapProbe("invalid", root, f"not a directory: {root}")
    # Keep this light for startup. Full family detection stays in workers/tools.
    return StartupAstapProbe("valid", root, "directory present")


def decide_startup_wizard(
    settings: Any,
    *,
    catalog_probe: Callable[[str | Path | None], StartupCatalogProbe] = default_catalog_probe,
    astap_probe: Callable[[str | Path | None], StartupAstapProbe] = default_astap_probe,
    required_version: int = STARTUP_WIZARD_VERSION,
) -> StartupWizardDecision:
    completed = bool(getattr(settings, "startup_wizard_completed", False))
    version = int(getattr(settings, "startup_wizard_version", 0) or 0)
    catalog_path = getattr(settings, "catalog_library_path", None)
    astap_path = getattr(settings, "db_root", None)

    catalog = catalog_probe(catalog_path) if catalog_path else StartupCatalogProbe("none")
    astap = astap_probe(astap_path) if astap_path else StartupAstapProbe("none")

    if catalog.state in {"missing", "invalid"}:
        return StartupWizardDecision(True, "repair", catalog, astap, "configured catalog library is unusable", completed, version)
    if catalog.usable:
        if completed and version >= required_version:
            return StartupWizardDecision(False, "ready", catalog, astap, "wizard already completed and library usable", completed, version)
        return StartupWizardDecision(True, "ready", catalog, astap, "catalog library is ready but wizard not completed", completed, version)

    if astap.usable:
        if completed and version >= required_version:
            return StartupWizardDecision(False, "astap_near_only", catalog, astap, "wizard already completed with ASTAP-only catalog", completed, version)
        return StartupWizardDecision(True, "astap_near_only", catalog, astap, "ASTAP catalog available without ZeSolver library", completed, version)

    if completed and version >= required_version:
        return StartupWizardDecision(False, "later", catalog, astap, "wizard completed without catalog; user chose to configure later", completed, version)
    return StartupWizardDecision(True, "fresh", catalog, astap, "fresh profile without configured catalog", completed, version)


def should_allow_legacy_family_prompt(settings: Any, decision: StartupWizardDecision | None = None) -> bool:
    mode = str(getattr(settings, "near_catalog_mode", "auto") or "auto").strip().lower().replace("_", "-")
    if mode != "legacy-index":
        return False
    if decision is not None and decision.catalog.usable:
        return False
    return True


def mark_startup_wizard_completed(settings: Any, *, version: int = STARTUP_WIZARD_VERSION) -> Any:
    settings.startup_wizard_version = int(version)
    settings.startup_wizard_completed = True
    return settings


def clear_invalid_catalog_selection(settings: Any) -> Any:
    settings.catalog_library_path = None
    settings.catalog_library_verification = None
    return settings


def startup_default_paths(
    *,
    platform_name: str | None = None,
    home: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Path]:
    return {
        "library_parent": default_library_parent(platform_name=platform_name, home=home, env=env),
        "cache_root": default_cache_root(platform_name=platform_name, home=home, env=env),
    }


try:  # pragma: no cover - exercised by GUI subprocess tests
    from PySide6 import QtCore, QtWidgets
except Exception:  # pragma: no cover
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]


if QtCore is not None:

    class StartupCatalogWorker(QtCore.QThread):
        progress = QtCore.Signal(object)
        discovered = QtCore.Signal(object, object, object)
        finished = QtCore.Signal(bool, object, str, str)

        def __init__(
            self,
            operation: str,
            payload: dict[str, Any],
            *,
            distribution_factory: Callable[..., CatalogDistributionService] = CatalogDistributionService,
            management_factory: Callable[..., CatalogLibraryManagementService] = CatalogLibraryManagementService,
        ) -> None:
            super().__init__()
            self.operation = str(operation)
            self.payload = dict(payload)
            self._distribution_factory = distribution_factory
            self._management_factory = management_factory
            self._cancel_event = threading.Event()

        def request_cancel(self) -> None:
            self._cancel_event.set()

        def run(self) -> None:
            try:
                result = self._run_operation()
                self.finished.emit(True, result, "", self.operation)
            except (DistributionCancelled, CatalogLibraryManagementCancelled) as exc:
                self.finished.emit(False, None, str(exc), self.operation)
            except (DistributionError, CatalogLibraryManagementError, CatalogLibraryError, Exception) as exc:
                self.finished.emit(False, None, str(exc), self.operation)

        def _run_operation(self) -> object:
            if self.operation == "official_install":
                service = self._distribution_factory(
                    progress_callback=self.progress.emit,
                    cancel_callback=self._cancel_event.is_set,
                )
                release, manifest = service.fetch_latest_distribution()
                parent_value = self.payload.get("install_parent")
                parent = Path(parent_value).expanduser() if parent_value else None
                plan = service.build_install_plan(release, manifest, parent=parent)
                storage = service.build_storage_plan(plan)
                self.discovered.emit(release, manifest, storage)
                return service.install_distribution(plan)
            if self.operation == "existing_library":
                path = Path(self.payload["path"]).expanduser()
                library = CatalogLibrary.open(path)
                resources = resolve_catalog_resources(catalog_library=library)
                return {"path": str(path), "resources": resources}
            if self.operation == "astap":
                root = Path(self.payload["path"]).expanduser()
                service = self._management_factory(
                    progress_callback=self.progress.emit,
                    cancel_callback=self._cancel_event.is_set,
                )
                families = service.detect_astap_families(root)
                if not families:
                    raise CatalogLibraryManagementError(f"ASTAP_SOURCE_EMPTY: {root}")
                return {"path": str(root), "families": families}
            if self.operation == "local_package":
                service = self._management_factory(
                    progress_callback=self.progress.emit,
                    cancel_callback=self._cancel_event.is_set,
                )
                result = service.install_package(
                    LibraryInstallOptions(
                        package_path=Path(self.payload["package_path"]).expanduser(),
                        destination=Path(self.payload["destination"]).expanduser(),
                    )
                )
                return result
            raise CatalogLibraryManagementError(f"UNKNOWN_STARTUP_OPERATION: {self.operation}")

else:  # pragma: no cover - import surface for non-GUI CLI environments
    StartupCatalogWorker = None  # type: ignore[assignment]


if QtWidgets is not None:

    class ZeSolverStartupWizard(QtWidgets.QWizard):
        librarySelected = QtCore.Signal(str)
        astapSelected = QtCore.Signal(str)
        completed = QtCore.Signal(str)

        def __init__(
            self,
            *,
            settings: Any,
            decision: StartupWizardDecision,
            save_settings: Callable[[Any], None],
            parent: Any = None,
            distribution_factory: Callable[..., CatalogDistributionService] = CatalogDistributionService,
            management_factory: Callable[..., CatalogLibraryManagementService] = CatalogLibraryManagementService,
        ) -> None:
            super().__init__(parent)
            self.settings = settings
            self.decision = decision
            self._save_settings = save_settings
            self._distribution_factory = distribution_factory
            self._management_factory = management_factory
            self._worker: StartupCatalogWorker | None = None
            self._operation_completed = False
            self._selected_choice: CatalogSourceChoice = "official"
            self.setWindowTitle("Assistant de demarrage ZeSolver")
            self.setWizardStyle(QtWidgets.QWizard.ModernStyle)
            self.setOption(QtWidgets.QWizard.NoBackButtonOnStartPage, True)
            self.setButtonText(QtWidgets.QWizard.CustomButton1, "Configurer plus tard")
            self.setOption(QtWidgets.QWizard.HaveCustomButton1, True)
            self.customButtonClicked.connect(self._on_custom_button)
            self._build_pages()
            self._sync_choice_visibility()

        def _build_pages(self) -> None:
            self.addPage(self._welcome_page())
            self.addPage(self._source_page())
            self.addPage(self._destination_page())
            self.addPage(self._progress_page())
            self.addPage(self._settings_page())
            self.addPage(self._summary_page())

        def _welcome_page(self) -> Any:
            page = QtWidgets.QWizardPage()
            page.setTitle("Bienvenue")
            layout = QtWidgets.QVBoxLayout(page)
            self.diagnostic_label = QtWidgets.QLabel(self._diagnostic_text())
            self.diagnostic_label.setWordWrap(True)
            layout.addWidget(self.diagnostic_label)
            return page

        def _source_page(self) -> Any:
            page = QtWidgets.QWizardPage()
            page.setTitle("Source catalogue")
            layout = QtWidgets.QVBoxLayout(page)
            self.official_radio = QtWidgets.QRadioButton("Installer la Bibliotheque ZeSolver officielle")
            self.existing_radio = QtWidgets.QRadioButton("Utiliser une Bibliotheque ZeSolver existante")
            self.astap_radio = QtWidgets.QRadioButton("Utiliser une base ASTAP existante - ZeNear uniquement")
            self.package_radio = QtWidgets.QRadioButton("Installer un paquet local - avance")
            for radio in (self.official_radio, self.existing_radio, self.astap_radio, self.package_radio):
                layout.addWidget(radio)
                radio.toggled.connect(self._sync_choice_visibility)
            if self.decision.mode == "ready":
                self.existing_radio.setChecked(True)
            elif self.decision.mode == "astap_near_only":
                self.astap_radio.setChecked(True)
            else:
                self.official_radio.setChecked(True)
            return page

        def _destination_page(self) -> Any:
            page = QtWidgets.QWizardPage()
            page.setTitle("Destination")
            layout = QtWidgets.QFormLayout(page)
            parent = getattr(self.settings, "catalog_library_install_parent", None) or str(default_library_parent())
            self.install_parent_edit = QtWidgets.QLineEdit(str(Path(parent).expanduser()))
            self.install_parent_browse = QtWidgets.QPushButton("Parcourir")
            self.install_parent_browse.clicked.connect(lambda: self._pick_directory(self.install_parent_edit))
            layout.addRow("Dossier parent", self._row(self.install_parent_edit, self.install_parent_browse))
            self.existing_library_edit = QtWidgets.QLineEdit(str(getattr(self.settings, "catalog_library_path", "") or ""))
            self.existing_library_browse = QtWidgets.QPushButton("Parcourir")
            self.existing_library_browse.clicked.connect(lambda: self._pick_directory(self.existing_library_edit))
            layout.addRow("Bibliotheque existante", self._row(self.existing_library_edit, self.existing_library_browse))
            self.astap_edit = QtWidgets.QLineEdit(str(getattr(self.settings, "db_root", "") or ""))
            self.astap_browse = QtWidgets.QPushButton("Parcourir")
            self.astap_browse.clicked.connect(lambda: self._pick_directory(self.astap_edit))
            layout.addRow("Base ASTAP", self._row(self.astap_edit, self.astap_browse))
            self.package_edit = QtWidgets.QLineEdit("")
            self.package_browse = QtWidgets.QPushButton("Parcourir")
            self.package_browse.clicked.connect(lambda: self._pick_file(self.package_edit))
            layout.addRow("Paquet local", self._row(self.package_edit, self.package_browse))
            self.package_destination_edit = QtWidgets.QLineEdit("")
            self.package_destination_browse = QtWidgets.QPushButton("Parcourir")
            self.package_destination_browse.clicked.connect(lambda: self._pick_directory(self.package_destination_edit))
            layout.addRow("Destination paquet", self._row(self.package_destination_edit, self.package_destination_browse))
            self.storage_label = QtWidgets.QLabel("")
            self.storage_label.setWordWrap(True)
            layout.addRow("Espace disque", self.storage_label)
            self.install_parent_edit.textChanged.connect(self._refresh_storage_preview)
            self._refresh_storage_preview()
            return page

        def _progress_page(self) -> Any:
            page = QtWidgets.QWizardPage()
            page.setTitle("Installation et validation")
            layout = QtWidgets.QVBoxLayout(page)
            self.progress_label = QtWidgets.QLabel("Pret.")
            self.progress_label.setWordWrap(True)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            self.progress_log = QtWidgets.QPlainTextEdit()
            self.progress_log.setReadOnly(True)
            self.progress_log.document().setMaximumBlockCount(500)
            row = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Demarrer")
            self.cancel_btn = QtWidgets.QPushButton("Annuler")
            self.cancel_btn.setEnabled(False)
            self.start_btn.clicked.connect(self._start_operation)
            self.cancel_btn.clicked.connect(self._cancel_worker)
            row.addWidget(self.start_btn)
            row.addWidget(self.cancel_btn)
            row.addStretch(1)
            layout.addWidget(self.progress_label)
            layout.addWidget(self.progress)
            layout.addLayout(row)
            layout.addWidget(self.progress_log, 1)
            return page

        def _settings_page(self) -> Any:
            page = QtWidgets.QWizardPage()
            page.setTitle("Dossier images et reglages")
            layout = QtWidgets.QFormLayout(page)
            self.sample_fits_edit = QtWidgets.QLineEdit(str(getattr(self.settings, "sample_fits", "") or ""))
            self.sample_fits_browse = QtWidgets.QPushButton("Parcourir")
            self.sample_fits_browse.clicked.connect(lambda: self._pick_directory(self.sample_fits_edit))
            layout.addRow("Dossier d'images", self._row(self.sample_fits_edit, self.sample_fits_browse))
            self.blind_enabled_check = QtWidgets.QCheckBox("Activer ZeBlind si la bibliotheque complete est disponible")
            self.blind_enabled_check.setChecked(bool(getattr(self.settings, "solver_blind_enabled", True)))
            layout.addRow(self.blind_enabled_check)
            return page

        def _summary_page(self) -> Any:
            page = QtWidgets.QWizardPage()
            page.setTitle("Resume")
            layout = QtWidgets.QVBoxLayout(page)
            self.summary_label = QtWidgets.QLabel("")
            self.summary_label.setWordWrap(True)
            layout.addWidget(self.summary_label)
            page.initializePage = self._update_summary  # type: ignore[method-assign]
            return page

        def _diagnostic_text(self) -> str:
            if self.decision.mode == "ready":
                return "Une bibliotheque ZeSolver est deja configuree. Vous pouvez terminer maintenant ou modifier l'installation."
            if self.decision.mode == "repair":
                return f"La configuration catalogue memorisee doit etre reparee: {self.decision.reason}."
            if self.decision.mode == "astap_near_only":
                return "Une base ASTAP est configuree. ZeNear peut fonctionner, mais la bibliotheque complete reste recommandee."
            return "Aucun catalogue exploitable n'est configure pour ce profil."

        def _current_choice(self) -> CatalogSourceChoice:
            if self.existing_radio.isChecked():
                return "existing_library"
            if self.astap_radio.isChecked():
                return "astap"
            if self.package_radio.isChecked():
                return "local_package"
            return "official"

        def _sync_choice_visibility(self) -> None:
            if not hasattr(self, "existing_library_edit"):
                return
            choice = self._current_choice()
            self._selected_choice = choice
            official = choice == "official"
            existing = choice == "existing_library"
            astap = choice == "astap"
            package = choice == "local_package"
            for widget in (self.install_parent_edit, self.install_parent_browse):
                widget.setEnabled(official)
            for widget in (self.existing_library_edit, self.existing_library_browse):
                widget.setEnabled(existing)
            for widget in (self.astap_edit, self.astap_browse):
                widget.setEnabled(astap)
            for widget in (self.package_edit, self.package_browse, self.package_destination_edit, self.package_destination_browse):
                widget.setEnabled(package)
            self._refresh_storage_preview()

        def _refresh_storage_preview(self) -> None:
            if not hasattr(self, "storage_label"):
                return
            try:
                parent = Path(self.install_parent_edit.text().strip()).expanduser()
                manifest = getattr(self, "_preview_manifest", None)
                if manifest is None:
                    self.storage_label.setText(f"Parent: {parent}\nCache: {default_cache_root()}")
                    return
                destination = resolve_library_destination(manifest, parent)
                self.storage_label.setText(f"Destination: {destination}\nCache: {default_cache_root()}")
            except Exception as exc:
                self.storage_label.setText(str(exc))

        def _start_operation(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                return
            choice = self._current_choice()
            payload: dict[str, Any]
            if choice == "official":
                payload = {"install_parent": self.install_parent_edit.text().strip()}
                operation = "official_install"
            elif choice == "existing_library":
                payload = {"path": self.existing_library_edit.text().strip()}
                operation = "existing_library"
            elif choice == "astap":
                payload = {"path": self.astap_edit.text().strip()}
                operation = "astap"
            elif choice == "local_package":
                payload = {
                    "package_path": self.package_edit.text().strip(),
                    "destination": self.package_destination_edit.text().strip(),
                }
                operation = "local_package"
            else:
                return
            self._worker = StartupCatalogWorker(
                operation,
                payload,
                distribution_factory=self._distribution_factory,
                management_factory=self._management_factory,
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.discovered.connect(self._on_discovered)
            self._worker.finished.connect(self._on_finished)
            self._set_busy(True)
            self.progress.setValue(0)
            self.progress_log.clear()
            self._append_log(operation)
            self._worker.start()

        def _on_discovered(self, _release: object, manifest: object, storage: object) -> None:
            self._preview_manifest = manifest
            requirements = getattr(storage, "requirements", ())
            lines = []
            for item in requirements:
                available = getattr(item, "available_bytes", None)
                required = getattr(item, "required_bytes", 0)
                lines.append(
                    f"{getattr(item, 'role', '?')}: requis {format_bytes_binary(required)}, disponible {format_bytes_binary(available or 0)}"
                )
            self.storage_label.setText("\n".join(lines) if lines else "Espace disque verifie.")

        def _on_progress(self, progress: object) -> None:
            message = str(getattr(progress, "message", "") or getattr(progress, "stage", "") or "operation")
            total = int(getattr(progress, "overall_total", 0) or 0)
            current = int(getattr(progress, "overall_current", 0) or 0)
            if total > 0:
                self.progress.setValue(max(0, min(100, int((current / total) * 100))))
            bytes_total = int(getattr(progress, "bytes_total", 0) or 0)
            bytes_current = int(getattr(progress, "bytes_current", 0) or 0)
            if bytes_total > 0:
                message = f"{message} - {format_bytes_binary(bytes_current)} / {format_bytes_binary(bytes_total)}"
            self.progress_label.setText(message)
            self._append_log(message)

        def _on_finished(self, ok: bool, result: object, error: str, operation: str) -> None:
            self._set_busy(False)
            self._worker = None
            if not ok:
                self.progress_label.setText(f"Echec: {error}")
                self._append_log(error)
                return
            self._operation_completed = True
            self.progress.setValue(100)
            if operation in {"official_install", "local_package"}:
                library_result = getattr(result, "library_result", result)
                path = str(getattr(library_result, "library_root", "") or "")
                if path:
                    self.librarySelected.emit(path)
                self.progress_label.setText("Bibliotheque installee et validee.")
            elif operation == "existing_library":
                path = str((result or {}).get("path", "") if isinstance(result, dict) else "")
                if path:
                    self.librarySelected.emit(path)
                self.progress_label.setText("Bibliotheque selectionnee.")
            elif operation == "astap":
                path = str((result or {}).get("path", "") if isinstance(result, dict) else "")
                if path:
                    self.astapSelected.emit(path)
                self.progress_label.setText("Base ASTAP selectionnee pour ZeNear.")
            self._append_log(self.progress_label.text())

        def _set_busy(self, busy: bool) -> None:
            self.start_btn.setEnabled(not busy)
            self.cancel_btn.setEnabled(busy)
            self.button(QtWidgets.QWizard.FinishButton).setEnabled(not busy)

        def _cancel_worker(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_cancel()
                self.cancel_btn.setEnabled(False)

        def _on_custom_button(self, which: int) -> None:
            if which != QtWidgets.QWizard.CustomButton1:
                return
            clear_invalid_catalog_selection(self.settings)
            mark_startup_wizard_completed(self.settings)
            self._save_settings(self.settings)
            self.completed.emit("later")
            self.accept()

        def _update_summary(self) -> None:
            choice = self._current_choice()
            if choice == "official" and not self._operation_completed:
                text = "L'installation officielle n'a pas encore ete lancee."
            elif choice == "existing_library":
                text = f"Bibliotheque: {self.existing_library_edit.text().strip() or '-'}"
            elif choice == "astap":
                text = f"ASTAP ZeNear uniquement: {self.astap_edit.text().strip() or '-'}"
            elif choice == "local_package":
                text = f"Paquet local: {self.package_edit.text().strip() or '-'}"
            else:
                text = "Bibliotheque officielle prete."
            self.summary_label.setText(text)

        def accept(self) -> None:
            choice = self._current_choice()
            if choice in {"official", "local_package"} and not self._operation_completed:
                QtWidgets.QMessageBox.information(
                    self,
                    "Assistant de demarrage ZeSolver",
                    "L'installation doit etre terminee avec succes avant de terminer ce parcours.",
                )
                return
            if choice == "existing_library" and not self._operation_completed:
                path = self.existing_library_edit.text().strip()
                if self.decision.catalog.usable and path:
                    self.librarySelected.emit(path)
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Assistant de demarrage ZeSolver",
                        "Validez la bibliotheque existante avant de terminer.",
                    )
                    return
            if choice == "astap" and not self._operation_completed:
                path = self.astap_edit.text().strip()
                if self.decision.astap.usable and path:
                    self.astapSelected.emit(path)
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Assistant de demarrage ZeSolver",
                        "Validez la base ASTAP avant de terminer.",
                    )
                    return
            self.settings.sample_fits = self.sample_fits_edit.text().strip() or None
            self.settings.solver_blind_enabled = bool(self.blind_enabled_check.isChecked())
            mark_startup_wizard_completed(self.settings)
            self._save_settings(self.settings)
            self.completed.emit(choice)
            super().accept()

        def reject(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_cancel()
                self._worker.wait(5000)
            super().reject()

        def _row(self, edit: Any, button: Any) -> Any:
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(edit, 1)
            layout.addWidget(button)
            return row

        def _pick_directory(self, edit: Any) -> None:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir un dossier", edit.text().strip() or str(Path.home()))
            if chosen:
                edit.setText(chosen)

        def _pick_file(self, edit: Any) -> None:
            chosen, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir un paquet", edit.text().strip() or str(Path.home()))
            if chosen:
                edit.setText(chosen)

        def _append_log(self, text: str) -> None:
            self.progress_log.appendPlainText(str(text))

        def closeEvent(self, event: Any) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_cancel()
                self._worker.wait(5000)
            super().closeEvent(event)

else:  # pragma: no cover - import surface for non-GUI CLI environments
    ZeSolverStartupWizard = None  # type: ignore[assignment]


__all__ = [
    "STARTUP_WIZARD_VERSION",
    "StartupAstapProbe",
    "StartupCatalogProbe",
    "StartupCatalogWorker",
    "StartupWizardDecision",
    "ZeSolverStartupWizard",
    "clear_invalid_catalog_selection",
    "decide_startup_wizard",
    "default_astap_probe",
    "default_catalog_probe",
    "mark_startup_wizard_completed",
    "should_allow_legacy_family_prompt",
    "startup_default_paths",
]
