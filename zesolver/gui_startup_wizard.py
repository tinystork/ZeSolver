"""First-run startup wizard for ZeSolver.

The decision layer in this module is GUI-free so tests can exercise startup
policy without constructing a QApplication.  The Qt dialog below only presents
those decisions and delegates catalog work to existing catalog services.
"""

from __future__ import annotations

import threading
import os
import sys
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
    DistributionTransferController,
    DistributionTransferState,
    LibraryInstallOptions,
    build_storage_plan,
    default_cache_root,
    default_library_parent,
    format_bytes_binary,
    resolve_library_destination,
    validate_library_parent,
)
from .catalog_resources import resolve_catalog_resources
from .gpu_support import (
    DistributionKind,
    EffectiveBackend,
    GpuCapabilityReport,
    GpuProvisioningPlan,
    GpuProvisioningResult,
    GpuRuntimeContext,
    ProvisioningStatus,
    PythonEnvironmentProvisioner,
    build_gpu_provisioning_plan,
    probe_gpu_capability,
)


STARTUP_WIZARD_VERSION = 1

CatalogState = Literal["none", "missing", "invalid", "ready_full", "ready_partial", "near_only"]
AstapState = Literal["none", "missing", "invalid", "valid"]
StartupWizardMode = Literal["fresh", "ready", "repair", "astap_near_only", "later"]
CatalogSourceChoice = Literal["official", "existing_library", "astap", "local_package", "later"]


def default_gpu_runtime_context() -> GpuRuntimeContext:
    """Return the conservative default runtime context for wizard diagnostics."""
    allow = os.environ.get("ZESOLVER_ALLOW_GPU_PROVISIONING", "").strip().lower() in {"1", "true", "yes"}
    kind = DistributionKind.SOURCE_MANAGED if allow else DistributionKind.UNKNOWN
    return GpuRuntimeContext(
        distribution_kind=kind,
        allow_environment_mutation=allow,
        python_executable=sys.executable if allow else None,
    )


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


@dataclass(frozen=True, slots=True)
class StartupWizardCompletionRequest:
    source: CatalogSourceChoice
    catalog_library_path: str | None = None
    astap_path: str | None = None
    image_directory: str | None = None
    blind_enabled: bool = True


@dataclass(frozen=True, slots=True)
class StartupWizardCompletionResult:
    ok: bool
    error: str = ""


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
            self._transfer_control = DistributionTransferController()

        def request_cancel(self) -> None:
            self._cancel_event.set()
            self._transfer_control.request_cancel()

        def request_pause(self) -> None:
            self._transfer_control.request_pause()

        def request_resume(self) -> None:
            self._transfer_control.request_resume()

        def request_resume_now(self) -> None:
            self._transfer_control.request_resume_now()

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
                    transfer_control=self._transfer_control,
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


    class StartupGpuProvisionWorker(QtCore.QThread):
        progress = QtCore.Signal(str)
        resultReady = QtCore.Signal(object)

        def __init__(self, plan: GpuProvisioningPlan) -> None:
            super().__init__()
            self.plan = plan
            self._cancel_event = threading.Event()

        def request_cancel(self) -> None:
            self._cancel_event.set()

        def run(self) -> None:
            try:
                provisioner = PythonEnvironmentProvisioner()
                result = provisioner.provision(
                    self.plan,
                    progress_callback=self.progress.emit,
                    cancel_token=self._cancel_event.is_set,
                )
            except Exception as exc:
                result = GpuProvisioningResult(ProvisioningStatus.INSTALL_FAILED, f"INSTALL_FAILED: {exc}")
            self.resultReady.emit(result)

else:  # pragma: no cover - import surface for non-GUI CLI environments
    StartupCatalogWorker = None  # type: ignore[assignment]
    StartupGpuProvisionWorker = None  # type: ignore[assignment]


if QtWidgets is not None:

    class ZeSolverStartupWizard(QtWidgets.QWizard):
        librarySelected = QtCore.Signal(str)
        astapSelected = QtCore.Signal(str)
        imageDirectorySelected = QtCore.Signal(str)
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
            completion_handler: Callable[[StartupWizardCompletionRequest], StartupWizardCompletionResult | bool] | None = None,
            gpu_report_factory: Callable[[GpuRuntimeContext], GpuCapabilityReport] | None = None,
            gpu_runtime_context: GpuRuntimeContext | None = None,
        ) -> None:
            super().__init__(parent)
            self.settings = settings
            self.decision = decision
            self._save_settings = save_settings
            self._distribution_factory = distribution_factory
            self._management_factory = management_factory
            self._completion_handler = completion_handler
            self._gpu_runtime_context = gpu_runtime_context or default_gpu_runtime_context()
            self._gpu_report_factory = gpu_report_factory or (lambda context: probe_gpu_capability(context))
            self._gpu_report: GpuCapabilityReport | None = None
            self._gpu_plan: GpuProvisioningPlan | None = None
            self._gpu_worker: StartupGpuProvisionWorker | None = None
            self._gpu_provision_result: object | None = None
            self._gpu_provisioning_active = False
            self._worker: StartupCatalogWorker | None = None
            self._completed_operation: str | None = None
            self._completed_operation_signature: tuple[str, ...] | None = None
            self._active_operation_signature: tuple[str, ...] | None = None
            self._operation_completed = False
            self._operation_state = DistributionTransferState.IDLE.value
            self._validated_library_path = self._normalized_path_text(decision.catalog.path) if decision.catalog.usable else ""
            self._validated_astap_path = self._normalized_path_text(decision.astap.path) if decision.astap.usable else ""
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
            self.addPage(self._gpu_page())
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

        def _gpu_page(self) -> Any:
            page = QtWidgets.QWizardPage()
            page.setTitle("Acceleration GPU")
            layout = QtWidgets.QVBoxLayout(page)
            self.gpu_status_label = QtWidgets.QLabel("")
            self.gpu_status_label.setWordWrap(True)
            self.gpu_detail_view = QtWidgets.QPlainTextEdit()
            self.gpu_detail_view.setReadOnly(True)
            self.gpu_detail_view.setMaximumHeight(120)
            row = QtWidgets.QHBoxLayout()
            self.gpu_install_btn = QtWidgets.QPushButton("Installer l'acceleration GPU")
            self.gpu_cpu_btn = QtWidgets.QPushButton("Continuer sur CPU")
            self.gpu_rediagnose_btn = QtWidgets.QPushButton("Relancer le diagnostic")
            self.gpu_install_btn.clicked.connect(self._install_gpu_support)
            self.gpu_cpu_btn.clicked.connect(self._choose_cpu_for_gpu)
            self.gpu_rediagnose_btn.clicked.connect(self._run_gpu_diagnostic)
            row.addWidget(self.gpu_install_btn)
            row.addWidget(self.gpu_cpu_btn)
            row.addWidget(self.gpu_rediagnose_btn)
            row.addStretch(1)
            layout.addWidget(self.gpu_status_label)
            layout.addWidget(self.gpu_detail_view)
            layout.addLayout(row)
            page.initializePage = self._initialize_gpu_page  # type: ignore[method-assign]
            return page

        def _initialize_gpu_page(self) -> None:
            if bool(getattr(self.settings, "gpu_user_cpu_selected", False)) and bool(getattr(self.settings, "gpu_diagnostic_completed", False)):
                self._set_gpu_page_text(
                    "Vous avez choisi de continuer sur CPU. Le diagnostic GPU reste accessible depuis les reglages.",
                    details=f"Dernier statut: {getattr(self.settings, 'gpu_last_reason_code', '-') or '-'}",
                )
                self.gpu_install_btn.setEnabled(False)
                return
            if self._gpu_report is None:
                self._run_gpu_diagnostic()
            else:
                self._refresh_gpu_page()

        def _run_gpu_diagnostic(self) -> None:
            try:
                self._gpu_report = self._gpu_report_factory(self._gpu_runtime_context)
            except Exception as exc:
                self._gpu_report = None
                self._gpu_plan = None
                self._set_gpu_page_text(
                    "Le diagnostic GPU n'a pas pu etre execute. ZeSolver utilisera le processeur.",
                    details=str(exc),
                )
                return
            self._gpu_plan = build_gpu_provisioning_plan(self._gpu_report, self._gpu_runtime_context)
            self.settings.gpu_diagnostic_schema_version = 1
            self.settings.gpu_diagnostic_completed = True
            self.settings.gpu_available = self._gpu_report.effective_backend == EffectiveBackend.CUDA
            self.settings.gpu_last_reason_code = self._gpu_report.reason_code.value
            self._refresh_gpu_page()

        def _refresh_gpu_page(self) -> None:
            report = self._gpu_report
            if report is None:
                self._set_gpu_page_text("Diagnostic GPU non execute.", details="")
                self.gpu_install_btn.setEnabled(False)
                return
            plan = self._gpu_plan or build_gpu_provisioning_plan(report, self._gpu_runtime_context)
            if report.effective_backend == EffectiveBackend.CUDA:
                title = "Acceleration GPU disponible"
                details = [
                    f"GPU: {', '.join(report.device_names) if report.device_names else '-'}",
                    f"CuPy: {report.cupy_version or '-'}",
                    f"Peripheriques: {report.device_count if report.device_count is not None else '-'}",
                    "Test: reussi",
                ]
            elif report.platform == "darwin":
                title = "CUDA n'est pas pris en charge dans cette configuration. ZeSolver utilisera le processeur."
                details = [report.human_message]
            elif plan.status == ProvisioningStatus.AVAILABLE:
                title = (
                    "Un GPU NVIDIA a ete detecte. L'acceleration GPU de ZeNear est facultative; "
                    "ZeSolver fonctionne aussi entierement sur CPU."
                )
                details = [
                    report.human_message,
                    f"Paquet propose: {plan.profile.package_requirement if plan.profile else '-'}",
                    f"Environnement: {plan.command[0] if plan.command else '-'}",
                    "Un redemarrage de ZeSolver sera necessaire apres installation.",
                ]
            elif report.cupy_package_state.value in {"broken", "conflict"}:
                title = "L'installation GPU est presente mais n'a pas reussi le test. ZeSolver continuera sur CPU."
                details = [report.human_message]
            else:
                title = "Aucun GPU NVIDIA utilisable n'a ete detecte. ZeSolver utilisera le processeur."
                details = [report.human_message]
            self._set_gpu_page_text(title, details="\n".join(details))
            self.gpu_install_btn.setEnabled(plan.status == ProvisioningStatus.AVAILABLE and bool(plan.command))

        def _set_gpu_page_text(self, text: str, *, details: str = "") -> None:
            if hasattr(self, "gpu_status_label"):
                self.gpu_status_label.setText(text)
            if hasattr(self, "gpu_detail_view"):
                self.gpu_detail_view.setPlainText(details)

        def _choose_cpu_for_gpu(self) -> None:
            if self._is_gpu_provisioning_running():
                self._cancel_gpu_provisioning()
                return
            self.settings.gpu_user_cpu_selected = True
            self.settings.gpu_restart_required = False
            self.settings.gpu_diagnostic_completed = True
            self._save_settings(self.settings)
            self._set_gpu_page_text(
                "Choix enregistre: ZeSolver continuera sur CPU.",
                details=f"Diagnostic: {getattr(self.settings, 'gpu_last_reason_code', '-') or '-'}",
            )
            self.gpu_install_btn.setEnabled(False)

        def _wizard_buttons(self) -> tuple[Any, ...]:
            buttons = []
            for which in (
                QtWidgets.QWizard.BackButton,
                QtWidgets.QWizard.NextButton,
                QtWidgets.QWizard.FinishButton,
                QtWidgets.QWizard.CancelButton,
                QtWidgets.QWizard.CustomButton1,
            ):
                button = self.button(which)
                if button is not None:
                    buttons.append(button)
            return tuple(buttons)

        def _set_gpu_provisioning_active(self, active: bool) -> None:
            self._gpu_provisioning_active = bool(active)
            for button in self._wizard_buttons():
                button.setEnabled(not active)
            if hasattr(self, "gpu_install_btn"):
                self.gpu_install_btn.setEnabled(False if active else bool(self._gpu_plan and self._gpu_plan.command))
            if hasattr(self, "gpu_rediagnose_btn"):
                self.gpu_rediagnose_btn.setEnabled(not active)
            if hasattr(self, "gpu_cpu_btn"):
                self.gpu_cpu_btn.setEnabled(True)
                self.gpu_cpu_btn.setText("Annuler l'installation GPU" if active else "Continuer sur CPU")

        def _is_gpu_provisioning_running(self) -> bool:
            worker = self._gpu_worker
            return bool(worker is not None and worker.isRunning())

        def _install_gpu_support(self) -> None:
            plan = self._gpu_plan
            if self._is_gpu_provisioning_running():
                return
            if plan is None or plan.status != ProvisioningStatus.AVAILABLE or not plan.command:
                QtWidgets.QMessageBox.information(self, "Acceleration GPU", "Aucune installation GPU n'est disponible dans ce contexte.")
                return
            answer = QtWidgets.QMessageBox.question(
                self,
                "Acceleration GPU",
                "Installer l'acceleration GPU optionnelle ?\n\n"
                f"Environnement: {plan.command[0]}\n"
                f"Paquet: {plan.command[-1]}\n\n"
                "ZeSolver restera utilisable sur CPU. Aucun pilote NVIDIA ni Toolkit systeme ne sera installe.",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self.settings.gpu_user_cpu_selected = True
                self.settings.gpu_last_reason_code = "DECLINED"
                self._save_settings(self.settings)
                return
            self._gpu_provision_result = None
            self._gpu_worker = StartupGpuProvisionWorker(plan)
            self._gpu_worker.progress.connect(lambda text: self.gpu_detail_view.appendPlainText(str(text)))
            self._gpu_worker.resultReady.connect(self._on_gpu_provision_result_ready)
            self._gpu_worker.finished.connect(self._on_gpu_worker_thread_finished)
            self._gpu_worker.finished.connect(self._gpu_worker.deleteLater)
            self._set_gpu_provisioning_active(True)
            self._set_gpu_page_text("Installation GPU en cours...", details="")
            self._gpu_worker.start()

        def _cancel_gpu_provisioning(self) -> None:
            worker = self._gpu_worker
            if worker is None:
                return
            worker.request_cancel()
            self._set_gpu_page_text("Annulation en cours...", details=self.gpu_detail_view.toPlainText())
            if hasattr(self, "gpu_cpu_btn"):
                self.gpu_cpu_btn.setEnabled(False)

        def _wait_for_gpu_provisioning_stop(self, timeout_ms: int = 5000) -> bool:
            worker = self._gpu_worker
            if worker is None or not worker.isRunning():
                return True
            worker.request_cancel()
            self._set_gpu_page_text("Annulation en cours...", details=self.gpu_detail_view.toPlainText() if hasattr(self, "gpu_detail_view") else "")
            if not worker.wait(timeout_ms):
                return False
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()
            return True

        def _on_gpu_provision_result_ready(self, result: object) -> None:
            self._gpu_provision_result = result
            status = str(getattr(result, "status", "") or "")
            message = str(getattr(result, "message", "") or "")
            restart_required = bool(getattr(result, "restart_required", False))
            self.settings.gpu_restart_required = restart_required
            self.settings.gpu_last_reason_code = status.split(".")[-1] if status else "UNKNOWN"
            self.settings.gpu_user_cpu_selected = not restart_required
            self._save_settings(self.settings)
            details = self.gpu_detail_view.toPlainText()
            output_tail = "\n".join(
                part
                for part in (
                    str(getattr(result, "stdout_tail", "") or "").strip(),
                    str(getattr(result, "stderr_tail", "") or "").strip(),
                )
                if part
            )
            if output_tail and output_tail not in details:
                details = (details + "\n" if details else "") + output_tail
            if restart_required:
                details = (details + "\n" if details else "") + "Relancez ZeSolver pour activer l'acceleration GPU."
            self._set_gpu_page_text(message or "Installation GPU terminee.", details=details)

        def _on_gpu_worker_thread_finished(self) -> None:
            self._set_gpu_provisioning_active(False)
            if self._gpu_provision_result is not None:
                self.gpu_detail_view.appendPlainText("Operation GPU terminee.")
            self._gpu_worker = None

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
            self.existing_library_edit.textChanged.connect(lambda _text: self._on_path_text_changed("existing_library"))
            layout.addRow("Bibliotheque existante", self._row(self.existing_library_edit, self.existing_library_browse))
            self.astap_edit = QtWidgets.QLineEdit(str(getattr(self.settings, "db_root", "") or ""))
            self.astap_browse = QtWidgets.QPushButton("Parcourir")
            self.astap_browse.clicked.connect(lambda: self._pick_directory(self.astap_edit))
            self.astap_edit.textChanged.connect(lambda _text: self._on_path_text_changed("astap"))
            layout.addRow("Base ASTAP", self._row(self.astap_edit, self.astap_browse))
            self.package_edit = QtWidgets.QLineEdit("")
            self.package_browse = QtWidgets.QPushButton("Parcourir")
            self.package_browse.clicked.connect(lambda: self._pick_file(self.package_edit))
            self.package_edit.textChanged.connect(lambda _text: self._on_path_text_changed("local_package"))
            layout.addRow("Paquet local", self._row(self.package_edit, self.package_browse))
            self.package_destination_edit = QtWidgets.QLineEdit("")
            self.package_destination_browse = QtWidgets.QPushButton("Parcourir")
            self.package_destination_browse.clicked.connect(lambda: self._pick_directory(self.package_destination_edit))
            self.package_destination_edit.textChanged.connect(lambda _text: self._on_path_text_changed("local_package"))
            layout.addRow("Destination paquet", self._row(self.package_destination_edit, self.package_destination_browse))
            self.storage_label = QtWidgets.QLabel("")
            self.storage_label.setWordWrap(True)
            layout.addRow("Espace disque", self.storage_label)
            self.install_parent_edit.textChanged.connect(self._refresh_storage_preview)
            self.install_parent_edit.textChanged.connect(lambda _text: self._on_path_text_changed("official"))
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
            self.pause_btn = QtWidgets.QPushButton("Mettre en pause")
            self.cancel_btn = QtWidgets.QPushButton("Annuler")
            self.cancel_btn.setEnabled(False)
            self.pause_btn.setVisible(False)
            self.start_btn.clicked.connect(self._primary_operation_action)
            self.pause_btn.clicked.connect(self._pause_worker)
            self.cancel_btn.clicked.connect(self._cancel_worker)
            row.addWidget(self.start_btn)
            row.addWidget(self.pause_btn)
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
            self._refresh_operation_completed_flag()
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

        @staticmethod
        def _normalized_path_text(value: object) -> str:
            text = str(value or "").strip()
            if not text:
                return ""
            try:
                return str(Path(text).expanduser())
            except Exception:
                return text

        def _current_library_path(self) -> str:
            if not hasattr(self, "existing_library_edit"):
                return ""
            return self._normalized_path_text(self.existing_library_edit.text())

        def _current_astap_path(self) -> str:
            if not hasattr(self, "astap_edit"):
                return ""
            return self._normalized_path_text(self.astap_edit.text())

        def _operation_signature(self, operation: str) -> tuple[str, ...]:
            if operation == "official_install":
                return (self._normalized_path_text(self.install_parent_edit.text()),)
            if operation == "existing_library":
                return (self._current_library_path(),)
            if operation == "astap":
                return (self._current_astap_path(),)
            if operation == "local_package":
                return (
                    self._normalized_path_text(self.package_edit.text()),
                    self._normalized_path_text(self.package_destination_edit.text()),
                )
            return ()

        def _operation_for_choice(self, choice: CatalogSourceChoice | None = None) -> str:
            selected = choice or self._current_choice()
            if selected == "official":
                return "official_install"
            if selected == "existing_library":
                return "existing_library"
            if selected == "astap":
                return "astap"
            if selected == "local_package":
                return "local_package"
            return ""

        def _is_current_library_validated(self) -> bool:
            path = self._current_library_path()
            return bool(path and self._validated_library_path and path == self._validated_library_path)

        def _is_current_astap_validated(self) -> bool:
            path = self._current_astap_path()
            return bool(path and self._validated_astap_path and path == self._validated_astap_path)

        def _is_current_operation_completed(self, choice: CatalogSourceChoice | None = None) -> bool:
            selected = choice or self._current_choice()
            if selected == "existing_library":
                return self._is_current_library_validated()
            if selected == "astap":
                return self._is_current_astap_validated()
            operation = self._operation_for_choice(selected)
            return bool(
                operation
                and self._completed_operation == operation
                and self._completed_operation_signature == self._operation_signature(operation)
                and (selected not in {"official", "local_package"} or self._validated_library_path)
            )

        def _current_image_directory(self) -> str:
            if not hasattr(self, "sample_fits_edit"):
                return ""
            text = self.sample_fits_edit.text().strip()
            if not text:
                return ""
            try:
                directory = Path(text).expanduser()
            except Exception:
                return text
            if not directory.is_dir():
                return text
            try:
                return str(directory.resolve())
            except Exception:
                return str(directory)

        def _refresh_operation_completed_flag(self) -> None:
            if not hasattr(self, "existing_library_edit"):
                self._operation_completed = False
                return
            self._operation_completed = self._is_current_operation_completed()

        def _on_path_text_changed(self, choice: CatalogSourceChoice) -> None:
            if choice == "existing_library" and self._current_library_path() != self._validated_library_path:
                self._validated_library_path = ""
            elif choice == "astap" and self._current_astap_path() != self._validated_astap_path:
                self._validated_astap_path = ""
            operation = self._operation_for_choice(choice)
            if self._completed_operation == operation and self._completed_operation_signature != self._operation_signature(operation):
                self._completed_operation = None
                self._completed_operation_signature = None
            self._refresh_operation_completed_flag()

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
            self._completed_operation = None
            self._completed_operation_signature = None
            self._active_operation_signature = self._operation_signature(operation)
            self._operation_completed = False
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
            state = str(getattr(progress, "transfer_state", "") or "")
            if state:
                self._set_operation_state(state)
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
            self._set_operation_state(DistributionTransferState.COMPLETED.value if ok else DistributionTransferState.FAILED.value)
            self._worker = None
            if not ok:
                self.progress_label.setText(f"Echec: {error}")
                self._append_log(error)
                self._active_operation_signature = None
                self._refresh_operation_completed_flag()
                return
            self._completed_operation = operation
            self._completed_operation_signature = self._active_operation_signature
            self._active_operation_signature = None
            self.progress.setValue(100)
            if operation in {"official_install", "local_package"}:
                library_result = getattr(result, "library_result", result)
                path = str(getattr(library_result, "library_root", "") or "")
                if path:
                    self._validated_library_path = self._normalized_path_text(path)
                self.progress_label.setText("Bibliotheque installee et validee.")
            elif operation == "existing_library":
                path = str((result or {}).get("path", "") if isinstance(result, dict) else "")
                if path:
                    self._validated_library_path = self._normalized_path_text(path)
                self.progress_label.setText("Bibliotheque selectionnee.")
            elif operation == "astap":
                path = str((result or {}).get("path", "") if isinstance(result, dict) else "")
                if path:
                    self._validated_astap_path = self._normalized_path_text(path)
                self.progress_label.setText("Base ASTAP selectionnee pour ZeNear.")
            self._refresh_operation_completed_flag()
            self._append_log(self.progress_label.text())

        def _set_busy(self, busy: bool) -> None:
            self._set_operation_state(DistributionTransferState.DOWNLOADING.value if busy else DistributionTransferState.IDLE.value)

        def _primary_operation_action(self) -> None:
            state = str(getattr(self, "_operation_state", DistributionTransferState.IDLE.value) or DistributionTransferState.IDLE.value)
            if state in {DistributionTransferState.IDLE.value, DistributionTransferState.FAILED.value}:
                self._start_operation()
            elif state == DistributionTransferState.DOWNLOADING.value:
                self._pause_worker()
            elif state == DistributionTransferState.RETRY_WAIT.value:
                self._resume_now_worker()
            elif state == DistributionTransferState.PAUSED.value:
                self._resume_worker()

        def _set_operation_state(self, state: str) -> None:
            normalized = str(state or DistributionTransferState.IDLE.value)
            valid = {item.value for item in DistributionTransferState}
            if normalized not in valid:
                normalized = DistributionTransferState.IDLE.value
            self._operation_state = normalized
            running = self._worker is not None and self._worker.isRunning()
            terminal = normalized in {DistributionTransferState.COMPLETED.value, DistributionTransferState.FAILED.value, DistributionTransferState.IDLE.value}
            self.cancel_btn.setEnabled(running and not terminal)
            if normalized in {
                DistributionTransferState.DOWNLOADING.value,
                DistributionTransferState.RETRY_WAIT.value,
                DistributionTransferState.PAUSED.value,
                DistributionTransferState.CANCELLING.value,
            }:
                self.cancel_btn.setText("Annuler")
            self.pause_btn.setVisible(normalized == DistributionTransferState.RETRY_WAIT.value)
            self.pause_btn.setEnabled(running and normalized == DistributionTransferState.RETRY_WAIT.value)
            if normalized == DistributionTransferState.DOWNLOADING.value:
                self.start_btn.setText("Pause")
                self.start_btn.setEnabled(running)
            elif normalized == DistributionTransferState.RETRY_WAIT.value:
                self.start_btn.setText("Reprendre maintenant")
                self.start_btn.setEnabled(running)
                self.pause_btn.setText("Mettre en pause")
            elif normalized == DistributionTransferState.PAUSED.value:
                self.start_btn.setText("Reprendre")
                self.start_btn.setEnabled(running)
            elif normalized == DistributionTransferState.CANCELLING.value:
                self.start_btn.setText("Annulation...")
                self.start_btn.setEnabled(False)
                self.cancel_btn.setEnabled(False)
            elif normalized == DistributionTransferState.FAILED.value:
                self.start_btn.setText("Reessayer")
                self.start_btn.setEnabled(True)
                self.cancel_btn.setText("Fermer")
                self.cancel_btn.setEnabled(True)
            elif normalized == DistributionTransferState.COMPLETED.value:
                self.start_btn.setText("Terminer")
                self.start_btn.setEnabled(False)
            else:
                self.start_btn.setText("Demarrer")
                self.start_btn.setEnabled(True)
                self.cancel_btn.setText("Annuler")
            self.button(QtWidgets.QWizard.FinishButton).setEnabled(normalized == DistributionTransferState.COMPLETED.value or not running)

        def _cancel_worker(self) -> None:
            if self._worker is None and self._operation_state == DistributionTransferState.FAILED.value:
                self.reject()
                return
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_cancel()
                self._set_operation_state(DistributionTransferState.CANCELLING.value)

        def _pause_worker(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_pause()
                self._set_operation_state(DistributionTransferState.PAUSED.value)

        def _resume_worker(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_resume()
                self._set_operation_state(DistributionTransferState.DOWNLOADING.value)

        def _resume_now_worker(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_resume_now()

        def _on_custom_button(self, which: int) -> None:
            if which != QtWidgets.QWizard.CustomButton1:
                return
            clear_invalid_catalog_selection(self.settings)
            mark_startup_wizard_completed(self.settings)
            self._save_settings(self.settings)
            self.completed.emit("later")
            super().accept()

        def _update_summary(self) -> None:
            choice = self._current_choice()
            ready = self._is_current_operation_completed(choice)
            if choice == "official" and not ready:
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
            if self._is_gpu_provisioning_running():
                QtWidgets.QMessageBox.information(
                    self,
                    "Acceleration GPU",
                    "L'installation GPU est encore en cours. Attendez la fin ou annulez-la.",
                )
                return
            choice = self._current_choice()
            ready = self._is_current_operation_completed(choice)
            image_directory = self._current_image_directory()
            if image_directory and not Path(image_directory).expanduser().is_dir():
                QtWidgets.QMessageBox.information(
                    self,
                    "Assistant de demarrage ZeSolver",
                    "Le dossier d'images indique n'existe pas ou n'est pas un repertoire.",
                )
                return
            if choice in {"official", "local_package"} and not ready:
                QtWidgets.QMessageBox.information(
                    self,
                    "Assistant de demarrage ZeSolver",
                    "L'installation doit etre terminee avec succes avant de terminer ce parcours.",
                )
                return
            request = StartupWizardCompletionRequest(
                source=choice,
                catalog_library_path=(
                    self._validated_library_path
                    if choice in {"existing_library", "official", "local_package"} and self._validated_library_path
                    else None
                ),
                astap_path=(self._validated_astap_path if choice == "astap" and self._validated_astap_path else None),
                image_directory=image_directory or None,
                blind_enabled=bool(self.blind_enabled_check.isChecked()),
            )
            if self._completion_handler is not None:
                try:
                    result = self._completion_handler(request)
                    if isinstance(result, bool):
                        result = StartupWizardCompletionResult(result)
                    elif not isinstance(result, StartupWizardCompletionResult):
                        result = StartupWizardCompletionResult(
                            bool(getattr(result, "ok", False)),
                            str(getattr(result, "error", "") or ""),
                        )
                except Exception as exc:
                    result = StartupWizardCompletionResult(False, str(exc))
                if not result.ok:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Assistant de demarrage ZeSolver",
                        result.error or "L'activation demandee a echoue.",
                    )
                    return
                self.completed.emit(choice)
                super().accept()
                return
            if choice == "existing_library":
                if not ready:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Assistant de demarrage ZeSolver",
                        "Validez la bibliotheque existante avant de terminer.",
                    )
                    return
                self.librarySelected.emit(self._validated_library_path)
            if choice == "astap":
                if not ready:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Assistant de demarrage ZeSolver",
                        "Validez la base ASTAP avant de terminer.",
                    )
                    return
                self.astapSelected.emit(self._validated_astap_path)
            if choice in {"official", "local_package"} and self._validated_library_path:
                self.librarySelected.emit(self._validated_library_path)
            self.settings.sample_fits = image_directory or None
            self.settings.solver_blind_enabled = bool(self.blind_enabled_check.isChecked())
            mark_startup_wizard_completed(self.settings)
            self._save_settings(self.settings)
            if image_directory:
                self.imageDirectorySelected.emit(image_directory)
            self.completed.emit(choice)
            super().accept()

        def reject(self) -> None:
            if not self._wait_for_gpu_provisioning_stop(5000):
                return
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
            if not self._wait_for_gpu_provisioning_stop(5000):
                event.ignore()
                return
            if self._worker is not None and self._worker.isRunning():
                box = QtWidgets.QMessageBox(self)
                box.setWindowTitle("Assistant de demarrage ZeSolver")
                box.setText("Le telechargement n'est pas termine.\nLes fichiers partiels seront conserves.")
                quit_btn = box.addButton("Quitter et conserver", QtWidgets.QMessageBox.AcceptRole)
                continue_btn = box.addButton("Continuer le telechargement", QtWidgets.QMessageBox.RejectRole)
                cancel_btn = box.addButton("Annuler l'installation", QtWidgets.QMessageBox.DestructiveRole)
                box.exec()
                clicked = box.clickedButton()
                if clicked is continue_btn:
                    event.ignore()
                    return
                if clicked is quit_btn or clicked is cancel_btn:
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
    "StartupWizardCompletionRequest",
    "StartupWizardCompletionResult",
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
