from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from zeblindsolver.profiles import HISTORICAL_PROFILE

from zesolver.core.blind_models import BlindSolveRequest
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration

from solver_pipeline_fixtures import near_resources


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zesolver_entrypoint_gui_development_audit", ROOT / "zesolver.py")
assert SPEC is not None and SPEC.loader is not None
zesolver_app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zesolver_app
SPEC.loader.exec_module(zesolver_app)


class FakeManifest:
    def __init__(self, root: Path) -> None:
        self.manifest_path = root / "manifest.json"
        self.enabled_index_paths = (root / "d50_2822_S_q40000.npz",)
        self.entries = ()


def test_development_dev_fields_are_not_product_settings() -> None:
    names = {field.name for field in fields(ProductSettings)}

    assert "dev_bucket_limit_override" not in names
    assert "dev_vote_percentile" not in names
    assert "dev_bucket_cap_S" not in names
    assert "dev_detect_k_sigma" not in names
    assert "dev_family_selection" not in names


def test_product_blind4d_runtime_does_not_receive_gui_dev_overrides_without_developer_channel(tmp_path: Path) -> None:
    configuration = build_solver_configuration(
        product_settings=ProductSettings(downsample=2),
        runtime_options=RuntimeOptions(),
    )
    request = BlindSolveRequest.from_solve_request(
        SimpleNamespace(input_path=tmp_path / "in.fit", output_path=None, overwrite_wcs=True, request_id="p3b1b"),
        configuration=configuration,
    )

    produced = ProductionBlindSolverPort().build_config(
        request,
        resources=near_resources(tmp_path, blind_count=1),
        configuration=configuration,
        loaded_manifest=FakeManifest(tmp_path),
    )

    assert configuration.developer_overrides_active is False
    assert not any(key.startswith("dev_") for key in configuration.legacy_solve_config_values)
    assert produced.bucket_limit_override == 0
    assert produced.vote_percentile == 40
    assert produced.bucket_cap_S == 0
    assert produced.bucket_cap_M == 0
    assert produced.bucket_cap_L == 0
    assert produced.detect_k_sigma == 3.0
    assert produced.detect_min_area == 5
    assert produced.downsample == 2


def test_legacy_blind_builder_consumes_development_bucket_vote_cap_and_detection_fields(tmp_path: Path) -> None:
    config = zesolver_app.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=("d50",),
        blind_index_path=tmp_path,
        blind_backend_profile=HISTORICAL_PROFILE,
        dev_bucket_limit_override=777,
        dev_vote_percentile=61,
        dev_bucket_cap_S=111,
        dev_bucket_cap_M=222,
        dev_bucket_cap_L=333,
        dev_detect_k_sigma=4.8,
        dev_detect_min_area=9,
        downsample=3,
    )

    produced = zesolver_app.build_blind_solve_config(config)

    assert produced.bucket_limit_override == 777
    assert produced.vote_percentile == 61
    assert produced.bucket_cap_S == 111
    assert produced.bucket_cap_M == 222
    assert produced.bucket_cap_L == 333
    assert produced.detect_k_sigma == 4.8
    assert produced.detect_min_area == 9
    assert produced.downsample == 3


def test_worker_control_has_single_performance_widget_source() -> None:
    source = (ROOT / "zesolver.py").read_text(encoding="utf-8")
    start = source.index("def _build_performance_tab")
    end = source.index("def _build_fast_solver_tab", start)
    block = source[start:end]

    assert "dev_workers_combo" not in source
    assert "self.workers_spin = QtWidgets.QSpinBox()" in block
    assert "self._settings.solver_workers = self._dev_workers_choice" in source
    assert "self.workers_spin.valueChanged.connect(self._on_workers_spin_changed)" in source
    assert "workers=self._effective_workers_for_run()" in source


def test_hash_rebuild_controls_start_quads_only_indexbuilder_not_a_solve_run() -> None:
    source = (ROOT / "zesolver.py").read_text(encoding="utf-8")
    start = source.index("def _rebuild_hash_level")
    end = source.index("def _populate_settings_ui", start)
    block = source[start:end]

    assert "IndexBuilder(" in block
    assert "quads_only=True" in block
    assert "levels=(level_key,)" in block
    assert "builder.start()" in block
    assert "_build_config(" not in block
    assert "ImageSolver(" not in block


def test_development_tab_builder_removed_and_hash_widgets_are_dialog_scoped() -> None:
    source = (ROOT / "zesolver.py").read_text(encoding="utf-8")
    build_ui = source[source.index("def _build_ui") : source.index("def _wrap_scroll_area")]

    assert "_build_dev_tab" not in source
    assert "dev_scroll" not in source
    assert "dev_tab" not in source
    assert "dev_hash_buttons" not in build_ui
    assert "def _build_historical_index_maintenance_dialog" in source
