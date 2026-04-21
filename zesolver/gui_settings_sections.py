from __future__ import annotations

from typing import Any


def build_blind_group(owner: Any, QtWidgets: Any):
    """Build and attach the blind-tuning group used by Settings tab."""
    owner.blind_group = QtWidgets.QGroupBox()
    owner.blind_group.setTitle(owner._text("settings_blind_group"))
    blind_form = QtWidgets.QFormLayout(owner.blind_group)

    owner.settings_blind_max_stars_label = QtWidgets.QLabel()
    owner.settings_blind_max_stars_spin = QtWidgets.QSpinBox()
    owner.settings_blind_max_stars_spin.setRange(100, 5000)
    owner.settings_blind_max_stars_spin.setValue(owner._settings.blind_max_stars)
    blind_form.addRow(owner.settings_blind_max_stars_label, owner.settings_blind_max_stars_spin)

    owner.settings_blind_max_quads_label = QtWidgets.QLabel()
    owner.settings_blind_max_quads_spin = QtWidgets.QSpinBox()
    owner.settings_blind_max_quads_spin.setRange(500, 100000)
    owner.settings_blind_max_quads_spin.setSingleStep(500)
    owner.settings_blind_max_quads_spin.setValue(owner._settings.blind_max_quads)
    blind_form.addRow(owner.settings_blind_max_quads_label, owner.settings_blind_max_quads_spin)

    owner.settings_blind_max_candidates_label = QtWidgets.QLabel()
    owner.settings_blind_max_candidates_spin = QtWidgets.QSpinBox()
    owner.settings_blind_max_candidates_spin.setRange(4, 64)
    owner.settings_blind_max_candidates_spin.setValue(owner._settings.blind_max_candidates)
    blind_form.addRow(owner.settings_blind_max_candidates_label, owner.settings_blind_max_candidates_spin)

    owner.settings_blind_pixel_tol_label = QtWidgets.QLabel()
    owner.settings_blind_pixel_tol_spin = QtWidgets.QDoubleSpinBox()
    owner.settings_blind_pixel_tol_spin.setRange(0.5, 10.0)
    owner.settings_blind_pixel_tol_spin.setDecimals(1)
    owner.settings_blind_pixel_tol_spin.setSingleStep(0.5)
    owner.settings_blind_pixel_tol_spin.setValue(owner._settings.blind_pixel_tolerance)
    blind_form.addRow(owner.settings_blind_pixel_tol_label, owner.settings_blind_pixel_tol_spin)

    owner.settings_blind_quality_inliers_label = QtWidgets.QLabel()
    owner.settings_blind_quality_inliers_spin = QtWidgets.QSpinBox()
    owner.settings_blind_quality_inliers_spin.setRange(4, 200)
    owner.settings_blind_quality_inliers_spin.setValue(owner._settings.blind_quality_inliers)
    blind_form.addRow(owner.settings_blind_quality_inliers_label, owner.settings_blind_quality_inliers_spin)

    owner.settings_blind_quality_rms_label = QtWidgets.QLabel()
    owner.settings_blind_quality_rms_spin = QtWidgets.QDoubleSpinBox()
    owner.settings_blind_quality_rms_spin.setRange(0.2, 5.0)
    owner.settings_blind_quality_rms_spin.setDecimals(2)
    owner.settings_blind_quality_rms_spin.setSingleStep(0.1)
    owner.settings_blind_quality_rms_spin.setValue(owner._settings.blind_quality_rms)
    blind_form.addRow(owner.settings_blind_quality_rms_label, owner.settings_blind_quality_rms_spin)

    owner.settings_blind_fast_check = QtWidgets.QCheckBox()
    owner.settings_blind_fast_check.setChecked(owner._settings.blind_fast_mode)
    blind_form.addRow(QtWidgets.QLabel(owner._text("settings_blind_fast_label")), owner.settings_blind_fast_check)

    return owner.blind_group


def build_presets_fov_reco_groups(owner: Any, QtWidgets: Any, preset_utils: Any, column: Any, form: Any) -> None:
    """Build and attach presets/FOV/recommendation settings groups."""
    owner.presets_group = QtWidgets.QGroupBox()
    owner.presets_group.setTitle(owner._text("presets_title"))
    presets_layout = QtWidgets.QVBoxLayout(owner.presets_group)
    owner.presets_combo = QtWidgets.QComboBox()
    preset_list = list(preset_utils.list_presets())
    for p in preset_list:
        owner.presets_combo.addItem(p.label, p.id)
    presets_layout.addWidget(owner.presets_combo)

    owner.fov_group = QtWidgets.QGroupBox()
    owner.fov_group.setTitle(owner._text("fov_mode_title"))
    fov_form = QtWidgets.QFormLayout(owner.fov_group)
    owner.fov_focal_spin = QtWidgets.QDoubleSpinBox()
    owner.fov_focal_spin.setRange(10.0, 6000.0)
    owner.fov_focal_spin.setDecimals(1)
    owner.fov_pixel_spin = QtWidgets.QDoubleSpinBox()
    owner.fov_pixel_spin.setRange(1.0, 20.0)
    owner.fov_pixel_spin.setDecimals(2)
    owner.fov_res_w_spin = QtWidgets.QSpinBox()
    owner.fov_res_w_spin.setRange(64, 20000)
    owner.fov_res_h_spin = QtWidgets.QSpinBox()
    owner.fov_res_h_spin.setRange(64, 20000)
    owner.fov_reducer_spin = QtWidgets.QDoubleSpinBox()
    owner.fov_reducer_spin.setRange(0.2, 2.0)
    owner.fov_reducer_spin.setDecimals(2)
    owner.fov_reducer_spin.setSingleStep(0.01)
    owner.fov_binning_spin = QtWidgets.QSpinBox()
    owner.fov_binning_spin.setRange(1, 8)

    owner.reco_group = QtWidgets.QGroupBox()
    owner.reco_group.setTitle(owner._text("recommendations_title"))
    reco_form = QtWidgets.QFormLayout(owner.reco_group)
    owner.reco_scale_label = QtWidgets.QLabel()
    owner.reco_scale_value = QtWidgets.QLabel("-")
    owner.reco_fov_label = QtWidgets.QLabel()
    owner.reco_fov_value = QtWidgets.QLabel("-")
    owner.reco_mag_label = QtWidgets.QLabel()
    owner.reco_mag_value = QtWidgets.QLabel("-")
    owner.reco_quads_label = QtWidgets.QLabel()
    owner.reco_quads_value = QtWidgets.QLabel("-")
    owner.reco_notes_label = QtWidgets.QLabel("")
    owner.reco_notes_label.setWordWrap(True)
    owner.compute_button = QtWidgets.QPushButton()

    fov_form.addRow(QtWidgets.QLabel(owner._text("focal_length_mm")), owner.fov_focal_spin)
    fov_form.addRow(QtWidgets.QLabel(owner._text("pixel_size_um")), owner.fov_pixel_spin)
    res_row = QtWidgets.QWidget()
    res_layout = QtWidgets.QHBoxLayout(res_row)
    res_layout.setContentsMargins(0, 0, 0, 0)
    res_layout.addWidget(owner.fov_res_w_spin)
    res_layout.addWidget(owner.fov_res_h_spin)
    fov_form.addRow(QtWidgets.QLabel(owner._text("resolution_px")), res_row)
    fov_form.addRow(QtWidgets.QLabel(owner._text("reducer_factor")), owner.fov_reducer_spin)
    fov_form.addRow(QtWidgets.QLabel(owner._text("binning")), owner.fov_binning_spin)
    fov_form.addRow(owner.compute_button)

    reco_form.addRow(owner.reco_scale_label, owner.reco_scale_value)
    reco_form.addRow(owner.reco_fov_label, owner.reco_fov_value)
    reco_form.addRow(owner.reco_mag_label, owner.reco_mag_value)
    reco_form.addRow(owner.reco_quads_label, owner.reco_quads_value)
    reco_form.addRow(owner.reco_notes_label)

    column.addLayout(form)
    column.addWidget(owner.presets_group)
    column.addWidget(owner.fov_group)
    column.addWidget(owner.reco_group)



def apply_settings_preset(owner: Any, preset_utils: Any, preset_id: str | None) -> None:
    """Apply selected preset values to FOV controls and refresh recommendations."""
    try:
        presets = {p.id: p for p in preset_utils.list_presets()}
        preset = presets.get(preset_id)
        if not preset:
            return
        owner.fov_focal_spin.setValue(preset.focal_mm)
        owner.fov_pixel_spin.setValue(preset.pixel_um)
        owner.fov_res_w_spin.setValue(preset.res_w)
        owner.fov_res_h_spin.setValue(preset.res_h)
        owner.fov_reducer_spin.setValue(preset.reducer)
        owner.fov_binning_spin.setValue(1)
        owner._on_compute_fov_clicked()
    except Exception:
        pass


def wire_settings_tab_callbacks(owner: Any, preset_utils: Any) -> None:
    """Connect Settings tab buttons/inputs to owner callbacks."""
    owner.settings_save_btn.clicked.connect(owner._on_save_settings_clicked)
    owner.settings_build_btn.clicked.connect(owner._on_build_index_clicked)
    owner.settings_run_blind_btn.clicked.connect(owner._on_run_blind_clicked)
    owner.settings_run_near_btn.clicked.connect(owner._on_run_near_clicked)

    owner.settings_db_browse.clicked.connect(
        lambda: owner._pick_settings_directory(owner.settings_db_edit)
    )

    def _sync_db_tab_text(text: str) -> None:
        try:
            if hasattr(owner, "db_tab_edit"):
                if owner.db_tab_edit.text().strip() != text.strip():
                    owner.db_tab_edit.setText(text)
        except Exception:
            pass

    owner.settings_db_edit.textChanged.connect(_sync_db_tab_text)
    owner.settings_db_edit.textChanged.connect(owner._on_db_root_text_changed)
    owner.settings_index_browse.clicked.connect(
        lambda: owner._pick_settings_directory(owner.settings_index_edit)
    )
    owner.settings_sample_browse.clicked.connect(owner._pick_settings_sample)

    owner.presets_combo.currentIndexChanged.connect(
        lambda idx: apply_settings_preset(owner, preset_utils, owner.presets_combo.itemData(idx))
    )

    saved_preset = getattr(owner._settings, "last_preset_id", None)
    if saved_preset:
        idx = owner.presets_combo.findData(saved_preset)
        if idx >= 0:
            owner.presets_combo.setCurrentIndex(idx)
            apply_settings_preset(owner, preset_utils, saved_preset)

    owner.compute_button.clicked.connect(owner._on_compute_fov_clicked)
