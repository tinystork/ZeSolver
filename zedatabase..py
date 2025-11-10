"""
ZeSolver Catalog Viewer (PySide6)
---------------------------------
Petit GUI pour explorer les bases HNSKY/ASTAP (familles 1476/290) via zewcs290.

Fonctions clés:
- Choix du répertoire racine de la base (ex: dossier contenant d50_*.1476, etc.)
- Détection/choix de la famille (D05, D20, D50, V50, G05, W08, H18… selon ce qui est présent)
- Ouverture du CatalogDB (utilise les layouts statiques, OK en base parcellaire)
- Liste des anneaux/tiles disponibles
- Requêtes "cone" (RA/DEC/rayon) et "box" (RA/DEC/largeur/hauteur)
- Affichage Matplotlib (scatter RA/Dec) + compteur d’étoiles
- Export CSV des résultats

Usage:
  python tools/zesolver_gui.py

Note: Place ce fichier dans tools/ du repo et assure-toi que le package zewcs290 est importable.
"""
from __future__ import annotations

import os
import sys
import csv
from pathlib import Path
from typing import Optional, List, Dict

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QLineEdit,
    QLabel,
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QListWidget,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
)

# Matplotlib embed
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import zewcs290
try:
    from zewcs290.catalog290 import CatalogDB, CatalogFamilySpec, CatalogTile, FAMILY_SPECS
except Exception as e:
    raise SystemExit(
        "Impossible d'importer zewcs290.catalog290. Assure-toi d'exécuter depuis la racine du repo \n"
        "ou d'avoir ajouté le projet au PYTHONPATH.\nErreur: %s" % (e,)
    )


class MplWidget(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_xlabel("RA (deg)")
        self.ax.set_ylabel("Dec (deg)")
        self.ax.grid(True, alpha=0.25)

    def clear(self):
        self.ax.clear()
        self.ax.set_xlabel("RA (deg)")
        self.ax.set_ylabel("Dec (deg)")
        self.ax.grid(True, alpha=0.25)
        self.draw_idle()

    def plot_points(self, ra_deg, dec_deg):
        self.ax.scatter(ra_deg, dec_deg, s=2)
        self.ax.invert_xaxis()  # convention carte du ciel
        self.ax.grid(True, alpha=0.25)
        self.draw_idle()


class ZeSolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeSolver – Catalog Viewer")
        self.resize(1200, 720)

        self.db: Optional[CatalogDB] = None
        self.current_stars = None  # numpy structured array
        self._spec_by_key: dict[str, CatalogFamilySpec] = {}
        self._ring_order: List[int] = []
        self._ring_tiles: Dict[int, List[CatalogTile]] = {}

        # Top: DB path + browse + family select + open
        path_label = QLabel("DB root:")
        self.path_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.on_browse)

        family_label = QLabel("Family:")
        self.family_combo = QComboBox()
        self.family_combo.setMinimumWidth(160)

        open_btn = QPushButton("Open DB")
        open_btn.clicked.connect(self.on_open_db)

        top_layout = QHBoxLayout()
        top_layout.addWidget(path_label)
        top_layout.addWidget(self.path_edit, 1)
        top_layout.addWidget(browse_btn)
        top_layout.addSpacing(16)
        top_layout.addWidget(family_label)
        top_layout.addWidget(self.family_combo)
        top_layout.addSpacing(8)
        top_layout.addWidget(open_btn)

        top_box = QWidget()
        top_box.setLayout(top_layout)

        # Left: rings/tiles lists
        left_box = QGroupBox("Rings & Tiles")
        self.rings_list = QListWidget()
        self.tiles_list = QListWidget()
        self.rings_list.itemSelectionChanged.connect(self.on_ring_selected)
        self.tiles_list.itemSelectionChanged.connect(self.on_tile_selected)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Rings:"))
        left_layout.addWidget(self.rings_list, 1)
        left_layout.addWidget(QLabel("Tiles (selected ring):"))
        left_layout.addWidget(self.tiles_list, 1)
        left_box.setLayout(left_layout)

        # Right: query form + plot
        right_box = QGroupBox("Queries & View")
        form = QFormLayout()

        self.ra_spin = QDoubleSpinBox()
        self.ra_spin.setRange(0.0, 360.0)
        self.ra_spin.setDecimals(6)
        self.ra_spin.setValue(180.0)

        self.dec_spin = QDoubleSpinBox()
        self.dec_spin.setRange(-90.0, 90.0)
        self.dec_spin.setDecimals(6)
        self.dec_spin.setValue(0.0)

        self.radius_arcmin = QDoubleSpinBox()
        self.radius_arcmin.setRange(0.01, 120.0)
        self.radius_arcmin.setDecimals(3)
        self.radius_arcmin.setValue(30.0)

        self.box_w_arcmin = QDoubleSpinBox()
        self.box_w_arcmin.setRange(0.01, 300.0)
        self.box_w_arcmin.setDecimals(3)
        self.box_w_arcmin.setValue(60.0)

        self.box_h_arcmin = QDoubleSpinBox()
        self.box_h_arcmin.setRange(0.01, 300.0)
        self.box_h_arcmin.setDecimals(3)
        self.box_h_arcmin.setValue(40.0)

        self.mag_limit = QDoubleSpinBox()
        self.mag_limit.setRange(-5.0, 30.0)
        self.mag_limit.setDecimals(2)
        self.mag_limit.setValue(18.0)

        self.max_stars = QSpinBox()
        self.max_stars.setRange(0, 2_000_000)
        self.max_stars.setValue(100_000)

        run_cone_btn = QPushButton("Run Cone")
        run_cone_btn.clicked.connect(self.on_run_cone)
        run_box_btn = QPushButton("Run Box")
        run_box_btn.clicked.connect(self.on_run_box)

        export_btn = QPushButton("Export CSV…")
        export_btn.clicked.connect(self.on_export_csv)

        form.addRow("RA (deg)", self.ra_spin)
        form.addRow("Dec (deg)", self.dec_spin)
        form.addRow("Cone radius (arcmin)", self.radius_arcmin)
        form.addRow("Box width (arcmin)", self.box_w_arcmin)
        form.addRow("Box height (arcmin)", self.box_h_arcmin)
        form.addRow("Mag limit", self.mag_limit)
        form.addRow("Max stars", self.max_stars)

        btns_row = QHBoxLayout()
        btns_row.addWidget(run_cone_btn)
        btns_row.addWidget(run_box_btn)
        btns_row.addStretch(1)
        btns_row.addWidget(export_btn)

        self.mpl = MplWidget()

        right_layout = QVBoxLayout()
        right_layout.addLayout(form)
        right_layout.addLayout(btns_row)
        right_layout.addWidget(self.mpl, 1)
        right_box.setLayout(right_layout)

        # Central splitter
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Main layout
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_box)
        main_layout.addWidget(splitter, 1)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")

        # Init family combo with a minimal hint (will refresh on path change)
        self.refresh_families(Path.cwd())

    # --- Helpers ---
    def refresh_families(self, root: Path):
        self.family_combo.clear()
        self._spec_by_key.clear()
        # Try to detect the families available under root by simple pattern scan
        specs: List[CatalogFamilySpec]
        build_all = getattr(CatalogFamilySpec, "build_all", None)
        if callable(build_all):
            raw_specs = build_all()
            if isinstance(raw_specs, dict):
                specs = list(raw_specs.values())
            else:
                specs = list(raw_specs)
        else:
            specs = list(FAMILY_SPECS.values())
        specs.sort(key=lambda spec: getattr(spec, "key", getattr(spec, "title", "")))
        present = []
        for spec in specs:
            pattern = spec.glob_pattern()
            if any(root.glob(pattern)):
                present.append(spec)
        if not present:
            # fallback: show all
            present = specs
        for spec in present:
            display = getattr(spec, "title", getattr(spec, "key", "Unnamed"))
            key = getattr(spec, "key", display)
            self.family_combo.addItem(display, key)
            self._spec_by_key[str(key)] = spec

    def on_browse(self):
        start = self.path_edit.text() or str(Path.cwd())
        d = QFileDialog.getExistingDirectory(self, "Select DB root", start)
        if d:
            self.path_edit.setText(d)
            self.refresh_families(Path(d))

    def on_open_db(self):
        root = Path(self.path_edit.text().strip())
        if not root.exists():
            QMessageBox.warning(self, "Path", "Le répertoire spécifié n'existe pas.")
            return
        spec_key = self.family_combo.currentData()
        if spec_key is None:
            QMessageBox.warning(self, "Family", "Aucune famille sélectionnée.")
            return
        spec_key = str(spec_key)
        spec = self._spec_by_key.get(spec_key)
        try:
            self.db = CatalogDB(root, families=[spec_key])
        except Exception as e:
            QMessageBox.critical(self, "Open DB", f"Échec d'ouverture: {e}")
            return

        self.populate_rings_and_tiles()
        spec_label = getattr(spec, "title", spec_key) if spec else spec_key
        self.statusBar().showMessage(f"DB ouverte: {spec_label} @ {root}")

    def populate_rings_and_tiles(self):
        self.rings_list.clear()
        self.tiles_list.clear()
        if not self.db:
            return
        tiles = getattr(self.db, "tiles", None)
        if tiles is None:
            QMessageBox.warning(self, "DB", "Cette version de zewcs290 ne fournit pas la liste des tiles.")
            return
        ring_map: Dict[int, List[CatalogTile]] = {}
        for tile in tiles:
            ring_map.setdefault(tile.ring_index, []).append(tile)
        self._ring_order = sorted(ring_map.keys())
        self._ring_tiles = {ring: sorted(entries, key=lambda t: t.tile_index) for ring, entries in ring_map.items()}
        for ring in self._ring_order:
            self.rings_list.addItem(f"Ring {ring:02d}  — {len(self._ring_tiles[ring])} tiles")

    def on_ring_selected(self):
        self.tiles_list.clear()
        if not self.db:
            return
        row = self.rings_list.currentRow()
        if row < 0:
            return
        if row >= len(self._ring_order):
            return
        ring_index = self._ring_order[row]
        entries = self._ring_tiles.get(ring_index, [])
        for tile in entries:
            self.tiles_list.addItem(f"{tile.tile_code} — {tile.path.name}")

    def on_tile_selected(self):
        # optional: in future, center view on tile bounds
        pass

    def _ensure_db(self) -> bool:
        if not self.db:
            QMessageBox.information(self, "DB", "Ouvre d'abord une base.")
            return False
        return True

    def on_run_cone(self):
        if not self._ensure_db():
            return
        ra = float(self.ra_spin.value())
        dec = float(self.dec_spin.value())
        radius_deg = float(self.radius_arcmin.value()) / 60.0
        mag_lim = float(self.mag_limit.value())
        max_stars = int(self.max_stars.value())
        try:
            stars = self.db.query_cone(ra_deg=ra, dec_deg=dec, radius_deg=radius_deg,
                                       mag_limit=mag_lim, max_stars=max_stars)
        except Exception as e:
            QMessageBox.critical(self, "Cone query", f"Erreur: {e}")
            return
        self.current_stars = stars
        self.update_plot()
        self.statusBar().showMessage(f"Cone: {len(stars)} étoiles")

    def on_run_box(self):
        if not self._ensure_db():
            return
        ra = float(self.ra_spin.value())
        dec = float(self.dec_spin.value())
        width_deg = float(self.box_w_arcmin.value()) / 60.0
        height_deg = float(self.box_h_arcmin.value()) / 60.0
        mag_lim = float(self.mag_limit.value())
        max_stars = int(self.max_stars.value())
        half_width = 0.5 * width_deg
        half_height = 0.5 * height_deg
        ra_min = ra - half_width
        ra_max = ra + half_width
        dec_min = dec - half_height
        dec_max = dec + half_height
        try:
            stars = self.db.query_box(
                ra_min_deg=ra_min,
                ra_max_deg=ra_max,
                dec_min_deg=dec_min,
                dec_max_deg=dec_max,
                mag_limit=mag_lim,
                max_stars=max_stars,
            )
        except Exception as e:
            QMessageBox.critical(self, "Box query", f"Erreur: {e}")
            return
        self.current_stars = stars
        self.update_plot()
        self.statusBar().showMessage(f"Box: {len(stars)} étoiles")

    def update_plot(self):
        self.mpl.clear()
        if self.current_stars is None or self.current_stars.size == 0:
            return
        ra = self.current_stars["ra_deg"]
        dec = self.current_stars["dec_deg"]
        self.mpl.plot_points(ra, dec)

    def on_export_csv(self):
        if self.current_stars is None or self.current_stars.size == 0:
            QMessageBox.information(self, "Export", "Aucun résultat à exporter.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Export CSV", "stars.csv", "CSV (*.csv)")
        if not out:
            return
        cols = list(self.current_stars.dtype.names)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in self.current_stars:
                w.writerow([row[c] for c in cols])
        self.statusBar().showMessage(f"Exporté: {out}")


def main(argv: List[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    w = ZeSolverGUI()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
