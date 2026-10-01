"""Desktop UI — PySide6 port of ViewController + UIManager + MainApp.

Light (white) theme, live face catalog table on top, status bar at bottom.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QMainWindow, QPushButton, QStatusBar,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..config import AppConfig
from ..detectors.ssd_haar import HaarFaceDetector, SsdFaceDetector
from ..detectors.yolo import YoloFaceDetector
from ..model_paths import ModelPaths
from ..path_validator import sanitize_filename
from ..processing.frames import FaceRow, FrameProcessor
from ..processing.vision_fps import FpsCalculator
from ..services.camera import CameraManager
from ..services.face_service import FaceDetectorService
from ..services.pipeline import DetectionPipeline
from ..services.prefs import PreferencesService
from ..services.update_security import UpdateService

LIGHT_QSS = """
QMainWindow, QWidget { background: #F8FAFC; color: #1E293B; font-size: 13px; }
QPushButton { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px;
    padding: 7px 16px; color: #1E293B; }
QPushButton:hover { background: #F1F5F9; border-color: #94A3B8; }
QPushButton:pressed { background: #E2E8F0; }
QPushButton#primary { background: #16A34A; color: #FFFFFF; border: none; font-weight: 600; }
QPushButton#primary:hover { background: #15803D; }
QComboBox { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; }
QTableWidget { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px;
    gridline-color: #E2E8F0; alternate-background-color: #F8FAFC; }
QHeaderView::section { background: #F1F5F9; color: #475569; border: none;
    padding: 5px; font-weight: 600; }
QLabel { color: #1E293B; }
QLabel#videoView { background: #111111; color: #DDDDDD; border-radius: 6px; }
QCheckBox { color: #334155; spacing: 6px; }
QStatusBar { background: #FFFFFF; border-top: 1px solid #E2E8F0; color: #475569; }
QStatusBar::item { border: none; }
"""

CATALOG_COLUMNS = ("ID", "Gender", "Confidence", "Engine", "Position")


def _to_pixmap(frame: np.ndarray, max_w: int = 640) -> QPixmap:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class CameraWorker(QThread):
    frame_ready = Signal(object)

    def __init__(self, manager: CameraManager):
        super().__init__()
        self.manager = manager

    def run(self):
        self.manager.start_capture(lambda f: self.frame_ready.emit(f))
        self.exec()

    def stop(self):
        self.manager.stop()
        self.quit()
        self.wait(2000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = AppConfig.get_instance()
        self.prefs = PreferencesService()
        self.setWindowTitle("Face Recognition & Analysis (Python)")
        self.resize(int(self.prefs.get_float("window_width", 1120)),
                    int(self.prefs.get_float("window_height", 720)))

        self.mp = ModelPaths.resolve()
        self.yolo = YoloFaceDetector(self.mp.yolo_path, self.cfg.yolo_confidence, self.cfg.yolo_nms) if self.mp.has_yolo() else None
        if self.yolo is not None and not self.yolo.is_available():
            self.yolo = None
        self.ssd = SsdFaceDetector(self.mp.ssd_model_path, self.mp.ssd_config_path, self.cfg.ssd_confidence) if self.mp.has_ssd() else None
        if self.ssd is not None and not self.ssd.is_available():
            self.ssd = None
        self.haar = HaarFaceDetector(self.mp.haar_path, self.cfg.haar_default_confidence,
                                     self.cfg.haar_scale_factor, self.cfg.haar_min_neighbors,
                                     self.cfg.haar_min_face_size) if self.mp.has_haar() else None
        self.face_service = None
        if self.mp.has_ssd() and self.mp.has_gender():
            try:
                self.face_service = FaceDetectorService(
                    str(self.mp.ssd_model_path), str(self.mp.ssd_config_path),
                    str(self.mp.gender_model_path), str(self.mp.gender_config_path), self.cfg)
            except Exception:
                self.face_service = None

        self.pipeline = DetectionPipeline(self.yolo, self.ssd, self.haar)
        self.processor = FrameProcessor(self.pipeline, self.face_service, self.cfg)
        self.camera = CameraManager(self.cfg)
        self.fps = FpsCalculator(self.cfg.fps_smoothing)
        self.updater = UpdateService(self.cfg)
        self.worker: CameraWorker | None = None
        self.frame_counter = 0
        self.camera_on = False
        self._catalog_key: tuple = ()
        self._last_processed: np.ndarray | None = None

        # ---- widgets ----
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # catalog table on top
        self.catalog = QTableWidget(0, len(CATALOG_COLUMNS))
        self.catalog.setHorizontalHeaderLabels(list(CATALOG_COLUMNS))
        self.catalog.setEditTriggers(QTableWidget.NoEditTriggers)
        self.catalog.setSelectionBehavior(QTableWidget.SelectRows)
        self.catalog.setSelectionMode(QTableWidget.SingleSelection)
        self.catalog.setAlternatingRowColors(True)
        self.catalog.verticalHeader().setVisible(False)
        self.catalog.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.catalog.setFixedHeight(140)
        layout.addWidget(self.catalog)

        top = QHBoxLayout()
        self.btn_image = QPushButton("Open Image")
        self.btn_camera = QPushButton("Start Camera")
        self.btn_camera.setObjectName("primary")
        self.btn_snapshot = QPushButton("Snapshot")
        self.btn_snapshot.setEnabled(False)
        self.camera_selector = QComboBox()
        top.addWidget(self.btn_image)
        top.addWidget(self.btn_camera)
        top.addWidget(self.btn_snapshot)
        top.addWidget(self.camera_selector)
        top.addStretch(1)
        layout.addLayout(top)

        views = QHBoxLayout()
        self.source_view = QLabel("Source")
        self.result_view = QLabel("Result")
        for v in (self.source_view, self.result_view):
            v.setObjectName("videoView")
            v.setAlignment(Qt.AlignCenter)
            v.setMinimumSize(480, 360)
        views.addWidget(self.source_view)
        views.addWidget(self.result_view)
        layout.addLayout(views)

        opts = QHBoxLayout()
        self.chk_adaptive = QCheckBox("Adaptive exposure")
        self.chk_bright = QCheckBox("Bright-light mode")
        self.chk_gender = QCheckBox("Gender recognition")
        self.chk_adaptive.setChecked(self.prefs.get_bool("adaptive_exposure_enabled", True))
        self.chk_bright.setChecked(self.prefs.get_bool("bright_light_mode_enabled", True))
        self.chk_gender.setChecked(self.prefs.get_bool("gender_recognition_enabled", False))
        for c in (self.chk_adaptive, self.chk_bright, self.chk_gender):
            opts.addWidget(c)
        opts.addStretch(1)
        layout.addLayout(opts)

        # status bar replaces the 4 stacked labels
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.engine_label = QLabel("Engine: --")
        self.fps_label = QLabel("FPS: --")
        self.exposure_label = QLabel("Exposure: Ready")
        for w in (self.engine_label, self.fps_label, self.exposure_label):
            self.status_bar.addPermanentWidget(w)
        self.status_bar.showMessage("Idle")

        # ---- signals ----
        self.btn_image.clicked.connect(self.select_image)
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.chk_adaptive.toggled.connect(self._on_adaptive)
        self.chk_bright.toggled.connect(self._on_bright)
        self.chk_gender.toggled.connect(lambda v: self.prefs.set_bool("gender_recognition_enabled", v))
        self.refresh_cameras()
        self.pipeline.set_bright_light_mode(self.chk_bright.isChecked())
        self.camera.set_adaptive(self.chk_adaptive.isChecked())
        self.updater.check_for_updates(self._on_update_found, lambda e: None)

    def refresh_cameras(self):
        self.camera_selector.clear()
        cams = CameraManager.detect_cameras()
        if not cams:
            self.camera_selector.addItem("No Camera Found")
        else:
            for i in cams:
                self.camera_selector.addItem(f"Camera {i}", i)
            last = self.prefs.get_int("last_camera_index", cams[0])
            idx = cams.index(last) if last in cams else 0
            self.camera_selector.setCurrentIndex(idx)

    def _selected_camera(self) -> int:
        data = self.camera_selector.currentData()
        return int(data) if data is not None else 0

    def _refresh_catalog(self, rows: list[FaceRow], engine: str) -> None:
        key = tuple((r.track_id, r.gender_label, r.gender_pct,
                     round(r.confidence, 3), r.x, r.y, r.w, r.h) for r in rows) + (engine,)
        if key == self._catalog_key:
            return
        self._catalog_key = key
        self.catalog.setRowCount(len(rows))
        for i, r in enumerate(rows):
            gender = f"{r.gender_label} {r.gender_pct}".strip() or "--"
            vals = (str(r.track_id), gender, f"{r.confidence:.0%}", engine, r.position)
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.catalog.setItem(i, j, item)

    def _show_result(self, display, processed, status: str, engine: str,
                     rows: list[FaceRow], fps_text: str, exposure: str) -> None:
        self.source_view.setPixmap(_to_pixmap(display))
        self.result_view.setPixmap(_to_pixmap(processed))
        self._last_processed = processed
        self.btn_snapshot.setEnabled(True)
        self._refresh_catalog(rows, engine)
        self.status_bar.showMessage(status)
        self.engine_label.setText(f"Engine: {engine}")
        self.fps_label.setText(f"FPS: {fps_text}")
        self.exposure_label.setText(exposure)

    def take_snapshot(self):
        if self._last_processed is None:
            self.status_bar.showMessage("Nothing to save yet — open an image or start the camera")
            return
        start_dir = self.prefs.get_string("last_snapshot_dir", str(Path.home() / "Documents"))
        stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
        name = sanitize_filename(f"snapshot_{stamp}.jpg")
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", str(Path(start_dir) / name),
                                              "JPEG (*.jpg);;PNG (*.png)")
        if not path:
            return
        if cv2.imwrite(path, self._last_processed):
            self.prefs.set_string("last_snapshot_dir", str(Path(path).parent))
            self.status_bar.showMessage(f"Snapshot saved: {path}")
        else:
            self.status_bar.showMessage(f"Failed to save snapshot: {path}")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        if self.camera_on:
            self.toggle_camera()
        img = cv2.imread(path)
        if img is None:
            self.status_bar.showMessage("Failed to read image")
            return
        self.processor.reset()
        res = self.processor.process(img, 1, self.chk_bright.isChecked(), self.chk_gender.isChecked())
        self._show_result(res.display, res.processed, res.status, res.engine_label,
                          res.faces, "Static", "Exposure: N/A")

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            if self.worker:
                self.worker.stop()
                self.worker = None
            self.camera.close()
            self.btn_camera.setText("Start Camera")
            return
        if not self.camera.open(self._selected_camera()):
            self.status_bar.showMessage("Failed to open camera")
            return
        self.prefs.set_int("last_camera_index", self._selected_camera())
        self.processor.reset()
        self.fps.reset()
        self.camera.set_adaptive(self.chk_adaptive.isChecked())
        self.worker = CameraWorker(self.camera)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.start()
        self.camera_on = True
        self.btn_camera.setText("Stop Camera")

    def on_frame(self, frame: np.ndarray):
        self.frame_counter += 1
        fps = self.fps.update()
        res = self.processor.process(frame, self.frame_counter,
                                     self.chk_bright.isChecked(), self.chk_gender.isChecked())
        self._show_result(res.display, res.processed, res.status, res.engine_label,
                          res.faces, f"{fps:.1f}", self.camera.status)

    def _on_adaptive(self, v: bool):
        self.prefs.set_bool("adaptive_exposure_enabled", v)
        self.camera.set_adaptive(v)

    def _on_bright(self, v: bool):
        self.prefs.set_bool("bright_light_mode_enabled", v)
        self.pipeline.set_bright_light_mode(v)

    def _on_update_found(self, url: str):
        self.status_bar.showMessage(f"New update found: {url}")

    def closeEvent(self, event):
        try:
            self.prefs.set_float("window_width", float(self.width()))
            self.prefs.set_float("window_height", float(self.height()))
            if self.worker:
                self.worker.stop()
            self.camera.close()
            self.pipeline.close()
        finally:
            super().closeEvent(event)


def run_app() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(LIGHT_QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
