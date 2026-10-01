"""Desktop UI — PySide6 port of ViewController + UIManager + MainApp."""
from __future__ import annotations

import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget)

from ..config import AppConfig
from ..detectors.ssd_haar import HaarFaceDetector, SsdFaceDetector
from ..detectors.yolo import YoloFaceDetector
from ..model_paths import ModelPaths
from ..processing.frames import FrameProcessor
from ..processing.vision_fps import FpsCalculator
from ..services.camera import CameraManager
from ..services.face_service import FaceDetectorService
from ..services.pipeline import DetectionPipeline
from ..services.prefs import PreferencesService
from ..services.update_security import UpdateService


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

        # ---- widgets ----
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        top = QHBoxLayout()
        self.btn_image = QPushButton("Open Image")
        self.btn_camera = QPushButton("Start Camera")
        self.camera_selector = QComboBox()
        top.addWidget(self.btn_image)
        top.addWidget(self.btn_camera)
        top.addWidget(self.camera_selector)
        layout.addLayout(top)

        views = QHBoxLayout()
        self.source_view = QLabel("Source")
        self.result_view = QLabel("Result")
        for v in (self.source_view, self.result_view):
            v.setAlignment(Qt.AlignCenter)
            v.setMinimumSize(480, 360)
            v.setStyleSheet("background:#222; color:#ddd;")
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
        layout.addLayout(opts)

        self.status_label = QLabel("Idle")
        self.engine_label = QLabel("Engine: --")
        self.fps_label = QLabel("FPS: --")
        self.exposure_label = QLabel("Exposure: Ready")
        for l in (self.status_label, self.engine_label, self.fps_label, self.exposure_label):
            layout.addWidget(l)

        # ---- signals ----
        self.btn_image.clicked.connect(self.select_image)
        self.btn_camera.clicked.connect(self.toggle_camera)
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

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        if self.camera_on:
            self.toggle_camera()
        img = cv2.imread(path)
        if img is None:
            self.status_label.setText("Failed to read image")
            return
        self.processor.reset()
        res = self.processor.process(img, 1, self.chk_bright.isChecked(), self.chk_gender.isChecked())
        self.source_view.setPixmap(_to_pixmap(res.display))
        self.result_view.setPixmap(_to_pixmap(res.processed))
        self.status_label.setText(res.status)
        self.engine_label.setText(f"Engine: {res.engine_label}")
        self.fps_label.setText("FPS: Static")
        self.exposure_label.setText("Exposure: N/A")

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
            self.status_label.setText("Failed to open camera")
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
        self.source_view.setPixmap(_to_pixmap(res.display))
        self.result_view.setPixmap(_to_pixmap(res.processed))
        self.status_label.setText(res.status)
        self.engine_label.setText(f"Engine: {res.engine_label}")
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.exposure_label.setText(self.camera.status)

    def _on_adaptive(self, v: bool):
        self.prefs.set_bool("adaptive_exposure_enabled", v)
        self.camera.set_adaptive(v)

    def _on_bright(self, v: bool):
        self.prefs.set_bool("bright_light_mode_enabled", v)
        self.pipeline.set_bright_light_mode(v)

    def _on_update_found(self, url: str):
        self.status_label.setText(f"New update found: {url}")
        self.status_label.setStyleSheet("color: orange;")

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
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
