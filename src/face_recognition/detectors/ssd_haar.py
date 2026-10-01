"""SSD + Haar detectors — port of SsdFaceDetector / HaarFaceDetector."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..models import FaceDetection
from .base import FaceDetector

log = logging.getLogger(__name__)


class SsdFaceDetector(FaceDetector):
    ENGINE = "SSD Caffe"

    def __init__(self, model_path: str | Path, config_path: str | Path,
                 confidence: float = 0.60, input_size: int = 300,
                 mean: tuple[float, float, float] = (104.0, 177.0, 123.0)):
        self._conf = float(confidence)
        self._size = int(input_size)
        self._mean = mean
        self._net = None
        try:
            self._net = cv2.dnn.readNetFromCaffe(str(config_path), str(model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            if self._net.empty():
                self._net = None
        except Exception as e:
            log.error("Failed to load SSD model: %s", e)
            self._net = None

    def name(self) -> str: return self.ENGINE
    def is_available(self) -> bool: return self._net is not None
    def confidence_threshold(self) -> float: return self._conf

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if not self.is_available() or frame is None or frame.size == 0:
            return []
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (self._size, self._size),
                                     self._mean, swapRB=False, crop=False)
        self._net.setInput(blob)
        dets = self._net.forward()
        out: list[FaceDetection] = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf <= self._conf:
                continue
            l, t, r, b = (float(dets[0, 0, i, k]) for k in range(3, 7))
            x1, y1 = max(0, int(l * W)), max(0, int(t * H))
            x2, y2 = min(W - 1, int(r * W)), min(H - 1, int(b * H))
            if x2 > x1 and y2 > y1:
                out.append(FaceDetection(x1, y1, x2 - x1, y2 - y1, [], conf, self.ENGINE))
        return out


class HaarFaceDetector(FaceDetector):
    ENGINE = "Haar"

    def __init__(self, cascade_path: str | Path, confidence: float = 0.75,
                 scale_factor: float = 1.1, min_neighbors: int = 7, min_size: int = 40):
        self._conf = float(confidence)
        self._scale = float(scale_factor)
        self._neighbors = int(min_neighbors)
        self._min_size = int(min_size)
        self._clf = cv2.CascadeClassifier(str(cascade_path))
        if self._clf.empty():
            log.error("Haar cascade not loaded: %s", cascade_path)

    def name(self) -> str: return self.ENGINE
    def is_available(self) -> bool: return not self._clf.empty()
    def confidence_threshold(self) -> float: return self._conf

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if not self.is_available() or frame is None or frame.size == 0:
            return []
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._clf.detectMultiScale(gray, self._scale, self._neighbors,
                                           0, (self._min_size, self._min_size))
        out = []
        for (x, y, w, h) in rects:
            if self._conf >= self.confidence_threshold():
                out.append(FaceDetection(int(x), int(y), int(w), int(h), [], self._conf, self.ENGINE))
        return out
