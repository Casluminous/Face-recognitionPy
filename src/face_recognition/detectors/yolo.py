"""YOLOv8-face detector — port of YoloFaceDetector.java."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..models import FaceDetection
from .base import FaceDetector

log = logging.getLogger(__name__)
INPUT_SIZE = (640, 640)


def _clamp_rect(x: float, y: float, w: float, h: float, W: int, H: int):
    x = min(max(x, 0), W - 1)
    y = min(max(y, 0), H - 1)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return int(x), int(y), int(w), int(h)


class YoloFaceDetector(FaceDetector):
    ENGINE = "YOLOv8"

    def __init__(self, model_path: str | Path, confidence: float = 0.65, nms: float = 0.45):
        self.model_path = Path(model_path)
        self._conf = float(confidence)
        self._nms = float(nms)
        self._session = None
        self._input_name = None
        if not str(model_path).strip() or not self.model_path.is_file():
            log.error("YOLO model not found: %s", model_path)
            return
        try:
            import onnxruntime as ort
            import os
            cpu = os.cpu_count() or 4
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = max(1, cpu // 2)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(self.model_path), sess_options=opts,
                                                 providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        except Exception as e:
            log.error("Failed to load YOLO model: %s", e)
            self._session = None

    def name(self) -> str: return self.ENGINE
    def is_available(self) -> bool: return self._session is not None
    def confidence_threshold(self) -> float: return self._conf

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if not self.is_available() or frame is None or frame.size == 0:
            return []
        H, W = frame.shape[:2]
        try:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, INPUT_SIZE, (0, 0, 0), swapRB=True, crop=False)
            out = self._session.run(None, {self._input_name: blob})[0]
            return self._process_output(np.asarray(out), W, H)
        except Exception as e:
            log.warning("YOLO detect failed: %s", e)
            return []

    def _process_output(self, data: np.ndarray, W: int, H: int) -> list[FaceDetection]:
        if data.size == 0:
            return []
        # normalize to (numChan, numPred) or detect transpose
        if data.ndim == 3:
            data = data[0]  # (C,N) or (N,C)
        if data.ndim != 2:
            return []
        # YOLOv8n-face: (16,8400) or (8400,16)
        transposed = data.shape[0] > 8000  # (8400,16) -> True
        if transposed:
            num_pred, num_chan = data.shape
        else:
            num_chan, num_pred = data.shape
        x_factor, y_factor = W / 640.0, H / 640.0
        boxes, confs, lms_list = [], [], []
        for i in range(num_pred):
            conf = float(data[i, 4] if transposed else data[4, i])
            if conf <= self._conf:
                continue
            if transposed:
                x, y, w, h = (float(data[i, c]) for c in range(4))
            else:
                x, y, w, h = (float(data[c, i]) for c in range(4))
            ax = W if x <= 1.01 else x_factor
            ay = H if y <= 1.01 else y_factor
            left, top = (x - 0.5 * w) * ax, (y - 0.5 * h) * ay
            bw, bh = w * ax, h * ay
            # landmarks 5 points -> channels 5..14 (same scale as box, mirror Java extractLandmarks)
            lms: list[tuple[float, float]] = []
            if num_chan >= 15:
                for k in range(5):
                    if transposed:
                        lx, ly = float(data[i, 5 + k * 2]), float(data[i, 5 + k * 2 + 1])
                    else:
                        lx, ly = float(data[5 + k * 2, i]), float(data[5 + k * 2 + 1, i])
                    lms.append((lx * ax, ly * ay))
            boxes.append([left, top, bw, bh])
            confs.append(conf)
            lms_list.append(lms)
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, confs, self._conf, self._nms)
        if len(indices) == 0:
            return []
        out: list[FaceDetection] = []
        for idx in np.asarray(indices).flatten():
            b = boxes[int(idx)]
            x, y, w, h = _clamp_rect(b[0], b[1], b[2], b[3], W, H)
            out.append(FaceDetection(x, y, w, h, lms_list[int(idx)], float(confs[int(idx)]), self.ENGINE))
        return out

    def close(self) -> None:
        self._session = None
