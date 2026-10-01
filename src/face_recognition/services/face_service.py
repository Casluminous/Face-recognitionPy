"""FaceDetectorService (SSD + gender) — port of Java version."""
from __future__ import annotations

import logging
import math
import threading

import cv2
import numpy as np

from ..config import AppConfig
from .gender import classify

log = logging.getLogger(__name__)


class FaceDetectorService:
    def __init__(self, ssd_model: str, ssd_config: str, gender_model: str, gender_config: str,
                 cfg: AppConfig | None = None, confidence: float = 0.35):
        self.cfg = cfg or AppConfig.get_instance()
        self._conf = float(confidence)
        self._lock = threading.Lock()
        self.face_net = cv2.dnn.readNetFromCaffe(ssd_config, ssd_model)
        try:
            self.gender_net = cv2.dnn.readNetFromCaffe(gender_config, gender_model)
            self._gender_ok = not self.gender_net.empty()
        except Exception as e:
            log.warning("Gender net unavailable: %s", e)
            self.gender_net = None
            self._gender_ok = False
        fm = self.cfg.parse_double_array("detection.face.mean", [104.0, 177.0, 123.0])
        gm = self.cfg.parse_double_array("detection.gender.mean",
                                         [78.4263377603, 87.7689143744, 114.895847746])
        self.face_mean = tuple(fm[:3])
        self.gender_mean = tuple(gm[:3])
        self.face_size = self.cfg.face_input_size
        self.gender_size = self.cfg.gender_input_size
        self.gender_min_conf = self.cfg.gender_min_confidence

    @property
    def gender_available(self) -> bool: return self._gender_ok

    def has_usable_landmarks(self, landmarks, rect, W: int, H: int) -> bool:
        if landmarks is None or len(landmarks) < 3:
            return False
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return False
        mx, my = w * 0.25, h * 0.25
        pts = landmarks[:3]
        for (px, py) in pts:
            if not (math.isfinite(px) and math.isfinite(py)):
                return False
            if not (0 <= px < W and 0 <= py < H):
                return False
            if not (x - mx <= px <= x + w + mx and y - my <= py <= y + h + my):
                return False
        (lx, ly), (rx, ry), (nx, ny) = pts[0], pts[1], pts[2]
        eye_dist = math.hypot(lx - rx, ly - ry)
        if not (w * 0.15 <= eye_dist <= w * 1.2):
            return False
        if abs(ly - ry) > h * 0.5:
            return False
        eye_y = (ly + ry) / 2
        if not (eye_y - h * 0.15 <= ny <= eye_y + h * 1.15):
            return False
        if not (min(lx, rx) - w * 0.35 <= nx <= max(lx, rx) + w * 0.35):
            return False
        return True

    def _aligned_face(self, image: np.ndarray, rect, landmarks):
        S = self.gender_size
        dst = np.array([[S * 0.32, S * 0.38], [S * 0.68, S * 0.38], [S * 0.50, S * 0.60]], dtype=np.float32)
        src = np.array(landmarks[:3], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(image, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _default_crop(self, image: np.ndarray, rect):
        H, W = image.shape[:2]
        x, y, w, h = rect
        padx, pady = round(w * 0.18), round(h * 0.22)
        x1, y1 = max(0, x - padx), max(0, y - pady)
        x2, y2 = min(W, x + w + padx), min(H, y + h + pady)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()

    def predict_gender(self, image: np.ndarray, rect, landmarks=None) -> tuple[str, str]:
        if not self._gender_ok or image is None or image.size == 0:
            return ("Unknown", "")
        H, W = image.shape[:2]
        face = None
        try:
            if landmarks and self.has_usable_landmarks(landmarks, rect, W, H):
                face = self._aligned_face(image, rect, landmarks)
            else:
                face = self._default_crop(image, rect)
        except Exception:
            face = None
        if face is None or face.size == 0:
            return ("Unknown", "")
        blob = cv2.dnn.blobFromImage(face, 1.0, (self.gender_size, self.gender_size),
                                     self.gender_mean, swapRB=False, crop=False)
        with self._lock:
            self.gender_net.setInput(blob)
            preds = self.gender_net.forward()
        p0, p1 = float(preds[0, 0]), float(preds[0, 1])
        return classify(p0, p1, self.gender_min_conf)

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        H, W = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (self.face_size, self.face_size),
                                     self.face_mean, swapRB=False, crop=False)
        with self._lock:
            self.face_net.setInput(blob)
            dets = self.face_net.forward()
        out = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf <= self._conf:
                continue
            l, t, r, b = (float(dets[0, 0, i, k]) for k in range(3, 7))
            x1, y1 = max(0, int(l * W)), max(0, int(t * H))
            x2, y2 = min(W - 1, int(r * W)), min(H - 1, int(b * H))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2 - x1, y2 - y1))
        return out

    def detect_and_draw(self, image: np.ndarray) -> int:
        rects = self.detect_faces(image)
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label, pct = self.predict_gender(image, (x, y, w, h))
            cv2.putText(image, f"{label} {pct}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return len(rects)

    def close(self) -> None:
        pass
