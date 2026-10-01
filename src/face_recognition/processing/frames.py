"""FrameProcessor — port of Java FrameProcessor."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import AppConfig
from ..processing.vision_fps import is_likely_overexposed, prepare_for_face_detection
from ..services.face_service import FaceDetectorService
from ..services.gender import GenderCacheManager
from ..services.pipeline import DetectionPipeline


@dataclass
class FrameResult:
    display: np.ndarray
    processed: np.ndarray
    status: str
    engine_label: str
    bright_scene: bool


class FrameProcessor:
    def __init__(self, pipeline: DetectionPipeline, face_service: FaceDetectorService | None = None,
                 cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig.get_instance()
        self.pipeline = pipeline
        self.face_service = face_service
        self.cache = GenderCacheManager(self.cfg.gender_vote_window)
        self._last_gender: dict[int, int] = {}

    def reset(self) -> None:
        self.cache.clear()
        self._last_gender.clear()
        self.pipeline.reset()

    def close(self) -> None:
        pass

    def _prepare_display(self, frame: np.ndarray) -> np.ndarray:
        f = frame
        if len(f.shape) == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        else:
            f = f.copy()
        max_w = self.cfg.display_max_width
        if f.shape[1] > max_w:
            scale = max_w / f.shape[1]
            f = cv2.resize(f, (max_w, round(f.shape[0] * scale)))
        return f

    def _should_predict(self, tid: int, cur: int) -> bool:
        last = self._last_gender.get(tid, -(1 << 62))
        has = self.cache.contains(tid)
        return (not has) or (cur - last) >= self.cfg.gender_prediction_interval

    def process(self, frame: np.ndarray, cur: int, bright_mode: bool, gender_on: bool) -> FrameResult:
        display = self._prepare_display(frame)
        bright_scene = bool(is_likely_overexposed(display, self.cfg))
        analysis = (prepare_for_face_detection(display, self.cfg)
                    if (bright_mode and bright_scene) else display.copy())
        det = self.pipeline.process(display, analysis, cur, bright_scene)
        out = display.copy()
        active: set[int] = set()
        for face in det.faces:
            active.add(face.track_id)
            x, y, w, h = face.x, face.y, face.w, face.h
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if gender_on and self.face_service is not None and self.face_service.gender_available:
                if self._should_predict(face.track_id, cur):
                    res = self.face_service.predict_gender(analysis, (x, y, w, h), face.landmarks)
                    self.cache.put(face.track_id, res)
                    self._last_gender[face.track_id] = cur
            if gender_on:
                info = self.cache.get(face.track_id)
                if info is not None:
                    cv2.putText(out, f"{info[0]} {info[1]}", (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self.cache.retain_all(active)
        for tid in list(self._last_gender.keys()):
            if tid not in active:
                del self._last_gender[tid]
        status = f"{len(det.faces)} face(s) detected"
        return FrameResult(display, out, status, det.engine_label, bright_scene)
