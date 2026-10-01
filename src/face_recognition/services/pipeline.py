"""DetectionPipeline — port of Java DetectionPipeline."""
from __future__ import annotations

import math

import numpy as np

from ..config import AppConfig
from ..models import DetectionResult, FaceDetection
from .tracker import FaceTracker


class DetectionPipeline:
    def __init__(self, yolo=None, ssd=None, haar=None, tracker: FaceTracker | None = None,
                 cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig.get_instance()
        self.yolo = yolo
        self.ssd = ssd
        self.haar = haar
        self.tracker = tracker or FaceTracker(self.cfg)
        self._yolo_streak = 0
        self._yolo_cooldown = 0
        self._last_frame = -(1 << 62)
        self._last_label = "Idle"
        self._bright_mode = False

    def set_bright_light_mode(self, b: bool) -> None: self._bright_mode = b
    def is_bright_light_mode(self) -> bool: return self._bright_mode
    def get_last_detector_label(self) -> str: return self._last_label
    def reset(self) -> None:
        self._yolo_streak = 0
        self._yolo_cooldown = 0
        self._last_frame = -(1 << 62)
        self._last_label = "Idle"
        self.tracker.reset()

    def close(self) -> None:
        for d in (self.yolo, self.ssd, self.haar):
            try:
                if d is not None:
                    d.close()
            except Exception:
                pass
        self.tracker.close()

    def process(self, frame: np.ndarray, normalized: np.ndarray, cur: int, is_bright: bool) -> DetectionResult:
        if self._yolo_cooldown > 0:
            self._yolo_cooldown -= 1
        fresh = self._should_run_fresh(cur, is_bright)
        res = self._run_fresh(frame, normalized, cur, is_bright) if fresh else self._run_tracker_only(frame, cur, is_bright)
        if not res.faces and not fresh:
            res = self._run_fresh(frame, normalized, cur, is_bright)
        return res

    def _should_run_fresh(self, cur: int, bright: bool) -> bool:
        if self.tracker.is_empty():
            return True
        if cur - self._last_frame >= self.cfg.detection_interval_frames:
            return True
        if bright and self._bright_mode and cur - self._last_frame >= 2:
            return True
        return False

    def _run_fresh(self, orig: np.ndarray, norm: np.ndarray, cur: int, bright: bool) -> DetectionResult:
        faces, label, conf = self._detect_with_fallbacks(orig, norm, bright)
        H, W = orig.shape[:2]
        filtered = self._filter(faces, W, H)
        tracked = self.tracker.update_with_detections(orig, filtered)
        self._last_label = label
        self._last_frame = cur
        return DetectionResult(tracked, label, False, conf)

    def _run_tracker_only(self, orig: np.ndarray, cur: int, bright: bool) -> DetectionResult:
        tracked = self.tracker.track(orig)
        if not tracked:
            return DetectionResult([], self._last_label, True, 0.0)
        suffix = " - Bright-Light" if bright and self._bright_mode else ""
        return DetectionResult(tracked, f"Tracker - {self._last_label}{suffix}", True, tracked[0].confidence)

    def _suffix(self, variant: str) -> str:
        return "" if variant == "Native" else f" - {variant}"

    def _detect_with_fallbacks(self, orig: np.ndarray, norm: np.ndarray, bright: bool):
        prefer_norm = bright and self._bright_mode
        H, W = orig.shape[:2]
        small_ratio = self.tracker.average_face_area_ratio(W, H) if not self.tracker.is_empty() else math.inf
        yolo_ok = self.yolo is not None and self.yolo.is_available() and self._yolo_cooldown == 0
        prefer_yolo = yolo_ok and (bright or small_ratio < self.cfg.small_face_ratio)
        sources = [(norm, True), (orig, False)] if prefer_norm else [(orig, False), (norm, True)]
        for src, is_norm in sources:
            if src is None:
                continue
            variant = ("Bright-Light" if is_norm and bright else ("Contrast" if is_norm else "Native"))
            if prefer_yolo:
                dets, label, conf = self._try_yolo(src, variant)
                if dets: return dets, label, conf
                dets, label, conf = self._try_ssd(src, variant)
                if dets: return dets, label, conf
            else:
                # mirror Java: SSD then YOLO per source when YOLO not preferred
                dets, label, conf = self._try_ssd(src, variant)
                if dets: return dets, label, conf
                dets, label, conf = self._try_yolo(src, variant)
                if dets: return dets, label, conf
        dets, label, conf = self._try_haar(norm if prefer_norm else orig,
                                           "Bright-Light" if prefer_norm and bright else ("Contrast" if prefer_norm else "Native"))
        if dets: return dets, label, conf
        if prefer_norm:
            dets, label, conf = self._try_haar(orig, "Native")
            if dets: return dets, label, conf
        return [], ("Searching (Bright-Light)" if bright else "Searching..."), 0.0

    def _filter(self, dets: list[FaceDetection], W: int, H: int) -> list[FaceDetection]:
        min_size = H * self.cfg.min_face_size_ratio
        out = []
        for d in dets:
            if d.w < min_size or d.h < min_size:
                continue
            ratio = d.w / max(1, d.h)
            if ratio < self.cfg.min_aspect_ratio or ratio > self.cfg.max_aspect_ratio:
                continue
            out.append(d)
        return out

    def _try_yolo(self, frame: np.ndarray, variant: str):
        if self.yolo is None or not self.yolo.is_available() or self._yolo_cooldown > 0:
            return [], "YOLO cooldown", 0.0
        dets = self.yolo.detect(frame)
        if dets:
            self._yolo_streak = 0
            return dets, f"YOLOv8{self._suffix(variant)}", dets[0].confidence
        self._yolo_streak += 1
        if self._yolo_streak >= self.cfg.yolo_failures_before_cooldown:
            self._yolo_cooldown = self.cfg.yolo_cooldown_frames
            self._yolo_streak = 0
        return [], "YOLO miss", 0.0

    def _try_ssd(self, frame: np.ndarray, variant: str):
        if self.ssd is None or not self.ssd.is_available():
            return [], "SSD miss", 0.0
        dets = self.ssd.detect(frame)
        if dets:
            return dets, f"SSD Caffe{self._suffix(variant)}", 1.0
        return [], "SSD miss", 0.0

    def _try_haar(self, frame: np.ndarray, variant: str):
        if self.haar is None or not self.haar.is_available():
            return [], "Haar miss", 0.0
        dets = self.haar.detect(frame)
        if dets:
            return dets, f"Haar{self._suffix(variant)}", 0.75
        return [], "Haar miss", 0.0
