"""VisionPreprocessor + FpsCalculator — port of Java versions."""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import AppConfig


def estimate_brightness(frame: np.ndarray) -> float:
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def is_likely_overexposed(frame: np.ndarray, cfg: AppConfig | None = None) -> bool:
    cfg = cfg or AppConfig.get_instance()
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    ratio = float(cv2.countNonZero(mask) / (gray.shape[0] * gray.shape[1]))
    return mean >= cfg.vision_brightness_threshold or ratio >= cfg.vision_highlight_ratio


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def prepare_for_face_detection(frame: np.ndarray, cfg: AppConfig | None = None) -> np.ndarray:
    cfg = cfg or AppConfig.get_instance()
    try:
        bgr = _ensure_bgr(frame)
        over = is_likely_overexposed(bgr, cfg)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = 3.2 if over else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        gamma = cfg.vision_aggressive_gamma if over else cfg.vision_mild_gamma
        lut = np.array([round(((i / 255.0) ** gamma) * 255) for i in range(256)], dtype=np.uint8)
        l = cv2.LUT(l, lut)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame.copy()


class FpsCalculator:
    def __init__(self, smoothing: float = 0.80):
        self.smoothing = smoothing
        self._fps = 0.0
        self._prev: int | None = None

    def update(self) -> float:
        now = time.monotonic_ns()
        if self._prev is None:
            self._prev = now
            return 0.0
        inst = 1e9 / max(1, now - self._prev)
        self._prev = now
        self._fps = inst if self._fps == 0 else self._fps * self.smoothing + inst * (1 - self.smoothing)
        return self._fps

    def reset(self) -> None:
        self._fps = 0.0
        self._prev = None

    @property
    def current(self) -> float: return self._fps
    def formatted(self) -> str: return f"{self._fps:.1f}"
