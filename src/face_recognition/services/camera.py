"""CameraManager — port of Java CameraManager (adaptive exposure)."""
from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from ..config import AppConfig
from ..processing.vision_fps import estimate_brightness

log = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig.get_instance()
        self.cap: cv2.VideoCapture | None = None
        self._active = False
        self._thread: threading.Thread | None = None
        self.camera_index = 0
        self.adaptive = True
        self._last_tune = -(1 << 62)
        self.brightness = 0.0
        self.status = "Exposure: Ready"

    @staticmethod
    def detect_cameras(max_index: int = 10) -> list[int]:
        found = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append(i)
                cap.release()
        return found

    def open(self, index: int) -> bool:
        self.close()
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_default_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_default_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            cap.read()
        self.cap = cap
        self.camera_index = index
        return True

    def start_capture(self, handler: Callable[[np.ndarray], None]) -> bool:
        if self.cap is None or not self.cap.isOpened():
            return False
        if self._active:
            return True
        self._active = True
        interval = self.cfg.camera_frame_interval_ms / 1000.0

        def loop():
            cur = 0
            while self._active:
                ok, frame = self.cap.read() if self.cap else (False, None)
                if not ok or frame is None:
                    time.sleep(interval)
                    continue
                cur += 1
                if self.adaptive:
                    try:
                        self.optimize_exposure(cur, frame)
                    except Exception as e:
                        log.debug("exposure tune failed: %s", e)
                try:
                    handler(frame.copy())
                except Exception as e:
                    log.debug("frame handler failed: %s", e)
                time.sleep(interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._active = False
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=2.0)

    def close(self) -> None:
        self.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # ---- exposure ----
    def _sample_brightness(self, n: int) -> float:
        if self.cap is None:
            return 0.0
        vals = []
        for _ in range(n):
            ok, f = self.cap.read()
            if ok and f is not None:
                vals.append(estimate_brightness(f))
        return float(sum(vals) / len(vals)) if vals else 0.0

    def optimize_exposure(self, cur: int, seed: np.ndarray | None = None) -> bool:
        if not self.adaptive or self.cap is None or not self.cap.isOpened():
            return False
        if cur - self._last_tune < self.cfg.exposure_retune_interval:
            return False
        base = float(estimate_brightness(seed)) if seed is not None else self._sample_brightness(3)
        self._last_tune = cur
        self.brightness = base
        if base <= 0 or base <= self.cfg.exposure_target:
            self.status = f"Exposure: Stable {base:.0f}"
            return False
        return self._adjust(base)

    def _build_candidates(self, orig: float) -> list[float]:
        if not math.isfinite(orig):
            return [-4.0, -5.0, -6.0, -7.0]
        if orig <= 0:
            return [orig - 1, orig - 2, orig - 3, orig - 4]
        return [orig * 0.8, orig * 0.65, orig * 0.5, orig * 0.35]

    def _set_manual(self) -> bool:
        modes = self.cfg.parse_double_array("camera.exposure.manual.modes", [1.0, 0.25, 0.0])
        for m in modes:
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, m)
                time.sleep(0.08)
            except Exception:
                continue
        return True

    def _adjust(self, base: float) -> bool:
        self._set_manual()
        try:
            orig = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        except Exception:
            orig = 0.0
        settle = self.cfg.exposure_settle_ms / 1000.0
        best, best_b = None, base
        for cand in self._build_candidates(orig):
            try:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, cand)
            except Exception:
                continue
            time.sleep(settle)
            cb = self._sample_brightness(2)
            if cb >= self.cfg.exposure_min and cb < best_b:
                best, best_b = cand, cb
            if cb <= self.cfg.exposure_target and base - cb >= self.cfg.exposure_drop:
                break
        if best is not None:
            try:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, best)
            except Exception:
                pass
            time.sleep(settle)
            self._retune_gain_wb(best_b)
            self.status = f"Exposure: Tuned {base:.0f} -> {best_b:.0f}"
            return True
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, orig)
        except Exception:
            pass
        self.status = "Exposure: Driver locked"
        return False

    def _retune_gain_wb(self, b: float) -> None:
        if b <= 185:
            return
        try:
            g = float(self.cap.get(cv2.CAP_PROP_GAIN))
            if g > 0:
                self.cap.set(cv2.CAP_PROP_GAIN, max(0, g - 1))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
        except Exception:
            pass
        time.sleep(0.05)

    def set_adaptive(self, enabled: bool) -> None:
        self.adaptive = enabled
        if enabled:
            self.status = "Exposure: Assist enabled"
            self._last_tune = -(1 << 62)
        else:
            try:
                for m in self.cfg.parse_double_array("camera.exposure.automatic.modes", [3.0, 0.75]):
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, m)
                    time.sleep(0.08)
            except Exception:
                pass
            self.status = "Exposure: Driver auto/manual hold"
