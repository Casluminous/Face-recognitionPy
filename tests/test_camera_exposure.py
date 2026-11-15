"""CameraManager exposure-tuning guard tests (no physical camera needed)."""
import numpy as np

from face_recognition.config import AppConfig
from face_recognition.services.camera import CameraManager


class _FakeCap:
    """Minimal VideoCapture stand-in whose driver rejects every write."""

    def __init__(self):
        self.set_calls = 0

    def isOpened(self):
        return True

    def set(self, *_args):
        self.set_calls += 1
        return False

    def get(self, *_args):
        return 0.0

    def read(self):
        return True, np.zeros((10, 10, 3), np.uint8)

    def release(self):
        pass


def test_exposure_driver_lock_stops_retuning():
    cfg = AppConfig.reset_for_tests({
        "camera.exposure.target.brightness": "10",
        "camera.exposure.retune.interval": "0",
        "camera.exposure.settle.ms": "0",
        "camera.exposure.manual.modes": "1.0",
    })
    cam = CameraManager(cfg)
    cam.cap = _FakeCap()
    bright = np.full((10, 10, 3), 200, np.uint8)

    assert cam.optimize_exposure(10, bright) is False
    assert cam._driver_locked is True
    assert "Driver locked" in cam.status

    calls = cam.cap.set_calls
    assert cam.optimize_exposure(20, bright) is False
    assert cam.cap.set_calls == calls
