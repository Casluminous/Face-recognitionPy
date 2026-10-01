"""Detector interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import FaceDetection


class FaceDetector(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[FaceDetection]: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def confidence_threshold(self) -> float: ...

    def close(self) -> None:
        pass
