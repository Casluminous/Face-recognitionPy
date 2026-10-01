"""Core dataclasses — port of FaceDetection / TrackedFace records."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class FaceDetection:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    engine: str = ""

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def landmarks_copy(self) -> list[tuple[float, float]]:
        return copy.deepcopy(self.landmarks)


@dataclass
class TrackedFace:
    track_id: int = 0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    tracked_only: bool = False

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class DetectionResult:
    faces: list[TrackedFace]
    engine_label: str
    tracker_only: bool
    confidence: float
