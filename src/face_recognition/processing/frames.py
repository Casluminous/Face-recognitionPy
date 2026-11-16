"""FrameProcessor — port of Java FrameProcessor."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import AppConfig
from ..processing.vision_fps import is_likely_overexposed, prepare_for_face_detection
from ..services.face_service import FaceDetectorService
from ..services.gender import GenderCacheManager
from ..services.pipeline import DetectionPipeline
from ..services.recognition import RecognitionService


@dataclass
class FaceRow:
    """One row for the catalog table: identity + prediction of a tracked face."""
    track_id: int = 0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    gender_label: str = ""
    gender_pct: str = ""
    confidence: float = 0.0
    person_name: str = ""
    person_code: str = ""
    person_score: float = 0.0

    @property
    def position(self) -> str:
        return f"({self.x},{self.y}) {self.w}x{self.h}"


@dataclass
class FrameResult:
    display: np.ndarray
    processed: np.ndarray
    status: str
    engine_label: str
    bright_scene: bool
    faces: list[FaceRow] = field(default_factory=list)


class FrameProcessor:
    def __init__(self, pipeline: DetectionPipeline, face_service: FaceDetectorService | None = None,
                 cfg: AppConfig | None = None, recognition: RecognitionService | None = None):
        self.cfg = cfg or AppConfig.get_instance()
        self.pipeline = pipeline
        self.face_service = face_service
        self.recognition = recognition
        self.cache = GenderCacheManager(self.cfg.gender_vote_window, self.cfg.gender_min_votes)
        self._warmup_interval = max(1, self.cfg.gender_warmup_interval)
        self._min_confirm = max(1, self.cfg.gender_min_track_confirm)
        self._last_gender: dict[int, int] = {}
        self._last_recognition: dict[int, int] = {}

    def reset(self) -> None:
        self.cache.clear()
        self._last_gender.clear()
        self._last_recognition.clear()
        if self.recognition is not None:
            self.recognition.invalidate()
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
        # Vote quickly until a label is actually decided (0.3-0.6s on camera),
        # then slow down to the normal interval to save CPU. Using the decided
        # label (not the raw vote count) keeps 2-1 splits from stalling.
        interval = (self.cfg.gender_prediction_interval
                    if self.cache.get(tid) is not None
                    else self._warmup_interval)
        last = self._last_gender.get(tid, -(1 << 62))
        return (cur - last) >= interval

    def _should_recognize(self, tid: int, cur: int) -> bool:
        decision = self.recognition.cache.get(tid) if self.recognition is not None else None
        interval = (self.cfg.recognition_update_interval
                    if decision is not None and decision.state != "pending"
                    else max(1, min(5, self.cfg.recognition_update_interval)))
        last = self._last_recognition.get(tid, -(1 << 62))
        return (cur - last) >= interval

    def process(self, frame: np.ndarray, cur: int, bright_mode: bool, gender_on: bool,
                static: bool = False, recognition_on: bool = True) -> FrameResult:
        """Process one frame.

        ``static`` is for single-shot images: a still has only one detection,
        so confirmation/vote history would hide it forever. There the first
        detection is trusted and gender is shown from a single prediction.
        ``recognition_on`` only disables identity matching; detection, tracking
        and gender continue to work.
        """
        display = self._prepare_display(frame)
        bright_scene = bool(is_likely_overexposed(display, self.cfg))
        analysis = (prepare_for_face_detection(display, self.cfg)
                    if (bright_mode and bright_scene) else display.copy())
        det = self.pipeline.process(display, analysis, cur, bright_scene)
        out = display.copy()
        active: set[int] = set()
        rows: list[FaceRow] = []
        gender_usable = (gender_on and self.face_service is not None
                         and self.face_service.gender_available)
        for face in det.faces:
            active.add(face.track_id)
            # Don't draw a box until a second detection confirms the face.
            # One-frame false positives (profile turn, fast motion) would
            # otherwise flash an extra green rectangle.
            if not static and face.confirm < self._min_confirm:
                continue
            x, y, w, h = face.x, face.y, face.w, face.h
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label, pct = "", ""
            if gender_usable:
                # Predict on the natural image (the CLAHE/gamma analysis frame
                # is for detection only — it skews skin tones).
                if static:
                    pred = self.face_service.predict_gender(display, (x, y, w, h), face.landmarks)
                    if pred is not None:
                        label, pct = pred
                else:
                    if self._should_predict(face.track_id, cur):
                        res = self.face_service.predict_gender(display, (x, y, w, h), face.landmarks)
                        if res is not None:
                            self.cache.put(face.track_id, res)
                        self._last_gender[face.track_id] = cur
                    info = self.cache.get(face.track_id)
                    if info is not None:
                        label, pct = info
                text = f"{label} {pct}".strip()
                if text:
                    cv2.putText(out, text, (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            person_name, person_code, person_score = "", "", 0.0
            if recognition_on and self.recognition is not None:
                if static:
                    match = self.recognition.recognize(display, (x, y, w, h), face.landmarks)
                    if match is not None:
                        person_name = match.name
                        person_code = match.code or ""
                        person_score = match.score
                    else:
                        person_name = "Unknown"
                else:
                    if self._should_recognize(face.track_id, cur):
                        self.recognition.observe(face.track_id, display, (x, y, w, h),
                                                 face.landmarks)
                        self._last_recognition[face.track_id] = cur
                    decision = self.recognition.cache.get(face.track_id)
                    if decision is not None and decision.state == "known" and decision.match:
                        person_name = decision.match.name
                        person_code = decision.match.code or ""
                        person_score = decision.match.score
                    elif decision is not None and decision.state == "unknown":
                        person_name = "Unknown"
                if person_name:
                    py = y - 30 if y >= 30 else y + h + 22
                    text = person_name if not person_code else f"{person_name} ({person_code})"
                    cv2.putText(out, text, (x, max(0, py)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            row = FaceRow(face.track_id, x, y, w, h, label, pct, face.confidence)
            row.person_name = person_name
            row.person_code = person_code
            row.person_score = person_score
            rows.append(row)
        self.cache.retain_all(active)
        if self.recognition is not None:
            self.recognition.retain_tracks(active)
        for tid in list(self._last_gender.keys()):
            if tid not in active:
                del self._last_gender[tid]
        for tid in list(self._last_recognition.keys()):
            if tid not in active:
                del self._last_recognition[tid]
        status = f"{len(rows)} face(s) detected"
        return FrameResult(display, out, status, det.engine_label, bright_scene, rows)
