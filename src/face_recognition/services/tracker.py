"""FaceTracker — port of Java FaceTracker (IoU + template matching)."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import AppConfig
from ..models import FaceDetection, TrackedFace


def compute_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(r: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = r
    return (x + w / 2.0, y + h / 2.0)


def _expand(r: tuple[int, int, int, int], W: int, H: int, ratio: float):
    x, y, w, h = r
    ex, ey = round(w * ratio), round(h * ratio)
    nx = max(0, x - ex)
    ny = max(0, y - ey)
    nw = min(W - nx, w + 2 * ex)
    nh = min(H - ny, h + 2 * ey)
    return (nx, ny, nw, nh)


def _clamp(r: tuple[int, int, int, int], W: int, H: int):
    x, y, w, h = r
    x = min(max(x, 0), W - 1)
    y = min(max(y, 0), H - 1)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)


@dataclass
class _Track:
    tid: int
    bbox: tuple[int, int, int, int]
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    template: np.ndarray | None = None
    missed: int = 0
    confirm: int = 1
    consec_tracked_only: int = 0
    last_score: float = 0.0
    tracked_only: bool = False


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class FaceTracker:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig.get_instance()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._lock = threading.Lock()

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._tracks) == 0

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._next_id = 1

    def close(self) -> None:
        self.reset()

    def average_face_area_ratio(self, W: int, H: int) -> float:
        with self._lock:
            if not self._tracks or W <= 0 or H <= 0:
                return 0.0
            return float(np.mean([t.bbox[2] * t.bbox[3] / (W * H) for t in self._tracks]))

    def update_with_detections(self, frame: np.ndarray, dets: list[FaceDetection]) -> list[TrackedFace]:
        gray = _to_gray(frame)
        H, W = gray.shape[:2]
        with self._lock:
            # greedy matching by score
            pairs: list[tuple[float, int, int]] = []
            for ti, t in enumerate(self._tracks):
                for di, d in enumerate(dets):
                    iou = compute_iou(t.bbox, d.bbox)
                    if iou < self.cfg.tracker_iou:
                        continue
                    ax, ay = _center(t.bbox)
                    bx, by = _center(d.bbox)
                    diag = math.hypot(max(t.bbox[2], d.bbox[2]), max(t.bbox[3], d.bbox[3])) or 1.0
                    penalty = min(1.0, math.hypot(ax - bx, ay - by) / diag)
                    score = iou * 0.75 + (1 - penalty) * 0.25
                    pairs.append((score, ti, di))
            pairs.sort(reverse=True)
            matched_t, matched_d = set(), set()
            for score, ti, di in pairs:
                if ti in matched_t or di in matched_d:
                    continue
                matched_t.add(ti)
                matched_d.add(di)
                self._update_from_detection(self._tracks[ti], dets[di], gray, False, score)
            for ti, t in enumerate(self._tracks):
                if ti not in matched_t:
                    t.missed += 1
            for di, d in enumerate(dets):
                if di not in matched_d:
                    self._create_track(d, gray, W, H)
            self._prune()
            return self._snapshot_locked()

    def track(self, frame: np.ndarray) -> list[TrackedFace]:
        gray = _to_gray(frame)
        H, W = gray.shape[:2]
        s = self.cfg.tracker_smoothing
        with self._lock:
            for t in self._tracks:
                safe = _clamp(t.bbox, W, H)
                if safe[2] < 8 or safe[3] < 8 or t.template is None:
                    t.missed += 1
                    continue
                search = _expand(safe, W, H, self.cfg.tracker_expansion)
                th, tw = t.template.shape[:2]
                if search[2] < tw or search[3] < th:
                    t.missed += 1
                    continue
                roi = gray[search[1]:search[1] + search[3], search[0]:search[0] + search[2]]
                res = cv2.matchTemplate(roi, t.template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val < self.cfg.tracker_match:
                    t.missed += 1
                    continue
                cand = (search[0] + max_loc[0], search[1] + max_loc[1], tw, th)
                old = t.bbox
                blended = (round(old[0] * s + cand[0] * (1 - s)),
                           round(old[1] * s + cand[1] * (1 - s)),
                           round(old[2] * s + cand[2] * (1 - s)),
                           round(old[3] * s + cand[3] * (1 - s)))
                dx, dy = blended[0] - old[0], blended[1] - old[1]
                t.bbox = _clamp(blended, W, H)
                t.landmarks = [(x + dx, y + dy) for x, y in t.landmarks]
                t.last_score = float(max_val)
                t.tracked_only = True
                t.consec_tracked_only += 1
                t.missed = 0
                self._refresh_locked(t, gray, W, H)
            self._prune()
            return self._snapshot_locked()

    # ---- internals (lock held) ----
    def _create_track(self, d: FaceDetection, gray: np.ndarray, W: int, H: int) -> None:
        t = _Track(self._next_id, _clamp(d.bbox, W, H), list(d.landmarks), None, 0, 1, 0, d.confidence, False)
        self._next_id += 1
        self._refresh_locked(t, gray, W, H)
        self._tracks.append(t)

    def _update_from_detection(self, t: _Track, d: FaceDetection, gray: np.ndarray,
                               tracked_only: bool, score: float) -> None:
        H, W = gray.shape[:2]
        s = self.cfg.tracker_smoothing
        old = t.bbox
        nb = d.bbox
        t.bbox = _clamp((round(old[0] * s + nb[0] * (1 - s)), round(old[1] * s + nb[1] * (1 - s)),
                         round(old[2] * s + nb[2] * (1 - s)), round(old[3] * s + nb[3] * (1 - s))), W, H)
        t.landmarks = list(d.landmarks)
        t.last_score = max(score, d.confidence)
        if not tracked_only:
            t.confirm += 1
            t.consec_tracked_only = 0
        t.missed = 0
        self._refresh_locked(t, gray, W, H)

    def _refresh_locked(self, t: _Track, gray: np.ndarray, W: int, H: int) -> None:
        r = _clamp(t.bbox, W, H)
        if r[2] < 8 or r[3] < 8:
            return
        t.template = gray[r[1]:r[1] + r[3], r[0]:r[0] + r[2]].copy()

    def _prune(self) -> None:
        mx = self.cfg.tracker_max_missed
        self._tracks = [t for t in self._tracks
                        if not (t.missed > mx or (t.confirm < 2 and t.consec_tracked_only > 3)
                                or (t.confirm < 3 and t.consec_tracked_only > 8))]

    def _snapshot_locked(self) -> list[TrackedFace]:
        out = [TrackedFace(t.tid, *t.bbox, list(t.landmarks), t.last_score, t.tracked_only)
               for t in self._tracks]
        out.sort(key=lambda f: f.track_id)
        return out
