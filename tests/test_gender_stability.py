"""Tests for gender-label stability (min-vote gating + crop quality gates).

Regression guard for the camera-startup bug: a single blurry/overexposed
frame used to label the face "Male" for several seconds.
"""
import numpy as np

from face_recognition.config import AppConfig
from face_recognition.models import DetectionResult, TrackedFace
from face_recognition.processing.frames import FrameProcessor
from face_recognition.services.face_service import crop_quality_ok, square_crop
from face_recognition.services.gender import GenderCacheManager


def test_cache_waits_for_min_votes():
    c = GenderCacheManager(vote_window=5, min_votes=3)
    c.put(1, ("Male", "91%"))
    assert c.get(1) is None
    c.put(1, ("Male", "93%"))
    assert c.get(1) is None
    c.put(1, ("Male", "90%"))
    assert c.get(1) == ("Male", "91%")


def test_single_vote_cannot_flip_label():
    c = GenderCacheManager(vote_window=5, min_votes=3)
    for pct in ("95%", "94%", "96%"):
        c.put(2, ("Female", pct))
    assert c.get(2)[0] == "Female"
    c.put(2, ("Male", "99%"))
    assert c.get(2)[0] == "Female"


def test_unknown_votes_stay_hidden_until_min_votes():
    c = GenderCacheManager(vote_window=5, min_votes=2)
    c.put(3, ("Unknown", "60%"))
    assert c.get(3) is None
    c.put(3, ("Unknown", "61%"))
    assert c.get(3) == ("Unknown", "")


def test_crop_quality_gate():
    checker = (np.indices((227, 227)).sum(axis=0) % 2 * 255).astype(np.uint8)
    sharp = np.dstack([checker] * 3)
    assert crop_quality_ok(sharp, 227, 8.0, 40.0, 225.0)

    flat = np.full((227, 227, 3), 128, np.uint8)
    assert not crop_quality_ok(flat, 227, 8.0, 40.0, 225.0)

    dark = np.full((227, 227, 3), 5, np.uint8)
    assert not crop_quality_ok(dark, 227, 0.0, 40.0, 225.0)

    blown = np.full((227, 227, 3), 250, np.uint8)
    assert not crop_quality_ok(blown, 227, 0.0, 40.0, 225.0)


def test_square_crop_centered_and_square():
    img = np.zeros((100, 120, 3), np.uint8)
    crop = square_crop(img, (40, 30, 20, 40), 0.30)
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[0] <= 100 and crop.shape[1] <= 120


class _StubPipeline:
    def __init__(self, face: TrackedFace):
        self.face = face

    def process(self, *args, **kwargs):
        return DetectionResult([self.face], "TEST", False, 1.0)

    def reset(self):
        pass


class _StubService:
    gender_available = True

    def __init__(self, result=("Male", "90%")):
        self.result = result
        self.calls = 0

    def predict_gender(self, image, rect, landmarks=None):
        self.calls += 1
        return self.result


def _cfg(**props):
    base = {
        "detection.gender.warmup.interval": "1",
        "detection.gender.min.votes": "3",
        "detection.gender.min.track.confirm": "2",
        "detection.gender.prediction.interval": "30",
    }
    base.update(props)
    return AppConfig.reset_for_tests(base)


def test_frame_processor_hides_label_until_votes_agree():
    cfg = _cfg()
    face = TrackedFace(track_id=1, x=100, y=100, w=120, h=120, confidence=0.9, confirm=2)
    svc = _StubService()
    proc = FrameProcessor(_StubPipeline(face), svc, cfg)
    frame = np.full((480, 640, 3), 128, np.uint8)

    r1 = proc.process(frame, 1, False, True)
    r2 = proc.process(frame, 2, False, True)
    assert r1.faces[0].gender_label == ""
    assert r2.faces[0].gender_label == ""
    assert svc.calls == 2

    r3 = proc.process(frame, 3, False, True)
    assert r3.faces[0].gender_label == "Male"
    assert svc.calls == 3

    # After the vote is decided the normal (slow) interval applies again.
    proc.process(frame, 4, False, True)
    assert svc.calls == 3


def test_frame_processor_skips_unconfirmed_tracks():
    cfg = _cfg()
    face = TrackedFace(track_id=7, x=100, y=100, w=120, h=120, confidence=0.9, confirm=1)
    svc = _StubService()
    proc = FrameProcessor(_StubPipeline(face), svc, cfg)
    res = proc.process(np.full((480, 640, 3), 128, np.uint8), 1, False, True)
    assert svc.calls == 0
    assert res.faces == []


def test_static_image_confirms_first_track_and_gender():
    """Static image processing gets only one frame; it must not wait for a
    second detection or three gender votes, or Open Image shows nothing."""
    cfg = _cfg()
    face = TrackedFace(track_id=9, x=100, y=100, w=120, h=120, confidence=0.9, confirm=1)
    svc = _StubService()
    proc = FrameProcessor(_StubPipeline(face), svc, cfg)
    res = proc.process(np.full((480, 640, 3), 128, np.uint8), 1, False, True, static=True)
    assert len(res.faces) == 1
    assert res.faces[0].gender_label == "Male"
    assert svc.calls == 1
    assert proc.cache.size() == 0


def test_frame_processor_ignores_rejected_crops():
    cfg = _cfg()
    face = TrackedFace(track_id=3, x=100, y=100, w=120, h=120, confidence=0.9, confirm=2)
    svc = _StubService(result=None)
    proc = FrameProcessor(_StubPipeline(face), svc, cfg)
    res = proc.process(np.full((480, 640, 3), 128, np.uint8), 1, False, True)
    assert svc.calls == 1
    assert res.faces[0].gender_label == ""
    assert proc.cache.size() == 0
