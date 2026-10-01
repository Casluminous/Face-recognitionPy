import numpy as np

from face_recognition.config import AppConfig
from face_recognition.models import FaceDetection
from face_recognition.processing.vision_fps import FpsCalculator, is_likely_overexposed
from face_recognition.services.gender import GenderCacheManager, classify
from face_recognition.services.pipeline import DetectionPipeline
from face_recognition.services.tracker import FaceTracker, compute_iou


def test_gender_classifier_gating():
    assert classify(0.8, 0.2, 0.7) == ("Male", "80%")
    assert classify(0.6, 0.4, 0.7) == ("Unknown", "60%")
    assert classify(-1, 0.5, 0.7) == ("Unknown", "")


def test_gender_cache_voting():
    c = GenderCacheManager(vote_window=5)
    c.put(1, ("Male", "80%"))
    c.put(1, ("Female", "90%"))
    c.put(1, ("Male", "82%"))
    label, pct = c.get(1)
    assert label == "Male"


def test_iou():
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert compute_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_tracker_and_pipeline_filter():
    cfg = AppConfig.reset_for_tests({})
    tracker = FaceTracker(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [FaceDetection(100, 100, 80, 80, [], 0.9, "TEST")]
    tracked = tracker.update_with_detections(frame, dets)
    assert len(tracked) == 1
    # too-small face filtered by pipeline
    pipe = DetectionPipeline(yolo=None, ssd=None, haar=None, tracker=FaceTracker(cfg), cfg=cfg)
    tiny = [FaceDetection(10, 10, 5, 5, [], 0.9, "TEST")]
    assert pipe._filter(tiny, 640, 480) == []


def test_overexposure_and_fps():
    cfg = AppConfig.reset_for_tests({})
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert is_likely_overexposed(white, cfg) is True
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert is_likely_overexposed(black, cfg) is False
    f = FpsCalculator()
    assert f.update() == 0.0


def test_config_defaults():
    cfg = AppConfig.reset_for_tests({})
    assert cfg.yolo_confidence == 0.65
    assert cfg.gender_min_confidence == 0.70
    assert cfg.tracker_iou == 0.45
