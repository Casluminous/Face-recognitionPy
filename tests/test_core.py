import numpy as np

from face_recognition.config import AppConfig
from face_recognition.models import FaceDetection
from face_recognition.processing.vision_fps import FpsCalculator, is_likely_overexposed
from face_recognition.services.gender import GenderCacheManager, classify
from face_recognition.services.pipeline import DetectionPipeline
from face_recognition.services.tracker import (
    FaceTracker,
    compute_containment,
    compute_iou,
)


def test_gender_classifier_gating():
    assert classify(0.8, 0.2, 0.7) == ("Male", "80%")
    assert classify(0.6, 0.4, 0.7) == ("Unknown", "60%")
    assert classify(-1, 0.5, 0.7) == ("Unknown", "")


def test_gender_cache_voting():
    c = GenderCacheManager(vote_window=5, min_votes=2)
    c.put(1, ("Male", "80%"))
    c.put(1, ("Female", "90%"))
    c.put(1, ("Male", "82%"))
    label, _pct = c.get(1)
    assert label == "Male"


def test_iou():
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert compute_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_containment():
    assert compute_containment((0, 0, 100, 100), (25, 25, 50, 50)) == 1.0
    assert compute_containment((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


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
    assert cfg.gender_min_votes == 3
    assert cfg.gender_warmup_interval == 4
    assert cfg.tracker_tracked_only_grace == 12
    assert cfg.tracker_containment == 0.80
    assert cfg.tracker_center_match == 0.90
    assert cfg.tracker_center_merge == 0.60
    assert cfg.detection_nms_iou == 0.45


def test_unconfirmed_track_survives_until_next_detection():
    """Regression: tracks were pruned after 4 tracker-only frames while the
    detection cycle is 5 frames, so every face got a new track ID and the
    gender vote window never filled (flaky single-frame Male labels)."""
    cfg = AppConfig.reset_for_tests({
        "tracker.tracked.only.grace.frames": "12",
        "detection.interval.frames": "5",
    })
    tracker = FaceTracker(cfg)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    dets = [FaceDetection(100, 100, 80, 80, [], 0.9, "TEST")]

    first = tracker.update_with_detections(frame, dets)
    tid = first[0].track_id
    assert first[0].confirm == 1

    for _ in range(10):
        tracked = tracker.track(frame)
        assert tracked and tracked[0].track_id == tid

    second = tracker.update_with_detections(frame, dets)
    assert second[0].track_id == tid
    assert second[0].confirm == 2


def test_pipeline_dedupes_duplicate_detector_boxes():
    """Regression: SSD anchor boxes for one face became 2-3 green rectangles."""
    cfg = AppConfig.reset_for_tests({})
    pipe = DetectionPipeline(cfg=cfg)
    dets = [
        FaceDetection(100, 100, 100, 100, [], 0.90, "SSD"),
        FaceDetection(104, 104, 96, 96, [], 0.80, "SSD"),
        FaceDetection(300, 100, 90, 90, [], 0.70, "SSD"),
    ]
    out = pipe._filter(dets, 640, 480)
    assert len(out) == 2
    assert out[0].confidence == 0.90


def test_tracker_dedupes_duplicate_boxes_in_one_frame():
    cfg = AppConfig.reset_for_tests({})
    tracker = FaceTracker(cfg)
    frame = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    dets = [
        FaceDetection(100, 100, 100, 100, [], 0.9, "SSD"),
        FaceDetection(105, 103, 98, 98, [], 0.8, "SSD"),
    ]
    tracks = tracker.update_with_detections(frame, dets)
    assert len(tracks) == 1


def test_tracker_matches_contained_box_instead_of_new_track():
    cfg = AppConfig.reset_for_tests({})
    tracker = FaceTracker(cfg)
    frame = np.random.default_rng(2).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    first = tracker.update_with_detections(
        frame, [FaceDetection(100, 100, 100, 100, [], 0.9, "SSD")])
    tid = first[0].track_id

    # IoU 0.36 (< 0.45) but fully contained -> same face, not a new track.
    second = tracker.update_with_detections(
        frame, [FaceDetection(115, 115, 60, 60, [], 0.8, "SSD")])
    assert len(second) == 1
    assert second[0].track_id == tid


def test_tracker_keeps_separate_faces():
    cfg = AppConfig.reset_for_tests({})
    tracker = FaceTracker(cfg)
    frame = np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    dets = [
        FaceDetection(50, 50, 80, 80, [], 0.9, "SSD"),
        FaceDetection(400, 300, 80, 80, [], 0.9, "SSD"),
    ]
    tracks = tracker.update_with_detections(frame, dets)
    assert len(tracks) == 2


def test_tracker_matches_fast_moving_face_by_center():
    """Regression: a face moving between detection cycles dropped to IoU 0.2
    and spawned a second green box that then followed the same face."""
    cfg = AppConfig.reset_for_tests({})
    tracker = FaceTracker(cfg)
    frame = np.random.default_rng(4).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    first = tracker.update_with_detections(
        frame, [FaceDetection(100, 100, 150, 150, [], 0.9, "SSD")])
    tid = first[0].track_id

    # 100 px jump, same size: IoU 0.20, containment 0.33, center ratio 0.67.
    second = tracker.update_with_detections(
        frame, [FaceDetection(200, 100, 150, 150, [], 0.9, "SSD")])
    assert len(second) == 1
    assert second[0].track_id == tid


def test_tracker_merges_lagging_duplicate_tracks_by_center():
    cfg = AppConfig.reset_for_tests({})
    tracker = FaceTracker(cfg)
    frame = np.random.default_rng(5).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    # Two same-size boxes 76 px apart: IoU 0.28 but clearly one face, so the
    # tracker must keep a single track even without detector NMS.
    dets = [
        FaceDetection(100, 100, 150, 150, [], 0.9, "SSD"),
        FaceDetection(176, 100, 150, 150, [], 0.8, "SSD"),
    ]
    assert len(tracker.update_with_detections(frame, dets)) == 1
