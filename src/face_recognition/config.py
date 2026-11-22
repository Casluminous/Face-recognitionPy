"""AppConfig — port of Java AppConfig + application.properties."""
from __future__ import annotations

import threading
from pathlib import Path


_DEFAULTS: dict[str, str] = {
    "camera.frame.interval.ms": "33",
    "camera.buffer.size": "1",
    "camera.default.width": "640",
    "camera.default.height": "480",
    "camera.shutdown.timeout.ms": "2000",
    "detection.interval.frames": "5",
    "detection.gender.prediction.interval": "30",
    "detection.gender.warmup.interval": "4",
    "detection.gender.min.votes": "3",
    "detection.gender.min.track.confirm": "2",
    "detection.gender.min.confidence": "0.70",
    "detection.gender.vote.window": "5",
    "detection.gender.min.face.px": "56",
    "detection.gender.min.sharpness": "3.0",
    "detection.gender.min.brightness": "40.0",
    "detection.gender.max.brightness": "225.0",
    "detection.exposure.retune.interval": "120",
    "detection.box.change.threshold": "0.15",
    "detection.min.face.size.ratio": "0.07",
    "detection.small.face.ratio": "0.045",
    "yolo.confidence.threshold": "0.65",
    "yolo.nms.threshold": "0.45",
    "yolo.failures.before.cooldown": "3",
    "yolo.cooldown.frames": "24",
    "ssd.confidence.threshold": "0.60",
    "camera.exposure.target.brightness": "165.0",
    "camera.exposure.min.brightness": "70.0",
    "camera.exposure.meaningful.drop": "10.0",
    "camera.exposure.settle.ms": "120",
    "camera.exposure.manual.modes": "1.0,0.25,0.0",
    "camera.exposure.automatic.modes": "3.0,0.75",
    "tracker.iou.threshold": "0.45",
    "tracker.match.threshold": "0.58",
    "tracker.search.expansion": "0.45",
    "tracker.rect.smoothing": "0.60",
    "tracker.max.missed.frames": "6",
    "tracker.tracked.only.grace.frames": "12",
    "tracker.containment.threshold": "0.80",
    "tracker.center.match.ratio": "0.90",
    "tracker.center.merge.ratio": "0.60",
    "tracker.merge.iou": "0.55",
    "tracker.merge.containment": "0.85",
    "recognition.enabled": "true",
    "recognition.detector": "yolo",
    "recognition.embedding.model": "face_recognition_sface_2021dec.onnx",
    "recognition.match.threshold": "0.36",
    "recognition.match.margin": "0.05",
    "recognition.min.votes": "3",
    "recognition.vote.window": "4",
    "recognition.update.interval.frames": "15",
    "recognition.min.face.px": "80",
    "recognition.min.sharpness": "3.0",
    "recognition.min.brightness": "40.0",
    "recognition.max.brightness": "225.0",
    "recognition.enrollment.min.images": "1",
    "recognition.enrollment.max.images": "5",
    "recognition.db.path": "data/recognition/people.sqlite3",
    "recognition.people.dir": "data/recognition/people",
    "image.display.max.width": "640",
    "update.check.timeout.seconds": "5",
    "update.download.timeout.seconds": "600",
    "update.github.api.url": "https://api.github.com/repos/OniAntou/Face-Recognition/releases/latest",
    "update.require.authenticode": "false",
    "update.trusted.signer.subject": "",
    "update.trusted.signer.thumbprint": "",
    "performance.ai.thread.priority": "-1",
    "fps.smoothing.factor": "0.80",
    "vision.brightness.threshold": "190.0",
    "vision.highlight.ratio.threshold": "0.16",
    "vision.aggressive.gamma": "1.35",
    "vision.mild.gamma": "1.10",
    "haar.min.face.size": "40",
    "haar.default.confidence": "0.75",
    "haar.scale.factor": "1.1",
    "haar.min.neighbors": "7",
    "detection.min.aspect.ratio": "0.5",
    "detection.max.aspect.ratio": "1.6",
    "detection.nms.iou": "0.45",
    "detection.nms.containment": "0.85",
    "detection.face.mean": "104.0,177.0,123.0",
    "detection.gender.mean": "78.4263377603,87.7689143744,114.895847746",
    "detection.face.input.size": "300",
    "detection.gender.input.size": "227",
    "detection.color.box": "0,255,0",
    "detection.color.label": "0,255,255",
}


def _candidate_property_files() -> list[Path]:
    import sys
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # PyInstaller one-file: config bundled into _MEIPASS, exe dir for overrides
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "config" / "application.properties")
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "config" / "application.properties")
    candidates += [
        Path.cwd() / "config" / "application.properties",
        here.parents[2] / "config" / "application.properties",  # project root/config
        Path(r"C:\Users\steal\Face-Recognition\src\main\resources\application.properties"),
    ]
    return candidates


class AppConfig:
    _instance: "AppConfig | None" = None
    _lock = threading.Lock()

    def __init__(self, props: dict[str, str] | None = None):
        self._props: dict[str, str] = dict(_DEFAULTS)
        if props:
            self._props.update(props)
        else:
            for f in _candidate_property_files():
                if f.is_file():
                    self._props.update(self._parse(f))
                    break

    @classmethod
    def get_instance(cls) -> "AppConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls, props: dict[str, str] | None = None) -> "AppConfig":
        with cls._lock:
            cls._instance = cls(props or {})
        return cls._instance

    @staticmethod
    def _parse(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def get_string(self, key: str, default: str = "") -> str:
        return self._props.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(float(self._props.get(key, str(default))))
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self._props.get(key, str(default)))
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        v = self._props.get(key)
        if v is None:
            return default
        return v.strip().lower() in ("true", "1", "yes", "y")

    def parse_double_array(self, key: str, default: list[float] | None = None) -> list[float]:
        raw = self._props.get(key)
        if raw is None or not raw.strip():
            return list(default or [])
        try:
            return [float(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            return list(default or [])

    # ---- typed shortcuts mirroring Java getters ----
    @property
    def camera_frame_interval_ms(self) -> int: return self.get_int("camera.frame.interval.ms", 33)
    @property
    def camera_default_width(self) -> int: return self.get_int("camera.default.width", 640)
    @property
    def camera_default_height(self) -> int: return self.get_int("camera.default.height", 480)
    @property
    def detection_interval_frames(self) -> int: return self.get_int("detection.interval.frames", 5)
    @property
    def gender_prediction_interval(self) -> int: return self.get_int("detection.gender.prediction.interval", 30)
    @property
    def gender_warmup_interval(self) -> int: return self.get_int("detection.gender.warmup.interval", 4)
    @property
    def gender_min_votes(self) -> int: return self.get_int("detection.gender.min.votes", 3)
    @property
    def gender_min_track_confirm(self) -> int: return self.get_int("detection.gender.min.track.confirm", 2)
    @property
    def gender_min_confidence(self) -> float: return self.get_float("detection.gender.min.confidence", 0.70)
    @property
    def gender_vote_window(self) -> int: return self.get_int("detection.gender.vote.window", 5)
    @property
    def gender_min_face_px(self) -> int: return self.get_int("detection.gender.min.face.px", 56)
    @property
    def gender_min_sharpness(self) -> float: return self.get_float("detection.gender.min.sharpness", 3.0)
    @property
    def gender_min_brightness(self) -> float: return self.get_float("detection.gender.min.brightness", 40.0)
    @property
    def gender_max_brightness(self) -> float: return self.get_float("detection.gender.max.brightness", 225.0)
    @property
    def exposure_retune_interval(self) -> int: return self.get_int("detection.exposure.retune.interval", 120)
    @property
    def min_face_size_ratio(self) -> float: return self.get_float("detection.min.face.size.ratio", 0.07)
    @property
    def small_face_ratio(self) -> float: return self.get_float("detection.small.face.ratio", 0.045)
    @property
    def yolo_confidence(self) -> float: return self.get_float("yolo.confidence.threshold", 0.65)
    @property
    def yolo_nms(self) -> float: return self.get_float("yolo.nms.threshold", 0.45)
    @property
    def yolo_failures_before_cooldown(self) -> int: return self.get_int("yolo.failures.before.cooldown", 3)
    @property
    def yolo_cooldown_frames(self) -> int: return self.get_int("yolo.cooldown.frames", 24)
    @property
    def ssd_confidence(self) -> float: return self.get_float("ssd.confidence.threshold", 0.60)
    @property
    def exposure_target(self) -> float: return self.get_float("camera.exposure.target.brightness", 165.0)
    @property
    def exposure_min(self) -> float: return self.get_float("camera.exposure.min.brightness", 70.0)
    @property
    def exposure_drop(self) -> float: return self.get_float("camera.exposure.meaningful.drop", 10.0)
    @property
    def exposure_settle_ms(self) -> int: return self.get_int("camera.exposure.settle.ms", 120)
    @property
    def tracker_iou(self) -> float: return self.get_float("tracker.iou.threshold", 0.45)
    @property
    def tracker_match(self) -> float: return self.get_float("tracker.match.threshold", 0.58)
    @property
    def tracker_expansion(self) -> float: return self.get_float("tracker.search.expansion", 0.45)
    @property
    def tracker_smoothing(self) -> float: return self.get_float("tracker.rect.smoothing", 0.60)
    @property
    def tracker_max_missed(self) -> int: return self.get_int("tracker.max.missed.frames", 6)
    @property
    def tracker_tracked_only_grace(self) -> int:
        return self.get_int("tracker.tracked.only.grace.frames", 12)
    @property
    def tracker_containment(self) -> float:
        return self.get_float("tracker.containment.threshold", 0.80)
    @property
    def tracker_center_match(self) -> float:
        return self.get_float("tracker.center.match.ratio", 0.90)
    @property
    def tracker_center_merge(self) -> float:
        return self.get_float("tracker.center.merge.ratio", 0.60)
    @property
    def tracker_merge_iou(self) -> float:
        return self.get_float("tracker.merge.iou", 0.55)
    @property
    def tracker_merge_containment(self) -> float:
        return self.get_float("tracker.merge.containment", 0.85)
    @property
    def recognition_enabled(self) -> bool:
        return self.get_bool("recognition.enabled", True)
    @property
    def recognition_detector(self) -> str:
        return self.get_string("recognition.detector", "yolo")
    @property
    def recognition_embedding_model(self) -> str:
        return self.get_string("recognition.embedding.model", "face_recognition_sface_2021dec.onnx")
    @property
    def recognition_match_threshold(self) -> float:
        return self.get_float("recognition.match.threshold", 0.36)
    @property
    def recognition_match_margin(self) -> float:
        return self.get_float("recognition.match.margin", 0.05)
    @property
    def recognition_min_votes(self) -> int:
        return self.get_int("recognition.min.votes", 3)
    @property
    def recognition_vote_window(self) -> int:
        return self.get_int("recognition.vote.window", 4)
    @property
    def recognition_update_interval(self) -> int:
        return self.get_int("recognition.update.interval.frames", 15)
    @property
    def recognition_min_face_px(self) -> int:
        return self.get_int("recognition.min.face.px", 80)
    @property
    def recognition_min_sharpness(self) -> float:
        return self.get_float("recognition.min.sharpness", 3.0)
    @property
    def recognition_min_brightness(self) -> float:
        return self.get_float("recognition.min.brightness", 40.0)
    @property
    def recognition_max_brightness(self) -> float:
        return self.get_float("recognition.max.brightness", 225.0)
    @property
    def recognition_enrollment_min_images(self) -> int:
        return self.get_int("recognition.enrollment.min.images", 1)
    @property
    def recognition_enrollment_max_images(self) -> int:
        return self.get_int("recognition.enrollment.max.images", 5)
    @property
    def recognition_db_path(self) -> str:
        return self.get_string("recognition.db.path", "data/recognition/people.sqlite3")
    @property
    def recognition_people_dir(self) -> str:
        return self.get_string("recognition.people.dir", "data/recognition/people")
    @property
    def display_max_width(self) -> int: return self.get_int("image.display.max.width", 640)
    @property
    def update_check_timeout(self) -> int: return self.get_int("update.check.timeout.seconds", 5)
    @property
    def update_download_timeout(self) -> int: return self.get_int("update.download.timeout.seconds", 600)
    @property
    def update_api_url(self) -> str: return self.get_string("update.github.api.url", _DEFAULTS["update.github.api.url"])
    @property
    def require_authenticode(self) -> bool: return self.get_bool("update.require.authenticode", False)
    @property
    def fps_smoothing(self) -> float: return self.get_float("fps.smoothing.factor", 0.80)
    @property
    def vision_brightness_threshold(self) -> float: return self.get_float("vision.brightness.threshold", 190.0)
    @property
    def vision_highlight_ratio(self) -> float: return self.get_float("vision.highlight.ratio.threshold", 0.16)
    @property
    def vision_aggressive_gamma(self) -> float: return self.get_float("vision.aggressive.gamma", 1.35)
    @property
    def vision_mild_gamma(self) -> float: return self.get_float("vision.mild.gamma", 1.10)
    @property
    def haar_min_face_size(self) -> int: return self.get_int("haar.min.face.size", 40)
    @property
    def haar_default_confidence(self) -> float: return self.get_float("haar.default.confidence", 0.75)
    @property
    def haar_scale_factor(self) -> float: return self.get_float("haar.scale.factor", 1.1)
    @property
    def haar_min_neighbors(self) -> int: return self.get_int("haar.min.neighbors", 7)
    @property
    def min_aspect_ratio(self) -> float: return self.get_float("detection.min.aspect.ratio", 0.5)
    @property
    def max_aspect_ratio(self) -> float: return self.get_float("detection.max.aspect.ratio", 1.6)
    @property
    def detection_nms_iou(self) -> float: return self.get_float("detection.nms.iou", 0.45)
    @property
    def detection_nms_containment(self) -> float:
        return self.get_float("detection.nms.containment", 0.85)
    @property
    def face_input_size(self) -> int: return self.get_int("detection.face.input.size", 300)
    @property
    def gender_input_size(self) -> int: return self.get_int("detection.gender.input.size", 227)
