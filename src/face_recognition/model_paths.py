"""ModelPaths — port of Java ModelPaths."""
from __future__ import annotations

import sys
from pathlib import Path

FACE_DIR = Path("data") / "models" / "face"
GENDER_DIR = Path("data") / "models" / "gender"
HAAR_DIR = Path("data") / "haarcascade"

YOLO_FILE = "yolov8n-face.onnx"
SSD_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
SSD_CONFIG = "deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_CONFIG = "gender_deploy.prototxt"
HAAR_FILE = "haarcascade_frontalface_default.xml"


class ModelPaths:
    def __init__(self, base: Path):
        self.base = base

    @classmethod
    def resolve(cls) -> "ModelPaths":
        candidates: list[Path] = []
        # PyInstaller one-file: data bundled into _MEIPASS, exe dir for external data
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "data")
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent / "data")
            candidates.append(Path(sys.executable).resolve().parent / "_internal" / "data")
        candidates.append(Path.cwd() / "data")
        # sibling Java checkout shares the same data layout
        candidates.append(Path(r"C:\Users\steal\Face-Recognition") / "data")
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            candidates.append(parent / "data")
            candidates.append(parent / "app" / "data")
        for c in candidates:
            if cls._has_complete(c):
                return cls(c)
        for c in candidates:
            if c.is_dir():
                return cls(c)
        return cls(Path.cwd() / "data")

    @staticmethod
    def _has_complete(d: Path) -> bool:
        files = [
            d / "models" / "face" / YOLO_FILE,
            d / "models" / "face" / SSD_MODEL,
            d / "models" / "face" / SSD_CONFIG,
            d / "models" / "gender" / GENDER_MODEL,
            d / "models" / "gender" / GENDER_CONFIG,
            d / "haarcascade" / HAAR_FILE,
        ]
        return all(f.is_file() for f in files)

    @property
    def yolo_path(self) -> Path: return self.base / "models" / "face" / YOLO_FILE
    @property
    def ssd_model_path(self) -> Path: return self.base / "models" / "face" / SSD_MODEL
    @property
    def ssd_config_path(self) -> Path: return self.base / "models" / "face" / SSD_CONFIG
    @property
    def gender_model_path(self) -> Path: return self.base / "models" / "gender" / GENDER_MODEL
    @property
    def gender_config_path(self) -> Path: return self.base / "models" / "gender" / GENDER_CONFIG
    @property
    def haar_path(self) -> Path: return self.base / "haarcascade" / HAAR_FILE

    def has_yolo(self) -> bool: return self.yolo_path.is_file()
    def has_ssd(self) -> bool: return self.ssd_model_path.is_file() and self.ssd_config_path.is_file()
    def has_gender(self) -> bool: return self.gender_model_path.is_file() and self.gender_config_path.is_file()
    def has_haar(self) -> bool: return self.haar_path.is_file()

    def missing_files(self) -> list[str]:
        out = []
        for p in [self.yolo_path, self.ssd_model_path, self.ssd_config_path,
                  self.gender_model_path, self.gender_config_path, self.haar_path]:
            if not p.is_file():
                out.append(str(p))
        return out
