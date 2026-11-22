"""SFace embedding service for 1:N face recognition.

The detector supplies a box (and optionally five YOLO landmarks). SFace turns
the aligned 112x112 crop into a normalized 128-d vector used by the matching
layer. Detection/gender models are deliberately untouched.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import AppConfig
from .face_service import crop_quality_ok, square_crop

log = logging.getLogger(__name__)

EMBEDDING_SIZE = 112
EMBEDDING_DIM = 128
MODEL_NAME = "sface_2021dec"


@dataclass(frozen=True)
class EmbeddingResult:
    vector: np.ndarray
    alignment: str
    quality: float


class FaceEmbeddingService:
    def __init__(self, model_path: str | Path, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig.get_instance()
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self._recognizer = None
        self._min_face_px = max(1, self.cfg.recognition_min_face_px)
        self._min_sharpness = self.cfg.recognition_min_sharpness
        self._min_brightness = self.cfg.recognition_min_brightness
        self._max_brightness = self.cfg.recognition_max_brightness
        if not self.model_path.is_file():
            log.error("SFace model not found: %s", self.model_path)
            return
        try:
            self._recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")
        except Exception as e:  # noqa: BLE001 - OpenCV/ONNX backends raise varied errors
            log.error("Failed to load SFace model: %s", e)
            self._recognizer = None

    @property
    def available(self) -> bool:
        return self._recognizer is not None

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    def embed(self, image: np.ndarray, rect: tuple[int, int, int, int],
              landmarks: list[tuple[float, float]] | None = None) -> EmbeddingResult | None:
        """Return a normalized embedding, or None when the crop is unusable."""
        prepared = self._prepare_face(image, rect, landmarks)
        if prepared is None:
            return None
        face, alignment = prepared
        vector = self._feature(face)
        if vector is None:
            return None
        return EmbeddingResult(vector=vector, alignment=alignment, quality=self._quality(face))

    def embed_augmented(self, image: np.ndarray, rect: tuple[int, int, int, int],
                        landmarks: list[tuple[float, float]] | None = None) -> EmbeddingResult | None:
        """Average embeddings from small one-image augmentations.

        Used for 1-image enrollment. Augmenting the aligned 112x112 crop avoids
        having to recompute face boxes/landmarks for flips and rotations.
        """
        prepared = self._prepare_face(image, rect, landmarks)
        if prepared is None:
            return None
        face, alignment = prepared
        variants = [
            face,
            cv2.flip(face, 1),
            self._gamma(face, 0.85),
            self._gamma(face, 1.15),
            self._rotate(face, 8),
            self._rotate(face, -8),
        ]
        vectors = [v for v in (self._feature(item) for item in variants) if v is not None]
        if not vectors:
            return None
        mean = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-12:
            return None
        return EmbeddingResult(vector=mean / norm, alignment=f"{alignment}+aug",
                               quality=self._quality(face))

    def _prepare_face(self, image: np.ndarray, rect: tuple[int, int, int, int],
                      landmarks: list[tuple[float, float]] | None = None
                      ) -> tuple[np.ndarray, str] | None:
        if not self.available or image is None or image.size == 0:
            return None
        x, y, w, h = (int(v) for v in rect)
        if w <= 0 or h <= 0 or min(w, h) < self._min_face_px:
            return None
        H, W = image.shape[:2]
        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            return None

        face = None
        alignment = "square"
        if landmarks and len(landmarks) >= 5:
            face = self._aligned_face(image, rect, landmarks)
            if face is not None:
                alignment = "landmarks"
        if face is None:
            face = square_crop(image, rect, 0.30)
            if face is None:
                return None
            interp = cv2.INTER_AREA if face.shape[0] > EMBEDDING_SIZE else cv2.INTER_LINEAR
            face = cv2.resize(face, (EMBEDDING_SIZE, EMBEDDING_SIZE), interpolation=interp)
        if face is None or face.size == 0:
            return None
        if not crop_quality_ok(face, EMBEDDING_SIZE, self._min_sharpness,
                               self._min_brightness, self._max_brightness):
            log.debug("recognition crop rejected (blur/brightness)")
            return None
        return face, alignment

    def _feature(self, face: np.ndarray) -> np.ndarray | None:
        try:
            with self._lock:
                feat = self._recognizer.feature(face)
        except Exception as e:  # noqa: BLE001 - OpenCV backend errors are not typed
            log.debug("SFace feature failed: %s", e)
            return None
        vector = np.asarray(feat, dtype=np.float32).reshape(-1)
        if vector.size != EMBEDDING_DIM or not np.all(np.isfinite(vector)):
            log.warning("unexpected SFace embedding size: %s", vector.size)
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return None
        return vector / norm

    @staticmethod
    def _quality(face: np.ndarray) -> float:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _gamma(face: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(1e-6, gamma)
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(face, table)

    @staticmethod
    def _rotate(face: np.ndarray, angle: float) -> np.ndarray:
        center = (face.shape[1] / 2.0, face.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(face, matrix, (face.shape[1], face.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _aligned_face(self, image: np.ndarray, rect: tuple[int, int, int, int],
                      landmarks: list[tuple[float, float]]) -> np.ndarray | None:
        try:
            x, y, w, h = rect
            points = []
            for px, py in landmarks[:5]:
                if not (math.isfinite(float(px)) and math.isfinite(float(py))):
                    return None
                points.extend((float(px), float(py)))
            face_row = np.array([x, y, w, h, *points], dtype=np.float32)
            with self._lock:
                aligned = self._recognizer.alignCrop(image, face_row)
            if aligned is None or aligned.size == 0:
                return None
            if aligned.shape[0] != EMBEDDING_SIZE or aligned.shape[1] != EMBEDDING_SIZE:
                aligned = cv2.resize(aligned, (EMBEDDING_SIZE, EMBEDDING_SIZE))
            return aligned
        except Exception as e:  # noqa: BLE001 - OpenCV alignment errors are not typed
            log.debug("SFace alignment failed: %s", e)
            return None

    def close(self) -> None:
        self._recognizer = None
