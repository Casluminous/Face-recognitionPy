"""Enrollment workflow: validate images, build embeddings, persist people."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from ..config import AppConfig
from ..detectors.base import FaceDetector
from ..models import FaceDetection
from .embedding import EmbeddingResult, FaceEmbeddingService
from .person_repository import EmbeddingInput, PersonRecord, PersonRepository

log = logging.getLogger(__name__)


@dataclass
class EnrollmentItem:
    image: np.ndarray
    face: FaceDetection
    embedding: EmbeddingResult
    source_path: str = ""


class EnrollmentService:
    def __init__(self, repository: PersonRepository, embedder: FaceEmbeddingService,
                 detector: FaceDetector | None, cfg: AppConfig | None = None):
        self.repository = repository
        self.embedder = embedder
        self.detector = detector
        self.cfg = cfg or AppConfig.get_instance()

    @property
    def available(self) -> bool:
        return self.embedder.available and self.detector is not None and self.detector.is_available()

    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]:
        if self.detector is None or not self.detector.is_available():
            return []
        return self.detector.detect(image)

    def make_item(self, image: np.ndarray, face: FaceDetection,
                  source_path: str = "") -> EnrollmentItem | None:
        result = self.embedder.embed(image, face.bbox, face.landmarks)
        if result is None:
            return None
        return EnrollmentItem(image=image.copy(), face=face, embedding=result,
                              source_path=source_path)

    def save(self, name: str, code: str | None, note: str,
             items: list[EnrollmentItem]) -> int:
        if not name.strip():
            raise ValueError("Tên không được để trống")
        if len(items) < self.cfg.recognition_enrollment_min_images:
            raise ValueError(f"Cần ít nhất {self.cfg.recognition_enrollment_min_images} ảnh hợp lệ")
        if len(items) > self.cfg.recognition_enrollment_max_images:
            raise ValueError(f"Chỉ được tối đa {self.cfg.recognition_enrollment_max_images} ảnh")
        if len(items) == 1:
            item = items[0]
            aug = getattr(self.embedder, "embed_augmented", None)
            if aug is not None:
                result = aug(item.image, item.face.bbox, item.face.landmarks)
                if result is not None:
                    item.embedding = result
        person_id = self.repository.create_person(name, code, note)
        created: list[Path] = []
        try:
            inputs: list[EmbeddingInput] = []
            for item in items:
                rel, path = self._write_image(person_id, item.image)
                created.append(path)
                inputs.append(EmbeddingInput(
                    vector=item.embedding.vector,
                    model=self.embedder.model_name,
                    alignment=item.embedding.alignment,
                    image_path=rel,
                    quality=item.embedding.quality,
                ))
            self.repository.add_embeddings(person_id, inputs)
            return person_id
        except Exception:
            self.repository.delete_person(person_id)
            for path in created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def add_images(self, person_id: int, items: list[EnrollmentItem]) -> int:
        person = self.repository.get_person(person_id)
        if person is None:
            raise KeyError(f"person not found: {person_id}")
        remaining = self.cfg.recognition_enrollment_max_images - person.image_count
        if len(items) > remaining:
            raise ValueError(f"Chỉ còn thêm được {max(0, remaining)} ảnh")
        created: list[Path] = []
        try:
            inputs: list[EmbeddingInput] = []
            for item in items:
                rel, path = self._write_image(person_id, item.image)
                created.append(path)
                inputs.append(EmbeddingInput(
                    vector=item.embedding.vector,
                    model=self.embedder.model_name,
                    alignment=item.embedding.alignment,
                    image_path=rel,
                    quality=item.embedding.quality,
                ))
            self.repository.add_embeddings(person_id, inputs)
            return len(inputs)
        except Exception:
            for path in created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def delete_person(self, person_id: int) -> bool:
        return self.repository.delete_person(person_id)

    def list_people(self) -> list[PersonRecord]:
        return self.repository.list_people()

    def load_image(self, image_path: str) -> np.ndarray | None:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.repository.people_dir / path
        return cv2.imread(str(path))

    def _write_image(self, person_id: int, image: np.ndarray) -> tuple[str, Path]:
        person_dir = self.repository.people_dir / str(person_id)
        person_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid4().hex}.jpg"
        path = person_dir / filename
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise OSError("không mã hóa được ảnh JPEG")
        path.write_bytes(buf.tobytes())
        return f"{person_id}/{filename}", path
