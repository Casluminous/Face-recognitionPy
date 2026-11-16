"""Enrollment service tests with stubbed detector/embedder (no model files)."""
from __future__ import annotations

import numpy as np

from face_recognition.config import AppConfig
from face_recognition.models import FaceDetection
from face_recognition.services.embedding import EmbeddingResult
from face_recognition.services.enrollment import EnrollmentItem, EnrollmentService
from face_recognition.services.person_repository import PersonRepository


class _Detector:
    def name(self):
        return "stub"

    def is_available(self):
        return True

    def confidence_threshold(self):
        return 0.5

    def detect(self, frame):
        return [FaceDetection(10, 10, 60, 60, [], 0.9, "stub")]


class _Embedder:
    available = True
    model_name = "stub"

    def embed(self, image, rect, landmarks=None):
        vec = np.zeros(128, dtype=np.float32)
        vec[0] = 1.0
        return EmbeddingResult(vec, "square", 5.0)


def _item():
    return EnrollmentItem(
        image=np.full((100, 100, 3), 128, np.uint8),
        face=FaceDetection(10, 10, 60, 60, [], 0.9, "stub"),
        embedding=EmbeddingResult(_unit(), "square", 5.0),
    )


def _unit():
    vec = np.zeros(128, dtype=np.float32)
    vec[0] = 1.0
    return vec


def test_enrollment_saves_images_and_embeddings(tmp_path):
    repo = PersonRepository(tmp_path / "people.sqlite3", tmp_path / "people")
    cfg = AppConfig.reset_for_tests({
        "recognition.enrollment.min.images": "3",
        "recognition.enrollment.max.images": "5",
    })
    svc = EnrollmentService(repo, _Embedder(), _Detector(), cfg)
    pid = svc.save("An", "A01", "note", [_item(), _item(), _item()])
    person = repo.get_person(pid)
    assert person is not None and person.image_count == 3
    embs = repo.get_embeddings(pid)
    assert len(embs) == 3
    for emb in embs:
        path = repo.people_dir / emb.image_path
        assert path.is_file()
    assert svc.add_images(pid, [_item()]) == 1
    assert repo.get_person(pid).image_count == 4
    assert svc.delete_person(pid) is True
    assert repo.get_person(pid) is None


def test_enrollment_rejects_too_few_images(tmp_path):
    repo = PersonRepository(tmp_path / "people.sqlite3", tmp_path / "people")
    cfg = AppConfig.reset_for_tests({"recognition.enrollment.min.images": "3"})
    svc = EnrollmentService(repo, _Embedder(), _Detector(), cfg)
    try:
        svc.save("An", None, "", [_item()])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted fewer than minimum images")
