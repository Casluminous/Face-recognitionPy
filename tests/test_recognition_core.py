"""Core tests for SFace matching, voting, and the enrollment repository.

These tests use synthetic vectors and do not require the 38 MB SFace model.
The real-model smoke test is separate so CI can run on a clean checkout.
"""
from __future__ import annotations

import numpy as np

from face_recognition.config import AppConfig
from face_recognition.services.embedding import EmbeddingResult, FaceEmbeddingService
from face_recognition.services.person_repository import (
    DuplicatePersonCodeError,
    EmbeddingInput,
    PersonRepository,
)
from face_recognition.services.recognition import (
    IdentityVoteCache,
    RecognitionService,
    cosine_similarity,
    match_embeddings,
    normalize_vector,
)


def _unit(index: int, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    return vec


def test_normalize_and_cosine():
    a = normalize_vector(np.array([2.0, 0.0], dtype=np.float32))
    b = normalize_vector(np.array([0.0, 3.0], dtype=np.float32))
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert cosine_similarity(a, a) == 1.0
    assert cosine_similarity(a, b) == 0.0


def test_match_embeddings_top2_and_margin():
    rows = [
        _person_embedding(1, "An", _unit(0)),
        _person_embedding(1, "An", _unit(0) * 0.9 + _unit(1) * 0.1),
        _person_embedding(2, "Binh", _unit(1)),
    ]
    match = match_embeddings(rows, _unit(0), threshold=0.36, margin=0.05)
    assert match is not None
    assert match.person_id == 1
    assert match.name == "An"


def test_match_embeddings_rejects_ambiguous_or_low_score():
    rows = [
        _person_embedding(1, "An", _unit(0)),
        _person_embedding(2, "Binh", _unit(0)),
    ]
    assert match_embeddings(rows, _unit(0), 0.36, 0.05) is None
    assert match_embeddings(rows, _unit(2), 0.36, 0.05) is None


def test_identity_vote_cache():
    cache = IdentityVoteCache(vote_window=4, min_votes=3)
    match = match_embeddings(
        [_person_embedding(1, "An", _unit(0))], _unit(0), 0.36, 0.05)
    assert match is not None
    cache.put(7, match)
    cache.put(7, match)
    assert cache.get(7).state == "pending"
    cache.put(7, match)
    decision = cache.get(7)
    assert decision.state == "known"
    assert decision.match is not None and decision.match.name == "An"
    cache.put(8, None)
    cache.put(8, None)
    cache.put(8, None)
    assert cache.get(8).state == "unknown"
    cache.retain_all({7})
    assert cache.get(8).state == "pending"


def test_repository_crud_and_cascade(tmp_path):
    repo = PersonRepository(tmp_path / "people.sqlite3", tmp_path / "people")
    p1 = repo.create_person("An", "A01", "lớp 1", [
        EmbeddingInput(_unit(0), "sface_2021dec", "landmarks", quality=5.0),
        EmbeddingInput(_unit(1), "sface_2021dec", "square", quality=6.0),
    ])
    p2 = repo.create_person("Bình", None, "")
    assert p1 > 0 and p2 > 0
    assert repo.get_person(p1).image_count == 2
    assert [p.name for p in repo.list_people()] == ["An", "Bình"]
    assert len(repo.get_embeddings(p1)) == 2
    assert len(repo.all_embeddings()) == 2

    repo.update_person(p2, "Bình", "B02", "ghi chú")
    assert repo.get_person(p2).code == "B02"

    assert repo.delete_person(p1) is True
    assert repo.get_person(p1) is None
    assert len(repo.get_embeddings(p1)) == 0
    assert len(repo.list_people()) == 1

    repo2 = PersonRepository(tmp_path / "people.sqlite3", tmp_path / "people")
    assert repo2.get_person(p2).name == "Bình"


def test_repository_rejects_duplicate_code_case_insensitive(tmp_path):
    repo = PersonRepository(tmp_path / "people.sqlite3", tmp_path / "people")
    repo.create_person("An", "A01")
    try:
        repo.create_person("An 2", "a01")
    except DuplicatePersonCodeError:
        pass
    else:
        raise AssertionError("duplicate code was accepted")


def test_embedding_service_missing_model_is_safe(tmp_path):
    svc = FaceEmbeddingService(tmp_path / "missing.onnx", AppConfig.reset_for_tests({}))
    assert svc.available is False
    img = np.zeros((100, 100, 3), np.uint8)
    assert svc.embed(img, (10, 10, 50, 50)) is None


class _StubEmbedder:
    model_name = "stub"

    def __init__(self, vector: np.ndarray):
        self.vector = normalize_vector(vector)
        self.available = True

    def embed(self, image, rect, landmarks=None):
        return EmbeddingResult(self.vector, "square", 5.0)


def test_recognition_service_known_and_unknown(tmp_path):
    repo = PersonRepository(tmp_path / "people.sqlite3", tmp_path / "people")
    repo.create_person("An", "A01", embeddings=[
        EmbeddingInput(_unit(0), "stub", "square"),
    ])
    cfg = AppConfig.reset_for_tests({
        "recognition.match.threshold": "0.50",
        "recognition.match.margin": "0.05",
        "recognition.min.votes": "3",
        "recognition.vote.window": "4",
    })
    svc = RecognitionService(_StubEmbedder(_unit(0)), repo, cfg)
    img = np.zeros((100, 100, 3), np.uint8)
    match = svc.recognize(img, (10, 10, 50, 50))
    assert match is not None and match.name == "An"
    assert svc.recognize(img, (10, 10, 50, 50)) is not None

    unknown = RecognitionService(_StubEmbedder(_unit(2)), repo, cfg)
    assert unknown.recognize(img, (10, 10, 50, 50)) is None
    for _ in range(3):
        unknown.observe(1, img, (10, 10, 50, 50))
    assert unknown.cache.get(1).state == "unknown"


def test_recognition_config_defaults():
    cfg = AppConfig.reset_for_tests({})
    assert cfg.recognition_enabled is True
    assert cfg.recognition_match_threshold == 0.36
    assert cfg.recognition_match_margin == 0.05
    assert cfg.recognition_min_face_px == 80
    assert cfg.recognition_enrollment_min_images == 3


def _person_embedding(person_id: int, name: str, vector: np.ndarray):
    from face_recognition.services.person_repository import PersonEmbedding
    return PersonEmbedding(person_id, name, None, "", vector, "test")
