"""1:N face matching and per-track identity voting."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from .embedding import FaceEmbeddingService
from .person_repository import PersonEmbedding, PersonRepository


@dataclass(frozen=True)
class PersonMatch:
    person_id: int
    name: str
    code: str | None
    note: str
    score: float
    margin: float


@dataclass(frozen=True)
class IdentityDecision:
    state: str  # "known", "unknown", "pending"
    match: PersonMatch | None = None


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("invalid embedding")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("zero embedding")
    return arr / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize_vector(a), normalize_vector(b)))


def match_embeddings(rows: list[PersonEmbedding], query: np.ndarray,
                     threshold: float, margin: float) -> PersonMatch | None:
    if not rows:
        return None
    query = normalize_vector(query)
    by_person: dict[int, list[float]] = defaultdict(list)
    info: dict[int, PersonEmbedding] = {}
    for row in rows:
        if row.vector.size != query.size:
            continue
        sim = float(np.dot(query, normalize_vector(row.vector)))
        by_person[row.person_id].append(sim)
        info[row.person_id] = row

    scored: list[tuple[float, int]] = []
    for person_id, sims in by_person.items():
        sims.sort(reverse=True)
        top = sims[:2]
        scored.append((float(sum(top) / len(top)), person_id))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    second = scored[1][0] if len(scored) > 1 else None
    gap = best_score - second if second is not None else 1.0
    if best_score < threshold or (second is not None and gap < margin):
        return None
    row = info[best_id]
    return PersonMatch(row.person_id, row.name, row.code, row.note,
                       best_score, gap)


class IdentityVoteCache:
    def __init__(self, vote_window: int = 4, min_votes: int = 3):
        self.vote_window = max(1, int(vote_window))
        self.min_votes = max(1, min(int(min_votes), self.vote_window))
        self._history: dict[int, deque[tuple[str, PersonMatch | None]]] = {}

    def put(self, track_id: int, match: PersonMatch | None) -> None:
        label = f"id:{match.person_id}" if match is not None else "unknown"
        history = self._history.setdefault(track_id, deque(maxlen=self.vote_window))
        history.append((label, match))

    def get(self, track_id: int) -> IdentityDecision:
        history = self._history.get(track_id)
        if not history:
            return IdentityDecision("pending")
        counts: dict[str, int] = defaultdict(int)
        latest: dict[str, PersonMatch | None] = {}
        for label, match in history:
            counts[label] += 1
            latest[label] = match
        label, count = max(counts.items(), key=lambda kv: kv[1])
        if count < self.min_votes:
            return IdentityDecision("pending")
        if label == "unknown":
            return IdentityDecision("unknown")
        return IdentityDecision("known", latest[label])

    def retain_all(self, active_track_ids: set[int]) -> None:
        for tid in list(self._history):
            if tid not in active_track_ids:
                del self._history[tid]

    def clear(self) -> None:
        self._history.clear()


class RecognitionService:
    def __init__(self, embedder: FaceEmbeddingService, repository: PersonRepository,
                 cfg: AppConfig | None = None):
        self.embedder = embedder
        self.repository = repository
        self.cfg = cfg or AppConfig.get_instance()
        self.cache = IdentityVoteCache(self.cfg.recognition_vote_window,
                                       self.cfg.recognition_min_votes)
        self._rows: list[PersonEmbedding] | None = None

    def refresh(self) -> None:
        self._rows = self.repository.all_embeddings()

    def _all_rows(self) -> list[PersonEmbedding]:
        if self._rows is None:
            self.refresh()
        return self._rows or []

    def invalidate(self) -> None:
        self._rows = None
        self.cache.clear()

    def recognize(self, image: np.ndarray, rect: tuple[int, int, int, int],
                  landmarks: list[tuple[float, float]] | None = None) -> PersonMatch | None:
        result = self.embedder.embed(image, rect, landmarks)
        if result is None:
            return None
        return match_embeddings(self._all_rows(), result.vector,
                                self.cfg.recognition_match_threshold,
                                self.cfg.recognition_match_margin)

    def observe(self, track_id: int, image: np.ndarray,
                rect: tuple[int, int, int, int],
                landmarks: list[tuple[float, float]] | None = None) -> IdentityDecision:
        self.cache.put(track_id, self.recognize(image, rect, landmarks))
        return self.cache.get(track_id)

    def retain_tracks(self, active_track_ids: set[int]) -> None:
        self.cache.retain_all(active_track_ids)
