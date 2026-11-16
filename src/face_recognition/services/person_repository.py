"""SQLite persistence for enrolled people and their face embeddings."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
EMBEDDING_DIM = 128

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    code TEXT COLLATE NOCASE UNIQUE,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    model TEXT NOT NULL,
    alignment TEXT NOT NULL,
    image_path TEXT NOT NULL DEFAULT '',
    quality REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON face_embeddings(person_id);
"""


class DuplicatePersonCodeError(ValueError):
    """Raised when an enrollment code is already used by another person."""


@dataclass(frozen=True)
class PersonRecord:
    id: int
    name: str
    code: str | None
    note: str
    created_at: str
    image_count: int = 0


@dataclass(frozen=True)
class EmbeddingInput:
    vector: np.ndarray
    model: str
    alignment: str = "square"
    image_path: str = ""
    quality: float = 0.0


@dataclass(frozen=True)
class EmbeddingRecord:
    id: int
    person_id: int
    vector: np.ndarray
    model: str
    alignment: str
    image_path: str
    quality: float
    created_at: str


@dataclass(frozen=True)
class PersonEmbedding:
    person_id: int
    name: str
    code: str | None
    note: str
    vector: np.ndarray
    model: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size != EMBEDDING_DIM:
        raise ValueError(f"expected {EMBEDDING_DIM}-d embedding, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("embedding has zero norm")
    return arr / norm


class PersonRepository:
    def __init__(self, db_path: str | Path, people_dir: str | Path):
        self.db_path = Path(db_path)
        self.people_dir = Path(people_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.people_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        try:
            yield con
        finally:
            con.close()

    def create_person(self, name: str, code: str | None = None, note: str = "",
                      embeddings: list[EmbeddingInput] | None = None) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name is required")
        clean_code = (code or "").strip() or None
        created = _now()
        prepared = [(self._blob(e.vector), e.model, e.alignment, e.image_path, float(e.quality))
                    for e in (embeddings or [])]
        try:
            with self._connect() as con:
                cur = con.execute(
                    "INSERT INTO persons(name, code, note, created_at) VALUES (?, ?, ?, ?)",
                    (clean_name, clean_code, note.strip(), created),
                )
                person_id = int(cur.lastrowid)
                for vector, model, alignment, image_path, quality in prepared:
                    con.execute(
                        "INSERT INTO face_embeddings(person_id, vector, model, alignment, "
                        "image_path, quality, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (person_id, vector, model, alignment, image_path, quality, created),
                    )
                con.commit()
                return person_id
        except sqlite3.IntegrityError as e:
            if "code" in str(e).lower() or "unique" in str(e).lower():
                raise DuplicatePersonCodeError(f"code already exists: {clean_code}") from e
            raise

    def add_embedding(self, person_id: int, embedding: EmbeddingInput) -> int:
        with self._connect() as con:
            if con.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
                raise KeyError(f"person not found: {person_id}")
            cur = con.execute(
                "INSERT INTO face_embeddings(person_id, vector, model, alignment, "
                "image_path, quality, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (person_id, self._blob(embedding.vector), embedding.model,
                 embedding.alignment, embedding.image_path, float(embedding.quality), _now()),
            )
            con.commit()
            return int(cur.lastrowid)

    def update_person(self, person_id: int, name: str, code: str | None, note: str) -> bool:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name is required")
        clean_code = (code or "").strip() or None
        try:
            with self._connect() as con:
                cur = con.execute(
                    "UPDATE persons SET name = ?, code = ?, note = ? WHERE id = ?",
                    (clean_name, clean_code, note.strip(), person_id),
                )
                con.commit()
                return cur.rowcount > 0
        except sqlite3.IntegrityError as e:
            if "code" in str(e).lower() or "unique" in str(e).lower():
                raise DuplicatePersonCodeError(f"code already exists: {clean_code}") from e
            raise

    def get_person(self, person_id: int) -> PersonRecord | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT p.*, (SELECT COUNT(*) FROM face_embeddings e WHERE e.person_id = p.id) "
                "AS image_count FROM persons p WHERE p.id = ?",
                (person_id,),
            ).fetchone()
            return self._person_row(row) if row is not None else None

    def list_people(self) -> list[PersonRecord]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT p.*, (SELECT COUNT(*) FROM face_embeddings e WHERE e.person_id = p.id) "
                "AS image_count FROM persons p ORDER BY p.name COLLATE NOCASE, p.id"
            ).fetchall()
            return [self._person_row(r) for r in rows]

    def get_embeddings(self, person_id: int) -> list[EmbeddingRecord]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM face_embeddings WHERE person_id = ? ORDER BY id",
                (person_id,),
            ).fetchall()
            return [self._embedding_row(r) for r in rows]

    def all_embeddings(self) -> list[PersonEmbedding]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT p.id AS person_id, p.name, p.code, p.note, e.vector, e.model "
                "FROM face_embeddings e JOIN persons p ON p.id = e.person_id ORDER BY e.id"
            ).fetchall()
            return [
                PersonEmbedding(
                    person_id=int(r["person_id"]),
                    name=str(r["name"]),
                    code=r["code"],
                    note=str(r["note"]),
                    vector=self._vector(r["vector"]),
                    model=str(r["model"]),
                )
                for r in rows
            ]

    def delete_person(self, person_id: int) -> bool:
        with self._connect() as con:
            image_paths = [str(r["image_path"]) for r in con.execute(
                "SELECT image_path FROM face_embeddings WHERE person_id = ? AND image_path != ''",
                (person_id,),
            ).fetchall()]
            cur = con.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            con.commit()
            deleted = cur.rowcount > 0
        if deleted:
            for raw in image_paths:
                path = Path(raw)
                if not path.is_absolute():
                    path = self.people_dir / path
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    log.warning("could not delete enrollment image %s: %s", path, e)
            person_dir = self.people_dir / str(person_id)
            try:
                person_dir.rmdir()
            except OSError:
                pass
        return deleted

    @staticmethod
    def _blob(vector: np.ndarray) -> bytes:
        return _normalize_vector(vector).tobytes()

    @staticmethod
    def _vector(blob: bytes) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32).copy()
        if arr.size != EMBEDDING_DIM:
            raise ValueError(f"corrupt embedding blob: {arr.size} values")
        return arr

    @staticmethod
    def _person_row(row: sqlite3.Row) -> PersonRecord:
        return PersonRecord(
            id=int(row["id"]), name=str(row["name"]), code=row["code"],
            note=str(row["note"]), created_at=str(row["created_at"]),
            image_count=int(row["image_count"]),
        )

    def _embedding_row(self, row: sqlite3.Row) -> EmbeddingRecord:
        return EmbeddingRecord(
            id=int(row["id"]), person_id=int(row["person_id"]),
            vector=self._vector(row["vector"]), model=str(row["model"]),
            alignment=str(row["alignment"]), image_path=str(row["image_path"]),
            quality=float(row["quality"]), created_at=str(row["created_at"]),
        )
