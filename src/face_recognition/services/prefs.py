"""PreferencesService — JSON file backing (port of Java Preferences node)."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class PreferencesService:
    def __init__(self, path: Path | None = None):
        self.path = path or (Path.home() / ".face_recognition_prefs.json")
        self._lock = threading.Lock()
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.is_file():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_string(self, key: str, default: str = "") -> str:
        with self._lock:
            v = self._data.get(key, default)
        return str(v)

    def set_string(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = str(value)
            self._save()

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get_string(key, str(default)))
        except ValueError:
            return default

    def set_int(self, key: str, value: int) -> None: self.set_string(key, str(value))

    def get_bool(self, key: str, default: bool = False) -> bool:
        return self.get_string(key, str(default)).lower() in ("true", "1", "yes")

    def set_bool(self, key: str, value: bool) -> None: self.set_string(key, "true" if value else "false")

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get_string(key, str(default)))
        except ValueError:
            return default

    def set_float(self, key: str, value: float) -> None: self.set_string(key, str(value))

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._save()
