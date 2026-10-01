"""PathValidator — port of Java PathValidator."""
from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "pbm", "pgm", "ppm", "tiff"}
MODEL_EXTS = {"onnx", "caffemodel", "prototxt", "xml"}


def is_valid_path(p: str | None) -> bool:
    if not p or "\x00" in p:
        return False
    parts = re.split(r"[\\/]", p)
    return all(seg not in ("..", "...") for seg in parts)


def _has_allowed_ext(p: str, allowed: set[str]) -> bool:
    suffix = Path(p).suffix.lower().lstrip(".")
    return suffix in allowed


def is_valid_image_path(p: str | None) -> bool:
    return bool(p) and is_valid_path(p) and _has_allowed_ext(p, IMAGE_EXTS)


def is_valid_model_path(p: str | Path | None) -> bool:
    if p is None:
        return False
    return is_valid_path(str(p)) and _has_allowed_ext(str(p), MODEL_EXTS)


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\.{2,}", ".", name).strip()
    return name[:255] or "file"


def resolve_safe_path(base: Path, user: str) -> Path:
    target = (base / user).resolve()
    base_r = base.resolve()
    if base_r not in target.parents and target != base_r:
        raise ValueError("Path escapes base directory")
    return target
