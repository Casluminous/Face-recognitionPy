"""Unicode-safe image read/write helpers for Windows paths."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path) -> np.ndarray | None:
    """Read an image without cv2.imread's ANSI-path limitation."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def imwrite(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> bool:
    """Write an image to a Unicode path, inferring the format from its suffix."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        suffix = ".jpg"
    ok, buf = cv2.imencode(suffix, image, params or [])
    if not ok:
        return False
    try:
        target.write_bytes(buf.tobytes())
        return True
    except OSError:
        return False
