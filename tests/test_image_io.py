"""Unicode path tests for the Windows image I/O helpers."""
from __future__ import annotations

import numpy as np

from face_recognition.image_io import imread, imwrite


def test_imread_imwrite_unicode_path(tmp_path):
    folder = tmp_path / "Thông tin cá nhân _ VAU Portal"
    folder.mkdir()
    path = folder / "imgi_3_default.png"
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    assert imwrite(path, image) is True
    loaded = imread(path)
    assert loaded is not None
    assert loaded.shape == image.shape
    assert int(loaded[0, 0, 1]) == 200


def test_imread_missing_returns_none(tmp_path):
    assert imread(tmp_path / "không tồn tại.png") is None
