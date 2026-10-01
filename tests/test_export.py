import csv

from face_recognition.processing.frames import FaceRow
from face_recognition.ui.main_window import CSV_HEADER, write_faces_csv


def _rows():
    return [
        FaceRow(1, 100, 120, 80, 80, "Male", "82%", 0.91),
        FaceRow(2, 300, 150, 70, 70, "", "", 0.75),
    ]


def test_write_faces_csv_content(tmp_path):
    p = tmp_path / "faces.csv"
    n = write_faces_csv(p, _rows(), "YOLOv8", "2026-10-01T08:00:00+07:00")
    assert n == 2
    with open(p, newline="", encoding="utf-8") as f:
        data = list(csv.reader(f))
    assert data[0] == list(CSV_HEADER)
    assert data[1][1:] == ["1", "Male 82%", "0.9100", "YOLOv8", "100", "120", "80", "80"]
    assert data[2][2] == ""  # no gender yet


def test_write_faces_csv_empty(tmp_path):
    p = tmp_path / "empty.csv"
    assert write_faces_csv(p, [], "Idle", "2026-10-01T08:00:00+07:00") == 0
    with open(p, newline="", encoding="utf-8") as f:
        assert list(csv.reader(f)) == [list(CSV_HEADER)]
