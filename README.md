# Face Recognition — Python port

Port 100% từ `Face-Recognition` Java (JavaFX + OpenCV + ONNX Runtime) sang Python.

Bản Java: **1.2.4** — YOLOv8-face + SSD ResNet-10 + Haar fallback, gender classification,
tracking, adaptive exposure, updater SHA-256/Authenticode.

## Cấu trúc

```text
src/face_recognition/
  config.py            # AppConfig (application.properties)
  model_paths.py       # ModelPaths
  path_validator.py
  models.py            # FaceDetection / TrackedFace / DetectionResult
  detectors/           # yolo.py, ssd_haar.py, base.py
  services/            # pipeline.py, tracker.py, face_service.py, gender.py,
                       # camera.py, prefs.py, update_security.py,
                       # embedding.py, enrollment.py, person_repository.py, recognition.py
  processing/          # frames.py (FrameProcessor), vision_fps.py
  ui/main_window.py    # PySide6 (thay JavaFX)
  ui/enroll_dialog.py  # đăng ký người mới
  ui/people_dialog.py  # quản lý người đã đăng ký
  cli.py               # thay FaceRecognitionCli
config/application.properties  # copy từ bản Java
```

Models dùng chung với bản Java (`C:\Users\steal\Face-Recognition\data` tự nhận qua `ModelPaths.resolve()`).

## Chạy

```bash
pip install -e ".[dev]"
python -m face_recognition            # app desktop PySide6
face-recognition-cli input.jpg output.jpg
pytest -q
```

Theo lựa chọn của anh: full port logic, UI PySide6, chạy bằng pip (không đóng exe installer).
`UpdateService`/`SecurityService` vẫn giữ verify SHA-256 + Authenticode fail-closed.

## Nhận diện khuôn mặt 1:N (YOLO + SFace)

Tính năng enroll không train lại YOLO. YOLO phát hiện khuôn mặt, SFace tạo embedding để so khớp.

1. Tải model SFace 2021dec từ OpenCV Zoo:

```text
https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Đặt tại:

```text
data/models/recognition/face_recognition_sface_2021dec.onnx
```

2. Mở app, bấm `Enroll Person`, chọn 3-5 ảnh, nhập Tên/Mã/Ghi chú và lưu.
3. Bật `Face recognition` để nhận diện trên ảnh tĩnh và webcam.
4. Dữ liệu lưu cục bộ:

```text
data/recognition/people.sqlite3
data/recognition/people/<person_id>/*.jpg
```

Nếu model thiếu, app vẫn chạy detect/gender bình thường và nút recognition bị tắt.
