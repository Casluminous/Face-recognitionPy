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
                       # camera.py, prefs.py, update_security.py
  processing/          # frames.py (FrameProcessor), vision_fps.py
  ui/main_window.py    # PySide6 (thay JavaFX)
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
