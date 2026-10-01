# MEMORY — Face-Recognition-Python (port từ Java)

## 2026-09-29 — Port full Java -> Python
- Đã làm: scaffold `C:\Users\steal\Face-Recognition-Python`, port 28 files Java (~5.9k lines) sang Python: config/model_paths/path_validator/models, detectors (yolo/ssd/haar), services (pipeline/tracker/face_service/gender+cache/camera/prefs/update_security), processing (frames/vision_fps), UI PySide6, CLI.
- Verify: `pytest 6 passed`; YOLO/SSD/Haar load model thật từ `C:\Users\steal\Face-Recognition\data`, detect ảnh blank = 0 faces; gender_available=True; pipeline end-to-end OK; CLI `input.jpg output.jpg` OK.
- Quyết định quan trọng: opencv-python phải `<5` (5.x bỏ readNetFromCaffe + CascadeClassifier) → pin `opencv-python>=4.9,<5`, hiện dùng 4.12.0.88 + numpy 2.5.3 (bất chấp pip resolver warning, runtime OK). Models dùng chung với bản Java, không copy 66MB.
- Đang dở: chưa test camera thật, chưa test ảnh có mặt người thật, chưa chạy GUI (`python -m face_recognition`).
- Việc tiếp theo: test với ảnh mặt thật so FPS/label với bản Java; `python -m face_recognition` kiểm tra UI; bổ sung pytest cho update/security/prefs.

## 2026-10-01 — Fix bug + build one-file exe + Inno Setup
- Đã làm: fix `ui/main_window.py` import `.` → `..` (GUI crash); fix YOLO landmarks dùng `ax/ay` theo Java; fix pipeline fallback SSD→YOLO per-source khi không prefer-YOLO; thêm `pythonpath=["src"]` cho pytest; dọn import thừa (cli/frames/update_security).
- Build: thêm frozen-paths (`sys._MEIPASS`/exe dir) vào `model_paths.py` + `config.py`; viết `face-recognition.spec` one-file bundle 69MB models + config; `pyinstaller` ra `dist/FaceRecognition.exe` 152MB, mở GUI sống >8s OK; viết `installer.iss` (Inno Setup 6, admin, shortcut).
- Verify: `pytest 6 passed`; CLI blank.jpg → 0 faces OK; `pyi-archive_viewer` đủ 6 model files + config + cv2/onnx/PySide6/numpy.
- Quyết định: theo anh chọn one-file + models nén trong + Inno Setup (khởi động chậm hơn one-folder, chấp nhận).

## 2026-10-01 — Fix exe crash ImportError
- Lỗi: `FaceRecognition.exe` báo `ImportError: attempted relative import` vì `__main__.py` dùng `from .ui...` — PyInstaller chạy entry như top-level script.
- Fix: chuyển sang absolute `from face_recognition.ui.main_window import run_app`; rebuild cả 2 exe, bản debug hết traceback, GUI sống >25s, chỉ còn log camera probe vô hại của OpenCV.
