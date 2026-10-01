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

## 2026-10-01 — Push lên GitHub
- Tạo repo private `Casluminous/Face-recognitionPy`, commit root `feat: port Java face recognition...` (34 files), push branch master. `.gitignore` loại build/dist/exe (exe 152MB vượt giới hạn 100MB của GitHub).

## 2026-10-01 — White theme + bảng catalog trên cùng
- Đã làm: `frames.py` thêm `FaceRow` + `FrameResult.faces` để fill bảng; viết lại `ui/main_window.py` — QSS trắng (#F8FAFC/#1E293B, nút chính xanh #16A34A), `QTableWidget` catalog 5 cột (ID/Gender/Confidence/Engine/Position) nằm trên cùng update live, status dồn vào `QStatusBar`.
- Verify: `pytest 6 passed`; lint chỉ còn warning style cũ; render offscreen + screenshot đúng layout (chữ hiện ô vuông do offscreen thiếu font, máy thật OK).

## 2026-10-01 — Nút Snapshot
- Đã làm: `ui/main_window.py` thêm nút Snapshot (disable đến khi có kết quả đầu tiên), lưu ảnh processed qua save-dialog, tên mặc định `snapshot_YYYYMMDD_HHMMSS.jpg`, nhớ thư mục qua prefs `last_snapshot_dir`, lỗi báo statusbar.
- Verify: `pytest 6 passed`; lint không thêm warning mới; test offscreen bấm snapshot ghi file JPG thật OK.

## 2026-10-01 — Nút Export CSV
- Đã làm: `ui/main_window.py` thêm nút Export CSV + helper `write_faces_csv()` (header timestamp,track_id,gender,confidence,engine,x,y,w,h), nhớ thư mục qua prefs `last_csv_dir`, catalog rỗng thì báo statusbar thay vì tạo file rỗng; thêm `tests/test_export.py` (2 test).
- Verify: `pytest 8 passed`; lint không thêm warning mới; test offscreen export CSV thật đúng nội dung.
