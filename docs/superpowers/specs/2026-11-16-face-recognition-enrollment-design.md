# Thiết kế: Nhận diện khuôn mặt 1:N bằng YOLOv8-face + SFace

- Ngày: 2026-11-16
- Trạng thái: Approved
- Phạm vi: thêm tính năng enrollment + nhận diện danh tính vào ứng dụng desktop hiện có
- Quyết định model: YOLOv8-face + SFace, không train lại YOLO

## 1. Bối cảnh

Ứng dụng hiện có:

- Phát hiện khuôn mặt bằng YOLOv8-face, SSD ResNet-10, Haar Cascade.
- Theo dõi khuôn mặt bằng IoU, containment, khoảng cách tâm và template matching.
- Phân loại giới tính bằng CaffeNet.
- Giao diện PySide6, xử lý ảnh tĩnh và webcam.

Ứng dụng chưa có khả năng nhận diện danh tính. YOLOv8-face hiện tại trả về hộp khuôn mặt; một số biến thể có thêm 5 landmark, nhưng model đang dùng chỉ trả bbox. Vì vậy tính năng mới dùng YOLO để phát hiện, tùy chọn dùng landmark nếu detector cung cấp, và dùng SFace để tạo embedding so khớp.

## 2. Mục tiêu

- Cho phép người dùng chọn 3–5 ảnh của một người, nhập Tên, Mã, Ghi chú và lưu hồ sơ.
- Tự động tạo embedding cho từng khuôn mặt đã chọn.
- Nhận diện 1:N trên ảnh tĩnh và webcam: hiển thị đúng tên nếu đã đăng ký, ngược lại hiển thị `Unknown`.
- Dữ liệu lưu cục bộ, không gửi ảnh/embedding ra ngoài.
- Không train lại YOLO hoặc SFace; thêm người mới chỉ cần enrollment.

## 3. Không thuộc phạm vi

- Train YOLO classifier cho từng người.
- Nhận diện 1:1 / xác thực danh tính.
- Chống giả mạo khuôn mặt (anti-spoofing).
- Đồng bộ cloud, nhiều máy, API mạng.
- Nhận diện cảm xúc, tuổi, dân tộc.
- Tự động tải model từ Internet khi chạy app.

## 4. Quyết định đã chốt

| Hạng mục | Quyết định |
|---|---|
| Kiểu nhận diện | 1:N, nhiều người |
| Số ảnh mỗi người | 3–5 ảnh |
| Nguồn nhận diện | Ảnh tĩnh + webcam |
| Thông tin người | Tên bắt buộc; Mã và Ghi chú tùy chọn |
| Khi không đủ giống | Hiển thị `Unknown`, không đoán người gần nhất |
| Detector cho recognition | YOLOv8-face nếu có; fallback SSD + crop vuông |
| Embedding | SFace 2021dec, 128-d |
| Lưu trữ | SQLite + ảnh gốc cục bộ |
| Train lại model | Không |

## 5. Kiến trúc

```text
[EnrollDialog]
   chọn 3-5 ảnh, nhập Tên/Mã/Ghi chú
        │
        ▼
[EnrollmentService]
   validate ảnh → YOLO detect + landmark → chọn mặt
   → SFace align + embedding → chuẩn hóa L2
        │
        ▼
[PersonRepository]
   SQLite: persons + face_embeddings
   ảnh gốc: data/recognition/people/<person_id>/

[Ảnh tĩnh / Webcam]
        │
        ▼
[FrameProcessor hiện có]
   detect → tracker → track đã xác nhận
        │
        ▼
[RecognitionService]
   lấy landmark YOLO nếu có → align → SFace
   → cosine với DB → gộp top-2 + margin + vote
        │
        ▼
[MainWindow]
   vẽ tên/Unknown + thêm cột Person vào catalog
```

### Thành phần mới

| Thành phần | Trách nhiệm |
|---|---|
| `FaceEmbeddingService` | Load SFace ONNX, align 112×112, tạo vector 128-d, chuẩn hóa L2 |
| `EnrollmentService` | Validate ảnh, phát hiện mặt, chọn mặt, tạo embeddings, lưu hồ sơ |
| `RecognitionService` | So vector với DB, gộp điểm, margin, vote, cache theo track |
| `PersonRepository` | CRUD SQLite cho persons/embeddings, quản lý ảnh, transaction |
| `EnrollDialog` | UI chọn ảnh, preview, chọn mặt, nhập thông tin |
| `PeopleDialog` | Danh sách người, sửa thông tin, thêm/xóa ảnh, xóa người |

## 6. Model và alignment

### Model

- Detector: `data/models/face/yolov8n-face.onnx` (đã có).
- Embedding: `data/models/recognition/face_recognition_sface_2021dec.onnx`.
- Nguồn model: OpenCV Zoo, model SFace 2021dec.
- Load bằng `cv2.FaceRecognizerSF.create(model_path, "")`.
- Nếu model thiếu: recognition bị disable, UI hiển thị cảnh báo rõ; enrollment bị chặn.

### Alignment

- Nếu detector trả về 5 landmark (YOLO keypoint model hoặc detector khác), tạo face row 15 phần tử:
  `[x, y, w, h, lx1, ly1, ..., lx5, ly5]`
- Gọi `FaceRecognizerSF.alignCrop(image, face_row)` để có ảnh 112×112.
- Model YOLOv8-face hiện tại chỉ trả bbox, nên mặc định:
  - Crop vuông quanh bbox, padding 30%.
  - Resize về 112×112.
  - Vẫn tạo embedding nhưng đánh dấu `alignment=fallback`.
- Có thể bổ sung YuNet/keypoint model ở phase sau để cải thiện alignment mà không đổi kiến trúc.
- Không dùng ảnh CLAHE/gamma cho embedding; dùng ảnh BGR gốc để tránh lệch màu.

### Chất lượng ảnh enroll

Ảnh được chấp nhận khi:

- Có đúng một khuôn mặt được chọn.
- Khuôn mặt tối thiểu 80 px cạnh ngắn.
- Độ sắc nét Laplacian ≥ 3.0.
- Độ sáng trung bình 40–225.
- Ảnh đọc được, không hỏng.

Ảnh không đạt bị đánh dấu lỗi trong dialog; người dùng có thể bỏ hoặc thay ảnh.

## 7. Database

### Vị trí

- DB: `data/recognition/people.sqlite3`
- Ảnh: `data/recognition/people/<person_id>/<uuid>.jpg`

### Schema

```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    code TEXT UNIQUE COLLATE NOCASE,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    model TEXT NOT NULL,
    alignment TEXT NOT NULL,
    image_path TEXT NOT NULL,
    quality REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_embeddings_person ON face_embeddings(person_id);
```

- `code` để `NULL` nếu người dùng bỏ trống; SQLite cho phép nhiều NULL.
- Vector lưu `float32`, 128 phần tử, đã chuẩn hóa L2.
- `model` lưu `sface_2021dec` để tránh so vector khác model.
- Bật `PRAGMA foreign_keys = ON` và `journal_mode = WAL`.
- Xóa người: xóa DB row (cascade) và xóa thư mục ảnh; nếu xóa file lỗi, ghi log và vẫn hoàn tất DB.
- Lưu người + ảnh + embeddings trong một transaction; nếu lỗi, rollback và dọn ảnh đã copy.

## 8. Matching

- Similarity: cosine = dot product của hai vector đã chuẩn hóa.
- Với mỗi người, lấy trung bình top-2 similarity trên các ảnh đã enroll.
- Chọn người có điểm cao nhất.
- Chấp nhận danh tính khi:
  - `top1_score >= recognition.match.threshold` (mặc định 0.36).
  - `top1_score - top2_score >= recognition.match.margin` (mặc định 0.05).
- Nếu không đạt: `Unknown`.
- Static image: dùng kết quả top-1 ngay, không vote.
- Camera:
  - Chỉ chạy khi track đã confirm.
  - Chạy mỗi 15 frame hoặc khi track mới.
  - Lưu tối đa 4 kết quả gần nhất/track.
  - Hiển thị tên khi có ít nhất 3 kết quả cùng một người.
  - Nếu 3 kết quả `Unknown`, hiển thị `Unknown`.
  - Xóa cache khi track biến mất.
- Không chạy recognition trên track chưa confirm.

## 9. UI

### Main window

- Thêm nút `Enroll Person`.
- Thêm nút `People`.
- Thêm checkbox `Face recognition` (mặc định bật nếu model có sẵn).
- Catalog thêm cột `Person`; giữ cột `Gender`.
- Trên hộp khuôn mặt:
  - Dòng 1: tên hoặc `Unknown`.
  - Dòng 2: giới tính + độ tin cậy (nếu bật gender).
- Nếu recognition tắt: chỉ vẽ hộp và gender như hiện tại.

### Enroll dialog

- Chọn 3–5 ảnh.
- Mỗi ảnh có thumbnail và trạng thái: hợp lệ / không có mặt / nhiều mặt / ảnh mờ.
- Nếu ảnh có nhiều mặt: hiện danh sách crop để chọn đúng người.
- Nhập:
  - Tên (bắt buộc).
  - Mã (tùy chọn, unique không phân biệt hoa thường).
  - Ghi chú (tùy chọn).
- Nút Lưu chỉ bật khi có ít nhất 3 ảnh hợp lệ và Tên không rỗng.
- Khi lưu: hiện tiến trình, khóa nút, báo lỗi nếu trùng mã/model lỗi/DB lỗi.

### People dialog

- Bảng: Tên, Mã, Số ảnh, Ngày tạo.
- Sửa Tên/Mã/Ghi chú.
- Thêm ảnh mới (tối đa 5).
- Xóa ảnh.
- Xóa người có xác nhận; xóa ảnh + embeddings.

## 10. Cấu hình

Thêm vào `config/application.properties` và `AppConfig`:

```properties
recognition.enabled=true
recognition.detector=yolo
recognition.embedding.model=data/models/recognition/face_recognition_sface_2021dec.onnx
recognition.match.threshold=0.36
recognition.match.margin=0.05
recognition.min.votes=3
recognition.vote.window=4
recognition.update.interval.frames=15
recognition.min.face.px=80
recognition.min.sharpness=3.0
recognition.min.brightness=40.0
recognition.max.brightness=225.0
recognition.enrollment.min.images=3
recognition.enrollment.max.images=5
recognition.db.path=data/recognition/people.sqlite3
recognition.people.dir=data/recognition/people
```

- `recognition.detector=yolo`: dùng YOLO cho enrollment/recognition nếu có; nếu không, fallback SSD + crop vuông.
- `recognition.enabled=false`: tắt toàn bộ recognition, không ảnh hưởng detect/gender.
- Các ngưỡng có thể chỉnh mà không sửa code.

## 11. Xử lý lỗi và edge case

- Ảnh không đọc được → báo lỗi, không thêm vào enrollment.
- Không có mặt → đánh dấu ảnh lỗi.
- Nhiều mặt → bắt buộc chọn một mặt.
- Mặt quá nhỏ/mờ/tối/cháy sáng → từ chối ảnh.
- Trùng mã → báo lỗi, không lưu.
- Tên rỗng → không cho lưu.
- Model SFace thiếu → disable recognition, hiện hướng dẫn đặt model.
- DB lỗi/khóa → báo lỗi, rollback; không làm hỏng dữ liệu cũ.
- Track mất → xóa cache recognition.
- Camera không có → không ảnh hưởng recognition trên ảnh tĩnh.
- Người lạ → `Unknown`, không gán tên.
- Xóa người khi đang nhận diện → cache được xóa ở lần track kế tiếp; không crash.

## 12. Quyền riêng tư

- Toàn bộ xử lý và lưu trữ cục bộ.
- Không gửi ảnh/embedding qua mạng.
- Ảnh enroll chỉ lưu khi người dùng bấm Lưu.
- Xóa người xóa cả ảnh gốc và vector.
- Không log ảnh/vector; log chỉ metadata lỗi.
- Cần sự đồng ý của người được enroll khi dùng ảnh thật.

## 13. Kiểm thử và tiêu chí chấp nhận

### Unit test

- Chuẩn hóa L2 và cosine similarity.
- Gộp top-2, margin, threshold.
- Vote window: 3/4 cùng người → hiện tên; 3/4 Unknown → Unknown.
- SQLite CRUD, unique code, cascade delete, transaction rollback.
- Validate ảnh: không mặt, nhiều mặt, mờ, tối, cháy sáng.
- Fallback alignment khi không có landmark.

### Integration/UI test

- Enroll 2 người × 3 ảnh.
- Nhận đúng trên 2 ảnh test/người.
- 3 ảnh người lạ → `Unknown`.
- Camera 10 giây: tên không nhấp nháy, không đổi liên tục.
- Tắt/mở app → dữ liệu còn.
- Xóa người → không còn nhận.
- Tắt checkbox recognition → không chạy recognition.
- Thiếu model → cảnh báo, app vẫn chạy detect/gender.

### Tiêu chí chấp nhận

- Enroll và nhận diện được ít nhất 2 người trên ảnh tĩnh.
- Camera nhận đúng người đã enroll trong điều kiện ánh sáng bình thường.
- Người chưa enroll không bị gán tên sai trong bộ test người lạ.
- FPS camera trung bình ≥ 15 trên máy thử nghiệm.
- UI không treo khi enroll hoặc khi recognition chạy.
- Dữ liệu tồn tại sau khi khởi động lại.

## 14. Hiệu năng

- SFace chỉ chạy trên track đã confirm, mỗi 15 frame.
- Static image: một lần/ảnh.
- Nếu recognition pass > 30 ms/face trên máy thử nghiệm, chuyển sang worker thread/queue ở Phase 3.
- Không chạy recognition trên frame bị drop.
- Cache embedding/identity theo track; không tính lại nếu không cần.

## 15. Phase triển khai

1. **Core**
   - Thêm model SFace, `FaceEmbeddingService`, `PersonRepository`.
   - Schema DB, config, unit test.
2. **Enrollment + People UI**
   - `EnrollDialog`, `PeopleDialog`.
   - Validate ảnh, chọn mặt, lưu/xóa.
3. **Recognition integration**
   - `RecognitionService`, tích hợp FrameProcessor.
   - Catalog cột Person, vẽ tên/Unknown.
   - Cache/vote theo track.
4. **Tuning + nghiệm thu**
   - Test ảnh/camera, chỉnh threshold.
   - README, hướng dẫn tải model, test suite.

## 16. Dataset cần chuẩn bị

Khi vào Phase 3/nghiệm thu, cần:

- 2–3 người.
- Mỗi người 3–5 ảnh enroll: chính diện, hơi trái/phải, ánh sáng khác nhau.
- Mỗi người 2–3 ảnh test không trùng ảnh enroll.
- 3–5 ảnh người lạ để test `Unknown`.
- Ảnh rõ mặt, không che khuất mạnh; có sự đồng ý của người trong ảnh.

Nếu chưa có, có thể dùng ảnh công khai/ảnh mẫu cho test kỹ thuật, nhưng không dùng làm dữ liệu thật.

## 17. Rủi ro và giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| SFace kém chính xác với góc mặt lạ | Yêu cầu 3–5 ảnh đa góc; threshold + margin; fallback Unknown |
| YOLO không có landmark (model hiện tại) | Fallback crop vuông; có thể bổ sung YuNet/keypoint model sau |
| Nhận nhầm người lạ | Margin top1-top2, vote, threshold; test người lạ |
| UI treo khi embedding | Chỉ chạy mỗi 15 frame; measure; worker nếu >30 ms |
| Mất dữ liệu khi xóa/sửa | Transaction + rollback + xóa file có kiểm soát |
| Model/license | Dùng OpenCV Zoo SFace; ghi nguồn trong README |
| Trùng mã | UNIQUE COLLATE NOCASE |
| Ảnh enroll chất lượng kém | Quality gate + preview lỗi từng ảnh |

## 18. Ngoài phạm vi / tương lai

- ArcFace 512-d nếu cần độ chính xác cao hơn.
- Anti-spoofing.
- Nhận diện 1:1.
- Đồng bộ nhiều máy.
- Train YOLO classifier với dataset lớn.
