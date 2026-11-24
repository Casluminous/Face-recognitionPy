"""Enrollment dialog: pick 3-5 images, choose one face per image, save a person."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ..image_io import imread
from ..services.enrollment import EnrollmentItem, EnrollmentService
from ..services.person_repository import DuplicatePersonCodeError


def _thumb(image: np.ndarray, face) -> QPixmap:
    view = image.copy()
    x, y, w, h = face.x, face.y, face.w, face.h
    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
    scale = min(1.0, 160.0 / max(1, view.shape[1]), 160.0 / max(1, view.shape[0]))
    if scale < 1.0:
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                  QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class EnrollDialog(QDialog):
    def __init__(self, service: EnrollmentService, parent=None, person_id: int | None = None):
        super().__init__(parent)
        self.service = service
        self.person_id = person_id
        self.items: list[EnrollmentItem] = []
        self.add_mode = person_id is not None
        self.setWindowTitle("Thêm ảnh cho người đã đăng ký" if self.add_mode else "Đăng ký người mới")
        self.resize(760, 620)

        root = QVBoxLayout(self)
        title = QLabel("Thêm ảnh cho người đã đăng ký" if self.add_mode else "Đăng ký người mới")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title)

        self.form = QFormLayout()
        self.name_edit = QLineEdit()
        self.code_edit = QLineEdit()
        self.note_edit = QTextEdit()
        self.note_edit.setFixedHeight(70)
        self.name_edit.setPlaceholderText("Bắt buộc")
        self.code_edit.setPlaceholderText("Tùy chọn, không trùng")
        self.form.addRow("Tên:", self.name_edit)
        self.form.addRow("Mã:", self.code_edit)
        self.form.addRow("Ghi chú:", self.note_edit)
        root.addLayout(self.form)

        hint = QLabel("Chọn 1-5 ảnh rõ mặt; 1 ảnh vẫn được nhưng nên thêm ảnh để ổn định hơn.")
        hint.setStyleSheet("color: #475569;")
        root.addWidget(hint)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(150, 150))
        self.list_widget.setSpacing(4)
        root.addWidget(self.list_widget, 1)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #475569;")
        root.addWidget(self.status)

        buttons = QHBoxLayout()
        self.btn_choose = QPushButton("Chọn ảnh")
        self.btn_remove = QPushButton("Bỏ ảnh đang chọn")
        self.btn_save = QPushButton("Lưu")
        self.btn_save.setObjectName("primary")
        self.btn_cancel = QPushButton("Hủy")
        buttons.addWidget(self.btn_choose)
        buttons.addWidget(self.btn_remove)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_save)
        root.addLayout(buttons)

        self.btn_choose.clicked.connect(self.choose_images)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_save.clicked.connect(self.save)
        self.btn_cancel.clicked.connect(self.reject)
        self.name_edit.textChanged.connect(self._update_actions)
        self.list_widget.itemSelectionChanged.connect(self._update_actions)

        if self.add_mode:
            person = self.service.repository.get_person(person_id)
            if person is None:
                QMessageBox.critical(self, "Lỗi", "Không tìm thấy người đã chọn.")
                self.reject()
                return
            self.name_edit.setText(person.name)
            self.code_edit.setText(person.code or "")
            self.note_edit.setPlainText(person.note)
            self.name_edit.setReadOnly(True)
            self.code_edit.setReadOnly(True)
            self.note_edit.setReadOnly(True)

        if not self.service.available:
            self.btn_choose.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.status.setText("Thiếu model SFace hoặc detector; không thể đăng ký.")
        self._update_actions()

    def _limit(self) -> int:
        if not self.add_mode:
            return self.service.cfg.recognition_enrollment_max_images
        person = self.service.repository.get_person(self.person_id)
        used = person.image_count if person is not None else 0
        return max(0, self.service.cfg.recognition_enrollment_max_images - used)

    def _update_actions(self) -> None:
        limit = self._limit()
        can_add = self.service.available and len(self.items) < limit
        self.btn_choose.setEnabled(can_add)
        self.btn_remove.setEnabled(bool(self.list_widget.selectedItems()))
        minimum = 1 if self.add_mode else self.service.cfg.recognition_enrollment_min_images
        self.btn_save.setEnabled(self.service.available and len(self.items) >= minimum
                                 and bool(self.name_edit.text().strip()))
        self.status.setText(f"Đã chọn {len(self.items)}/{limit} ảnh hợp lệ.")

    def choose_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Chọn ảnh", "", "Ảnh (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not paths:
            return
        skipped = []
        self.setCursor(Qt.WaitCursor)
        try:
            for raw in paths:
                if len(self.items) >= self._limit():
                    break
                path = Path(raw)
                image = imread(path)
                if image is None:
                    skipped.append(f"{path.name}: không đọc được")
                    continue
                faces = self.service.detect_faces(image)
                if not faces:
                    skipped.append(f"{path.name}: không có khuôn mặt")
                    continue
                face = faces[0]
                if len(faces) > 1:
                    labels = [f"Mặt {i + 1} ({f.x},{f.y}) {f.w}x{f.h}"
                              for i, f in enumerate(faces)]
                    choice, ok = QInputDialog.getItem(
                        self, "Chọn khuôn mặt", f"{path.name}: chọn khuôn mặt cần đăng ký",
                        labels, 0, False)
                    if not ok:
                        continue
                    face = faces[labels.index(choice)]
                item = self.service.make_item(image, face, str(path))
                if item is None:
                    skipped.append(f"{path.name}: ảnh mờ/tối/quá nhỏ")
                    continue
                self.items.append(item)
                entry = QListWidgetItem(QIcon(_thumb(image, face)), path.name)
                entry.setToolTip(str(path))
                self.list_widget.addItem(entry)
        finally:
            self.unsetCursor()
            self._update_actions()
        if skipped:
            QMessageBox.information(self, "Một số ảnh bị bỏ qua", "\n".join(skipped[:12]))

    def remove_selected(self) -> None:
        rows = sorted({self.list_widget.row(i) for i in self.list_widget.selectedItems()},
                      reverse=True)
        for row in rows:
            if 0 <= row < len(self.items):
                self.items.pop(row)
                self.list_widget.takeItem(row)
        self._update_actions()

    def save(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Thiếu thông tin", "Tên không được để trống.")
            return
        self.setCursor(Qt.WaitCursor)
        try:
            if self.add_mode:
                count = self.service.add_images(self.person_id, self.items)
                QMessageBox.information(self, "Đã lưu", f"Đã thêm {count} ảnh.")
            else:
                self.service.save(name, self.code_edit.text().strip(), self.note_edit.toPlainText(),
                                  self.items)
                QMessageBox.information(self, "Đã lưu", "Đã đăng ký người mới.")
            self.accept()
        except DuplicatePersonCodeError:
            QMessageBox.warning(self, "Trùng mã", "Mã này đã được dùng cho người khác.")
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Không lưu được", str(e))
        except Exception as e:  # noqa: BLE001 - keep the dialog alive on backend errors
            QMessageBox.critical(self, "Lỗi", f"Không lưu được dữ liệu: {e}")
        finally:
            self.unsetCursor()
