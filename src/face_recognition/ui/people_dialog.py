"""People manager: list, edit, add images, and delete enrolled identities."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from ..services.enrollment import EnrollmentService
from ..services.person_repository import DuplicatePersonCodeError, PersonRecord
from .enroll_dialog import EnrollDialog


class PeopleDialog(QDialog):
    def __init__(self, service: EnrollmentService, parent=None):
        super().__init__(parent)
        self.service = service
        self.changed = False
        self.setWindowTitle("Danh sách người đã đăng ký")
        self.resize(760, 520)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Người đã đăng ký"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Tên", "Mã", "Số ảnh", "Ngày tạo"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_actions)
        root.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_add = QPushButton("Thêm ảnh")
        self.btn_edit = QPushButton("Sửa thông tin")
        self.btn_delete = QPushButton("Xóa người")
        self.btn_close = QPushButton("Đóng")
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_edit)
        buttons.addWidget(self.btn_delete)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_close)
        root.addLayout(buttons)

        self.btn_add.clicked.connect(self.add_images)
        self.btn_edit.clicked.connect(self.edit_person)
        self.btn_delete.clicked.connect(self.delete_person)
        self.btn_close.clicked.connect(self.accept)
        self.reload()

    def _selected(self) -> PersonRecord | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        row = rows[0].row()
        item = self.table.item(row, 0)
        if item is None:
            return None
        people = self.service.list_people()
        return people[row] if 0 <= row < len(people) else None

    def _update_actions(self) -> None:
        has = self._selected() is not None
        self.btn_add.setEnabled(has)
        self.btn_edit.setEnabled(has)
        self.btn_delete.setEnabled(has)

    def reload(self) -> None:
        people = self.service.list_people()
        self.table.setRowCount(len(people))
        for i, p in enumerate(people):
            vals = (p.name, p.code or "", str(p.image_count), p.created_at[:10])
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)
        self._update_actions()

    def add_images(self) -> None:
        person = self._selected()
        if person is None:
            return
        if person.image_count >= self.service.cfg.recognition_enrollment_max_images:
            QMessageBox.information(self, "Đã đủ ảnh", "Người này đã có số ảnh tối đa.")
            return
        dialog = EnrollDialog(self.service, self, person_id=person.id)
        if dialog.exec():
            self.changed = True
            self.reload()

    def edit_person(self) -> None:
        person = self._selected()
        if person is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Sửa thông tin")
        form = QFormLayout(dialog)
        name = QLineEdit(person.name)
        code = QLineEdit(person.code or "")
        note = QTextEdit(person.note)
        note.setFixedHeight(80)
        form.addRow("Tên:", name)
        form.addRow("Mã:", code)
        form.addRow("Ghi chú:", note)
        row = QHBoxLayout()
        ok = QPushButton("Lưu")
        cancel = QPushButton("Hủy")
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(ok)
        form.addRow(row)
        ok.clicked.connect(dialog.accept)
        cancel.clicked.connect(dialog.reject)
        if not dialog.exec():
            return
        try:
            updated = self.service.repository.update_person(
                person.id, name.text(), code.text().strip(), note.toPlainText())
            if not updated:
                QMessageBox.warning(self, "Không tìm thấy", "Người này không còn tồn tại.")
            else:
                self.changed = True
                self.reload()
        except DuplicatePersonCodeError:
            QMessageBox.warning(self, "Trùng mã", "Mã này đã được dùng cho người khác.")
        except ValueError as e:
            QMessageBox.warning(self, "Thiếu thông tin", str(e))

    def delete_person(self) -> None:
        person = self._selected()
        if person is None:
            return
        answer = QMessageBox.question(
            self, "Xóa người",
            f"Xóa '{person.name}' cùng toàn bộ ảnh và embedding?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        if self.service.delete_person(person.id):
            self.changed = True
            self.reload()
