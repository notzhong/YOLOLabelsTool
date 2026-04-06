"""
验证窗口
"""

import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QSize
from PySide6.QtGui import QImage, QPixmap, QCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QGroupBox, QSlider, QDoubleSpinBox,
    QMessageBox, QRubberBand
)

from src.utils.i18n import tr

try:
    import dxcam
    DXCAM_AVAILABLE = True
except Exception:
    DXCAM_AVAILABLE = False


_user32 = ctypes.windll.user32


class RegionSelector(QDialog):
    """全选"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.CrossCursor)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._origin = QPoint()
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._selected_rect: Optional[QRect] = None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._origin = event.pos()
        self._rubber_band.setGeometry(QRect(self._origin, QSize()))
        self._rubber_band.show()

    def mouseMoveEvent(self, event):
        if not self._rubber_band.isVisible():
            return
        rect = QRect(self._origin, event.pos()).normalized()
        self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._selected_rect = self._rubber_band.geometry()
        self._rubber_band.hide()
        self.accept()

    @staticmethod
    def get_region(parent=None) -> Optional[QRect]:
        selector = RegionSelector(parent)
        if selector.exec() == QDialog.Accepted:
            rect = selector._selected_rect
            if rect and rect.width() > 0 and rect.height() > 0:
                return rect
        return None


class ValidationDialog(QDialog):
    """验证窗口"""

    def __init__(self, parent, model_manager):
        super().__init__(parent)
        self.setWindowTitle(tr("validation_window"))
        self.resize(900, 600)

        self.model_manager = model_manager
        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.is_running = False

        self.current_hwnd: Optional[int] = None
        self.current_rect: Optional[QRect] = None
        self.capture_region: Optional[Tuple[int, int, int, int]] = None

        self._init_ui()
        self._update_model_status()

        self.pick_timer = QTimer(self)
        self.pick_timer.timeout.connect(self._update_window_pick)
        self.picking_window = False

    def _init_ui(self):
        layout = QHBoxLayout(self)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # 模
        model_group = QGroupBox(tr("model"))
        model_layout = QVBoxLayout(model_group)
        self.model_status_label = QLabel("")
        self.btn_load_model = QPushButton(tr("load_model"))
        self.btn_load_model.clicked.connect(self._load_model)
        model_layout.addWidget(self.model_status_label)
        model_layout.addWidget(self.btn_load_model)
        left_panel.addWidget(model_group)

        # 源选
        source_group = QGroupBox(tr("validation_source"))
        source_layout = QVBoxLayout(source_group)

        self.source_combo = QComboBox()
        self.source_combo.addItems([
            tr("source_window"),
            tr("source_region"),
            tr("source_image")
        ])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)

        # 选
        self.window_info_label = QLabel(tr("window_not_selected"))
        self.btn_pick_window = QPushButton(tr("pick_window"))
        self.btn_pick_window.clicked.connect(self._toggle_pick_window)
        source_layout.addWidget(self.window_info_label)
        source_layout.addWidget(self.btn_pick_window)

        # 选
        self.region_info_label = QLabel(tr("region_not_selected"))
        self.btn_pick_region = QPushButton(tr("pick_region"))
        self.btn_pick_region.clicked.connect(self._pick_region)
        source_layout.addWidget(self.region_info_label)
        source_layout.addWidget(self.btn_pick_region)

        # 图片选
        image_row = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.btn_browse_image = QPushButton(tr("browse"))
        self.btn_browse_image.clicked.connect(self._browse_image)
        image_row.addWidget(self.image_path_edit, 1)
        image_row.addWidget(self.btn_browse_image)
        source_layout.addLayout(image_row)

        left_panel.addWidget(source_group)

        # 
        params_group = QGroupBox(tr("model_params"))
        params_layout = QVBoxLayout(params_group)

        conf_row = QHBoxLayout()
        conf_label = QLabel(tr("confidence"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(int(self.model_manager.confidence_threshold * 100))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.00)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(self.model_manager.confidence_threshold)
        self.conf_spin.setDecimals(2)
        self.conf_slider.valueChanged.connect(self._on_conf_slider)
        self.conf_spin.valueChanged.connect(self._on_conf_spin)
        conf_row.addWidget(conf_label)
        conf_row.addWidget(self.conf_slider, 1)
        conf_row.addWidget(self.conf_spin)
        params_layout.addLayout(conf_row)

        iou_row = QHBoxLayout()
        iou_label = QLabel(tr("iou_threshold"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(int(self.model_manager.iou_threshold * 100))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.00)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(self.model_manager.iou_threshold)
        self.iou_spin.setDecimals(2)
        self.iou_slider.valueChanged.connect(self._on_iou_slider)
        self.iou_spin.valueChanged.connect(self._on_iou_spin)
        iou_row.addWidget(iou_label)
        iou_row.addWidget(self.iou_slider, 1)
        iou_row.addWidget(self.iou_spin)
        params_layout.addLayout(iou_row)

        left_panel.addWidget(params_group)

        # 始/停止
        self.btn_toggle = QPushButton(tr("start_detect"))
        self.btn_toggle.clicked.connect(self._toggle_detect)
        left_panel.addWidget(self.btn_toggle)

        self.status_label = QLabel(tr("ready"))
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()

        # 预
        self.preview_label = QLabel(tr("no_image_loaded"))
        self.preview_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.preview_label, 1)

        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)

        self._on_source_changed()

    def _update_model_status(self):
        if self.model_manager.is_model_loaded():
            info = self.model_manager.get_model_info()
            name = Path(info.get("path", "")).name
            self.model_status_label.setText(
                tr("model_loaded_status").replace("{model_name}", name)
            )
        else:
            self.model_status_label.setText(tr("model_not_loaded"))

    def _load_model(self):
        if not self.model_manager.is_available():
            QMessageBox.warning(self, tr("warning"), tr("ultralytics_not_installed"))
            return
        model_path, _ = QFileDialog.getOpenFileName(
            self, tr("select_pretrained_model_file"), str(Path.cwd()), tr("model_file_filter")
        )
        if model_path:
            success = self.model_manager.load_model(model_path)
            if success:
                self._update_model_status()
            else:
                QMessageBox.warning(
                    self,
                    tr("warning"),
                    tr("load_model_failed").replace("{model_path}", model_path)
                )

    def _on_source_changed(self):
        source = self.source_combo.currentText()
        is_window = source == tr("source_window")
        is_region = source == tr("source_region")
        is_image = source == tr("source_image")

        self.window_info_label.setVisible(is_window)
        self.btn_pick_window.setVisible(is_window)

        self.region_info_label.setVisible(is_region)
        self.btn_pick_region.setVisible(is_region)

        self.image_path_edit.setVisible(is_image)
        self.btn_browse_image.setVisible(is_image)

    def _toggle_pick_window(self):
        self.picking_window = not self.picking_window
        if self.picking_window:
            self.btn_pick_window.setText(tr("picking_window"))
            self.pick_timer.start(100)
            self.setCursor(Qt.CrossCursor)
        else:
            self.btn_pick_window.setText(tr("pick_window"))
            self.pick_timer.stop()
            self.unsetCursor()

    def _update_window_pick(self):
        pos = QCursor.pos()
        point = wintypes.POINT(pos.x(), pos.y())
        hwnd = _user32.WindowFromPoint(point)
        if hwnd:
            title = self._get_window_title(hwnd)
            self.current_hwnd = hwnd
            self.window_info_label.setText(
                tr("window_selected").replace("{title}", title)
            )

    def _get_window_title(self, hwnd: int) -> str:
        length = _user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        return title or f"HWND:{hwnd}"

    def _pick_region(self):
        rect = RegionSelector.get_region(self)
        if rect:
            self.current_rect = rect
            self.region_info_label.setText(
                tr("region_selected").replace(
                    "{rect}",
                    f"{rect.x()},{rect.y()} {rect.width()}x{rect.height()}"
                )
            )

    def _browse_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("select_image_file_dialog_title"),
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif)"
        )
        if image_path:
            self.image_path_edit.setText(image_path)

    def _toggle_detect(self):
        if self.is_running:
            self._stop()
            return
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return

        source = self.source_combo.currentText()
        if source == tr("source_image"):
            self._run_image_once()
            return

        if not DXCAM_AVAILABLE:
            QMessageBox.warning(self, tr("warning"), tr("dxcam_not_installed"))
            return

        if source == tr("source_window") and not self.current_hwnd:
            QMessageBox.warning(self, tr("warning"), tr("window_not_selected"))
            return
        if source == tr("source_region") and not self.current_rect:
            QMessageBox.warning(self, tr("warning"), tr("region_not_selected"))
            return

        self._start()

    def _start(self):
        self.is_running = True
        self.btn_toggle.setText(tr("stop_detect"))
        self.timer.start(120)
        self.status_label.setText(tr("detecting_status"))

        if self.camera is None and DXCAM_AVAILABLE:
            self.camera = dxcam.create()

    def _stop(self):
        self.is_running = False
        self.btn_toggle.setText(tr("start_detect"))
        self.timer.stop()
        self.status_label.setText(tr("ready"))

    def _on_tick(self):
        frame = self._capture_frame()
        if frame is None:
            return
        self._run_detection(frame)

    def _capture_frame(self) -> Optional[np.ndarray]:
        source = self.source_combo.currentText()
        if source == tr("source_window"):
            if not self.current_hwnd:
                return None
            rect = wintypes.RECT()
            _user32.GetWindowRect(self.current_hwnd, ctypes.byref(rect))
            self.capture_region = (rect.left, rect.top, rect.right, rect.bottom)
        elif source == tr("source_region"):
            if not self.current_rect:
                return None
            r = self.current_rect
            self.capture_region = (r.x(), r.y(), r.x() + r.width(), r.y() + r.height())
        else:
            return None

        if not self.camera:
            return None
        frame = self.camera.grab(region=self.capture_region)
        if frame is None:
            return None

        # 转为 BGR
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _run_image_once(self):
        image_path = self.image_path_edit.text().strip()
        if not image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.warning(self, tr("warning"), tr("cannot_load_image") + image_path)
            return
        self._run_detection(img)

    def _run_detection(self, frame: np.ndarray):
        detections = self.model_manager.predict_image(frame)
        vis = frame.copy()
        for det in detections:
            x = int(det["x"])
            y = int(det["y"])
            w = int(det["width"])
            h = int(det["height"])
            conf = det.get("confidence", 0)
            cls_id = det.get("class_id", 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{cls_id}:{conf:.2f}",
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        self._update_preview(vis)

    def _update_preview(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        target_size = self.preview_label.size()
        if target_size.width() > 0 and target_size.height() > 0:
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)

    def _on_conf_slider(self, value: int):
        conf_value = value / 100.0
        self.conf_spin.blockSignals(True)
        self.conf_spin.setValue(conf_value)
        self.conf_spin.blockSignals(False)
        self.model_manager.set_confidence_threshold(conf_value)

    def _on_conf_spin(self, value: float):
        slider_value = int(value * 100)
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(slider_value)
        self.conf_slider.blockSignals(False)
        self.model_manager.set_confidence_threshold(value)

    def _on_iou_slider(self, value: int):
        iou_value = value / 100.0
        self.iou_spin.blockSignals(True)
        self.iou_spin.setValue(iou_value)
        self.iou_spin.blockSignals(False)
        self.model_manager.set_iou_threshold(iou_value)

    def _on_iou_spin(self, value: float):
        slider_value = int(value * 100)
        self.iou_slider.blockSignals(True)
        self.iou_slider.setValue(slider_value)
        self.iou_slider.blockSignals(False)
        self.model_manager.set_iou_threshold(value)

    def closeEvent(self, event):
        self._stop()
        if self.pick_timer.isActive():
            self.pick_timer.stop()
        self.picking_window = False
        self.unsetCursor()
        super().closeEvent(event)
