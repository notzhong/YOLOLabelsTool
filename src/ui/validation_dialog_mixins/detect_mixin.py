"""
实时检测循环 Mixin

从 validation_dialog.py 拆分（P2 重构）：相机创建/释放、定时抓帧、
推理执行与预览渲染（含 dxcam 可用性守卫）。
通过 self 访问 ValidationDialog 的属性与其他 Mixin 方法。
"""

from ctypes import wintypes
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from src.utils.unicode_text import _draw_unicode_texts_batch
from src.utils.i18n import tr
from src.utils.logger import get_logger_simple
from src.utils.win32_helpers import get_user32

logger = get_logger_simple(__name__)

try:
    import dxcam

    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
except AttributeError:
    # dxcam 仅支持 Windows：在导入链深处直接调用 Windows 专属 API，
    # 非 Windows 平台以 AttributeError 形式失败（而非 ImportError）
    DXCAM_AVAILABLE = False


class DetectMixin:
    """实时检测 Mixin：相机生命周期、抓帧、推理与预览渲染"""


    def _toggle_detect(self):
        if self.is_running:
            self._stop()
            return

        if not self.model_manager.is_model_loaded():
            logger.warning("开始检测被拒: 模型未加载")
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return

        source = self._selected_source()

        if source == self.SOURCE_IMAGE:
            self._run_image_once()
            return

        if not DXCAM_AVAILABLE:
            logger.warning("dxcam 不可用，无法开始实时检测")
            QMessageBox.warning(self, tr("warning"), tr("dxcam_not_installed"))
            return

        if source == self.SOURCE_WINDOW and not self.current_hwnd:
            logger.warning("开始检测被拒: 未选择窗口")
            QMessageBox.warning(self, tr("warning"), tr("window_not_selected"))
            return

        if source == self.SOURCE_REGION and not self.current_rect:
            logger.warning("开始检测被拒: 未选择区域")
            QMessageBox.warning(self, tr("warning"), tr("region_not_selected"))
            return

        if self.picking_window:
            self._stop_window_pick(confirmed=False)

        self._start()

    def _create_camera(self):
        # output_color is available in newer dxcam; keep fallback for compatibility.
        try:
            return dxcam.create(output_color="BGR")
        except TypeError:
            return dxcam.create()

    def _release_camera(self):
        """Release dxcam camera resources."""
        if self.camera is not None:
            try:
                # Try to call stop() if available
                if hasattr(self.camera, 'stop'):
                    self.camera.stop()
                # Try to call release() if available
                if hasattr(self.camera, 'release'):
                    self.camera.release()
            except Exception:
                pass
            self.camera = None

    def _start(self):
        logger.info(f"实时检测开始, 来源={self._selected_source()}")
        self._release_camera()

        if DXCAM_AVAILABLE:
            self.camera = self._create_camera()

        self.is_running = True
        self.btn_toggle.setText(tr("stop_detect"))
        self.timer.start(120)
        self.status_label.setText(tr("detecting_status"))

    def _stop(self):
        logger.info("实时检测停止")
        self.is_running = False
        self.btn_toggle.setText(tr("start_detect"))
        self.timer.stop()
        self.status_label.setText(tr("ready"))
        self._release_camera()

    def _on_tick(self):
        try:
            frame = self._capture_frame()
            if frame is None:
                return
            self._run_detection(frame)
        except Exception as e:
            # Log error but don't stop the timer
            logger.exception(f"Error in detection tick: {e}")
            # Optionally update status label
            self.status_label.setText(tr("detecting_status") + f" (Error: {e})")

    def _capture_frame(self) -> Optional[np.ndarray]:
        source = self._selected_source()
        if source == self.SOURCE_WINDOW:
            if not self.current_hwnd:
                # No window selected
                return None
            # Check if window is still valid
            if not get_user32().IsWindow(wintypes.HWND(self.current_hwnd)):
                # Window no longer exists
                self.current_hwnd = None
                self.window_info_label.setText(tr("window_not_selected"))
                # Stop detection since window is gone
                if self.is_running:
                    self._stop()
                    QMessageBox.warning(self, tr("warning"), tr("window_closed"))
                return None
            region = self._window_rect(self.current_hwnd)
            if not region:
                # Window may be minimized or invisible
                return None
            self.capture_region = region
        elif source == self.SOURCE_REGION:
            if not self.current_rect:
                return None
            r = self.current_rect
            # 将逻辑像素矩形转换为物理像素区域
            physical_region = self._logical_to_physical_rect(r)
            # 规范化区域（裁剪到屏幕边界内）
            region = self._normalize_region(physical_region)
            if not region:
                logger.warning(f"区域无效或超出屏幕边界: {physical_region}")
                return None
            self.capture_region = region
        else:
            return None

        if not self.camera:
            return None

        try:
            frame = self.camera.grab(region=self.capture_region)
        except ValueError as e:
            # 捕获区域无效错误（例如超出屏幕范围）
            logger.exception(f"捕获区域无效: {e}, 区域: {self.capture_region}")
            return None

        if frame is None:
            # Recreate camera once to recover from occasional dxcam invalid state.
            self._release_camera()
            self.camera = self._create_camera()
            if self.camera:
                try:
                    frame = self.camera.grab(region=self.capture_region)
                except ValueError as e:
                    logger.error(f"重新创建相机后捕获区域仍无效: {e}")
                    return None
        if frame is None:
            return None

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _read_image(self, image_path: str) -> Optional[np.ndarray]:
        # Use imdecode to support non-ascii paths on Windows.
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        return cv2.imread(image_path)

    def _run_image_once(self):
        image_path = self.image_path_edit.text().strip()
        if not image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        img = self._read_image(image_path)
        if img is None:
            QMessageBox.warning(self, tr("warning"), tr("cannot_load_image") + image_path)
            return

        self.status_label.setText(tr("detecting_status"))
        self._run_detection(img)
        self.status_label.setText(tr("ready"))

    def _run_detection(self, frame: np.ndarray):
        detections = self.model_manager.predict_image(frame)

        # 无检测时直接使用原帧，避免不必要的内存拷贝
        if not detections:
            self._update_preview(frame)
            return

        vis = frame.copy()
        labels_batch = []

        for det in detections:
            x = int(det.get("x", 0))
            y = int(det.get("y", 0))
            w = int(det.get("width", 0))
            h = int(det.get("height", 0))
            conf = float(det.get("confidence", 0))
            cls_id = det.get("class_id", 0)

            label = det.get("class_name", str(cls_id))
            if self.show_confidence:
                label += f":{conf:.2f}"

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            labels_batch.append((label, (x, y), self.label_font_size,
                                (0, 255, 0), max(1, int(self.label_font_size * 2))))

        _draw_unicode_texts_batch(vis, labels_batch)
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

    def _on_show_conf_toggled(self, checked: bool):
        self.show_confidence = checked

    def closeEvent(self, event):
        self._stop()
        if self.picking_window:
            self._stop_window_pick(confirmed=False)
        super().closeEvent(event)
