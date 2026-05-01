"""Validation dialog for realtime model verification."""

import ctypes
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QCursor, QGuiApplication, QScreen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.utils.i18n import tr
from src.utils.logger import get_logger_simple
from src.utils.widget_helpers import SliderSpinBoxBinder
from src.ui.region_selector import WindowHighlighter, RegionSelector
from src.utils.win32_helpers import (
    get_user32, to_root_window, get_window_title,
    VK_LBUTTON, VK_RBUTTON,
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN,
)

logger = get_logger_simple(__name__)

try:
    import dxcam

    DXCAM_AVAILABLE = True
except Exception:
    DXCAM_AVAILABLE = False



class ValidationDialog(QDialog):
    """Validation dialog for model testing."""

    SOURCE_WINDOW = 0
    SOURCE_REGION = 1
    SOURCE_IMAGE = 2

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

        # Global window-pick state:
        # poll cursor location + mouse button edges, then confirm/cancel by click.
        self.pick_timer = QTimer(self)
        self.pick_timer.timeout.connect(self._update_window_pick)
        self.picking_window = False
        self._pick_prev_left_down = False
        self._pick_prev_right_down = False
        self._pick_ignore_until = 0.0
        self._pick_cursor_owned = False
        # Window highlighter
        self._highlighter = None

        # 文件对话框路径记忆
        self._last_browse_path = str(Path.cwd())

        # Display settings
        self.label_font_size = 0.5
        self.show_confidence = True

        self._init_ui()
        self._update_model_status()

    def _init_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        left_container = QWidget()
        left_panel = QVBoxLayout(left_container)
        left_panel.setContentsMargins(5, 5, 5, 5)

        model_group = QGroupBox(tr("model"))
        model_layout = QVBoxLayout(model_group)
        self.model_status_label = QLabel("")
        self.btn_load_model = QPushButton(tr("load_model"))
        self.btn_load_model.clicked.connect(self._load_model)
        model_layout.addWidget(self.model_status_label)
        model_layout.addWidget(self.btn_load_model)
        left_panel.addWidget(model_group)

        source_group = QGroupBox(tr("validation_source"))
        source_layout = QVBoxLayout(source_group)

        self.source_combo = QComboBox()
        self.source_combo.addItems([tr("source_window"), tr("source_region"), tr("source_image")])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)

        self.window_info_label = QLabel(tr("window_not_selected"))
        self.btn_pick_window = QPushButton(tr("pick_window"))
        self.btn_pick_window.clicked.connect(self._pick_window)
        source_layout.addWidget(self.window_info_label)
        source_layout.addWidget(self.btn_pick_window)

        self.region_info_label = QLabel(tr("region_not_selected"))
        self.btn_pick_region = QPushButton(tr("pick_region"))
        self.btn_pick_region.clicked.connect(self._pick_region)
        source_layout.addWidget(self.region_info_label)
        source_layout.addWidget(self.btn_pick_region)

        image_row = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.btn_browse_image = QPushButton(tr("browse"))
        self.btn_browse_image.clicked.connect(self._browse_image)
        image_row.addWidget(self.image_path_edit, 1)
        image_row.addWidget(self.btn_browse_image)
        source_layout.addLayout(image_row)

        left_panel.addWidget(source_group)

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
        conf_row.addWidget(conf_label)
        conf_row.addWidget(self.conf_slider, 1)
        conf_row.addWidget(self.conf_spin)
        params_layout.addLayout(conf_row)

        self._conf_binder = SliderSpinBoxBinder(
            self.conf_slider, self.conf_spin, divider=100,
            on_value_changed=self.model_manager.set_confidence_threshold
        )

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
        iou_row.addWidget(iou_label)
        iou_row.addWidget(self.iou_slider, 1)
        iou_row.addWidget(self.iou_spin)
        params_layout.addLayout(iou_row)

        self._iou_binder = SliderSpinBoxBinder(
            self.iou_slider, self.iou_spin, divider=100,
            on_value_changed=self.model_manager.set_iou_threshold
        )

        left_panel.addWidget(params_group)

        display_group = QGroupBox(tr("display_settings"))
        display_layout = QVBoxLayout(display_group)

        font_row = QHBoxLayout()
        font_label = QLabel(tr("label_font_size"))
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setRange(1, 20)
        self.font_size_slider.setValue(int(self.label_font_size * 10))
        self.font_size_spin = QDoubleSpinBox()
        self.font_size_spin.setRange(0.1, 2.0)
        self.font_size_spin.setSingleStep(0.1)
        self.font_size_spin.setValue(self.label_font_size)
        self.font_size_spin.setDecimals(1)
        font_row.addWidget(font_label)
        font_row.addWidget(self.font_size_slider, 1)
        font_row.addWidget(self.font_size_spin)
        display_layout.addLayout(font_row)

        self._font_binder = SliderSpinBoxBinder(
            self.font_size_slider, self.font_size_spin, divider=10,
            on_value_changed=lambda v: setattr(self, 'label_font_size', v)
        )

        self.show_conf_check = QCheckBox(tr("show_confidence"))
        self.show_conf_check.setChecked(self.show_confidence)
        self.show_conf_check.toggled.connect(self._on_show_conf_toggled)
        display_layout.addWidget(self.show_conf_check)

        left_panel.addWidget(display_group)

        self.btn_toggle = QPushButton(tr("start_detect"))
        self.btn_toggle.clicked.connect(self._toggle_detect)
        left_panel.addWidget(self.btn_toggle)

        self.status_label = QLabel(tr("ready"))
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()

        self.preview_label = QLabel(tr("no_image_loaded"))
        self.preview_label.setAlignment(Qt.AlignCenter)

        splitter.addWidget(left_container)
        splitter.addWidget(self.preview_label)
        splitter.setStretchFactor(0, 0)  # 左侧不拉伸
        splitter.setStretchFactor(1, 1)  # 右侧自适应拉伸
        splitter.setSizes([280, 600])

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        self._on_source_changed()

    def _selected_source(self) -> int:
        return self.source_combo.currentIndex()

    def _update_model_status(self):
        if self.model_manager.is_model_loaded():
            info = self.model_manager.get_model_info()
            name = Path(info.get("path", "")).name
            self.model_status_label.setText(tr("model_loaded_status").replace("{model_name}", name))
        else:
            self.model_status_label.setText(tr("model_not_loaded"))

    def _load_model(self):
        if not self.model_manager.is_available():
            QMessageBox.warning(self, tr("warning"), tr("ultralytics_not_installed"))
            return
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("select_pretrained_model_file"),
            self._last_browse_path,
            tr("model_file_filter"),
        )
        if model_path:
            self._last_browse_path = str(Path(model_path).parent)
            if self.model_manager.load_model(model_path):
                self._update_model_status()
            else:
                QMessageBox.warning(
                    self,
                    tr("warning"),
                    tr("load_model_failed").replace("{model_path}", model_path),
                )

    def _on_source_changed(self):
        source = self._selected_source()
        is_window = source == self.SOURCE_WINDOW
        is_region = source == self.SOURCE_REGION
        is_image = source == self.SOURCE_IMAGE

        if self.picking_window and not is_window:
            self._stop_window_pick(confirmed=False)

        self.window_info_label.setVisible(is_window)
        self.btn_pick_window.setVisible(is_window)

        self.region_info_label.setVisible(is_region)
        self.btn_pick_region.setVisible(is_region)

        self.image_path_edit.setVisible(is_image)
        self.btn_browse_image.setVisible(is_image)

    def _pick_window(self):
        if self.picking_window:
            self._stop_window_pick(confirmed=False)
            return

        self.picking_window = True
        self.btn_pick_window.setText(tr("picking_window"))
        _user32 = get_user32()
        self._pick_prev_left_down = bool(_user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        self._pick_prev_right_down = bool(_user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        # Ignore button events for 300ms to swallow the release from clicking the "pick" button
        self._pick_ignore_until = time.monotonic() + 0.3
        if QApplication.overrideCursor() is None:
            QApplication.setOverrideCursor(Qt.CrossCursor)
            self._pick_cursor_owned = True
        self.pick_timer.start(40)

    def _stop_window_pick(self, confirmed: bool):
        """Exit pick mode and restore cursor/button state."""
        self.picking_window = False
        if self.pick_timer.isActive():
            self.pick_timer.stop()
        self.btn_pick_window.setText(tr("pick_window"))
        if self._pick_cursor_owned:
            QApplication.restoreOverrideCursor()
            self._pick_cursor_owned = False

        # Remove window highlighter
        self._cleanup_highlighter()

        if confirmed and self.current_hwnd:
            title = get_window_title(self.current_hwnd)
            self.window_info_label.setText(tr("window_selected").replace("{title}", title))
        else:
            # Clear selection if cancelled or not confirmed
            self.current_hwnd = None
            self.window_info_label.setText(tr("window_not_selected"))

    def _update_window_pick(self):
        _user32 = get_user32()
        # Read cursor-under-window globally, not limited by dialog focus.
        point = wintypes.POINT(QCursor.pos().x(), QCursor.pos().y())
        hwnd = int(_user32.WindowFromPoint(point) or 0)
        hwnd = to_root_window(hwnd)

        own_hwnd = int(self.winId())
        valid_candidate = False
        if hwnd and hwnd != own_hwnd:
            if _user32.IsWindow(wintypes.HWND(hwnd)) and _user32.IsWindowVisible(wintypes.HWND(hwnd)):
                valid_candidate = True

        # Update window highlighter based on current cursor position
        self._update_window_highlighter(hwnd if valid_candidate else 0)

        # Update label with the window under cursor
        if valid_candidate:
            if hwnd != self.current_hwnd:
                self.current_hwnd = hwnd
                title = get_window_title(hwnd)
                self.window_info_label.setText(tr("window_selected").replace("{title}", title))
        else:
            if self.current_hwnd is not None:
                self.current_hwnd = None
                self.window_info_label.setText(tr("window_not_selected"))

        now = time.monotonic()
        # Ignore button events during cooldown period (swallow the button click that started pick mode)
        if now < self._pick_ignore_until:
            return

        left_down = bool(_user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        right_down = bool(_user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        # Convert key level state into edge events (pressed this tick).
        left_pressed = left_down and not self._pick_prev_left_down
        right_pressed = right_down and not self._pick_prev_right_down
        self._pick_prev_left_down = left_down
        self._pick_prev_right_down = right_down

        # Right click cancels pick mode
        if right_pressed:
            self._stop_window_pick(confirmed=False)
            return

        # Left click on a valid window confirms the selection
        if left_pressed and valid_candidate:
            self.current_hwnd = hwnd
            self._stop_window_pick(confirmed=True)

    def _update_window_highlighter(self, hwnd: int):
        """Update or create window highlighter for the given window handle."""
        if not hwnd:
            self._cleanup_highlighter()
            return

        # Get window rectangle in physical pixels
        physical_rect = self._window_rect(hwnd)
        if not physical_rect:
            self._cleanup_highlighter()
            return

        # Convert physical pixels to logical pixels for Qt
        logical_rect = self._physical_to_logical_rect(physical_rect)

        # Create or update highlighter
        if self._highlighter is None:
            self._highlighter = WindowHighlighter()
        self._highlighter.set_target_rect(logical_rect)
        self._highlighter.show()

    def _cleanup_highlighter(self):
        """Remove window highlighter."""
        if self._highlighter is not None:
            self._highlighter.close()
            self._highlighter = None

    def _pick_region(self):
        # 创建区域选择器
        selector = RegionSelector(self)

        try:
            result = selector.exec()

            if result == QDialog.Accepted:
                rect = selector._selected_rect
                if rect and rect.width() > 1 and rect.height() > 1:
                    self.current_rect = rect
                    self.region_info_label.setText(
                        tr("region_selected").replace(
                            "{rect}", f"{rect.x()},{rect.y()} {rect.width()}x{rect.height()}"
                        )
                    )
                else:
                    logger.info("选择的区域无效")
        except Exception as e:
            logger.exception(f"区域选择过程中发生异常: {e}")
        finally:
            selector.close()
            self.raise_()
            self.activateWindow()

    def _browse_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("select_image_file_dialog_title"),
            self._last_browse_path,
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif)",
        )
        if image_path:
            self.image_path_edit.setText(image_path)
            self._last_browse_path = str(Path(image_path).parent)

    def _toggle_detect(self):
        if self.is_running:
            self._stop()
            return

        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return

        source = self._selected_source()

        if source == self.SOURCE_IMAGE:
            self._run_image_once()
            return

        if not DXCAM_AVAILABLE:
            QMessageBox.warning(self, tr("warning"), tr("dxcam_not_installed"))
            return

        if source == self.SOURCE_WINDOW and not self.current_hwnd:
            QMessageBox.warning(self, tr("warning"), tr("window_not_selected"))
            return

        if source == self.SOURCE_REGION and not self.current_rect:
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
        # Ensure old camera is released before creating new one
        self._release_camera()

        if DXCAM_AVAILABLE:
            self.camera = self._create_camera()

        self.is_running = True
        self.btn_toggle.setText(tr("stop_detect"))
        self.timer.start(120)
        self.status_label.setText(tr("detecting_status"))

    def _stop(self):
        self.is_running = False
        self.btn_toggle.setText(tr("start_detect"))
        self.timer.stop()
        self.status_label.setText(tr("ready"))
        # Release camera resources when not detecting
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

    def _get_screen_bounds(self) -> Tuple[int, int, int, int]:
        """获取虚拟桌面的物理边界坐标（所有显示器的联合区域）"""
        _user32 = get_user32()
        # 使用Windows API获取虚拟屏幕的边界
        try:
            x = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            y = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            width = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            height = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
            if width > 0 and height > 0:
                return x, y, x + width, y + height
        except Exception:
            pass

        # 回退到Qt获取所有屏幕的联合区域
        screens = QGuiApplication.screens()
        if screens:
            # 获取逻辑坐标的联合区域
            virtual_rect = screens[0].geometry()
            for screen in screens[1:]:
                virtual_rect = virtual_rect.united(screen.geometry())

            # 转换为物理像素（考虑DPI缩放）
            # 注意：不同显示器可能有不同的DPI，这里使用主屏幕的DPI作为近似
            primary_screen = QGuiApplication.primaryScreen()
            dpr = primary_screen.devicePixelRatio() if primary_screen else 1.0

            x = int(virtual_rect.x() * dpr)
            y = int(virtual_rect.y() * dpr)
            width = int(virtual_rect.width() * dpr)
            height = int(virtual_rect.height() * dpr)
            return x, y, x + width, y + height

        # 默认返回 3840x2160，与错误信息中的分辨率一致
        return 0, 0, 3840, 2160

    def _normalize_region(self, rect: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        left, top, right, bottom = [int(v) for v in rect]
        if right <= left or bottom <= top:
            return None

        # 裁剪到屏幕边界内
        screen_left, screen_top, screen_right, screen_bottom = self._get_screen_bounds()
        left = max(left, screen_left)
        top = max(top, screen_top)
        right = min(right, screen_right)
        bottom = min(bottom, screen_bottom)

        # 再次检查有效性
        if right <= left or bottom <= top:
            return None

        return left, top, right, bottom

    def _physical_to_logical_rect(self, physical_rect: Tuple[int, int, int, int]) -> QRect:
        """将物理像素矩形转换为逻辑像素矩形（考虑DPI缩放）"""
        left, top, right, bottom = physical_rect
        width = right - left
        height = bottom - top

        # 先使用主屏幕DPI进行粗略转换，找到包含窗口的屏幕
        primary_screen = QGuiApplication.primaryScreen()
        primary_dpr = primary_screen.devicePixelRatio() if primary_screen else 1.0

        # 粗略的逻辑坐标用于查找屏幕
        rough_logical_left = left / primary_dpr
        rough_logical_top = top / primary_dpr
        rough_center_x = rough_logical_left + (width / primary_dpr) / 2
        rough_center_y = rough_logical_top + (height / primary_dpr) / 2
        rough_center_point = QPoint(int(rough_center_x), int(rough_center_y))

        # 找到包含该点的屏幕
        screen = QGuiApplication.screenAt(rough_center_point)
        if not screen:
            # 回退到主屏幕
            screen = primary_screen

        # 使用实际屏幕的DPI进行精确转换
        if screen:
            dpr = screen.devicePixelRatio()
            if dpr > 0:
                # 物理像素转换为逻辑像素
                logical_left = left / dpr
                logical_top = top / dpr
                logical_width = width / dpr
                logical_height = height / dpr
                return QRect(int(logical_left), int(logical_top),
                           int(logical_width), int(logical_height))

        # 如果没有找到屏幕或dpr无效，直接使用原始坐标（假设无缩放）
        return QRect(left, top, width, height)

    def _logical_to_physical_rect(self, logical_rect: QRect) -> Tuple[int, int, int, int]:
        """将逻辑像素矩形转换为物理像素区域（考虑DPI缩放）"""
        # 获取矩形中心点所在的屏幕
        center = logical_rect.center()
        screen = QGuiApplication.screenAt(center)
        if not screen:
            # 回退到主屏幕
            screen = QGuiApplication.primaryScreen()

        if screen:
            dpr = screen.devicePixelRatio()
            if dpr > 0:
                # 逻辑像素转换为物理像素
                left = int(logical_rect.left() * dpr)
                top = int(logical_rect.top() * dpr)
                right = int(logical_rect.right() * dpr)
                bottom = int(logical_rect.bottom() * dpr)
                return left, top, right, bottom

        # 如果没有找到屏幕或dpr无效，直接使用原始坐标（假设无缩放）
        return logical_rect.left(), logical_rect.top(), logical_rect.right(), logical_rect.bottom()

    def _window_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        _user32 = get_user32()
        if not _user32.IsWindow(wintypes.HWND(hwnd)):
            return None
        rect = wintypes.RECT()
        ok = _user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
        if not ok:
            return None
        return self._normalize_region((rect.left, rect.top, rect.right, rect.bottom))

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
        vis = frame.copy()

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
            cv2.putText(
                vis,
                label,
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.label_font_size,
                (0, 255, 0),
                max(1, int(self.label_font_size * 2)),
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

    def _on_show_conf_toggled(self, checked: bool):
        self.show_confidence = checked

    def closeEvent(self, event):
        self._stop()
        if self.picking_window:
            self._stop_window_pick(confirmed=False)
        super().closeEvent(event)
