"""Validation dialog for realtime model verification."""

import ctypes
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QSize, QEvent
from PySide6.QtGui import QImage, QPixmap, QCursor, QGuiApplication, QWindow, QScreen
from PySide6.QtWidgets import (
    QApplication,
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
    QRubberBand,
    QSlider,
    QVBoxLayout,
)

from src.utils.i18n import tr
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)

try:
    import dxcam

    DXCAM_AVAILABLE = True
except Exception:
    DXCAM_AVAILABLE = False


_user32 = ctypes.windll.user32
# WinAPI constants used by window picking.
GA_ROOT = 2
MONITOR_DEFAULTTONEAREST = 2
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
# Virtual screen metrics
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def _configure_user32() -> None:
    """Set explicit WinAPI signatures to avoid HWND truncation on 64-bit."""
    _user32.WindowFromPoint.argtypes = [wintypes.POINT]
    _user32.WindowFromPoint.restype = wintypes.HWND

    _user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    _user32.GetAncestor.restype = wintypes.HWND

    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int

    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int

    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetWindowRect.restype = wintypes.BOOL

    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.IsWindow.restype = wintypes.BOOL

    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL

    _user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    _user32.GetAsyncKeyState.restype = ctypes.c_short

    _user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.MonitorFromWindow.restype = wintypes.HANDLE

    _user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    _user32.MonitorFromPoint.restype = wintypes.HANDLE

    _user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFOEXW)]
    _user32.GetMonitorInfoW.restype = wintypes.BOOL


_configure_user32()


def _to_root_window(hwnd: int) -> int:
    """Promote child/control HWND to top-level window HWND."""
    if not hwnd:
        return 0
    root = int(_user32.GetAncestor(wintypes.HWND(hwnd), GA_ROOT) or 0)
    return root if root else hwnd


def _get_window_title(hwnd: int) -> str:
    length = _user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
    title = buf.value.strip()
    return title if title else f"HWND:{hwnd}"


class WindowHighlighter(QDialog):
    """Transparent overlay that draws a colored frame around a target window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background-color: transparent;")

        self._border_color = Qt.red
        self._border_width = 4

    def set_target_rect(self, rect: QRect):
        """Move and resize the overlay to surround the given rectangle."""
        # Expand the rectangle to make room for the border
        expanded = rect.adjusted(-self._border_width, -self._border_width,
                                 self._border_width, self._border_width)
        self.setGeometry(expanded)

    def paintEvent(self, event):
        """Draw a colored border around the overlay."""
        from PySide6.QtGui import QPainter, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self._border_color, self._border_width)
        painter.setPen(pen)
        # Draw rectangle inset by half border width to keep border fully visible
        half = self._border_width // 2
        rect = self.rect().adjusted(half, half, -half, -half)
        painter.drawRect(rect)


class RegionSelector(QDialog):
    """Fullscreen region selector on virtual desktop."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)  # 设置为模态对话框，阻塞父窗口直到选择完成
        # 窗口标志：无边框，置顶，作为普通窗口
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window
        self.setWindowFlags(flags)
        self.setCursor(Qt.CrossCursor)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)  # 不自动删除窗口
        # 设置对话框覆盖整个虚拟桌面
        virtual_geometry = self._virtual_geometry()
        logger.debug(f"RegionSelector初始化，虚拟几何: {virtual_geometry}, 父窗口: {parent}, 父窗口ID: {parent.winId() if parent else 'None'}")
        self.setGeometry(virtual_geometry)

        # 确保能接收键盘和鼠标事件
        self.setFocusPolicy(Qt.StrongFocus)
        # 安装事件过滤器来监控窗口事件
        self.installEventFilter(self)

        # 设置轻微透明度，让用户知道正在选择区域
        self.setWindowOpacity(0.5)  # 增加透明度到50%，更容易看到
        # 设置半透明颜色，让用户知道这是覆盖层
        self.setStyleSheet("background-color: rgba(128, 128, 128, 128);")  # 灰色半透明背景

        self._origin_global = QPoint()
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._selected_rect: Optional[QRect] = None
        self._selection_started = False  # 标记是否已开始选择
        logger.debug("RegionSelector初始化完成")

    def showEvent(self, event):
        """窗口显示事件"""
        logger.debug(f"RegionSelector.showEvent: 窗口显示，几何: {self.geometry()}")
        super().showEvent(event)

    def hideEvent(self, event):
        """窗口隐藏事件"""
        logger.debug("RegionSelector.hideEvent: 窗口隐藏")
        super().hideEvent(event)

    def eventFilter(self, obj, event):
        """事件过滤器，监控窗口事件"""
        from PySide6.QtCore import QEvent
        if obj == self:
            event_type = event.type()
            if event_type == QEvent.Close:
                logger.debug(f"RegionSelector.eventFilter: 接收到Close事件")
            elif event_type == QEvent.WindowActivate:
                logger.debug("RegionSelector.eventFilter: 窗口激活")
            elif event_type == QEvent.WindowDeactivate:
                logger.debug("RegionSelector.eventFilter: 窗口失活")
            elif event_type == QEvent.FocusIn:
                logger.debug("RegionSelector.eventFilter: 获得焦点")
            elif event_type == QEvent.FocusOut:
                logger.debug("RegionSelector.eventFilter: 失去焦点")
        return super().eventFilter(obj, event)

    @staticmethod
    def _virtual_geometry() -> QRect:
        """返回虚拟桌面的逻辑像素几何（所有屏幕的联合区域）"""
        screens = QGuiApplication.screens()
        if not screens:
            return QRect(0, 0, 1920, 1080)
        rect = screens[0].geometry()
        for screen in screens[1:]:
            rect = rect.united(screen.geometry())
        return rect

    def keyPressEvent(self, event):
        logger.debug(f"RegionSelector.keyPressEvent: {event.key()}")
        if event.key() == Qt.Key_Escape:
            logger.debug("ESC键按下，取消区域选择")
            self.reject()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        logger.debug(f"RegionSelector.mousePressEvent: button={event.button()}, pos={event.position()}, global={event.globalPosition()}")
        if event.button() != Qt.LeftButton:
            logger.debug("非左键按下，忽略")
            return super().mousePressEvent(event)

        # 标记选择已开始
        self._selection_started = True
        logger.debug("选择已开始")

        # 记录全局起始点
        self._origin_global = event.globalPosition().toPoint()
        logger.debug(f"设置起始点: {self._origin_global}")

        # 转换为本地坐标设置橡皮筋
        origin_local = self.mapFromGlobal(self._origin_global)
        logger.debug(f"起始点本地坐标: {origin_local}")

        self._rubber_band.setGeometry(QRect(origin_local, QSize()))
        self._rubber_band.show()
        logger.debug("橡皮筋显示")

    def mouseMoveEvent(self, event):
        if not self._rubber_band.isVisible():
            logger.debug("mouseMoveEvent: 橡皮筋不可见，忽略")
            return super().mouseMoveEvent(event)

        # 获取当前全局位置并转换为本地坐标
        current_global = event.globalPosition().toPoint()
        current_local = self.mapFromGlobal(current_global)
        logger.debug(f"mouseMoveEvent: 当前位置 global={current_global}, local={current_local}")

        # 构建矩形（从起始点到当前点）
        origin_local = self.mapFromGlobal(self._origin_global)
        rect = QRect(origin_local, current_local).normalized()
        logger.debug(f"设置橡皮筋矩形: {rect}")
        self._rubber_band.setGeometry(rect)
        event.accept()

    def mouseReleaseEvent(self, event):
        logger.debug(f"RegionSelector.mouseReleaseEvent: button={event.button()}, pos={event.position()}, 橡皮筋可见: {self._rubber_band.isVisible()}")
        if event.button() != Qt.LeftButton:
            logger.debug("非左键释放，忽略")
            return super().mouseReleaseEvent(event)

        # 只有当橡皮筋可见（即已经开始选择）时才处理释放事件
        if not self._rubber_band.isVisible():
            logger.debug("橡皮筋不可见，忽略释放事件（可能是从按钮点击传递过来的释放事件）")
            return super().mouseReleaseEvent(event)

        # 获取释放时的全局位置
        current_global = event.globalPosition().toPoint()
        logger.debug(f"释放位置: {current_global}, 起始位置: {self._origin_global}")

        # 直接使用全局坐标构建矩形（逻辑像素）
        self._selected_rect = QRect(self._origin_global, current_global).normalized()
        logger.debug(f"选择的矩形: {self._selected_rect}")

        # 检查选择的区域是否有效（最小尺寸）
        if self._selected_rect.width() < 5 or self._selected_rect.height() < 5:
            logger.debug(f"选择的区域太小: {self._selected_rect.width()}x{self._selected_rect.height()}，忽略")
            self._rubber_band.hide()
            return

        self._rubber_band.hide()
        logger.debug("隐藏橡皮筋，接受选择")
        self.accept()
        event.accept()

    def closeEvent(self, event):
        logger.debug(f"RegionSelector.closeEvent - 选择矩形: {self._selected_rect}")
        # 如果窗口被关闭而没有明确accept/reject，则reject
        if self._selected_rect is None:
            logger.debug("窗口被关闭，没有选择区域，自动reject")
            self.reject()
        super().closeEvent(event)



class ValidationDialog(QDialog):
    """Validation dialog for model testing."""

    SOURCE_WINDOW = 0
    SOURCE_REGION = 1
    SOURCE_IMAGE = 2
    DRAG_THRESHOLD = 2  # 像素，最小拖动距离阈值（防止误点击）

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
        # For drag-to-select
        self._pick_drag_start_pos = None
        self._pick_drag_start_time = 0.0
        self._pick_drag_start_hwnd = None
        self._pick_dragging = False
        # Window highlighter
        self._highlighter = None

        self._init_ui()
        self._update_model_status()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

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

        self.btn_toggle = QPushButton(tr("start_detect"))
        self.btn_toggle.clicked.connect(self._toggle_detect)
        left_panel.addWidget(self.btn_toggle)

        self.status_label = QLabel(tr("ready"))
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()

        self.preview_label = QLabel(tr("no_image_loaded"))
        self.preview_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.preview_label, 1)

        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)

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
            str(Path.cwd()),
            tr("model_file_filter"),
        )
        if model_path:
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
            # Second click on the same button cancels pick mode.
            self._stop_window_pick(confirmed=False)
            return

        self.picking_window = True
        self.btn_pick_window.setText(tr("picking_window"))
        self._pick_prev_left_down = bool(_user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        self._pick_prev_right_down = bool(_user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        # No ignore period - drag threshold prevents accidental selection
        self._pick_ignore_until = 0.0
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

        # Clean up drag state
        self._pick_drag_start_pos = None
        self._pick_drag_start_hwnd = None
        self._pick_dragging = False

        # Remove window highlighter
        self._cleanup_highlighter()

        if confirmed and self.current_hwnd:
            title = _get_window_title(self.current_hwnd)
            self.window_info_label.setText(tr("window_selected").replace("{title}", title))
        else:
            # Clear selection if cancelled or not confirmed
            self.current_hwnd = None
            self.window_info_label.setText(tr("window_not_selected"))

    def _update_window_pick(self):
        # Read cursor-under-window globally, not limited by dialog focus.
        point = wintypes.POINT(QCursor.pos().x(), QCursor.pos().y())
        hwnd = int(_user32.WindowFromPoint(point) or 0)
        hwnd = _to_root_window(hwnd)

        own_hwnd = int(self.winId())
        valid_candidate = False
        if hwnd and hwnd != own_hwnd:
            if _user32.IsWindow(wintypes.HWND(hwnd)) and _user32.IsWindowVisible(wintypes.HWND(hwnd)):
                valid_candidate = True

        # Update window highlighter based on current cursor position
        self._update_window_highlighter(hwnd if valid_candidate else 0)

        # Update current_hwnd and label only if not dragging
        if not self._pick_dragging:
            if hwnd != self.current_hwnd:
                if valid_candidate:
                    self.current_hwnd = hwnd
                    # Update label with window title
                    title = _get_window_title(hwnd)
                    self.window_info_label.setText(tr("window_selected").replace("{title}", title))
                else:
                    self.current_hwnd = None
                    self.window_info_label.setText(tr("window_not_selected"))

        now = time.monotonic()
        left_down = bool(_user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        right_down = bool(_user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        # Convert key level state into edge events (pressed this tick).
        left_pressed = left_down and not self._pick_prev_left_down
        left_released = not left_down and self._pick_prev_left_down
        right_pressed = right_down and not self._pick_prev_right_down
        self._pick_prev_left_down = left_down
        self._pick_prev_right_down = right_down

        # Handle right click cancel (ignore time limit for cancel)
        if right_pressed:
            self._stop_window_pick(confirmed=False)
            return

        # Handle left button events
        if left_pressed:
            # Mouse button just pressed - start tracking drag
            self._pick_drag_start_pos = QCursor.pos()
            self._pick_drag_start_time = now
            self._pick_dragging = True
        elif left_released and self._pick_drag_start_pos is not None:
            # Mouse button released - check if we should confirm selection
            release_pos = QCursor.pos()
            dx = release_pos.x() - self._pick_drag_start_pos.x()
            dy = release_pos.y() - self._pick_drag_start_pos.y()
            distance = abs(dx) + abs(dy)  # 曼哈顿距离

            if distance >= self.DRAG_THRESHOLD and valid_candidate:
                # 拖动距离足够且释放时在有效窗口上 - 确认选择
                self.current_hwnd = hwnd
                self._stop_window_pick(confirmed=True)
            else:
                # 拖动距离不足或释放位置无效 - 取消选择
                self._stop_window_pick(confirmed=False)
            # Reset drag state
            self._pick_drag_start_pos = None
        elif left_down and self._pick_drag_start_pos is not None:
            # Mouse is being held down - update highlighter based on current window
            # No action needed here, highlighter is updated elsewhere
            pass
        else:
            # No button activity - just update highlighter
            pass

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

        # Debug output
        logger.debug(f"Window highlighter - HWND: {hwnd}")
        logger.debug(f"  Physical rect: {physical_rect}")
        logger.debug(f"  Logical rect: {logical_rect}")
        logger.debug(f"  Screen DPR: {QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() else 1.0}")

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
        logger.debug("_pick_region 开始，创建区域选择器")

        # 创建区域选择器
        selector = RegionSelector(self)
        logger.debug(f"RegionSelector创建完成，父窗口: {self}, selector窗口: {selector}")

        try:
            logger.debug("显示RegionSelector (exec)")
            result = selector.exec()
            logger.debug(f"RegionSelector.exec返回结果: {result}")

            if result == QDialog.Accepted:
                rect = selector._selected_rect
                if rect and rect.width() > 1 and rect.height() > 1:
                    logger.debug(f"区域选择完成: {rect.x()},{rect.y()} {rect.width()}x{rect.height()}")
                    self.current_rect = rect
                    self.region_info_label.setText(
                        tr("region_selected").replace(
                            "{rect}", f"{rect.x()},{rect.y()} {rect.width()}x{rect.height()}"
                        )
                    )
                else:
                    logger.debug("选择的区域无效")
            else:
                logger.debug("区域选择取消")
        except Exception as e:
            logger.exception(f"区域选择过程中发生异常: {e}")
        finally:
            logger.debug("区域选择过程结束，清理资源")
            # 确保选择器窗口被关闭
            selector.close()

    def _browse_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("select_image_file_dialog_title"),
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif)",
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
            if not _user32.IsWindow(wintypes.HWND(self.current_hwnd)):
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

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{cls_id}:{conf:.2f}",
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
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
        if self.picking_window:
            self._stop_window_pick(confirmed=False)
        super().closeEvent(event)
