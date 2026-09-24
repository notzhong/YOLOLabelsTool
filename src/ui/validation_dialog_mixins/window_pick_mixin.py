"""
窗口/区域拾取 Mixin

从 validation_dialog.py 拆分（P2 重构）：窗口拾取（光标轮询 + 高亮）、
区域选择器与 DPI 屏幕坐标换算（含 win32 调用）。
通过 self 访问 ValidationDialog 的属性与其他 Mixin 方法。
"""

import ctypes
import time
from ctypes import wintypes
from typing import Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog

from src.ui.region_selector import RegionSelector, WindowHighlighter
from src.utils.i18n import tr
from src.utils.logger import get_logger_simple
from src.utils.win32_helpers import (
    SM_CXVIRTUALSCREEN,
    SM_CYVIRTUALSCREEN,
    SM_XVIRTUALSCREEN,
    SM_YVIRTUALSCREEN,
    VK_LBUTTON,
    VK_RBUTTON,
    get_user32,
    get_window_title,
    to_root_window,
)

logger = get_logger_simple(__name__)


class WindowPickMixin:
    """窗口/区域拾取 Mixin：光标轮询、窗口高亮、区域选择与坐标换算"""


    def _pick_window(self):
        if self.picking_window:
            self._stop_window_pick(confirmed=False)
            return

        self._pick_saved_hwnd = self.current_hwnd  # 保存已确认的窗口，取消时恢复
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
        elif not confirmed:
            # 取消时恢复之前已确认的窗口选择
            saved = getattr(self, '_pick_saved_hwnd', None)
            self.current_hwnd = saved
            if saved:
                title = get_window_title(saved)
                self.window_info_label.setText(tr("window_selected").replace("{title}", title))
            else:
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

    def _window_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        _user32 = get_user32()
        if not _user32.IsWindow(wintypes.HWND(hwnd)):
            return None
        rect = wintypes.RECT()
        ok = _user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
        if not ok:
            return None
        return self._normalize_region((rect.left, rect.top, rect.right, rect.bottom))

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
