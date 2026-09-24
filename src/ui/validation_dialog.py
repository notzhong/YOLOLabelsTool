"""Validation dialog for realtime model verification.

P2 重构：实现拆分至 src/ui/validation_dialog_mixins/ 包（界面构建 / 窗口区域拾取 /
实时检测循环 + Unicode 文本绘制），本模块保留对话框骨架与对外类名。
"""

from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import QRect, QTimer
from PySide6.QtWidgets import QDialog

from src.ui.validation_dialog_mixins import (
    DXCAM_AVAILABLE,
    DetectMixin,
    ValidationUiMixin,
    WindowPickMixin,
    _draw_unicode_text,
    _draw_unicode_texts_batch,
    _get_unicode_font,
)
from src.utils.i18n import tr

__all__ = [
    "DXCAM_AVAILABLE",
    "ValidationDialog",
    "_draw_unicode_text",
    "_draw_unicode_texts_batch",
    "_get_unicode_font",
]


class ValidationDialog(
    ValidationUiMixin,
    WindowPickMixin,
    DetectMixin,
    QDialog,
):
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
