"""ValidationDialog 职责拆分的 Mixin 与绘制工具模块（P2 重构）"""

from .detect_mixin import DXCAM_AVAILABLE, DetectMixin
from .ui_mixin import ValidationUiMixin
from .unicode_text import (
    _draw_unicode_text,
    _draw_unicode_texts_batch,
    _get_unicode_font,
)
from .window_pick_mixin import WindowPickMixin

__all__ = ["DXCAM_AVAILABLE", "DetectMixin", "ValidationUiMixin",
           "WindowPickMixin", "_draw_unicode_text",
           "_draw_unicode_texts_batch", "_get_unicode_font"]
