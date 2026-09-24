"""ValidationDialog 职责拆分的 Mixin 模块（P2 重构）

绘制工具（Unicode 文本）已迁至 Qt-free 的 src/utils/unicode_text.py，
此处为兼容既有导入路径做再导出。
"""

from .detect_mixin import DXCAM_AVAILABLE, DetectMixin
from .ui_mixin import ValidationUiMixin
from .window_pick_mixin import WindowPickMixin

from src.utils.unicode_text import (
    _draw_unicode_text,
    _draw_unicode_texts_batch,
    _get_unicode_font,
)

__all__ = ["DXCAM_AVAILABLE", "DetectMixin", "ValidationUiMixin",
           "WindowPickMixin", "_draw_unicode_text",
           "_draw_unicode_texts_batch", "_get_unicode_font"]
