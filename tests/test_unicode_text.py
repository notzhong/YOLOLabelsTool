"""src/utils/unicode_text.py 单元测试（纯函数，无 Qt 依赖，可在 CI 无 PySide6 环境运行）"""

import numpy as np

from src.utils.unicode_text import (
    _UNICODE_FONT_CACHE,
    _draw_unicode_text,
    _draw_unicode_texts_batch,
    _get_unicode_font,
)


def test_get_unicode_font_uses_cache_on_size_match():
    class _FakeFont:
        size = 12

    _UNICODE_FONT_CACHE[0] = None
    fake = _FakeFont()
    _UNICODE_FONT_CACHE[0] = fake
    try:
        assert _get_unicode_font(12) is fake
        # size 不匹配时重新加载（不依赖平台是否有中文字体）
        assert _get_unicode_font(13) is not fake
    finally:
        _UNICODE_FONT_CACHE[0] = None


def test_get_unicode_font_returns_usable_font():
    _UNICODE_FONT_CACHE[0] = None
    try:
        font = _get_unicode_font(16)
        assert hasattr(font, "getbbox")
    finally:
        _UNICODE_FONT_CACHE[0] = None


def test_draw_unicode_texts_batch_empty_is_noop():
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    _draw_unicode_texts_batch(img, [])
    assert img.sum() == 0


def test_draw_unicode_text_draws_pixels():
    img = np.zeros((60, 240, 3), dtype=np.uint8)
    _draw_unicode_text(img, "car 0.95", (5, 30), 0.5, (0, 255, 0), 1)
    assert img.sum() > 0


def test_draw_unicode_texts_batch_multiple_items():
    img = np.zeros((90, 240, 3), dtype=np.uint8)
    _draw_unicode_texts_batch(
        img,
        [
            ("person 0.95", (5, 30), 0.5, (0, 255, 0), 1),
            ("", (5, 60), 0.5, (255, 0, 0), 1),
            ("dog 0.12", (5, 90), 1.0, (0, 0, 255), 2),
        ],
    )
    assert img.sum() > 0
