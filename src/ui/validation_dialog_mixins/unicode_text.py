"""
Unicode 文本绘制工具

从 validation_dialog.py 拆分（P2 重构）：cv2.putText 无法渲染中文，
这里用 Pillow 在 OpenCV BGR 图像上绘制 Unicode 文本。
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Unicode font fallback chain for cv2.putText (which can't handle Chinese)
_UNICODE_FONT_CACHE = [None]

def _get_unicode_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a PIL font capable of rendering Unicode (Chinese) text."""
    if _UNICODE_FONT_CACHE[0] is not None and _UNICODE_FONT_CACHE[0].size == size:
        return _UNICODE_FONT_CACHE[0]
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",          # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",         # SimHei
        "C:/Windows/Fonts/msyhbd.ttc",         # Microsoft YaHei Bold
        "C:/Windows/Fonts/yahei.ttf",          # YaHei fallback
    ]
    font = None
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _UNICODE_FONT_CACHE[0] = font
    return font

def _draw_unicode_text(img: np.ndarray, text: str, pos, font_scale: float,
                       color, thickness: int):
    """Draw text (including Unicode) on an OpenCV BGR image using Pillow."""
    _draw_unicode_texts_batch(img, [(text, pos, font_scale, color, thickness)])

def _draw_unicode_texts_batch(img: np.ndarray,
                              items: list):
    """Batch-draw multiple Unicode text labels in a single PIL round-trip.

    Each item: (text, pos, font_scale, color, thickness).
    """
    if not items:
        return

    # Convert OpenCV BGR to PIL RGB once
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    for text, pos, font_scale, color, thickness in items:
        if not text:
            continue
        font_size = max(10, int(font_scale * 20))
        font = _get_unicode_font(font_size)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        tx, ty = pos[0], pos[1] - th - 2
        bg = color[::-1] if isinstance(color, tuple) and len(color) == 3 else (0, 255, 0)
        draw.rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], fill=bg)
        draw.text((tx, ty), text, font=font, fill=(0, 0, 0))

    # Convert back to BGR once
    img[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
