"""⚠️ 一次性重构脚本（会直接重写 src/ui/validation_dialog.py，请勿随意执行）

P2 重构：按 AST 从 validation_dialog.py 机械抽取 Mixin 与绘制工具（方法体逐字节保留）。
基线为 ffcc902:src/ui/validation_dialog.py —— v2.4.0 已完成拆分，日常维护请改用
tools/verify_split.py 做保真校验。
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src/ui/validation_dialog.py"
OUT_DIR = ROOT / "src/ui/validation_dialog_mixins"
OUT_DIR.mkdir(exist_ok=True)

src = SRC.read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)
tree = ast.parse(src)
cls = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.ClassDef) and n.name == "ValidationDialog"
)

spans = {}
for node in cls.body:
    if isinstance(node, ast.FunctionDef):
        start = node.lineno - 1
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        spans[node.name] = (start, node.end_lineno - 1)

# ---------- 1. 模块级：Unicode 文本绘制工具（原样搬移） ----------
mod_spans = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "_get_unicode_font", "_draw_unicode_text", "_draw_unicode_texts_batch"
    ):
        mod_spans.append((node.lineno - 1, node.end_lineno - 1))
    elif isinstance(node, ast.Assign) and any(
        getattr(t, "id", "") == "_UNICODE_FONT_CACHE" for t in node.targets
    ):
        start = node.lineno - 1
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        mod_spans.append((start, node.end_lineno - 1))

mod_spans.sort()
assert len(mod_spans) == 4, f"模块级跨度数量异常: {mod_spans}"
draw_body = "\n\n".join(
    "".join(lines[s:e + 1]).rstrip("\n") for s, e in mod_spans
)

UNICODE_TEXT_HEADER = '''"""
Unicode 文本绘制工具

从 validation_dialog.py 拆分（P2 重构）：cv2.putText 无法渲染中文，
这里用 Pillow 在 OpenCV BGR 图像上绘制 Unicode 文本。
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
'''

(OUT_DIR / "unicode_text.py").write_text(
    UNICODE_TEXT_HEADER + "\n\n" + draw_body + "\n", encoding="utf-8"
)
print(f"unicode_text.py: {(UNICODE_TEXT_HEADER + draw_body).count(chr(10))} 行")

# ---------- 2. 类方法拆分 ----------
UI_HEADER = '''"""
界面构建 Mixin

从 validation_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 ValidationDialog 的属性与其他 Mixin 方法。
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSlider,
    QSplitter, QVBoxLayout, QWidget,
)

from src.utils.i18n import tr
from src.utils.widget_helpers import SliderSpinBoxBinder
'''

PICK_HEADER = '''"""
窗口/区域拾取 Mixin

从 validation_dialog.py 拆分（P2 重构）：窗口拾取（光标轮询 + 高亮）、
区域选择器与 DPI 屏幕坐标换算（含 win32 调用）。
通过 self 访问 ValidationDialog 的属性与其他 Mixin 方法。
"""

import ctypes
import time
from ctypes import wintypes
from typing import Optional, Tuple

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog

from src.ui.region_selector import RegionSelector, WindowHighlighter
from src.utils.i18n import tr
from src.utils.logger import get_logger_simple
from src.utils.win32_helpers import (
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN,
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
    VK_LBUTTON, VK_RBUTTON,
    get_user32, get_window_title, to_root_window,
)

logger = get_logger_simple(__name__)
'''

DETECT_HEADER = '''"""
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

from src.ui.validation_dialog_mixins.unicode_text import _draw_unicode_texts_batch
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
'''

GROUPS = {
    "ui_mixin": (
        "ValidationUiMixin",
        "界面构建与来源控制 Mixin：左栏控件、模型加载、参数行与来源切换",
        UI_HEADER,
        ["_init_ui", "_selected_source", "_update_model_status", "_load_model",
         "_add_slider_row", "_on_source_changed", "_browse_image"],
    ),
    "window_pick_mixin": (
        "WindowPickMixin",
        "窗口/区域拾取 Mixin：光标轮询、窗口高亮、区域选择与坐标换算",
        PICK_HEADER,
        ["_pick_window", "_stop_window_pick", "_update_window_pick",
         "_update_window_highlighter", "_cleanup_highlighter", "_pick_region",
         "_window_rect", "_get_screen_bounds", "_normalize_region",
         "_physical_to_logical_rect", "_logical_to_physical_rect"],
    ),
    "detect_mixin": (
        "DetectMixin",
        "实时检测 Mixin：相机生命周期、抓帧、推理与预览渲染",
        DETECT_HEADER,
        ["_toggle_detect", "_create_camera", "_release_camera", "_start",
         "_stop", "_on_tick", "_capture_frame", "_read_image",
         "_run_image_once", "_run_detection", "_update_preview",
         "_on_show_conf_toggled", "closeEvent"],
    ),
}

all_names = [n for _, _, _, names in GROUPS.values() for n in names]
missing = [n for n in all_names if n not in spans]
assert not missing, f"方法未找到: {missing}"
assert len(all_names) == len(set(all_names)), "方法重复分组"

OUT_DIR.mkdir(exist_ok=True)
for fname, (cls_name, doc, header, names) in GROUPS.items():
    body = "\n\n".join(
        "".join(lines[s:e + 1]).rstrip("\n")
        for s, e in (spans[n] for n in names)
    )
    content = header
    content += f"\n\nclass {cls_name}:\n    \"\"\"{doc}\"\"\"\n\n\n"
    content += body + "\n"
    (OUT_DIR / f"{fname}.py").write_text(content, encoding="utf-8")
    print(f"{fname}.py: {content.count(chr(10))} 行, {len(names)} 个方法")

(OUT_DIR / "__init__.py").write_text(
    '"""ValidationDialog 职责拆分的 Mixin 与绘制工具模块（P2 重构）"""\n\n'
    "from .detect_mixin import DXCAM_AVAILABLE, DetectMixin\n"
    "from .ui_mixin import ValidationUiMixin\n"
    "from .unicode_text import (\n"
    "    _draw_unicode_text,\n"
    "    _draw_unicode_texts_batch,\n"
    "    _get_unicode_font,\n"
    ")\n"
    "from .window_pick_mixin import WindowPickMixin\n\n"
    '__all__ = ["DXCAM_AVAILABLE", "DetectMixin", "ValidationUiMixin",\n'
    '           "WindowPickMixin", "_draw_unicode_text",\n'
    '           "_draw_unicode_texts_batch", "_get_unicode_font"]\n',
    encoding="utf-8",
)

# ---------- 3. 重写 validation_dialog.py（仅保留骨架） ----------
MAIN_HEADER = '''"""Validation dialog for realtime model verification.

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


'''


def method_text(name):
    s, e = spans[name]
    return "".join(lines[s:e + 1]).rstrip("\n")


new_src = MAIN_HEADER + method_text("__init__") + "\n"
SRC.write_text(new_src, encoding="utf-8")
print(f"validation_dialog.py rewritten: {new_src.count(chr(10))} 行")

# ---------- 校验：生成的方法体与原文逐字节一致 ----------
gen_tree = ast.parse(new_src)
gen_cls = next(
    n for n in ast.walk(gen_tree)
    if isinstance(n, ast.ClassDef) and n.name == "ValidationDialog"
)
gen_methods = {n.name for n in gen_cls.body if isinstance(n, ast.FunctionDef)}
assert gen_methods == {"__init__"}, gen_methods

for fname, (cls_name, doc, header, names) in GROUPS.items():
    gen = (OUT_DIR / f"{fname}.py").read_text(encoding="utf-8")
    gen_cls2 = next(
        n for n in ast.walk(ast.parse(gen))
        if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    for node in gen_cls2.body:
        if isinstance(node, ast.FunctionDef):
            orig_node = next(
                n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == node.name
            )
            assert ast.get_source_segment(gen, node) == ast.get_source_segment(
                src, orig_node
            ), node.name

gen_draw = (OUT_DIR / "unicode_text.py").read_text(encoding="utf-8")
gen_draw_tree = ast.parse(gen_draw)
orig_top = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "_get_unicode_font", "_draw_unicode_text", "_draw_unicode_texts_batch"
    ):
        orig_top[node.name] = node
    elif isinstance(node, ast.Assign) and any(
        getattr(t, "id", "") == "_UNICODE_FONT_CACHE" for t in node.targets
    ):
        orig_top["_UNICODE_FONT_CACHE"] = node
for node in gen_draw_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in orig_top:
        assert ast.get_source_segment(gen_draw, node) == ast.get_source_segment(
            src, orig_top[node.name]
        ), node.name
print("校验通过：所有方法体与绘制工具与拆分前原文逐字节一致")

