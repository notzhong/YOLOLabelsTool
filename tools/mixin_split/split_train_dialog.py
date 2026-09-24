"""⚠️ 一次性重构脚本（会直接重写 src/ui/train_dialog.py，请勿随意执行）

P2 重构：按 AST 从 train_dialog.py 机械抽取 4 个 Mixin（方法体逐字节保留）。
基线为 ffcc902:src/ui/train_dialog.py —— v2.4.0 已完成拆分，日常维护请改用
tools/verify_split.py 做保真校验。
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src/ui/train_dialog.py"
OUT_DIR = ROOT / "src/ui/train_dialog_mixins"

src = SRC.read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)
tree = ast.parse(src)
cls = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.ClassDef) and n.name == "TrainDialog"
)

spans = {}
for node in cls.body:
    if isinstance(node, ast.FunctionDef):
        start = node.lineno - 1
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        spans[node.name] = (start, node.end_lineno - 1)

TABS_HEADER = '''"""
标签页构建 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from src.utils.i18n import tr
'''

BROWSE_HEADER = '''"""
文件浏览与开关联动 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QLineEdit

from src.utils.i18n import tr
'''

CONFIG_HEADER = '''"""
配置收集与持久化 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

import json
from pathlib import Path
from typing import Any, Dict

from PySide6.QtWidgets import QMessageBox

from src.utils.i18n import tr
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)
'''

ACTIONS_HEADER = '''"""
对话框生命周期与训练启动 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

from PySide6.QtWidgets import QMessageBox

from src.utils.i18n import tr
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)
'''

GROUPS = {
    "tabs_mixin": (
        "TrainTabsMixin",
        "标签页构建 Mixin：五个参数标签页的控件创建",
        TABS_HEADER,
        ["create_basic_tab", "create_params_tab", "create_optimizer_tab",
         "create_augment_tab", "create_advanced_tab"],
    ),
    "browse_mixin": (
        "TrainBrowseMixin",
        "文件浏览 Mixin：模型/数据/输出/检查点路径选择与恢复-增量互斥开关",
        BROWSE_HEADER,
        ["on_resume_toggled", "on_incremental_toggled", "on_augment_toggled",
         "_browse_open_file", "_browse_directory", "browse_model_file",
         "browse_data_yaml", "browse_output_dir", "browse_resume_file",
         "browse_incremental_file"],
    ),
    "config_mixin": (
        "TrainConfigMixin",
        "配置读写 Mixin：UI 与配置字典互转、校验、INI 自动保存与信号连接",
        CONFIG_HEADER,
        ["save_config", "load_config", "collect_config_from_ui",
         "load_config_to_ui", "validate_config", "load_last_config",
         "save_last_config", "connect_config_change_signals",
         "on_config_changed", "save_config_on_exit"],
    ),
    "actions_mixin": (
        "TrainActionsMixin",
        "对话生命周期与训练启动 Mixin：关闭/取消/默认值重置/启动训练",
        ACTIONS_HEADER,
        ["closeEvent", "reject", "accept", "log_message", "reset_to_defaults",
         "start_training"],
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
    '"""TrainDialog 职责拆分的 Mixin 模块（P2 重构）"""\n\n'
    "from .actions_mixin import TrainActionsMixin\n"
    "from .browse_mixin import TrainBrowseMixin\n"
    "from .config_mixin import TrainConfigMixin\n"
    "from .tabs_mixin import TrainTabsMixin\n\n"
    '__all__ = ["TrainActionsMixin", "TrainBrowseMixin", "TrainConfigMixin",\n'
    '           "TrainTabsMixin"]\n',
    encoding="utf-8",
)

# ---------- 重写 train_dialog.py（仅保留骨架） ----------
MAIN_HEADER = '''"""
模型训练配置对话框

P2 重构：按职责拆分至 src/ui/train_dialog_mixins/ 包，
本模块仅保留对话框骨架（__init__ / init_ui）与对外类名 TrainDialog。
"""

import configparser
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from yolo_tool import YOLOTrainer

from src.ui.train_dialog_mixins import (
    TrainActionsMixin,
    TrainBrowseMixin,
    TrainConfigMixin,
    TrainTabsMixin,
)
from src.utils.i18n import tr


class TrainDialog(
    TrainTabsMixin,
    TrainBrowseMixin,
    TrainConfigMixin,
    TrainActionsMixin,
    QDialog,
):
    """模型训练配置对话框"""


'''


def method_text(name):
    s, e = spans[name]
    return "".join(lines[s:e + 1]).rstrip("\n")


new_src = MAIN_HEADER + "\n".join(
    method_text(n) for n in ["__init__", "init_ui"]
) + "\n"
SRC.write_text(new_src, encoding="utf-8")
print(f"train_dialog.py rewritten: {new_src.count(chr(10))} 行")

# ---------- 校验：生成的方法体与原文逐字节一致 ----------
gen_tree = ast.parse(new_src)
gen_cls = next(
    n for n in ast.walk(gen_tree)
    if isinstance(n, ast.ClassDef) and n.name == "TrainDialog"
)
gen_methods = {n.name for n in gen_cls.body if isinstance(n, ast.FunctionDef)}
assert gen_methods == {"__init__", "init_ui"}, gen_methods

for fname, (cls_name, doc, header, names) in GROUPS.items():
    gen = (OUT_DIR / f"{fname}.py").read_text(encoding="utf-8")
    gen_cls2 = next(
        n for n in ast.walk(ast.parse(gen))
        if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    orig_cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "TrainDialog"
    )
    for node in gen_cls2.body:
        if isinstance(node, ast.FunctionDef):
            orig_node = next(
                n for n in orig_cls.body
                if isinstance(n, ast.FunctionDef) and n.name == node.name
            )
            assert ast.get_source_segment(gen, node) == ast.get_source_segment(
                src, orig_node
            ), node.name
print("校验通过：所有方法体与拆分前原文逐字节一致")
