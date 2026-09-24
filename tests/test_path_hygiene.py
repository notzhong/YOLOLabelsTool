"""路径卫生守卫：应用数据路径禁止使用 CWD 相对字面量（回归 B2）

跨 CWD 启动（桌面快捷方式 / `python /path/main.py`）时，CWD 相对路径会把
配置/QSS/图标/标注数据写到错误位置：设置"失忆"、主题失效、标注看似丢失。
统一使用 src.utils.logger._get_app_root() 锚定应用根目录。
纯文本检查，无需 Qt，可在 CI 直接运行。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# CWD 相对字面量 → 必须改为 _get_app_root() 锚定
FORBIDDEN = [
    (re.compile(r"""Path\(\s*['"]config/"""), "config/ 配置路径"),
    (re.compile(r"""Path\(\s*['"]qss/"""), "qss/ 主题路径"),
    (re.compile(r"""Path\(\s*['"]icon\.ico"""), "icon.ico 图标路径"),
    (
        re.compile(r"""_annotation_dir\s*=\s*['"]annotations['"]"""),
        "annotations 标注目录",
    ),
]


def _iter_source_files():
    yield ROOT / "main.py"
    for directory in (ROOT / "src", ROOT / "yolo_tool"):
        yield from sorted(directory.rglob("*.py"))


def test_no_cwd_relative_data_paths():
    offenders = []
    for file in _iter_source_files():
        text = file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern, why in FORBIDDEN:
                if pattern.search(line):
                    rel = file.relative_to(ROOT)
                    offenders.append(f"{rel}:{lineno}: {why} -> {line.strip()}")
    assert not offenders, "发现 CWD 相对数据路径：\n" + "\n".join(offenders)
