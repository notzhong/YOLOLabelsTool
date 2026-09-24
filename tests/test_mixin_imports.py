"""Mixin 拆分的导入卫生守卫（回归 f3d4a66/56b5234 的相对导入断链问题）

Mixin 文件位于 `src/ui/<xxx>_mixins/` 包内，函数体里延迟写 `from .dialog import ...`
会在运行时解析为 `<xxx>_mixins.dialog` → ModuleNotFoundError（新增类别/训练/导出
按钮点击即崩）。本测试为纯文本检查，无需 Qt，可在 CI 直接运行。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MIXIN_PACKAGES = [
    "main_window_mixins",
    "train_dialog_mixins",
    "validation_dialog_mixins",
]

# 匹配 `from .module import ...`（排除 `from .__init__` 与 `from ..`，均不适用此处）
_RELATIVE_IMPORT = re.compile(r"^from\s+\.\w")


def _relative_imports_in(pkg: str):
    offenders = []
    pkg_dir = ROOT / "src" / "ui" / pkg
    assert pkg_dir.is_dir(), f"目录不存在: {pkg_dir}"
    for file in sorted(pkg_dir.glob("*.py")):
        if file.name == "__init__.py":
            continue  # 包内互相导入相对写法正确
        for lineno, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if _RELATIVE_IMPORT.match(stripped):
                rel = file.relative_to(ROOT)
                offenders.append(f"{rel}:{lineno}: {stripped}")
    return offenders


def test_mixin_modules_have_no_relative_imports():
    """Mixin 文件一律使用绝对导入（拆分后包路径已变，相对导入必然断链）"""
    offenders = []
    for pkg in MIXIN_PACKAGES:
        offenders.extend(_relative_imports_in(pkg))
    assert not offenders, (
        "Mixin 中发现相对导入，会解析到不存在的模块并导致运行时 "
        "ModuleNotFoundError:\n" + "\n".join(offenders)
    )


def test_known_absolute_imports_resolve():
    """关键延迟导入路径保持可解析（与运行时相同的模块路径）"""
    import importlib

    for module in (
        "src.ui.class_dialog",
        "src.ui.train_dialog",
        "src.ui.export_dialog",
        "src.ui.train_progress_dialog",
        "src.ui.validation_dialog",
    ):
        # 只验证模块路径可解析，不触发 Qt 导入（spec 不执行模块）
        assert importlib.util.find_spec(module) is not None, module
