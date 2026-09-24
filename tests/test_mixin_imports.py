"""Mixin 拆分的导入卫生守卫（回归 f3d4a66/56b5234 的相对导入断链问题）

Mixin 文件位于 `src/ui/<xxx>_mixins/` 包内，函数体里延迟写 `from .dialog import ...`
会在运行时解析为 `<xxx>_mixins.dialog` → ModuleNotFoundError（新增类别/训练/导出
按钮点击即崩）。本测试为纯静态检查，**不 import src.ui**（避免拖入 PySide6，
保证无 Qt 环境——如 CI——可直接运行）。
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


UI_FILES = {
    "src/ui/main_window_mixins/class_actions_mixin.py": ("src.ui.class_dialog",),
    "src/ui/main_window_mixins/model_actions_mixin.py": (
        "src.ui.train_dialog",
        "src.ui.export_dialog",
    ),
    "src/ui/train_dialog_mixins/actions_mixin.py": ("src.ui.train_progress_dialog",),
}


def test_mixin_absolute_imports_point_to_existing_files():
    """Mixin 延迟导入的绝对模块路径必须真实存在。

    静态校验（不 import src.ui，避免拖入 PySide6 破坏无 Qt 环境），
    比 find_spec 更严格：同时校验引用写法与文件落位一致。
    """
    for rel, modules in UI_FILES.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for module in modules:
            assert f"from {module} import" in text, f"{rel} 未引用 {module}"
            target = ROOT / (module.replace(".", "/") + ".py")
            assert target.is_file(), f"{rel} 引用的 {module} 不存在（{target}）"
