"""AST 检查：类（含 Mixin 组合）中"只读未写"的 self 属性 → AttributeError 高危点"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def check(class_files, cls_name, label):
    written, read = set(), set()
    for f in class_files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name
            ) and node.value.id == "self":
                # 赋值目标 / aug 赋值 / 带删除 → written；仅 Load → read
                if isinstance(node.ctx, ast.Store):
                    written.add(node.attr)
                elif isinstance(node.ctx, ast.Del):
                    written.add(node.attr)
                else:
                    read.add(node.attr)
            # 形参 self.xxx 形式（__init__ 入参默认无此形态，忽略）
    # 也把 __init__ 里的 self.x = ... 已覆盖（Store）
    # 通过 super().__init__ / 父类（QDialog/QMainWindow）提供的属性白名单
    qt_provided = {
        "windowTitle", "setWindowTitle", "setModal", "resize", "parent",
        "close", "show", "hide", "setEnabled", "isVisible", "raise_",
        "activateWindow", "winId", "setStyleSheet", "setFont", "setCursor",
        "addAction", "setToolTip", "setContentsMargins", "layout", "setModal",
        "update", "repaint", "setMinimumSize", "setMaximumSize", "setFixedSize",
        "setWindowFlags", "setWindowIcon", "setCentralWidget", "menuBar",
        "statusBar", "setStatusBar", "setGeometry", "move", "pos", "size",
        "setAccessibleName", "showFullScreen", "setStyleSheet",
    }
    # 类体内的 class 属性（SOURCE_* 之类）也算写入
    src_all = ""
    for f in class_files:
        src_all += f.read_text(encoding="utf-8")
    tree = ast.parse(src_all)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            written.add(t.id)
                elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    written.add(item.target.id)
                elif isinstance(item, ast.AugAssign) and isinstance(
                    item.target, ast.Name
                ):
                    written.add(item.target.id)

    suspects = sorted(read - written - qt_provided)
    print(f"[{label}] 只读未写属性 {len(suspects)} 个:")
    for name in suspects:
        print(f"    self.{name}")
    return suspects


def main():
    groups = {
        "MainWindow": (
            [ROOT / "src/ui/main_window.py",
             *sorted((ROOT / "src/ui/main_window_mixins").glob("*.py"))],
            "MainWindow",
        ),
        "TrainDialog": (
            [ROOT / "src/ui/train_dialog.py",
             *sorted((ROOT / "src/ui/train_dialog_mixins").glob("*.py"))],
            "TrainDialog",
        ),
        "ValidationDialog": (
            [ROOT / "src/ui/validation_dialog.py",
             *sorted((ROOT / "src/ui/validation_dialog_mixins").glob("*.py"))],
            "ValidationDialog",
        ),
        "ExportDialog": (
            [ROOT / "src/ui/export_dialog.py"], "ExportDialog",
        ),
    }
    total = 0
    for label, (files, cls) in groups.items():
        total += len(check(files, cls, label))
    print(f"合计疑似点: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
