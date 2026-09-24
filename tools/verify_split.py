"""拆分结果门禁：当前 Mixin 包内的方法体必须与拆分前原文逐字节一致"""

import ast
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git_show(rev_path):
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", rev_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"无法读取拆分前基线 {rev_path}。\n"
            f"本脚本需要包含 P2 重构提交的 git 历史（基线：f3d4a66^ 与 ffcc902）。\n"
            f"若仓库为浅克隆或尚未包含这些提交，请先同步历史；\n"
            f"仅验证当前行为可改用 tools/verify_flow_fixes.py、tools/verify_decision_fixes.py。\n"
            f"git 报错：{proc.stderr.strip()}"
        )
    return proc.stdout


def class_nodes(src, cls_name):
    tree = ast.parse(src)
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def file_methods(text):
    """收集文件中所有类的方法（Mixin 文件名与类名一一对应，此处不关心类名）"""
    out = {}
    for cls in [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.ClassDef)]:
        for n in cls.body:
            if isinstance(n, ast.FunctionDef):
                out[n.name] = n
    return out


# 有意变更白名单：拆分后修复的延迟相对导入（绝对化）与行为缺陷修复
INTENTIONAL = {
    "add_class", "edit_class",           # main_window_mixins/class_actions_mixin.py
    "train_model", "export_model",       # main_window_mixins/model_actions_mixin.py
    "start_training",                    # train_dialog_mixins/actions_mixin.py
    "apply_theme",                       # B2: QSS 路径改用 _get_app_root()
    "load_image_folder_by_path",         # B14: 切文件夹清 undo/redo 栈
    "close_image_folder",                # B14: 关文件夹清 undo/redo 栈
    "_handle_image_after_removal",       # B3: 移除后按路径重定位当前图片
    "batch_auto_annotate",               # B5: 批量后刷新当前画布
    "load_image",                         # B6: 换图即清 undo/redo 栈
}


def check(mixin_dir, cls_name, origin_src, label):
    origin = class_nodes(origin_src, cls_name)
    seen = set()
    waived = []
    for f in sorted(mixin_dir.glob("*.py")):
        if f.name == "__init__.py":
            continue
        text = f.read_text(encoding="utf-8")
        for name, node in file_methods(text).items():
            assert name in origin, f"{f.name}: {name} 不在原文中"
            same = ast.get_source_segment(text, node) == ast.get_source_segment(
                origin_src, origin[name]
            )
            if not same:
                assert name in INTENTIONAL, f"{f.name}: {name} 与原文不一致（非白名单）"
                waived.append(name)
            seen.add(name)
    assert seen <= set(origin), (
        f"{label}: 出现原文之外的方法 {sorted(seen - set(origin))}"
    )
    print(
        f"{label}: {len(seen) - len(waived)} 个方法体与原文逐字节一致"
        f"（主文件保留 {len(origin) - len(seen)} 个；有意变更 {len(waived)} 个: {sorted(set(waived))}）"
    )


check(
    ROOT / "src/ui/main_window_mixins",
    "MainWindow",
    git_show("f3d4a66^:src/ui/main_window.py"),
    "main_window_mixins",
)
check(
    ROOT / "src/ui/train_dialog_mixins",
    "TrainDialog",
    git_show("ffcc902:src/ui/train_dialog.py"),
    "train_dialog_mixins",
)
check(
    ROOT / "src/ui/validation_dialog_mixins",
    "ValidationDialog",
    git_show("ffcc902:src/ui/validation_dialog.py"),
    "validation_dialog_mixins",
)

# 绘制工具为模块级函数，单独比对
valid_origin = git_show("ffcc902:src/ui/validation_dialog.py")
origin_funcs = {
    n.name: n
    for n in ast.parse(valid_origin).body
    if isinstance(n, ast.FunctionDef)
}
draw_text = (ROOT / "src/ui/validation_dialog_mixins/unicode_text.py").read_text(
    encoding="utf-8"
)
draw_funcs = [
    n for n in ast.parse(draw_text).body if isinstance(n, ast.FunctionDef)
]
for node in draw_funcs:
    assert ast.get_source_segment(draw_text, node) == ast.get_source_segment(
        valid_origin, origin_funcs[node.name]
    ), node.name
print(f"unicode_text.py: {len(draw_funcs)} 个绘制函数与原文逐字节一致")
print("VERIFY PASSED")
