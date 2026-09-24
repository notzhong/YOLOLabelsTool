"""⚠️ 一次性重构脚本（会直接重写 src/ui/main_window_mixins/*.py，请勿随意执行）

修正 f3d4a66 拆分 main_window.py 时的重复空行：用拆分前原文
（f3d4a66^:src/ui/main_window.py）重新生成方法体。v2.4.0 已修正，日常维护请改用
tools/verify_split.py 做保真校验。
"""

import ast
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src/ui/main_window_mixins"

original = subprocess.run(
    ["git", "-C", str(ROOT), "show", "f3d4a66^:src/ui/main_window.py"],
    capture_output=True, text=True, check=True,
).stdout
lines = original.splitlines(keepends=True)
tree = ast.parse(original)
cls = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.ClassDef) and n.name == "MainWindow"
)
spans = {}
for node in cls.body:
    if isinstance(node, ast.FunctionDef):
        start = node.lineno - 1
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        spans[node.name] = (start, node.end_lineno - 1)

GROUPS = {
    "theme_language_mixin": (
        "ThemeLanguageMixin",
        ["load_qss_style", "apply_theme", "update_title_colors_for_theme",
         "switch_to_dark_theme", "switch_to_light_theme", "switch_to_colorful_theme",
         "switch_to_eyecare_theme", "switch_language", "update_ui_texts",
         "update_menu_texts", "update_button_texts", "update_panel_titles",
         "update_other_ui_elements"],
    ),
    "panels_mixin": (
        "PanelsMixin",
        ["create_left_panel", "create_center_panel", "create_right_panel",
         "_load_splitter_sizes"],
    ),
    "image_actions_mixin": (
        "ImageActionsMixin",
        ["load_image_folder", "load_image_folder_by_path", "close_image_folder",
         "load_image", "update_image_list", "update_image_info", "update_stats",
         "update_statistics_panel", "on_image_item_clicked",
         "_on_image_list_context_menu", "_delete_all_unannotated_images",
         "_remove_selected_images", "_remove_all_unannotated_images",
         "_handle_image_after_removal", "prev_image", "next_image",
         "fit_to_window", "zoom_in", "zoom_out", "reset_view",
         "_update_scale_status"],
    ),
    "class_actions_mixin": (
        "ClassActionsMixin",
        ["_select_next_class", "_select_prev_class", "update_class_list",
         "_on_class_item_double_clicked", "_on_class_list_context_menu",
         "on_class_item_clicked", "add_class", "edit_class", "delete_class",
         "clear_all_classes", "load_classes_from_yaml", "save_classes_to_yaml"],
    ),
    "model_actions_mixin": (
        "ModelActionsMixin",
        ["load_model", "show_model_info", "open_validation_window",
         "auto_annotate_current", "batch_auto_annotate", "unload_model",
         "update_model_info_panel", "train_model", "export_model"],
    ),
}

for fname, (cls_name, names) in GROUPS.items():
    path = PKG / f"{fname}.py"
    content = path.read_text(encoding="utf-8")
    body = "\n\n".join(
        "".join(lines[s:e + 1]).rstrip("\n")
        for s, e in (spans[n] for n in names)
    )
    file_lines = content.splitlines(keepends=True)
    idx = next(i for i, ln in enumerate(file_lines) if ln.startswith(f"class {cls_name}:"))
    prefix = file_lines[:idx + 2]  # class 行 + 类 docstring 行
    while prefix and prefix[-1].strip() == "":
        prefix.pop()
    new = "".join(prefix) + "\n\n\n" + body + "\n"
    path.write_text(new, encoding="utf-8")
    print(f"{fname}.py: {new.count(chr(10))} 行, {len(names)} 个方法")

    # 校验：方法体与拆分前原文逐字节一致
    gen = path.read_text(encoding="utf-8")
    gen_cls = next(
        n for n in ast.walk(ast.parse(gen))
        if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    gen_names = [n.name for n in gen_cls.body if isinstance(n, ast.FunctionDef)]
    assert gen_names == names, (fname, gen_names)
    for node in gen_cls.body:
        if isinstance(node, ast.FunctionDef):
            orig_node = next(
                n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == node.name
            )
            assert ast.get_source_segment(gen, node) == ast.get_source_segment(
                original, orig_node
            ), f"{fname}:{node.name}"

print("校验通过：main_window_mixins 方法体与拆分前原文逐字节一致")
