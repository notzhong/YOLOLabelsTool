"""未加载图片时进行标注操作的守卫回归（纯静态检查，无 Qt 依赖）

回归场景：启动后不加载任何图片，直接在画布上拖拽画框 →
`TypeError: argument should be a str or an os.PathLike object ... not 'NoneType'`
（栈底是 `Path(image_path).stem`，image_path 为 None）。

本次修复分三层，每层都靠"人肉记得加守卫"维持，一但有人删掉就会静默复发，
而且复发点全在 Qt 代码里——CI 环境没装 PySide6，跑不了真实 UI 测试。
所以这里用静态文本检查把守卫钉住，风格对齐 test_mixin_imports.py：
**不 import src.ui**，只读源码做断言。

被钉住的三层：
  1. canvas.mousePressEvent / mouseReleaseEvent —— 无图不进绘制
  2. main_window.add_annotation_with_command / _on_canvas_annotation_created —— 无图直接返回
  3. 画布状态复位 —— 走 canvas.clear_image()，而不是裸 _scene.clear()
     （裸 clear 会残留 _image_item，导致 has_image 永远为真）
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CANVAS = ROOT / "src" / "ui" / "annotation_canvas.py"
MAIN_WINDOW = ROOT / "src" / "ui" / "main_window.py"
IMAGE_ACTIONS = ROOT / "src" / "ui" / "main_window_mixins" / "image_actions_mixin.py"
PANELS = ROOT / "src" / "ui" / "main_window_mixins" / "panels_mixin.py"
ZH_INI = ROOT / "translations" / "zh_CN.ini"
EN_INI = ROOT / "translations" / "en_US.ini"

# 匹配函数体：从 `def name(` 到下一个同级 `def ` 之前
_FUNC_RE_CACHE = {}


def _class_body(source: str, class_name: str) -> str:
    """取出指定 class 的源码片段（到下一个顶层 class 之前）

    必须先按类收窄：AnnotationRectItem 与 AnnotationCanvas 都有
    mousePressEvent / mouseReleaseEvent，且同为 4 空格缩进，
    不做类限定会取到前一个类的同名方法。
    """
    key = f"__class__{class_name}"
    if key not in _FUNC_RE_CACHE:
        _FUNC_RE_CACHE[key] = re.compile(
            rf"^class {re.escape(class_name)}\b.*?(?=^class |\Z)", re.S | re.M
        )
    match = _FUNC_RE_CACHE[key].search(source)
    assert match, f"未找到 class {class_name}（类名可能被改动，请同步本守卫）"
    return match.group(0)


def _func_body(source: str, name: str, class_name: str = None) -> str:
    """取出指定函数的源码片段（到下一个顶层 def/装饰器之前）

    class_name 给定时，只在类内部查找（处理同名方法歧义）。
    """
    scope = _class_body(source, class_name) if class_name else source
    cache_key = f"{class_name or ''}.{name}"
    if cache_key not in _FUNC_RE_CACHE:
        _FUNC_RE_CACHE[cache_key] = re.compile(
            rf"^    def {re.escape(name)}\(.*?(?=^    def |^    @|\Z)", re.S | re.M
        )
    match = _FUNC_RE_CACHE[cache_key].search(scope)
    assert match, f"未找到函数 {cache_key}（签名可能被改动，请同步本守卫）"
    return match.group(0)


# ---- 第 1 层：画布不响应无图时的绘制手势 ----


class TestCanvasDrawingGuard:
    @classmethod
    @pytest.fixture(scope="class")
    def canvas_src(cls) -> str:
        return CANVAS.read_text(encoding="utf-8")

    def test_canvas_exposes_has_image(self, canvas_src):
        """画布必须对外暴露 has_image 判据（唯一可信的"有图"来源）"""
        assert re.search(r"def has_image\(", canvas_src), (
            "AnnotationCanvas 缺少 has_image 属性；各守卫将退回到直接读 "
            "_image_item，而它会随 _scene.clear() 残留而说谎"
        )

    def test_canvas_exposes_clear_image(self, canvas_src):
        assert re.search(r"def clear_image\(", canvas_src), (
            "AnnotationCanvas 缺少 clear_image()；调用方会退回裸 "
            "_scene.clear()，导致 _image_item 残留"
        )

    def test_clear_image_resets_image_item(self, canvas_src):
        body = _func_body(canvas_src, "clear_image", class_name="AnnotationCanvas")
        assert "_image_item = None" in body, (
            "clear_image() 必须把 _image_item 复位为 None，"
            "否则关夹后 has_image 仍为真、又能拖出标注"
        )

    def test_mouse_press_refuses_without_image(self, canvas_src):
        body = _func_body(
            canvas_src, "mousePressEvent", class_name="AnnotationCanvas"
        )
        assert "has_image" in body, (
            "mousePressEvent 未检查 has_image：无图时仍会进入绘制，"
            "用户能拖出矩形，最终在 main_window 侧崩溃"
        )

    def test_mouse_release_refuses_without_image(self, canvas_src):
        body = _func_body(
            canvas_src, "mouseReleaseEvent", class_name="AnnotationCanvas"
        )
        assert "has_image" in body, (
            "mouseReleaseEvent 未检查 has_image：应在发出 "
            "annotation_created 前再兜一层"
        )

    def test_display_image_sets_image_item(self, canvas_src):
        body = _func_body(canvas_src, "display_image", class_name="AnnotationCanvas")
        assert "_image_item = QGraphicsPixmapItem" in body, (
            "display_image 必须设置 _image_item，否则 has_image 永远为假，"
            "正常图片也无法标注"
        )


# ---- 第 2 层：主窗口槽位在无图时直接返回 ----


class TestMainWindowSlotGuard:
    @classmethod
    @pytest.fixture(scope="class")
    def main_src(cls) -> str:
        return MAIN_WINDOW.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "func",
        [
            "_on_canvas_annotation_created",
            "add_annotation_with_command",
        ],
    )
    def test_annotation_slot_guards_current_image_path(self, main_src, func):
        """两个标注入口都必须先判断 current_image_path 再往下走

        本 bug 的根因就是 _on_canvas_annotation_created 漏了这道判断，
        而它的 7 个同类槽位（delete/save/clear/export...）都有。
        """
        body = _func_body(main_src, func)
        assert re.search(r"if not self\.current_image_path:", body), (
            f"{func} 缺少 `if not self.current_image_path:` 守卫；"
            "无图时 current_image_path 为 None，会一路炸到 Path(None)"
        )
        # 守卫之后必须真的 return，而不是继续执行
        assert re.search(
            r"if not self\.current_image_path:.*?\n\s+return", body, re.S
        ), f"{func} 的守卫没有 return，等于没拦"

    def test_blocked_annotation_logs_warning(self, main_src):
        """被拦下的请求要留日志：否则用户侧"点了没反应"无从排查"""
        body = _func_body(main_src, "add_annotation_with_command")
        assert "logger.warning" in body or "logger.error" in body, (
            "add_annotation_with_command 拦截时未记录日志；"
            "静默 return 会让该缺陷无从诊断"
        )


# ---- 第 3 层：所有"不再显示图片"的路径都走 clear_image() ----


class TestCanvasStateReset:
    @classmethod
    @pytest.fixture(scope="class")
    def actions_src(cls) -> str:
        return IMAGE_ACTIONS.read_text(encoding="utf-8")

    def test_no_raw_scene_clear_left(self, actions_src):
        """不得再裸调 canvas._scene.clear()（那是本 bug 的帮凶之一）"""
        offenders = [
            line.strip()
            for line in actions_src.splitlines()
            if "canvas._scene.clear()" in line
        ]
        assert not offenders, (
            "image_actions_mixin 仍直接清场景，会残留 _image_item：\n"
            + "\n".join(offenders)
            + "\n请改用 canvas.clear_image()"
        )

    @pytest.mark.parametrize(
        "func",
        ["close_image_folder", "_handle_image_after_removal"],
    )
    def test_reset_paths_use_clear_image(self, actions_src, func):
        body = _func_body(actions_src, func)
        assert "self.canvas.clear_image()" in body, (
            f"{func} 未调用 canvas.clear_image()，画布状态将与 "
            "current_image_path 不一致"
        )

    def test_failed_pixmap_load_rolls_back_current_image_path(self, actions_src):
        """解码失败的图片必须回滚 current_image_path，而不是留着半同步状态"""
        body = _func_body(actions_src, "load_image")
        assert body.count("self.current_image_path = None") >= 1, (
            "load_image 在 pixmap.isNull() 分支未把 current_image_path 置回 "
            "None：会在「有当前图片」与「画布无图」之间留下矛盾状态"
        )


# ---- 第 4 层：用户反馈与翻译完整性 ----
#
# 被拦下的标注必须给用户一句可见反馈，否则"拖了没反应"无从得知原因。
# 注意 CI 不校验真实 translations/（tests/test_i18n.py 用 monkeypatch 的临时
# 目录），所以 key 只加一侧 INI 不会有任何测试报警 —— 这里补上这道守卫。

NEW_KEYS = ["canvas_no_image_hint", "annotation_ignored_no_image"]


def _ini_keys(path: Path) -> set:
    """取出 INI [translations] 段的所有 key"""
    text = path.read_text(encoding="utf-8")
    keys = set()
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == "[translations]"
            continue
        if in_section and stripped and not stripped.startswith("#") and "=" in stripped:
            keys.add(stripped.split("=", 1)[0].strip())
    return keys


class TestUserFeedbackAndI18n:
    @classmethod
    @pytest.fixture(scope="class")
    def canvas_src(cls) -> str:
        return CANVAS.read_text(encoding="utf-8")

    @classmethod
    @pytest.fixture(scope="class")
    def main_src(cls) -> str:
        return MAIN_WINDOW.read_text(encoding="utf-8")

    def test_canvas_emits_status_message_on_blocked_drag(self, canvas_src):
        body = _func_body(
            canvas_src, "mousePressEvent", class_name="AnnotationCanvas"
        )
        assert "status_message.emit" in body, (
            "无图时画布只静默 return，用户看不到任何反馈（点了没反应）；"
            "应通过 status_message 信号给状态栏一句提示"
        )
        assert "canvas_no_image_hint" in body, (
            "画布提示文案未走 tr 翻译，将硬编码中文，英文界面显示中文"
        )

    def test_status_message_signal_is_connected(self):
        """status_message 此前定义了但从未连接 —— 发什么都无人接收"""
        src = PANELS.read_text(encoding="utf-8")
        assert "canvas.status_message.connect" in src, (
            "AnnotationCanvas.status_message 未连接到主窗口：画布发出的"
            "状态提示不会显示在状态栏上"
        )

    def test_slot_reports_status_on_blocked_annotation(self, main_src):
        body = _func_body(main_src, "add_annotation_with_command")
        assert "update_status" in body, (
            "add_annotation_with_command 拦截时未更新状态栏，用户无感知"
        )
        assert "annotation_ignored_no_image" in body, (
            "槽位提示文案未走 tr 翻译"
        )

    def test_blocked_annotation_still_logs(self, main_src):
        """状态栏是给用户的；日志是给排查的，两者不能互相替代"""
        body = _func_body(main_src, "add_annotation_with_command")
        assert "logger.warning" in body or "logger.error" in body

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_new_keys_exist_in_both_ini_files(self, key):
        """新 key 必须同时存在于 zh_CN 与 en_US，否则英文界面回退成中文"""
        for ini in (ZH_INI, EN_INI):
            assert key in _ini_keys(ini), (
                f"{ini.name} 缺少 key {key} —— i18n 回退链会退到 en_US 或 "
                "直接显示 key 本身"
            )

    def test_new_keys_are_non_empty_in_both_languages(self):
        """空值等于没有翻译"""
        for ini in (ZH_INI, EN_INI):
            text = ini.read_text(encoding="utf-8")
            for key in NEW_KEYS:
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith(f"{key} ="):
                        value = stripped.split("=", 1)[1].strip()
                        assert value, f"{ini.name} 的 {key} 值为空"
                        break

