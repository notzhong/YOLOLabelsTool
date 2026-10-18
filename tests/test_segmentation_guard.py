"""分割标注（多边形 / YOLO-seg）功能守卫（纯静态检查，无 Qt 依赖）

2.5.0 新增分割标注：画布绘制/编辑多边形、任务模式切换、YOLO 标签导入、
框↔多边形互转、导出时临时转换。这些能力横跨画布事件层、窗口 Mixin 与
配置持久化，任何一处接线被删/改都会静默失效（UI 行为测试只在
tools/verify_polygon_flow.py 里跑，CI 不执行），所以用静态文本检查把
关键接线钉住，风格对齐 test_no_image_annotation_guard.py：
**不 import src.ui**，只读源码做断言。

钉住的四层：
  1. 画布：多边形图形项事件（手势捕获=一步撤销）、绘制状态机、
     最少 3 顶点守卫、文本标签点击穿透
  2. 主窗口：任务模式持久化、导出 task 透传与临时转换确认、Ctrl+Z 拦截
  3. Mixin 装配：SegmentationMixin 注册与各 Mixin 接线
  4. i18n：分割功能涉及的所有 tr() key 在中英文文件中必须双侧齐全
"""

import configparser
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CANVAS = ROOT / "src" / "ui" / "annotation_canvas.py"
MAIN_WINDOW = ROOT / "src" / "ui" / "main_window.py"
MIXINS_INIT = ROOT / "src" / "ui" / "main_window_mixins" / "__init__.py"
SEGMENTATION_MIXIN = ROOT / "src" / "ui" / "main_window_mixins" / "segmentation_mixin.py"
PANELS = ROOT / "src" / "ui" / "main_window_mixins" / "panels_mixin.py"
IMAGE_ACTIONS = ROOT / "src" / "ui" / "main_window_mixins" / "image_actions_mixin.py"
MODEL_ACTIONS = ROOT / "src" / "ui" / "main_window_mixins" / "model_actions_mixin.py"
THEME_LANGUAGE = ROOT / "src" / "ui" / "main_window_mixins" / "theme_language_mixin.py"
ZH_INI = ROOT / "translations" / "zh_CN.ini"
EN_INI = ROOT / "translations" / "en_US.ini"

# 匹配函数体：从 `def name(` 到下一个同级 def/装饰器之前
_FUNC_RE_CACHE = {}


def _class_body(source: str, class_name: str) -> str:
    """取出指定 class 的源码片段（到下一个顶层 class 之前）"""
    key = f"__class__{class_name}"
    if key not in _FUNC_RE_CACHE:
        _FUNC_RE_CACHE[key] = re.compile(
            rf"^class {re.escape(class_name)}\b.*?(?=^class |\Z)", re.S | re.M
        )
    match = _FUNC_RE_CACHE[key].search(source)
    assert match, f"未找到 class {class_name}（类名可能被改动，请同步本守卫）"
    return match.group(0)


def _func_body(source: str, name: str, class_name: str = None) -> str:
    """取出指定函数的源码片段（到下一个同级 def/装饰器之前）"""
    scope = _class_body(source, class_name) if class_name else source
    cache_key = f"{class_name or ''}.{name}"
    if cache_key not in _FUNC_RE_CACHE:
        _FUNC_RE_CACHE[cache_key] = re.compile(
            rf"^    def {re.escape(name)}\(.*?(?=^    def |^    @|\Z)", re.S | re.M
        )
    match = _FUNC_RE_CACHE[cache_key].search(scope)
    assert match, f"未找到函数 {cache_key}（签名可能被改动，请同步本守卫）"
    return match.group(0)


# 2.5.0 分割标注新增的全部 tr() key（双侧 INI 必须同时具备）
SEGMENTATION_KEYS = [
    # 任务模式
    "task_mode_label", "task_mode_detect", "task_mode_segment", "task_mode_switched",
    "task_mode_mismatch_polygon_hint", "task_mode_mismatch_box_hint",
    # 标签导入
    "segmentation_menu",
    "import_annotation_current", "import_annotation_batch",
    "import_annotation_dialog_title", "txt_file_filter",
    "import_overwrite_warning", "import_failed", "import_nothing_parsed",
    "import_complete_title", "import_result", "import_complete",
    "import_batch_title", "import_batch_dialog_title", "import_batch_no_match",
    "import_batch_confirmation", "import_batch_overwrite_warning",
    "import_batch_in_progress", "import_batch_result",
    # 框 ↔ 多边形互转
    "convert_current_to_segment", "convert_current_to_detect",
    "convert_folder_to_segment", "convert_folder_to_detect",
    "convert_folder_title", "convert_folder_to_segment_confirmation",
    "convert_folder_to_detect_confirmation", "convert_folder_in_progress",
    "convert_complete", "convert_nothing_to_do",
    # 导出临时转换确认
    "export_convert_confirm_title", "export_convert_polygon_to_box",
    "export_convert_box_to_polygon",
    # 模型任务适配提示与其他
    "auto_convert_box_to_polygon_hint", "auto_convert_polygon_to_box_hint",
    "annotation_updated", "polygon_min_points_hint", "delete_vertex",
]


# ---- 第 1 层：画布多边形交互 ----


class TestCanvasPolygon:
    @classmethod
    @pytest.fixture(scope="class")
    def canvas_src(cls) -> str:
        return CANVAS.read_text(encoding="utf-8")

    def test_polygon_item_exists(self, canvas_src):
        """多边形图形项必须存在（QGraphicsPolygonItem 子类）"""
        assert "class AnnotationPolygonItem(QGraphicsPolygonItem):" in canvas_src, (
            "AnnotationPolygonItem 被删除/改名：分割标注的显示与编辑全部失效"
        )

    def test_polygon_item_captures_edit_gesture(self, canvas_src):
        """拖动/框选必须走手势捕获：开始=快照，结束=一步撤销"""
        press = _func_body(canvas_src, "mousePressEvent", class_name="AnnotationPolygonItem")
        assert "_begin_edit_capture" in press, (
            "多边形按下未调用 _begin_edit_capture：顶点拖动将无法撤销（Q7）"
        )
        release = _func_body(canvas_src, "mouseReleaseEvent", class_name="AnnotationPolygonItem")
        assert "_end_edit_capture" in release, (
            "多边形释放未调用 _end_edit_capture：手势结束不会生成撤销步骤"
        )

    def test_polygon_item_syncs_annotation_points(self, canvas_src):
        """图形项顶点必须同步回 annotation.points（保存/导出依赖）"""
        body = _func_body(canvas_src, "_apply_points", class_name="AnnotationPolygonItem")
        assert "self.annotation.points" in body, (
            "_apply_points 未回写 annotation.points：拖动后保存的仍是旧坐标"
        )

    def test_delete_vertex_min_points_guard(self, canvas_src):
        """顶点删除必须保留最少 3 个顶点"""
        body = _func_body(canvas_src, "delete_vertex", class_name="AnnotationPolygonItem")
        assert "polygon_min_points_hint" in body and "<= 3" in body, (
            "delete_vertex 缺少最少 3 顶点守卫：可把多边形删成退化形状"
        )

    def test_finish_polygon_requires_three_points(self, canvas_src):
        """闭合绘制时少于 3 个顶点必须拒绝"""
        body = _func_body(canvas_src, "_finish_polygon", class_name="AnnotationCanvas")
        assert "len(self._poly_points) < 3" in body, (
            "_finish_polygon 缺少 <3 顶点守卫：会创建退化多边形"
        )
        assert "PolygonAnnotation(" in body, (
            "_finish_polygon 未构造 PolygonAnnotation：创建的多边形没有落点"
        )

    def test_canvas_exposes_task_mode_api(self, canvas_src):
        """画布任务模式接口必须存在（窗口层依赖）"""
        for name in ("set_task_mode", "is_polygon_drawing", "polygon_undo_last_point"):
            assert re.search(rf"def {name}\(", canvas_src), (
                f"AnnotationCanvas 缺少 {name}：窗口层任务模式切换/Ctrl+Z 拦截将失效"
            )

    def test_drawing_keys_abort_and_pop(self, canvas_src):
        """绘制中的 Esc 取消 / Backspace 回退顶点必须保留"""
        body = _func_body(canvas_src, "keyPressEvent", class_name="AnnotationCanvas")
        assert "Key_Escape" in body and "Key_Backspace" in body, (
            "keyPressEvent 缺少 Esc/Backspace 处理：绘制中无法取消/回退顶点"
        )

    def test_text_label_does_not_swallow_mouse(self, canvas_src):
        """文本标签必须点击穿透（否则标签覆盖处的顶点/边无法操作）"""
        body = _func_body(canvas_src, "_draw_one_annotation", class_name="AnnotationCanvas")
        assert "setAcceptedMouseButtons" in body and "NoButton" in body, (
            "标注文本标签未设置为不接收鼠标：标签覆盖区域会吞掉标注点击"
        )


# ---- 第 2 层：主窗口接线 ----


class TestMainWindowWiring:
    @classmethod
    @pytest.fixture(scope="class")
    def main_src(cls) -> str:
        return MAIN_WINDOW.read_text(encoding="utf-8")

    def test_toolbar_has_task_mode_combo(self, main_src):
        body = _func_body(main_src, "init_toolbar")
        assert "task_mode_combo" in body, "工具栏缺少任务模式下拉框（Q9）"
        assert '"detect"' in body and '"segment"' in body, (
            "任务模式下拉框缺少 detect/segment 数据项"
        )

    def test_task_mode_persisted(self, main_src):
        load_body = _func_body(main_src, "load_settings")
        assert "'task_mode'" in load_body, "load_settings 未读取 task_mode：重启后模式丢失"
        save_body = _func_body(main_src, "save_settings")
        assert "set('preferences', 'task_mode'" in save_body, (
            "save_settings 未持久化 task_mode"
        )

    def test_export_passes_task_and_confirms_conversion(self, main_src):
        for func in ("export_yolo_format", "export_dataset_split"):
            body = _func_body(main_src, func)
            assert "confirm_export_conversion" in body, (
                f"{func} 未做导出前临时转换确认（Q10）"
            )
            assert "task=self.task_mode" in body, (
                f"{func} 未把任务模式传给导出器：导出格式与模式不匹配"
            )

    def test_undo_intercepts_polygon_drawing(self, main_src):
        body = _func_body(main_src, "undo")
        assert "is_polygon_drawing" in body and "polygon_undo_last_point" in body, (
            "undo 缺少绘制中拦截：绘制多边形时 Ctrl+Z 会撤销旧命令、打断绘制（Q11）"
        )

    def test_edit_gesture_slots(self, main_src):
        started = _func_body(main_src, "_on_canvas_edit_started")
        assert "get_annotations" in started and "deepcopy" in started, (
            "手势开始未深拷贝当前标注：撤销快照会被原地修改污染"
        )
        finished = _func_body(main_src, "_on_canvas_edit_finished")
        assert "EditAnnotationsCommand" in finished and "push_command" in finished, (
            "手势结束未生成/压入编辑命令：多边形编辑将不可撤销（Q7/Q12）"
        )


# ---- 第 3 层：Mixin 装配 ----


class TestMixinWiring:
    def test_mixin_registered(self):
        init_src = MIXINS_INIT.read_text(encoding="utf-8")
        assert "from .segmentation_mixin import SegmentationMixin" in init_src, (
            "main_window_mixins/__init__ 未导出 SegmentationMixin"
        )
        assert "SegmentationMixin" in init_src

        main_src = MAIN_WINDOW.read_text(encoding="utf-8")
        assert "SegmentationMixin," in main_src, (
            "MainWindow 未混入 SegmentationMixin：分割菜单/导入/转换全部不可用"
        )

    def test_segmentation_mixin_methods(self):
        src = SEGMENTATION_MIXIN.read_text(encoding="utf-8")
        for name in (
            "_apply_task_mode", "_on_task_mode_changed", "hint_task_mode_mismatch",
            "import_segmentation_current", "import_segmentation_batch",
            "convert_current_to_segment", "convert_current_to_detect",
            "convert_folder_to_segment", "convert_folder_to_detect",
            "confirm_export_conversion", "_model_task_adaptation_hint",
        ):
            assert re.search(rf"def {re.escape(name)}\(", src), (
                f"SegmentationMixin 缺少 {name}（被删/改名后窗口接线会崩）"
            )

    def test_panels_connects_gesture_signals(self):
        src = PANELS.read_text(encoding="utf-8")
        assert "edit_gesture_started.connect" in src, (
            "panels_mixin 未连接 edit_gesture_started：手势快照不会捕获"
        )
        assert "edit_gesture_finished.connect" in src, (
            "panels_mixin 未连接 edit_gesture_finished：编辑手势不会进入撤销栈"
        )

    def test_image_actions_hints_task_mismatch(self):
        src = IMAGE_ACTIONS.read_text(encoding="utf-8")
        assert "hint_task_mode_mismatch" in src, (
            "打开文件夹后未调用任务模式不匹配提示（Q9）"
        )

    def test_model_actions_passes_task_mode(self):
        src = MODEL_ACTIONS.read_text(encoding="utf-8")
        assert src.count("convert_to_annotations(detections, task=self.task_mode)") >= 2, (
            "自动标注（单个/批量）未按任务模式转换检测结果（Q16）："
            "segment 模式下会存成矩形"
        )
        assert "_model_task_adaptation_hint" in src, (
            "自动标注未给出模型任务适配提示（Q16）"
        )

    def test_theme_language_retranslates_new_ui(self):
        src = THEME_LANGUAGE.read_text(encoding="utf-8")
        assert "task_mode_combo" in src and "setItemText" in src, (
            "语言切换未刷新任务模式下拉框文本"
        )
        assert "segmentation_menu" in src, (
            "语言切换未刷新分割菜单与动作文本"
        )


# ---- 第 4 层：i18n 完整性 ----


class TestI18nParity:
    @staticmethod
    def _ini_keys(path: Path) -> set:
        """取出 [translations] 段的所有 key（大小写敏感，与 i18n 加载器一致）"""
        config = configparser.ConfigParser()
        config.optionxform = lambda option: option
        config.read(path, encoding="utf-8")
        return set(config["translations"])

    @classmethod
    @pytest.fixture(scope="class")
    def zh_keys(cls) -> set:
        return cls._ini_keys(ZH_INI)

    @classmethod
    @pytest.fixture(scope="class")
    def en_keys(cls) -> set:
        return cls._ini_keys(EN_INI)

    @pytest.mark.parametrize("key", SEGMENTATION_KEYS)
    def test_new_key_in_both_inis(self, key, zh_keys, en_keys):
        """新增 key 必须双侧齐全：只加一侧时 CI 不会报警，运行期回退到英文/key 本身"""
        assert key in zh_keys, f"zh_CN.ini 缺少分割功能 key: {key}"
        assert key in en_keys, f"en_US.ini 缺少分割功能 key: {key}"

    def test_tr_keys_in_segmentation_sources_exist(self, zh_keys, en_keys):
        """分割相关源码中所有字面量 tr("key") 必须能命中双侧 INI"""
        sources = [SEGMENTATION_MIXIN, CANVAS]
        pattern = re.compile(r'tr\(\s*"([A-Za-z0-9_]+)"')
        missing = []
        for path in sources:
            for key in pattern.findall(path.read_text(encoding="utf-8")):
                if key not in zh_keys or key not in en_keys:
                    missing.append(f"{path.name}: {key}")
        assert not missing, f"存在双侧 INI 缺失的 tr() key: {missing}"
