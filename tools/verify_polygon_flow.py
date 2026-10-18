"""分割标注（多边形 / YOLO-seg）全流程验收脚本（offscreen，需 PySide6）

覆盖 2.5.0 分割标注功能从画布到窗口层的整条链路：

  A. 初始化：工具栏任务模式下拉框 / 分割菜单 / 默认 detect
  B. segment 模式绘制：单击加顶点 + 双击闭合 → PolygonAnnotation 进入管理器与撤销栈
  C. 绘制控制：Backspace 回退顶点、Esc 取消
  D. detect 模式矩形绘制回归
  E. 顶点拖动：手势开始/结束信号；恰好一步撤销；画布与缓存对象一致性；
     Ctrl+Z（窗口 undo）恢复顶点、redo 再应用
  F. 绘制中 Ctrl+Z 拦截（撤销栈非空，窗口 undo 路径）：回退顶点而非撤销命令
  G. 导入当前图片 YOLO 标签（框 + 多边形 + 无效行）：解析、跳过、一步撤销
  H. 当前图片转换：框 → 多边形（可撤销）
  I. segment 导出：导出前临时转换确认；data.yaml 含 task: segment；标签为分割格式
  J. 任务模式持久化：切换后写入 config.ini [preferences] task_mode
  K. 批量导入标签文件夹：按文件名主干与图片匹配
  L. 文件夹批量转换：直接落盘、清空撤销栈
  M. 空撤销栈绘制中 Ctrl+Z：QAction 被禁用时按键落到画布（真实按键）仍回退顶点

跑法（Windows，offscreen 无需真实显示器；需 PySide6，如 conda 环境 yolo）：
  python tools\\verify_polygon_flow.py
退出码 0 表示全部通过。
"""

import configparser
import copy
import os
import pathlib
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

app = QApplication.instance() or QApplication([])

# offscreen 下模态对话框无人点击会永久阻塞：全部替换为记账用实现
_DIALOG_CALLS = {"warning": 0, "critical": 0, "information": 0, "question": 0}
_question_answer = {"value": QMessageBox.Yes}


def _stub_dialog(kind):
    def _fn(*args, **kwargs):
        _DIALOG_CALLS[kind] += 1
        return QMessageBox.Yes

    return staticmethod(_fn)


def _stub_question(*args, **kwargs):
    _DIALOG_CALLS["question"] += 1
    return _question_answer["value"]


QMessageBox.warning = _stub_dialog("warning")
QMessageBox.critical = _stub_dialog("critical")
QMessageBox.information = _stub_dialog("information")
QMessageBox.question = staticmethod(_stub_question)

# 文件对话框返回值由各场景预置
_FILE_DIALOG = {"open": None, "save": None, "dir": None}

QFileDialog.getOpenFileName = staticmethod(
    lambda *a, **k: (_FILE_DIALOG["open"], "txt")
)
QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (_FILE_DIALOG["save"], "yaml")
)
QFileDialog.getExistingDirectory = staticmethod(
    lambda *a, **k: _FILE_DIALOG["dir"]
)

from PIL import Image  # noqa: E402

from src.core.annotation import Annotation, PolygonAnnotation  # noqa: E402
from src.ui.main_window import MainWindow  # noqa: E402


def view_pos_of(canvas, scene_pos):
    """场景坐标 → 视图（viewport）坐标"""
    return canvas.mapFromScene(scene_pos)


def click(canvas, scene_pos):
    """在场景坐标处单击

    必须走 QTest 真实事件流：直接手搓 QMouseEvent 调用 canvas.mousePressEvent
    时，视图→场景→图形项的二次分发不可靠（图像项永远收不到事件），
    多边形顶点拖动等依赖图形项事件的交互会静默失败。
    """
    QTest.mouseClick(
        canvas.viewport(), Qt.LeftButton, Qt.NoModifier, view_pos_of(canvas, scene_pos)
    )


def double_click(canvas, scene_pos):
    QTest.mouseDClick(
        canvas.viewport(), Qt.LeftButton, Qt.NoModifier, view_pos_of(canvas, scene_pos)
    )


def drag_on_canvas(canvas, start_scene, end_scene):
    """按下 → 移动 → 释放（矩形绘制 / 顶点拖动）"""
    start = view_pos_of(canvas, start_scene)
    end = view_pos_of(canvas, end_scene)
    QTest.mousePress(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(canvas.viewport(), end)
    QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, end)


def key_event(canvas, key):
    canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))


def find_polygon_item(canvas, annotation):
    for item in canvas._scene.items():
        if getattr(item, "annotation", None) is annotation:
            return item
    return None


def main():
    work = pathlib.Path(tempfile.mkdtemp(prefix="seg_verify_"))
    img_dir = work / "images"
    img_dir.mkdir()
    Image.new("RGB", (320, 240), (30, 60, 90)).save(img_dir / "img0.png")
    Image.new("RGB", (320, 240), (90, 60, 30)).save(img_dir / "img1.png")

    ann_dir = work / "annotations"
    ann_dir.mkdir()
    cfg_path = work / "config.ini"

    w = MainWindow()
    # 重定向配置与标注目录，避免污染真实数据
    w.config_file_path = cfg_path
    w.annotation_manager._annotation_dir = str(ann_dir)
    if w.image_manager.get_image_count() > 0:
        w.close_image_folder()   # 丢弃真实配置里可能自动加载的文件夹
    w._apply_task_mode("detect", persist=False)

    # ---- A. 初始化 ----
    assert w.task_mode == "detect"
    assert w.canvas.task_mode == "detect"
    assert w.task_mode_combo.count() == 2
    assert w.task_mode_combo.itemData(0) == "detect"
    assert w.task_mode_combo.itemData(1) == "segment"
    assert w.segmentation_menu is not None
    assert w.action_import_annotation is not None
    assert w.action_convert_folder_to_segment is not None
    print("[OK] A 初始化：任务模式下拉框 / 分割菜单 / 默认 detect")

    # ---- 准备 ----
    w.load_image_folder_by_path(str(img_dir))
    assert w.canvas.has_image and w.current_image_path
    current = w.current_image_path
    assert pathlib.Path(current).stem == "img0"
    assert w.annotation_manager.get_annotations(current) == []

    c = w.canvas

    # ---- B. segment 模式绘制多边形 ----
    w._apply_task_mode("segment", persist=False)
    click(c, QPointF(60, 40))
    click(c, QPointF(200, 60))
    click(c, QPointF(160, 180))
    assert c.is_polygon_drawing(), "三次单击后应处于绘制状态"
    double_click(c, QPointF(100, 120))   # 距最后顶点足够远 → 作为第 4 个顶点并闭合
    assert not c.is_polygon_drawing()
    anns = w.annotation_manager.get_annotations(current)
    assert len(anns) == 1 and isinstance(anns[0], PolygonAnnotation)
    assert len(anns[0].points) == 4
    assert w.annotation_manager.can_undo()
    print("[OK] B segment 绘制：4 顶点多边形已创建并进入撤销栈")

    # ---- C. Backspace / Esc ----
    click(c, QPointF(250, 60))
    click(c, QPointF(300, 90))
    assert c.is_polygon_drawing() and len(c._poly_points) == 2
    key_event(c, Qt.Key_Backspace)
    assert len(c._poly_points) == 1
    key_event(c, Qt.Key_Escape)
    assert not c.is_polygon_drawing() and not c._poly_points
    print("[OK] C Backspace 回退顶点、Esc 取消绘制")

    # ---- D. detect 模式矩形绘制回归 ----
    w._apply_task_mode("detect", persist=False)
    drag_on_canvas(c, QPointF(240, 30), QPointF(310, 120))
    anns = w.annotation_manager.get_annotations(current)
    assert sum(1 for a in anns if isinstance(a, Annotation)) == 1
    assert sum(1 for a in anns if isinstance(a, PolygonAnnotation)) == 1
    print("[OK] D detect 模式矩形绘制正常（回归）")

    # ---- E. 顶点拖动：一步撤销 + 对象一致性 ----
    poly_ann = next(a for a in anns if isinstance(a, PolygonAnnotation))
    item = find_polygon_item(c, poly_ann)
    assert item is not None, "画布上应存在多边形图形项"

    gesture = {"started": 0, "finished": 0}
    c.edit_gesture_started.connect(
        lambda: gesture.__setitem__("started", gesture["started"] + 1)
    )
    c.edit_gesture_finished.connect(
        lambda: gesture.__setitem__("finished", gesture["finished"] + 1)
    )

    before_points = [(round(x), round(y)) for x, y in poly_ann.points]
    before_stack = len(w.annotation_manager._undo_stack)

    vertex = item.polygon()[0]
    end_scene = QPointF(vertex.x() + 12, vertex.y() + 8)
    drag_on_canvas(c, vertex, end_scene)

    assert gesture["started"] == 1 and gesture["finished"] == 1, gesture
    after_points = [(round(x), round(y)) for x, y in poly_ann.points]
    assert after_points != before_points, "顶点拖动应改变坐标"
    assert len(w.annotation_manager._undo_stack) == before_stack + 1, (
        "一次手势应恰好一步撤销"
    )

    # 对象一致性：编辑手势结束后画布对象仍与缓存同一批（删除/改类依赖 identity）
    cache = w.annotation_manager.get_annotations(current)
    canvas_anns = c.get_annotation_items()
    assert len(cache) == len(canvas_anns)
    assert all(any(ca is cc for cc in cache) for ca in canvas_anns), (
        "画布与缓存对象身份不一致"
    )

    w.undo()   # Ctrl+Z：顶点回到原位
    restored = w.annotation_manager.get_annotations(current)
    restored_poly = next(a for a in restored if isinstance(a, PolygonAnnotation))
    assert [(round(x), round(y)) for x, y in restored_poly.points] == before_points

    w.redo()
    redone = w.annotation_manager.get_annotations(current)
    redone_poly = next(a for a in redone if isinstance(a, PolygonAnnotation))
    assert [(round(x), round(y)) for x, y in redone_poly.points] == after_points
    print("[OK] E 顶点拖动：一步撤销 / 对象一致 / undo-redo 正常")

    # ---- F. 绘制中 Ctrl+Z 拦截 ----
    w._apply_task_mode("segment", persist=False)
    click(c, QPointF(300, 210))
    click(c, QPointF(312, 225))
    assert c.is_polygon_drawing(), "右下角空白处应能开始新绘制"
    stack_len = len(w.annotation_manager._undo_stack)
    w.undo()   # 应为回退顶点，而不是撤销命令
    assert c.is_polygon_drawing() and len(c._poly_points) == 1
    assert len(w.annotation_manager._undo_stack) == stack_len
    key_event(c, Qt.Key_Escape)
    print("[OK] F 绘制中 Ctrl+Z：回退顶点且不动撤销栈")

    # ---- G. 导入当前图片标签 ----
    labels_src = work / "labels_src"
    labels_src.mkdir()
    label_file = labels_src / "import_me.txt"
    label_file.write_text(
        "0 0.5 0.5 0.1 0.1\n"            # 检测框
        "0 0.1 0.2 0.3 0.2 0.3 0.6\n"    # 3 顶点多边形
        "not a valid line\n",             # 无效行
        encoding="utf-8",
    )
    _FILE_DIALOG["open"] = str(label_file)

    pre_import = copy.deepcopy(w.annotation_manager.get_annotations(current))
    w.import_segmentation_current()
    imported = w.annotation_manager.get_annotations(current)
    assert len(imported) == 2, [type(a).__name__ for a in imported]
    assert sum(1 for a in imported if isinstance(a, PolygonAnnotation)) == 1
    w.undo()
    undone = w.annotation_manager.get_annotations(current)
    assert [a.to_dict() for a in undone] == [a.to_dict() for a in pre_import]
    print("[OK] G 导入当前图片：框+多边形解析、跳过无效行、一步撤销")

    # ---- H. 当前图片转换：框 → 多边形 ----
    w.convert_current_to_segment()
    conv = w.annotation_manager.get_annotations(current)
    assert conv and all(isinstance(a, PolygonAnnotation) for a in conv)
    w.undo()
    back = w.annotation_manager.get_annotations(current)
    assert any(isinstance(a, Annotation) for a in back)
    print("[OK] H 当前图片转换：框→多边形且可撤销恢复")

    # ---- I. segment 导出 ----
    w._apply_task_mode("segment", persist=False)
    export_dir = work / "seg_export"
    _FILE_DIALOG["save"] = str(export_dir / "data.yaml")
    questions_before = _DIALOG_CALLS["question"]
    w.export_yolo_format()
    assert _DIALOG_CALLS["question"] == questions_before + 1, (
        "导出前应触发临时转换确认（当前含 1 个框）"
    )

    import yaml
    yaml_path = export_dir / "data.yaml"
    assert yaml_path.exists(), "导出目录应包含 data.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data.get("task") == "segment"

    label_files = [p for p in export_dir.rglob("*.txt") if p.stem == "img0"]
    assert label_files, "导出目录应包含 img0 的标签文件"
    for line in label_files[0].read_text(encoding="utf-8").splitlines():
        parts = line.split()
        assert len(parts) >= 7 and (len(parts) - 1) % 2 == 0, line
    print("[OK] I segment 导出：临时转换确认 + task:segment + 分割格式标签")

    # ---- J. 任务模式持久化 ----
    w._apply_task_mode("detect", persist=False)   # 先归位，保证切换真实触发
    idx = w.task_mode_combo.findData("segment")
    w.task_mode_combo.setCurrentIndex(idx)
    assert w.task_mode == "segment" and w.canvas.task_mode == "segment"
    cfg = configparser.ConfigParser()
    cfg.read(str(cfg_path), encoding="utf-8")
    assert cfg.get("preferences", "task_mode") == "segment"
    print("[OK] J 任务模式持久化：config.ini 写入 task_mode=segment")

    # ---- K. 批量导入标签文件夹 ----
    batch_dir = work / "labels_batch"
    batch_dir.mkdir()
    (batch_dir / "img1.txt").write_text(
        "0 0.2 0.2 0.4 0.2 0.4 0.6 0.2 0.6\n",   # 4 顶点多边形
        encoding="utf-8",
    )
    _FILE_DIALOG["dir"] = str(batch_dir)
    w.import_segmentation_batch()
    img1_path = next(
        p for p in w.image_manager.get_all_image_paths()
        if pathlib.Path(p).stem == "img1"
    )
    img1_anns = w.annotation_manager.get_annotations(img1_path)
    assert len(img1_anns) == 1 and isinstance(img1_anns[0], PolygonAnnotation)
    print("[OK] K 批量导入：按文件名匹配 img1.txt")

    # ---- L. 文件夹批量转换 ----
    w.convert_folder_to_detect()
    img0_anns = w.annotation_manager.get_annotations(current)
    img1_anns = w.annotation_manager.get_annotations(img1_path)
    assert img0_anns and all(isinstance(a, Annotation) for a in img0_anns)
    assert img1_anns and all(isinstance(a, Annotation) for a in img1_anns)
    assert not w.annotation_manager.can_undo(), "批量转换后应清空撤销栈"
    print("[OK] L 文件夹批量转换：全部转框、撤销栈已清空")

    # ---- M. 空撤销栈绘制中 Ctrl+Z（QAction 禁用时的真实按键路径）----
    # L 批量转换后撤销栈已清空且调用了 update_undo_redo_actions()：
    # action_undo 处于禁用状态，Ctrl+Z 快捷键不会触发窗口 undo()，
    # 按键直接落到画布——画布必须自己兜住"绘制中回退顶点"。
    assert not w.annotation_manager.can_undo()
    assert not w.action_undo.isEnabled(), "本场景前置条件：撤销 QAction 应处于禁用"
    click(c, QPointF(20, 205))   # 左下空白区（在所有标注外接框之外）
    click(c, QPointF(60, 235))
    assert c.is_polygon_drawing() and len(c._poly_points) == 2
    stack_len = len(w.annotation_manager._undo_stack)
    QTest.keyClick(c, Qt.Key_Z, Qt.ControlModifier)   # 真实按键，非直调 undo()
    assert len(c._poly_points) == 1, (
        "空撤销栈时 QAction 被禁用：绘制中 Ctrl+Z 应回退顶点（画布兜底）"
    )
    assert c.is_polygon_drawing()
    assert len(w.annotation_manager._undo_stack) == stack_len
    key_event(c, Qt.Key_Escape)
    print("[OK] M 空撤销栈绘制中 Ctrl+Z：真实按键落到画布仍回退顶点")

    w.close()
    print("\n=== 分割标注流程验收：全部通过 ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError:
        traceback.print_exc()
        print("\n!!! 验收失败（断言不通过）")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print("\n!!! 验收失败（异常）")
        sys.exit(1)
