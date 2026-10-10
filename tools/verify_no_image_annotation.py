"""未加载图片时进行标注操作的复验（offscreen，需 PySide6）

对应缺陷：启动后不加载任何图片，直接在画布上拖拽画框，抛
`TypeError: ... not 'NoneType'`（栈底 Path(image_path).stem，image_path 为 None）。

本脚本覆盖三条真实路径，都是 CI 跑不到的（CI 不装 PySide6，且 tests/ 禁止
import src.ui）：
  A. 冷启动空画布：真实派发鼠标按下→移动→松开，断言不抛异常、不产生标注
  B. 关夹后残留画布状态：load_folder → close_image_folder → 再拖拽
     （这一条专测 _image_item 残留导致 has_image 说谎的回归）
  C. 绕过画布信号直调 add_annotation_with_command（tools/ 里的老用法）
     + 图片解码失败后 current_image_path 必须回滚

跑法：python tools/verify_no_image_annotation.py
"""

import os
import pathlib
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])

# offscreen 下模态对话框无人点击会永久阻塞，这里把它们换成记账用的空实现：
# 仍然走完对话框调用路径（否则测不到），但不会挂住。
_DIALOG_CALLS = {"warning": 0, "critical": 0, "information": 0, "question": 0}


def _stub_dialog(kind):
    def _fn(*args, **kwargs):
        _DIALOG_CALLS[kind] += 1
        return QMessageBox.Yes

    return staticmethod(_fn)


QMessageBox.warning = _stub_dialog("warning")
QMessageBox.critical = _stub_dialog("critical")
QMessageBox.information = _stub_dialog("information")
QMessageBox.question = _stub_dialog("question")

from PIL import Image  # noqa: E402

from src.core.annotation import Annotation  # noqa: E402
from src.ui.main_window import MainWindow  # noqa: E402


def make_event(event_type, pos, button=Qt.LeftButton):
    """构造画布鼠标事件（view 坐标 → 由 canvas.mapToScene 自行换算）"""
    return QMouseEvent(
        event_type,
        QPointF(float(pos[0]), float(pos[1])),
        button,
        button,
        Qt.NoModifier,
    )


def drag_on_canvas(canvas, start=(60, 60), end=(200, 180)):
    """在画布上完成一次"按下→拖动→松开"，模拟用户画框"""
    from PySide6.QtCore import QEvent

    canvas.mousePressEvent(make_event(QEvent.MouseButtonPress, start))
    canvas.mouseMoveEvent(make_event(QEvent.MouseMove, (130, 120)))
    canvas.mouseReleaseEvent(make_event(QEvent.MouseButtonRelease, end))


def status_text(window) -> str:
    return window.status_label.text()


def expect_no_annotation(label, fn):
    """执行 fn，断言不抛异常且没有产生任何标注"""
    try:
        fn()
    except Exception:
        print(f"[FAIL] {label}: 仍然抛异常", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
    print(f"[OK] {label}: 未抛异常", flush=True)


def main():
    # 用独立临时目录 + 独立标注目录：AnnotationManager 的标注目录锚定应用根
    # （_get_app_root()/annotations），若沿用真实目录，本次运行写入的标注会
    # 残留在仓库里，下次运行加载图片时旧标注会落在光标下 —— 按下命中标注项
    # 而不是开始绘制，导致"正常拖拽"断言假失败。
    base = pathlib.Path(tempfile.mkdtemp(prefix="imgtest_noimg_"))
    ann_dir = pathlib.Path(tempfile.mkdtemp(prefix="annotations_noimg_"))
    for i in range(2):
        Image.new("RGB", (320, 240), (i * 90, 40, 40)).save(base / f"img{i}.png")

    w = MainWindow()
    # 关夹/切图会调 save_settings() 把语言、主题写回 config.ini —— 那会污染
    # 开发者本机的真实配置（本次 D 段切英文就曾把 language 写成 en_US）。
    # 把 config 路径改到临时文件，脚本对仓库零副作用；原配置不读也不回写。
    w.config_file_path = ann_dir.parent / "config.ini"
    w.annotation_manager._annotation_dir = str(ann_dir)

    # ---- A. 冷启动，未加载任何图片 ----
    assert w.current_image_path is None, "前置条件不成立：启动即带图片"
    assert w.canvas.has_image is False, "空画布 has_image 应为 False"
    expect_no_annotation(
        "A 冷启动空画布拖拽画框",
        lambda: drag_on_canvas(w.canvas),
    )
    assert w.annotation_manager.can_undo() is False, "A BUG: 空画布拖拽竟进了撤销栈"
    assert w.canvas.get_annotation_items() == [], "A BUG: 空画布竟产生了标注项"
    assert status_text(w), (
        "A BUG: 无图时拖拽没有给用户任何提示 —— 状态栏仍为空，"
        "用户只会觉得'拖了没反应'"
    )
    print(
        f"[OK] A 冷启动：无标注、撤销栈为空、画布 has_image 仍为 False，"
        f"且状态栏已提示「{status_text(w)}」",
        flush=True,
    )

    # ---- B. 加载后关夹，再拖拽（测 _image_item 残留回归）----
    w.load_image_folder_by_path(str(base))
    w.load_image(0)
    assert w.canvas.has_image is True, "前置条件不成立：已加载图片但 has_image 为假"
    expect_no_annotation(
        "B 已加载图片拖拽（应正常产生标注）",
        lambda: drag_on_canvas(w.canvas),
    )
    assert w.annotation_manager.can_undo() is True, "B BUG: 正常图片拖拽后应可撤销"
    print("[OK] B 已加载图片：正常标注仍可用（守卫没有误伤正常流程）", flush=True)

    w.close_image_folder()
    assert w.current_image_path is None, "B BUG: 关夹后 current_image_path 未清空"
    assert w.canvas.has_image is False, (
        "B BUG: 关夹后画布 has_image 仍为真 —— _image_item 残留，"
        "这正是原缺陷的隐藏复发点"
    )
    expect_no_annotation(
        "B 关夹后再次拖拽画框",
        lambda: drag_on_canvas(w.canvas),
    )
    assert w.annotation_manager.can_undo() is False, "B BUG: 关夹后拖拽竟进了撤销栈"
    print("[OK] B 关夹后：画布状态正确复位，拖拽被安全忽略", flush=True)

    # ---- C. 绕过画布信号直调 + 解码失败的图片 ----
    w.close_image_folder()
    expect_no_annotation(
        "C 无图时直调 add_annotation_with_command",
        lambda: w.add_annotation_with_command(Annotation(1, 1, 10, 10, 0)),
    )
    assert w.annotation_manager.can_undo() is False, "C BUG: 无图时直调竟进了撤销栈"
    assert status_text(w), "C BUG: 无图时直调未给出状态栏提示，用户无感知"
    print(
        f"[OK] C 无图直调 add_annotation_with_command：被安全忽略，"
        f"且已提示「{status_text(w)}」",
        flush=True,
    )

    # 解码失败的图片必须回滚 current_image_path，不能留半同步状态
    bad = base / "broken.png"
    bad.write_bytes(b"this is not a real png")
    w.image_manager.load_folder(str(base))
    bad_index = w.image_manager.get_all_image_paths().index(str(bad))
    w.load_image(bad_index)
    assert w.current_image_path is None, (
        "C BUG: 图片解码失败后 current_image_path 未回滚，"
        "画布与主窗口状态矛盾，后续标注会写到无效路径"
    )
    assert w.canvas.has_image is False, "C BUG: 解码失败后画布仍认为有图"
    assert _DIALOG_CALLS["warning"] >= 1, (
        "C BUG: 图片解码失败未给出用户提示 —— 上面的状态回滚断言可能是"
        "走了别的分支才通过的，而非被测的回滚逻辑"
    )
    print(
        f"[OK] C 解码失败的图片：current_image_path 正确回滚，画布复位，"
        f"且已提示用户（warning 调用 {_DIALOG_CALLS['warning']} 次）",
        flush=True,
    )

    # ---- D. 英文界面下提示也必须是英文（防"只加一侧 INI"）----
    from src.utils.i18n import TranslationManager

    tm = TranslationManager.instance()
    prev_lang = tm.get_current_language()
    tm.switch_language("en_US")
    w.close_image_folder()
    drag_on_canvas(w.canvas)
    hint = status_text(w)
    assert hint, "D BUG: 英文界面下无图拖拽无提示"
    # 只查中文字符不够：tr() 找不到 key 时会原样返回 key 名
    # （如 "canvas_no_image_hint"），那也不是可用的英文文案。
    assert hint != "canvas_no_image_hint", (
        f"D BUG: 英文界面提示回退成了 key 名「{hint}」—— "
        "en_US.ini 里缺 canvas_no_image_hint"
    )
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in hint), (
        f"D BUG: 英文界面下提示仍是中文「{hint}」—— "
        "en_US.ini 可能漏了 canvas_no_image_hint"
    )
    print(f"[OK] D 英文界面提示为英文：「{hint}」", flush=True)
    tm.switch_language(prev_lang)

    print("\n全部复验通过：未加载图片时的标注操作不再抛异常，正常流程未受影响。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
