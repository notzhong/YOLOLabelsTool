"""B2/B3/B5/B14 修复后的运行时复验（offscreen）"""

import json
import os
import pathlib
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPO = pathlib.Path(__file__).resolve().parents[1]

# ---- B2-a: 从其他 CWD 启动，配置/QSS/标注目录必须仍锚定应用根 ----
os.chdir("/tmp")
shutil.rmtree("/tmp/config", ignore_errors=True)
shutil.rmtree("/tmp/annotations", ignore_errors=True)  # 清理修复前复现留下的残留

from PIL import Image  # noqa: E402

base = pathlib.Path("/tmp/imgtest_fix")  # noqa: E402
shutil.rmtree(base, ignore_errors=True)
base.mkdir()
for i in range(6):
    Image.new("RGB", (40, 30), (i * 40, 0, 0)).save(base / f"img{i}.png")

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog  # noqa: E402

app = QApplication([])

# 弹窗全部打桩
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.critical = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)

from src.ui.main_window import MainWindow  # noqa: E402
from src.core.annotation import AnnotationManager  # noqa: E402

mgr = AnnotationManager()
assert str(mgr._annotation_dir) == str(REPO / "annotations"), mgr._annotation_dir
assert not pathlib.Path("/tmp/annotations").exists(), "标注目录被写到了 CWD！"
print("B2-a 标注目录锚定应用根 OK:", mgr._annotation_dir)

w = MainWindow()
assert str(w.config_file_path) == str(REPO / "config" / "config.ini")
assert not pathlib.Path("/tmp/config").exists(), "配置写到了 CWD！"
assert w.styleSheet(), "QSS 主题未加载（styleSheet 为空）"
print("B2-b 配置/QSS 锚定应用根 OK，主题已应用")

# ---- B14: 关文件夹后 undo 栈必须清空 ----
w.load_image_folder_by_path(str(base))
from src.core.annotation import Annotation  # noqa: E402

w.add_annotation_with_command(Annotation(1, 1, 10, 10, 0))
assert w.annotation_manager.can_undo(), "绘制后应可撤销"
w.close_image_folder()
assert not w.annotation_manager.can_undo(), "B14 BUG: 关夹后撤销栈未清空"
assert not w.annotation_manager.can_redo(), "B14 BUG: 关夹后重做栈未清空"
print("B14 关文件夹清栈 OK")

# ---- B3: 移除前面的图片后应停留在当前图片 ----
w.load_image_folder_by_path(str(base))
order = [w.image_manager.get_image_path(i) for i in range(6)]
w.load_image(4)
w.image_list_widget.setCurrentRow(1)  # 选中第 1 行并移除
w._remove_selected_images()
assert w.current_image_path == order[4], (
    f"B3 BUG: 应显示 {pathlib.Path(order[4]).name}，实际 "
    f"{pathlib.Path(w.current_image_path or '').name}"
)
print("B3 移除后按路径重定位 OK（停留在 img4）")

# ---- B5: 批量标注后当前画布必须刷新，随后切图不得覆盖批量结果 ----
from src.core.annotation import Annotation as Ann  # noqa: E402

w.load_image_folder_by_path(str(base))
current = w.image_manager.get_image_path(0)
w.load_image(0)
w.add_annotation_with_command(Ann(1, 1, 10, 10, 0))  # 旧的手工标注（已落盘）

w.model_manager.is_available = lambda: True
w.model_manager.is_model_loaded = lambda: True
w.model_manager.predict = lambda path: ["det"]
w.model_manager.convert_to_annotations = lambda d: [Ann(5, 5, 20, 20, 1)]

w.batch_auto_annotate()

items = w.canvas.get_annotation_items()
assert len(items) == 1 and (items[0].x, items[0].width, items[0].class_id) == (5, 20, 1), (
    f"B5 BUG: 批量后画布未刷新: {[(a.x, a.width, a.class_id) for a in items]}"
)
# 模拟随后切图（load_image 会先把画布存盘）
w.next_image()
path = w.annotation_manager.get_annotation_path(current)
data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
assert len(data) == 1 and data[0]["x"] == 5, f"B5 BUG: 切图后批量结果被覆盖: {data}"
print("B5 批量标注画布刷新 OK（切图未覆盖批量结果）")

print("ALL FLOW FIXES VERIFIED")
