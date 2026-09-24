"""B13（重新配置流程）/ B6（撤销栈按图）/ runs 目录锚定 的复验（offscreen）"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
REPO = pathlib.Path(__file__).resolve().parents[1]

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

app = QApplication([])
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.critical = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)

# ---- runs 训练输出目录锚定（从 /tmp 启动） ----
os.chdir("/tmp")
from yolo_tool import YOLOTrainer  # noqa: E402

expected = str(REPO / "runs" / "train")
assert YOLOTrainer().get_default_config()["output_dir"] == expected
print("runs 输出目录锚定 OK:", expected)

from src.core.annotation import Annotation  # noqa: E402
from src.ui.main_window import MainWindow  # noqa: E402

# ---- B6: 换图清 undo/redo，撤销不再跨图生效 ----
base = pathlib.Path("/tmp/imgtest_b6")
shutil.rmtree(base, ignore_errors=True)
base.mkdir()
from PIL import Image  # noqa: E402

for i in range(2):
    Image.new("RGB", (40, 30), (i * 100, 0, 0)).save(base / f"img{i}.png")

w = MainWindow()
w.load_image_folder_by_path(str(base))
paths = [w.image_manager.get_image_path(i) for i in range(2)]
w.load_image(0)
w.add_annotation_with_command(Annotation(1, 1, 10, 10, 0))
assert w.annotation_manager.can_undo(), "绘制后应可撤销"

w.next_image()  # 切到第二张
assert not w.annotation_manager.can_undo(), "B6 BUG: 换图后撤销栈未清空"
assert not w.annotation_manager.can_redo(), "B6 BUG: 换图后重做栈未清空"

w.add_annotation_with_command(Annotation(2, 2, 12, 12, 0))  # 第二张新标注
w.undo()  # 只应撤销第二张
saved = json.loads(
    pathlib.Path(w.annotation_manager.get_annotation_path(paths[0])).read_text(
        encoding="utf-8"
    )
)
assert saved, "B6 BUG: 第一张图片的标注被跨图撤销误删"
assert not w.annotation_manager.has_annotations(paths[1]), "B6 BUG: 第二张未撤销"
print("B6 换图清栈 OK（撤销作用域限定在当前图片）")

# ---- B13: 训练"重新配置"流程 ----
from src.ui.train_dialog import TrainDialog  # noqa: E402
from src.ui.train_progress_dialog import TrainProgressDialog  # noqa: E402

# (a) 进度对话框：reconfigure 设置标记 + 断开 trainer 信号
trainer = YOLOTrainer()
progress = TrainProgressDialog(trainer, parent=None)
assert progress.reconfigure_requested is False
progress.reconfigure_training()
assert progress.reconfigure_requested is True
assert progress.result() == QDialog.Rejected
detached = False
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    trainer.training_started.disconnect(progress.on_training_started)
    detached = any("Failed to disconnect" in str(item.message) for item in caught)
assert detached, "B13 BUG: 重新配置后 trainer 信号仍连接（连接会累积）"
print("B13-a 重新配置标记 + trainer 信号断开 OK")

# (b) TrainDialog：模拟"重新配置" → 不关闭配置对话框、开始按钮恢复
tmp = pathlib.Path(tempfile.mkdtemp())
model, data_yaml = tmp / "fake.pt", tmp / "data.yaml"
model.write_text("")
data_yaml.write_text("path: .\n")
TrainProgressDialog.start_training = lambda self: True


def make_dialog():
    dialog = TrainDialog()
    dialog.model_path_edit.setText(str(model))
    dialog.data_yaml_edit.setText(str(data_yaml))
    dialog.trainer.setup = lambda cfg: True
    return dialog


def exec_reconfigure(self):
    self.reconfigure_requested = True
    return QDialog.Rejected


def exec_close(self):
    return QDialog.Accepted


TrainProgressDialog.exec = exec_reconfigure
d1 = make_dialog()
d1.start_training()
assert d1.result() != QDialog.Accepted, "B13 BUG: 重新配置后不应关闭配置对话框"
assert d1.btn_start.isEnabled(), "B13 BUG: 重新配置后开始按钮未恢复"
print("B13-b 重新配置后留在配置对话框 OK")

# (c) 正常关闭进度窗 → 关闭配置对话框（原有行为保持）
TrainProgressDialog.exec = exec_close
d2 = make_dialog()
d2.start_training()
assert d2.result() == QDialog.Accepted, "B13 BUG: 正常关闭后应接受并关闭配置对话框"
assert not d2.btn_start.isEnabled()
print("B13-c 正常关闭后关闭配置对话框 OK")

print("ALL DECISION FIXES VERIFIED")
