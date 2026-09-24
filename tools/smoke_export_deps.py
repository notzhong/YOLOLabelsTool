"""导出依赖"询问后安装"流程 offscreen 冒烟（含冻结环境降级路径）"""

import os
import pathlib
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# 与真实启动一致：先走 main.setup_environment，确认收回 ultralytics 自动安装权
import main  # noqa: E402

main.setup_environment()
assert os.environ["YOLO_AUTOINSTALL"] == "False"

from ultralytics.utils import AUTOINSTALL  # noqa: E402

assert AUTOINSTALL is False, f"AUTOINSTALL={AUTOINSTALL}"

from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication, QApplication as _QA  # noqa: E402

app = QApplication([])

from src.ui import export_dialog as ed  # noqa: E402
from src.utils import export_deps  # noqa: E402


# ---- 用假 QMessageBox 替换 export_dialog 命名空间（避免任何真实弹窗） ----
class FakeMessageBox:
    Yes, No = 1, 2  # 支持代码里的 Yes | No 位运算
    Warning, ActionRole, RejectRole = "warning", "action", "reject"
    answer = 2  # No
    clicks_copy = False
    info_calls = []
    critical_calls = []

    def __init__(self, parent=None):
        self._buttons = []
        self._clicked = None

    def setIcon(self, *_):
        pass

    def setWindowTitle(self, *_):
        pass

    def setText(self, *_):
        pass

    def addButton(self, text, role):
        button = type("Btn", (), {"text": text})()
        self._buttons.append(button)
        if FakeMessageBox.clicks_copy and text and ("复制" in text or "Copy" in text):
            self._clicked = button
        return button

    def exec(self):
        pass

    def clickedButton(self):
        return self._clicked

    @classmethod
    def question(cls, *args, **kwargs):
        return cls.answer

    @classmethod
    def information(cls, *args, **kwargs):
        cls.info_calls.append(args)

    @classmethod
    def critical(cls, *args, **kwargs):
        cls.critical_calls.append(args)

    @classmethod
    def warning(cls, *args, **kwargs):
        return None


ed.QMessageBox = FakeMessageBox

dialog = ed.ExportDialog(default_model_path="/tmp/does-not-exist.pt")

# ---- 1) 真实环境依赖检测：已装齐的格式无缺失，未装的精确报缺 ----
assert dialog.check_dependencies("ONNX") == [], dialog.check_dependencies("ONNX")
assert dialog.check_dependencies("TensorRT") == []
assert dialog.check_dependencies("OpenVINO") == []
assert dialog.check_dependencies("TFLite") == ["tensorflow"]
assert dialog.check_dependencies("ncnn") == []

# ---- 2) 翻译文案中的字面 \n 被还原为真实换行 ----
text = ed._text("deps_install_ask_msg", "x").format(fmt="ONNX", packages="a")
assert "\n" in text and "\\n" not in text, repr(text)

# ---- 3) 用户拒绝 → 中止，不安装 ----
FakeMessageBox.answer = FakeMessageBox.No
assert dialog._resolve_missing_deps("TFLite", ["tensorflow"]) is False

# ---- 4) 用户同意 → 触发后台安装（拦截 _start_install 避免真装 tensorflow） ----
recorded = {}
dialog._start_install = lambda pkgs: (recorded.setdefault("pkgs", list(pkgs)), False)[1]
FakeMessageBox.answer = FakeMessageBox.Yes
assert dialog._resolve_missing_deps("TFLite", ["tensorflow"]) is False
assert recorded["pkgs"] == ["tensorflow"], recorded

# ---- 5) 冻结环境 → 复制安装命令（不执行 pip） ----
FakeMessageBox.clicks_copy = True
ed.is_frozen = lambda: True
started = []
dialog._start_install = lambda pkgs: started.append(pkgs) or False
assert dialog._resolve_missing_deps("ONNX", ["onnxslim>=0.1.82"]) is False
assert started == [], started
assert FakeMessageBox.info_calls, "复制后应弹提示"
clipboard = _QA.clipboard().text()
assert clipboard.startswith("pip install ") and "onnxslim>=0.1.82" in clipboard, clipboard
ed.is_frozen = export_deps.is_frozen

# ---- 6) 真实 InstallWorker：安装已满足的依赖（无网络下载），验证线程管线 ----
results, logs = [], []
worker = ed.InstallWorker(["Pillow>=10.0.0"], parent=dialog)
worker.progress.connect(lambda m: logs.append(m))
worker.finished.connect(lambda ok, msg: (results.append((ok, msg)), loop.quit()))
loop = QEventLoop()
worker.start()
deadline = time.time() + 60
while not results and time.time() < deadline:
    app.processEvents()
    time.sleep(0.05)
assert results and results[0][0] is True, (results, logs[:20])
assert any("pip install" in line for line in logs), logs[:3]
assert any("already satisfied" in line.lower() for line in logs), logs[:20]

# ---- 7) 失败路径：非法包名 → 非零退出码 ----
results.clear()
worker2 = ed.InstallWorker(["this-package-does-not-exist-xyz123"], parent=dialog)
worker2.finished.connect(lambda ok, msg: (results.append((ok, msg)), loop.quit()))
worker2.start()
deadline = time.time() + 120
while not results and time.time() < deadline:
    app.processEvents()
    time.sleep(0.05)
assert results and results[0][0] is False, results

print("EXPORT DEPS SMOKE PASSED")
