"""P2 拆分后的 offscreen 冒烟：两个对话框实例化 + 关键路径调用"""

import os
import pathlib
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

# ---------------- TrainDialog ----------------
from src.core.model_manager import ModelManager  # noqa: E402
from src.ui.train_dialog import TrainDialog  # noqa: E402
from src.ui.train_dialog_mixins import (  # noqa: E402
    TrainActionsMixin,
    TrainBrowseMixin,
    TrainConfigMixin,
    TrainTabsMixin,
)
from src.utils.win32_helpers import PlatformError, is_windows  # noqa: E402

d = TrainDialog()
for mixin in (TrainTabsMixin, TrainBrowseMixin, TrainConfigMixin, TrainActionsMixin):
    assert isinstance(d, mixin), mixin
assert isinstance(d, TrainTabsMixin)

cfg = d.collect_config_from_ui()
assert {"model_path", "data_yaml", "epochs", "imgsz", "batch", "optimizer",
        "augment", "device"} <= set(cfg), sorted(cfg)

d.config["epochs"] = 123
d.load_config_to_ui()
assert d.epochs_spin.value() == 123
assert d.collect_config_from_ui()["epochs"] == 123

# 互斥开关 + 增强联动（通过控件信号驱动，覆盖 toggled 连接）
d.resume_checkbox.setChecked(True)
assert d.resume_path_edit.isEnabled()
d.incremental_checkbox.setChecked(True)
assert not d.resume_checkbox.isChecked()
assert d.incremental_checkbox.isChecked()
assert d.incremental_hint.isVisibleTo(d)
assert not d.resume_path_edit.isEnabled()
d.augment_checkbox.setChecked(False)
assert not d.mixup_spin.isEnabled()
d.augment_checkbox.setChecked(True)
assert d.mixup_spin.isEnabled()

d.on_config_changed()
d.save_config_on_exit()
d.close()
print(f"TrainDialog OK ({len(cfg)} 个配置键, 5 个标签页)")

# ---------------- ValidationDialog ----------------
from src.ui.validation_dialog import DXCAM_AVAILABLE, ValidationDialog  # noqa: E402
from src.ui.validation_dialog_mixins import unicode_text  # noqa: E402

v = ValidationDialog(None, ModelManager())
v._update_model_status()
for i in (v.SOURCE_WINDOW, v.SOURCE_REGION, v.SOURCE_IMAGE):
    v.source_combo.setCurrentIndex(i)
    v._on_source_changed()
v._selected_source()

sb = v._get_screen_bounds() if is_windows() else None
if is_windows():
    assert sb[2] > sb[0] and sb[3] > sb[1]
    rect = (sb[0] + 1, sb[1] + 1, sb[0] + 31, sb[1] + 31)
    assert v._normalize_region(rect) == rect
    assert v._normalize_region((50, 50, 10, 10)) is None
else:
    # 现状：win32 守卫后非 Windows 调用即抛 PlatformError（拆分前同样行为）
    for call in (v._get_screen_bounds, lambda: v._normalize_region((10, 10, 50, 50))):
        try:
            call()
        except PlatformError:
            pass
        else:
            raise AssertionError("非 Windows 平台应抛 PlatformError")
logical = QRect(10, 20, 300, 200)
back = v._physical_to_logical_rect(v._logical_to_physical_rect(logical))
# right()/bottom() 为闭区间，往返存在 1px 量化差（原实现行为）
assert abs(back.x() - 10) <= 1 and abs(back.y() - 20) <= 1
assert abs(back.width() - 300) <= 1 and abs(back.height() - 200) <= 1
assert v._capture_frame() is None  # 未选窗口 -> None

frame = np.zeros((120, 200, 3), dtype=np.uint8)
v._update_preview(frame)

img = np.zeros((60, 240, 3), dtype=np.uint8)
unicode_text._draw_unicode_texts_batch(
    img, [("person 0.95", (5, 30), 0.5, (0, 255, 0), 1)]
)
assert img.sum() > 0

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    Image.fromarray(np.full((20, 20, 3), 7, dtype=np.uint8)).save(tmp.name)
    read_back = v._read_image(tmp.name)
assert read_back is not None and read_back.shape[:2] == (20, 20)

v.close()
print(f"ValidationDialog OK, DXCAM_AVAILABLE={DXCAM_AVAILABLE}")
print("SMOKE PASSED")
