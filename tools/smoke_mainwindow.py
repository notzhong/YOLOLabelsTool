"""MainWindow 回归冒烟：mixin 空行修正后重新验证实例化与文案刷新链路"""

import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

from src.ui.main_window import MainWindow  # noqa: E402
from src.ui.main_window_mixins import (  # noqa: E402
    ClassActionsMixin,
    ImageActionsMixin,
    ModelActionsMixin,
    PanelsMixin,
    ThemeLanguageMixin,
)

w = MainWindow()
for mixin in (ThemeLanguageMixin, PanelsMixin, ImageActionsMixin,
              ClassActionsMixin, ModelActionsMixin):
    assert isinstance(w, mixin), mixin

w.update_ui_texts()
w.update_menu_texts()
w.update_button_texts()
w.update_panel_titles()
w.update_other_ui_elements()
assert w.windowTitle()
assert w.centralWidget() is not None
print("MainWindow OK:", w.windowTitle())
