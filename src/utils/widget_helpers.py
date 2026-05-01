"""
UI 组件辅助工具：消除 Slider / SpinBox 双向同步重复代码
"""

from typing import Callable, Optional

from PySide6.QtWidgets import QSlider, QDoubleSpinBox


class SliderSpinBoxBinder:
    """在 QSlider（整数范围）和 QDoubleSpinBox（浮点值）之间建立双向同步。

    典型场景：置信度阈值滑块(1-100) + SpinBox(0.01-1.00)。

    用法::

        binder = SliderSpinBoxBinder(
            slider, spinbox, divider=100,
            on_value_changed=self.model_manager.set_confidence_threshold
        )
        # binder 保持引用即可，无需手动管理连接
    """

    def __init__(
        self,
        slider: QSlider,
        spinbox: QDoubleSpinBox,
        divider: float,
        on_value_changed: Optional[Callable[[float], None]] = None,
    ):
        self.slider = slider
        self.spinbox = spinbox
        self.divider = divider
        self._callback = on_value_changed

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, value: int) -> None:
        float_value = value / self.divider
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(float_value)
        self.spinbox.blockSignals(False)
        if self._callback:
            self._callback(float_value)

    def _on_spinbox_changed(self, value: float) -> None:
        slider_value = int(value * self.divider)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        if self._callback:
            self._callback(value)
