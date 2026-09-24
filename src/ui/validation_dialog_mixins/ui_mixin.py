"""
界面构建 Mixin

从 validation_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 ValidationDialog 的属性与其他 Mixin 方法。
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.utils.i18n import tr
from src.utils.widget_helpers import SliderSpinBoxBinder


class ValidationUiMixin:
    """界面构建与来源控制 Mixin：左栏控件、模型加载、参数行与来源切换"""


    def _init_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        left_container = QWidget()
        left_panel = QVBoxLayout(left_container)
        left_panel.setContentsMargins(5, 5, 5, 5)

        model_group = QGroupBox(tr("model"))
        model_layout = QVBoxLayout(model_group)
        self.model_status_label = QLabel("")
        self.btn_load_model = QPushButton(tr("load_model"))
        self.btn_load_model.clicked.connect(self._load_model)
        model_layout.addWidget(self.model_status_label)
        model_layout.addWidget(self.btn_load_model)
        left_panel.addWidget(model_group)

        source_group = QGroupBox(tr("validation_source"))
        source_layout = QVBoxLayout(source_group)

        self.source_combo = QComboBox()
        self.source_combo.addItems([tr("source_window"), tr("source_region"), tr("source_image")])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)

        self.window_info_label = QLabel(tr("window_not_selected"))
        self.btn_pick_window = QPushButton(tr("pick_window"))
        self.btn_pick_window.clicked.connect(self._pick_window)
        source_layout.addWidget(self.window_info_label)
        source_layout.addWidget(self.btn_pick_window)

        self.region_info_label = QLabel(tr("region_not_selected"))
        self.btn_pick_region = QPushButton(tr("pick_region"))
        self.btn_pick_region.clicked.connect(self._pick_region)
        source_layout.addWidget(self.region_info_label)
        source_layout.addWidget(self.btn_pick_region)

        image_row = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.btn_browse_image = QPushButton(tr("browse"))
        self.btn_browse_image.clicked.connect(self._browse_image)
        image_row.addWidget(self.image_path_edit, 1)
        image_row.addWidget(self.btn_browse_image)
        source_layout.addLayout(image_row)

        left_panel.addWidget(source_group)

        params_group = QGroupBox(tr("model_params"))
        params_layout = QVBoxLayout(params_group)

        self._conf_binder = self._add_slider_row(
            tr("confidence"), params_layout,
            slider_range=(1, 100), spin_range=(0.01, 1.00),
            spin_step=0.01, decimals=2, divider=100,
            initial=self.model_manager.confidence_threshold,
            on_value_changed=self.model_manager.set_confidence_threshold,
        )

        self._iou_binder = self._add_slider_row(
            tr("iou_threshold"), params_layout,
            slider_range=(1, 100), spin_range=(0.01, 1.00),
            spin_step=0.01, decimals=2, divider=100,
            initial=self.model_manager.iou_threshold,
            on_value_changed=self.model_manager.set_iou_threshold,
        )

        left_panel.addWidget(params_group)

        display_group = QGroupBox(tr("display_settings"))
        display_layout = QVBoxLayout(display_group)

        self._font_binder = self._add_slider_row(
            tr("label_font_size"), display_layout,
            slider_range=(1, 20), spin_range=(0.1, 2.0),
            spin_step=0.1, decimals=1, divider=10,
            initial=self.label_font_size,
            on_value_changed=lambda v: setattr(self, 'label_font_size', v),
        )

        self.show_conf_check = QCheckBox(tr("show_confidence"))
        self.show_conf_check.setChecked(self.show_confidence)
        self.show_conf_check.toggled.connect(self._on_show_conf_toggled)
        display_layout.addWidget(self.show_conf_check)

        left_panel.addWidget(display_group)

        self.btn_toggle = QPushButton(tr("start_detect"))
        self.btn_toggle.clicked.connect(self._toggle_detect)
        left_panel.addWidget(self.btn_toggle)

        self.status_label = QLabel(tr("ready"))
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()

        self.preview_label = QLabel(tr("no_image_loaded"))
        self.preview_label.setAlignment(Qt.AlignCenter)

        splitter.addWidget(left_container)
        splitter.addWidget(self.preview_label)
        splitter.setStretchFactor(0, 0)  # 左侧不拉伸
        splitter.setStretchFactor(1, 1)  # 右侧自适应拉伸
        splitter.setSizes([280, 600])

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        self._on_source_changed()

    def _selected_source(self) -> int:
        return self.source_combo.currentIndex()

    def _update_model_status(self):
        if self.model_manager.is_model_loaded():
            info = self.model_manager.get_model_info()
            name = Path(info.get("path", "")).name
            self.model_status_label.setText(tr("model_loaded_status").replace("{model_name}", name))
        else:
            self.model_status_label.setText(tr("model_not_loaded"))

    def _load_model(self):
        if not self.model_manager.is_available():
            QMessageBox.warning(self, tr("warning"), tr("ultralytics_not_installed"))
            return
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("select_pretrained_model_file"),
            self._last_browse_path,
            tr("model_file_filter"),
        )
        if model_path:
            self._last_browse_path = str(Path(model_path).parent)
            if self.model_manager.load_model(model_path):
                self._update_model_status()
            else:
                QMessageBox.warning(
                    self,
                    tr("warning"),
                    tr("load_model_failed").replace("{model_path}", model_path),
                )

    def _add_slider_row(self, label_text: str, parent_layout,
                        slider_range: tuple, spin_range: tuple,
                        spin_step: float, decimals: int, divider: float,
                        initial: float, on_value_changed):
        """Create a labeled slider + spinbox row and return its binder."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(*slider_range)
        slider.setValue(int(initial * divider))
        spin = QDoubleSpinBox()
        spin.setRange(*spin_range)
        spin.setSingleStep(spin_step)
        spin.setValue(initial)
        spin.setDecimals(decimals)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return SliderSpinBoxBinder(slider, spin, divider=divider,
                                   on_value_changed=on_value_changed)

    def _on_source_changed(self):
        source = self._selected_source()
        is_window = source == self.SOURCE_WINDOW
        is_region = source == self.SOURCE_REGION
        is_image = source == self.SOURCE_IMAGE

        if self.picking_window and not is_window:
            self._stop_window_pick(confirmed=False)

        self.window_info_label.setVisible(is_window)
        self.btn_pick_window.setVisible(is_window)

        self.region_info_label.setVisible(is_region)
        self.btn_pick_region.setVisible(is_region)

        self.image_path_edit.setVisible(is_image)
        self.btn_browse_image.setVisible(is_image)

    def _browse_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("select_image_file_dialog_title"),
            self._last_browse_path,
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif)",
        )
        if image_path:
            self.image_path_edit.setText(image_path)
            self._last_browse_path = str(Path(image_path).parent)
