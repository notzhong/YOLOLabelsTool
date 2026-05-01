"""提取的右侧面板组件：StatsPanel（统计面板）和 ModelInfoPanel（模型信息面板）"""

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QDoubleSpinBox,
)
from PySide6.QtGui import QFont
from src.utils.i18n import tr
from src.utils.widget_helpers import SliderSpinBoxBinder


class StatsPanel(QGroupBox):
    """标注统计面板"""

    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stats_counts = {"total": 0, "annotated": 0, "unannotated": 0}
        self._build_ui()

    def _build_ui(self):
        self.setTitle(tr("annotation_stats"))
        layout = QVBoxLayout(self)

        self.stats_total_label = QLabel(tr("total_images_zero"))
        layout.addWidget(self.stats_total_label)

        self.stats_annotated_label = QLabel(tr("annotated_images_zero"))
        layout.addWidget(self.stats_annotated_label)

        self.stats_unannotated_label = QLabel(tr("unannotated_images_zero"))
        layout.addWidget(self.stats_unannotated_label)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(
            [tr("class_id"), tr("class_name"), tr("annotation_count")]
        )
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setMaximumHeight(200)
        layout.addWidget(self.stats_table)

        self.btn_refresh_stats = QPushButton(tr("refresh_stats"))
        self.btn_refresh_stats.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(self.btn_refresh_stats)

        layout.addStretch()

    def update_statistics(self, image_manager, annotation_manager, class_manager):
        """更新统计信息"""
        total_images = image_manager.get_image_count()

        if total_images == 0:
            self.stats_total_label.setText(tr("total_images_zero"))
            self.stats_annotated_label.setText(tr("annotated_images_zero"))
            self.stats_unannotated_label.setText(tr("unannotated_images_zero"))
            self.stats_table.setRowCount(0)
            return

        all_anns = annotation_manager.get_all_annotations()
        annotated_paths = set(all_anns.keys())
        annotated_count = len(annotated_paths)
        unannotated_count = total_images - annotated_count

        class_counts = {}
        for _, annotations in all_anns.items():
            for annotation in annotations:
                class_id = annotation.class_id
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        self._stats_counts = {
            "total": total_images,
            "annotated": annotated_count,
            "unannotated": unannotated_count,
        }

        self.stats_total_label.setText(
            tr("total_images_label") + f" {total_images}"
        )
        self.stats_annotated_label.setText(
            tr("annotated_images_label") + f" {annotated_count}"
        )
        self.stats_unannotated_label.setText(
            tr("unannotated_images_label") + f" {unannotated_count}"
        )

        self.stats_table.setRowCount(len(class_counts))
        self.stats_table.setSortingEnabled(False)

        sorted_class_ids = sorted(class_counts.keys())
        for row, class_id in enumerate(sorted_class_ids):
            count = class_counts[class_id]
            class_info = class_manager.get_class(class_id)
            class_name = (
                class_info["name"]
                if class_info
                else tr("unknown_class").replace("{class_id}", str(class_id))
            )

            id_item = QTableWidgetItem(str(class_id))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.stats_table.setItem(row, 0, id_item)

            name_item = QTableWidgetItem(class_name)
            self.stats_table.setItem(row, 1, name_item)

            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.stats_table.setItem(row, 2, count_item)

        self.stats_table.setSortingEnabled(True)
        self.stats_table.resizeColumnsToContents()

    def update_language(self):
        """语言切换时更新文本"""
        self.setTitle(tr("annotation_stats"))
        self.stats_table.setHorizontalHeaderLabels(
            [tr("class_id"), tr("class_name"), tr("annotation_count")]
        )
        self.btn_refresh_stats.setText(tr("refresh_stats"))

        if self._stats_counts["total"] != 0:
            c = self._stats_counts
            self.stats_total_label.setText(
                f"{tr('total_images_label')} {c['total']}"
            )
            self.stats_annotated_label.setText(
                f"{tr('annotated_images_label')} {c['annotated']}"
            )
            self.stats_unannotated_label.setText(
                f"{tr('unannotated_images_label')} {c['unannotated']}"
            )
        else:
            self.stats_total_label.setText(tr("total_images_zero"))
            self.stats_annotated_label.setText(tr("annotated_images_zero"))
            self.stats_unannotated_label.setText(tr("unannotated_images_zero"))


class ModelInfoPanel(QGroupBox):
    """模型信息面板"""

    confidence_changed = Signal(float)
    iou_changed = Signal(float)
    unload_requested = Signal()
    train_requested = Signal()
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        self.setTitle(tr("model_info_panel"))
        layout = QVBoxLayout(self)

        # 模型名称
        self.model_name_label = QLabel(tr("model_not_loaded"))
        layout.addWidget(self.model_name_label)

        # 模型路径
        self.model_path_label = QLabel(tr("model_path_none"))
        path_font = QFont()
        path_font.setPointSize(9)
        self.model_path_label.setFont(path_font)
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.model_path_label)

        # 类别数量
        self.model_classes_label = QLabel(tr("model_classes_zero"))
        layout.addWidget(self.model_classes_label)

        # 置信度阈值
        conf_layout = QHBoxLayout()
        self.conf_label = QLabel(tr("confidence"))
        self.conf_label.setFixedWidth(70)
        conf_layout.addWidget(self.conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        conf_layout.addWidget(self.conf_slider)

        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.00)
        self.conf_spinbox.setSingleStep(0.01)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.setFixedWidth(60)
        conf_layout.addWidget(self.conf_spinbox)

        self._conf_binder = SliderSpinBoxBinder(
            self.conf_slider, self.conf_spinbox, divider=100,
            on_value_changed=lambda v: self.confidence_changed.emit(v),
        )

        layout.addLayout(conf_layout)

        # IoU 阈值
        iou_layout = QHBoxLayout()
        self.iou_label = QLabel(tr("iou_threshold"))
        self.iou_label.setFixedWidth(70)
        iou_layout.addWidget(self.iou_label)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.setTickInterval(10)
        iou_layout.addWidget(self.iou_slider)

        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.01, 1.00)
        self.iou_spinbox.setSingleStep(0.01)
        self.iou_spinbox.setValue(0.45)
        self.iou_spinbox.setDecimals(2)
        self.iou_spinbox.setFixedWidth(60)
        iou_layout.addWidget(self.iou_spinbox)

        self._iou_binder = SliderSpinBoxBinder(
            self.iou_slider, self.iou_spinbox, divider=100,
            on_value_changed=lambda v: self.iou_changed.emit(v),
        )

        layout.addLayout(iou_layout)

        # 默认禁用阈值控件
        self.conf_slider.setEnabled(False)
        self.conf_spinbox.setEnabled(False)
        self.iou_slider.setEnabled(False)
        self.iou_spinbox.setEnabled(False)

        # 状态指示器
        self.model_status_indicator = QLabel("● " + tr("model_status_unloaded"))
        self.model_status_indicator.setStyleSheet(
            "color: #ff6b6b; font-weight: bold;"
        )
        layout.addWidget(self.model_status_indicator)

        # 操作按钮
        btn_layout = QHBoxLayout()

        self.btn_unload_model = QPushButton(tr("unload_model"))
        self.btn_unload_model.clicked.connect(self.unload_requested.emit)
        self.btn_unload_model.setEnabled(False)
        btn_layout.addWidget(self.btn_unload_model)

        self.btn_train_model = QPushButton(tr("train_model_btn"))
        self.btn_train_model.clicked.connect(self.train_requested.emit)
        btn_layout.addWidget(self.btn_train_model)

        self.btn_refresh_info = QPushButton(tr("refresh_info"))
        self.btn_refresh_info.clicked.connect(self.refresh_requested.emit)
        btn_layout.addWidget(self.btn_refresh_info)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def update_info(self, model_manager):
        """根据模型管理器状态更新面板内容"""
        model_info = model_manager.get_model_info()

        if not model_info["loaded"]:
            self.model_name_label.setText(tr("model_not_loaded"))
            self.model_path_label.setText(
                f"{tr('model_path')} {tr('none')}"
            )
            self.model_classes_label.setText(f"{tr('model_classes')} 0")
            self.model_status_indicator.setText("● " + tr("model_status_unloaded"))
            self.model_status_indicator.setStyleSheet(
                "color: #ff6b6b; font-weight: bold;"
            )
            self.conf_slider.setEnabled(False)
            self.conf_spinbox.setEnabled(False)
            self.iou_slider.setEnabled(False)
            self.iou_spinbox.setEnabled(False)
            self.btn_unload_model.setEnabled(False)
            return

        model_name = (
            os.path.basename(model_info["path"])
            if model_info["path"]
            else tr("unknown_model")
        )
        self.model_name_label.setText(f"{tr('model_label')} {model_name}")
        self.model_path_label.setText(
            f"{tr('model_path')} {model_info['path']}"
        )
        self.model_classes_label.setText(
            f"{tr('model_classes')} {model_info['class_count']}"
        )
        self.model_status_indicator.setText("● " + tr("model_status_loaded"))
        self.model_status_indicator.setStyleSheet(
            "color: #6bff6b; font-weight: bold;"
        )

        conf_value = model_info["confidence_threshold"]
        iou_value = model_info["iou_threshold"]

        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(int(conf_value * 100))
        self.conf_slider.setEnabled(True)
        self.conf_slider.blockSignals(False)

        self.conf_spinbox.blockSignals(True)
        self.conf_spinbox.setValue(conf_value)
        self.conf_spinbox.setEnabled(True)
        self.conf_spinbox.blockSignals(False)

        self.iou_slider.blockSignals(True)
        self.iou_slider.setValue(int(iou_value * 100))
        self.iou_slider.setEnabled(True)
        self.iou_slider.blockSignals(False)

        self.iou_spinbox.blockSignals(True)
        self.iou_spinbox.setValue(iou_value)
        self.iou_spinbox.setEnabled(True)
        self.iou_spinbox.blockSignals(False)

        self.btn_unload_model.setEnabled(True)

    def update_language(self, model_manager):
        """语言切换时更新文本"""
        self.setTitle(tr("model_info_panel"))
        self.conf_label.setText(tr("confidence"))
        self.iou_label.setText(tr("iou_threshold"))
        self.btn_unload_model.setText(tr("unload_model"))
        self.btn_train_model.setText(tr("train_model_btn"))
        self.btn_refresh_info.setText(tr("refresh_info"))
        self.update_info(model_manager)
