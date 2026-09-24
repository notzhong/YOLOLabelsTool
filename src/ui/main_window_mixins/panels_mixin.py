"""
面板构建 Mixin：左/中/右三栏与分割器的创建

从 main_window.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 MainWindow 的属性与其他 Mixin 方法。
"""

import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget, QLabel,
    QPushButton, QGroupBox,
    QAbstractItemView,
)
from PySide6.QtCore import Qt

from src.ui.annotation_canvas import AnnotationCanvas
from src.ui.panels import StatsPanel, ModelInfoPanel
from src.utils.i18n import tr

class PanelsMixin:
    """面板构建 Mixin：左/中/右三栏与分割器的创建"""



    def create_left_panel(self) -> QWidget:
        """创建左侧图片列表面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        self.left_panel_title_label = QLabel(tr("image_list"))
        self.left_panel_title_label.setAlignment(Qt.AlignCenter)
        self.left_panel_title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        layout.addWidget(self.left_panel_title_label)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        self.btn_load_folder = QPushButton(tr("load_folder"))
        self.btn_load_folder.clicked.connect(self.load_image_folder)
        btn_layout.addWidget(self.btn_load_folder)

        self.btn_close_folder = QPushButton(tr("close_folder"))
        self.btn_close_folder.clicked.connect(self.close_image_folder)
        btn_layout.addWidget(self.btn_close_folder)
        
        self.btn_prev = QPushButton(tr("previous_image"))
        self.btn_prev.clicked.connect(self.prev_image)
        btn_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton(tr("next_image"))
        self.btn_next.clicked.connect(self.next_image)
        btn_layout.addWidget(self.btn_next)
        
        layout.addLayout(btn_layout)
        
        # 图片列表
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.on_image_item_clicked)
        self.image_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list_widget.customContextMenuRequested.connect(self._on_image_list_context_menu)
        self.image_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.image_list_widget)
        
        # 统计信息
        self.stats_label = QLabel(tr("no_image_loaded"))
        self.stats_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        layout.addWidget(self.stats_label)
        
        return panel

    def create_center_panel(self) -> QWidget:
        """创建中间图片显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        self.center_panel_title_label = QLabel(tr("image_annotation"))
        self.center_panel_title_label.setAlignment(Qt.AlignCenter)
        self.center_panel_title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        layout.addWidget(self.center_panel_title_label)
        
        # 工具栏
        tool_layout = QHBoxLayout()
        
        self.btn_fit = QPushButton(tr("fit_to_window"))
        self.btn_fit.clicked.connect(self.fit_to_window)
        tool_layout.addWidget(self.btn_fit)
        
        self.btn_zoom_in = QPushButton(tr("zoom_in"))
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        tool_layout.addWidget(self.btn_zoom_in)
        
        self.btn_zoom_out = QPushButton(tr("zoom_out"))
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        tool_layout.addWidget(self.btn_zoom_out)
        
        self.btn_reset = QPushButton(tr("reset"))
        self.btn_reset.clicked.connect(self.reset_view)
        tool_layout.addWidget(self.btn_reset)
        
        tool_layout.addStretch()
        
        layout.addLayout(tool_layout)
        
        # 图片显示区域 — 用 AnnotationCanvas 替换原 QGraphicsView
        self.canvas = AnnotationCanvas()
        self.canvas.set_class_manager(self.class_manager)
        self.canvas.annotation_created.connect(self._on_canvas_annotation_created)
        self.canvas.annotation_changed.connect(self.save_annotations)
        self.canvas.annotation_deleted.connect(self._on_canvas_annotation_deleted)

        layout.addWidget(self.canvas)
        
        # 状态信息
        self.image_info_label = QLabel(tr("no_image_loaded"))
        self.image_info_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        layout.addWidget(self.image_info_label)
        
        return panel

    def create_right_panel(self) -> QWidget:
        """创建右侧类别管理面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        self.right_panel_title_label = QLabel(tr("class_management"))
        self.right_panel_title_label.setAlignment(Qt.AlignCenter)
        self.right_panel_title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        layout.addWidget(self.right_panel_title_label)
        
        # 创建垂直分割器，允许用户调整各区域高度
        self._right_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self._right_splitter, 1)  # 第二个参数为1表示填充剩余空间
        
        # 类别列表 - 保存引用
        self.class_group = QGroupBox(tr("annotation_classes"))
        class_layout = QVBoxLayout(self.class_group)
        
        self.class_list_widget = QListWidget()
        self.class_list_widget.itemClicked.connect(self.on_class_item_clicked)
        self.class_list_widget.itemDoubleClicked.connect(self._on_class_item_double_clicked)
        self.class_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.class_list_widget.customContextMenuRequested.connect(self._on_class_list_context_menu)
        class_layout.addWidget(self.class_list_widget)
        
        # 类别操作按钮
        class_btn_layout = QHBoxLayout()
        
        self.btn_add_class = QPushButton(tr("add"))
        self.btn_add_class.clicked.connect(self.add_class)
        class_btn_layout.addWidget(self.btn_add_class)
        
        self.btn_edit_class = QPushButton(tr("edit"))
        self.btn_edit_class.clicked.connect(self.edit_class)
        class_btn_layout.addWidget(self.btn_edit_class)
        
        self.btn_delete_class = QPushButton(tr("delete"))
        self.btn_delete_class.clicked.connect(self.delete_class)
        class_btn_layout.addWidget(self.btn_delete_class)
        
        self.btn_clear_classes = QPushButton(tr("clear_all_classes"))  # 清空按钮，文本将通过翻译设置
        self.btn_clear_classes.clicked.connect(self.clear_all_classes)
        class_btn_layout.addWidget(self.btn_clear_classes)
        
        class_layout.addLayout(class_btn_layout)
        
        self._right_splitter.addWidget(self.class_group)
        
        # 标注操作 - 保存引用
        self.annotation_group = QGroupBox(tr("annotation_operations"))
        annotation_layout = QVBoxLayout(self.annotation_group)
        
        self.btn_delete_annotation = QPushButton(tr("delete_selected_annotation"))
        self.btn_delete_annotation.clicked.connect(self.delete_selected_annotation)
        annotation_layout.addWidget(self.btn_delete_annotation)
        
        self.btn_clear_all = QPushButton(tr("clear_all_annotations"))
        self.btn_clear_all.clicked.connect(self.clear_all_annotations)
        annotation_layout.addWidget(self.btn_clear_all)
        
        annotation_layout.addStretch()
        
        self._right_splitter.addWidget(self.annotation_group)
        
        # 导出操作 - 保存引用
        self.export_group = QGroupBox(tr("data_export"))
        export_layout = QVBoxLayout(self.export_group)
        
        self.btn_export_yolo = QPushButton(tr("export_yolo"))
        self.btn_export_yolo.clicked.connect(self.export_yolo_format)
        export_layout.addWidget(self.btn_export_yolo)
        
        self.btn_export_split = QPushButton(tr("export_dataset_split"))
        self.btn_export_split.clicked.connect(self.export_dataset_split)
        export_layout.addWidget(self.btn_export_split)
        
        export_layout.addStretch()
        
        self._right_splitter.addWidget(self.export_group)
        
        # 标注统计面板
        self.stats_panel = StatsPanel()
        self.stats_panel.refresh_requested.connect(self._on_stats_refresh)
        self._right_splitter.addWidget(self.stats_panel)
        
        # 模型信息面板（默认隐藏）
        self.model_info_panel = ModelInfoPanel()
        self.model_info_panel.setVisible(False)
        self.model_info_panel.confidence_changed.connect(self._on_panel_conf_changed)
        self.model_info_panel.iou_changed.connect(self._on_panel_iou_changed)
        self.model_info_panel.unload_requested.connect(self.unload_model)
        self.model_info_panel.train_requested.connect(self.train_model)
        self.model_info_panel.refresh_requested.connect(self._on_model_info_refresh)
        
        self._right_splitter.addWidget(self.model_info_panel)

        # 设置分割器的初始大小比例
        self._right_splitter.setSizes(self._load_splitter_sizes('right_splitter', [150, 100, 100, 200, 150]))
        
        # 设置分割器手柄样式
        self._right_splitter.setHandleWidth(6)
        
        return panel

    def _load_splitter_sizes(self, key: str, default: list) -> list:
        """从配置文件加载分割器大小"""
        try:
            if self.config.has_option("window", key):
                sizes = json.loads(self.config.get("window", key))
                if isinstance(sizes, list) and len(sizes) == len(default):
                    return sizes
        except Exception:
            pass
        return default
