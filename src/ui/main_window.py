"""
主窗口实现
"""

import os
import json
import configparser
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QLabel, QPushButton,
    QFileDialog, QMessageBox, QToolBar, QStatusBar,
    QMenuBar, QMenu, QDockWidget, QFrame, QSizePolicy,
    QScrollArea, QGroupBox, QComboBox, QCheckBox,
    QProgressDialog, QApplication, QAbstractItemView
)
from PySide6.QtCore import (
    Qt,
)
from PySide6.QtGui import (
    QPixmap, QColor,
    QAction, QKeySequence, QIcon, QShortcut,
)

from src.ui.annotation_canvas import AnnotationCanvas
from src.ui.panels import StatsPanel, ModelInfoPanel

from src.core.annotation import Annotation, AnnotationManager
from src.core.image_manager import ImageManager
from src.core.class_manager import ClassManager
from src.core.model_manager import ModelManager
from src.utils.yolo_exporter import YOLOExporter

from src.utils.logger import get_logger_simple
from src.utils.i18n import TranslationManager, tr, T


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 主题相关 - 必须在任何方法调用之前初始化
        self.current_theme = "dark"  # 默认使用黑夜主题
        
        # 配置管理器
        self.config_file_path = Path("config/config.ini")
        self.config = configparser.ConfigParser()
        
        # 日志记录器
        self.logger = get_logger_simple(__name__)
        
        # 初始化管理器
        self.image_manager = ImageManager()
        self.class_manager = ClassManager()
        self.annotation_manager = AnnotationManager()
        self.model_manager = ModelManager()
        self.yolo_exporter = YOLOExporter()
        
        # 当前状态
        self.current_image_path: Optional[str] = None
        self.current_image_index: int = 0
        self.selected_class_id: int = 0
        # 统计缓存移至 StatsPanel._stats_counts
        self._last_browse_path = str(Path.cwd())  # 上次浏览路径
        self._last_folder_path = ""  # 上次打开的图片文件夹

        # 加载设置
        self.load_settings()
        
        # 初始化UI
        self.init_ui()
        self.canvas.selected_class_id = self.selected_class_id  # 同步初始类别
        self.init_actions()
        self.init_menus()
        # 同步主题菜单勾选状态（必须在菜单创建后执行）
        self.action_dark_theme.setChecked(self.current_theme == "dark")
        self.action_light_theme.setChecked(self.current_theme == "light")
        self.action_colorful_theme.setChecked(self.current_theme == "colorful")
        self.init_toolbar()
        self.init_statusbar()
        
        # 更新类别列表
        self.update_class_list()

        # 自动加载上次打开的文件夹
        if self._last_folder_path and Path(self._last_folder_path).exists():
            self.logger.info(f"自动加载上次打开的文件夹: {self._last_folder_path}")
            self.load_image_folder_by_path(self._last_folder_path)

        # 设置窗口属性
        self.setWindowTitle(tr("yolo_label_tool"))
        self.resize(1200, 800)
    
    def init_ui(self):
        """初始化用户界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建分割器
        self._main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self._main_splitter)
        
        # 左侧面板 - 图片列表
        left_panel = self.create_left_panel()
        self._main_splitter.addWidget(left_panel)
        
        # 中间面板 - 图片显示
        center_panel = self.create_center_panel()
        self._main_splitter.addWidget(center_panel)
        
        # 右侧面板 - 类别管理
        right_panel = self.create_right_panel()
        self._main_splitter.addWidget(right_panel)
        
        # 设置分割器初始大小（后续会被配置覆盖）
        self._main_splitter.setSizes(self._load_splitter_sizes('main_splitter', [200, 600, 200]))
        
        # 加载QSS样式
        self.load_qss_style()

        if Path('icon.ico').exists():
            self.setWindowIcon(QPixmap('icon.ico'))

        # 设置键盘快捷键
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        """设置键盘快捷键"""
        # ←/→ 翻图
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_image)
        # Tab/Shift+Tab 循环选择类别
        QShortcut(QKeySequence(Qt.Key_Tab), self, self._select_next_class)
        QShortcut(QKeySequence(Qt.SHIFT | Qt.Key_Tab), self, self._select_prev_class)

    def _select_next_class(self):
        """选择下一个类别"""
        count = self.class_list_widget.count()
        if count == 0:
            return
        current_row = self.class_list_widget.currentRow()
        next_row = (current_row + 1) % count
        self.class_list_widget.setCurrentRow(next_row)
        item = self.class_list_widget.item(next_row)
        if item:
            self.on_class_item_clicked(item)
            self.update_status(
                tr("selected_class").replace(
                    "{class_name}", self.class_manager.get_class_name(self.selected_class_id)
                )
            )

    def _select_prev_class(self):
        """选择上一个类别"""
        count = self.class_list_widget.count()
        if count == 0:
            return
        current_row = self.class_list_widget.currentRow()
        prev_row = (current_row - 1) % count
        self.class_list_widget.setCurrentRow(prev_row)
        item = self.class_list_widget.item(prev_row)
        if item:
            self.on_class_item_clicked(item)
            self.update_status(
                tr("selected_class").replace(
                    "{class_name}", self.class_manager.get_class_name(self.selected_class_id)
                )
            )

    def load_qss_style(self):
        """加载QSS样式文件"""
        self.logger.info(f"加载QSS样式，当前主题: {self.current_theme}")
        self.apply_theme(self.current_theme)
    
    def apply_theme(self, theme_name: str):
        """应用指定主题"""
        if theme_name == "dark":
            qss_path = Path("qss/dark_theme.qss")
        elif theme_name == "light":
            qss_path = Path("qss/light_theme.qss")
        elif theme_name == "colorful":
            qss_path = Path("qss/colorful_theme.qss")
        else:
            self.logger.warning(f"未知主题: {theme_name}, 使用默认主题")
            return
        
        if qss_path.exists():
            try:
                with open(qss_path, 'r', encoding='utf-8') as f:
                    qss_content = f.read()
                self.setStyleSheet(qss_content)
                self.current_theme = theme_name
                
                # 根据主题更新标题标签颜色
                self.update_title_colors_for_theme(theme_name)
                # 更新主题菜单选中标记（菜单存在时）
                if hasattr(self, 'action_dark_theme'):
                    self.action_dark_theme.setChecked(theme_name == "dark")
                    self.action_light_theme.setChecked(theme_name == "light")
                    self.action_colorful_theme.setChecked(theme_name == "colorful")
                
                self.logger.info(f"已应用主题: {theme_name}")
            except Exception as e:
                self.logger.error(f"加载主题样式文件失败: {e}, 将使用Qt默认样式")
        else:
            self.logger.warning(f"主题文件不存在: {qss_path}, 将使用Qt默认样式")
    
    def update_title_colors_for_theme(self, theme_name: str):
        """根据主题更新标题标签颜色"""
        if theme_name == "light":
            # 白天主题：黑色标题，深灰色状态
            title_style = "font-weight: bold; font-size: 14px; color: #333333;"
            status_style = "color: #777777; font-size: 12px;"
        else:
            # 黑夜/炫彩主题：白色标题，灰色状态
            title_style = "font-weight: bold; font-size: 14px; color: #ffffff;"
            status_style = "color: #aaaaaa; font-size: 12px;"
        
        # 查找并更新所有标题标签
        for widget in self.findChildren(QLabel):
            current_style = widget.styleSheet()
            if "font-weight: bold; font-size: 14px; color:" in current_style:
                widget.setStyleSheet(title_style)
            elif "color: #aaaaaa; font-size: 12px;" in current_style:
                widget.setStyleSheet(status_style)
    
    def switch_to_dark_theme(self):
        """切换到黑夜主题"""
        self.apply_theme("dark")
        self.save_settings()
    
    def switch_to_light_theme(self):
        """切换到白天主题"""
        self.apply_theme("light")
        self.save_settings()
    
    def switch_to_colorful_theme(self):
        """切换到炫彩主题"""
        self.apply_theme("colorful")
        self.save_settings()

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

    def init_actions(self):
        """初始化动作"""
        # 文件操作
        self.action_open_folder = QAction(tr("open_folder"), self)
        self.action_open_folder.setShortcut(QKeySequence.Open)
        self.action_open_folder.triggered.connect(self.load_image_folder)

        self.action_close_folder = QAction(tr("close_folder"), self)
        self.action_close_folder.triggered.connect(self.close_image_folder)

        self.action_save = QAction(tr("save_annotations"), self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self.save_annotations)
        
        self.action_export = QAction(tr("export_yolo_format"), self)
        self.action_export.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export.triggered.connect(self.export_yolo_format)
        
        self.action_exit = QAction(tr("exit"), self)
        self.action_exit.setShortcut(QKeySequence.Quit)
        self.action_exit.triggered.connect(self.close)
        
        # 编辑操作
        self.action_undo = QAction(tr("undo"), self)
        self.action_undo.setShortcut(QKeySequence.Undo)
        self.action_undo.triggered.connect(self.undo)
        
        self.action_redo = QAction(tr("redo"), self)
        self.action_redo.setShortcut(QKeySequence.Redo)
        self.action_redo.triggered.connect(self.redo)
        
        self.action_delete = QAction(tr("delete_selected"), self)
        self.action_delete.setShortcut(QKeySequence.Delete)
        self.action_delete.triggered.connect(self.delete_selected_annotation)
        
        # 视图操作
        self.action_zoom_in = QAction(tr("zoom_in"), self)
        self.action_zoom_in.setShortcut(QKeySequence.ZoomIn)
        self.action_zoom_in.triggered.connect(self.zoom_in)
        
        self.action_zoom_out = QAction(tr("zoom_out"), self)
        self.action_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.action_zoom_out.triggered.connect(self.zoom_out)
        
        self.action_fit = QAction(tr("fit_to_window"), self)
        self.action_fit.setShortcut("Ctrl+F")
        self.action_fit.triggered.connect(self.fit_to_window)
        
        # 模型操作
        self.action_load_model = QAction(tr("load_model"), self)
        self.action_load_model.setShortcut("Ctrl+M")
        self.action_load_model.triggered.connect(self.load_model)
        
        self.action_model_info = QAction(tr("model_info"), self)
        self.action_model_info.triggered.connect(self.show_model_info)

        self.action_validation_window = QAction(tr("validation_window"), self)
        self.action_validation_window.triggered.connect(self.open_validation_window)

        # 标注操作
        self.action_auto_annotate = QAction(tr("auto_annotate"), self)
        self.action_auto_annotate.setShortcut("Ctrl+A")
        self.action_auto_annotate.triggered.connect(self.auto_annotate_current)
        
        self.action_batch_auto_annotate = QAction(tr("batch_auto_annotate"), self)
        self.action_batch_auto_annotate.setShortcut("Ctrl+Shift+A")
        self.action_batch_auto_annotate.triggered.connect(self.batch_auto_annotate)
        
        # 训练操作
        self.action_train_model = QAction(tr("train_model"), self)
        self.action_train_model.setShortcut("Ctrl+T")
        self.action_train_model.triggered.connect(self.train_model)
    
    def init_menus(self):
        """初始化菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        self.file_menu = menubar.addMenu(tr("file"))
        self.file_menu.addAction(self.action_open_folder)
        self.file_menu.addAction(self.action_close_folder)
        self.file_menu.addAction(self.action_save)
        self.file_menu.addAction(self.action_export)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_exit)
        
        # 编辑菜单
        self.edit_menu = menubar.addMenu(tr("edit"))
        self.edit_menu.addAction(self.action_undo)
        self.edit_menu.addAction(self.action_redo)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.action_delete)
        
        # 视图菜单
        self.view_menu = menubar.addMenu(tr("view"))
        self.view_menu.addAction(self.action_zoom_in)
        self.view_menu.addAction(self.action_zoom_out)
        self.view_menu.addAction(self.action_fit)
        
        # 类别菜单
        self.class_menu = menubar.addMenu(tr("classes"))
        self.action_load_yaml = QAction(tr("load_yaml"), self)
        self.action_load_yaml.triggered.connect(self.load_classes_from_yaml)
        self.class_menu.addAction(self.action_load_yaml)
        
        self.action_save_yaml = QAction(tr("save_yaml"), self)
        self.action_save_yaml.triggered.connect(self.save_classes_to_yaml)
        self.class_menu.addAction(self.action_save_yaml)
        
        self.class_menu.addSeparator()
        
        # 主题菜单
        self.theme_menu = menubar.addMenu(tr("theme"))
        
        self.action_dark_theme = QAction(tr("dark_theme"), self)
        self.action_dark_theme.setCheckable(True)
        self.action_dark_theme.triggered.connect(self.switch_to_dark_theme)
        self.theme_menu.addAction(self.action_dark_theme)

        self.action_light_theme = QAction(tr("light_theme"), self)
        self.action_light_theme.setCheckable(True)
        self.action_light_theme.triggered.connect(self.switch_to_light_theme)
        self.theme_menu.addAction(self.action_light_theme)

        self.action_colorful_theme = QAction(tr("colorful_theme"), self)
        self.action_colorful_theme.setCheckable(True)
        self.action_colorful_theme.triggered.connect(self.switch_to_colorful_theme)
        self.theme_menu.addAction(self.action_colorful_theme)
        
        # 模型菜单
        self.model_menu = menubar.addMenu(tr("model"))
        self.model_menu.addAction(self.action_load_model)
        self.model_menu.addAction(self.action_model_info)
        self.model_menu.addAction(self.action_validation_window)
        self.model_menu.addSeparator()
        self.model_menu.addAction(self.action_train_model)
        self.model_menu.addSeparator()
        self.model_menu.addAction(self.action_auto_annotate)
        self.model_menu.addAction(self.action_batch_auto_annotate)
        
        # 标注菜单
        self.annotate_menu = menubar.addMenu(tr("annotate"))
        self.annotate_menu.addAction(self.action_auto_annotate)
        self.annotate_menu.addAction(self.action_batch_auto_annotate)
        
        # 语言菜单
        self.language_menu = menubar.addMenu(tr("language"))
        
        self.action_chinese = QAction(tr("chinese"), self)
        self.action_chinese.triggered.connect(lambda: self.switch_language("zh_CN"))
        self.language_menu.addAction(self.action_chinese)
        
        self.action_english = QAction(tr("english"), self)
        self.action_english.triggered.connect(lambda: self.switch_language("en_US"))
        self.language_menu.addAction(self.action_english)
    
    def init_toolbar(self):
        """初始化工具栏"""
        self.toolbar = self.addToolBar(tr("main_toolbar"))
        self.toolbar.setMovable(False)
        
        self.toolbar.addAction(self.action_open_folder)
        self.toolbar.addAction(self.action_save)
        self.toolbar.addAction(self.action_export)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_undo)
        self.toolbar.addAction(self.action_redo)
        self.toolbar.addAction(self.action_delete)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_zoom_in)
        self.toolbar.addAction(self.action_zoom_out)
        self.toolbar.addAction(self.action_fit)
    
    def init_statusbar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 状态标签
        self.status_label = QLabel(tr("ready"))
        self.status_bar.addWidget(self.status_label, 1)
        
        # 图片信息
        self.status_image_info = QLabel("")
        self.status_bar.addPermanentWidget(self.status_image_info)
        
        # 标注信息
        self.status_annotation_info = QLabel("")
        self.status_bar.addPermanentWidget(self.status_annotation_info)
    
    def load_settings(self):
        """加载设置"""
        # 确保配置文件存在
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.config_file_path.exists():
            # 如果配置文件不存在，创建默认配置
            self._create_default_config()
        else:
            # 读取配置文件
            self.config.read(self.config_file_path, encoding='utf-8')
            
            # 加载窗口大小和位置
            if self.config.has_option('window', 'geometry'):
                geometry_data = self.config.get('window', 'geometry')
                if geometry_data:
                    try:
                        geometry = bytes.fromhex(geometry_data)
                        self.restoreGeometry(geometry)
                    except Exception as e:
                        self.logger.error(f"恢复窗口几何形状失败: {e}")
            
            # 加载类别设置
            if self.config.has_option('classes', 'data'):
                classes_data_str = self.config.get('classes', 'data')
                if classes_data_str:
                    try:
                        classes_data = json.loads(classes_data_str)
                        self.class_manager.load_from_list(classes_data)
                        # 不在UI初始化前更新列表，将在init_ui后调用
                    except json.JSONDecodeError as e:
                        self.logger.error(f"解析类别数据失败: {e}")
            
            # 加载主题设置
            if self.config.has_option('preferences', 'theme'):
                saved_theme = self.config.get('preferences', 'theme')
                if saved_theme in ['dark', 'light', 'colorful']:
                    self.current_theme = saved_theme
                    self.logger.info(f"加载保存的主题: {self.current_theme}")
            
            # 加载语言设置
            if self.config.has_option('preferences', 'language'):
                saved_language = self.config.get('preferences', 'language')
                if saved_language in ['zh_CN', 'en_US']:
                    # 设置翻译管理器语言并加载翻译文件
                    from src.utils.i18n import TranslationManager
                    translation_manager = TranslationManager.instance()
                    translation_manager.current_language = saved_language
                    translation_manager.load_translation_files()
                    self.logger.info(f"加载保存的语言: {saved_language}")
        
        # 加载最近文件夹
        if self.config.has_option("preferences", "last_folder"):
            self._last_folder_path = self.config.get("preferences", "last_folder")
            if self._last_folder_path:
                self.logger.info(f"加载保存的文件夹: {self._last_folder_path}")
    
    def save_settings(self):
        """保存设置"""
        try:
            # 确保配置文件存在
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建或更新配置
            if not self.config.has_section('window'):
                self.config.add_section('window')
            if not self.config.has_section('classes'):
                self.config.add_section('classes')
            if not self.config.has_section('preferences'):
                self.config.add_section('preferences')
            
            # 保存窗口状态
            geometry_bytes = self.saveGeometry()
            if geometry_bytes and not geometry_bytes.isEmpty():
                # 将QByteArray转换为bytes再转换为hex
                geometry_data = bytes(geometry_bytes.data()).hex()
                self.config.set('window', 'geometry', geometry_data)
            else:
                self.config.set('window', 'geometry', '')

            # 保存分割器位置
            if hasattr(self, '_main_splitter'):
                self.config.set('window', 'main_splitter',
                                json.dumps(self._main_splitter.sizes()))
            if hasattr(self, '_right_splitter'):
                self.config.set('window', 'right_splitter',
                                json.dumps(self._right_splitter.sizes()))
            
            # 保存类别设置
            classes_data = self.class_manager.get_classes_list()
            if classes_data:
                classes_data_str = json.dumps(classes_data, ensure_ascii=False)
                self.config.set('classes', 'data', classes_data_str)
            else:
                self.config.set('classes', 'data', '[]')
            
            # 保存主题设置
            self.config.set('preferences', 'theme', self.current_theme)
            
            # 保存语言设置
            from src.utils.i18n import TranslationManager
            translation_manager = TranslationManager.instance()
            self.config.set('preferences', 'language', translation_manager.get_current_language())

            # 保存最近文件夹
            self.config.set('preferences', 'last_folder', self._last_folder_path)
            
            # 保存配置文件
            with open(self.config_file_path, 'w', encoding='utf-8') as configfile:
                self.config.write(configfile)
                
        except Exception as e:
            self.logger.error(f"保存设置失败: {e}")
            # 即使保存失败也不影响程序关闭
    
    def _create_default_config(self):
        """创建默认配置文件"""
        # 创建默认配置节
        if not self.config.has_section('window'):
            self.config.add_section('window')
        if not self.config.has_section('classes'):
            self.config.add_section('classes')
        if not self.config.has_section('preferences'):
            self.config.add_section('preferences')
        
        # 设置默认值
        self.config.set('window', 'width', '1200')
        self.config.set('window', 'height', '800')
        self.config.set('window', 'geometry', '')
        
        self.config.set('classes', 'data', '[]')
        
        self.config.set('preferences', 'recent_folder', '')
        self.config.set('preferences', 'default_export_path', '')
        self.config.set('preferences', 'auto_save', 'false')
        
        # 保存配置文件
        with open(self.config_file_path, 'w', encoding='utf-8') as configfile:
            self.config.write(configfile)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.save_settings()
        event.accept()
    
    # ==================== 图片管理方法 ====================
    
    def load_image_folder(self):
        """通过对话框加载图片文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, tr("select_image_folder_dialog_title"),
            self._last_browse_path
        )

        if folder_path:
            self.load_image_folder_by_path(folder_path)

    def load_image_folder_by_path(self, folder_path: str):
        """加载指定路径的图片文件夹（供对话框和自动加载共用）"""
        self._last_browse_path = folder_path
        self._last_folder_path = folder_path
        self.image_manager.load_folder(folder_path)
        self.update_image_list()
        self.update_stats()

        if self.image_manager.get_image_count() > 0:
            self.load_image(0)

    def close_image_folder(self):
        """关闭当前文件夹，下次启动不再自动打开"""
        self.image_manager._image_paths.clear()
        self.image_manager._current_folder = None
        self.image_manager.clear_cache()
        self.annotation_manager._annotations.clear()
        self._last_folder_path = ""
        self.current_image_path = None
        self.current_image_index = 0
        self.canvas._scene.clear()
        self.canvas._crosshair_items = None
        self.update_image_list()
        self.update_stats()
        self.image_info_label.setText(tr("no_image_loaded"))
        self.status_image_info.setText("")
        self.status_annotation_info.setText("")
        self.save_settings()
    
    def load_image(self, index: int):
        """加载指定索引的图片"""
        if 0 <= index < self.image_manager.get_image_count():
            # 切图前自动保存当前标注
            if self.current_image_path:
                annotations = self.canvas.get_annotation_items()
                self.annotation_manager.save_annotations(self.current_image_path, annotations)

            image_path = self.image_manager.get_image_path(index)
            self.current_image_path = image_path
            self.current_image_index = index

            # 加载图片
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 交由 canvas 显示（内部会清除旧场景）
                self.canvas.display_image(pixmap)

                # 更新状态
                self.update_image_info()

                # 加载标注
                self.load_annotations_for_current_image()

                # 选中列表项
                self.image_list_widget.setCurrentRow(index)
            else:
                QMessageBox.warning(self, tr("error"), f"{tr('cannot_load_image')}{image_path}")
    
    def update_image_list(self):
        """更新图片列表"""
        self.image_list_widget.clear()
        
        for i in range(self.image_manager.get_image_count()):
            image_path = self.image_manager.get_image_path(i)
            image_name = os.path.basename(image_path)
            
            # 检查是否有标注
            has_annotations = self.annotation_manager.has_annotations(image_path)
            
            item_text = image_name
            if has_annotations:
                item_text += " ✓"
            
            self.image_list_widget.addItem(item_text)
    
    def update_image_info(self):
        """更新图片信息"""
        if self.current_image_path:
            image_name = os.path.basename(self.current_image_path)
            image_size = self.canvas.get_image_size()
            if image_size:
                info = f"{image_name} | {image_size[0]}x{image_size[1]}"
                self.image_info_label.setText(info)
                self.status_image_info.setText(f"{tr('image_name_label')}: {image_name}")
                return
        self.image_info_label.setText(tr("no_image_loaded"))
        self.status_image_info.setText("")
    
    def update_stats(self):
        """更新统计信息"""
        count = self.image_manager.get_image_count()
        self.stats_label.setText(tr("total_images_count").replace("{count}", str(count)))
    
    def update_statistics_panel(self):
        """更新标注统计面板（委托给 StatsPanel）"""
        self.stats_panel.update_statistics(
            self.image_manager, self.annotation_manager, self.class_manager
        )
    
    def on_image_item_clicked(self, item):
        """图片列表项点击事件"""
        index = self.image_list_widget.row(item)
        self.load_image(index)

    def _on_image_list_context_menu(self, pos):
        """图片列表右键菜单"""
        menu = QMenu(self)
        action_delete = menu.addAction(tr("delete_unannotated"))
        action_delete.triggered.connect(self._delete_all_unannotated_images)

        action_remove = menu.addAction(tr("remove_from_list"))
        action_remove.triggered.connect(self._remove_selected_images)
        action_remove.setEnabled(len(self.image_list_widget.selectedItems()) > 0)

        menu.exec(self.image_list_widget.mapToGlobal(pos))

    def _delete_all_unannotated_images(self):
        """删除所有未标注的图片（同时删除本地文件）"""
        count = self.image_manager.get_image_count()
        if count == 0:
            return

        # 收集未标注的图片索引
        unannotated_indices = []
        for i in range(count):
            image_path = self.image_manager.get_image_path(i)
            if not self.annotation_manager.has_annotations(image_path):
                unannotated_indices.append(i)

        if not unannotated_indices:
            QMessageBox.information(self, tr("info"), tr("no_unannotated_images"))
            return

        # 确认对话框
        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_delete_unannotated").replace("{count}", str(len(unannotated_indices))),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 先收集路径再操作
        paths_to_delete = [self.image_manager.get_image_path(i) for i in unannotated_indices]
        current_image_path = self.current_image_path

        # 删除本地文件和标注文件
        deleted_count = 0
        for image_path in paths_to_delete:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    deleted_count += 1
            except OSError as e:
                self.logger.error(f"删除图片失败: {image_path}, {e}")
            # 清理对应的标注文件
            if self.annotation_manager.has_annotations(image_path):
                self.annotation_manager.clear_annotations(image_path)

        # 从列表中移除（反向遍历保持索引正确）
        current_was_deleted = current_image_path in paths_to_delete
        for i in sorted(unannotated_indices, reverse=True):
            self.image_manager.remove_image(i)

        # 处理当前显示的图片
        self._handle_image_after_removal(current_was_deleted)

        # 更新界面
        self.update_image_list()
        self.update_stats()
        self.update_status(tr("images_deleted").replace("{count}", str(deleted_count)))

    def _remove_selected_images(self):
        """从列表中移除选中的图片（不删除本地文件）"""
        selected_items = self.image_list_widget.selectedItems()
        if not selected_items:
            return

        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_remove_selected").replace("{count}", str(len(selected_items))),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 收集选中的行号和路径
        selected = []
        for item in selected_items:
            row = self.image_list_widget.row(item)
            path = self.image_manager.get_image_path(row)
            if path is not None:
                selected.append((row, path))

        current_image_path = self.current_image_path
        current_was_removed = any(path == current_image_path for _, path in selected)

        # 反向遍历移除
        for row, _ in sorted(selected, key=lambda x: x[0], reverse=True):
            self.image_manager.remove_image(row)

        # 处理当前显示的图片
        self._handle_image_after_removal(current_was_removed)

        # 更新界面
        self.update_image_list()
        self.update_stats()
        self.update_status(tr("images_removed").replace("{count}", str(len(selected))))

    def _handle_image_after_removal(self, current_was_affected: bool):
        """处理移除图片后的当前图片状态"""
        total = self.image_manager.get_image_count()

        if total == 0:
            self.current_image_path = None
            self.current_image_index = 0
            self.canvas._scene.clear()
            self.canvas._crosshair_items = None
            self.image_info_label.setText(tr("no_image_loaded"))
            self.status_image_info.setText("")
            self.status_annotation_info.setText("")
            return

        if current_was_affected or self.current_image_path is None:
            self.load_image(0)
        else:
            # 当前图片还在列表中，但索引可能已变化，找回来
            new_index = self.current_image_index
            if new_index >= total:
                new_index = total - 1
            self.load_image(new_index)
    
    def prev_image(self):
        """上一张图片"""
        if self.image_manager.get_image_count() > 0:
            new_index = (self.current_image_index - 1) % self.image_manager.get_image_count()
            self.load_image(new_index)
    
    def next_image(self):
        """下一张图片"""
        if self.image_manager.get_image_count() > 0:
            new_index = (self.current_image_index + 1) % self.image_manager.get_image_count()
            self.load_image(new_index)
    
    # ==================== 视图操作 ====================
    
    def fit_to_window(self):
        """适应窗口大小"""
        self.canvas.fit_to_window()
        self._update_scale_status()

    def zoom_in(self):
        """放大"""
        self.canvas.zoom_in()
        self._update_scale_status()

    def zoom_out(self):
        """缩小"""
        self.canvas.zoom_out()
        self._update_scale_status()

    def reset_view(self):
        """重置视图"""
        self.canvas.reset_view()
        self.update_status(tr("view_reset_message"))

    def _update_scale_status(self):
        """更新缩放显示"""
        factor = self.canvas.get_scale_factor()
        self.update_status(tr("zoom_status").replace("{scale_factor}", f"{factor:.2f}"))
    
    # ==================== 类别管理 ====================
    
    def update_class_list(self):
        """更新类别列表"""
        self.class_list_widget.clear()
        
        classes = self.class_manager.get_classes()
        for class_id, class_info in sorted(classes.items()):
            class_name = class_info["name"]
            color = class_info["color"]
            
            item_text = f"{class_id}: {class_name}"
            self.class_list_widget.addItem(item_text)
            
            # 设置背景颜色
            item = self.class_list_widget.item(self.class_list_widget.count() - 1)
            item.setData(Qt.UserRole, class_id)
            item.setBackground(QColor(*color))
            
            # 设置文字颜色为对比色
            brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
            text_color = QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)
            item.setForeground(text_color)
    
    def on_class_item_clicked(self, item):
        """类别列表项点击事件"""
        # 优先从UserRole读取真实class_id，避免行号和class_id错位
        class_id = item.data(Qt.UserRole)
        if class_id is None:
            # 兼容旧数据：从文本中回退解析
            item_text = item.text()
            if ": " in item_text:
                try:
                    class_id_str = item_text.split(": ")[0]
                    class_id = int(class_id_str)
                except ValueError:
                    self.update_status(tr("cannot_parse_class_id_error"))
                    return
            else:
                self.update_status(tr("invalid_class_format_error"))
                return

        self.selected_class_id = int(class_id)
        self.canvas.selected_class_id = self.selected_class_id
        self.update_status(tr("selected_class").replace("{class_name}", self.class_manager.get_class_name(self.selected_class_id)))
    
    def add_class(self):
        """添加类别"""
        from .class_dialog import ClassDialog

        dialog = ClassDialog(self)
        # 预先分配一个不与现有颜色重复的颜色并显示在对话框中
        auto_color = self.class_manager._generate_color()
        dialog.set_color(auto_color)

        if dialog.exec():
            class_name, color = dialog.get_values()
            self.class_manager.add_class(class_name, color)
            self.update_class_list()
    
    def edit_class(self):
        """编辑类别"""
        selected_items = self.class_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, tr("warning"), tr("select_class"))
            return
        
        selected_item = selected_items[0]
        class_id = selected_item.data(Qt.UserRole)
        if class_id is None:
            QMessageBox.warning(self, tr("warning"), tr("cannot_parse_class_id_error"))
            return

        class_id = int(class_id)
        class_info = self.class_manager.get_class(class_id)
        
        # 检查类别是否存在
        if not class_info:
            QMessageBox.warning(self, tr("warning"), tr("class_not_exist"))
            return
        
        from .class_dialog import ClassDialog
        
        dialog = ClassDialog(self)
        dialog.set_values(class_info["name"], class_info["color"])
        
        if dialog.exec():
            class_name, color = dialog.get_values()
            self.class_manager.update_class(class_id, class_name, color)
            self.update_class_list()
    
    def delete_class(self):
        """删除类别"""
        selected_items = self.class_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, tr("warning"), tr("select_class"))
            return
        
        selected_item = selected_items[0]
        class_id = selected_item.data(Qt.UserRole)
        if class_id is None:
            QMessageBox.warning(self, tr("warning"), tr("cannot_parse_class_id_error"))
            return

        class_id = int(class_id)
        
        # 获取要删除的类别名称
        class_info = self.class_manager.get_class(class_id)
        if not class_info:
            QMessageBox.warning(self, tr("warning"), tr("class_does_not_exist"))
            return
        
        class_name = class_info["name"]
        
        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_delete_class").replace("{class_name}", class_name),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 如果删除的是当前选中的类别，重置选中状态
            if class_id == self.selected_class_id:
                self.selected_class_id = 0
                self.canvas.selected_class_id = 0
                if self.class_manager.get_class_count() > 0:
                    # 选择第一个可用的类别
                    available_classes = list(self.class_manager.get_classes().keys())
                    if available_classes:
                        self.selected_class_id = available_classes[0]
                        self.canvas.selected_class_id = self.selected_class_id
            
            # 删除类别
            self.class_manager.delete_class(class_id)
            self.update_class_list()
            
            # 更新状态
            self.update_status(tr("class_deleted").replace("{class_name}", class_name))
    
    def clear_all_classes(self):
        """清空所有类别"""
        if self.class_manager.get_class_count() == 0:
            QMessageBox.information(self, tr("info"), tr("no_classes_to_clear"))
            return
        
        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_clear_all_classes"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清空所有类别
            self.class_manager.clear_all()
            self.update_class_list()
            
            # 重置选中的类别
            self.selected_class_id = 0
            self.canvas.selected_class_id = 0
            
            # 更新状态
            self.update_status(tr("all_classes_cleared"))
    
    # ==================== 标注管理 ====================
    
    def load_annotations_for_current_image(self):
        """加载当前图片的标注"""
        if self.current_image_path:
            annotations = self.annotation_manager.get_annotations(self.current_image_path)
            self.draw_annotations(annotations)
    
    def draw_annotations(self, annotations: List[Annotation]):
        """绘制标注框 — 委托给 canvas"""
        self.canvas.draw_annotations(annotations)
    
    def save_annotations(self):
        """保存标注"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded_warning"))
            return

        annotations = self.canvas.get_annotation_items()
        self.annotation_manager.save_annotations(self.current_image_path, annotations)
        self.update_status(tr("annotations_saved").replace("{count}", str(len(annotations))))
    
    def delete_selected_annotation(self):
        """删除选中标注（仅删除当前选中的标注框）"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        sel_annotation = self.canvas.get_selected_annotation()
        if sel_annotation is None:
            QMessageBox.warning(self, tr("warning"), tr("no_annotation_selected"))
            return

        annotation_index = self.canvas.get_selected_annotation_index()

        if annotation_index >= 0:
            from src.core.annotation import DeleteAnnotationCommand
            command = DeleteAnnotationCommand(
                self.annotation_manager,
                self.current_image_path,
                annotation_index
            )
            self.annotation_manager.execute_command(command)

        self.load_annotations_for_current_image()
        self.update_image_list()
        self.update_status(tr("annotation_deleted"))
        self.update_undo_redo_actions()
    
    def clear_all_annotations(self):
        """清除所有标注"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("clear_all_annotations_confirmation"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.clear_annotation_items()
            self.annotation_manager.clear_annotations(self.current_image_path)
            self.update_image_list()
            self.update_status(tr("annotations_cleared"))
    
    # ==================== 导出功能 ====================
    
    def export_yolo_format(self):
        """导出YOLO格式"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        default_path = str(Path(self._last_browse_path) / "data.yaml")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("export_yolo_format_dialog_title"),
            default_path,
            tr("yaml_file_filter"),
        )

        if save_path:
            output_dir = str(Path(save_path).parent)
            self._last_browse_path = output_dir
            yaml_filename = Path(save_path).name
            if not yaml_filename.lower().endswith((".yaml", ".yml")):
                yaml_filename += ".yaml"

            try:
                self.yolo_exporter.export(
                    self.image_manager,
                    self.annotation_manager,
                    self.class_manager,
                    output_dir,
                    yaml_filename=yaml_filename,
                )
                QMessageBox.information(self, tr("success"), f"{tr('export_success')} {output_dir}")
            except Exception as e:
                QMessageBox.critical(self, tr("error"), f"{tr('export_failed')} {str(e)}")

    def export_dataset_split(self):
        """导出数据集划分"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, tr("export_dataset_split_dialog_title"),
            self._last_browse_path
        )

        if output_dir:
            self._last_browse_path = output_dir
            try:
                from src.utils.dataset_splitter import DatasetSplitter
                
                splitter = DatasetSplitter()
                splitter.split_and_export(
                    self.image_manager,
                    self.annotation_manager,
                    output_dir,
                    class_manager=self.class_manager,
                )
                QMessageBox.information(self, tr("success"), f"{tr('dataset_split_export_success')} {output_dir}")
            except Exception as e:
                QMessageBox.critical(self, tr("error"), f"{tr('export_failed')} {str(e)}")
    
    # ==================== 模型辅助标注 ====================
    
    # ==================== 撤销/重做方法 ====================
    
    def undo(self):
        """撤销"""
        if self.annotation_manager.can_undo():
            success = self.annotation_manager.undo()
            if success:
                # 重新加载标注以更新UI
                self.load_annotations_for_current_image()
                self.update_status(tr("undo_success"))
                self.update_undo_redo_actions()
            else:
                self.update_status(tr("undo_failed"))
        else:
            self.update_status(tr("no_undo_action"))
    
    def redo(self):
        """重做"""
        if self.annotation_manager.can_redo():
            success = self.annotation_manager.redo()
            if success:
                # 重新加载标注以更新UI
                self.load_annotations_for_current_image()
                self.update_status(tr("redo_success"))
                self.update_undo_redo_actions()
            else:
                self.update_status(tr("redo_failed"))
        else:
            self.update_status(tr("no_redo_action"))
    
    def update_undo_redo_actions(self):
        """更新撤销/重做菜单和按钮状态"""
        can_undo = self.annotation_manager.can_undo()
        can_redo = self.annotation_manager.can_redo()

        # self.action_undo/redo 与工具栏按钮是同一 QAction 对象，直接设置即可
        self.action_undo.setEnabled(can_undo)
        self.action_redo.setEnabled(can_redo)
    
    # ==================== 其他方法 ====================
    
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_label.setText(message)
    
    
    def load_classes_from_yaml(self):
        """从YAML文件加载类别"""
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, tr("load_classes_from_yaml_dialog"),
            self._last_browse_path,
            tr("yaml_file_filter")
        )
        
        if yaml_path:
            success = self.class_manager.import_from_yaml(yaml_path)
            if success:
                self.update_class_list()
                QMessageBox.information(self, tr("success"), tr("load_yaml_success") + f" {yaml_path}")
            else:
                QMessageBox.warning(self, tr("warning"), tr("load_yaml_failed"))
    
    def save_classes_to_yaml(self):
        """保存类别到YAML文件"""
        if self.class_manager.get_class_count() == 0:
            QMessageBox.warning(self, tr("warning"), tr("no_classes_to_save"))
            return

        yaml_path, _ = QFileDialog.getSaveFileName(
            self, tr("save_classes_to_yaml_dialog"),
            self._last_browse_path,
            tr("yaml_file_filter")
        )
        
        if yaml_path:
            # 确保文件扩展名
            if not yaml_path.lower().endswith(('.yaml', '.yml')):
                yaml_path += '.yaml'
            
            # 导出YAML
            self.class_manager.export_to_yaml(yaml_path)
            QMessageBox.information(self, tr("success"), tr("save_yaml_success") + f" {yaml_path}")
    
    # ==================== Canvas 信号处理 ====================

    def add_annotation_with_command(self, annotation):
        """通过命令模式添加标注（支持撤销）"""
        from src.core.annotation import AddAnnotationCommand
        command = AddAnnotationCommand(
            self.annotation_manager,
            self.current_image_path,
            annotation,
        )
        self.annotation_manager.execute_command(command)
        self.update_image_list()
        self.load_annotations_for_current_image()
        self.update_undo_redo_actions()

    def _on_canvas_annotation_created(self, annotation):
        """用户通过画布绘制了一个新标注"""
        self.add_annotation_with_command(annotation)
        class_name = self.class_manager.get_class_name(self.selected_class_id)
        self.update_status(tr("annotation_created").replace("{class_name}", class_name))

    def _on_canvas_annotation_deleted(self, annotation, index: int = -1):
        """用户通过右键菜单删除了一个标注"""
        if not self.current_image_path:
            return

        if index < 0:
            return

        from src.core.annotation import DeleteAnnotationCommand
        command = DeleteAnnotationCommand(
            self.annotation_manager,
            self.current_image_path,
            index
        )
        self.annotation_manager.execute_command(command)

        self.load_annotations_for_current_image()
        self.update_image_list()
        self.update_status(tr("annotation_deleted"))
        self.update_undo_redo_actions()

    # ==================== 面板信号处理 ====================

    def _on_stats_refresh(self):
        """统计面板刷新按钮"""
        self.stats_panel.update_statistics(
            self.image_manager, self.annotation_manager, self.class_manager
        )

    def _on_panel_conf_changed(self, value: float):
        """模型信息面板置信度阈值变化"""
        if self.model_manager.is_model_loaded():
            self.model_manager.set_confidence_threshold(value)
            self.update_status(
                tr("confidence_threshold_set").replace("{value}", f"{value:.2f}")
            )

    def _on_panel_iou_changed(self, value: float):
        """模型信息面板IoU阈值变化"""
        if self.model_manager.is_model_loaded():
            self.model_manager.set_iou_threshold(value)
            self.update_status(
                tr("iou_threshold_set").replace("{value}", f"{value:.2f}")
            )

    def _on_model_info_refresh(self):
        """模型信息面板刷新按钮"""
        self.model_info_panel.update_info(self.model_manager)

    # ==================== 模型管理方法 ====================

    # ==================== 模型管理方法 ====================
    
    def load_model(self):
        """加载YOLO模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, tr("model_file_selection"),
            self._last_browse_path,
            tr("model_file_filter")
        )
        
        if model_path:
            # 检查YOLO库是否可用
            if not self.model_manager.is_available():
                QMessageBox.critical(
                    self, tr("error"),
                    tr("ultralytics_not_installed")
                )
                return
            
            success = self.model_manager.load_model(model_path)
            if success:
                model_info = self.model_manager.get_model_info()
                class_count = model_info["class_count"]
                classes = model_info["classes"]
                
                # 询问是否将模型类别添加到类别管理器
                if classes:
                    reply = QMessageBox.question(
                        self, tr("import_model_classes"),
                        tr("import_model_classes_confirmation").replace("{class_count}", str(class_count)),
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.Yes:
                        # 导入模型类别
                        for class_id, class_name in classes.items():
                            # 生成随机颜色
                            import random
                            color = (
                                random.randint(50, 255),
                                random.randint(50, 255),
                                random.randint(50, 255)
                            )
                            # 添加或更新类别
                            self.class_manager.add_or_update_class(class_id, class_name, color)
                        
                        self.update_class_list()
                        QMessageBox.information(self, tr("success"), tr("success"))
                
                # 更新模型信息面板
                self.model_info_panel.update_info(self.model_manager)

                # 如果模型信息面板是隐藏的，自动显示它
                if not self.model_info_panel.isVisible():
                    self.model_info_panel.setVisible(True)
                    self.action_model_info.setText(tr("hide_model_info"))
                
                QMessageBox.information(
                    self, tr("success"),
                    tr("load_model_success").replace("{model_path}", model_path).replace("{class_count}", str(class_count))
                )
                self.update_status(tr("model_loaded_status").replace("{model_name}", os.path.basename(model_path)))
            else:
                QMessageBox.critical(
                    self, tr("error"),
                    tr("load_model_failed").replace("{model_path}", model_path)
                )
    
    def show_model_info(self):
        """显示/隐藏模型信息面板"""
        # 切换面板可见性
        is_visible = not self.model_info_panel.isVisible()
        self.model_info_panel.setVisible(is_visible)

        # 如果显示面板，更新内容
        if is_visible:
            self.model_info_panel.update_info(self.model_manager)
            # 确保UI文本使用当前语言
            self.update_other_ui_elements()
        
        # 更新菜单文本
        if is_visible:
            self.action_model_info.setText(tr("hide_model_info"))
            self.update_status(tr("model_info_panel_shown"))
        else:
            self.action_model_info.setText(tr("show_model_info"))
            self.update_status(tr("model_info_panel_hidden"))

    def open_validation_window(self):
        """打开验证窗口"""
        try:
            from src.ui.validation_dialog import ValidationDialog
            dialog = ValidationDialog(self, self.model_manager)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"打开验证窗口失败: {e}")
            QMessageBox.critical(
                self, tr("error"),
                tr("open_validation_window_failed").replace("{error}", str(e))
            )

    def auto_annotate_current(self):
        """自动标注当前图片"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return

        # 如果已有标注，确认是否覆盖
        if self.annotation_manager.has_annotations(self.current_image_path):
            reply = QMessageBox.question(
                self, tr("confirm"),
                tr("auto_annotation_overwrite_warning"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            # 显示进度
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # 使用模型进行推理
            detections = self.model_manager.predict(self.current_image_path)
            
            if not detections:
                QApplication.restoreOverrideCursor()
                QMessageBox.information(self, tr("result"), tr("no_target_detected"))
                return
            
            # 将检测结果转换为标注
            annotations = self.model_manager.convert_to_annotations(detections)
            
            if not annotations:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, tr("warning"), tr("convert_detections_failed"))
                return
            
            # 通过 canvas 清空并绘制新标注
            self.canvas.draw_annotations(annotations)

            # 保存标注
            self.annotation_manager.save_annotations(self.current_image_path, annotations)
            
            # 更新状态
            self.update_status(tr("auto_annotation_complete").replace("{detection_count}", str(len(annotations))))
            QApplication.restoreOverrideCursor()
            
            QMessageBox.information(
                self, tr("complete"),
                tr("auto_annotation_complete").replace("{detection_count}", str(len(annotations)))
            )
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, tr("error"), f"{tr('auto_annotation_failed')}: {str(e)}")
    
    def batch_auto_annotate(self):
        """批量自动标注"""
        
        if self.image_manager.get_image_count() == 0:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return
        
        # 确认批量标注
        msg = (tr("batch_annotation_confirmation").replace("{image_count}", str(self.image_manager.get_image_count()))
               + "\n\n⚠ " + tr("batch_overwrite_warning"))
        reply = QMessageBox.question(
            self, tr("confirm_batch_annotation"),
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # 创建进度对话框
            progress_dialog = QProgressDialog(tr("batch_annotation_in_progress"), tr("cancel"), 
                                               0, self.image_manager.get_image_count(), self)
            progress_dialog.setWindowTitle(tr("batch_annotation_progress"))
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()
            
            success_count = 0
            total_detections = 0
            
            for i in range(self.image_manager.get_image_count()):
                if progress_dialog.wasCanceled():
                    break
                
                image_path = self.image_manager.get_image_path(i)
                
                try:
                    # 推理
                    detections = self.model_manager.predict(image_path)
                    annotations = self.model_manager.convert_to_annotations(detections)
                    
                    if annotations:
                        # 保存标注
                        self.annotation_manager.save_annotations(image_path, annotations)
                        success_count += 1
                        total_detections += len(annotations)
                    
                except Exception as e:
                    self.logger.error(f"标注图片 {os.path.basename(image_path)} 失败: {e}")
                
                # 更新进度
                progress_dialog.setValue(i + 1)
                QApplication.processEvents()
            
            progress_dialog.close()
            
            # 更新图片列表显示
            self.update_image_list()
            
            # 显示结果
            QMessageBox.information(
                self, tr("batch_annotation_complete"),
                tr("batch_annotation_result").replace("{success_count}", str(success_count))
                                                  .replace("{total_count}", str(self.image_manager.get_image_count()))
                                                  .replace("{total_detections}", str(total_detections))
            )
            
            self.update_status(tr("batch_annotation_complete_status").replace("{image_count}", str(success_count)))
            
        except Exception as e:
            QMessageBox.critical(self, tr("error"), tr("batch_annotation_failed").replace("{error}", str(e)))
    
    # ==================== 模型参数调整方法 ====================
    
    def unload_model(self):
        """卸载模型"""
        if not self.model_manager.is_model_loaded():
            QMessageBox.information(self, tr("info"), tr("no_model_loaded"))
            return
        
        reply = QMessageBox.question(
            self, tr("confirm_unload"),
            tr("unload_model_confirmation"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.model_manager.unload_model()
            self.model_info_panel.update_info(self.model_manager)
            
            self.update_status(tr("model_unloaded"))
            QMessageBox.information(self, tr("success"), tr("model_unloaded"))
    
    def update_model_info_panel(self):
        """更新模型信息面板（委托给 ModelInfoPanel）"""
        self.model_info_panel.update_info(self.model_manager)
    
    def train_model(self):
        """训练模型"""
        # 检查是否加载了模型，如果有则获取模型路径作为默认值
        default_model_path = ""
        if self.model_manager.is_model_loaded():
            model_info = self.model_manager.get_model_info()
            default_model_path = model_info.get("path", "")
        
        # 创建训练配置对话框
        from .train_dialog import TrainDialog
        dialog = TrainDialog(self, default_model_path)
        
        if dialog.exec():
            # 对话框已确认，训练将在对话框内部启动
            self.update_status(tr("training_config_complete"))
    
    def switch_language(self, language: str):
        """切换语言"""
        from src.utils.i18n import TranslationManager, tr
        
        self.logger.info(f"开始切换语言到: {language}")
        translation_manager = TranslationManager.instance()
        success = translation_manager.switch_language(language)
        
        if success:
            self.logger.info(f"语言切换成功，当前语言: {translation_manager.get_current_language()}")
            # 测试翻译是否立即生效
            self.logger.info(f"测试翻译 'open_folder': {tr('open_folder')}")
            self.logger.info(f"测试翻译 'file': {tr('file')}")
            # 更新所有UI文本
            self.update_ui_texts()
            self.update_status(tr("language_switched").replace("{language}", language))
            QMessageBox.information(self, tr("success"), tr("language_switched").replace("{language}", language))
            self.logger.info(f"UI文本已更新")
        else:
            self.logger.error(f"语言切换失败: {language}")
            QMessageBox.warning(self, tr("warning"), tr("language_switch_failed").replace("{language}", language))
    
    def update_ui_texts(self):
        """更新UI文本（语言切换后重新设置所有文本）"""
        
        # 窗口标题
        self.setWindowTitle(tr("yolo_label_tool"))
        
        # 更新菜单文本
        self.update_menu_texts()
        
        # 更新按钮文本
        self.update_button_texts()
        
        # 更新面板标题
        self.update_panel_titles()
        
        # 更新状态栏
        self.status_label.setText(tr("ready"))
        
        # 更新其他UI元素
        self.update_other_ui_elements()
    
    def update_menu_texts(self):
        """更新菜单文本"""

        self.file_menu.setTitle(tr("file"))
        self.edit_menu.setTitle(tr("edit"))
        self.view_menu.setTitle(tr("view"))
        self.class_menu.setTitle(tr("classes"))
        self.theme_menu.setTitle(tr("theme"))
        self.model_menu.setTitle(tr("model"))
        self.annotate_menu.setTitle(tr("annotate"))
        self.language_menu.setTitle(tr("language"))

        self.action_open_folder.setText(tr("open_folder"))
        self.action_save.setText(tr("save_annotations"))
        self.action_export.setText(tr("export_yolo_format"))
        self.action_exit.setText(tr("exit"))

        self.action_undo.setText(tr("undo"))
        self.action_redo.setText(tr("redo"))
        self.action_delete.setText(tr("delete_selected"))

        self.action_zoom_in.setText(tr("zoom_in"))
        self.action_zoom_out.setText(tr("zoom_out"))
        self.action_fit.setText(tr("fit_to_window"))

        self.action_load_model.setText(tr("load_model"))
        self.action_model_info.setText(tr("model_info"))
        self.action_train_model.setText(tr("train_model"))
        self.action_auto_annotate.setText(tr("auto_annotate"))
        self.action_batch_auto_annotate.setText(tr("batch_auto_annotate"))

        self.action_load_yaml.setText(tr("load_yaml"))
        self.action_save_yaml.setText(tr("save_yaml"))

        self.action_dark_theme.setText(tr("dark_theme"))
        self.action_light_theme.setText(tr("light_theme"))
        self.action_colorful_theme.setText(tr("colorful_theme"))

        self.action_chinese.setText(tr("chinese"))
        self.action_english.setText(tr("english"))

    def update_button_texts(self):
        """更新按钮文本"""

        self.btn_load_folder.setText(tr("load_folder"))
        self.btn_prev.setText(tr("previous_image"))
        self.btn_next.setText(tr("next_image"))

        self.btn_fit.setText(tr("fit_to_window"))
        self.btn_zoom_in.setText(tr("zoom_in"))
        self.btn_zoom_out.setText(tr("zoom_out"))
        self.btn_reset.setText(tr("reset"))

        self.btn_add_class.setText(tr("add"))
        self.btn_edit_class.setText(tr("edit"))
        self.btn_delete_class.setText(tr("delete"))
        self.btn_clear_classes.setText(tr("clear_all_classes"))

        self.btn_delete_annotation.setText(tr("delete_selected_annotation"))
        self.btn_clear_all.setText(tr("clear_all_annotations"))

        self.btn_export_yolo.setText(tr("export_yolo"))
        self.btn_export_split.setText(tr("export_dataset_split"))

    def update_panel_titles(self):
        """更新面板标题"""

        self.left_panel_title_label.setText(tr("image_list"))
        self.center_panel_title_label.setText(tr("image_annotation"))
        self.right_panel_title_label.setText(tr("class_management"))

    def update_other_ui_elements(self):
        """更新其他UI元素"""

        self.class_group.setTitle(tr("annotation_classes"))
        self.annotation_group.setTitle(tr("annotation_operations"))
        self.export_group.setTitle(tr("data_export"))

        self.stats_panel.update_language()
        self.model_info_panel.update_language(self.model_manager)

        if self.image_manager.get_image_count() == 0:
            self.stats_label.setText(tr("no_image_loaded"))

        if not self.current_image_path:
            self.image_info_label.setText(tr("no_image_loaded"))

