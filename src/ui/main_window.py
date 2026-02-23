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
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem, QGraphicsEllipseItem,
    QMenuBar, QMenu, QDockWidget, QFrame, QSizePolicy,
    QScrollArea, QGroupBox, QFormLayout, QLineEdit,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QProgressDialog, QApplication, QSlider, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import (
    Qt, QSize, QPoint, QPointF, QRect, QRectF, QTimer,
    Signal, Slot, QEvent
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor,
    QAction, QKeySequence, QFont, QIcon, QCursor,
    QMouseEvent, QWheelEvent, QKeyEvent, QResizeEvent
)

from src.core.annotation import Annotation, AnnotationManager
from src.core.image_manager import ImageManager
from src.core.class_manager import ClassManager
from src.core.model_manager import ModelManager
from src.utils.yolo_exporter import YOLOExporter

from src.utils.logger import get_logger_simple
from src.utils.i18n import TranslationManager, tr, T


class AnnotationRectItem(QGraphicsRectItem):
    """标注框图形项，支持选中状态和拖拽编辑"""
    
    def __init__(self, x, y, width, height, annotation, color, parent=None):
        super().__init__(x, y, width, height, parent)
        self.annotation = annotation
        self.color = color
        
        # 拖拽状态
        self.is_selected = False
        self.is_dragging = False
        self.is_resizing = False
        self.drag_start_pos = QPointF()
        self.original_rect = QRectF()
        self.resize_handle = None  # 调整大小的控制点位置：None, 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        
        # 控制点大小
        self.handle_size = 8
        
        # 设置可拖拽和可选中
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        
        # 右键菜单
        self.context_menu = None
        
        self.update_appearance()
    
    def hoverMoveEvent(self, event):
        """鼠标悬停事件，显示调整控制点"""
        if self.is_selected:
            pos = event.pos()
            rect = self.rect()
            
            # 检查鼠标是否在控制点附近
            handle_positions = {
                'top-left': QPointF(rect.left(), rect.top()),
                'top-right': QPointF(rect.right(), rect.top()),
                'bottom-left': QPointF(rect.left(), rect.bottom()),
                'bottom-right': QPointF(rect.right(), rect.bottom())
            }
            
            for handle_name, handle_pos in handle_positions.items():
                if (pos - handle_pos).manhattanLength() < self.handle_size * 2:
                    # 根据控制点位置设置鼠标形状
                    if handle_name in ['top-left', 'bottom-right']:
                        self.setCursor(Qt.SizeFDiagCursor)
                    elif handle_name in ['top-right', 'bottom-left']:
                        self.setCursor(Qt.SizeBDiagCursor)
                    self.resize_handle = handle_name
                    return
            
            # 如果不在控制点附近，检查是否在边框附近
            if (abs(pos.x() - rect.left()) < 5 or abs(pos.x() - rect.right()) < 5 or
                abs(pos.y() - rect.top()) < 5 or abs(pos.y() - rect.bottom()) < 5):
                self.setCursor(Qt.SizeAllCursor)
                self.resize_handle = 'edge'
            else:
                self.setCursor(Qt.ArrowCursor)
                self.resize_handle = None
        super().hoverMoveEvent(event)
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            # 取消其他所有标注框的选中状态
            self.deselect_all_other_annotations()
            
            if self.resize_handle:
                # 开始调整大小
                self.is_resizing = True
                self.drag_start_pos = event.pos()
                self.original_rect = self.rect()
            else:
                # 开始拖拽
                self.is_dragging = True
                self.drag_start_pos = event.pos()
                self.original_rect = self.rect()
            
            # 设置选中状态
            if not self.is_selected:
                self.set_selected(True)
            event.accept()
        
        elif event.button() == Qt.RightButton:
            # 显示右键菜单
            self.show_context_menu(event.screenPos())
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def deselect_all_other_annotations(self):
        """取消所有其他标注框的选中状态"""
        scene = self.scene()
        if scene:
            for item in scene.items():
                if (hasattr(item, 'is_annotation_item') and 
                    item != self and 
                    hasattr(item, 'is_selected') and 
                    item.is_selected):
                    item.set_selected(False)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.is_dragging:
            # 计算移动距离
            delta = event.pos() - self.drag_start_pos
            new_rect = self.original_rect.translated(delta)
            
            # 更新矩形位置
            self.setRect(new_rect)
            
            # 更新标注数据
            self.annotation.x = new_rect.x()
            self.annotation.y = new_rect.y()
            
            event.accept()
        
        elif self.is_resizing:
            # 调整大小
            delta = event.pos() - self.drag_start_pos
            new_rect = QRectF(self.original_rect)
            
            if self.resize_handle == 'top-left':
                new_rect.setTopLeft(self.original_rect.topLeft() + delta)
            elif self.resize_handle == 'top-right':
                new_rect.setTopRight(self.original_rect.topRight() + delta)
            elif self.resize_handle == 'bottom-left':
                new_rect.setBottomLeft(self.original_rect.bottomLeft() + delta)
            elif self.resize_handle == 'bottom-right':
                new_rect.setBottomRight(self.original_rect.bottomRight() + delta)
            elif self.resize_handle == 'edge':
                # 如果是边框拖拽，调整整个大小（保持中心点不变）
                new_rect = self.original_rect.adjusted(-delta.x()/2, -delta.y()/2, delta.x()/2, delta.y()/2)
            
            # 确保矩形大小合理
            if new_rect.width() > 10 and new_rect.height() > 10:
                self.setRect(new_rect)
                
                # 更新标注数据
                self.annotation.x = new_rect.x()
                self.annotation.y = new_rect.y()
                self.annotation.width = new_rect.width()
                self.annotation.height = new_rect.height()
            
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            if self.is_dragging or self.is_resizing:
                # 保存标注变更到文件
                if self.scene() and hasattr(self.scene().parent(), 'save_annotations'):
                    self.scene().parent().save_annotations()
                
                self.is_dragging = False
                self.is_resizing = False
                event.accept()
            else:
                super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)
    
    def set_selected(self, selected: bool):
        """设置选中状态"""
        self.is_selected = selected
        self.update_appearance()
    
    def update_appearance(self):
        """更新外观"""
        if self.is_selected:
            # 选中时使用更粗的边框和实心填充
            self.setPen(QPen(QColor(255, 255, 0), 3))  # 黄色边框
            self.setBrush(QBrush(QColor(255, 255, 0, 30)))  # 半透明黄色填充
            
            # 绘制控制点
            self.paint_handles()
        else:
            # 非选中状态，使用类别颜色
            self.setPen(QPen(self.color, 2))
            self.setBrush(QBrush(self.color, Qt.Dense4Pattern))
            self.resize_handle = None
    
    def paint_handles(self):
        """绘制调整大小的控制点"""
        # 控制点会在paint方法中绘制
        self.update()
    
    def paint(self, painter, option, widget=None):
        """自定义绘制"""
        super().paint(painter, option, widget)
        
        # 如果被选中，绘制控制点
        if self.is_selected:
            rect = self.rect()
            handle_size = self.handle_size
            
            # 绘制四个角的控制点
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            corners = [
                rect.topLeft(),
                rect.topRight(),
                rect.bottomLeft(),
                rect.bottomRight()
            ]
            
            for corner in corners:
                painter.drawRect(
                    corner.x() - handle_size/2,
                    corner.y() - handle_size/2,
                    handle_size,
                    handle_size
                )
    
    def show_context_menu(self, screen_pos):
        """显示右键菜单"""
        if not self.context_menu:
            self.context_menu = QMenu()
            
            # 修改类别子菜单
            self.class_menu = QMenu(tr("modify_class"), self.context_menu)
            self.context_menu.addMenu(self.class_menu)
            
            # 删除标注
            delete_action = QAction(tr("delete_annotation"), self.context_menu)
            delete_action.triggered.connect(self.delete_annotation)
            self.context_menu.addAction(delete_action)
        
        # 更新类别子菜单
        self.class_menu.clear()
        
        # 获取当前窗口实例以访问类别管理器
        scene = self.scene()
        if scene and hasattr(scene.parent(), 'class_manager'):
            class_manager = scene.parent().class_manager
            classes = class_manager.get_classes()
            
            for class_id, class_info in sorted(classes.items()):
                class_name = class_info["name"]
                color = class_info["color"]
                
                action = QAction(class_name, self.class_menu)
                action.setData(class_id)  # 存储类别ID
                action.triggered.connect(lambda checked, cid=class_id: self.change_class(cid))
                
                # 设置图标颜色
                pixmap = QPixmap(16, 16)
                pixmap.fill(QColor(*color))
                action.setIcon(QIcon(pixmap))
                
                self.class_menu.addAction(action)
        
        # 显示菜单
        self.context_menu.exec(screen_pos)
    
    def change_class(self, class_id):
        """修改标注类别"""
        self.annotation.class_id = class_id
        
        # 更新颜色
        scene = self.scene()
        if scene and hasattr(scene.parent(), 'class_manager'):
            class_manager = scene.parent().class_manager
            class_info = class_manager.get_class(class_id)
            if class_info:
                self.color = QColor(*class_info["color"])
                self.update_appearance()
        
        # 保存变更
        if scene and hasattr(scene.parent(), 'save_annotations'):
            scene.parent().save_annotations()
    
    def delete_annotation(self):
        """删除标注"""
        scene = self.scene()
        if scene and hasattr(scene.parent(), 'delete_selected_annotation'):
            # 确保当前项被选中
            if not self.is_selected:
                self.set_selected(True)
            scene.parent().delete_selected_annotation()


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
        self.is_drawing: bool = False
        self.drawing_start_point: Optional[QPoint] = None
        self.drawing_end_point: Optional[QPoint] = None
        self.selected_class_id: int = 0
        self.scale_factor: float = 1.0
        self.temp_rect_item = None  # 临时绘制项
        self.selected_annotations = []  # 存储选中的标注项
        
        # 加载设置
        self.load_settings()
        
        # 初始化UI
        self.init_ui()
        self.init_actions()
        self.init_menus()
        self.init_toolbar()
        self.init_statusbar()
        
        # 更新类别列表（如果已从配置文件加载）
        if hasattr(self, 'class_list_widget'):
            self.update_class_list()
        
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
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板 - 图片列表
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 中间面板 - 图片显示
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # 右侧面板 - 类别管理
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割器初始大小
        splitter.setSizes([200, 600, 200])
        
        # 加载QSS样式
        self.load_qss_style()
    
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
                
                self.logger.info(f"已应用主题: {theme_name}")
            except Exception as e:
                self.logger.error(f"加载主题样式文件失败: {e}, 将使用Qt默认样式")
        else:
            self.logger.warning(f"主题文件不存在: {qss_path}, 将使用Qt默认样式")
    
    def update_title_colors_for_theme(self, theme_name: str):
        """根据主题更新标题标签颜色"""
        if theme_name == "dark":
            # 黑夜主题：白色标题，灰色状态
            title_style = "font-weight: bold; font-size: 14px; color: #ffffff;"
            status_style = "color: #aaaaaa; font-size: 12px;"
        else:
            # 白天主题：黑色标题，深灰色状态
            title_style = "font-weight: bold; font-size: 14px; color: #333333;"
            status_style = "color: #777777; font-size: 12px;"
        
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
        
        # 图片显示区域
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        
        # 设置鼠标事件
        self.graphics_view.viewport().installEventFilter(self)
        
        layout.addWidget(self.graphics_view)
        
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
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter, 1)  # 第二个参数为1表示填充剩余空间
        
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
        
        class_layout.addLayout(class_btn_layout)
        
        splitter.addWidget(self.class_group)
        
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
        
        splitter.addWidget(self.annotation_group)
        
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
        
        splitter.addWidget(self.export_group)
        
        # 标注统计面板
        self.stats_group = QGroupBox(tr("annotation_stats"))
        stats_layout = QVBoxLayout(self.stats_group)
        
        # 统计信息标签
        self.stats_total_label = QLabel(tr("total_images_zero"))
        stats_layout.addWidget(self.stats_total_label)
        
        self.stats_annotated_label = QLabel(tr("annotated_images_zero"))
        stats_layout.addWidget(self.stats_annotated_label)
        
        self.stats_unannotated_label = QLabel(tr("unannotated_images_zero"))
        stats_layout.addWidget(self.stats_unannotated_label)
        
        # 类别统计表格（使用QTableWidget）
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels([tr("class_id"), tr("class_name"), tr("annotation_count")])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_table)
        
        # 刷新统计按钮
        self.btn_refresh_stats = QPushButton(tr("refresh_stats"))
        self.btn_refresh_stats.clicked.connect(self.update_statistics_panel)
        stats_layout.addWidget(self.btn_refresh_stats)
        
        stats_layout.addStretch()
        
        splitter.addWidget(self.stats_group)
        
        # 模型信息面板（默认隐藏）
        self.model_info_group = QGroupBox(tr("model_info_panel"))
        self.model_info_group.setVisible(False)  # 默认隐藏
        
        model_info_layout = QVBoxLayout(self.model_info_group)
        
        # 模型名称标签
        self.model_name_label = QLabel(tr("model_not_loaded"))
        model_info_layout.addWidget(self.model_name_label)
        
        # 模型路径
        self.model_path_label = QLabel(tr("model_path_none"))
        model_path_font = QFont()
        model_path_font.setPointSize(9)
        self.model_path_label.setFont(model_path_font)
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: #888888;")
        model_info_layout.addWidget(self.model_path_label)
        
        # 类别数量
        self.model_classes_label = QLabel(tr("model_classes_zero"))
        model_info_layout.addWidget(self.model_classes_label)
        
        # 置信度阈值调整
        conf_layout = QHBoxLayout()
        self.conf_label = QLabel(tr("confidence"))
        self.conf_label.setFixedWidth(70)
        conf_layout.addWidget(self.conf_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)  # 0.01-1.00 的百分比
        self.conf_slider.setValue(25)  # 默认 0.25
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.on_confidence_slider_changed)
        conf_layout.addWidget(self.conf_slider)
        
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.00)
        self.conf_spinbox.setSingleStep(0.01)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.valueChanged.connect(self.on_confidence_spinbox_changed)
        self.conf_spinbox.setFixedWidth(60)
        conf_layout.addWidget(self.conf_spinbox)
        
        model_info_layout.addLayout(conf_layout)
        
        # IoU 阈值调整
        iou_layout = QHBoxLayout()
        self.iou_label = QLabel(tr("iou_threshold"))
        self.iou_label.setFixedWidth(70)
        iou_layout.addWidget(self.iou_label)
        
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)  # 0.01-1.00 的百分比
        self.iou_slider.setValue(45)  # 默认 0.45
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.setTickInterval(10)
        self.iou_slider.valueChanged.connect(self.on_iou_slider_changed)
        iou_layout.addWidget(self.iou_slider)
        
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.01, 1.00)
        self.iou_spinbox.setSingleStep(0.01)
        self.iou_spinbox.setValue(0.45)
        self.iou_spinbox.setDecimals(2)
        self.iou_spinbox.valueChanged.connect(self.on_iou_spinbox_changed)
        self.iou_spinbox.setFixedWidth(60)
        iou_layout.addWidget(self.iou_spinbox)
        
        model_info_layout.addLayout(iou_layout)
        
        # 状态指示器
        self.model_status_indicator = QLabel(tr("model_status_unloaded"))
        self.model_status_indicator.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        model_info_layout.addWidget(self.model_status_indicator)
        
        # 模型操作按钮
        model_btn_layout = QHBoxLayout()
        
        self.btn_unload_model = QPushButton(tr("unload_model"))
        self.btn_unload_model.clicked.connect(self.unload_model)
        self.btn_unload_model.setEnabled(False)
        model_btn_layout.addWidget(self.btn_unload_model)
        
        self.btn_train_model = QPushButton(tr("train_model_btn"))
        self.btn_train_model.clicked.connect(self.train_model)
        self.btn_train_model.setEnabled(True)
        model_btn_layout.addWidget(self.btn_train_model)
        
        self.btn_refresh_info = QPushButton(tr("refresh_info"))
        self.btn_refresh_info.clicked.connect(self.update_model_info_panel)
        model_btn_layout.addWidget(self.btn_refresh_info)
        
        model_info_layout.addLayout(model_btn_layout)
        
        model_info_layout.addStretch()
        
        splitter.addWidget(self.model_info_group)
        
        # 设置分割器的初始大小比例
        splitter.setSizes([150, 100, 100, 200, 150])
        
        # 设置分割器手柄样式
        splitter.setHandleWidth(6)
        
        return panel
    
    def init_actions(self):
        """初始化动作"""
        # 文件操作
        self.action_open_folder = QAction(tr("open_folder"), self)
        self.action_open_folder.setShortcut(QKeySequence.Open)
        self.action_open_folder.triggered.connect(self.load_image_folder)
        
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
        self.action_dark_theme.triggered.connect(self.switch_to_dark_theme)
        self.theme_menu.addAction(self.action_dark_theme)
        
        self.action_light_theme = QAction(tr("light_theme"), self)
        self.action_light_theme.triggered.connect(self.switch_to_light_theme)
        self.theme_menu.addAction(self.action_light_theme)
        
        # 模型菜单
        self.model_menu = menubar.addMenu(tr("model"))
        self.model_menu.addAction(self.action_load_model)
        self.model_menu.addAction(self.action_model_info)
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
                if saved_theme in ['dark', 'light']:
                    self.current_theme = saved_theme
                    self.logger.info(f"加载保存的主题: {self.current_theme}")
            
            # 加载语言设置
            if self.config.has_option('preferences', 'language'):
                saved_language = self.config.get('preferences', 'language')
                if saved_language in ['zh_CN', 'en_US']:
                    # 设置翻译管理器语言（但不立即更新UI，将在init_ui后应用）
                    from src.utils.i18n import TranslationManager
                    translation_manager = TranslationManager.instance()
                    translation_manager.current_language = saved_language
                    self.logger.info(f"加载保存的语言: {saved_language}")
    
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
        """加载图片文件夹"""
        # 默认路径改为当前程序的运行路径
        current_dir = Path.cwd()
        folder_path = QFileDialog.getExistingDirectory(
            self, tr("select_image_folder_dialog_title"), 
            str(current_dir)
        )
        
        if folder_path:
            self.image_manager.load_folder(folder_path)
            self.update_image_list()
            self.update_stats()
            
            if self.image_manager.get_image_count() > 0:
                self.load_image(0)
    
    def load_image(self, index: int):
        """加载指定索引的图片"""
        if 0 <= index < self.image_manager.get_image_count():
            image_path = self.image_manager.get_image_path(index)
            self.current_image_path = image_path
            self.current_image_index = index
            
            # 加载图片
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 清除场景
                self.graphics_scene.clear()
                
                # 添加图片
                self.image_item = QGraphicsPixmapItem(pixmap)
                self.graphics_scene.addItem(self.image_item)
                
                # 更新视图
                self.graphics_view.setSceneRect(self.image_item.boundingRect())
                
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
        if self.current_image_path and hasattr(self, 'image_item') and self.image_item:
            image_name = os.path.basename(self.current_image_path)
            image_size = self.image_item.pixmap().size()
            
            info = f"{image_name} | {image_size.width()}x{image_size.height()}"
            self.image_info_label.setText(info)
            self.status_image_info.setText(f"{tr('image_name_label')}: {image_name}")
        else:
            self.image_info_label.setText(tr("no_image_loaded"))
            self.status_image_info.setText("")
    
    def update_stats(self):
        """更新统计信息"""
        from src.utils.i18n import tr
        count = self.image_manager.get_image_count()
        self.stats_label.setText(tr("total_images_count").replace("{count}", str(count)))
    
    def update_statistics_panel(self):
        """更新标注统计面板"""
        try:
            # 获取图片总数
            total_images = self.image_manager.get_image_count()
            
            if total_images == 0:
                self.stats_total_label.setText(tr("total_images_zero"))
                self.stats_annotated_label.setText(tr("annotated_images_zero"))
                self.stats_unannotated_label.setText(tr("unannotated_images_zero"))
                self.stats_table.setRowCount(0)
                return
            
            # 统计已标注和未标注的图片
            annotated_count = 0
            unannotated_count = 0
            class_counts = {}  # 统计各类别标注数量
            
            for i in range(total_images):
                image_path = self.image_manager.get_image_path(i)
                has_annotations = self.annotation_manager.has_annotations(image_path)
                
                if has_annotations:
                    annotated_count += 1
                    
                    # 获取当前图片的标注，统计类别
                    annotations = self.annotation_manager.get_annotations(image_path)
                    for annotation in annotations:
                        class_id = annotation.class_id
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                else:
                    unannotated_count += 1
            
            # 更新标签
            self.stats_total_label.setText(tr("total_images_label") + f" {total_images}")
            self.stats_annotated_label.setText(tr("annotated_images_label") + f" {annotated_count}")
            self.stats_unannotated_label.setText(tr("unannotated_images_label") + f" {unannotated_count}")
            
            # 更新类别统计表格
            self.stats_table.setRowCount(len(class_counts))
            self.stats_table.setSortingEnabled(False)  # 在填充数据时禁用排序
            
            # 按类别ID排序
            sorted_class_ids = sorted(class_counts.keys())
            
            for row, class_id in enumerate(sorted_class_ids):
                count = class_counts[class_id]
                
                # 获取类别名称
                class_info = self.class_manager.get_class(class_id)
                class_name = class_info["name"] if class_info else tr("unknown_class").replace("{class_id}", str(class_id))
                
                # 设置ID单元格
                id_item = QTableWidgetItem(str(class_id))
                id_item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(row, 0, id_item)
                
                # 设置名称单元格
                name_item = QTableWidgetItem(class_name)
                self.stats_table.setItem(row, 1, name_item)
                
                # 设置数量单元格
                count_item = QTableWidgetItem(str(count))
                count_item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(row, 2, count_item)
            
            self.stats_table.setSortingEnabled(True)  # 重新启用排序
            self.stats_table.resizeColumnsToContents()
            
        except Exception as e:
            self.logger.error(f"更新统计面板失败: {e}")
            QMessageBox.warning(self, tr("error"), tr("update_stats_failed") + str(e))
    
    def on_image_item_clicked(self, item):
        """图片列表项点击事件"""
        index = self.image_list_widget.row(item)
        self.load_image(index)
    
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
        # 检查场景中是否有图片项
        image_items = [item for item in self.graphics_scene.items() 
                      if isinstance(item, QGraphicsPixmapItem)]
        
        if not image_items:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
            
        image_item = image_items[0]
        self.graphics_view.fitInView(image_item, Qt.KeepAspectRatio)
        self.update_scale_factor()
    
    def zoom_in(self):
        """放大"""
        self.graphics_view.scale(1.25, 1.25)
        self.update_scale_factor()
    
    def zoom_out(self):
        """缩小"""
        self.graphics_view.scale(0.8, 0.8)
        self.update_scale_factor()
    
    def reset_view(self):
        """重置视图"""
        self.graphics_view.resetTransform()
        self.scale_factor = 1.0
        self.update_status(tr("view_reset_message"))
    
    def update_scale_factor(self):
        """更新缩放因子"""
        transform = self.graphics_view.transform()
        self.scale_factor = transform.m11()
        self.update_status(tr("zoom_status").replace("{scale_factor}", f"{self.scale_factor:.2f}"))
    
    # ==================== 类别管理 ====================
    
    def update_class_list(self):
        """更新类别列表"""
        self.class_list_widget.clear()
        
        classes = self.class_manager.get_classes()
        for class_id, class_info in classes.items():
            class_name = class_info["name"]
            color = class_info["color"]
            
            item_text = f"{class_id}: {class_name}"
            self.class_list_widget.addItem(item_text)
            
            # 设置背景颜色
            item = self.class_list_widget.item(self.class_list_widget.count() - 1)
            item.setBackground(QColor(*color))
            
            # 设置文字颜色为对比色
            brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
            text_color = QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)
            item.setForeground(text_color)
    
    def on_class_item_clicked(self, item):
        """类别列表项点击事件"""
        # 从列表项文本中提取class_id
        item_text = item.text()
        if ": " in item_text:
            try:
                class_id_str = item_text.split(": ")[0]
                self.selected_class_id = int(class_id_str)
                self.update_status(tr("selected_class").replace("{class_name}", self.class_manager.get_class_name(self.selected_class_id)))
            except ValueError:
                self.update_status(tr("cannot_parse_class_id_error"))
        else:
            self.update_status(tr("invalid_class_format_error"))
    
    def add_class(self):
        """添加类别"""
        from .class_dialog import ClassDialog
        
        dialog = ClassDialog(self)
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
        
        index = self.class_list_widget.row(selected_items[0])
        class_info = self.class_manager.get_class(index)
        
        # 检查类别是否存在
        if not class_info:
            QMessageBox.warning(self, tr("warning"), tr("class_not_exist"))
            return
        
        from .class_dialog import ClassDialog
        
        dialog = ClassDialog(self)
        dialog.set_values(class_info["name"], class_info["color"])
        
        if dialog.exec():
            class_name, color = dialog.get_values()
            self.class_manager.update_class(index, class_name, color)
            self.update_class_list()
    
    def delete_class(self):
        """删除类别"""
        selected_items = self.class_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, tr("warning"), tr("select_class"))
            return
        
        index = self.class_list_widget.row(selected_items[0])
        
        # 获取要删除的类别名称
        class_info = self.class_manager.get_class(index)
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
            if index == self.selected_class_id:
                self.selected_class_id = 0
                if self.class_manager.get_class_count() > 0:
                    # 选择第一个可用的类别
                    available_classes = list(self.class_manager.get_classes().keys())
                    if available_classes:
                        self.selected_class_id = available_classes[0]
            
            # 删除类别
            self.class_manager.delete_class(index)
            self.update_class_list()
            
            # 更新状态
            self.update_status(tr("class_deleted").replace("{class_name}", class_name))
    
    # ==================== 标注管理 ====================
    
    def load_annotations_for_current_image(self):
        """加载当前图片的标注"""
        if self.current_image_path:
            annotations = self.annotation_manager.get_annotations(self.current_image_path)
            self.draw_annotations(annotations)
    
    def draw_annotations(self, annotations: List[Annotation]):
        """绘制标注框"""
        # 清除现有标注图形项
        for item in self.graphics_scene.items():
            if hasattr(item, 'is_annotation_item') and item.is_annotation_item:
                self.graphics_scene.removeItem(item)
        
        # 绘制新的标注框
        for annotation in annotations:
            self.draw_annotation_box(annotation)
    
    def draw_annotation_box(self, annotation: Annotation):
        """绘制单个标注框"""
            # 获取类别颜色（检查类别是否存在）
        class_info = self.class_manager.get_class(annotation.class_id)
        if not class_info:
            # 如果类别不存在，使用默认灰色
            color = QColor(128, 128, 128)
            class_name = tr("unknown_class").replace("{class_id}", str(annotation.class_id))
        else:
            color = QColor(*class_info["color"])
            class_name = class_info["name"]
        
        # 创建自定义矩形框项
        rect_item = AnnotationRectItem(
            annotation.x, annotation.y,
            annotation.width, annotation.height,
            annotation, color
        )
        
        # 设置属性
        rect_item.is_annotation_item = True
        
        # 添加到场景
        self.graphics_scene.addItem(rect_item)
        
        # 添加类别标签
        text_item = QGraphicsTextItem(class_name)
        text_item.setDefaultTextColor(color)
        text_item.setPos(annotation.x, annotation.y - 20)
        
        # 设置属性
        text_item.is_annotation_item = True
        text_item.associated_rect_item = rect_item  # 链接到矩形项
        
        self.graphics_scene.addItem(text_item)
    
    def save_annotations(self):
        """保存标注"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded_warning"))
            return
        
        # 获取当前图片的所有标注
        annotations = []
        for item in self.graphics_scene.items():
            if hasattr(item, 'annotation'):
                annotations.append(item.annotation)
        
        self.annotation_manager.save_annotations(self.current_image_path, annotations)
        self.update_status(tr("annotations_saved").replace("{count}", str(len(annotations))))
    
    def delete_selected_annotation(self):
        """删除选中标注（仅删除当前选中的标注框）"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        # 获取当前选中的标注项（理论上应该只有一个）
        selected_rect_items = []
        for item in self.graphics_scene.items():
            if (hasattr(item, 'annotation') and 
                hasattr(item, 'is_selected') and 
                item.is_selected):
                selected_rect_items.append(item)
        
        if not selected_rect_items:
            QMessageBox.warning(self, tr("warning"), tr("no_annotation_selected"))
            return
        
        # 只处理第一个选中的标注项（确保只删除当前选中的）
        rect_item = selected_rect_items[0]
        
        # 获取当前所有标注，找到选中的索引
        annotations = self.annotation_manager.get_annotations(self.current_image_path)
        
        # 查找标注在列表中的索引
        annotation_index = -1
        for i, ann in enumerate(annotations):
            if (abs(ann.x - rect_item.annotation.x) < 1 and 
                abs(ann.y - rect_item.annotation.y) < 1 and 
                abs(ann.width - rect_item.annotation.width) < 1 and 
                abs(ann.height - rect_item.annotation.height) < 1 and 
                ann.class_id == rect_item.annotation.class_id):
                annotation_index = i
                break
        
        if annotation_index >= 0:
            # 使用命令模式删除标注
            from src.core.annotation import DeleteAnnotationCommand
            command = DeleteAnnotationCommand(
                self.annotation_manager,
                self.current_image_path,
                annotation_index
            )
            self.annotation_manager.execute_command(command)
        
        # 重新加载标注以更新UI
        self.load_annotations_for_current_image()
        
        # 更新图片列表显示状态
        self.update_image_list()
        
        self.update_status(tr("annotation_deleted"))
        
        # 更新菜单状态
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
            # 清除图形项
            items_to_remove = []
            for item in self.graphics_scene.items():
                if hasattr(item, 'is_annotation_item'):
                    items_to_remove.append(item)
            
            for item in items_to_remove:
                self.graphics_scene.removeItem(item)
            
            # 清除标注数据
            self.annotation_manager.clear_annotations(self.current_image_path)
            
            # 清除缓存，确保下次加载时从文件读取
            if self.current_image_path in self.annotation_manager._annotations:
                del self.annotation_manager._annotations[self.current_image_path]
            
            # 更新图片列表显示
            self.update_image_list()
            
            self.update_status(tr("annotations_cleared").replace("{count}", str(len(items_to_remove))))
    
    # ==================== 导出功能 ====================
    
    def export_yolo_format(self):
        """导出YOLO格式"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, tr("export_yolo_format_dialog_title"), 
            str(Path.cwd())
        )
        
        if output_dir:
            try:
                self.yolo_exporter.export(
                    self.image_manager,
                    self.annotation_manager,
                    self.class_manager,
                    output_dir
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
            str(Path.cwd())
        )
        
        if output_dir:
            try:
                from src.utils.dataset_splitter import DatasetSplitter
                
                splitter = DatasetSplitter()
                splitter.split_and_export(
                    self.image_manager,
                    self.annotation_manager,
                    output_dir
                )
                QMessageBox.information(self, tr("success"), f"{tr('dataset_split_export_success')} {output_dir}")
            except Exception as e:
                QMessageBox.critical(self, tr("error"), f"{tr('export_failed')} {str(e)}")
    
    # ==================== 模型辅助标注 ====================
    
    def auto_annotate_current(self):
        """自动标注当前图片"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        QMessageBox.information(self, tr("info"), tr("model_help_needed"))
        # TODO: 实现模型加载和推理
    
    def batch_auto_annotate(self):
        """批量自动标注"""
        if self.image_manager.get_image_count() == 0:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        QMessageBox.information(self, tr("info"), tr("batch_model_help_needed"))
        # TODO: 实现批量推理
    
    # ==================== 撤销/重做方法 ====================
    
    def undo(self):
        """撤销"""
        from src.utils.i18n import tr
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
        from src.utils.i18n import tr
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
        
        # 更新动作状态
        self.action_undo.setEnabled(can_undo)
        self.action_redo.setEnabled(can_redo)
        
        # 更新按钮状态
        for action in self.toolbar.actions():
            if action.text() == tr("undo"):
                action.setEnabled(can_undo)
            elif action.text() == tr("redo"):
                action.setEnabled(can_redo)
    
    # ==================== 其他方法 ====================
    
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_label.setText(message)
    
    
    def load_classes_from_yaml(self):
        """从YAML文件加载类别"""
        current_dir = Path.cwd()
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, tr("load_classes_from_yaml_dialog"), 
            str(current_dir),
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
        
        current_dir = Path.cwd()
        yaml_path, _ = QFileDialog.getSaveFileName(
            self, tr("save_classes_to_yaml_dialog"), 
            str(current_dir),
            tr("yaml_file_filter")
        )
        
        if yaml_path:
            # 确保文件扩展名
            if not yaml_path.lower().endswith(('.yaml', '.yml')):
                yaml_path += '.yaml'
            
            # 导出YAML
            self.class_manager.export_to_yaml(yaml_path)
            QMessageBox.information(self, tr("success"), tr("save_yaml_success") + f" {yaml_path}")
    
    def eventFilter(self, obj, event):
        """事件过滤器"""
        if obj is self.graphics_view.viewport():
            if event.type() == QEvent.MouseButtonPress:
                return self.on_graphics_view_mouse_press(event)
            elif event.type() == QEvent.MouseMove:
                return self.on_graphics_view_mouse_move(event)
            elif event.type() == QEvent.MouseButtonRelease:
                return self.on_graphics_view_mouse_release(event)
            elif event.type() == QEvent.Wheel:
                return self.on_graphics_view_wheel(event)
        
        return super().eventFilter(obj, event)
    
    def on_graphics_view_mouse_press(self, event: QMouseEvent):
        """图形视图鼠标按下事件"""
        scene_pos = self.graphics_view.mapToScene(event.pos())
        
        # 检查是否点击了标注框
        clicked_items = self.graphics_view.items(event.pos())
        for item in clicked_items:
            if hasattr(item, 'annotation') and isinstance(item, QGraphicsRectItem):
                # 点击的是标注框，不开始绘制，让事件传递给标注框处理
                self.is_drawing = False
                return False  # 返回 False 让事件继续传递
        
        if event.button() == Qt.LeftButton:
            # 开始绘制标注框
            self.is_drawing = True
            self.drawing_start_point = scene_pos
            self.drawing_end_point = scene_pos
            
            # 设置十字形鼠标
            self.graphics_view.viewport().setCursor(Qt.CrossCursor)
            
            # 创建临时绘制项
            self.temp_rect_item = QGraphicsRectItem()
            self.temp_rect_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
            self.graphics_scene.addItem(self.temp_rect_item)
            
            return True
        
        return False
    
    def on_graphics_view_mouse_move(self, event: QMouseEvent):
        """图形视图鼠标移动事件"""
        # 总是设置十字形鼠标
        self.graphics_view.viewport().setCursor(Qt.CrossCursor)
        
        # 获取场景坐标
        scene_pos = self.graphics_view.mapToScene(event.pos())
        
        # 如果正在绘制，更新临时矩形
        if self.is_drawing and hasattr(self, 'temp_rect_item') and self.temp_rect_item:
            self.drawing_end_point = scene_pos
            
            # 更新临时矩形框
            if self.drawing_start_point:
                rect = QRectF(
                    min(self.drawing_start_point.x(), scene_pos.x()),
                    min(self.drawing_start_point.y(), scene_pos.y()),
                    abs(self.drawing_start_point.x() - scene_pos.x()),
                    abs(self.drawing_start_point.y() - scene_pos.y())
                )
                if self.temp_rect_item:
                    self.temp_rect_item.setRect(rect)
        else:
            # 如果不是正在绘制，绘制十字准线
            self.draw_crosshair(scene_pos)
        
        return True
    
    def draw_crosshair(self, scene_pos: QPointF):
        """绘制十字准线"""
        # 移除现有的十字准线
        for item in self.graphics_scene.items():
            if hasattr(item, 'is_crosshair') and item.is_crosshair:
                self.graphics_scene.removeItem(item)
        
        # 获取场景边界
        scene_rect = self.graphics_scene.sceneRect()
        if scene_rect.isNull():
            return
        
        # 创建垂直线
        vline = QGraphicsLineItem(scene_rect.x(), scene_pos.y(), scene_rect.x() + scene_rect.width(), scene_pos.y())
        vline.setPen(QPen(QColor(255, 255, 0, 128), 1, Qt.DashLine))  # 半透明黄色的虚线
        vline.is_crosshair = True
        self.graphics_scene.addItem(vline)
        
        # 创建水平线
        hline = QGraphicsLineItem(scene_pos.x(), scene_rect.y(), scene_pos.x(), scene_rect.y() + scene_rect.height())
        hline.setPen(QPen(QColor(255, 255, 0, 128), 1, Qt.DashLine))  # 半透明黄色的虚线
        hline.is_crosshair = True
        self.graphics_scene.addItem(hline)
        
        # 在交点处画一个小圆点
        dot = QGraphicsEllipseItem(scene_pos.x() - 2, scene_pos.y() - 2, 4, 4)
        dot.setBrush(QBrush(QColor(255, 0, 0, 200)))  # 半透明的红色
        dot.setPen(QPen(Qt.NoPen))
        dot.is_crosshair = True
        self.graphics_scene.addItem(dot)
    
    def on_graphics_view_mouse_release(self, event: QMouseEvent):
        """图形视图鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            
            # 移除临时绘制项
            if hasattr(self, 'temp_rect_item'):
                self.graphics_scene.removeItem(self.temp_rect_item)
                delattr(self, 'temp_rect_item')
            
            # 创建标注
            if self.drawing_start_point and self.drawing_end_point:
                rect = QRectF(
                    min(self.drawing_start_point.x(), self.drawing_end_point.x()),
                    min(self.drawing_start_point.y(), self.drawing_end_point.y()),
                    abs(self.drawing_start_point.x() - self.drawing_end_point.x()),
                    abs(self.drawing_start_point.y() - self.drawing_end_point.y())
                )
                
                # 确保矩形大小合理
                if rect.width() > 10 and rect.height() > 10:
                    annotation = Annotation(
                        rect.x(), rect.y(),
                        rect.width(), rect.height(),
                        self.selected_class_id
                    )
                    
                    # 使用命令模式添加标注
                    self.add_annotation_with_command(annotation)
                    
                self.update_status(tr("annotation_created").replace("{class_name}", self.class_manager.get_class_name(self.selected_class_id)))
            
            self.drawing_start_point = None
            self.drawing_end_point = None
            
            return True
        
        return False
    
    def add_annotation_with_command(self, annotation: Annotation):
        """使用命令模式添加标注"""
        if not self.current_image_path:
            return
        
        # 创建添加命令
        from src.core.annotation import AddAnnotationCommand
        command = AddAnnotationCommand(
            self.annotation_manager,
            self.current_image_path,
            annotation
        )
        
        # 执行命令
        self.annotation_manager.execute_command(command)
        
        # 更新UI显示
        self.draw_annotation_box(annotation)
        
        # 更新菜单状态
        self.update_undo_redo_actions()
    
    def on_graphics_view_wheel(self, event: QWheelEvent):
        """图形视图滚轮事件"""
        # 按住Ctrl键时缩放
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True
        
        return False
    
    # ==================== 模型管理方法 ====================
    
    def load_model(self):
        """加载YOLO模型"""
        current_dir = Path.cwd()
        model_path, _ = QFileDialog.getOpenFileName(
            self, tr("model_file_selection"),
            str(current_dir),
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
                self.update_model_info_panel()
                
                # 如果模型信息面板是隐藏的，自动显示它
                if not self.model_info_group.isVisible():
                    self.model_info_group.setVisible(True)
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
        is_visible = not self.model_info_group.isVisible()
        self.model_info_group.setVisible(is_visible)
        
        # 如果显示面板，更新内容
        if is_visible:
            self.update_model_info_panel()
            # 确保UI文本使用当前语言
            self.update_other_ui_elements()
        
        # 更新菜单文本
        if is_visible:
            self.action_model_info.setText(tr("hide_model_info"))
            self.update_status(tr("model_info_panel_shown"))
        else:
            self.action_model_info.setText(tr("show_model_info"))
            self.update_status(tr("model_info_panel_hidden"))
    
    def auto_annotate_current(self):
        """自动标注当前图片"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
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
            
            # 清空现有标注
            for item in self.graphics_scene.items():
                if hasattr(item, 'is_annotation_item') and item.is_annotation_item:
                    self.graphics_scene.removeItem(item)
            
            # 绘制新标注
            for annotation in annotations:
                self.draw_annotation_box(annotation)
            
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
        from src.utils.i18n import tr
        
        if self.image_manager.get_image_count() == 0:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return
        
        # 确认批量标注
        reply = QMessageBox.question(
            self, tr("confirm_batch_annotation"),
            tr("batch_annotation_confirmation").replace("{image_count}", str(self.image_manager.get_image_count())),
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
    
    def on_confidence_slider_changed(self, value: int):
        """置信度滑块值改变"""
        # 将滑块值(1-100)转换为0.01-1.00
        conf_value = value / 100.0
        self.conf_spinbox.blockSignals(True)
        self.conf_spinbox.setValue(conf_value)
        self.conf_spinbox.blockSignals(False)
        
        # 更新模型管理器中的置信度阈值
        if self.model_manager.is_model_loaded():
            self.model_manager.set_confidence_threshold(conf_value)
            self.update_status(tr("confidence_threshold_set").replace("{value}", f"{conf_value:.2f}"))
    
    def on_confidence_spinbox_changed(self, value: float):
        """置信度SpinBox值改变"""
        from src.utils.i18n import tr
        # 将值(0.01-1.00)转换为滑块值(1-100)
        slider_value = int(value * 100)
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(slider_value)
        self.conf_slider.blockSignals(False)
        
        # 更新模型管理器中的置信度阈值
        if self.model_manager.is_model_loaded():
            self.model_manager.set_confidence_threshold(value)
            self.update_status(tr("confidence_threshold_set").replace("{value}", f"{value:.2f}"))
    
    def on_iou_slider_changed(self, value: int):
        """IoU滑块值改变"""
        # 将滑块值(1-100)转换为0.01-1.00
        iou_value = value / 100.0
        self.iou_spinbox.blockSignals(True)
        self.iou_spinbox.setValue(iou_value)
        self.iou_spinbox.blockSignals(False)
        
        # 更新模型管理器中的IoU阈值
        if self.model_manager.is_model_loaded():
            self.model_manager.set_iou_threshold(iou_value)
            self.update_status(tr("iou_threshold_set").replace("{value}", f"{iou_value:.2f}"))
    
    def on_iou_spinbox_changed(self, value: float):
        """IoU SpinBox值改变"""
        from src.utils.i18n import tr
        # 将值(0.01-1.00)转换为滑块值(1-100)
        slider_value = int(value * 100)
        self.iou_slider.blockSignals(True)
        self.iou_slider.setValue(slider_value)
        self.iou_slider.blockSignals(False)
        
        # 更新模型管理器中的IoU阈值
        if self.model_manager.is_model_loaded():
            self.model_manager.set_iou_threshold(value)
            self.update_status(tr("iou_threshold_set").replace("{value}", f"{value:.2f}"))
    
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
            # 卸载模型
            self.model_manager.model = None
            self.model_manager.model_path = None
            self.model_manager.model_loaded = False
            self.model_manager.class_names = {}
            
            # 更新UI状态
            self.btn_unload_model.setEnabled(False)
            self.conf_slider.setEnabled(False)
            self.conf_spinbox.setEnabled(False)
            self.iou_slider.setEnabled(False)
            self.iou_spinbox.setEnabled(False)
            
            # 更新模型信息面板
            self.update_model_info_panel()
            
            self.update_status(tr("model_unloaded"))
            QMessageBox.information(self, tr("success"), tr("model_unloaded"))
    
    def update_model_info_panel(self):
        """更新模型信息面板内容"""
        from src.utils.i18n import tr
        
        model_info = self.model_manager.get_model_info()
        
        if not model_info["loaded"]:
            self.model_name_label.setText(tr("model_not_loaded"))
            self.model_path_label.setText(f"{tr('model_path')} {tr('none')}")
            self.model_classes_label.setText(f"{tr('model_classes')} 0")
            self.model_status_indicator.setText("● " + tr("model_status_unloaded"))
            self.model_status_indicator.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            
            # 禁用控件
            self.conf_slider.setEnabled(False)
            self.conf_spinbox.setEnabled(False)
            self.iou_slider.setEnabled(False)
            self.iou_spinbox.setEnabled(False)
            self.btn_unload_model.setEnabled(False)
            return
        
        # 更新模型信息
        model_name = os.path.basename(model_info["path"]) if model_info["path"] else tr("unknown_model")
        self.model_name_label.setText(f"{tr('model_label')} {model_name}")
        self.model_path_label.setText(f"{tr('model_path')} {model_info['path']}")
        self.model_classes_label.setText(f"{tr('model_classes')} {model_info['class_count']}")
        self.model_status_indicator.setText("● " + tr("model_status_loaded"))
        self.model_status_indicator.setStyleSheet("color: #6bff6b; font-weight: bold;")
        
        # 更新阈值控件
        conf_value = model_info["confidence_threshold"]
        iou_value = model_info["iou_threshold"]
        
        # 更新滑块和SpinBox
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
        
        # 启用卸载按钮
        self.btn_unload_model.setEnabled(True)
    
    def train_model(self):
        """训练模型"""
        from src.utils.i18n import tr
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
        from src.utils.i18n import tr
        
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
        from src.utils.i18n import tr
        
        # 使用已保存的菜单引用
        if hasattr(self, 'file_menu'):
            self.file_menu.setTitle(tr("file"))
        if hasattr(self, 'edit_menu'):
            self.edit_menu.setTitle(tr("edit"))
        if hasattr(self, 'view_menu'):
            self.view_menu.setTitle(tr("view"))
        if hasattr(self, 'class_menu'):
            self.class_menu.setTitle(tr("classes"))
        if hasattr(self, 'theme_menu'):
            self.theme_menu.setTitle(tr("theme"))
        if hasattr(self, 'model_menu'):
            self.model_menu.setTitle(tr("model"))
        if hasattr(self, 'annotate_menu'):
            self.annotate_menu.setTitle(tr("annotate"))
        if hasattr(self, 'language_menu'):
            self.language_menu.setTitle(tr("language"))
        
        # 更新动作文本
        if hasattr(self, 'action_open_folder'):
            self.action_open_folder.setText(tr("open_folder"))
        if hasattr(self, 'action_save'):
            self.action_save.setText(tr("save_annotations"))
        if hasattr(self, 'action_export'):
            self.action_export.setText(tr("export_yolo_format"))
        if hasattr(self, 'action_exit'):
            self.action_exit.setText(tr("exit"))
        
        if hasattr(self, 'action_undo'):
            self.action_undo.setText(tr("undo"))
        if hasattr(self, 'action_redo'):
            self.action_redo.setText(tr("redo"))
        if hasattr(self, 'action_delete'):
            self.action_delete.setText(tr("delete_selected"))
        
        if hasattr(self, 'action_zoom_in'):
            self.action_zoom_in.setText(tr("zoom_in"))
        if hasattr(self, 'action_zoom_out'):
            self.action_zoom_out.setText(tr("zoom_out"))
        if hasattr(self, 'action_fit'):
            self.action_fit.setText(tr("fit_to_window"))
        
        if hasattr(self, 'action_load_model'):
            self.action_load_model.setText(tr("load_model"))
        if hasattr(self, 'action_model_info'):
            self.action_model_info.setText(tr("model_info"))
        if hasattr(self, 'action_train_model'):
            self.action_train_model.setText(tr("train_model"))
        if hasattr(self, 'action_auto_annotate'):
            self.action_auto_annotate.setText(tr("auto_annotate"))
        if hasattr(self, 'action_batch_auto_annotate'):
            self.action_batch_auto_annotate.setText(tr("batch_auto_annotate"))
        
        if hasattr(self, 'action_load_yaml'):
            self.action_load_yaml.setText(tr("load_yaml"))
        if hasattr(self, 'action_save_yaml'):
            self.action_save_yaml.setText(tr("save_yaml"))
        
        if hasattr(self, 'action_dark_theme'):
            self.action_dark_theme.setText(tr("dark_theme"))
        if hasattr(self, 'action_light_theme'):
            self.action_light_theme.setText(tr("light_theme"))
        
        if hasattr(self, 'action_chinese'):
            self.action_chinese.setText(tr("chinese"))
        if hasattr(self, 'action_english'):
            self.action_english.setText(tr("english"))
    
    def update_button_texts(self):
        """更新按钮文本"""
        from src.utils.i18n import tr
        
        # 左侧面板按钮
        if hasattr(self, 'btn_load_folder'):
            self.btn_load_folder.setText(tr("load_folder"))
        if hasattr(self, 'btn_prev'):
            self.btn_prev.setText(tr("previous_image"))
        if hasattr(self, 'btn_next'):
            self.btn_next.setText(tr("next_image"))
        
        # 中间面板按钮
        if hasattr(self, 'btn_fit'):
            self.btn_fit.setText(tr("fit_to_window"))
        if hasattr(self, 'btn_zoom_in'):
            self.btn_zoom_in.setText(tr("zoom_in"))
        if hasattr(self, 'btn_zoom_out'):
            self.btn_zoom_out.setText(tr("zoom_out"))
        if hasattr(self, 'btn_reset'):
            self.btn_reset.setText(tr("reset"))
        
        # 右侧面板按钮
        if hasattr(self, 'btn_add_class'):
            self.btn_add_class.setText(tr("add"))
        if hasattr(self, 'btn_edit_class'):
            self.btn_edit_class.setText(tr("edit"))
        if hasattr(self, 'btn_delete_class'):
            self.btn_delete_class.setText(tr("delete"))
        
        if hasattr(self, 'btn_delete_annotation'):
            self.btn_delete_annotation.setText(tr("delete_selected_annotation"))
        if hasattr(self, 'btn_clear_all'):
            self.btn_clear_all.setText(tr("clear_all_annotations"))
        
        if hasattr(self, 'btn_export_yolo'):
            self.btn_export_yolo.setText(tr("export_yolo"))
        if hasattr(self, 'btn_export_split'):
            self.btn_export_split.setText(tr("export_dataset_split"))
        
        if hasattr(self, 'btn_refresh_stats'):
            self.btn_refresh_stats.setText(tr("refresh_stats"))
        
        if hasattr(self, 'btn_unload_model'):
            self.btn_unload_model.setText(tr("unload_model"))
        if hasattr(self, 'btn_train_model'):
            self.btn_train_model.setText(tr("train_model_btn"))
        if hasattr(self, 'btn_refresh_info'):
            self.btn_refresh_info.setText(tr("refresh_info"))
    
    def update_panel_titles(self):
        """更新面板标题"""
        from src.utils.i18n import tr
        
        # 左侧面板标题
        if hasattr(self, 'left_panel_title_label'):
            self.left_panel_title_label.setText(tr("image_list"))
        
        # 中间面板标题
        if hasattr(self, 'center_panel_title_label'):
            self.center_panel_title_label.setText(tr("image_annotation"))
        
        # 右侧面板标题
        if hasattr(self, 'right_panel_title_label'):
            self.right_panel_title_label.setText(tr("class_management"))
    
    def update_other_ui_elements(self):
        """更新其他UI元素"""
        from src.utils.i18n import tr
        
        # 更新GroupBox标题
        if hasattr(self, 'class_group'):
            self.class_group.setTitle(tr("annotation_classes"))
        if hasattr(self, 'annotation_group'):
            self.annotation_group.setTitle(tr("annotation_operations"))
        if hasattr(self, 'export_group'):
            self.export_group.setTitle(tr("data_export"))
        if hasattr(self, 'stats_group'):
            self.stats_group.setTitle(tr("annotation_stats"))
        if hasattr(self, 'model_info_group'):
            self.model_info_group.setTitle(tr("model_info_panel"))
        
        # 更新状态信息
        if hasattr(self, 'stats_label') and self.image_manager.get_image_count() == 0:
            self.stats_label.setText(tr("no_image_loaded"))
        
        if hasattr(self, 'image_info_label') and not self.current_image_path:
            self.image_info_label.setText(tr("no_image_loaded"))
        
        if hasattr(self, 'model_name_label') and not self.model_manager.is_model_loaded():
            self.model_name_label.setText(tr("model_not_loaded"))
        
        # 更新统计信息标签
        if hasattr(self, 'stats_total_label') and hasattr(self, 'stats_annotated_label') and hasattr(self, 'stats_unannotated_label'):
            # 获取当前值
            total_text = self.stats_total_label.text()
            annotated_text = self.stats_annotated_label.text()
            unannotated_text = self.stats_unannotated_label.text()
            
            # 提取数字
            import re
            total_match = re.search(r'\d+', total_text)
            annotated_match = re.search(r'\d+', annotated_text)
            unannotated_match = re.search(r'\d+', unannotated_text)
            
            if total_match:
                total_num = total_match.group()
                self.stats_total_label.setText(f"{tr('total_images')} {total_num}")
            
            if annotated_match:
                annotated_num = annotated_match.group()
                self.stats_annotated_label.setText(f"{tr('annotated_images')} {annotated_num}")
            
            if unannotated_match:
                unannotated_num = unannotated_match.group()
                self.stats_unannotated_label.setText(f"{tr('unannotated_images')} {unannotated_num}")
        
        # 更新表格表头
        if hasattr(self, 'stats_table'):
            self.stats_table.setHorizontalHeaderLabels([tr("class_id"), tr("class_name"), tr("annotation_count")])
        
        # 更新模型信息面板标签
        if hasattr(self, 'model_path_label'):
            path_text = self.model_path_label.text()
            # 提取路径值
            if path_text.startswith("路径:"):
                path_value = path_text[3:].strip()
                self.model_path_label.setText(f"{tr('model_path')} {path_value}")
            elif path_text.startswith("Path:"):
                path_value = path_text[5:].strip()
                self.model_path_label.setText(f"{tr('model_path')} {path_value}")
        
        if hasattr(self, 'model_classes_label'):
            classes_text = self.model_classes_label.text()
            if classes_text.startswith("类别数:"):
                classes_value = classes_text[4:].strip()
                self.model_classes_label.setText(f"{tr('model_classes')} {classes_value}")
            elif classes_text.startswith("Classes:"):
                classes_value = classes_text[8:].strip()
                self.model_classes_label.setText(f"{tr('model_classes')} {classes_value}")
        
        # 更新模型状态指示器
        if hasattr(self, 'model_status_indicator'):
            status_text = self.model_status_indicator.text()
            if "未加载" in status_text or "not loaded" in status_text:
                self.model_status_indicator.setText("● " + tr("model_not_loaded"))
            elif "已加载" in status_text or "loaded" in status_text:
                self.model_status_indicator.setText("● " + tr("model_loaded"))
        
        # 更新置信度和IoU标签
        if hasattr(self, 'conf_label'):
            self.conf_label.setText(tr("confidence"))
        if hasattr(self, 'iou_label'):
            self.iou_label.setText(tr("iou_threshold"))
