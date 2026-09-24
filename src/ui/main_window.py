"""
主窗口实现
"""

import json
import configparser
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QSplitter, QLabel, QFileDialog, QMessageBox, QStatusBar
)
from PySide6.QtCore import (
    Qt,
)
from PySide6.QtGui import (
    QPixmap, QAction, QKeySequence, QShortcut,
)


from src.core.annotation import Annotation, AnnotationManager
from src.core.image_manager import ImageManager
from src.core.class_manager import ClassManager
from src.core.model_manager import ModelManager
from src.utils.yolo_exporter import YOLOExporter

from src.utils.logger import _get_app_root, get_logger_simple
from src.utils.i18n import tr

from src.ui.main_window_mixins import (
    ClassActionsMixin,
    ImageActionsMixin,
    ModelActionsMixin,
    PanelsMixin,
    ThemeLanguageMixin,
)


class MainWindow(
    ThemeLanguageMixin,
    PanelsMixin,
    ImageActionsMixin,
    ClassActionsMixin,
    ModelActionsMixin,
    QMainWindow,
):
    """主窗口类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 主题相关 - 必须在任何方法调用之前初始化
        self.current_theme = "dark"  # 默认使用黑夜主题
        
        # 配置管理器（应用根目录锚定，避免随启动 CWD 漂移）
        self.config_file_path = _get_app_root() / "config" / "config.ini"
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
        self.action_eyecare_theme.setChecked(self.current_theme == "eyecare")
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

        icon_path = _get_app_root() / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QPixmap(str(icon_path)))

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

        self.action_export_model = QAction(tr("export_model"), self)
        self.action_export_model.triggered.connect(self.export_model)
    
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

        self.action_eyecare_theme = QAction(tr("eyecare_theme"), self)
        self.action_eyecare_theme.setCheckable(True)
        self.action_eyecare_theme.triggered.connect(self.switch_to_eyecare_theme)
        self.theme_menu.addAction(self.action_eyecare_theme)

        # 模型菜单
        self.model_menu = menubar.addMenu(tr("model"))
        self.model_menu.addAction(self.action_load_model)
        self.model_menu.addAction(self.action_model_info)
        self.model_menu.addAction(self.action_validation_window)
        self.model_menu.addSeparator()
        self.model_menu.addAction(self.action_train_model)
        self.model_menu.addAction(self.action_export_model)

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
        self.toolbar.addAction(self.action_auto_annotate)
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
                if saved_theme in ['dark', 'light', 'colorful', 'eyecare']:
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
















    
    # ==================== 视图操作 ====================





    
    # ==================== 类别管理 ====================








    
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
        """删除选中标注（通过 annotation 对象定位索引，避免场景 item 顺序漂移导致误删）"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        sel_annotation = self.canvas.get_selected_annotation()
        if sel_annotation is None:
            QMessageBox.warning(self, tr("warning"), tr("no_annotation_selected"))
            return

        # 通过 annotation 对象在列表中查找索引，而非依赖 scene 的 annotation_index
        annotations = self.annotation_manager.get_annotations(self.current_image_path)
        annotation_index = -1
        for i, ann in enumerate(annotations):
            if ann is sel_annotation:
                annotation_index = i
                break

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

        # 通过 annotation 对象在列表中定位索引
        annotations = self.annotation_manager.get_annotations(self.current_image_path)
        annotation_index = -1
        for i, ann in enumerate(annotations):
            if ann is annotation:
                annotation_index = i
                break

        if annotation_index < 0:
            return

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





    
    # ==================== 模型参数调整方法 ====================










