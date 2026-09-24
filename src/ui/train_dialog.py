"""
模型训练配置对话框

P2 重构：按职责拆分至 src/ui/train_dialog_mixins/ 包，
本模块仅保留对话框骨架（__init__ / init_ui）与对外类名 TrainDialog。
"""

import configparser
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui.train_dialog_mixins import (
    TrainActionsMixin,
    TrainBrowseMixin,
    TrainConfigMixin,
    TrainTabsMixin,
)
from src.utils.i18n import tr
from yolo_tool import YOLOTrainer


class TrainDialog(
    TrainTabsMixin,
    TrainBrowseMixin,
    TrainConfigMixin,
    TrainActionsMixin,
    QDialog,
):
    """模型训练配置对话框"""


    def __init__(self, parent=None, default_model_path: str = ""):
        super().__init__(parent)
        
        # 训练器实例
        self.trainer = YOLOTrainer()
        
        # 默认模型路径
        self.default_model_path = default_model_path
        
        # 配置文件路径
        self.config_file_path = Path("config/config.ini")
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_parser = configparser.ConfigParser()
        
        # 训练配置
        self.config = self.trainer.get_default_config()
        if default_model_path:
            self.config['model_path'] = default_model_path
        
        # 配置修改标志
        self.config_modified = False

        # 文件对话框路径记忆
        self._last_browse_path = str(Path.cwd())
        
        self.init_ui()
        self.load_config_to_ui()
        
        # 自动加载上次保存的配置
        self.load_last_config()
        
        # 设置窗口属性
        self.setWindowTitle(tr("train_config"))
        self.setModal(True)
        self.resize(800, 700)
        
        # 连接控件修改信号
        self.connect_config_change_signals()
    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout(self)
        
        # 创建标签页
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # 基本设置标签页
        basic_tab = QWidget()
        tabs.addTab(basic_tab, tr("basic_settings"))
        self.create_basic_tab(basic_tab)
        
        # 训练参数标签页
        params_tab = QWidget()
        tabs.addTab(params_tab, tr("training_parameters"))
        self.create_params_tab(params_tab)
        
        # 优化器标签页
        optimizer_tab = QWidget()
        tabs.addTab(optimizer_tab, tr("optimizer"))
        self.create_optimizer_tab(optimizer_tab)
        
        # 数据增强标签页
        augment_tab = QWidget()
        tabs.addTab(augment_tab, tr("data_augmentation"))
        self.create_augment_tab(augment_tab)
        
        # 高级参数标签页
        advanced_tab = QWidget()
        tabs.addTab(advanced_tab, tr("advanced_params"))
        self.create_advanced_tab(advanced_tab)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 重置默认参数按钮
        self.btn_reset_default = QPushButton(tr("reset_default_params_btn"))
        self.btn_reset_default.clicked.connect(self.reset_to_defaults)
        self.btn_reset_default.setStyleSheet("background-color: #2196F3; color: white;")
        button_layout.addWidget(self.btn_reset_default)
        
        button_layout.addStretch()
        
        # 保存/加载配置按钮
        self.btn_save_config = QPushButton(tr("save_config_btn"))
        self.btn_save_config.clicked.connect(self.save_config)
        button_layout.addWidget(self.btn_save_config)
        
        self.btn_load_config = QPushButton(tr("load_config_btn"))
        self.btn_load_config.clicked.connect(self.load_config)
        button_layout.addWidget(self.btn_load_config)
        
        button_layout.addStretch()
        
        # 开始/取消按钮
        self.btn_start = QPushButton(tr("start_training_btn"))
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.btn_start)
        
        self.btn_cancel = QPushButton(tr("cancel_btn"))
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        main_layout.addLayout(button_layout)
