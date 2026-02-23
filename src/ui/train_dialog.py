"""
模型训练配置对话框
"""
import json
import os
import configparser
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFileDialog, QMessageBox, QTabWidget,
    QWidget, QTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from yolo_tool import YOLOTrainer


class TrainDialog(QDialog):
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
        
        self.init_ui()
        self.load_config_to_ui()
        
        # 自动加载上次保存的配置
        self.load_last_config()
        
        # 设置窗口属性
        self.setWindowTitle("YOLO模型训练配置")
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
        tabs.addTab(basic_tab, "基本设置")
        self.create_basic_tab(basic_tab)
        
        # 训练参数标签页
        params_tab = QWidget()
        tabs.addTab(params_tab, "训练参数")
        self.create_params_tab(params_tab)
        
        # 优化器标签页
        optimizer_tab = QWidget()
        tabs.addTab(optimizer_tab, "优化器")
        self.create_optimizer_tab(optimizer_tab)
        
        # 数据增强标签页
        augment_tab = QWidget()
        tabs.addTab(augment_tab, "数据增强")
        self.create_augment_tab(augment_tab)
        
        # 高级参数标签页
        advanced_tab = QWidget()
        tabs.addTab(advanced_tab, "高级参数")
        self.create_advanced_tab(advanced_tab)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 重置默认参数按钮
        self.btn_reset_default = QPushButton("重置默认参数")
        self.btn_reset_default.clicked.connect(self.reset_to_defaults)
        self.btn_reset_default.setStyleSheet("background-color: #2196F3; color: white;")
        button_layout.addWidget(self.btn_reset_default)
        
        button_layout.addStretch()
        
        # 保存/加载配置按钮
        self.btn_save_config = QPushButton("保存配置")
        self.btn_save_config.clicked.connect(self.save_config)
        button_layout.addWidget(self.btn_save_config)
        
        self.btn_load_config = QPushButton("加载配置")
        self.btn_load_config.clicked.connect(self.load_config)
        button_layout.addWidget(self.btn_load_config)
        
        button_layout.addStretch()
        
        # 开始/取消按钮
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.btn_start)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        main_layout.addLayout(button_layout)
    
    def create_basic_tab(self, parent: QWidget):
        """创建基本设置标签页"""
        layout = QVBoxLayout(parent)
        
        # 预训练模型
        model_group = QGroupBox("预训练模型")
        model_layout = QFormLayout(model_group)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择预训练模型文件 (.pt)")
        model_layout.addRow("模型路径:", self.model_path_edit)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addRow("", browse_btn)
        
        layout.addWidget(model_group)
        
        # 数据集配置
        data_group = QGroupBox("数据集配置")
        data_layout = QFormLayout(data_group)
        
        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setPlaceholderText("选择数据集配置文件 (.yaml)")
        data_layout.addRow("数据集YAML:", self.data_yaml_edit)
        
        browse_yaml_btn = QPushButton("浏览...")
        browse_yaml_btn.clicked.connect(self.browse_data_yaml)
        data_layout.addRow("", browse_yaml_btn)
        
        layout.addWidget(data_group)
        
        # 输出设置
        output_group = QGroupBox("输出设置")
        output_layout = QFormLayout(output_group)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择训练结果输出目录")
        output_layout.addRow("输出目录:", self.output_dir_edit)
        
        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        output_layout.addRow("", browse_output_btn)
        
        self.run_name_edit = QLineEdit("train")
        output_layout.addRow("运行名称:", self.run_name_edit)
        
        layout.addWidget(output_group)
        
        # 恢复训练
        resume_group = QGroupBox("恢复训练")
        resume_layout = QFormLayout(resume_group)
        
        self.resume_checkbox = QCheckBox("恢复训练")
        self.resume_checkbox.toggled.connect(self.on_resume_toggled)
        resume_layout.addRow(self.resume_checkbox)
        
        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setPlaceholderText("选择检查点文件 (.pt)")
        self.resume_path_edit.setEnabled(False)
        resume_layout.addRow("检查点:", self.resume_path_edit)
        
        browse_resume_btn = QPushButton("浏览...")
        browse_resume_btn.clicked.connect(self.browse_resume_file)
        browse_resume_btn.setEnabled(False)
        self.resume_browse_btn = browse_resume_btn
        resume_layout.addRow("", browse_resume_btn)
        
        layout.addWidget(resume_group)
        
        layout.addStretch()
    
    def create_params_tab(self, parent: QWidget):
        """创建训练参数标签页"""
        layout = QVBoxLayout(parent)
        
        # 训练参数
        train_group = QGroupBox("训练参数")
        train_layout = QFormLayout(train_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(300)
        train_layout.addRow("训练轮数 (epochs):", self.epochs_spin)
        
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        train_layout.addRow("输入图像尺寸:", self.imgsz_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        train_layout.addRow("批量大小 (batch):", self.batch_spin)
        
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(8)
        train_layout.addRow("数据加载线程数:", self.workers_spin)
        
        layout.addWidget(train_group)
        
        # 设备设置
        device_group = QGroupBox("设备设置")
        device_layout = QFormLayout(device_group)
        
        self.device_combo = QComboBox()
        
        # 动态检测可用设备
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    for i in range(gpu_count):
                        try:
                            gpu_name = torch.cuda.get_device_name(i)
                            self.device_combo.addItem(f"{i} ({gpu_name})")
                        except:
                            self.device_combo.addItem(f"{i} (GPU {i})")
                    # CPU选项
                    self.device_combo.addItem("-1 (CPU)")
                    # 默认选择第一个GPU
                    self.device_combo.setCurrentIndex(0)
                else:
                    self.device_combo.addItem("-1 (CPU)")
                    self.device_combo.setToolTip("未检测到CUDA设备，将使用CPU训练")
            else:
                self.device_combo.addItem("-1 (CPU)")
                self.device_combo.setToolTip("CUDA不可用，将使用CPU训练")
        except ImportError:
            # 如果torch未安装，添加默认选项
            self.device_combo.addItems(["0 (GPU)", "-1 (CPU)"])
            self.device_combo.setToolTip("PyTorch未安装，设备选择可能不准确")
        
        device_layout.addRow("训练设备:", self.device_combo)
        
        layout.addWidget(device_group)
        
        # 其他设置
        other_group = QGroupBox("其他设置")
        other_layout = QFormLayout(other_group)
        
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(40)
        other_layout.addRow("早停耐心值:", self.patience_spin)
        
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(40)
        other_layout.addRow("关闭马赛克增强轮数:", self.close_mosaic_spin)
        
        self.rect_checkbox = QCheckBox("矩形训练")
        other_layout.addRow(self.rect_checkbox)
        
        self.cache_checkbox = QCheckBox("缓存数据集")
        self.cache_checkbox.setChecked(True)
        other_layout.addRow(self.cache_checkbox)
        
        self.amp_checkbox = QCheckBox("自动混合精度训练")
        self.amp_checkbox.setChecked(True)
        other_layout.addRow(self.amp_checkbox)
        
        self.plots_checkbox = QCheckBox("生成可视化图表")
        self.plots_checkbox.setChecked(True)
        other_layout.addRow(self.plots_checkbox)
        
        self.verbose_checkbox = QCheckBox("详细日志输出")
        self.verbose_checkbox.setChecked(True)
        other_layout.addRow(self.verbose_checkbox)
        
        layout.addWidget(other_group)
        
        layout.addStretch()
    
    def create_optimizer_tab(self, parent: QWidget):
        """创建优化器标签页"""
        layout = QVBoxLayout(parent)
        
        optimizer_group = QGroupBox("优化器设置")
        optimizer_layout = QFormLayout(optimizer_group)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "SGD", "Adam", "RMSprop"])
        optimizer_layout.addRow("优化器:", self.optimizer_combo)
        
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(0.00001, 0.1)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setValue(0.01)
        self.lr0_spin.setDecimals(5)
        optimizer_layout.addRow("初始学习率 (lr0):", self.lr0_spin)
        
        self.cos_lr_checkbox = QCheckBox("使用余弦退火学习率调度")
        self.cos_lr_checkbox.setChecked(True)
        optimizer_layout.addRow(self.cos_lr_checkbox)
        
        layout.addWidget(optimizer_group)
        layout.addStretch()
    
    def create_augment_tab(self, parent: QWidget):
        """创建数据增强标签页"""
        layout = QVBoxLayout(parent)
        
        augment_group = QGroupBox("数据增强")
        augment_layout = QFormLayout(augment_group)
        
        self.augment_checkbox = QCheckBox("启用数据增强")
        self.augment_checkbox.setChecked(True)
        self.augment_checkbox.toggled.connect(self.on_augment_toggled)
        augment_layout.addRow(self.augment_checkbox)
        
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setSingleStep(0.1)
        self.mixup_spin.setValue(0.0)
        augment_layout.addRow("Mixup增强强度:", self.mixup_spin)
        
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setSingleStep(1.0)
        self.degrees_spin.setValue(0.0)
        augment_layout.addRow("旋转角度 (degrees):", self.degrees_spin)
        
        self.shear_spin = QDoubleSpinBox()
        self.shear_spin.setRange(0.0, 1.0)
        self.shear_spin.setSingleStep(0.1)
        self.shear_spin.setValue(0.0)
        augment_layout.addRow("剪切变换强度 (shear):", self.shear_spin)
        
        self.perspective_spin = QDoubleSpinBox()
        self.perspective_spin.setRange(0.0, 1.0)
        self.perspective_spin.setSingleStep(0.1)
        self.perspective_spin.setValue(0.0)
        augment_layout.addRow("透视变换强度:", self.perspective_spin)
        
        self.flipud_spin = QDoubleSpinBox()
        self.flipud_spin.setRange(0.0, 1.0)
        self.flipud_spin.setSingleStep(0.1)
        self.flipud_spin.setValue(0.0)
        augment_layout.addRow("上下翻转概率:", self.flipud_spin)
        
        # 添加HSV增强参数
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 0.5)
        self.hsv_h_spin.setSingleStep(0.01)
        self.hsv_h_spin.setValue(0.015)
        augment_layout.addRow("色调增强 (hsv_h):", self.hsv_h_spin)
        
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setSingleStep(0.1)
        self.hsv_s_spin.setValue(0.7)
        augment_layout.addRow("饱和度增强 (hsv_s):", self.hsv_s_spin)
        
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setSingleStep(0.1)
        self.hsv_v_spin.setValue(0.4)
        augment_layout.addRow("明度增强 (hsv_v):", self.hsv_v_spin)
        
        layout.addWidget(augment_group)
        layout.addStretch()
    
    def create_advanced_tab(self, parent: QWidget):
        """创建高级参数标签页"""
        layout = QVBoxLayout(parent)
        
        # 优化器高级参数
        optimizer_advanced_group = QGroupBox("优化器高级参数")
        optimizer_layout = QFormLayout(optimizer_advanced_group)
        
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.01)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setDecimals(4)
        optimizer_layout.addRow("权重衰减 (weight_decay):", self.weight_decay_spin)
        
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setSingleStep(0.01)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setDecimals(3)
        optimizer_layout.addRow("动量 (momentum):", self.momentum_spin)
        
        layout.addWidget(optimizer_advanced_group)
        
        # 学习率预热
        warmup_group = QGroupBox("学习率预热")
        warmup_layout = QFormLayout(warmup_group)
        
        self.warmup_epochs_spin = QDoubleSpinBox()
        self.warmup_epochs_spin.setRange(0.0, 10.0)
        self.warmup_epochs_spin.setSingleStep(0.5)
        self.warmup_epochs_spin.setValue(3.0)
        warmup_layout.addRow("预热轮数 (warmup_epochs):", self.warmup_epochs_spin)
        
        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_momentum_spin.setRange(0.0, 1.0)
        self.warmup_momentum_spin.setSingleStep(0.1)
        self.warmup_momentum_spin.setValue(0.8)
        warmup_layout.addRow("预热动量 (warmup_momentum):", self.warmup_momentum_spin)
        
        self.warmup_bias_lr_spin = QDoubleSpinBox()
        self.warmup_bias_lr_spin.setRange(0.0, 0.5)
        self.warmup_bias_lr_spin.setSingleStep(0.05)
        self.warmup_bias_lr_spin.setValue(0.1)
        warmup_layout.addRow("预热偏置学习率 (warmup_bias_lr):", self.warmup_bias_lr_spin)
        
        layout.addWidget(warmup_group)
        
        # 损失权重
        loss_weights_group = QGroupBox("损失权重")
        loss_layout = QFormLayout(loss_weights_group)
        
        self.box_weight_spin = QDoubleSpinBox()
        self.box_weight_spin.setRange(0.0, 20.0)
        self.box_weight_spin.setSingleStep(0.5)
        self.box_weight_spin.setValue(7.5)
        loss_layout.addRow("边框损失权重 (box):", self.box_weight_spin)
        
        self.cls_weight_spin = QDoubleSpinBox()
        self.cls_weight_spin.setRange(0.0, 5.0)
        self.cls_weight_spin.setSingleStep(0.1)
        self.cls_weight_spin.setValue(0.5)
        loss_layout.addRow("分类损失权重 (cls):", self.cls_weight_spin)
        
        self.dfl_weight_spin = QDoubleSpinBox()
        self.dfl_weight_spin.setRange(0.0, 5.0)
        self.dfl_weight_spin.setSingleStep(0.1)
        self.dfl_weight_spin.setValue(1.5)
        loss_layout.addRow("分布焦点损失权重 (dfl):", self.dfl_weight_spin)
        
        layout.addWidget(loss_weights_group)
        
        # 正则化和其他
        other_advanced_group = QGroupBox("正则化和其他")
        other_layout = QFormLayout(other_advanced_group)
        
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.2)
        self.label_smoothing_spin.setSingleStep(0.01)
        self.label_smoothing_spin.setValue(0.0)
        other_layout.addRow("标签平滑 (label_smoothing):", self.label_smoothing_spin)
        
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.0)
        other_layout.addRow("Dropout率 (dropout):", self.dropout_spin)
        
        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setSingleStep(0.1)
        self.fliplr_spin.setValue(0.5)
        other_layout.addRow("左右翻转概率 (fliplr):", self.fliplr_spin)
        
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setSingleStep(0.1)
        self.mosaic_spin.setValue(1.0)
        other_layout.addRow("马赛克增强概率 (mosaic):", self.mosaic_spin)
        
        self.copy_paste_spin = QDoubleSpinBox()
        self.copy_paste_spin.setRange(0.0, 1.0)
        self.copy_paste_spin.setSingleStep(0.1)
        self.copy_paste_spin.setValue(0.0)
        other_layout.addRow("复制粘贴增强概率 (copy_paste):", self.copy_paste_spin)
        
        layout.addWidget(other_advanced_group)
        
        layout.addStretch()
    
    def on_resume_toggled(self, checked: bool):
        """恢复训练复选框状态改变"""
        self.resume_path_edit.setEnabled(checked)
        self.resume_browse_btn.setEnabled(checked)
    
    def on_augment_toggled(self, checked: bool):
        """数据增强复选框状态改变"""
        # 启用/禁用所有增强参数控件
        for widget in [self.mixup_spin, self.degrees_spin, 
                       self.shear_spin, self.perspective_spin, 
                       self.flipud_spin]:
            widget.setEnabled(checked)
    
    def browse_model_file(self):
        """浏览模型文件"""
        current_dir = Path.cwd()
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择预训练模型文件",
            str(current_dir),
            "PyTorch模型文件 (*.pt)"
        )
        
        if model_path:
            self.model_path_edit.setText(model_path)
    
    def browse_data_yaml(self):
        """浏览数据集配置文件"""
        current_dir = Path.cwd()
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集配置文件",
            str(current_dir),
            "YAML文件 (*.yaml *.yml)"
        )
        
        if yaml_path:
            self.data_yaml_edit.setText(yaml_path)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        current_dir = Path.cwd()
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择训练结果输出目录",
            str(current_dir)
        )
        
        if output_dir:
            self.output_dir_edit.setText(output_dir)
    
    def browse_resume_file(self):
        """浏览检查点文件"""
        current_dir = Path.cwd()
        resume_path, _ = QFileDialog.getOpenFileName(
            self, "选择检查点文件",
            str(current_dir),
            "PyTorch模型文件 (*.pt)"
        )
        
        if resume_path:
            self.resume_path_edit.setText(resume_path)
    
    def save_config(self):
        """保存训练配置到文件"""
        current_dir = Path.cwd()
        config_path, _ = QFileDialog.getSaveFileName(
            self, "保存训练配置",
            str(current_dir),
            "JSON文件 (*.json)"
        )
        
        if config_path:
            # 确保文件扩展名
            if not config_path.lower().endswith('.json'):
                config_path += '.json'
            
            # 收集当前UI配置
            config = self.collect_config_from_ui()
            
            try:
                # 保存到文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "成功", f"配置已保存到: {config_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
    
    def load_config(self):
        """从文件加载训练配置"""
        current_dir = Path.cwd()
        config_path, _ = QFileDialog.getOpenFileName(
            self, "加载训练配置",
            str(current_dir),
            "JSON文件 (*.json)"
        )
        
        if config_path:
            try:
                # 从文件加载
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新配置并加载到UI
                self.config.update(config)
                self.load_config_to_ui()
                
                QMessageBox.information(self, "成功", f"配置已加载: {config_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
    
    def collect_config_from_ui(self) -> Dict[str, Any]:
        """从UI收集配置"""
        config = {}
        
        # 基本设置
        config['model_path'] = self.model_path_edit.text()
        config['data_yaml'] = self.data_yaml_edit.text()
        config['output_dir'] = self.output_dir_edit.text()
        config['run_name'] = self.run_name_edit.text()
        config['resume'] = self.resume_checkbox.isChecked()
        if config['resume']:
            config['resume'] = self.resume_path_edit.text()
        
        # 训练参数
        config['epochs'] = self.epochs_spin.value()
        config['imgsz'] = self.imgsz_spin.value()
        config['batch'] = self.batch_spin.value()
        config['workers'] = self.workers_spin.value()
        
        # 设备设置
        device_text = self.device_combo.currentText()
        device_value = device_text.split()[0]
        try:
            config['device'] = int(device_value)
        except ValueError:
            config['device'] = 0
        
        # 其他设置
        config['patience'] = self.patience_spin.value()
        config['close_mosaic'] = self.close_mosaic_spin.value()
        config['rect'] = self.rect_checkbox.isChecked()
        config['cache'] = self.cache_checkbox.isChecked()
        config['amp'] = self.amp_checkbox.isChecked()
        config['plots'] = self.plots_checkbox.isChecked()
        config['verbose'] = self.verbose_checkbox.isChecked()
        
        # 优化器设置
        config['optimizer'] = self.optimizer_combo.currentText()
        config['lr0'] = self.lr0_spin.value()
        config['cos_lr'] = self.cos_lr_checkbox.isChecked()
        
        # 数据增强
        config['augment'] = self.augment_checkbox.isChecked()
        config['mixup'] = self.mixup_spin.value()
        config['degrees'] = self.degrees_spin.value()
        config['shear'] = self.shear_spin.value()
        config['perspective'] = self.perspective_spin.value()
        config['flipud'] = self.flipud_spin.value()
        config['hsv_h'] = self.hsv_h_spin.value()
        config['hsv_s'] = self.hsv_s_spin.value()
        config['hsv_v'] = self.hsv_v_spin.value()
        
        # 高级参数
        config['weight_decay'] = self.weight_decay_spin.value()
        config['momentum'] = self.momentum_spin.value()
        config['warmup_epochs'] = self.warmup_epochs_spin.value()
        config['warmup_momentum'] = self.warmup_momentum_spin.value()
        config['warmup_bias_lr'] = self.warmup_bias_lr_spin.value()
        config['box'] = self.box_weight_spin.value()
        config['cls'] = self.cls_weight_spin.value()
        config['dfl'] = self.dfl_weight_spin.value()
        config['label_smoothing'] = self.label_smoothing_spin.value()
        config['dropout'] = self.dropout_spin.value()
        config['fliplr'] = self.fliplr_spin.value()
        config['mosaic'] = self.mosaic_spin.value()
        config['copy_paste'] = self.copy_paste_spin.value()
        
        return config
    
    def load_config_to_ui(self):
        """将配置加载到UI"""
        # 基本设置
        self.model_path_edit.setText(self.config.get('model_path', ''))
        self.data_yaml_edit.setText(self.config.get('data_yaml', ''))
        self.output_dir_edit.setText(self.config.get('output_dir', ''))
        self.run_name_edit.setText(self.config.get('run_name', 'train'))
        
        # 恢复训练
        resume = self.config.get('resume', False)
        self.resume_checkbox.setChecked(bool(resume))
        if isinstance(resume, str) and resume:
            self.resume_path_edit.setText(resume)
        
        # 训练参数
        self.epochs_spin.setValue(self.config.get('epochs', 300))
        self.imgsz_spin.setValue(self.config.get('imgsz', 640))
        self.batch_spin.setValue(self.config.get('batch', 16))
        self.workers_spin.setValue(self.config.get('workers', 8))
        
        # 设备设置
        device = self.config.get('device', 0)
        
        # 查找对应的设备选项
        device_found = False
        if self.device_combo.count() > 0:
            # 尝试找到匹配的设备索引
            for i in range(self.device_combo.count()):
                item_text = self.device_combo.itemText(i)
                # 从文本中提取设备号
                if item_text.startswith(f"{device} ("):
                    self.device_combo.setCurrentIndex(i)
                    device_found = True
                    break
                # 特殊处理CPU
                elif device == -1 and "CPU" in item_text:
                    self.device_combo.setCurrentIndex(i)
                    device_found = True
                    break
            
            # 如果没找到，使用第一个选项
            if not device_found and self.device_combo.count() > 0:
                self.device_combo.setCurrentIndex(0)
        
        # 其他设置
        self.patience_spin.setValue(self.config.get('patience', 40))
        self.close_mosaic_spin.setValue(self.config.get('close_mosaic', 40))
        self.rect_checkbox.setChecked(self.config.get('rect', False))
        self.cache_checkbox.setChecked(self.config.get('cache', True))
        self.amp_checkbox.setChecked(self.config.get('amp', True))
        self.plots_checkbox.setChecked(self.config.get('plots', True))
        self.verbose_checkbox.setChecked(self.config.get('verbose', True))
        
        # 优化器设置
        optimizer = self.config.get('optimizer', 'AdamW')
        index = self.optimizer_combo.findText(optimizer)
        if index >= 0:
            self.optimizer_combo.setCurrentIndex(index)
        
        self.lr0_spin.setValue(self.config.get('lr0', 0.0008))
        self.cos_lr_checkbox.setChecked(self.config.get('cos_lr', True))
        
        # 数据增强
        augment = self.config.get('augment', True)
        self.augment_checkbox.setChecked(augment)
        self.mixup_spin.setValue(self.config.get('mixup', 0.0))
        self.degrees_spin.setValue(self.config.get('degrees', 0.0))
        self.shear_spin.setValue(self.config.get('shear', 0.0))
        self.perspective_spin.setValue(self.config.get('perspective', 0.0))
        self.flipud_spin.setValue(self.config.get('flipud', 0.0))
        self.hsv_h_spin.setValue(self.config.get('hsv_h', 0.015))
        self.hsv_s_spin.setValue(self.config.get('hsv_s', 0.7))
        self.hsv_v_spin.setValue(self.config.get('hsv_v', 0.4))
        
        # 启用/禁用增强参数控件
        self.on_augment_toggled(augment)
        
        # 高级参数
        self.weight_decay_spin.setValue(self.config.get('weight_decay', 0.0005))
        self.momentum_spin.setValue(self.config.get('momentum', 0.937))
        self.warmup_epochs_spin.setValue(self.config.get('warmup_epochs', 3.0))
        self.warmup_momentum_spin.setValue(self.config.get('warmup_momentum', 0.8))
        self.warmup_bias_lr_spin.setValue(self.config.get('warmup_bias_lr', 0.1))
        self.box_weight_spin.setValue(self.config.get('box', 7.5))
        self.cls_weight_spin.setValue(self.config.get('cls', 0.5))
        self.dfl_weight_spin.setValue(self.config.get('dfl', 1.5))
        self.label_smoothing_spin.setValue(self.config.get('label_smoothing', 0.0))
        self.dropout_spin.setValue(self.config.get('dropout', 0.0))
        self.fliplr_spin.setValue(self.config.get('fliplr', 0.5))
        self.mosaic_spin.setValue(self.config.get('mosaic', 1.0))
        self.copy_paste_spin.setValue(self.config.get('copy_paste', 0.0))
    
    def validate_config(self) -> bool:
        """验证配置"""
        config = self.collect_config_from_ui()
        
        # 检查必要参数
        if not config['model_path']:
            QMessageBox.warning(self, "警告", "请选择预训练模型文件")
            self.model_path_edit.setFocus()
            return False
        
        if not config['data_yaml']:
            QMessageBox.warning(self, "警告", "请选择数据集配置文件")
            self.data_yaml_edit.setFocus()
            return False
        
        # 检查文件是否存在
        if not Path(config['model_path']).exists():
            QMessageBox.warning(self, "警告", f"模型文件不存在: {config['model_path']}")
            self.model_path_edit.setFocus()
            return False
        
        if not Path(config['data_yaml']).exists():
            QMessageBox.warning(self, "警告", f"数据集配置文件不存在: {config['data_yaml']}")
            self.data_yaml_edit.setFocus()
            return False
        
        # 如果启用了恢复训练，检查检查点文件
        if config['resume'] and isinstance(config['resume'], str):
            if not Path(config['resume']).exists():
                QMessageBox.warning(self, "警告", f"检查点文件不存在: {config['resume']}")
                self.resume_path_edit.setFocus()
                return False
        
        return True
    
    def load_last_config(self):
        """加载上次保存的训练配置"""
        try:
            if self.config_file_path.exists():
                self.config_parser.read(self.config_file_path)
                
                # 检查是否启用自动保存配置
                auto_save = self.config_parser.getboolean('training', 'auto_save_config', fallback=True)
                if not auto_save:
                    return
                
                # 尝试从INI文件加载JSON配置
                if self.config_parser.has_option('training', 'last_config'):
                    config_json = self.config_parser.get('training', 'last_config')
                    if config_json:
                        last_config = json.loads(config_json)
                        # 更新当前配置
                        self.config.update(last_config)
                        # 加载到UI
                        self.load_config_to_ui()
                        self.log_message("已加载上次保存的配置")
                elif self.config_parser.has_option('training', 'last_config_path'):
                    # 从文件路径加载配置
                    config_path = self.config_parser.get('training', 'last_config_path')
                    if config_path and Path(config_path).exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            last_config = json.load(f)
                            self.config.update(last_config)
                            self.load_config_to_ui()
                            self.log_message(f"已从文件加载上次配置: {config_path}")
        except Exception as e:
            # 加载失败时不中断程序
            print(f"加载上次配置失败: {e}")
    
    def save_last_config(self):
        """保存当前配置到配置文件"""
        try:
            # 收集当前配置
            current_config = self.collect_config_from_ui()
            
            # 转换为JSON字符串
            config_json = json.dumps(current_config, ensure_ascii=False)
            
            # 更新配置解析器
            if not self.config_parser.has_section('training'):
                self.config_parser.add_section('training')
            
            self.config_parser.set('training', 'last_config', config_json)
            self.config_parser.set('training', 'auto_save_config', 'true')
            
            # 保存到文件
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                self.config_parser.write(f)
            
            self.log_message("配置已自动保存")
        except Exception as e:
            # 保存失败时不中断程序
            print(f"自动保存配置失败: {e}")
    
    def connect_config_change_signals(self):
        """连接所有控件的修改信号，以便在配置发生变化时自动保存"""
        # 文本输入框
        text_edits = [
            self.model_path_edit,
            self.data_yaml_edit,
            self.output_dir_edit,
            self.run_name_edit,
            self.resume_path_edit
        ]
        
        for text_edit in text_edits:
            text_edit.textChanged.connect(self.on_config_changed)
        
        # 数值输入框
        spin_boxes = [
            self.epochs_spin,
            self.imgsz_spin,
            self.batch_spin,
            self.workers_spin,
            self.patience_spin,
            self.close_mosaic_spin,
            self.mixup_spin,
            self.degrees_spin,
            self.shear_spin,
            self.perspective_spin,
            self.flipud_spin,
            self.lr0_spin,
            # 新增的数值控件
            self.hsv_h_spin,
            self.hsv_s_spin,
            self.hsv_v_spin,
            self.weight_decay_spin,
            self.momentum_spin,
            self.warmup_epochs_spin,
            self.warmup_momentum_spin,
            self.warmup_bias_lr_spin,
            self.box_weight_spin,
            self.cls_weight_spin,
            self.dfl_weight_spin,
            self.label_smoothing_spin,
            self.dropout_spin,
            self.fliplr_spin,
            self.mosaic_spin,
            self.copy_paste_spin
        ]
        
        for spin_box in spin_boxes:
            if hasattr(spin_box, 'valueChanged'):
                spin_box.valueChanged.connect(self.on_config_changed)
            elif hasattr(spin_box, 'textChanged'):
                spin_box.textChanged.connect(self.on_config_changed)
        
        # 组合框
        combo_boxes = [
            self.device_combo,
            self.optimizer_combo
        ]
        
        for combo_box in combo_boxes:
            combo_box.currentIndexChanged.connect(self.on_config_changed)
        
        # 复选框
        check_boxes = [
            self.resume_checkbox,
            self.rect_checkbox,
            self.cache_checkbox,
            self.amp_checkbox,
            self.plots_checkbox,
            self.verbose_checkbox,
            self.cos_lr_checkbox,
            self.augment_checkbox
        ]
        
        for check_box in check_boxes:
            check_box.toggled.connect(self.on_config_changed)
        
        # 浏览按钮不连接，因为它们会触发文本输入框的textChanged信号
    
    def on_config_changed(self, *args):
        """配置发生变化时的处理"""
        self.config_modified = True
        
        # 如果配置被修改，可以在这里添加实时保存逻辑
        # 注意：实时保存可能会影响性能，所以只在关键操作时保存
        pass
    
    def save_config_on_exit(self):
        """退出时自动保存配置"""
        if self.config_modified:
            self.save_last_config()
    
    def closeEvent(self, event):
        """关闭事件处理"""
        self.save_config_on_exit()
        event.accept()
    
    def reject(self):
        """取消按钮处理"""
        self.save_config_on_exit()
        super().reject()
    
    def accept(self):
        """接受按钮处理（开始训练）"""
        self.save_config_on_exit()
        super().accept()
    
    def log_message(self, message: str):
        """记录消息到控制台"""
        print(f"[TrainDialog] {message}")
    
    def reset_to_defaults(self):
        """重置所有参数到市场最优默认值"""
        reply = QMessageBox.question(
            self,
            "重置默认参数",
            "确定要重置所有参数到市场最优默认值吗？\n当前的自定义配置将被覆盖。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 使用训练器的默认配置（市场最优参数）
            default_config = self.trainer.get_default_config()
            
            # 保留文件路径信息（如果已设置）
            if self.model_path_edit.text():
                default_config['model_path'] = self.model_path_edit.text()
            if self.data_yaml_edit.text():
                default_config['data_yaml'] = self.data_yaml_edit.text()
            if self.output_dir_edit.text():
                default_config['output_dir'] = self.output_dir_edit.text()
            
            # 更新配置
            self.config.update(default_config)
            
            # 重新加载到UI
            self.load_config_to_ui()
            
            # 标记配置已修改
            self.config_modified = True
            
            QMessageBox.information(
                self,
                "重置成功",
                "所有参数已重置到市场最优默认值\n（文件路径信息已保留）"
            )
    
    def start_training(self):
        """开始训练"""
        if not self.validate_config():
            return
        
        # 收集配置
        config = self.collect_config_from_ui()
        
        # 自动保存配置
        self.save_last_config()
        
        # 设置训练器
        if not self.trainer.setup(config):
            QMessageBox.critical(self, "错误", "训练参数设置失败")
            return
        
        # 创建训练进度对话框
        from .train_progress_dialog import TrainProgressDialog
        progress_dialog = TrainProgressDialog(self.trainer, parent=self)
        
        # 连接训练完成信号
        def on_training_finished(success: bool, message: str):
            self.btn_start.setEnabled(True)
            if success:
                QMessageBox.information(self, "训练完成", message)
            else:
                QMessageBox.critical(self, "训练失败", message)
        
        self.trainer.training_finished.connect(on_training_finished)
        
        # 禁用开始按钮
        self.btn_start.setEnabled(False)
        
        # 显示进度对话框并开始训练
        if progress_dialog.start_training():
            self.accept()  # 关闭配置对话框
        else:
            self.btn_start.setEnabled(True)
