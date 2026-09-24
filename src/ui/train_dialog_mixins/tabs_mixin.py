"""
标签页构建 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.utils.i18n import tr


class TrainTabsMixin:
    """标签页构建 Mixin：五个参数标签页的控件创建"""


    def create_basic_tab(self, parent: QWidget):
        """创建基本设置标签页"""
        layout = QVBoxLayout(parent)
        
        # 预训练模型
        model_group = QGroupBox(tr("pretrained_model"))
        model_layout = QFormLayout(model_group)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText(tr("select_pretrained_model_file"))
        model_layout.addRow(tr("model_path_label"), self.model_path_edit)
        
        browse_btn = QPushButton(tr("browse_button_label"))
        browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addRow("", browse_btn)
        
        layout.addWidget(model_group)
        
        # 数据集配置
        data_group = QGroupBox(tr("dataset_config"))
        data_layout = QFormLayout(data_group)
        
        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setPlaceholderText(tr("select_dataset_config_file"))
        data_layout.addRow(tr("data_yaml_label"), self.data_yaml_edit)
        
        browse_yaml_btn = QPushButton(tr("browse_button_label"))
        browse_yaml_btn.clicked.connect(self.browse_data_yaml)
        data_layout.addRow("", browse_yaml_btn)
        
        layout.addWidget(data_group)
        
        # 输出设置
        output_group = QGroupBox(tr("output_settings"))
        output_layout = QFormLayout(output_group)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(tr("select_output_dir"))
        output_layout.addRow(tr("output_dir_label"), self.output_dir_edit)
        
        browse_output_btn = QPushButton(tr("browse_button_label"))
        browse_output_btn.clicked.connect(self.browse_output_dir)
        output_layout.addRow("", browse_output_btn)
        
        self.run_name_edit = QLineEdit("train")
        output_layout.addRow(tr("running_name_label"), self.run_name_edit)
        
        layout.addWidget(output_group)
        
        # 恢复训练
        resume_group = QGroupBox(tr("resume_training"))
        resume_layout = QFormLayout(resume_group)
        
        self.resume_checkbox = QCheckBox(tr("resume_training_checkbox"))
        self.resume_checkbox.toggled.connect(self.on_resume_toggled)
        resume_layout.addRow(self.resume_checkbox)
        
        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setPlaceholderText(tr("select_checkpoint_file"))
        self.resume_path_edit.setEnabled(False)
        resume_layout.addRow(tr("resume_checkpoint_label"), self.resume_path_edit)
        
        browse_resume_btn = QPushButton(tr("browse_button_label"))
        browse_resume_btn.clicked.connect(self.browse_resume_file)
        browse_resume_btn.setEnabled(False)
        self.resume_browse_btn = browse_resume_btn
        resume_layout.addRow("", browse_resume_btn)
        
        layout.addWidget(resume_group)

        # 增量训练
        incremental_group = QGroupBox(tr("incremental_training", "增量训练"))
        incremental_layout = QFormLayout(incremental_group)

        self.incremental_checkbox = QCheckBox(tr("incremental_training_checkbox", "启用增量训练"))
        self.incremental_checkbox.toggled.connect(self.on_incremental_toggled)
        incremental_layout.addRow(self.incremental_checkbox)

        self.incremental_path_edit = QLineEdit()
        self.incremental_path_edit.setPlaceholderText(tr("select_incremental_checkpoint", "选择已训练的权重文件 (.pt)"))
        self.incremental_path_edit.setEnabled(False)
        incremental_layout.addRow(tr("incremental_checkpoint_label", "权重文件:"), self.incremental_path_edit)

        browse_incremental_btn = QPushButton(tr("browse_button_label"))
        browse_incremental_btn.clicked.connect(self.browse_incremental_file)
        browse_incremental_btn.setEnabled(False)
        self.incremental_browse_btn = browse_incremental_btn
        incremental_layout.addRow("", browse_incremental_btn)

        self.incremental_hint = QLabel(tr("incremental_hint", "使用已训练的模型权重在新数据集上继续学习，保留旧知识的同时学习新数据"))
        self.incremental_hint.setWordWrap(True)
        self.incremental_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.incremental_hint.setVisible(False)
        incremental_layout.addRow(self.incremental_hint)

        layout.addWidget(incremental_group)

        layout.addStretch()

    def create_params_tab(self, parent: QWidget):
        """创建训练参数标签页"""
        layout = QVBoxLayout(parent)
        
        # 训练参数
        train_group = QGroupBox(tr("training_parameters"))
        train_layout = QFormLayout(train_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(300)
        train_layout.addRow(tr("epochs_label"), self.epochs_spin)
        
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        train_layout.addRow(tr("imgsz_label"), self.imgsz_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(4)
        train_layout.addRow(tr("batch_label"), self.batch_spin)
        
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(4)
        train_layout.addRow(tr("workers_label"), self.workers_spin)
        
        layout.addWidget(train_group)
        
        # 设备设置
        device_group = QGroupBox(tr("device_settings"))
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
                        except Exception:
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
        
        device_layout.addRow(tr("device_label"), self.device_combo)
        
        layout.addWidget(device_group)
        
        # 其他设置
        other_group = QGroupBox(tr("other_settings"))
        other_layout = QFormLayout(other_group)
        
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(40)
        other_layout.addRow(tr("patience_label"), self.patience_spin)
        
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(40)
        other_layout.addRow(tr("close_mosaic_label"), self.close_mosaic_spin)
        
        self.rect_checkbox = QCheckBox(tr("rectangular_training"))
        other_layout.addRow(self.rect_checkbox)
        
        self.cache_checkbox = QCheckBox(tr("cache_dataset"))
        self.cache_checkbox.setChecked(True)
        other_layout.addRow(self.cache_checkbox)
        
        self.amp_checkbox = QCheckBox(tr("amp_training"))
        self.amp_checkbox.setChecked(True)
        other_layout.addRow(self.amp_checkbox)
        
        self.plots_checkbox = QCheckBox(tr("generate_plots"))
        self.plots_checkbox.setChecked(True)
        other_layout.addRow(self.plots_checkbox)
        
        self.verbose_checkbox = QCheckBox(tr("verbose_logging"))
        self.verbose_checkbox.setChecked(True)
        other_layout.addRow(self.verbose_checkbox)
        
        layout.addWidget(other_group)
        
        layout.addStretch()

    def create_optimizer_tab(self, parent: QWidget):
        """创建优化器标签页"""
        layout = QVBoxLayout(parent)
        
        optimizer_group = QGroupBox(tr("optimizer_settings"))
        optimizer_layout = QFormLayout(optimizer_group)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "SGD", "Adam", "RMSprop"])
        optimizer_layout.addRow(tr("optimizer"), self.optimizer_combo)
        
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(0.00001, 0.1)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setValue(0.01)
        self.lr0_spin.setDecimals(5)
        optimizer_layout.addRow(tr("initial_learning_rate"), self.lr0_spin)

        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setRange(0.0, 1.0)
        self.lrf_spin.setSingleStep(0.01)
        self.lrf_spin.setValue(0.01)
        self.lrf_spin.setDecimals(4)
        optimizer_layout.addRow(tr("final_lr_factor"), self.lrf_spin)

        self.cos_lr_checkbox = QCheckBox(tr("use_cosine_lr"))
        self.cos_lr_checkbox.setChecked(True)
        optimizer_layout.addRow(self.cos_lr_checkbox)
        
        layout.addWidget(optimizer_group)
        layout.addStretch()

    def create_augment_tab(self, parent: QWidget):
        """创建数据增强标签页"""
        layout = QVBoxLayout(parent)
        
        augment_group = QGroupBox(tr("data_augmentation_settings"))
        augment_layout = QFormLayout(augment_group)
        
        self.augment_checkbox = QCheckBox(tr("enable_augmentation"))
        self.augment_checkbox.setChecked(True)
        self.augment_checkbox.toggled.connect(self.on_augment_toggled)
        augment_layout.addRow(self.augment_checkbox)
        
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setSingleStep(0.1)
        self.mixup_spin.setValue(0.0)
        augment_layout.addRow(tr("mixup_augmentation_strength"), self.mixup_spin)
        
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setSingleStep(1.0)
        self.degrees_spin.setValue(0.0)
        augment_layout.addRow(tr("rotation_degrees_label"), self.degrees_spin)
        
        self.shear_spin = QDoubleSpinBox()
        self.shear_spin.setRange(0.0, 1.0)
        self.shear_spin.setSingleStep(0.1)
        self.shear_spin.setValue(0.0)
        augment_layout.addRow(tr("shear_transformation_strength"), self.shear_spin)
        
        self.perspective_spin = QDoubleSpinBox()
        self.perspective_spin.setRange(0.0, 1.0)
        self.perspective_spin.setSingleStep(0.1)
        self.perspective_spin.setValue(0.0)
        augment_layout.addRow(tr("perspective_transformation_strength_label"), self.perspective_spin)
        
        self.flip_up_down_spin = QDoubleSpinBox()
        self.flip_up_down_spin.setRange(0.0, 1.0)
        self.flip_up_down_spin.setSingleStep(0.1)
        self.flip_up_down_spin.setValue(0.0)
        augment_layout.addRow(tr("flip_up_down_probability_label"), self.flip_up_down_spin)
        
        # 添加HSV增强参数
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 0.5)
        self.hsv_h_spin.setSingleStep(0.01)
        self.hsv_h_spin.setValue(0.015)
        augment_layout.addRow(tr("hue_augmentation"), self.hsv_h_spin)
        
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setSingleStep(0.1)
        self.hsv_s_spin.setValue(0.7)
        augment_layout.addRow(tr("saturation_augmentation"), self.hsv_s_spin)
        
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setSingleStep(0.1)
        self.hsv_v_spin.setValue(0.4)
        augment_layout.addRow(tr("value_augmentation"), self.hsv_v_spin)
        
        layout.addWidget(augment_group)
        layout.addStretch()

    def create_advanced_tab(self, parent: QWidget):
        """创建高级参数标签页"""
        layout = QVBoxLayout(parent)
        
        # 优化器高级参数
        optimizer_advanced_group = QGroupBox(tr("optimizer_advanced_params"))
        optimizer_layout = QFormLayout(optimizer_advanced_group)
        
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.01)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setDecimals(4)
        optimizer_layout.addRow(tr("weight_decay"), self.weight_decay_spin)
        
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setSingleStep(0.01)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setDecimals(3)
        optimizer_layout.addRow(tr("momentum"), self.momentum_spin)
        
        layout.addWidget(optimizer_advanced_group)
        
        # 学习率预热
        warmup_group = QGroupBox(tr("learning_rate_warmup"))
        warmup_layout = QFormLayout(warmup_group)
        
        self.warmup_epochs_spin = QDoubleSpinBox()
        self.warmup_epochs_spin.setRange(0.0, 10.0)
        self.warmup_epochs_spin.setSingleStep(0.5)
        self.warmup_epochs_spin.setValue(3.0)
        warmup_layout.addRow(tr("warmup_epochs"), self.warmup_epochs_spin)
        
        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_momentum_spin.setRange(0.0, 1.0)
        self.warmup_momentum_spin.setSingleStep(0.1)
        self.warmup_momentum_spin.setValue(0.8)
        warmup_layout.addRow(tr("warmup_momentum"), self.warmup_momentum_spin)
        
        self.warmup_bias_lr_spin = QDoubleSpinBox()
        self.warmup_bias_lr_spin.setRange(0.0, 0.5)
        self.warmup_bias_lr_spin.setSingleStep(0.05)
        self.warmup_bias_lr_spin.setValue(0.1)
        warmup_layout.addRow(tr("warmup_bias_lr"), self.warmup_bias_lr_spin)
        
        layout.addWidget(warmup_group)
        
        # 损失权重
        loss_weights_group = QGroupBox(tr("loss_weights"))
        loss_layout = QFormLayout(loss_weights_group)
        
        self.box_weight_spin = QDoubleSpinBox()
        self.box_weight_spin.setRange(0.0, 20.0)
        self.box_weight_spin.setSingleStep(0.5)
        self.box_weight_spin.setValue(7.5)
        loss_layout.addRow(tr("box_loss_weight"), self.box_weight_spin)
        
        self.cls_weight_spin = QDoubleSpinBox()
        self.cls_weight_spin.setRange(0.0, 5.0)
        self.cls_weight_spin.setSingleStep(0.1)
        self.cls_weight_spin.setValue(0.5)
        loss_layout.addRow(tr("cls_loss_weight"), self.cls_weight_spin)
        
        self.dfl_weight_spin = QDoubleSpinBox()
        self.dfl_weight_spin.setRange(0.0, 5.0)
        self.dfl_weight_spin.setSingleStep(0.1)
        self.dfl_weight_spin.setValue(1.5)
        loss_layout.addRow(tr("dfl_loss_weight"), self.dfl_weight_spin)
        
        layout.addWidget(loss_weights_group)
        
        # 正则化和其他
        other_advanced_group = QGroupBox(tr("regularization_and_other_section"))
        other_layout = QFormLayout(other_advanced_group)
        
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 1.0)
        self.label_smoothing_spin.setSingleStep(0.1)
        self.label_smoothing_spin.setValue(0.4)
        other_layout.addRow(tr("erasing"), self.label_smoothing_spin)
        
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.0)
        other_layout.addRow(tr("dropout_rate"), self.dropout_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(0)
        other_layout.addRow(tr("random_seed"), self.seed_spin)

        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setSingleStep(0.1)
        self.fliplr_spin.setValue(0.5)
        other_layout.addRow(tr("flip_lr_probability"), self.fliplr_spin)
        
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setSingleStep(0.1)
        self.mosaic_spin.setValue(1.0)
        other_layout.addRow(tr("mosaic_probability"), self.mosaic_spin)
        
        self.copy_paste_spin = QDoubleSpinBox()
        self.copy_paste_spin.setRange(0.0, 1.0)
        self.copy_paste_spin.setSingleStep(0.1)
        self.copy_paste_spin.setValue(0.0)
        other_layout.addRow(tr("copy_paste_probability"), self.copy_paste_spin)
        
        layout.addWidget(other_advanced_group)
        
        layout.addStretch()
