"""
配置收集与持久化 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

import json
from pathlib import Path
from typing import Any, Dict

from PySide6.QtWidgets import QFileDialog, QMessageBox

from src.utils.i18n import tr
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)


class TrainConfigMixin:
    """配置读写 Mixin：UI 与配置字典互转、校验、INI 自动保存与信号连接"""


    def save_config(self):
        """保存训练配置到文件"""
        config_path, _ = QFileDialog.getSaveFileName(
            self, tr("save_config_dialog_title"),
            self._last_browse_path,
            "JSON文件 (*.json)"
        )
        
        if config_path:
            # 保存配置
            if not config_path.lower().endswith('.json'):
                config_path += '.json'
            self._last_browse_path = str(Path(config_path).parent)

            # 收集当前UI配置
            config = self.collect_config_from_ui()
            
            try:
                # 保存到文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, tr("success"), tr("config_saved_successfully") + f" {config_path}")
            except Exception as e:
                logger.exception(f"保存配置失败: {e}")
                QMessageBox.critical(self, tr("error"), tr("save_config_failed_error") + f" {str(e)}")

    def load_config(self):
        """从文件加载训练配置"""
        config_path, _ = QFileDialog.getOpenFileName(
            self, tr("load_config_dialog_title"),
            self._last_browse_path,
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
                
                QMessageBox.information(self, tr("success"), tr("config_loaded_successfully") + f" {config_path}")
            except Exception as e:
                logger.exception(f"加载配置失败: {e}")
                QMessageBox.critical(self, tr("error"), tr("load_config_failed_error") + f" {str(e)}")

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

        config['incremental'] = self.incremental_checkbox.isChecked()
        if config['incremental']:
            config['incremental'] = self.incremental_path_edit.text()
        
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
        config['lrf'] = self.lrf_spin.value()
        config['cos_lr'] = self.cos_lr_checkbox.isChecked()
        
        # 数据增强
        config['augment'] = self.augment_checkbox.isChecked()
        config['mixup'] = self.mixup_spin.value()
        config['degrees'] = self.degrees_spin.value()
        config['shear'] = self.shear_spin.value()
        config['perspective'] = self.perspective_spin.value()
        config['flipud'] = self.flip_up_down_spin.value()
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
        config['erasing'] = self.label_smoothing_spin.value()
        config['dropout'] = self.dropout_spin.value()
        config['seed'] = self.seed_spin.value()
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

        # 增量训练
        incremental = self.config.get('incremental', False)
        self.incremental_checkbox.setChecked(bool(incremental))
        if isinstance(incremental, str) and incremental:
            self.incremental_path_edit.setText(incremental)
        
        # 训练参数
        self.epochs_spin.setValue(self.config.get('epochs', 300))
        self.imgsz_spin.setValue(self.config.get('imgsz', 640))
        self.batch_spin.setValue(self.config.get('batch', 4))
        self.workers_spin.setValue(self.config.get('workers', 4))
        
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
        
        self.lr0_spin.setValue(self.config.get('lr0', 0.01))
        self.lrf_spin.setValue(self.config.get('lrf', 0.01))
        self.cos_lr_checkbox.setChecked(self.config.get('cos_lr', True))
        
        # 数据增强
        augment = self.config.get('augment', True)
        self.augment_checkbox.setChecked(augment)
        self.mixup_spin.setValue(self.config.get('mixup', 0.0))
        self.degrees_spin.setValue(self.config.get('degrees', 0.0))
        self.shear_spin.setValue(self.config.get('shear', 0.0))
        self.perspective_spin.setValue(self.config.get('perspective', 0.0))
        self.flip_up_down_spin.setValue(self.config.get('flipud', 0.0))
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
        self.label_smoothing_spin.setValue(self.config.get('erasing', 0.4))
        self.dropout_spin.setValue(self.config.get('dropout', 0.0))
        self.seed_spin.setValue(self.config.get('seed', 0))
        self.fliplr_spin.setValue(self.config.get('fliplr', 0.5))
        self.mosaic_spin.setValue(self.config.get('mosaic', 1.0))
        self.copy_paste_spin.setValue(self.config.get('copy_paste', 0.0))

    def validate_config(self) -> bool:
        """验证配置"""
        config = self.collect_config_from_ui()

        is_resume = config['resume'] and isinstance(config['resume'], str) and Path(config['resume']).exists()
        is_incremental = config['incremental'] and isinstance(config['incremental'], str)

        # 恢复训练 / 增量训练时不需要预训练 model_path
        if not is_resume and not is_incremental:
            if not config['model_path']:
                logger.warning("训练验证失败: 未指定模型路径")
                QMessageBox.warning(self, tr("warning"), tr("validation_failed_model_file"))
                self.model_path_edit.setFocus()
                return False

            if not Path(config['model_path']).exists():
                logger.warning(f"训练验证失败: 模型文件不存在 {config['model_path']}")
                QMessageBox.warning(self, tr("warning"), tr("model_file_not_exists") + f" {config['model_path']}")
                self.model_path_edit.setFocus()
                return False

        if not config['data_yaml']:
            logger.warning("训练验证失败: 未指定 data.yaml")
            QMessageBox.warning(self, tr("warning"), tr("validation_failed_data_yaml"))
            self.data_yaml_edit.setFocus()
            return False

        if not Path(config['data_yaml']).exists():
            logger.warning(f"训练验证失败: data.yaml 不存在 {config['data_yaml']}")
            QMessageBox.warning(self, tr("warning"), tr("data_yaml_not_exists") + f" {config['data_yaml']}")
            self.data_yaml_edit.setFocus()
            return False

        # 如果启用了恢复训练，检查检查点文件
        if config['resume'] and isinstance(config['resume'], str):
            if not Path(config['resume']).exists():
                logger.warning(f"训练验证失败: 恢复检查点不存在 {config['resume']}")
                QMessageBox.warning(self, tr("warning"), tr("checkpoint_file_not_exists") + f" {config['resume']}")
                self.resume_path_edit.setFocus()
                return False

        # 如果启用了增量训练，检查权重文件
        if is_incremental:
            if not Path(config['incremental']).exists():
                logger.warning(f"训练验证失败: 增量训练权重文件不存在 {config['incremental']}")
                QMessageBox.warning(self, tr("warning"), tr("incremental_file_not_exists", "增量训练权重文件不存在") + f" {config['incremental']}")
                self.incremental_path_edit.setFocus()
                return False

        return True

    def load_last_config(self):
        """加载上次保存的训练配置"""
        try:
            if self.config_file_path.exists():
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    self.config_parser.read_file(f)
                
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
            logger.error(f"加载上次训练配置失败: {e}")

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
            
            self.log_message(tr("config_auto_saved"))
        except Exception as e:
            # 保存失败时不中断程序
            logger.error(tr("config_auto_save_failed").replace("{error}", str(e)))

    def connect_config_change_signals(self):
        """连接所有控件的修改信号，以便在配置发生变化时自动保存"""
        # 文本输入框
        text_edits = [
            self.model_path_edit,
            self.data_yaml_edit,
            self.output_dir_edit,
            self.run_name_edit,
            self.resume_path_edit,
            self.incremental_path_edit
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
            self.flip_up_down_spin,
            self.lr0_spin,
            self.lrf_spin,
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
            self.seed_spin,
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
            self.incremental_checkbox,
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
