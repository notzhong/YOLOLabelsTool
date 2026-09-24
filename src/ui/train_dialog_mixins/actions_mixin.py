"""
对话框生命周期与训练启动 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

from PySide6.QtWidgets import QMessageBox

from src.utils.i18n import tr
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)


class TrainActionsMixin:
    """对话生命周期与训练启动 Mixin：关闭/取消/默认值重置/启动训练"""


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
        logger.info(f"[TrainDialog] {message}")

    def reset_to_defaults(self):
        """重置所有参数到市场最优默认值"""
        reply = QMessageBox.question(
            self,
            tr("reset_default_params"),
            tr("reset_params_confirmation"),
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
                tr("success"),
                tr("reset_params_complete")
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
            logger.error(f"训练器 setup 失败, config keys: {list(config.keys())}")
            QMessageBox.critical(self, tr("error"), tr("train_params_setup_failed"))
            return
        
        # 创建训练进度对话框
        from .train_progress_dialog import TrainProgressDialog
        progress_dialog = TrainProgressDialog(self.trainer, parent=self)
        
        # 断开旧连接，防止信号累积
        try:
            self.trainer.training_finished.disconnect()
        except Exception:
            pass

        # 连接训练完成信号
        def on_training_finished(success: bool, message: str):
            self.btn_start.setEnabled(True)
            if success:
                logger.info(f"训练完成: {message}")
                QMessageBox.information(self, tr("training_completed"), message)
            else:
                logger.error(f"训练失败: {message}")
                QMessageBox.critical(self, tr("training_failed"), message)

        self.trainer.training_finished.connect(on_training_finished)
        
        # 禁用开始按钮
        self.btn_start.setEnabled(False)
        
        # 显示进度对话框并开始训练
        if progress_dialog.start_training():
            self.accept()  # 关闭配置对话框
        else:
            self.btn_start.setEnabled(True)
