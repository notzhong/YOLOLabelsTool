"""
训练进度对话框 - 显示训练进度和日志
"""
import time
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QProgressBar, QMessageBox, QGroupBox, QWidget
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor

from yolo_tool import YOLOTrainer
from src.utils.i18n import tr


class TrainProgressDialog(QDialog):
    """训练进度对话框"""
    
    def __init__(self, trainer: YOLOTrainer, parent=None):
        super().__init__(parent)
        
        self.trainer = trainer
        self.start_time: Optional[datetime] = None
        
        self.init_ui()
        
        # 设置窗口属性
        self.setWindowTitle(tr("training_progress"))
        self.setModal(True)
        self.resize(800, 600)
        
        # 连接信号
        self.trainer.training_started.connect(self.on_training_started)
        self.trainer.progress_updated.connect(self.on_progress_updated)
        self.trainer.log_message.connect(self.on_log_message)
        self.trainer.training_finished.connect(self.on_training_finished)
        self.trainer.training_stopped.connect(self.on_training_stopped)
    
    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout(self)
        
        # 训练信息
        info_group = QGroupBox(tr("training_info"))
        info_layout = QVBoxLayout(info_group)
        
        # 状态标签
        self.status_label = QLabel(tr("ready_to_start_training_status"))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        info_layout.addWidget(self.progress_bar)
        
        # 详细信息
        details_layout = QHBoxLayout()
        
        # 已用时间
        self.time_label = QLabel(tr("elapsed_time_label"))
        details_layout.addWidget(self.time_label)
        
        # 当前epoch
        self.epoch_label = QLabel(tr("current_epoch_label"))
        details_layout.addWidget(self.epoch_label)
        
        # 损失值
        self.loss_label = QLabel(tr("loss_value_label"))
        details_layout.addWidget(self.loss_label)
        
        details_layout.addStretch()
        info_layout.addLayout(details_layout)
        
        main_layout.addWidget(info_group)
        
        # 日志输出
        log_group = QGroupBox(tr("training_log_title"))
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_group)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        # 重新配置按钮（训练失败时显示）
        self.btn_reconfigure = QPushButton(tr("reconfigure_training_btn"))
        self.btn_reconfigure.clicked.connect(self.reconfigure_training)
        self.btn_reconfigure.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_reconfigure.setVisible(False)
        button_layout.addWidget(self.btn_reconfigure)
        
        # 停止训练按钮
        self.btn_stop = QPushButton(tr("stop_training_btn_label"))
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop.setEnabled(False)
        button_layout.addWidget(self.btn_stop)
        
        # 关闭按钮（默认隐藏）
        self.btn_close = QPushButton(tr("close_dialog_btn"))
        self.btn_close.clicked.connect(self.accept)
        self.btn_close.setVisible(False)
        button_layout.addWidget(self.btn_close)
        
        main_layout.addLayout(button_layout)
        
        # 计时器更新已用时间
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
    
    def start_training(self) -> bool:
        """开始训练并显示对话框"""
        if self.trainer.start_training():
            self.show()
            return True
        else:
            QMessageBox.critical(self, tr("error"), tr("unable_to_start_training"))
            return False
    
    @Slot()
    def on_training_started(self):
        """训练开始事件"""
        self.start_time = datetime.now()
        self.status_label.setText(tr("training_in_progress_status"))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        
        self.btn_stop.setEnabled(True)
        self.btn_close.setVisible(False)
        
        # 启动计时器
        self.timer.start(1000)  # 每秒更新一次
        
        self.log_message(tr("training_started"))
    
    @Slot(int, dict)
    def on_progress_updated(self, epoch: int, metrics: dict):
        """进度更新事件"""
        # 更新进度条（假设总epochs在配置中）
        total_epochs = self.trainer.config.get('epochs', 300)
        progress = int((epoch / total_epochs) * 100) if total_epochs > 0 else 0
        self.progress_bar.setValue(progress)
        
        # 更新标签
        self.epoch_label.setText(tr("training_progress_current_epoch_format").format(epoch, total_epochs))
        
        # 更新损失值
        if metrics:
            loss = metrics.get('loss', 'N/A')
            self.loss_label.setText(tr("training_progress_loss_format").format(loss))
        else:
            self.loss_label.setText(tr("training_progress_loss_default"))
    
    @Slot(str)
    def on_log_message(self, message: str):
        """日志消息事件"""
        self.log_message(message)
    
    @Slot(bool, str)
    def on_training_finished(self, success: bool, message: str):
        """训练完成事件"""
        # 停止计时器
        self.timer.stop()
        
        # 更新状态
        if success:
            self.status_label.setText(tr("training_completed_status"))
            self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(tr("training_failed_status_label"))
            self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #f44336;")
        
        # 更新按钮状态
        self.btn_stop.setEnabled(False)
        self.btn_close.setVisible(True)
        
        # 训练失败时显示重新配置按钮
        if not success:
            self.btn_reconfigure.setVisible(True)
        
        # 记录完成消息
        self.log_message(message)
        self.log_message(tr("training_ended"))
    
    @Slot()
    def on_training_stopped(self):
        """训练停止事件"""
        self.status_label.setText(tr("training_stopped_status"))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #ff9800;")
        
        self.btn_stop.setEnabled(False)
        self.btn_close.setVisible(True)
        
        self.log_message(tr("training_stopped_message"))
    
    def reconfigure_training(self):
        """重新配置训练"""
        try:
            # 隐藏重新配置按钮
            self.btn_reconfigure.setVisible(False)
            self.btn_close.setVisible(False)
            
            # 清理训练器资源
            if hasattr(self.trainer, 'cleanup'):
                self.trainer.cleanup()
            else:
                # 如果cleanup方法不存在，尝试其他清理方式
                self.trainer.is_training = False
                self.trainer.should_stop = False
                if self.trainer.model is not None:
                    self.trainer.model = None
            
            # 重置UI状态
            self.status_label.setText(tr("ready_to_reconfigure"))
            self.status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.progress_bar.setValue(0)
            self.epoch_label.setText(tr("training_progress_epoch_default"))
            self.loss_label.setText(tr("training_progress_loss_default"))
            self.time_label.setText(tr("training_progress_time_default"))
            
            # 停止计时器
            if self.timer.isActive():
                self.timer.stop()
            
            # 记录操作
            self.log_message(tr("reconfiguring_training"))
            
            # 关闭当前对话框并返回拒绝结果
            # 父窗口可以检测到拒绝结果并重新打开配置对话框
            self.done(QDialog.Rejected)
            
        except Exception as e:
            self.log_message(tr("reconfigure_failed_prefix") + str(e))
            QMessageBox.critical(self, tr("error"), tr("reconfigure_failed_prefix") + str(e))
            
            # 恢复按钮状态
            self.btn_reconfigure.setVisible(True)
            self.btn_close.setVisible(True)
    
    def stop_training(self):
        """停止训练"""
        reply = QMessageBox.question(
            self, tr("stop_training_confirm_title"),
            tr("stop_training_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.trainer.stop_training()
            self.btn_stop.setEnabled(False)
            self.log_message(tr("stopping_training"))
    
    def log_message(self, message: str):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        # 追加到文本编辑框
        self.log_text.append(log_line)
        
        # 滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
        # 保持一定行数，避免过大
        max_lines = 1000
        lines = self.log_text.toPlainText().split('\n')
        if len(lines) > max_lines:
            self.log_text.setPlainText('\n'.join(lines[-max_lines:]))
    
    def update_elapsed_time(self):
        """更新已用时间"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            time_str = tr("training_progress_elapsed_time_format").format(int(hours), int(minutes), int(seconds))
            self.time_label.setText(time_str)
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.trainer.is_training:
            reply = QMessageBox.question(
                self, tr("close_dialog_confirm_title"),
                tr("close_dialog_confirm_msg"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        
        # 停止计时器
        if self.timer.isActive():
            self.timer.stop()
        
        event.accept()
