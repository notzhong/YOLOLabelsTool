"""
Class Edit Dialog
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QColorDialog, QWidget,
    QFormLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette

from src.utils.i18n import tr


class ClassDialog(QDialog):
    """Class Edit Dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.class_name = ""
        self.color = (255, 0, 0)  # 默认红色
        
        self.init_ui()
        
        # 设置窗口属性
        self.setWindowTitle(tr("class_dialog_title"))
        self.setModal(True)
        self.resize(300, 150)
        
        # 更新UI文本
        self.update_ui_texts()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 表单布局
        form_layout = QFormLayout()
        
        # 类别名称 - 保存标签引用
        self.name_label = QLabel(tr("class_name_label_dialog"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("enter_class_name"))
        form_layout.addRow(self.name_label, self.name_edit)
        
        # 颜色选择 - 保存标签引用
        self.color_label = QLabel(tr("class_color_label"))
        color_layout = QHBoxLayout()
        
        self.color_preview = QWidget()
        self.color_preview.setFixedSize(50, 25)
        self.color_preview.setAutoFillBackground(True)
        self.update_color_preview()
        
        self.color_btn = QPushButton(tr("choose_color_btn"))
        self.color_btn.clicked.connect(self.select_color)
        
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(self.color_btn)
        color_layout.addStretch()
        
        # 创建容器小部件以容纳颜色布局
        color_widget = QWidget()
        color_widget.setLayout(color_layout)
        form_layout.addRow(self.color_label, color_widget)
        
        layout.addLayout(form_layout)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.ok_btn = QPushButton(tr("ok"))
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton(tr("cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def select_color(self):
        """选择颜色"""
        current_color = QColor(*self.color)
        color = QColorDialog.getColor(current_color, self, tr("select_color_dialog_title"))
        
        if color.isValid():
            self.color = (color.red(), color.green(), color.blue())
            self.update_color_preview()
    
    def update_color_preview(self):
        """更新颜色预览"""
        palette = self.color_preview.palette()
        palette.setColor(QPalette.Window, QColor(*self.color))
        self.color_preview.setPalette(palette)
    
    def get_values(self):
        """获取对话框中的值"""
        return self.class_name, self.color
    
    def set_values(self, class_name: str, color: tuple):
        """设置对话框中的值"""
        self.class_name = class_name
        self.color = color
        
        self.name_edit.setText(class_name)
        self.update_color_preview()
    
    def update_ui_texts(self):
        """更新UI文本"""
        # 更新窗口标题
        self.setWindowTitle(tr("class_dialog_title"))
        
        # 更新表单标签（需要找到表单布局并更新）
        # 由于表单布局没有保存行引用，我们无法直接更新
        # 这里主要更新按钮文本
        if hasattr(self, 'ok_btn'):
            self.ok_btn.setText(tr("ok"))
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setText(tr("cancel"))
        if hasattr(self, 'color_btn'):
            self.color_btn.setText(tr("select"))
    
    def accept(self):
        """接受对话框"""
        self.class_name = self.name_edit.text().strip()
        
        if not self.class_name:
            self.name_edit.setFocus()
            return
        
        super().accept()
