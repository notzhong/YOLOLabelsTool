"""
类别编辑对话框
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QColorDialog, QWidget,
    QFormLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette


class ClassDialog(QDialog):
    """类别编辑对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.class_name = ""
        self.color = (255, 0, 0)  # 默认红色
        
        self.init_ui()
        
        # 设置窗口属性
        self.setWindowTitle("类别编辑")
        self.setModal(True)
        self.resize(300, 150)
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 表单布局
        form_layout = QFormLayout()
        
        # 类别名称
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入类别名称")
        form_layout.addRow("类别名称:", self.name_edit)
        
        # 颜色选择
        color_layout = QHBoxLayout()
        
        self.color_preview = QWidget()
        self.color_preview.setFixedSize(50, 25)
        self.color_preview.setAutoFillBackground(True)
        self.update_color_preview()
        
        self.color_btn = QPushButton("选择颜色")
        self.color_btn.clicked.connect(self.select_color)
        
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(self.color_btn)
        color_layout.addStretch()
        
        form_layout.addRow("颜色:", color_layout)
        
        layout.addLayout(form_layout)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def select_color(self):
        """选择颜色"""
        current_color = QColor(*self.color)
        color = QColorDialog.getColor(current_color, self, "选择类别颜色")
        
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
    
    def accept(self):
        """接受对话框"""
        self.class_name = self.name_edit.text().strip()
        
        if not self.class_name:
            self.name_edit.setFocus()
            return
        
        super().accept()