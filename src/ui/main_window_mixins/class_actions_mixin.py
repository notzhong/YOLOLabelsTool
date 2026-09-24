"""
类别管理 Mixin：类别列表、增删改、YAML 导入导出、键盘循环选择

从 main_window.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 MainWindow 的属性与其他 Mixin 方法。
"""


from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QMenu,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from src.utils.i18n import tr

class ClassActionsMixin:
    """类别管理 Mixin：类别列表、增删改、YAML 导入导出、键盘循环选择"""



    def _select_next_class(self):
        """选择下一个类别"""
        count = self.class_list_widget.count()
        if count == 0:
            return
        current_row = self.class_list_widget.currentRow()
        next_row = (current_row + 1) % count
        self.class_list_widget.setCurrentRow(next_row)
        item = self.class_list_widget.item(next_row)
        if item:
            self.on_class_item_clicked(item)
            self.update_status(
                tr("selected_class").replace(
                    "{class_name}", self.class_manager.get_class_name(self.selected_class_id)
                )
            )

    def _select_prev_class(self):
        """选择上一个类别"""
        count = self.class_list_widget.count()
        if count == 0:
            return
        current_row = self.class_list_widget.currentRow()
        prev_row = (current_row - 1) % count
        self.class_list_widget.setCurrentRow(prev_row)
        item = self.class_list_widget.item(prev_row)
        if item:
            self.on_class_item_clicked(item)
            self.update_status(
                tr("selected_class").replace(
                    "{class_name}", self.class_manager.get_class_name(self.selected_class_id)
                )
            )

    def update_class_list(self):
        """更新类别列表"""
        self.class_list_widget.clear()
        
        classes = self.class_manager.get_classes()
        for class_id, class_info in sorted(classes.items()):
            class_name = class_info["name"]
            color = class_info["color"]
            
            item_text = f"{class_id}: {class_name}"
            self.class_list_widget.addItem(item_text)
            
            # 设置背景颜色
            item = self.class_list_widget.item(self.class_list_widget.count() - 1)
            item.setData(Qt.UserRole, class_id)
            item.setBackground(QColor(*color))
            
            # 设置文字颜色为对比色
            brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
            text_color = QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)
            item.setForeground(text_color)

    def _on_class_item_double_clicked(self, item):
        """类别列表项双击 → 编辑"""
        self.edit_class()

    def _on_class_list_context_menu(self, pos):
        """类别列表右键菜单"""
        item = self.class_list_widget.itemAt(pos)
        if item is None:
            return
        self.class_list_widget.setCurrentItem(item)
        menu = QMenu(self)
        action_edit = menu.addAction(tr("edit"))
        action_edit.triggered.connect(self.edit_class)
        action_delete = menu.addAction(tr("delete"))
        action_delete.triggered.connect(self.delete_class)
        menu.exec(self.class_list_widget.mapToGlobal(pos))

    def on_class_item_clicked(self, item):
        """类别列表项点击事件"""
        # 优先从UserRole读取真实class_id，避免行号和class_id错位
        class_id = item.data(Qt.UserRole)
        if class_id is None:
            # 兼容旧数据：从文本中回退解析
            item_text = item.text()
            if ": " in item_text:
                try:
                    class_id_str = item_text.split(": ")[0]
                    class_id = int(class_id_str)
                except ValueError:
                    self.update_status(tr("cannot_parse_class_id_error"))
                    return
            else:
                self.update_status(tr("invalid_class_format_error"))
                return

        self.selected_class_id = int(class_id)
        self.canvas.selected_class_id = self.selected_class_id
        self.update_status(tr("selected_class").replace("{class_name}", self.class_manager.get_class_name(self.selected_class_id)))

    def add_class(self):
        """添加类别"""
        from src.ui.class_dialog import ClassDialog

        dialog = ClassDialog(self)
        # 预先分配一个不与现有颜色重复的颜色并显示在对话框中
        auto_color = self.class_manager._generate_color()
        dialog.set_color(auto_color)

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
        
        selected_item = selected_items[0]
        class_id = selected_item.data(Qt.UserRole)
        if class_id is None:
            QMessageBox.warning(self, tr("warning"), tr("cannot_parse_class_id_error"))
            return

        class_id = int(class_id)
        class_info = self.class_manager.get_class(class_id)
        
        # 检查类别是否存在
        if not class_info:
            QMessageBox.warning(self, tr("warning"), tr("class_not_exist"))
            return
        
        from src.ui.class_dialog import ClassDialog
        
        dialog = ClassDialog(self)
        dialog.set_values(class_info["name"], class_info["color"])
        
        if dialog.exec():
            class_name, color = dialog.get_values()
            self.class_manager.update_class(class_id, class_name, color)
            self.update_class_list()

    def delete_class(self):
        """删除类别"""
        selected_items = self.class_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, tr("warning"), tr("select_class"))
            return
        
        selected_item = selected_items[0]
        class_id = selected_item.data(Qt.UserRole)
        if class_id is None:
            QMessageBox.warning(self, tr("warning"), tr("cannot_parse_class_id_error"))
            return

        class_id = int(class_id)
        
        # 获取要删除的类别名称
        class_info = self.class_manager.get_class(class_id)
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
            if class_id == self.selected_class_id:
                self.selected_class_id = 0
                self.canvas.selected_class_id = 0
                if self.class_manager.get_class_count() > 0:
                    # 选择第一个可用的类别
                    available_classes = list(self.class_manager.get_classes().keys())
                    if available_classes:
                        self.selected_class_id = available_classes[0]
                        self.canvas.selected_class_id = self.selected_class_id
            
            # 删除类别
            self.class_manager.delete_class(class_id)
            self.update_class_list()
            
            # 更新状态
            self.update_status(tr("class_deleted").replace("{class_name}", class_name))

    def clear_all_classes(self):
        """清空所有类别"""
        if self.class_manager.get_class_count() == 0:
            QMessageBox.information(self, tr("info"), tr("no_classes_to_clear"))
            return
        
        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_clear_all_classes"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清空所有类别
            self.class_manager.clear_all()
            self.update_class_list()
            
            # 重置选中的类别
            self.selected_class_id = 0
            self.canvas.selected_class_id = 0
            
            # 更新状态
            self.update_status(tr("all_classes_cleared"))

    def load_classes_from_yaml(self):
        """从YAML文件加载类别"""
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, tr("load_classes_from_yaml_dialog"),
            self._last_browse_path,
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

        yaml_path, _ = QFileDialog.getSaveFileName(
            self, tr("save_classes_to_yaml_dialog"),
            self._last_browse_path,
            tr("yaml_file_filter")
        )
        
        if yaml_path:
            # 确保文件扩展名
            if not yaml_path.lower().endswith(('.yaml', '.yml')):
                yaml_path += '.yaml'
            
            # 导出YAML
            self.class_manager.export_to_yaml(yaml_path)
            QMessageBox.information(self, tr("success"), tr("save_yaml_success") + f" {yaml_path}")
