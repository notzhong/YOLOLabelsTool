"""
图片操作 Mixin：文件夹加载/关闭、列表浏览、删除/移除、缩放与视图

从 main_window.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 MainWindow 的属性与其他 Mixin 方法。
"""

import os

from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QMenu,
)
from PySide6.QtGui import QPixmap

from src.utils.i18n import tr

class ImageActionsMixin:
    """图片操作 Mixin：文件夹加载/关闭、列表浏览、删除/移除、缩放与视图"""



    def load_image_folder(self):
        """通过对话框加载图片文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, tr("select_image_folder_dialog_title"),
            self._last_browse_path
        )

        if folder_path:
            self.load_image_folder_by_path(folder_path)

    def load_image_folder_by_path(self, folder_path: str):
        """加载指定路径的图片文件夹（供对话框和自动加载共用）"""
        self._last_browse_path = folder_path
        self._last_folder_path = folder_path
        # 新文件夹 = 新的撤销上下文：清空指向旧文件夹图片的 undo/redo 栈
        self.annotation_manager._undo_stack.clear()
        self.annotation_manager._redo_stack.clear()
        self.update_undo_redo_actions()
        self.image_manager.load_folder(folder_path)
        self.update_image_list()
        self.update_stats()

        if self.image_manager.get_image_count() > 0:
            self.load_image(0)

    def close_image_folder(self):
        """关闭当前文件夹，下次启动不再自动打开"""
        self.image_manager._image_paths.clear()
        self.image_manager._current_folder = None
        self.image_manager.clear_cache()
        self.annotation_manager._annotations.clear()
        # 关夹后 undo/redo 栈中的命令指向已关闭的图片，必须清空，
        # 否则 Ctrl+Z 仍会按旧路径把标注写回磁盘（孤儿标注文件）
        self.annotation_manager._undo_stack.clear()
        self.annotation_manager._redo_stack.clear()
        self.update_undo_redo_actions()
        self._last_folder_path = ""
        self.current_image_path = None
        self.current_image_index = 0
        self.canvas._scene.clear()
        self.canvas._crosshair_items = None
        self.update_image_list()
        self.update_stats()
        self.image_info_label.setText(tr("no_image_loaded"))
        self.status_image_info.setText("")
        self.status_annotation_info.setText("")
        self.save_settings()

    def load_image(self, index: int):
        """加载指定索引的图片"""
        if 0 <= index < self.image_manager.get_image_count():
            # 切图前自动保存当前标注
            if self.current_image_path:
                annotations = self.canvas.get_annotation_items()
                self.annotation_manager.save_annotations(self.current_image_path, annotations)

            image_path = self.image_manager.get_image_path(index)
            self.current_image_path = image_path
            self.current_image_index = index

            # 加载图片
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 交由 canvas 显示（内部会清除旧场景）
                self.canvas.display_image(pixmap)

                # 更新状态
                self.update_image_info()

                # 加载标注
                self.load_annotations_for_current_image()

                # 选中列表项
                self.image_list_widget.setCurrentRow(index)
            else:
                QMessageBox.warning(self, tr("error"), f"{tr('cannot_load_image')}{image_path}")

    def update_image_list(self):
        """更新图片列表"""
        self.image_list_widget.clear()
        
        for i in range(self.image_manager.get_image_count()):
            image_path = self.image_manager.get_image_path(i)
            image_name = os.path.basename(image_path)
            
            # 检查是否有标注
            has_annotations = self.annotation_manager.has_annotations(image_path)
            
            item_text = image_name
            if has_annotations:
                item_text += " ✓"
            
            self.image_list_widget.addItem(item_text)

    def update_image_info(self):
        """更新图片信息"""
        if self.current_image_path:
            image_name = os.path.basename(self.current_image_path)
            image_size = self.canvas.get_image_size()
            if image_size:
                info = f"{image_name} | {image_size[0]}x{image_size[1]}"
                self.image_info_label.setText(info)
                self.status_image_info.setText(f"{tr('image_name_label')}: {image_name}")
                return
        self.image_info_label.setText(tr("no_image_loaded"))
        self.status_image_info.setText("")

    def update_stats(self):
        """更新统计信息"""
        count = self.image_manager.get_image_count()
        self.stats_label.setText(tr("total_images_count").replace("{count}", str(count)))

    def update_statistics_panel(self):
        """更新标注统计面板（委托给 StatsPanel）"""
        self.stats_panel.update_statistics(
            self.image_manager, self.annotation_manager, self.class_manager
        )

    def on_image_item_clicked(self, item):
        """图片列表项点击事件"""
        index = self.image_list_widget.row(item)
        self.load_image(index)

    def _on_image_list_context_menu(self, pos):
        """图片列表右键菜单"""
        menu = QMenu(self)
        action_delete = menu.addAction(tr("delete_unannotated"))
        action_delete.triggered.connect(self._delete_all_unannotated_images)

        menu.addSeparator()

        action_remove = menu.addAction(tr("remove_from_list"))
        action_remove.triggered.connect(self._remove_selected_images)
        action_remove.setEnabled(len(self.image_list_widget.selectedItems()) > 0)

        action_remove_unannotated = menu.addAction(tr("remove_all_unannotated"))
        action_remove_unannotated.triggered.connect(self._remove_all_unannotated_images)

        menu.exec(self.image_list_widget.mapToGlobal(pos))

    def _delete_all_unannotated_images(self):
        """删除所有未标注的图片（同时删除本地文件）"""
        count = self.image_manager.get_image_count()
        if count == 0:
            return

        # 收集未标注的图片索引
        unannotated_indices = []
        for i in range(count):
            image_path = self.image_manager.get_image_path(i)
            if not self.annotation_manager.has_annotations(image_path):
                unannotated_indices.append(i)

        if not unannotated_indices:
            QMessageBox.information(self, tr("info"), tr("no_unannotated_images"))
            return

        # 确认对话框
        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_delete_unannotated").replace("{count}", str(len(unannotated_indices))),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 先收集路径再操作
        paths_to_delete = [self.image_manager.get_image_path(i) for i in unannotated_indices]
        current_image_path = self.current_image_path

        # 删除本地文件和标注文件
        deleted_count = 0
        for image_path in paths_to_delete:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    deleted_count += 1
            except OSError as e:
                self.logger.error(f"删除图片失败: {image_path}, {e}")
            # 清理对应的标注文件
            if self.annotation_manager.has_annotations(image_path):
                self.annotation_manager.clear_annotations(image_path)

        # 从列表中移除（反向遍历保持索引正确）
        current_was_deleted = current_image_path in paths_to_delete
        for i in sorted(unannotated_indices, reverse=True):
            self.image_manager.remove_image(i)

        # 处理当前显示的图片
        self._handle_image_after_removal(current_was_deleted)

        # 更新界面
        self.update_image_list()
        self.update_stats()
        self.update_status(tr("images_deleted").replace("{count}", str(deleted_count)))

    def _remove_selected_images(self):
        """从列表中移除选中的图片（不删除本地文件）"""
        selected_items = self.image_list_widget.selectedItems()
        if not selected_items:
            return

        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_remove_selected").replace("{count}", str(len(selected_items))),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 收集选中的行号和路径
        selected = []
        for item in selected_items:
            row = self.image_list_widget.row(item)
            path = self.image_manager.get_image_path(row)
            if path is not None:
                selected.append((row, path))

        current_image_path = self.current_image_path
        current_was_removed = any(path == current_image_path for _, path in selected)

        # 反向遍历移除
        for row, _ in sorted(selected, key=lambda x: x[0], reverse=True):
            self.image_manager.remove_image(row)

        # 处理当前显示的图片
        self._handle_image_after_removal(current_was_removed)

        # 更新界面
        self.update_image_list()
        self.update_stats()
        self.update_status(tr("images_removed").replace("{count}", str(len(selected))))

    def _remove_all_unannotated_images(self):
        """从列表中移除所有未标注的图片（不删除本地文件）"""
        count = self.image_manager.get_image_count()
        if count == 0:
            return

        unannotated_indices = []
        for i in range(count):
            image_path = self.image_manager.get_image_path(i)
            if not self.annotation_manager.has_annotations(image_path):
                unannotated_indices.append(i)

        if not unannotated_indices:
            QMessageBox.information(self, tr("info"), tr("no_unannotated_images"))
            return

        reply = QMessageBox.question(
            self, tr("confirm"),
            tr("confirm_remove_unannotated").replace("{count}", str(len(unannotated_indices))),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        current_image_path = self.current_image_path
        current_was_removed = any(
            self.image_manager.get_image_path(i) == current_image_path
            for i in unannotated_indices
        )

        for i in sorted(unannotated_indices, reverse=True):
            self.image_manager.remove_image(i)

        self._handle_image_after_removal(current_was_removed)

        self.update_image_list()
        self.update_stats()
        self.update_status(tr("images_removed").replace("{count}", str(len(unannotated_indices))))

    def _handle_image_after_removal(self, current_was_affected: bool):
        """处理移除图片后的当前图片状态"""
        total = self.image_manager.get_image_count()

        if total == 0:
            self.current_image_path = None
            self.current_image_index = 0
            self.canvas._scene.clear()
            self.canvas._crosshair_items = None
            self.image_info_label.setText(tr("no_image_loaded"))
            self.status_image_info.setText("")
            self.status_annotation_info.setText("")
            return

        if current_was_affected or self.current_image_path is None:
            self.load_image(0)
        else:
            # 当前图片还在列表中，但移除前面的图片会改变索引 → 按路径重定位
            paths = [
                self.image_manager.get_image_path(i) for i in range(total)
            ]
            if self.current_image_path in paths:
                self.load_image(paths.index(self.current_image_path))
            else:
                self.load_image(0)

    def prev_image(self):
        """上一张图片"""
        if self.image_manager.get_image_count() > 0:
            new_index = (self.current_image_index - 1) % self.image_manager.get_image_count()
            self.load_image(new_index)

    def next_image(self):
        """下一张图片"""
        if self.image_manager.get_image_count() > 0:
            new_index = (self.current_image_index + 1) % self.image_manager.get_image_count()
            self.load_image(new_index)

    def fit_to_window(self):
        """适应窗口大小"""
        self.canvas.fit_to_window()
        self._update_scale_status()

    def zoom_in(self):
        """放大"""
        self.canvas.zoom_in()
        self._update_scale_status()

    def zoom_out(self):
        """缩小"""
        self.canvas.zoom_out()
        self._update_scale_status()

    def reset_view(self):
        """重置视图"""
        self.canvas.reset_view()
        self.update_status(tr("view_reset_message"))

    def _update_scale_status(self):
        """更新缩放显示"""
        factor = self.canvas.get_scale_factor()
        self.update_status(tr("zoom_status").replace("{scale_factor}", f"{factor:.2f}"))
