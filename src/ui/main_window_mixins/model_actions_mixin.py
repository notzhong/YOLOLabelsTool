"""
模型操作 Mixin：模型加载/卸载/信息、自动标注（单个/批量）、训练/导出/验证对话框

从 main_window.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 MainWindow 的属性与其他 Mixin 方法。
"""

import os

from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QProgressDialog, QApplication,
)
from PySide6.QtCore import Qt

from src.utils.i18n import tr

class ModelActionsMixin:
    """模型操作 Mixin：模型加载/卸载/信息、自动标注（单个/批量）、训练/导出/验证对话框"""



    def load_model(self):
        """加载YOLO模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, tr("model_file_selection"),
            self._last_browse_path,
            tr("model_file_filter")
        )
        
        if model_path:
            # 检查YOLO库是否可用
            if not self.model_manager.is_available():
                QMessageBox.critical(
                    self, tr("error"),
                    tr("ultralytics_not_installed")
                )
                return
            
            success = self.model_manager.load_model(model_path)
            if success:
                model_info = self.model_manager.get_model_info()
                class_count = model_info["class_count"]
                classes = model_info["classes"]
                
                # 询问是否将模型类别添加到类别管理器
                if classes:
                    reply = QMessageBox.question(
                        self, tr("import_model_classes"),
                        tr("import_model_classes_confirmation").replace("{class_count}", str(class_count)),
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.Yes:
                        # 导入模型类别
                        for class_id, class_name in classes.items():
                            # 生成随机颜色
                            import random
                            color = (
                                random.randint(50, 255),
                                random.randint(50, 255),
                                random.randint(50, 255)
                            )
                            # 添加或更新类别
                            self.class_manager.add_or_update_class(class_id, class_name, color)
                        
                        self.update_class_list()
                        QMessageBox.information(self, tr("success"), tr("success"))
                
                # 更新模型信息面板
                self.model_info_panel.update_info(self.model_manager)

                # 如果模型信息面板是隐藏的，自动显示它
                if not self.model_info_panel.isVisible():
                    self.model_info_panel.setVisible(True)
                    self.action_model_info.setText(tr("hide_model_info"))
                
                QMessageBox.information(
                    self, tr("success"),
                    tr("load_model_success").replace("{model_path}", model_path).replace("{class_count}", str(class_count))
                )
                self.update_status(tr("model_loaded_status").replace("{model_name}", os.path.basename(model_path)))
            else:
                QMessageBox.critical(
                    self, tr("error"),
                    tr("load_model_failed").replace("{model_path}", model_path)
                )

    def show_model_info(self):
        """显示/隐藏模型信息面板"""
        # 切换面板可见性
        is_visible = not self.model_info_panel.isVisible()
        self.model_info_panel.setVisible(is_visible)

        # 如果显示面板，更新内容
        if is_visible:
            self.model_info_panel.update_info(self.model_manager)
            # 确保UI文本使用当前语言
            self.update_other_ui_elements()
        
        # 更新菜单文本
        if is_visible:
            self.action_model_info.setText(tr("hide_model_info"))
            self.update_status(tr("model_info_panel_shown"))
        else:
            self.action_model_info.setText(tr("show_model_info"))
            self.update_status(tr("model_info_panel_hidden"))

    def open_validation_window(self):
        """打开验证窗口"""
        try:
            from src.ui.validation_dialog import ValidationDialog
            dialog = ValidationDialog(self, self.model_manager)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"打开验证窗口失败: {e}")
            QMessageBox.critical(
                self, tr("error"),
                tr("open_validation_window_failed").replace("{error}", str(e))
            )

    def auto_annotate_current(self):
        """自动标注当前图片"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return

        # 如果已有标注，确认是否覆盖
        if self.annotation_manager.has_annotations(self.current_image_path):
            reply = QMessageBox.question(
                self, tr("confirm"),
                tr("auto_annotation_overwrite_warning"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            # 显示进度
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # 使用模型进行推理
            detections = self.model_manager.predict(self.current_image_path)
            
            if not detections:
                QApplication.restoreOverrideCursor()
                QMessageBox.information(self, tr("result"), tr("no_target_detected"))
                return
            
            # 将检测结果转换为标注
            annotations = self.model_manager.convert_to_annotations(detections)
            
            if not annotations:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, tr("warning"), tr("convert_detections_failed"))
                return
            
            # 通过 canvas 清空并绘制新标注
            self.canvas.draw_annotations(annotations)

            # 保存标注
            self.annotation_manager.save_annotations(self.current_image_path, annotations)
            
            # 更新状态
            self.update_status(tr("auto_annotation_complete").replace("{detection_count}", str(len(annotations))))
            QApplication.restoreOverrideCursor()
            
            QMessageBox.information(
                self, tr("complete"),
                tr("auto_annotation_complete").replace("{detection_count}", str(len(annotations)))
            )
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, tr("error"), f"{tr('auto_annotation_failed')}: {str(e)}")

    def batch_auto_annotate(self):
        """批量自动标注"""
        
        if self.image_manager.get_image_count() == 0:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return
        
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, tr("warning"), tr("no_model_loaded"))
            return
        
        # 确认批量标注
        msg = (tr("batch_annotation_confirmation").replace("{image_count}", str(self.image_manager.get_image_count()))
               + "\n\n⚠ " + tr("batch_overwrite_warning"))
        reply = QMessageBox.question(
            self, tr("confirm_batch_annotation"),
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # 创建进度对话框
            progress_dialog = QProgressDialog(tr("batch_annotation_in_progress"), tr("cancel"), 
                                               0, self.image_manager.get_image_count(), self)
            progress_dialog.setWindowTitle(tr("batch_annotation_progress"))
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()
            
            success_count = 0
            total_detections = 0
            
            for i in range(self.image_manager.get_image_count()):
                if progress_dialog.wasCanceled():
                    break
                
                image_path = self.image_manager.get_image_path(i)
                
                try:
                    # 推理
                    detections = self.model_manager.predict(image_path)
                    annotations = self.model_manager.convert_to_annotations(detections)
                    
                    if annotations:
                        # 保存标注
                        self.annotation_manager.save_annotations(image_path, annotations)
                        success_count += 1
                        total_detections += len(annotations)
                    
                except Exception as e:
                    self.logger.error(f"标注图片 {os.path.basename(image_path)} 失败: {e}")
                
                # 更新进度
                progress_dialog.setValue(i + 1)
                QApplication.processEvents()
            
            progress_dialog.close()
            
            # 更新图片列表显示
            self.update_image_list()
            
            # 显示结果
            QMessageBox.information(
                self, tr("batch_annotation_complete"),
                tr("batch_annotation_result").replace("{success_count}", str(success_count))
                                                  .replace("{total_count}", str(self.image_manager.get_image_count()))
                                                  .replace("{total_detections}", str(total_detections))
            )
            
            self.update_status(tr("batch_annotation_complete_status").replace("{image_count}", str(success_count)))
            
        except Exception as e:
            QMessageBox.critical(self, tr("error"), tr("batch_annotation_failed").replace("{error}", str(e)))

    def unload_model(self):
        """卸载模型"""
        if not self.model_manager.is_model_loaded():
            QMessageBox.information(self, tr("info"), tr("no_model_loaded"))
            return
        
        reply = QMessageBox.question(
            self, tr("confirm_unload"),
            tr("unload_model_confirmation"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.model_manager.unload_model()
            self.model_info_panel.update_info(self.model_manager)
            
            self.update_status(tr("model_unloaded"))
            QMessageBox.information(self, tr("success"), tr("model_unloaded"))

    def update_model_info_panel(self):
        """更新模型信息面板（委托给 ModelInfoPanel）"""
        self.model_info_panel.update_info(self.model_manager)

    def train_model(self):
        """训练模型"""
        # 检查是否加载了模型，如果有则获取模型路径作为默认值
        default_model_path = ""
        if self.model_manager.is_model_loaded():
            model_info = self.model_manager.get_model_info()
            default_model_path = model_info.get("path", "")
        
        # 创建训练配置对话框
        from .train_dialog import TrainDialog
        dialog = TrainDialog(self, default_model_path)
        
        if dialog.exec():
            # 对话框已确认，训练将在对话框内部启动
            self.update_status(tr("training_config_complete"))

    def export_model(self):
        """导出模型"""
        default_model_path = ""
        if self.model_manager.is_model_loaded():
            model_info = self.model_manager.get_model_info()
            default_model_path = model_info.get("path", "")

        from .export_dialog import ExportDialog
        dialog = ExportDialog(self, default_model_path)
        dialog.exec()
