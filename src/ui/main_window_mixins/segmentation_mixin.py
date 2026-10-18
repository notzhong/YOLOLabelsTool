"""
分割标注 Mixin：任务模式切换、YOLO 标签导入、框/多边形互转、导出前的临时转换确认

2.5.0 分割标注功能新增：通过 self 访问 MainWindow 的属性与其他 Mixin 方法。
"""

import copy
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QProgressDialog, QApplication,
)
from PySide6.QtCore import Qt

from src.core.annotation import Annotation, EditAnnotationsCommand, PolygonAnnotation
from src.utils.i18n import tr


class SegmentationMixin:
    """分割标注 Mixin：任务模式、标签导入、标注转换、导出临时转换确认"""

    # ==================== 任务模式 ====================

    def _apply_task_mode(self, mode: str, persist: bool = True):
        """应用任务模式：窗口状态 + 画布 + 工具栏下拉框（可选持久化）"""
        if mode not in ("detect", "segment"):
            mode = "detect"
        self.task_mode = mode
        self.canvas.set_task_mode(mode)

        combo = getattr(self, 'task_mode_combo', None)
        if combo is not None:
            index = combo.findData(mode)
            if index >= 0 and combo.currentIndex() != index:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

        if persist:
            self.save_settings()

    def _on_task_mode_changed(self, index: int):
        """工具栏任务模式切换：切换画布绘制行为并持久化"""
        combo = getattr(self, 'task_mode_combo', None)
        if combo is None:
            return

        mode = combo.itemData(index)
        if mode not in ("detect", "segment") or mode == self.task_mode:
            return

        self._apply_task_mode(mode)
        self.update_status(
            tr("task_mode_switched").replace("{mode}", combo.itemText(index))
        )

    def hint_task_mode_mismatch(self):
        """打开文件夹后：标注内容与任务模式明显不匹配时在状态栏提示一次"""
        if self.image_manager.get_image_count() == 0:
            return

        polygon_count = 0
        box_count = 0
        for image_path in self.image_manager.get_all_image_paths():
            for ann in self.annotation_manager.get_annotations(image_path):
                if isinstance(ann, PolygonAnnotation):
                    polygon_count += 1
                else:
                    box_count += 1

        if self.task_mode == "detect" and polygon_count > box_count:
            self.update_status(
                tr("task_mode_mismatch_polygon_hint").replace("{count}", str(polygon_count))
            )
        elif self.task_mode == "segment" and box_count > polygon_count:
            self.update_status(
                tr("task_mode_mismatch_box_hint").replace("{count}", str(box_count))
            )

    # ==================== 标签导入 ====================

    def import_segmentation_current(self):
        """导入当前图片的 YOLO 标签文件（框与多边形混合可解析，作为一步可撤销操作）"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        image_size = self.canvas.get_image_size()
        if not image_size:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        label_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("import_annotation_dialog_title"),
            str(Path(self.current_image_path).parent),
            tr("txt_file_filter"),
        )
        if not label_path:
            return

        if self.annotation_manager.has_annotations(self.current_image_path):
            reply = QMessageBox.question(
                self, tr("confirm"), tr("import_overwrite_warning"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            QMessageBox.critical(
                self, tr("error"), tr("import_failed").replace("{error}", str(e))
            )
            return

        width, height = image_size
        old_state = copy.deepcopy(
            self.annotation_manager.get_annotations(self.current_image_path)
        )
        imported, skipped = self.annotation_manager.import_from_yolo_format(
            self.current_image_path, lines, width, height
        )

        if imported == 0:
            # 没有有效行（空文件/全部无效）：恢复原标注，避免误清空
            self.annotation_manager.save_annotations(self.current_image_path, old_state)
            self.load_annotations_for_current_image()
            QMessageBox.warning(self, tr("warning"), tr("import_nothing_parsed"))
            return

        # 导入作为一步可撤销操作：命令快照负责撤销，其余照常刷新
        new_state = copy.deepcopy(
            self.annotation_manager.get_annotations(self.current_image_path)
        )
        if old_state != new_state:
            command = EditAnnotationsCommand(
                self.annotation_manager, self.current_image_path, old_state, new_state
            )
            self.annotation_manager.push_command(command)

        self._refresh_annotation_views()
        self.update_undo_redo_actions()
        QMessageBox.information(
            self, tr("import_complete_title"),
            tr("import_result")
                .replace("{imported}", str(imported))
                .replace("{skipped}", str(skipped))
        )
        self.update_status(
            tr("import_complete")
                .replace("{imported}", str(imported))
                .replace("{skipped}", str(skipped))
        )

    def import_segmentation_batch(self):
        """批量导入文件夹中的 YOLO 标签（按文件名主干匹配图片，不可撤销、直接落盘）"""
        if self.image_manager.get_image_count() == 0:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        folder = QFileDialog.getExistingDirectory(
            self, tr("import_batch_dialog_title"), self._last_browse_path
        )
        if not folder:
            return
        self._last_browse_path = folder

        # 收集可匹配的 (图片路径, 标签路径)
        candidates = []
        for index in range(self.image_manager.get_image_count()):
            image_path = self.image_manager.get_image_path(index)
            label_path = Path(folder) / f"{Path(image_path).stem}.txt"
            if label_path.exists():
                candidates.append((image_path, str(label_path)))

        if not candidates:
            QMessageBox.information(self, tr("info"), tr("import_batch_no_match"))
            return

        overwrite_count = sum(
            1 for image_path, _ in candidates
            if self.annotation_manager.has_annotations(image_path)
        )

        message = (
            tr("import_batch_confirmation")
            .replace("{match}", str(len(candidates)))
            .replace("{total}", str(self.image_manager.get_image_count()))
        )
        if overwrite_count:
            message += "\n\n⚠ " + tr("import_batch_overwrite_warning").replace(
                "{count}", str(overwrite_count)
            )

        reply = QMessageBox.question(
            self, tr("import_batch_title"), message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        progress = QProgressDialog(
            tr("import_batch_in_progress"), tr("cancel"), 0, len(candidates), self
        )
        progress.setWindowTitle(tr("import_batch_title"))
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        imported_files = 0
        imported_total = 0
        skipped_total = 0

        for index, (image_path, label_path) in enumerate(candidates):
            if progress.wasCanceled():
                break

            image_info = self.image_manager.get_image_info(image_path)
            if image_info is None:
                skipped_total += 1
                progress.setValue(index + 1)
                continue

            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError as e:
                self.logger.error(f"读取标签文件失败 {label_path}: {e}")
                progress.setValue(index + 1)
                continue

            old_state = copy.deepcopy(
                self.annotation_manager.get_annotations(image_path)
            )
            width, height = image_info
            imported, skipped = self.annotation_manager.import_from_yolo_format(
                image_path, lines, width, height
            )

            if imported == 0:
                # 空文件/全部无效：保留原标注
                self.annotation_manager.save_annotations(image_path, old_state)
                skipped_total += skipped
            else:
                imported_files += 1
                imported_total += imported
                skipped_total += skipped

            progress.setValue(index + 1)
            QApplication.processEvents()

        progress.close()

        # 批量导入直接落盘、不可撤销：清空撤销栈，避免旧命令把导入结果覆盖回去
        self.annotation_manager._undo_stack.clear()
        self.annotation_manager._redo_stack.clear()
        self.update_undo_redo_actions()
        self._refresh_annotation_views()

        QMessageBox.information(
            self, tr("import_batch_title"),
            tr("import_batch_result")
                .replace("{files}", str(imported_files))
                .replace("{imported}", str(imported_total))
                .replace("{skipped}", str(skipped_total))
        )
        self.update_status(
            tr("import_complete")
                .replace("{imported}", str(imported_total))
                .replace("{skipped}", str(skipped_total))
        )

    # ==================== 标注转换（框 ↔ 多边形）====================

    @staticmethod
    def _needs_conversion(annotation, target_task: str) -> bool:
        """标注类型与目标任务是否不一致（需要转换）"""
        if target_task == "segment":
            return isinstance(annotation, Annotation)
        return isinstance(annotation, PolygonAnnotation)

    @staticmethod
    def _convert_annotation_list(annotations, target_task: str):
        """按目标任务转换标注列表（不改动传入对象）"""
        converted = []
        for ann in annotations:
            if target_task == "segment":
                if isinstance(ann, PolygonAnnotation):
                    converted.append(copy.deepcopy(ann))
                else:
                    converted.append(PolygonAnnotation.from_box(ann))
            else:
                if isinstance(ann, PolygonAnnotation):
                    converted.append(Annotation.from_polygon(ann))
                else:
                    converted.append(copy.deepcopy(ann))
        return converted

    def _convert_annotations_for_image(
        self, image_path: str, target_task: str, undoable: bool = True
    ) -> int:
        """把单张图片的标注转换为目标类型，返回实际转换的标注数"""
        old_state = copy.deepcopy(self.annotation_manager.get_annotations(image_path))
        changed = sum(1 for ann in old_state if self._needs_conversion(ann, target_task))
        if changed == 0:
            return 0

        new_state = self._convert_annotation_list(old_state, target_task)
        if undoable:
            command = EditAnnotationsCommand(
                self.annotation_manager, image_path, old_state, new_state
            )
            self.annotation_manager.execute_command(command)
            if image_path == self.current_image_path:
                self.load_annotations_for_current_image()
            self.update_undo_redo_actions()
        else:
            self.annotation_manager.save_annotations(image_path, new_state)
        return changed

    def convert_current_to_segment(self):
        """当前图片：转为分割标注（矩形 → 四点多边形，可撤销）"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        changed = self._convert_annotations_for_image(
            self.current_image_path, "segment", undoable=True
        )
        if changed == 0:
            QMessageBox.information(self, tr("info"), tr("convert_nothing_to_do"))
            return
        self.update_status(tr("convert_complete").replace("{count}", str(changed)))

    def convert_current_to_detect(self):
        """当前图片：转为检测标注（多边形 → 外接矩形，可撤销）"""
        if not self.current_image_path:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        changed = self._convert_annotations_for_image(
            self.current_image_path, "detect", undoable=True
        )
        if changed == 0:
            QMessageBox.information(self, tr("info"), tr("convert_nothing_to_do"))
            return
        self.update_status(tr("convert_complete").replace("{count}", str(changed)))

    def convert_folder_to_segment(self):
        """整个文件夹：批量转为分割标注（不可撤销、直接保存）"""
        self._convert_folder_annotations("segment")

    def convert_folder_to_detect(self):
        """整个文件夹：批量转为检测标注（不可撤销、直接保存）"""
        self._convert_folder_annotations("detect")

    def _convert_folder_annotations(self, target_task: str):
        """文件夹批量转换：确认 + 进度，直接落盘、不可撤销"""
        paths = self.image_manager.get_all_image_paths()
        if not paths:
            QMessageBox.warning(self, tr("warning"), tr("no_image_loaded"))
            return

        affected_images = 0
        affected_annotations = 0
        for image_path in paths:
            annotations = self.annotation_manager.get_annotations(image_path)
            count = sum(
                1 for ann in annotations if self._needs_conversion(ann, target_task)
            )
            if count:
                affected_images += 1
                affected_annotations += count

        if affected_images == 0:
            QMessageBox.information(self, tr("info"), tr("convert_nothing_to_do"))
            return

        if target_task == "segment":
            message = tr("convert_folder_to_segment_confirmation")
        else:
            message = tr("convert_folder_to_detect_confirmation")
        message = (
            message
            .replace("{images}", str(affected_images))
            .replace("{annotations}", str(affected_annotations))
        )

        reply = QMessageBox.question(
            self, tr("convert_folder_title"), message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        progress = QProgressDialog(
            tr("convert_folder_in_progress"), tr("cancel"), 0, len(paths), self
        )
        progress.setWindowTitle(tr("convert_folder_title"))
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        converted_total = 0
        for index, image_path in enumerate(paths):
            if progress.wasCanceled():
                break
            converted_total += self._convert_annotations_for_image(
                image_path, target_task, undoable=False
            )
            progress.setValue(index + 1)
            QApplication.processEvents()

        progress.close()

        # 批量转换直接落盘、不可撤销：清空撤销栈
        self.annotation_manager._undo_stack.clear()
        self.annotation_manager._redo_stack.clear()
        self.update_undo_redo_actions()
        self._refresh_annotation_views()

        QMessageBox.information(
            self, tr("success"),
            tr("convert_complete").replace("{count}", str(converted_total))
        )
        self.update_status(
            tr("convert_complete").replace("{count}", str(converted_total))
        )

    # ==================== 导出临时转换确认 ====================

    def confirm_export_conversion(self) -> bool:
        """导出前确认：存在与任务模式不匹配的标注时列出数量并询问（存储数据不变）"""
        mismatch_count = 0
        for image_path in self.image_manager.get_all_image_paths():
            for ann in self.annotation_manager.get_annotations(image_path):
                if self._needs_conversion(ann, self.task_mode):
                    mismatch_count += 1

        if mismatch_count == 0:
            return True

        if self.task_mode == "segment":
            message = tr("export_convert_box_to_polygon")
        else:
            message = tr("export_convert_polygon_to_box")
        message = message.replace("{count}", str(mismatch_count))

        reply = QMessageBox.question(
            self, tr("export_convert_confirm_title"), message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        return reply == QMessageBox.Yes

    # ==================== 模型任务适配提示 ====================

    def _model_task_adaptation_hint(self) -> str:
        """模型任务类型与当前模式不一致时的适配说明；一致或未知时返回空串"""
        model_task = self.model_manager.model_task
        if not model_task or model_task == self.task_mode:
            return ""
        if self.task_mode == "segment":
            return tr("auto_convert_box_to_polygon_hint")
        return tr("auto_convert_polygon_to_box_hint")

    # ==================== 内部工具 ====================

    def _refresh_annotation_views(self):
        """标注被导入/转换流程改写后：刷新列表、统计与画布"""
        self.update_image_list()
        self.update_statistics_panel()
        self.load_annotations_for_current_image()
