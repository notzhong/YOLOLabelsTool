"""
文件浏览与开关联动 Mixin

从 train_dialog.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 TrainDialog 的属性与其他 Mixin 方法。
"""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QLineEdit

from src.utils.i18n import tr


class TrainBrowseMixin:
    """文件浏览 Mixin：模型/数据/输出/检查点路径选择与恢复-增量互斥开关"""


    def on_resume_toggled(self, checked: bool):
        """恢复训练复选框状态改变"""
        self.resume_path_edit.setEnabled(checked)
        self.resume_browse_btn.setEnabled(checked)

        # 恢复训练与增量训练互斥
        if checked and self.incremental_checkbox.isChecked():
            self.incremental_checkbox.setChecked(False)

    def on_incremental_toggled(self, checked: bool):
        """增量训练复选框状态改变"""
        self.incremental_path_edit.setEnabled(checked)
        self.incremental_browse_btn.setEnabled(checked)
        self.incremental_hint.setVisible(checked)

        # 恢复训练与增量训练互斥
        if checked and self.resume_checkbox.isChecked():
            self.resume_checkbox.setChecked(False)

    def on_augment_toggled(self, checked: bool):
        """数据增强复选框状态改变"""
        # 启用/禁用所有增强参数控件
        for widget in [self.mixup_spin, self.degrees_spin,
                       self.shear_spin, self.perspective_spin,
                       self.flip_up_down_spin, self.hsv_h_spin,
                       self.hsv_s_spin, self.hsv_v_spin]:
            widget.setEnabled(checked)

    def _browse_open_file(self, target_edit: QLineEdit, title_key: str, file_filter: str):
        """通用文件浏览方法"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr(title_key), self._last_browse_path, file_filter
        )
        if file_path:
            target_edit.setText(file_path)
            self._last_browse_path = str(Path(file_path).parent)

    def _browse_directory(self, target_edit: QLineEdit, title_key: str):
        """通用目录浏览方法"""
        dir_path = QFileDialog.getExistingDirectory(
            self, tr(title_key), self._last_browse_path
        )
        if dir_path:
            target_edit.setText(dir_path)
            self._last_browse_path = dir_path

    def browse_model_file(self):
        self._browse_open_file(self.model_path_edit, "browse_model_file_dialog", "PyTorch模型文件 (*.pt)")

    def browse_data_yaml(self):
        self._browse_open_file(self.data_yaml_edit, "browse_data_yaml_dialog", "YAML文件 (*.yaml *.yml)")

    def browse_output_dir(self):
        self._browse_directory(self.output_dir_edit, "browse_output_dir_dialog")

    def browse_resume_file(self):
        self._browse_open_file(self.resume_path_edit, "browse_resume_file_dialog", "PyTorch模型文件 (*.pt)")

    def browse_incremental_file(self):
        self._browse_open_file(self.incremental_path_edit, "browse_incremental_file_dialog", "PyTorch模型文件 (*.pt)")
