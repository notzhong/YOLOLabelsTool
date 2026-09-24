"""
主题与语言 Mixin：QSS 主题应用/切换、语言切换与 UI 文本刷新

从 main_window.py 拆分（P2 重构）：方法体保持原样，仅按职责归类。
通过 self 访问 MainWindow 的属性与其他 Mixin 方法。
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QLabel,
    QMessageBox,
)

from src.utils.i18n import tr

class ThemeLanguageMixin:
    """主题与语言 Mixin：QSS 主题应用/切换、语言切换与 UI 文本刷新"""



    def load_qss_style(self):
        """加载QSS样式文件"""
        self.logger.info(f"加载QSS样式，当前主题: {self.current_theme}")
        self.apply_theme(self.current_theme)

    def apply_theme(self, theme_name: str):
        """应用指定主题"""
        if theme_name == "dark":
            qss_path = Path("qss/dark_theme.qss")
        elif theme_name == "light":
            qss_path = Path("qss/light_theme.qss")
        elif theme_name == "colorful":
            qss_path = Path("qss/colorful_theme.qss")
        elif theme_name == "eyecare":
            qss_path = Path("qss/eyecare_theme.qss")
        else:
            self.logger.warning(f"未知主题: {theme_name}, 使用默认主题")
            return
        
        if qss_path.exists():
            try:
                with open(qss_path, 'r', encoding='utf-8') as f:
                    qss_content = f.read()
                self.setStyleSheet(qss_content)
                self.current_theme = theme_name
                
                # 根据主题更新标题标签颜色
                self.update_title_colors_for_theme(theme_name)
                # 更新主题菜单选中标记（菜单存在时）
                if hasattr(self, 'action_dark_theme'):
                    self.action_dark_theme.setChecked(theme_name == "dark")
                    self.action_light_theme.setChecked(theme_name == "light")
                    self.action_colorful_theme.setChecked(theme_name == "colorful")
                    self.action_eyecare_theme.setChecked(theme_name == "eyecare")
                
                self.logger.info(f"已应用主题: {theme_name}")
            except Exception as e:
                self.logger.error(f"加载主题样式文件失败: {e}, 将使用Qt默认样式")
        else:
            self.logger.warning(f"主题文件不存在: {qss_path}, 将使用Qt默认样式")

    def update_title_colors_for_theme(self, theme_name: str):
        """根据主题更新标题标签颜色"""
        if theme_name == "light":
            # 白天主题：黑色标题，深灰色状态
            title_style = "font-weight: bold; font-size: 14px; color: #333333;"
            status_style = "color: #777777; font-size: 12px;"
        else:
            # 黑夜/炫彩/护眼主题：白色/暖色标题，灰色状态
            title_style = "font-weight: bold; font-size: 14px; color: #ffffff;"
            status_style = "color: #aaaaaa; font-size: 12px;"
        
        # 查找并更新所有标题标签
        for widget in self.findChildren(QLabel):
            current_style = widget.styleSheet()
            if "font-weight: bold; font-size: 14px; color:" in current_style:
                widget.setStyleSheet(title_style)
            elif "color: #aaaaaa; font-size: 12px;" in current_style:
                widget.setStyleSheet(status_style)

    def switch_to_dark_theme(self):
        """切换到黑夜主题"""
        self.apply_theme("dark")
        self.save_settings()

    def switch_to_light_theme(self):
        """切换到白天主题"""
        self.apply_theme("light")
        self.save_settings()

    def switch_to_colorful_theme(self):
        """切换到炫彩主题"""
        self.apply_theme("colorful")
        self.save_settings()

    def switch_to_eyecare_theme(self):
        """切换到护眼主题"""
        self.apply_theme("eyecare")
        self.save_settings()

    def switch_language(self, language: str):
        """切换语言"""
        from src.utils.i18n import TranslationManager, tr
        
        self.logger.info(f"开始切换语言到: {language}")
        translation_manager = TranslationManager.instance()
        success = translation_manager.switch_language(language)
        
        if success:
            self.logger.info(f"语言切换成功，当前语言: {translation_manager.get_current_language()}")
            # 测试翻译是否立即生效
            self.logger.info(f"测试翻译 'open_folder': {tr('open_folder')}")
            self.logger.info(f"测试翻译 'file': {tr('file')}")
            # 更新所有UI文本
            self.update_ui_texts()
            self.update_status(tr("language_switched").replace("{language}", language))
            QMessageBox.information(self, tr("success"), tr("language_switched").replace("{language}", language))
            self.logger.info(f"UI文本已更新")
        else:
            self.logger.error(f"语言切换失败: {language}")
            QMessageBox.warning(self, tr("warning"), tr("language_switch_failed").replace("{language}", language))

    def update_ui_texts(self):
        """更新UI文本（语言切换后重新设置所有文本）"""
        
        # 窗口标题
        self.setWindowTitle(tr("yolo_label_tool"))
        
        # 更新菜单文本
        self.update_menu_texts()
        
        # 更新按钮文本
        self.update_button_texts()
        
        # 更新面板标题
        self.update_panel_titles()
        
        # 更新状态栏
        self.status_label.setText(tr("ready"))
        
        # 更新其他UI元素
        self.update_other_ui_elements()

    def update_menu_texts(self):
        """更新菜单文本"""

        self.file_menu.setTitle(tr("file"))
        self.edit_menu.setTitle(tr("edit"))
        self.view_menu.setTitle(tr("view"))
        self.class_menu.setTitle(tr("classes"))
        self.theme_menu.setTitle(tr("theme"))
        self.model_menu.setTitle(tr("model"))
        self.annotate_menu.setTitle(tr("annotate"))
        self.language_menu.setTitle(tr("language"))

        self.action_open_folder.setText(tr("open_folder"))
        self.action_save.setText(tr("save_annotations"))
        self.action_export.setText(tr("export_yolo_format"))
        self.action_exit.setText(tr("exit"))

        self.action_undo.setText(tr("undo"))
        self.action_redo.setText(tr("redo"))
        self.action_delete.setText(tr("delete_selected"))

        self.action_zoom_in.setText(tr("zoom_in"))
        self.action_zoom_out.setText(tr("zoom_out"))
        self.action_fit.setText(tr("fit_to_window"))

        self.action_load_model.setText(tr("load_model"))
        self.action_model_info.setText(tr("model_info"))
        self.action_train_model.setText(tr("train_model"))
        self.action_export_model.setText(tr("export_model"))
        self.action_auto_annotate.setText(tr("auto_annotate"))
        self.action_batch_auto_annotate.setText(tr("batch_auto_annotate"))

        self.action_load_yaml.setText(tr("load_yaml"))
        self.action_save_yaml.setText(tr("save_yaml"))

        self.action_dark_theme.setText(tr("dark_theme"))
        self.action_light_theme.setText(tr("light_theme"))
        self.action_colorful_theme.setText(tr("colorful_theme"))
        self.action_eyecare_theme.setText(tr("eyecare_theme"))

        self.action_chinese.setText(tr("chinese"))
        self.action_english.setText(tr("english"))

    def update_button_texts(self):
        """更新按钮文本"""

        self.btn_load_folder.setText(tr("load_folder"))
        self.btn_prev.setText(tr("previous_image"))
        self.btn_next.setText(tr("next_image"))

        self.btn_fit.setText(tr("fit_to_window"))
        self.btn_zoom_in.setText(tr("zoom_in"))
        self.btn_zoom_out.setText(tr("zoom_out"))
        self.btn_reset.setText(tr("reset"))

        self.btn_add_class.setText(tr("add"))
        self.btn_edit_class.setText(tr("edit"))
        self.btn_delete_class.setText(tr("delete"))
        self.btn_clear_classes.setText(tr("clear_all_classes"))

        self.btn_delete_annotation.setText(tr("delete_selected_annotation"))
        self.btn_clear_all.setText(tr("clear_all_annotations"))

        self.btn_export_yolo.setText(tr("export_yolo"))
        self.btn_export_split.setText(tr("export_dataset_split"))

    def update_panel_titles(self):
        """更新面板标题"""

        self.left_panel_title_label.setText(tr("image_list"))
        self.center_panel_title_label.setText(tr("image_annotation"))
        self.right_panel_title_label.setText(tr("class_management"))

    def update_other_ui_elements(self):
        """更新其他UI元素"""

        self.class_group.setTitle(tr("annotation_classes"))
        self.annotation_group.setTitle(tr("annotation_operations"))
        self.export_group.setTitle(tr("data_export"))

        self.stats_panel.update_language()
        self.model_info_panel.update_language(self.model_manager)

        if self.image_manager.get_image_count() == 0:
            self.stats_label.setText(tr("no_image_loaded"))

        if not self.current_image_path:
            self.image_info_label.setText(tr("no_image_loaded"))
