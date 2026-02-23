"""
国际化翻译管理模块
提供中英文双语支持
"""
import configparser
import json
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtCore import QCoreApplication, QLocale, QTranslator, QLibraryInfo
from PySide6.QtWidgets import QApplication
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)


class TranslationManager:
    """翻译管理器类"""
    
    _instance = None
    
    # 最小后备翻译（仅用于无法加载文件时）
    _fallback_translations = {
        "zh_CN": {
            "ok": "确定",
            "cancel": "取消",
            "yes": "是",
            "no": "否",
            "save": "保存",
            "load": "加载",
            "add": "添加",
            "edit": "编辑",
            "delete": "删除",
            "close": "关闭",
            "error": "错误",
            "success": "成功",
        },
        "en_US": {
            "ok": "OK",
            "cancel": "Cancel",
            "yes": "Yes",
            "no": "No",
            "save": "Save",
            "load": "Load",
            "add": "Add",
            "edit": "Edit",
            "delete": "Delete",
            "close": "Close",
            "error": "Error",
            "success": "Success",
        }
    }
    
    def __init__(self):
        self.translations = {}
        self.current_language = "zh_CN"
        self.translator = QTranslator()
        
        # 初始化翻译
        self._init_translations()
        
        # 尝试加载外部翻译文件
        self.load_translation_files()
    
    def _init_translations(self):
        """初始化翻译字典结构"""
        self.translations = {}
        for lang in self._fallback_translations.keys():
            self.translations[lang] = {}
    
    @classmethod
    def instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_translation_files(self):
        """从文件加载翻译"""
        translation_dir = Path("translations")
        if not translation_dir.exists():
            logger.warning(f"翻译目录不存在: {translation_dir}")
            # 使用后备翻译
            self.translations = self._fallback_translations.copy()
            return
        
        # 尝试加载当前语言的翻译文件
        lang_file = translation_dir / f"{self.current_language}.ini"
        if not lang_file.exists():
            logger.warning(f"翻译文件不存在: {lang_file}")
            # 使用后备翻译
            if self.current_language in self._fallback_translations:
                self.translations[self.current_language] = self._fallback_translations[self.current_language].copy()
            return
        
        try:
            config = configparser.ConfigParser()
            # 读取时保持键的大小写
            config.optionxform = lambda option: option
            config.read(lang_file, encoding='utf-8')
            
            if 'translations' not in config:
                logger.error(f"翻译文件格式错误，缺少 [translations] 部分: {lang_file}")
                return
            
            # 清空当前语言的翻译
            if self.current_language not in self.translations:
                self.translations[self.current_language] = {}
            
            # 加载翻译
            for key, value in config['translations'].items():
                self.translations[self.current_language][key] = value
            
            logger.info(f"成功加载翻译文件: {lang_file}, 包含 {len(self.translations[self.current_language])} 条翻译")
            
        except Exception as e:
            logger.error(f"加载翻译文件失败: {e}")
            # 使用后备翻译
            if self.current_language in self._fallback_translations:
                self.translations[self.current_language] = self._fallback_translations[self.current_language].copy()
    
    def tr(self, key: str, default: Optional[str] = None) -> str:
        """翻译函数"""
        # 如果当前语言没有翻译数据，尝试加载
        if self.current_language not in self.translations or not self.translations[self.current_language]:
            self.load_translation_files()
        
        # 从当前语言翻译中查找
        if self.current_language in self.translations:
            lang_translations = self.translations[self.current_language]
            if key in lang_translations:
                return lang_translations[key]
        
        # 回退到后备翻译
        if self.current_language in self._fallback_translations:
            fallback_translations = self._fallback_translations[self.current_language]
            if key in fallback_translations:
                return fallback_translations[key]
        
        # 返回默认值或键本身
        return default or key
    
    def switch_language(self, language: str):
        """切换语言"""
        if language not in ['zh_CN', 'en_US']:
            logger.warning(f"不支持的语言: {language}")
            return False
        
        if language == self.current_language:
            return True
        
        self.current_language = language
        
        # 重新加载翻译文件
        self.load_translation_files()
        
        # 更新Qt翻译系统
        app = QApplication.instance()
        if app:
            # 移除旧的翻译器
            app.removeTranslator(self.translator)
            
            # 创建新的翻译器
            self.translator = QTranslator()
            
            # 尝试加载Qt标准库的翻译
            qt_translator = QTranslator()
            if qt_translator.load(QLocale(language), "qt", "_", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
                app.installTranslator(qt_translator)
            
            # 尝试加载Qt基础库的翻译
            qtbase_translator = QTranslator()
            if qtbase_translator.load(QLocale(language), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
                app.installTranslator(qtbase_translator)
            
            # 安装我们的翻译器（虽然我们使用字典，但安装一个空翻译器以触发重翻译）
            app.installTranslator(self.translator)
        
        return True
    
    def get_supported_languages(self):
        """获取支持的语言列表"""
        return ['zh_CN', 'en_US']
    
    def get_current_language(self):
        """获取当前语言"""
        return self.current_language
    
    def save_translation_file(self, language: str):
        """保存翻译文件到INI"""
        translation_dir = Path("translations")
        translation_dir.mkdir(parents=True, exist_ok=True)
        
        lang_file = translation_dir / f"{language}.ini"
        translations = self.translations.get(language, {})
        
        try:
            config = configparser.ConfigParser()
            config['translations'] = {}
            
            # 按字母顺序排序
            sorted_items = sorted(translations.items(), key=lambda x: x[0])
            
            for key, value in sorted_items:
                config['translations'][key] = value
            
            with open(lang_file, 'w', encoding='utf-8') as f:
                config.write(f)
            
            logger.info(f"翻译文件已保存: {lang_file}")
            return True
        except Exception as e:
            logger.error(f"保存翻译文件失败: {e}")
            return False


# 全局翻译函数，方便使用
def tr(key: str, default: Optional[str] = None) -> str:
    """全局翻译函数"""
    return TranslationManager.instance().tr(key, default)


# 快捷方式
T = tr