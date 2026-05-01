"""
国际化翻译管理模块
提供中英文双语支持
"""
import configparser
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QCoreApplication
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)


class TranslationManager:
    """翻译管理器类"""

    _instance = None

    def __init__(self):
        self.translations = {}
        self.current_language = "zh_CN"
        self._available_languages = ['zh_CN', 'en_US']

        # 加载所有语言翻译文件
        self.load_all_translations()

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all_translations(self):
        """从文件加载所有语言的翻译"""
        translation_dir = Path("translations")
        if not translation_dir.exists():
            logger.warning(f"翻译目录不存在: {translation_dir}")
            return

        for lang in self._available_languages:
            lang_file = translation_dir / f"{lang}.ini"
            if not lang_file.exists():
                logger.warning(f"翻译文件不存在: {lang_file}")
                continue

            try:
                config = configparser.ConfigParser()
                config.optionxform = lambda option: option
                config.read(lang_file, encoding='utf-8')

                if 'translations' not in config:
                    logger.error(f"翻译文件格式错误，缺少 [translations] 部分: {lang_file}")
                    continue

                self.translations[lang] = {}
                for key, value in config['translations'].items():
                    self.translations[lang][key] = value

                logger.info(f"成功加载翻译文件: {lang_file}, 包含 {len(self.translations[lang])} 条翻译")
            except Exception as e:
                logger.error(f"加载翻译文件失败 {lang_file}: {e}")

    def load_translation_files(self):
        """兼容旧接口：仅重载当前语言"""
        self.load_all_translations()

    def tr(self, key: str, default: Optional[str] = None) -> str:
        """翻译函数（回退链: 当前语言 → en_US → default/key）"""
        # 如果当前语言没有翻译数据，尝试加载
        if self.current_language not in self.translations or not self.translations[self.current_language]:
            self.load_translation_files()

        # 从当前语言翻译中查找
        if self.current_language in self.translations:
            lang_translations = self.translations[self.current_language]
            if key in lang_translations:
                return lang_translations[key]

        # 回退到 en_US
        if self.current_language != 'en_US' and 'en_US' in self.translations:
            if key in self.translations['en_US']:
                return self.translations['en_US'][key]

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

        return True

    def get_supported_languages(self):
        """获取支持的语言列表"""
        return ['zh_CN', 'en_US']

    def get_current_language(self):
        """获取当前语言"""
        return self.current_language

    def save_translation_file(self, language: str):
        """保存翻译文件到INI（合并现有翻译，不覆盖）"""
        translation_dir = Path("translations")
        translation_dir.mkdir(parents=True, exist_ok=True)

        lang_file = translation_dir / f"{language}.ini"
        new_translations = self.translations.get(language, {})

        try:
            # 先读取现有翻译
            config = configparser.ConfigParser()
            config.optionxform = lambda option: option
            config.read(lang_file, encoding='utf-8')

            if 'translations' not in config:
                config['translations'] = {}

            # 合并新翻译到现有翻译（覆盖同键）
            for key, value in new_translations.items():
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