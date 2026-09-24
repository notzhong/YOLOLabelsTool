"""MainWindow 职责拆分的 Mixin 模块（P2 重构）"""

from .theme_language_mixin import ThemeLanguageMixin
from .panels_mixin import PanelsMixin
from .image_actions_mixin import ImageActionsMixin
from .class_actions_mixin import ClassActionsMixin
from .model_actions_mixin import ModelActionsMixin

__all__ = ["ThemeLanguageMixin", "PanelsMixin", "ImageActionsMixin",
           "ClassActionsMixin", "ModelActionsMixin"]
