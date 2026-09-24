"""TrainDialog 职责拆分的 Mixin 模块（P2 重构）"""

from .actions_mixin import TrainActionsMixin
from .browse_mixin import TrainBrowseMixin
from .config_mixin import TrainConfigMixin
from .tabs_mixin import TrainTabsMixin

__all__ = ["TrainActionsMixin", "TrainBrowseMixin", "TrainConfigMixin",
           "TrainTabsMixin"]
