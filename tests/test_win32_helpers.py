"""src/utils/win32_helpers.py 单元测试

Linux 上验证平台守卫行为；Windows 分支的真实 WinAPI 调用只能在 CI 的
windows runner 上覆盖（见 .github/workflows/ci.yml 的平台矩阵）。
"""

import sys

import pytest

from src.utils import win32_helpers as wh
from src.utils.win32_helpers import (
    GA_ROOT,
    MONITORINFOEXW,
    PlatformError,
    get_user32,
    get_window_title,
    is_windows,
    to_root_window,
)


class TestPlatformDetection:
    def test_is_windows_matches_sys_platform(self):
        assert is_windows() == (sys.platform == "win32")

    def test_constants_exposed(self):
        assert GA_ROOT == 2


@pytest.mark.skipif(sys.platform == "win32", reason="守卫行为仅在非 Windows 平台可测")
class TestNonWindowsGuards:
    def test_get_user32_raises_platform_error(self):
        with pytest.raises(PlatformError, match="only available on Windows"):
            get_user32()

    def test_to_root_window_raises_platform_error(self):
        with pytest.raises(PlatformError):
            to_root_window(12345)

    def test_get_window_title_raises_platform_error(self):
        with pytest.raises(PlatformError):
            get_window_title(12345)

    def test_monitorinfo_is_empty_struct_on_linux(self):
        # 非 Windows 下仅提供空壳结构体，保证模块可导入
        assert MONITORINFOEXW._fields_ == []

    def test_importable_without_windows(self):
        # 模块导入本身不应有任何 Windows 专属副作用
        assert callable(wh.get_user32)
