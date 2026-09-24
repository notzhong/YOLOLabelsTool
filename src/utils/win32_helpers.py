"""Win32 API helpers for window picking and screen capture.

Windows 专属：在非 Windows 平台导入本模块将抛出 PlatformError，
调用方应通过 :func:`is_windows` 先行判断。
"""

import ctypes
import sys
from ctypes import wintypes


def is_windows() -> bool:
    """当前是否运行于 Windows"""
    return sys.platform == "win32"


class PlatformError(RuntimeError):
    """在非 Windows 平台调用 Win32 专属功能时抛出"""


# WinAPI constants used by window picking.
GA_ROOT = 2
MONITOR_DEFAULTTONEAREST = 2
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
# Virtual screen metrics
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


def _require_windows():
    if not is_windows():
        raise PlatformError(
            "Win32 API helpers are only available on Windows; "
            f"current platform is {sys.platform!r}"
        )


class MONITORINFOEXW(ctypes.Structure):
    # 结构体定义本身跨平台安全，仅在实例化时访问 wintypes 成员
    _fields_ = []


def _platform_guard(fn):
    """非 Windows 平台统一抛 PlatformError 的守卫装饰器"""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _require_windows()
        return fn(*args, **kwargs)

    return wrapper


if is_windows():
    class MONITORINFOEXW(ctypes.Structure):  # noqa: F811
        """MONITORINFOEXW 结构体（仅 Windows 平台定义完整字段）"""
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]


    def _configure_user32(user32) -> None:
        """Set explicit WinAPI signatures to avoid HWND truncation on 64-bit."""
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND

        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND

        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int

        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int

        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL

        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsWindow.restype = wintypes.BOOL

        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL

        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = ctypes.c_short

        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.MonitorFromWindow.restype = wintypes.HANDLE

        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user32.MonitorFromPoint.restype = wintypes.HANDLE

        user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFOEXW)]
        user32.GetMonitorInfoW.restype = wintypes.BOOL

    # Lazily-initialized user32 instance
    _user32 = None


    def get_user32():
        """Get the configured user32 instance."""
        global _user32
        if _user32 is None:
            _user32 = ctypes.windll.user32
            _configure_user32(_user32)
        return _user32


    def to_root_window(hwnd: int) -> int:
        """Promote child/control HWND to top-level window HWND."""
        if not hwnd:
            return 0
        user32 = get_user32()
        root = int(user32.GetAncestor(wintypes.HWND(hwnd), GA_ROOT) or 0)
        return root if root else hwnd


    def get_window_title(hwnd: int) -> str:
        user32 = get_user32()
        length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
        title = buf.value.strip()
        return title if title else f"HWND:{hwnd}"

else:
    @_platform_guard
    def get_user32():
        raise PlatformError("user32 is only available on Windows")


    @_platform_guard
    def to_root_window(hwnd: int) -> int:
        raise PlatformError("window handles are only available on Windows")


    @_platform_guard
    def get_window_title(hwnd: int) -> str:
        raise PlatformError("window titles are only available on Windows")
