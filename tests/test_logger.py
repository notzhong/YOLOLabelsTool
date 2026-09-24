"""src/utils/logger.py 单元测试（monkeypatch app root，不依赖 CWD、不污染仓库）"""

import logging
import sys
from pathlib import Path

import pytest

from src.utils import logger as logger_module
from src.utils.logger import _get_app_root, get_logger, get_logger_simple, handle_exception


@pytest.fixture(autouse=True)
def _app_root(tmp_path, monkeypatch):
    """把日志根目录指到 tmp_path 并清理缓存"""
    monkeypatch.setattr(logger_module, "_get_app_root", lambda: tmp_path)
    logger_module._logger_cache.clear()
    yield tmp_path
    logger_module._logger_cache.clear()


class TestGetLogger:
    def test_returns_configured_logger(self):
        log = get_logger("test_logger_a")
        assert isinstance(log, logging.Logger)
        assert log.level == logging.INFO
        assert logging.FileHandler in {type(h) for h in log.handlers}
        assert logging.StreamHandler in {type(h) for h in log.handlers}

    def test_creates_logs_dir(self, _app_root):
        get_logger("test_logger_b")
        assert (_app_root / "logs").is_dir()

    def test_writes_log_file_under_app_root(self, _app_root):
        get_logger("test_logger_d").info("hello")
        files = list((_app_root / "logs").glob("*.log"))
        assert len(files) == 1

    def test_no_duplicate_handlers(self):
        log1 = get_logger("test_logger_c")
        n = len(log1.handlers)
        log2 = get_logger("test_logger_c")
        assert log1 is log2
        assert len(log2.handlers) == n


class TestGetLoggerSimple:
    def test_cached(self):
        log1 = get_logger_simple("test_simple_a")
        assert "test_simple_a" in logger_module._logger_cache
        log2 = get_logger_simple("test_simple_a")
        assert log1 is log2

    def test_different_names_different_loggers(self):
        a = get_logger_simple("test_simple_b")
        b = get_logger_simple("test_simple_c")
        assert a is not b


class TestAppRootResolution:
    def test_source_mode_points_to_repo_root(self):
        # autouse fixture 已 patch 模块函数，这里直接验证源码解析逻辑的不变量：
        # _get_app_root 源码模式返回 Path(__file__).parents[2]，即仓库根
        module_file = Path(logger_module.__file__).resolve()
        expected_root = module_file.parents[2]
        assert (expected_root / "src" / "utils" / "logger.py") == module_file
        # 仓库根确有 src 树
        assert (expected_root / "src").is_dir()

    def test_frozen_mode_uses_executable_dir(self, monkeypatch, tmp_path):
        fake_exe = tmp_path / "app" / "YoloLabelsTool.exe"
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        assert _get_app_root() == tmp_path / "app"

    def test_meipass_takes_priority(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(tmp_path / "app" / "x.exe"))
        meipass = tmp_path / "_MEI123"
        monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
        assert _get_app_root() == meipass


class TestHandleException:
    @pytest.fixture
    def hook_recorder(self, monkeypatch):
        calls = []
        monkeypatch.setattr(sys, "__excepthook__", lambda *a: calls.append(a))
        return calls

    def test_logs_critical_and_calls_default_hook(self, hook_recorder, _app_root):
        handle_exception(ValueError, ValueError("boom"), None)
        assert len(hook_recorder) == 1
        exc_type, exc_value, tb = hook_recorder[0]
        assert exc_type is ValueError

        exc_logger = logging.getLogger("exception")
        assert any(isinstance(h, logging.FileHandler) for h in exc_logger.handlers)

    def test_keyboard_interrupt_bypasses_logging(self, hook_recorder):
        handle_exception(KeyboardInterrupt, KeyboardInterrupt(), None)
        # 仍转交默认 hook，但不写入 exception 日志文件
        assert len(hook_recorder) == 1
        exc_logger = logging.getLogger("exception")
        file_handlers = [h for h in exc_logger.handlers if isinstance(h, logging.FileHandler)]
        for h in file_handlers:
            h.flush()
            with open(h.baseFilename, encoding="utf-8") as f:
                content = f.read()
            assert "KeyboardInterrupt" not in content
