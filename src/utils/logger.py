"""
日志模块 - 替换项目中所有的 print 语句
增强功能：捕获程序异常报错到日志文件
"""

import os
import sys
import traceback
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
import threading


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        level: 日志级别，默认为 INFO
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果已经配置过处理器，直接返回
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 创建 logs 目录（如果不存在）
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 按日期命名的日志文件
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{date_str}.log"
    
    # 日志格式 - 增强版，包含异常追踪
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器 - 按日期滚动
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# 全局日志记录器
_logger_cache = {}


def get_logger_simple(name: str) -> logging.Logger:
    """
    简化版获取日志记录器（推荐使用）
    
    Args:
        name: 日志记录器名称
    
    Returns:
        logging.Logger: 日志记录器
    """
    if name in _logger_cache:
        return _logger_cache[name]
    
    logger = get_logger(name)
    _logger_cache[name] = logger
    return logger


# ==================== 异常捕获功能 ====================

def handle_exception(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
    """
    全局异常处理器 - 捕获未处理的异常
    
    Args:
        exc_type: 异常类型
        exc_value: 异常值
        exc_traceback: 异常追踪
    """
    # 忽略 KeyboardInterrupt 异常（用户按 Ctrl+C）
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # 获取异常记录器
    logger = get_logger_simple('exception')
    
    # 记录异常信息
    logger.critical(
        f"未捕获的异常: {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
    # 记录完整的异常追踪
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(f"异常追踪:\n{error_msg}")
    
    # 调用默认的异常处理器（程序仍会退出）
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
    """
    线程异常处理器 - 捕获线程中的未处理异常（Python 3.8+）
    
    Args:
        args: 线程异常参数
    """
    # 获取异常记录器
    logger = get_logger_simple('thread_exception')
    
    # 记录线程异常
    thread_name = args.thread.name if args.thread else "未知线程"
    logger.error(
        f"线程 {thread_name} 发生未捕获异常: {args.exc_type.__name__}: {args.exc_value}",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


def setup_exception_hook() -> None:
    """
    设置异常捕获钩子
    调用此函数以启用全局异常捕获
    """
    # 设置全局异常钩子
    sys.excepthook = handle_exception
    
    # 设置线程异常钩子（Python 3.8+）
    if hasattr(threading, 'excepthook'):
        threading.excepthook = handle_thread_exception
    
    # 记录已启用异常捕获
    logger = get_logger_simple('logger')
    logger.info("异常捕获钩子已启用")


def log_exceptions(func: Callable) -> Callable:
    """
    装饰器：自动记录函数异常
    
    Args:
        func: 要装饰的函数
    
    Returns:
        Callable: 装饰后的函数
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 获取函数所在模块的日志记录器
            module_name = func.__module__ if hasattr(func, '__module__') else 'unknown'
            logger = get_logger_simple(module_name)
            
            # 记录异常
            logger.exception(f"函数 {func.__name__} 发生异常")
            
            # 重新抛出异常
            raise
    
    return wrapper


# ==================== 测试函数 ====================

def test_logger():
    """测试日志功能"""
    logger = get_logger(__name__)
    
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")
    
    logger.info(f"日志文件已创建: logs/{datetime.now().strftime('%Y-%m-%d')}.log")


def test_exception_capture():
    """测试异常捕获功能"""
    logger = get_logger_simple(__name__)
    logger.info("开始测试异常捕获功能...")
    
    @log_exceptions
    def faulty_function():
        """会抛出异常的函数"""
        raise ValueError("这是一个测试异常")
    
    try:
        faulty_function()
    except ValueError as e:
        logger.info(f"成功捕获测试异常: {e}")
    
    logger.info("异常捕获测试完成")


if __name__ == "__main__":
    # 启用异常捕获
    setup_exception_hook()
    
    # 测试日志功能
    test_logger()
    
    # 测试异常捕获
    test_exception_capture()
    
    # 测试未捕获异常（注释掉以避免程序退出）
    # raise RuntimeError("这是一个未捕获的异常测试")
