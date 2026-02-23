#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 标注工具 - 主程序入口
"""

import sys
import traceback
from pathlib import Path


def setup_environment():
    """设置环境，添加src目录到Python路径"""
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    return current_dir


def init_logging():
    """初始化日志系统（仅在主进程中调用）"""
    try:
        from src.utils.logger import setup_exception_hook, get_logger_simple
        
        # 设置异常捕获钩子
        setup_exception_hook()
        
        # 获取日志记录器
        logger = get_logger_simple(__name__)
        logger.info("=" * 60)
        logger.info("YOLO Label Tool 启动")
        logger.info(f"工作目录: {Path(__file__).parent}")
        logger.info(f"Python 版本: {sys.version}")
        logger.info("=" * 60)
        
        return logger
        
    except Exception as e:
        print(f"无法初始化日志系统: {e}")
        print("程序将继续运行，但日志功能可能不可用")
        return None


def main():
    """主程序入口"""
    # 设置环境
    setup_environment()
    
    # 初始化日志（仅主进程执行）
    logger = init_logging()
    
    try:
        # 延迟导入，避免多进程重复执行模块级代码
        from PySide6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow
        
        # 如果日志初始化失败，创建一个简单的日志记录器
        if logger is None:
            class SimpleLogger:
                def info(self, msg): print(f"[INFO] {msg}")
                def warning(self, msg): print(f"[WARNING] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
                def critical(self, msg): print(f"[CRITICAL] {msg}")
            logger = SimpleLogger()
        
        app = QApplication(sys.argv)
        
        # 设置应用程序信息
        app.setApplicationName("YOLO Label Tool")
        app.setOrganizationName("YoloLabelTool")
        app.setApplicationVersion("1.0.0")
        
        # 初始化翻译管理器
        from src.utils.i18n import TranslationManager
        translation_manager = TranslationManager.instance()
        logger.info(f"翻译管理器初始化完成，当前语言: {translation_manager.get_current_language()}")
        
        # 创建并显示主窗口
        window = MainWindow()
        window.show()
        
        # 记录应用程序启动
        logger.info("主窗口创建成功，应用程序启动")
        
        # 运行应用程序
        return_code = app.exec()
        
        # 记录应用程序退出
        logger.info(f"应用程序退出，返回码: {return_code}")
        return return_code
        
    except Exception as e:
        # 捕获主函数中的异常
        error_msg = f"应用程序启动失败: {e}\n{traceback.format_exc()}"
        print(error_msg)
        
        # 尝试记录到日志
        if logger and hasattr(logger, 'critical'):
            logger.critical(error_msg)
        else:
            print(f"[CRITICAL] {error_msg}")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)