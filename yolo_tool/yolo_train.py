"""
YOLO模型训练器 - 支持异步训练、配置保存和恢复训练
"""
import os
import sys
import json
import logging
import threading
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Union
import traceback

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from PySide6.QtCore import QObject, Signal

from src.utils.logger import get_logger_simple

_logger = get_logger_simple(__name__)


class ProgressCallback:
    """YOLO训练进度回调，用于实时更新训练进度"""
    
    def __init__(self, trainer):
        self.trainer = trainer
    
    def on_train_epoch_end(self, trainer):
        """在每个训练epoch结束时调用，同时检查是否需要停止训练"""
        try:
            # 检查是否需要停止训练
            if self.trainer.should_stop:
                trainer.stop = True
                self.trainer.log_message.emit("用户请求停止，将在当前epoch完成后停止训练")
                return

            # 获取当前epoch和指标
            epoch = trainer.epoch
            
            # 提取关键指标 - ultralytics results 是 dict_keys 对象
            metrics = {}
            
            # 尝试从训练器获取指标
            if hasattr(trainer, 'results'):
                results = trainer.results
                if results:
                    # ultralytics results 是 dict_keys 对象，需要转换为字典
                    if hasattr(results, 'keys'):
                        # 如果是字典
                        for key in results.keys():
                            value = getattr(results, key, None)
                            if isinstance(value, (int, float)):
                                metrics[key] = value
                    else:
                        # 尝试直接访问常见指标
                        common_metrics = ['loss', 'val_loss', 'precision', 'recall', 'mAP50', 'mAP50-95']
                        for metric in common_metrics:
                            if hasattr(results, metric):
                                value = getattr(results, metric, None)
                                if isinstance(value, (int, float)):
                                    metrics[metric] = value
            
            # 发送进度更新信号（ultralytics epoch 是 0-indexed，显示时 +1）
            self.trainer.progress_updated.emit(epoch + 1, metrics)
            self.trainer.log_message.emit(f"Epoch {epoch + 1}/{trainer.epochs} 完成")
        except Exception as e:
            self.trainer.log_message.emit(f"回调错误: {str(e)[:100]}")
    
    def on_train_batch_end(self, trainer):
        """在每个训练批次结束时调用，用于获取更实时的损失数据，同时检查是否需要停止训练"""
        try:
            # 检查是否需要停止训练
            if self.trainer.should_stop:
                trainer.stop = True
                return

            # 获取当前epoch
            epoch = trainer.epoch
            
            # 尝试获取损失值
            metrics = {}
            if hasattr(trainer, 'loss_names') and hasattr(trainer, 'loss_items'):
                # 提取损失项
                loss_names = trainer.loss_names
                loss_items = trainer.loss_items
                
                if loss_names and loss_items:
                    for i, name in enumerate(loss_names):
                        if i < len(loss_items):
                            metrics[name] = float(loss_items[i])
            
            # 如果有损失数据，发送更新（ultralytics epoch 是 0-indexed，显示时 +1）
            if metrics:
                self.trainer.progress_updated.emit(epoch + 1, metrics)
        except Exception:
            # 忽略批次更新错误，不影响主流程
            pass


class YOLOTrainer(QObject):
    """YOLO模型训练器，支持异步训练和进度跟踪"""

    # 信号定义
    progress_updated = Signal(int, dict)      # epoch, metrics
    log_message = Signal(str)                 # 日志消息
    training_started = Signal()               # 训练开始
    training_finished = Signal(bool, str)     # 成功, 消息
    training_stopped = Signal()               # 训练被停止

    def __init__(self):
        super().__init__()
        self.model: Optional[YOLO] = None
        self.is_training: bool = False
        self.should_stop: bool = False
        self.config: Dict[str, Any] = {}
        self.train_thread: Optional[threading.Thread] = None
        self._callbacks_to_add: list = []

        # log_message 信号一旦触发就自动写日志文件
        self.log_message.connect(lambda msg: _logger.info(msg))

    def _log(self, msg: str, level: str = "info"):
        """同时发送 UI 信号和写入日志文件（带级别控制）"""
        self.log_message.emit(msg)
        getattr(_logger, level)(msg)
        
    def load_config(self, config_path: Union[str, Path]) -> bool:
        """从JSON文件加载训练配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self._log(f"已加载训练配置: {config_path}")
            return True
        except Exception as e:
            self._log(f"加载配置失败: {e}")
            return False
    
    def save_config(self, config_path: Union[str, Path]) -> bool:
        """保存训练配置到JSON文件"""
        try:
            # 确保目录存在
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self._log(f"已保存训练配置: {config_path}")
            return True
        except Exception as e:
            self._log(f"保存配置失败: {e}")
            return False
    
    def setup(self, config: Dict[str, Any]) -> bool:
        """设置训练参数"""
        self.config = config.copy()

        # 确定实际使用的模型路径
        resume = self.config.get('resume', False)
        incremental = self.config.get('incremental', False)
        is_resume = isinstance(resume, str) and resume
        is_incremental = isinstance(incremental, str) and incremental

        if is_resume:
            model_file = resume
        elif is_incremental:
            model_file = incremental
        else:
            model_file = self.config.get('model_path', '')

        # 检查必要的参数
        if not is_resume and not is_incremental:
            if 'model_path' not in config:
                self._log(f"缺少必要参数: model_path")
                return False
            if not Path(config['model_path']).exists():
                self._log(f"模型文件不存在: {config['model_path']}")
                return False

        if 'data_yaml' not in config:
            self._log(f"缺少必要参数: data_yaml")
            return False

        # 检查文件是否存在
        if not Path(model_file).exists():
            self._log(f"模型文件不存在: {model_file}")
            return False

        if not Path(config['data_yaml']).exists():
            self._log(f"数据集配置文件不存在: {config['data_yaml']}")
            return False

        # 设置默认输出目录
        if 'output_dir' not in self.config:
            self.config['output_dir'] = str(Path.cwd() / "runs" / "train")

        self._log("训练参数设置完成")
        return True
    
    def _training_worker(self):
        """训练工作线程"""
        # GUI 环境下 sys.stdout 可能为 None，导致 ultralytics TQDM 崩溃
        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w', encoding='utf-8')

        # 确保 stdout/stderr 使用 UTF-8 编码，避免 GBK 无法编码 ✅ 等字符
        for _s in (sys.stdout, sys.stderr):
            if hasattr(_s, 'reconfigure'):
                try:
                    _s.reconfigure(encoding='utf-8', errors='replace')
                except Exception:
                    pass

        # 修复已有的日志处理器：移除 None stream 的处理器，或将其重定向到 devnull
        for logger_name in list(logging.root.manager.loggerDict):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    if handler.stream is None:
                        logger.removeHandler(handler)

        try:
            self.is_training = True
            self.should_stop = False
            self.training_started.emit()

            # Windows 下 multiprocessing spawn 模式可导致 DataLoader 子进程持有
            # 的文件句柄与 torch.save 冲突，引发 "I/O operation on closed file"
            # 限制 workers 为安全上限 2，同时不超过 batch size
            is_windows = sys.platform == 'win32'
            workers = self.config.get('workers', 0)
            if is_windows and workers > 2:
                safe_workers = min(2, self.config.get('batch', 4))
                self._log(
                    f"⚠ Windows 下 workers={workers} 可能引发 I/O 错误, 已自动降至 {safe_workers}"
                )
                self.config['workers'] = safe_workers

            # 加载模型
            # 优先级: resume > incremental > pretrained model_path
            resume = self.config.get('resume', False)
            incremental = self.config.get('incremental', False)
            is_resume = isinstance(resume, str) and resume
            is_incremental = isinstance(incremental, str) and incremental

            if is_resume:
                model_path = resume
            elif is_incremental:
                model_path = incremental
            else:
                model_path = self.config.get('model_path')

            self._log(f"加载模型: {model_path}")
            self.model = YOLO(model_path)

            # 清理模型中可能携带的旧版/非兼容参数（例如某些自定义checkpoint参数）
            # 增量训练需要激进清理，因为旧 checkpoint 的 data/project/nc 等与新训练冲突
            self._sanitize_model_overrides(aggressive=is_incremental)

            if is_incremental:
                self._log("增量训练模式：已加载训练过的模型权重，将使用全新优化器在新数据集上训练")

            # 添加进度回调 - 使用健壮的注册方式
            progress_callback = ProgressCallback(self)
            self._register_callbacks(progress_callback)

            # 准备训练参数（包含所有市场最优参数）
            train_args = {
                'data': self.config.get('data_yaml'),
                'epochs': self.config.get('epochs', 300),
                'imgsz': self.config.get('imgsz', 640),
                'batch': self.config.get('batch', 4),
                'device': self.config.get('device', 0),
                'workers': self.config.get('workers', 4),
                'optimizer': self.config.get('optimizer', 'AdamW'),
                'lr0': self.config.get('lr0', 0.01),
                'lrf': self.config.get('lrf', 0.01),
                'cos_lr': self.config.get('cos_lr', True),
                'close_mosaic': self.config.get('close_mosaic', 40),
                'patience': self.config.get('patience', 40),
                'rect': self.config.get('rect', False),
                'cache': self.config.get('cache', True),
                'augment': self.config.get('augment', True),
                'amp': self.config.get('amp', True),
                'plots': self.config.get('plots', True),
                'verbose': self.config.get('verbose', True),
                'mixup': self.config.get('mixup', 0.0),
                'degrees': self.config.get('degrees', 0.0),
                'shear': self.config.get('shear', 0.0),
                'perspective': self.config.get('perspective', 0.0),
                'flipud': self.config.get('flipud', 0.0),
                'project': self.config.get('output_dir'),
                'name': self.config.get('run_name', 'train'),
                'exist_ok': True,  # 允许覆盖现有运行
                'resume': is_resume,
                # 新增优化器参数
                'weight_decay': self.config.get('weight_decay', 0.0005),
                'momentum': self.config.get('momentum', 0.937),
                # 新增学习率预热参数
                'warmup_epochs': self.config.get('warmup_epochs', 3.0),
                'warmup_momentum': self.config.get('warmup_momentum', 0.8),
                'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
                # 新增损失权重参数
                'box': self.config.get('box', 7.5),
                'cls': self.config.get('cls', 0.5),
                'dfl': self.config.get('dfl', 1.5),
                # 随机擦除增强（替代已移除的 label_smoothing）
                'erasing': self.config.get('erasing', 0.4),
                # 新增HSV增强参数
                'hsv_h': self.config.get('hsv_h', 0.015),
                'hsv_s': self.config.get('hsv_s', 0.7),
                'hsv_v': self.config.get('hsv_v', 0.4),
                # 新增其他增强参数
                'fliplr': self.config.get('fliplr', 0.5),
                'mosaic': self.config.get('mosaic', 1.0),
                'copy_paste': self.config.get('copy_paste', 0.0),
                # 新增正则化参数
                'dropout': self.config.get('dropout', 0.0),
                # 随机种子
                'seed': self.config.get('seed', 0),
            }
            
            # 移除None值
            train_args = {k: v for k, v in train_args.items() if v is not None}

            # 仅保留当前ultralytics版本支持的参数，避免版本差异导致训练失败
            train_args = self._filter_supported_train_args(train_args)
            
            self._log(f"开始训练，共 {train_args['epochs']} 个epochs")
            self._log(f"输出目录: {train_args['project']}")
            
            # 执行训练
            results = self.model.train(**train_args)

            if self.should_stop:
                self._log("训练被用户停止")
                self.training_stopped.emit()

                # 清理内存
                self._cleanup_resources()
                self.is_training = False
                return  # 提前返回，避免触发 training_finished 覆盖停止状态

            # 获取最佳权重路径
            best_model_path = Path(train_args['project']) / train_args['name'] / 'weights' / 'best.pt'
            if best_model_path.exists():
                message = f"训练完成！最佳模型保存于: {best_model_path}"
            else:
                message = "训练完成"

            self._log("训练完成")
            success = True

            # 清理内存
            self._cleanup_resources()

            self.is_training = False
            self.training_finished.emit(success, message)
            
        except Exception as e:
            error_msg = f"训练过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg)
            self.is_training = False
            
            # 清理内存
            self._cleanup_resources()
            
            self.training_finished.emit(False, error_msg)

    def _sanitize_model_overrides(self, aggressive: bool = False):
        """清理模型内置overrides中的不兼容参数，避免影响train参数校验。

        Args:
            aggressive: True 时额外清理上次训练遗留的 data/project/name/classes
                        等字段，用于增量训练场景。
        """
        try:
            if not self.model or not hasattr(self.model, 'overrides'):
                return

            overrides = getattr(self.model, 'overrides', None)
            if not isinstance(overrides, dict):
                return

            valid_keys = set(DEFAULT_CFG_DICT.keys())
            removed_keys = [k for k in list(overrides.keys()) if k not in valid_keys]

            if aggressive:
                # 增量训练时必须清除旧训练的参数，让 train() 使用新传入的值
                aggressive_remove = {'data', 'project', 'name', 'nc', 'classes', 'batch',
                                     'epochs', 'imgsz', 'device', 'workers', 'cache',
                                     'optimizer', 'lr0', 'lrf', 'cos_lr', 'patience',
                                     'close_mosaic', 'rect', 'augment', 'amp', 'plots',
                                     'verbose', 'mixup', 'degrees', 'shear', 'perspective',
                                     'flipud', 'fliplr', 'mosaic', 'hsv_h', 'hsv_s', 'hsv_v',
                                     'weight_decay', 'momentum', 'warmup_epochs',
                                     'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl',
                                     'erasing', 'dropout', 'seed', 'copy_paste', 'resume'}
                for k in aggressive_remove:
                    overrides.pop(k, None)

            for k in removed_keys:
                overrides.pop(k, None)

            if removed_keys or (aggressive and aggressive_remove):
                self._log(
                    f"已清理模型旧训练参数，避免与新训练冲突"
                )
        except Exception as e:
            self._log(f"清理模型overrides失败（忽略）: {e}")

    def _filter_supported_train_args(self, train_args: Dict[str, Any]) -> Dict[str, Any]:
        """过滤并返回当前ultralytics版本支持的训练参数。"""
        valid_keys = set(DEFAULT_CFG_DICT.keys())
        # 常用运行参数一般也在DEFAULT_CFG_DICT中，这里为稳妥保留一组关键键
        valid_keys.update({'data', 'project', 'name', 'exist_ok', 'resume'})

        filtered_args = {}
        removed_keys = []

        for key, value in train_args.items():
            if key in valid_keys:
                filtered_args[key] = value
            else:
                removed_keys.append(key)

        if removed_keys:
            self._log(
                f"检测到当前版本不支持的训练参数，已忽略: {', '.join(sorted(removed_keys))}"
            )

        return filtered_args
    
    def start_training(self):
        """开始训练（异步）"""
        if self.is_training:
            self._log("训练已在运行中")
            return False
        
        if not self.config:
            self._log("请先设置训练参数")
            return False
        
        # 创建并启动训练线程
        self.train_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.train_thread.start()
        
        self._log("训练线程已启动")
        return True
    
    def stop_training(self):
        """停止训练"""
        if not self.is_training:
            self._log("没有正在运行的训练")
            return False
        
        self.should_stop = True
        self._log("正在停止训练...")
        return True
    
    def resume_training(self, checkpoint_path: Union[str, Path]) -> bool:
        """从检查点恢复训练"""
        if not Path(checkpoint_path).exists():
            self._log(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        self.config['resume'] = checkpoint_path
        self._log(f"将恢复训练从: {checkpoint_path}")
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认训练配置（市场最优参数）"""
        return {
            'model_path': '',
            'data_yaml': '',
            'output_dir': str(Path.cwd() / "runs" / "train"),
            'run_name': 'train',
            'epochs': 300,
            'imgsz': 640,
            'batch': 4,
            'device': 0,
            'workers': 4,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'cos_lr': True,
            'close_mosaic': 40,
            'patience': 40,
            'rect': False,
            'cache': True,
            'augment': True,
            'amp': True,
            'plots': True,
            'verbose': True,
            'mixup': 0.0,
            'degrees': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'resume': False,
            'incremental': False,
            # 新增优化器参数
            'weight_decay': 0.0005,
            'momentum': 0.937,
            # 新增学习率预热参数
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            # 新增损失权重参数
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            # 随机擦除增强（替代已移除的 label_smoothing）
            'erasing': 0.4,
            # 新增HSV增强参数
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            # 新增其他增强参数
            'fliplr': 0.5,
            'mosaic': 1.0,
            'copy_paste': 0.0,
            # 新增正则化参数
            'dropout': 0.0,
            # 随机种子
            'seed': 0,
        }
    
    def _register_callbacks(self, callback_instance):
        """注册训练回调，兼容不同版本的ultralytics API"""
        try:
            # 检查add_callback方法是否存在
            if not hasattr(self.model, 'add_callback'):
                self._log("⚠ 模型没有add_callback方法，将通过训练参数传递回调")
                return
            
            # 方法1：尝试两个参数的版本（事件名称, 回调函数）
            try:
                # 使用lambda包装器确保正确的参数传递
                self.model.add_callback('on_train_epoch_end', 
                    lambda trainer: callback_instance.on_train_epoch_end(trainer))
                self.model.add_callback('on_train_batch_end', 
                    lambda trainer: callback_instance.on_train_batch_end(trainer))
                self._log("✓ 使用 add_callback(event, func) 注册回调")
                return
            except TypeError as e:
                # 如果两个参数版本失败，尝试一个参数的版本（回调对象）
                if "missing 1 required positional argument" in str(e):
                    # 错误提示缺少一个参数，说明可能是单参数版本
                    try:
                        self.model.add_callback(callback_instance)
                        self._log("✓ 使用 add_callback(callback_instance) 注册回调")
                        return
                    except Exception as inner_e:
                        self._log(f"单参数版本也失败: {inner_e}")
                else:
                    self._log(f"两参数版本失败: {e}")
            
            # 方法2：通过callbacks参数（训练时传递）
            if hasattr(self.model, 'callbacks'):
                # 尝试添加到callbacks列表
                if isinstance(self.model.callbacks, list):
                    self.model.callbacks.append(callback_instance)
                    self._log("✓ 添加到 model.callbacks 列表")
                    return
            
            # 方法3：通过训练参数传递（在train_args中添加callbacks）
            self._log("⚠ 无法直接注册回调，将通过训练参数传递")
            
        except Exception as e:
            self._log(f"注册回调失败（不影响训练）: {e}")
    
    def _cleanup_resources(self):
        """清理训练资源，释放内存"""
        try:
            # 清理模型
            if self.model is not None:
                # 尝试清理YOLO模型内部资源
                if hasattr(self.model, 'trainer'):
                    # 清理训练器
                    trainer = self.model.trainer
                    if trainer is not None:
                        # 清理数据加载器
                        if hasattr(trainer, 'train_loader'):
                            trainer.train_loader = None
                        if hasattr(trainer, 'val_loader'):
                            trainer.val_loader = None
                        # 清理优化器
                        if hasattr(trainer, 'optimizer'):
                            trainer.optimizer = None
                        # 清理调度器
                        if hasattr(trainer, 'scheduler'):
                            trainer.scheduler = None
                
                # 显式删除模型引用
                del self.model
                self.model = None
            
            # 清理CUDA缓存（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self._log("CUDA缓存已清理")
            except ImportError:
                pass
            
            # 强制垃圾回收
            gc.collect()
            
            self._log("资源清理完成")
        except Exception as e:
            self._log(f"清理资源时出错: {e}")
    
    def cleanup(self):
        """清理所有资源，释放内存"""
        self._cleanup_resources()
        
        # 清理配置数据
        self.config.clear()
        
        # 停止计时器（如果有）
        if self.train_thread and self.train_thread.is_alive():
            # 线程是daemon线程，会随主线程结束
            pass
            
        self._log("训练器资源完全清理完成")


# 保留原函数以保持向后兼容性
def train_model(model_path="yolo26m.pt", data_yaml="dnf.yaml", **kwargs):
    """原训练函数的兼容版本"""
    trainer = YOLOTrainer()
    config = trainer.get_default_config()
    config.update({
        'model_path': model_path,
        'data_yaml': data_yaml,
    })
    config.update(kwargs)
    
    if trainer.setup(config):
        # 注意：这个版本是同步的，会阻塞
        trainer._training_worker()