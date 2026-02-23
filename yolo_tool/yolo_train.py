"""
YOLO模型训练器 - 支持异步训练、配置保存和恢复训练
"""
import os
import json
import threading
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Union
import traceback

from ultralytics import YOLO

from PySide6.QtCore import QObject, Signal


class ProgressCallback:
    """YOLO训练进度回调，用于实时更新训练进度"""
    
    def __init__(self, trainer):
        self.trainer = trainer
    
    def on_train_epoch_end(self, trainer):
        """在每个训练epoch结束时调用"""
        try:
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
            
            # 发送进度更新信号
            self.trainer.progress_updated.emit(epoch, metrics)
            self.trainer.log_message.emit(f"Epoch {epoch}/{trainer.epochs} 完成")
        except Exception as e:
            self.trainer.log_message.emit(f"回调错误: {str(e)[:100]}")
    
    def on_train_batch_end(self, trainer):
        """在每个训练批次结束时调用，用于获取更实时的损失数据"""
        try:
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
            
            # 如果有损失数据，发送更新
            if metrics:
                self.trainer.progress_updated.emit(epoch, metrics)
        except Exception as e:
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
        self._callbacks_to_add: list = []  # 存储需要通过train_args传递的回调
        
    def load_config(self, config_path: Union[str, Path]) -> bool:
        """从JSON文件加载训练配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.log_message.emit(f"已加载训练配置: {config_path}")
            return True
        except Exception as e:
            self.log_message.emit(f"加载配置失败: {e}")
            return False
    
    def save_config(self, config_path: Union[str, Path]) -> bool:
        """保存训练配置到JSON文件"""
        try:
            # 确保目录存在
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.log_message.emit(f"已保存训练配置: {config_path}")
            return True
        except Exception as e:
            self.log_message.emit(f"保存配置失败: {e}")
            return False
    
    def setup(self, config: Dict[str, Any]) -> bool:
        """设置训练参数"""
        self.config = config.copy()
        
        # 检查必要的参数
        required_keys = ['model_path', 'data_yaml']
        for key in required_keys:
            if key not in config:
                self.log_message.emit(f"缺少必要参数: {key}")
                return False
        
        # 检查文件是否存在
        if not Path(config['model_path']).exists():
            self.log_message.emit(f"模型文件不存在: {config['model_path']}")
            return False
        
        if not Path(config['data_yaml']).exists():
            self.log_message.emit(f"数据集配置文件不存在: {config['data_yaml']}")
            return False
        
        # 设置默认输出目录
        if 'output_dir' not in config:
            self.config['output_dir'] = str(Path.cwd() / "runs" / "train")
        
        self.log_message.emit("训练参数设置完成")
        return True
    
    def _training_worker(self):
        """训练工作线程"""
        try:
            self.is_training = True
            self.should_stop = False
            self.training_started.emit()
            
            # 加载模型
            model_path = self.config.get('model_path')
            resume = self.config.get('resume', False)
            
            self.log_message.emit(f"加载模型: {model_path}")
            self.model = YOLO(model_path)
            
            # 添加进度回调 - 使用健壮的注册方式
            progress_callback = ProgressCallback(self)
            self._register_callbacks(progress_callback)
            
            # 准备训练参数
            train_args = {
                'data': self.config.get('data_yaml'),
                'epochs': self.config.get('epochs', 300),
                'imgsz': self.config.get('imgsz', 640),
                'batch': self.config.get('batch', 16),
                'device': self.config.get('device', 0),
                'workers': self.config.get('workers', 8),
                'optimizer': self.config.get('optimizer', 'AdamW'),
                'lr0': self.config.get('lr0', 0.0008),
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
                'resume': resume,
            }
            
            # 移除None值
            train_args = {k: v for k, v in train_args.items() if v is not None}
            
            self.log_message.emit(f"开始训练，共 {train_args['epochs']} 个epochs")
            self.log_message.emit(f"输出目录: {train_args['project']}")
            
            # 执行训练
            results = self.model.train(**train_args)
            
            if self.should_stop:
                self.log_message.emit("训练被用户停止")
                self.training_stopped.emit()
                success = False
                message = "训练被用户停止"
                
                # 清理内存
                self._cleanup_resources()
            else:
                # 获取最佳权重路径
                best_model_path = Path(train_args['project']) / train_args['name'] / 'weights' / 'best.pt'
                if best_model_path.exists():
                    message = f"训练完成！最佳模型保存于: {best_model_path}"
                else:
                    message = "训练完成"
                
                self.log_message.emit("训练完成")
                success = True
                
                # 清理内存
                self._cleanup_resources()
            
            self.is_training = False
            self.training_finished.emit(success, message)
            
        except Exception as e:
            error_msg = f"训练过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg)
            self.is_training = False
            
            # 清理内存
            self._cleanup_resources()
            
            self.training_finished.emit(False, error_msg)
    
    def start_training(self):
        """开始训练（异步）"""
        if self.is_training:
            self.log_message.emit("训练已在运行中")
            return False
        
        if not self.config:
            self.log_message.emit("请先设置训练参数")
            return False
        
        # 创建并启动训练线程
        self.train_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.train_thread.start()
        
        self.log_message.emit("训练线程已启动")
        return True
    
    def stop_training(self):
        """停止训练"""
        if not self.is_training:
            self.log_message.emit("没有正在运行的训练")
            return False
        
        self.should_stop = True
        self.log_message.emit("正在停止训练...")
        return True
    
    def resume_training(self, checkpoint_path: Union[str, Path]) -> bool:
        """从检查点恢复训练"""
        if not Path(checkpoint_path).exists():
            self.log_message.emit(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        self.config['resume'] = checkpoint_path
        self.log_message.emit(f"将恢复训练从: {checkpoint_path}")
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认训练配置"""
        return {
            'model_path': '',
            'data_yaml': '',
            'output_dir': str(Path.cwd() / "runs" / "train"),
            'run_name': 'train',
            'epochs': 300,
            'imgsz': 640,
            'batch': 16,
            'device': 0,
            'workers': 8,
            'optimizer': 'AdamW',
            'lr0': 0.0008,
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
        }
    
    def _register_callbacks(self, callback_instance):
        """注册训练回调，兼容不同版本的ultralytics API"""
        try:
            # 检查add_callback方法是否存在
            if not hasattr(self.model, 'add_callback'):
                self.log_message.emit("⚠ 模型没有add_callback方法，将通过训练参数传递回调")
                return
            
            # 方法1：尝试两个参数的版本（事件名称, 回调函数）
            try:
                # 使用lambda包装器确保正确的参数传递
                self.model.add_callback('on_train_epoch_end', 
                    lambda trainer: callback_instance.on_train_epoch_end(trainer))
                self.model.add_callback('on_train_batch_end', 
                    lambda trainer: callback_instance.on_train_batch_end(trainer))
                self.log_message.emit("✓ 使用 add_callback(event, func) 注册回调")
                return
            except TypeError as e:
                # 如果两个参数版本失败，尝试一个参数的版本（回调对象）
                if "missing 1 required positional argument" in str(e):
                    # 错误提示缺少一个参数，说明可能是单参数版本
                    try:
                        self.model.add_callback(callback_instance)
                        self.log_message.emit("✓ 使用 add_callback(callback_instance) 注册回调")
                        return
                    except Exception as inner_e:
                        self.log_message.emit(f"单参数版本也失败: {inner_e}")
                else:
                    self.log_message.emit(f"两参数版本失败: {e}")
            
            # 方法2：通过callbacks参数（训练时传递）
            if hasattr(self.model, 'callbacks'):
                # 尝试添加到callbacks列表
                if isinstance(self.model.callbacks, list):
                    self.model.callbacks.append(callback_instance)
                    self.log_message.emit("✓ 添加到 model.callbacks 列表")
                    return
            
            # 方法3：通过训练参数传递（在train_args中添加callbacks）
            self.log_message.emit("⚠ 无法直接注册回调，将通过训练参数传递")
            
        except Exception as e:
            self.log_message.emit(f"注册回调失败（不影响训练）: {e}")
    
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
                    self.log_message.emit("CUDA缓存已清理")
            except ImportError:
                pass
            
            # 强制垃圾回收
            gc.collect()
            
            self.log_message.emit("资源清理完成")
        except Exception as e:
            self.log_message.emit(f"清理资源时出错: {e}")
    
    def cleanup(self):
        """清理所有资源，释放内存"""
        self._cleanup_resources()
        
        # 清理配置数据
        self.config.clear()
        
        # 停止计时器（如果有）
        if self.train_thread and self.train_thread.is_alive():
            # 线程是daemon线程，会随主线程结束
            pass
            
        self.log_message.emit("训练器资源完全清理完成")


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