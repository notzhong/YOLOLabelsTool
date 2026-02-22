"""
标注数据结构和标注管理器
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from src.utils.logger import get_logger_simple


@dataclass
class Annotation:
    """标注数据结构"""
    x: float          # 矩形左上角x坐标
    y: float          # 矩形左上角y坐标
    width: float      # 矩形宽度
    height: float     # 矩形高度
    class_id: int     # 类别ID
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Annotation':
        """从字典创建"""
        return cls(
            x=data.get('x', 0),
            y=data.get('y', 0),
            width=data.get('width', 0),
            height=data.get('height', 0),
            class_id=data.get('class_id', 0)
        )
    
    def to_yolo_format(self, image_width: int, image_height: int) -> List[float]:
        """
        转换为YOLO格式（归一化坐标）
        格式: [class_id, x_center, y_center, width, height]
        """
        # 计算中心点坐标
        x_center = (self.x + self.width / 2) / image_width
        y_center = (self.y + self.height / 2) / image_height
        
        # 计算归一化宽度和高度
        norm_width = self.width / image_width
        norm_height = self.height / image_height
        
        return [
            self.class_id,
            x_center,
            y_center,
            norm_width,
            norm_height
        ]
    
    @classmethod
    def from_yolo_format(
        cls, 
        yolo_data: List[float], 
        image_width: int, 
        image_height: int
    ) -> 'Annotation':
        """
        从YOLO格式创建标注
        """
        if len(yolo_data) != 5:
            raise ValueError(f"YOLO格式数据长度应为5，实际为{len(yolo_data)}")
        
        class_id = int(yolo_data[0])
        x_center, y_center, norm_width, norm_height = yolo_data[1:]
        
        # 转换回像素坐标
        width = norm_width * image_width
        height = norm_height * image_height
        x = x_center * image_width - width / 2
        y = y_center * image_height - height / 2
        
        return cls(x=x, y=y, width=width, height=height, class_id=class_id)


class Command:
    """命令模式基类"""
    
    def __init__(self, annotation_manager: 'AnnotationManager', image_path: str):
        self.annotation_manager = annotation_manager
        self.image_path = image_path
        self.old_state: Optional[List[Annotation]] = None
        self.new_state: Optional[List[Annotation]] = None
    
    def execute(self):
        """执行命令"""
        pass
    
    def undo(self):
        """撤销命令"""
        pass
    
    def redo(self):
        """重做命令"""
        pass


class AddAnnotationCommand(Command):
    """添加标注命令"""
    
    def __init__(self, annotation_manager: 'AnnotationManager', image_path: str, annotation: Annotation):
        super().__init__(annotation_manager, image_path)
        self.annotation = annotation
    
    def execute(self):
        """执行添加操作"""
        # 保存当前状态
        self.old_state = self.annotation_manager.get_annotations(self.image_path).copy()
        
        # 执行添加
        annotations = self.annotation_manager.get_annotations(self.image_path)
        annotations.append(self.annotation)
        self.annotation_manager.save_annotations(self.image_path, annotations)
        
        # 保存新状态
        self.new_state = annotations.copy()
    
    def undo(self):
        """撤销添加"""
        if self.old_state is not None:
            self.annotation_manager.save_annotations(self.image_path, self.old_state)
    
    def redo(self):
        """重做添加"""
        if self.new_state is not None:
            self.annotation_manager.save_annotations(self.image_path, self.new_state)


class DeleteAnnotationCommand(Command):
    """删除标注命令"""
    
    def __init__(self, annotation_manager: 'AnnotationManager', image_path: str, annotation_index: int):
        super().__init__(annotation_manager, image_path)
        self.annotation_index = annotation_index
        self.deleted_annotation: Optional[Annotation] = None
    
    def execute(self):
        """执行删除操作"""
        # 保存当前状态
        self.old_state = self.annotation_manager.get_annotations(self.image_path).copy()
        
        # 执行删除
        annotations = self.annotation_manager.get_annotations(self.image_path)
        if 0 <= self.annotation_index < len(annotations):
            self.deleted_annotation = annotations[self.annotation_index]
            annotations.pop(self.annotation_index)
            self.annotation_manager.save_annotations(self.image_path, annotations)
        
        # 保存新状态
        self.new_state = annotations.copy()
    
    def undo(self):
        """撤销删除"""
        if self.old_state is not None:
            self.annotation_manager.save_annotations(self.image_path, self.old_state)
    
    def redo(self):
        """重做删除"""
        if self.new_state is not None:
            self.annotation_manager.save_annotations(self.image_path, self.new_state)


class AnnotationManager:
    """标注管理器"""
    
    def __init__(self):
        self._annotations: Dict[str, List[Annotation]] = {}
        self._annotation_dir = "annotations"
        
        # 历史记录栈
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        
        # 日志记录器
        self.logger = get_logger_simple(__name__)
        
        # 创建标注目录
        Path(self._annotation_dir).mkdir(exist_ok=True)
    
    def get_annotation_path(self, image_path: str) -> str:
        """获取标注文件路径"""
        image_name = Path(image_path).stem
        return os.path.join(self._annotation_dir, f"{image_name}.json")
    
    def save_annotations(self, image_path: str, annotations: List[Annotation]):
        """保存标注到文件"""
        if not annotations:
            return
        
        annotation_path = self.get_annotation_path(image_path)
        
        # 转换为可序列化的字典列表
        annotations_data = [ann.to_dict() for ann in annotations]
        
        # 保存为JSON
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        
        # 同时更新内存中的缓存
        self._annotations[image_path] = annotations
    
    def load_annotations(self, image_path: str) -> List[Annotation]:
        """从文件加载标注"""
        annotation_path = self.get_annotation_path(image_path)
        
        if not os.path.exists(annotation_path):
            return []
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotations_data = json.load(f)
            
            annotations = [Annotation.from_dict(data) for data in annotations_data]
            self._annotations[image_path] = annotations
            return annotations
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"加载标注文件失败: {e}")
            return []
    
    def get_annotations(self, image_path: str) -> List[Annotation]:
        """获取指定图片的标注（优先从缓存中获取）"""
        if image_path in self._annotations:
            return self._annotations[image_path]
        
        return self.load_annotations(image_path)
    
    def has_annotations(self, image_path: str) -> bool:
        """检查是否有标注"""
        annotation_path = self.get_annotation_path(image_path)
        return os.path.exists(annotation_path)
    
    def add_annotation(self, image_path: str, annotation: Annotation):
        """添加单个标注"""
        annotations = self.get_annotations(image_path)
        annotations.append(annotation)
        self.save_annotations(image_path, annotations)
    
    def delete_annotation(self, image_path: str, annotation_index: int):
        """删除指定索引的标注"""
        annotations = self.get_annotations(image_path)
        if 0 <= annotation_index < len(annotations):
            annotations.pop(annotation_index)
            self.save_annotations(image_path, annotations)
    
    def clear_annotations(self, image_path: str):
        """清除指定图片的所有标注"""
        self._annotations[image_path] = []
        
        # 删除标注文件
        annotation_path = self.get_annotation_path(image_path)
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
    
    def get_all_annotations(self) -> Dict[str, List[Annotation]]:
        """获取所有图片的标注"""
        return self._annotations.copy()
    
    def export_to_yolo_format(self, image_path: str, image_width: int, image_height: int) -> List[str]:
        """导出为YOLO格式文本行"""
        annotations = self.get_annotations(image_path)
        yolo_lines = []
        
        for ann in annotations:
            yolo_data = ann.to_yolo_format(image_width, image_height)
            line = " ".join(f"{val:.6f}" for val in yolo_data)
            yolo_lines.append(line)
        
        return yolo_lines
    
    def import_from_yolo_format(
        self, 
        image_path: str, 
        yolo_lines: List[str], 
        image_width: int, 
        image_height: int
    ):
        """从YOLO格式导入标注"""
        annotations = []
        
        for line in yolo_lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                yolo_data = [float(part) for part in parts]
                annotation = Annotation.from_yolo_format(yolo_data, image_width, image_height)
                annotations.append(annotation)
            except (ValueError, IndexError) as e:
                self.logger.warning(f"解析YOLO格式行失败: {line}, 错误: {e}")
        
        self.save_annotations(image_path, annotations)
    
    def get_statistics(self) -> Dict:
        """获取标注统计信息"""
        total_images = len(self._annotations)
        total_annotations = 0
        class_counts = {}
        
        for image_path, annotations in self._annotations.items():
            total_annotations += len(annotations)
            for ann in annotations:
                class_counts[ann.class_id] = class_counts.get(ann.class_id, 0) + 1
        
        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "class_counts": class_counts
        }
    
    # ==================== 撤销/重做功能 ====================
    
    def execute_command(self, command: Command):
        """执行命令"""
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()  # 执行新命令时清空重做栈
    
    def undo(self) -> bool:
        """撤销操作"""
        if not self._undo_stack:
            return False
        
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        return True
    
    def redo(self) -> bool:
        """重做操作"""
        if not self._redo_stack:
            return False
        
        command = self._redo_stack.pop()
        command.redo()
        self._undo_stack.append(command)
        return True
    
    def can_undo(self) -> bool:
        """检查是否可以撤销"""
        return len(self._undo_stack) > 0
    
    def can_redo(self) -> bool:
        """检查是否可以重做"""
        return len(self._redo_stack) > 0
    
    def clear_history(self):
        """清空历史记录"""
        self._undo_stack.clear()
        self._redo_stack.clear()
