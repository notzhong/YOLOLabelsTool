"""
标注数据结构和标注管理器
"""

import os
import copy
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path

from src.utils.logger import _get_app_root, get_logger_simple


@dataclass
class Annotation:
    """标注数据结构"""
    x: float          # 矩形左上角x坐标
    y: float          # 矩形左上角y坐标
    width: float      # 矩形宽度
    height: float     # 矩形高度
    class_id: int     # 类别ID
    
    def to_dict(self) -> Dict:
        """转换为字典（写入 type 字段区分框/多边形；无 type 的旧文件按框处理）"""
        data = asdict(self)
        data["type"] = "box"
        return data
    
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

    @classmethod
    def from_polygon(cls, polygon: 'PolygonAnnotation') -> 'Annotation':
        """由多边形生成其外接矩形标注"""
        x, y, width, height = polygon.bounding_box()
        return cls(x=x, y=y, width=width, height=height, class_id=polygon.class_id)


@dataclass
class PolygonAnnotation:
    """多边形标注数据结构（YOLO-seg 分割任务）

    points 为像素坐标顶点列表（至少 3 个顶点），序列化时以
    {"type": "polygon", "points": [[x, y], ...], "class_id": n} 存储。
    """

    points: List[Tuple[float, float]]
    class_id: int

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "type": "polygon",
            "points": [[float(x), float(y)] for x, y in self.points],
            "class_id": self.class_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolygonAnnotation':
        """从字典创建"""
        points = []
        for point in data.get('points', []) or []:
            points.append((float(point[0]), float(point[1])))
        return cls(points=points, class_id=int(data.get('class_id', 0)))

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """外接矩形 (x, y, width, height)"""
        if not self.points:
            return 0.0, 0.0, 0.0, 0.0
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, max_x - min_x, max_y - min_y

    def to_yolo_seg_format(self, image_width: int, image_height: int) -> List[float]:
        """转换为 YOLO 分割格式（归一化坐标）

        格式: [class_id, x1, y1, x2, y2, ...]
        """
        data = [self.class_id]
        for x, y in self.points:
            data.append(x / image_width)
            data.append(y / image_height)
        return data

    @classmethod
    def from_yolo_seg_format(
        cls,
        yolo_data: List[float],
        image_width: int,
        image_height: int
    ) -> 'PolygonAnnotation':
        """从 YOLO 分割格式创建（字段数须为 1+2k 且 k>=3）"""
        if len(yolo_data) < 7 or len(yolo_data) % 2 == 0:
            raise ValueError(
                f"YOLO分割格式数据长度应为1+2k(k>=3)，实际为{len(yolo_data)}"
            )

        class_id = int(yolo_data[0])
        coords = yolo_data[1:]
        points = []
        for i in range(0, len(coords), 2):
            points.append((coords[i] * image_width, coords[i + 1] * image_height))

        return cls(points=points, class_id=class_id)

    @classmethod
    def from_box(cls, box: Annotation) -> 'PolygonAnnotation':
        """矩形转换为 4 顶点矩形多边形（左上 → 右上 → 右下 → 左下）"""
        x, y, width, height = box.x, box.y, box.width, box.height
        return cls(
            points=[(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
            class_id=box.class_id,
        )


AnnotationLike = Union[Annotation, PolygonAnnotation]


def annotation_from_dict(data: Dict) -> AnnotationLike:
    """由字典创建标注对象：type=polygon 为多边形，其余（含旧文件无 type）按框处理"""
    if data.get('type') == 'polygon':
        return PolygonAnnotation.from_dict(data)
    return Annotation.from_dict(data)


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
    
    def __init__(self, annotation_manager: 'AnnotationManager', image_path: str, annotation: AnnotationLike):
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


class EditAnnotationsCommand(Command):
    """编辑标注命令（顶点拖动、整体平移、顶点增删、改类、格式转换）

    以修改前/后的完整标注列表快照记录一次编辑手势，支持框与多边形混合列表。
    快照在构造时深拷贝，避免后续画布上的原地修改污染命令状态。
    """

    def __init__(
        self,
        annotation_manager: 'AnnotationManager',
        image_path: str,
        old_state: List[AnnotationLike],
        new_state: List[AnnotationLike],
    ):
        super().__init__(annotation_manager, image_path)
        self.old_state = copy.deepcopy(old_state)
        self.new_state = copy.deepcopy(new_state)

    def execute(self):
        """执行编辑操作（写入新状态）"""
        if self.new_state is not None:
            self.annotation_manager.save_annotations(
                self.image_path, copy.deepcopy(self.new_state)
            )

    def undo(self):
        """撤销编辑"""
        if self.old_state is not None:
            self.annotation_manager.save_annotations(
                self.image_path, copy.deepcopy(self.old_state)
            )

    def redo(self):
        """重做编辑"""
        self.execute()


class AnnotationManager:
    """标注管理器"""

    # 撤销栈最大深度，防止内存无限增长
    MAX_UNDO_SIZE = 100

    def __init__(self):
        self._annotations: Dict[str, List[AnnotationLike]] = {}
        # 标注目录锚定应用根目录：随 CWD 漂移会导致跨目录启动时标注"丢失"
        self._annotation_dir = str(_get_app_root() / "annotations")

        # 历史记录栈
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        
        # 日志记录器
        self.logger = get_logger_simple(__name__)
        
        # 创建标注目录
        Path(self._annotation_dir).mkdir(exist_ok=True)
    
    def _legacy_annotation_path(self, image_path: str) -> str:
        """获取旧版标注文件路径（仅使用文件名）"""
        if not image_path:
            raise ValueError(f"image_path 不能为空: {image_path!r}")
        image_name = Path(image_path).stem
        return os.path.join(self._annotation_dir, f"{image_name}.json")

    def get_annotation_path(self, image_path: str) -> str:
        """获取标注文件路径

        image_path 为空（未加载图片）时抛出 ValueError：Path(None) 的
        TypeError 只会暴露 pathlib 内部细节，看不出是"没加载图片"这个真实原因。
        """
        if not image_path:
            raise ValueError(f"image_path 不能为空: {image_path!r}")
        image_name = Path(image_path).stem
        # 基于全路径生成短 Hash，避免同名文件冲突
        digest = hashlib.md5(image_path.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return os.path.join(self._annotation_dir, f"{image_name}_{digest}.json")
    
    def save_annotations(self, image_path: str, annotations: List[AnnotationLike]):
        """保存标注到文件

        写路径对空 image_path 快速失败：静默返回会让调用方以为标注已落盘，
        实际上是丢数据；抛 ValueError 则问题在源头就暴露。
        """
        if not image_path:
            raise ValueError(
                f"拒绝保存标注：image_path 为空，{len(annotations)} 条标注会被丢弃"
            )
        annotation_path = self.get_annotation_path(image_path)

        if not annotations:
            # 空标注时删除对应文件，避免残留旧标注
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
            # 同步更新内存缓存
            self._annotations[image_path] = []
            return
        
        # 转换为可序列化的字典列表
        annotations_data = [ann.to_dict() for ann in annotations]
        
        # 保存为JSON
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        
        # 同时更新内存中的缓存
        self._annotations[image_path] = annotations
    
    def load_annotations(self, image_path: str) -> List[AnnotationLike]:
        """从文件加载标注（未加载图片时返回空列表，不抛异常）"""
        if not image_path:
            return []
        annotation_path = self.get_annotation_path(image_path)
        
        if not os.path.exists(annotation_path):
            legacy_path = self._legacy_annotation_path(image_path)
            if not os.path.exists(legacy_path):
                return []
            annotation_path = legacy_path
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotations_data = json.load(f)
            
            annotations = [annotation_from_dict(data) for data in annotations_data]
            self._annotations[image_path] = annotations
            if annotation_path == self._legacy_annotation_path(image_path):
                try:
                    self.save_annotations(image_path, annotations)
                except Exception:
                    pass
            return annotations
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"加载标注文件失败: {e}")
            return []
    
    def get_annotations(self, image_path: str) -> List[AnnotationLike]:
        """获取指定图片的标注（优先从缓存中获取）

        未加载图片时返回空列表而非抛异常——读路径不该因为"没有图"而崩溃，
        拦截该情况的职责在 UI 层。注意不写入缓存，避免 None 污染
        _annotations 进而虚增 get_statistics 的图片数。
        """
        if not image_path:
            return []
        if image_path in self._annotations:
            return self._annotations[image_path]
        
        return self.load_annotations(image_path)
    
    def has_annotations(self, image_path: str) -> bool:
        """检查是否有标注（未加载图片时返回 False）"""
        if not image_path:
            return False
        annotation_path = self.get_annotation_path(image_path)
        if os.path.exists(annotation_path):
            return True
        legacy_path = self._legacy_annotation_path(image_path)
        return os.path.exists(legacy_path)
    
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
        if not image_path:
            raise ValueError(f"拒绝清除标注：image_path 为空: {image_path!r}")
        self._annotations[image_path] = []
        
        # 删除新/旧标注文件
        paths_to_remove = [
            self.get_annotation_path(image_path),
            self._legacy_annotation_path(image_path),
        ]
        for annotation_path in paths_to_remove:
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
    
    def get_all_annotations(self) -> Dict[str, List[AnnotationLike]]:
        """获取所有图片的标注"""
        return self._annotations.copy()
    
    def export_to_yolo_format(
        self, image_path: str, image_width: int, image_height: int, task: str = "detect"
    ) -> List[str]:
        """导出为YOLO格式文本行（task="segment" 输出多边形分割格式）"""
        # 局部导入避免 utils 层与 core 层的模块级循环依赖
        from src.utils.yolo_exporter import annotation_to_yolo_lines

        annotations = self.get_annotations(image_path)
        return annotation_to_yolo_lines(
            annotations, image_width, image_height, task=task
        )
    
    def import_from_yolo_format(
        self, 
        image_path: str, 
        yolo_lines: List[str], 
        image_width: int, 
        image_height: int
    ) -> Tuple[int, int]:
        """从YOLO格式导入标注

        解析规则：5 个字段按检测框解析；1+2k（k>=3）个字段按多边形解析；
        其余行视为无效并跳过。返回 (成功导入数, 跳过行数)。
        """
        annotations: List[AnnotationLike] = []
        skipped = 0
        
        for line in yolo_lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split()
                field_count = len(parts)
                if field_count == 5:
                    yolo_data = [float(part) for part in parts]
                    annotations.append(Annotation.from_yolo_format(yolo_data, image_width, image_height))
                elif field_count >= 7 and field_count % 2 == 1:
                    yolo_data = [float(part) for part in parts]
                    annotations.append(
                        PolygonAnnotation.from_yolo_seg_format(yolo_data, image_width, image_height)
                    )
                else:
                    skipped += 1
                    self.logger.warning(f"跳过字段数不合法的YOLO行（{field_count} 个字段）: {line}")
            except (ValueError, IndexError) as e:
                skipped += 1
                self.logger.warning(f"解析YOLO格式行失败: {line}, 错误: {e}")
        
        self.save_annotations(image_path, annotations)
        return len(annotations), skipped
    
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
        self._push_undo(command)

    def push_command(self, command: Command):
        """将已应用的命令压入撤销栈

        用于"命令效果已自行写入缓存"的场景（如画布上的多边形编辑手势：
        画布对象与缓存共享同一批标注对象，直接 save_annotations 即可保持
        对象一致性，只有撤销/重做快照需要由命令持有）。
        """
        self._push_undo(command)

    def _push_undo(self, command: Command):
        """压入撤销栈并维护栈上限/重做栈"""
        self._undo_stack.append(command)
        # 限制撤销栈大小，防止内存无限增长
        if len(self._undo_stack) > self.MAX_UNDO_SIZE:
            self._undo_stack.pop(0)
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
