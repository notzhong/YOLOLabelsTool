"""
模型管理器 - 用于加载YOLO模型进行自动标注
"""

from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger_simple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    # 显式兜底 None：保证模块属性契约在无 ultralytics 环境（CI）下同样成立，
    # 便于测试注入 FakeYOLO，也避免 NameError
    YOLO = None
    YOLO_AVAILABLE = False

from .annotation import Annotation, PolygonAnnotation


class ModelManager:
    """YOLO模型管理器"""
    
    def __init__(self):
        self.model: Optional[Any] = None
        self.model_path: Optional[str] = None
        self.confidence_threshold: float = 0.25
        self.iou_threshold: float = 0.45
        self.model_loaded: bool = False
        self.class_names: Dict[int, str] = {}
        self.model_task: Optional[str] = None  # ultralytics 模型任务类型（detect/segment/...）
        self.logger = get_logger_simple(__name__)
        
        # 在初始化时检查YOLO库是否可用
        if not YOLO_AVAILABLE:
            self.logger.warning("ultralytics 库未安装，自动标注功能将不可用")
    
    def is_available(self) -> bool:
        """检查YOLO库是否可用"""
        return YOLO_AVAILABLE
    
    def load_model(self, model_path: str) -> bool:
        """加载YOLO模型"""
        if not self.is_available():
            self.logger.error("ultralytics 库未安装，无法加载模型")
            return False
        
        try:
            model_path = str(Path(model_path).resolve())
            self.logger.info(f"加载模型: {model_path}")
            
            # 加载模型
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.model_loaded = True
            # 记录模型任务类型：ultralytics 在加载时设置 task 属性（检测/分割等）
            self.model_task = getattr(self.model, 'task', None)
            if self.model_task:
                self.logger.info(f"模型任务类型: {self.model_task}")
            
            # 获取类别名称
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = {i: name for i, name in self.model.names.items()}
                self.logger.info(f"模型包含 {len(self.class_names)} 个类别: {list(self.class_names.values())}")
            else:
                self.logger.warning("无法获取模型类别名称")
                self.class_names = {}
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            self.model_loaded = False
            return False
    
    def _process_boxes(
        self,
        boxes,
        img_width: int,
        img_height: int,
        mask_polygons: Optional[List[Optional[List[tuple]]]] = None,
    ) -> List[Dict[str, Any]]:
        """处理模型输出的 boxes，提取统一格式的检测结果列表

        mask_polygons 为与 boxes 一一对应的掩码多边形（像素坐标），
        存在时附加到对应检测结果的 'polygon_px' 字段。
        """
        detections = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            width = x2 - x1
            height = y2 - y1
            if width < 5 or height < 5:
                continue
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            detection = {
                'x': x1, 'y': y1, 'width': width, 'height': height,
                'confidence': conf, 'class_id': cls_id, 'class_name': class_name,
            }
            if mask_polygons is not None and index < len(mask_polygons):
                polygon_px = mask_polygons[index]
                if polygon_px:
                    detection['polygon_px'] = polygon_px
            detections.append(detection)
        return detections

    def _extract_mask_polygons(
        self, result, img_width: int, img_height: int
    ) -> Optional[List[Optional[List[tuple]]]]:
        """提取分割掩码的多边形顶点（result.masks.xyn 为归一化坐标）

        返回与 boxes 一一对应的像素坐标多边形列表（无效掩码处为 None）；
        非分割模型结果（无 masks）返回 None。
        """
        masks = getattr(result, 'masks', None)
        if masks is None:
            return None
        xyn = getattr(masks, 'xyn', None)
        if xyn is None:
            return None

        polygons: List[Optional[List[tuple]]] = []
        for poly in xyn:
            points = []
            try:
                for x, y in poly:
                    px = min(max(float(x) * img_width, 0.0), float(img_width))
                    py = min(max(float(y) * img_height, 0.0), float(img_height))
                    points.append((px, py))
            except (TypeError, ValueError, IndexError):
                points = []
            polygons.append(points if len(points) >= 3 else None)
        return polygons

    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """对图片进行推理，返回检测结果"""
        if not self.model_loaded or not self.model:
            self.logger.error("模型未加载")
            return []
        
        try:
            # 运行推理
            results = self.model(
                image_path, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                # 获取图片尺寸
                if hasattr(result, 'orig_shape'):
                    img_height, img_width = result.orig_shape
                else:
                    # 从原始图片读取尺寸
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                    else:
                        img_width, img_height = 640, 640  # 默认值

                mask_polygons = self._extract_mask_polygons(result, img_width, img_height)
                detections.extend(
                    self._process_boxes(result.boxes, img_width, img_height, mask_polygons)
                )

            self.logger.info(f"检测到 {len(detections)} 个目标")
            return detections
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return []
    
    def predict_image(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """对numpy数组格式的图片进行推理"""
        if not self.model_loaded or not self.model:
            self.logger.error("模型未加载")
            return []
        
        try:
            # 运行推理
            results = self.model(
                image_array, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                # 获取图片尺寸
                img_height, img_width = image_array.shape[:2]

                mask_polygons = self._extract_mask_polygons(result, img_width, img_height)
                detections.extend(
                    self._process_boxes(result.boxes, img_width, img_height, mask_polygons)
                )

            return detections
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model_loaded:
            return {"loaded": False, "path": None, "class_count": 0, "task": None}
        
        return {
            "loaded": True,
            "path": self.model_path,
            "class_count": len(self.class_names),
            "classes": self.class_names,
            "task": self.model_task,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold
        }
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_iou_threshold(self, threshold: float):
        """设置IoU阈值"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model_loaded

    def unload_model(self):
        """卸载模型并释放资源"""
        if self.model is not None:
            try:
                del self.model
            except Exception as e:
                self.logger.warning(f"卸载模型时出错: {e}")
            self.model = None
        self.model_path = None
        self.model_loaded = False
        self.class_names = {}
        self.model_task = None
        self.logger.info("模型已卸载")

    def convert_to_annotations(
        self, detections: List[Dict[str, Any]], task: str = "detect"
    ) -> List[Any]:
        """将检测结果转换为标注对象列表

        task="detect"：生成矩形标注（分割掩码存在时取其外接框）；
        task="segment"：生成多边形标注（无掩码的检测结果退化为 4 顶点矩形）。
        """
        annotations: List[Any] = []
        
        for detection in detections:
            class_id = detection['class_id']
            polygon_px = detection.get('polygon_px')
            has_polygon = isinstance(polygon_px, (list, tuple)) and len(polygon_px) >= 3

            if task == "segment":
                if has_polygon:
                    annotations.append(PolygonAnnotation(
                        points=[(float(x), float(y)) for x, y in polygon_px],
                        class_id=class_id,
                    ))
                else:
                    box = Annotation(
                        x=detection['x'],
                        y=detection['y'],
                        width=detection['width'],
                        height=detection['height'],
                        class_id=class_id,
                    )
                    annotations.append(PolygonAnnotation.from_box(box))
            else:
                if has_polygon:
                    polygon = PolygonAnnotation(
                        points=[(float(x), float(y)) for x, y in polygon_px],
                        class_id=class_id,
                    )
                    annotations.append(Annotation.from_polygon(polygon))
                else:
                    annotations.append(Annotation(
                        x=detection['x'],
                        y=detection['y'],
                        width=detection['width'],
                        height=detection['height'],
                        class_id=class_id  # 使用模型返回的class_id
                    ))
        
        return annotations