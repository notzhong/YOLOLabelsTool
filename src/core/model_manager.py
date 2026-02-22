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
    YOLO_AVAILABLE = False

from .annotation import Annotation


class ModelManager:
    """YOLO模型管理器"""
    
    def __init__(self):
        self.model: Optional[Any] = None
        self.model_path: Optional[str] = None
        self.confidence_threshold: float = 0.25
        self.iou_threshold: float = 0.45
        self.model_loaded: bool = False
        self.class_names: Dict[int, str] = {}
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
                
                boxes = result.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 确保坐标在图片范围内
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 确保框大小合理
                    if width < 5 or height < 5:
                        continue
                    
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    # 获取类别名称
                    class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append({
                        'x': x1,
                        'y': y1,
                        'width': width,
                        'height': height,
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': class_name
                    })
            
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
                
                boxes = result.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 确保坐标在图片范围内
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 确保框大小合理
                    if width < 5 or height < 5:
                        continue
                    
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    # 获取类别名称
                    class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append({
                        'x': x1,
                        'y': y1,
                        'width': width,
                        'height': height,
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': class_name
                    })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model_loaded:
            return {"loaded": False, "path": None, "class_count": 0}
        
        return {
            "loaded": True,
            "path": self.model_path,
            "class_count": len(self.class_names),
            "classes": self.class_names,
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
    
    def convert_to_annotations(self, detections: List[Dict[str, Any]]) -> List[Annotation]:
        """将检测结果转换为Annotation对象列表"""
        annotations = []
        
        for detection in detections:
            annotation = Annotation(
                x=detection['x'],
                y=detection['y'],
                width=detection['width'],
                height=detection['height'],
                class_id=detection['class_id']  # 使用模型返回的class_id
            )
            annotations.append(annotation)
        
        return annotations