"""
图片管理器
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger_simple


class ImageManager:
    """图片管理器"""
    
    # 支持的图片格式
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    
    def __init__(self):
        self._image_paths: List[str] = []
        self._current_folder: Optional[str] = None
        self._image_cache: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
        self._max_cache_size = 50  # 最大缓存图片数
        self.logger = get_logger_simple(__name__)
        
    def load_folder(self, folder_path: str) -> bool:
        """加载文件夹中的所有图片"""
        if not os.path.isdir(folder_path):
            return False
        
        self._current_folder = folder_path
        self._image_paths.clear()
        self._image_cache.clear()
        
        # 扫描所有支持的图片文件
        for ext in self.SUPPORTED_EXTENSIONS:
            pattern = f"*{ext}"
            for file_path in Path(folder_path).glob(pattern):
                self._image_paths.append(str(file_path))
            
            # 扫描大写扩展名
            pattern = f"*{ext.upper()}"
            for file_path in Path(folder_path).glob(pattern):
                self._image_paths.append(str(file_path))
        
        # 去重并排序
        self._image_paths = sorted(list(set(self._image_paths)))
        
        return len(self._image_paths) > 0
    
    def get_image_path(self, index: int) -> Optional[str]:
        """获取指定索引的图片路径"""
        if 0 <= index < len(self._image_paths):
            return self._image_paths[index]
        return None
    
    def get_image_count(self) -> int:
        """获取图片数量"""
        return len(self._image_paths)
    
    def get_image_info(self, image_path: str) -> Optional[Tuple[int, int]]:
        """获取图片尺寸信息"""
        try:
            if image_path in self._image_cache:
                _, size = self._image_cache[image_path]
                return size
            
            # 使用PIL获取图片尺寸（更快，不加载完整图片）
            with Image.open(image_path) as img:
                size = img.size  # (width, height)
                return size
        except Exception as e:
            self.logger.error(f"获取图片信息失败: {image_path}, 错误: {e}")
            return None
    
    def load_image(self, image_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """加载图片为numpy数组"""
        if not os.path.exists(image_path):
            return None
        
        # 检查缓存
        if use_cache and image_path in self._image_cache:
            img_array, _ = self._image_cache[image_path]
            return img_array.copy()
        
        try:
            # 使用OpenCV加载图片
            img_array = cv2.imread(image_path)
            if img_array is None:
                # 尝试使用PIL作为备选
                pil_img = Image.open(image_path)
                img_array = np.array(pil_img)
                # 如果是RGBA，转换为RGB
                if img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif len(img_array.shape) == 2:  # 灰度图
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif img_array.shape[2] == 3:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if img_array is not None:
                # 获取图片尺寸
                height, width = img_array.shape[:2]
                
                # 更新缓存
                if use_cache:
                    self._update_cache(image_path, img_array, (width, height))
                
                return img_array
        except Exception as e:
            self.logger.error(f"加载图片失败: {image_path}, 错误: {e}")
        
        return None
    
    def _update_cache(self, image_path: str, img_array: np.ndarray, size: Tuple[int, int]):
        """更新图片缓存"""
        # 如果缓存已满，移除最旧的图片
        if len(self._image_cache) >= self._max_cache_size:
            # 简单策略：移除第一个缓存项
            oldest_key = next(iter(self._image_cache))
            del self._image_cache[oldest_key]
        
        # 添加新缓存
        self._image_cache[image_path] = (img_array.copy(), size)
    
    def clear_cache(self):
        """清除图片缓存"""
        self._image_cache.clear()
    
    def get_image_thumbnail(self, image_path: str, max_size: Tuple[int, int] = (100, 100)) -> Optional[np.ndarray]:
        """获取缩略图"""
        img_array = self.load_image(image_path, use_cache=True)
        if img_array is None:
            return None
        
        # 计算缩略图尺寸
        height, width = img_array.shape[:2]
        max_width, max_height = max_size
        
        # 计算缩放比例
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 缩放图片
        thumbnail = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return thumbnail
    
    def get_folder_path(self) -> Optional[str]:
        """获取当前加载的文件夹路径"""
        return self._current_folder
    
    def get_all_image_paths(self) -> List[str]:
        """获取所有图片路径"""
        return self._image_paths.copy()
    
    def get_next_image_index(self, current_index: int) -> int:
        """获取下一张图片的索引"""
        if len(self._image_paths) == 0:
            return 0
        return (current_index + 1) % len(self._image_paths)
    
    def get_prev_image_index(self, current_index: int) -> int:
        """获取上一张图片的索引"""
        if len(self._image_paths) == 0:
            return 0
        return (current_index - 1) % len(self._image_paths)
    
    def find_image_by_name(self, name: str) -> Optional[int]:
        """根据文件名查找图片索引"""
        name_lower = name.lower()
        for i, path in enumerate(self._image_paths):
            if name_lower in Path(path).name.lower():
                return i
        return None
    
    def export_image_with_annotations(
        self, 
        image_path: str, 
        annotations: List, 
        output_path: str
    ) -> bool:
        """导出带标注框的图片"""
        try:
            img_array = self.load_image(image_path, use_cache=False)
            if img_array is None:
                return False
            
            # 绘制标注框
            img_with_boxes = img_array.copy()
            height, width = img_array.shape[:2]
            
            for ann in annotations:
                # 提取标注信息
                if hasattr(ann, 'to_dict'):
                    ann_dict = ann.to_dict()
                    x = int(ann_dict.get('x', 0))
                    y = int(ann_dict.get('y', 0))
                    w = int(ann_dict.get('width', 0))
                    h = int(ann_dict.get('height', 0))
                    class_id = ann_dict.get('class_id', 0)
                else:
                    x = int(ann.get('x', 0))
                    y = int(ann.get('y', 0))
                    w = int(ann.get('width', 0))
                    h = int(ann.get('height', 0))
                    class_id = ann.get('class_id', 0)
                
                # 绘制矩形框
                color = self._get_class_color(class_id)
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
                
                # 添加类别标签
                label = f"Class {class_id}"
                font_scale = 0.5
                thickness = 1
                
                # 计算文本尺寸
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = x
                text_y = max(y - 5, text_size[1] + 5)
                
                # 绘制文本背景
                cv2.rectangle(
                    img_with_boxes, 
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    color,
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    img_with_boxes, 
                    label, 
                    (text_x + 3, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
            
            # 保存图片
            cv2.imwrite(output_path, img_with_boxes)
            return True
            
        except Exception as e:
            self.logger.error(f"导出带标注的图片失败: {e}")
            return False
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """根据类别ID生成颜色"""
        # 使用不同的颜色映射
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
            (128, 0, 0),    # 深红
            (0, 128, 0),    # 深绿
            (0, 0, 128),    # 深蓝
            (128, 128, 0),  # 橄榄色
        ]
        return colors[class_id % len(colors)]
    
    def batch_resize_images(
        self, 
        output_folder: str, 
        target_size: Tuple[int, int] = (640, 640),
        keep_aspect_ratio: bool = True
    ) -> List[str]:
        """批量调整图片尺寸"""
        if not self._image_paths:
            return []
        
        output_paths = []
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        target_width, target_height = target_size
        
        for image_path in self._image_paths:
            try:
                img_array = self.load_image(image_path, use_cache=False)
                if img_array is None:
                    continue
                
                height, width = img_array.shape[:2]
                
                if keep_aspect_ratio:
                    # 保持宽高比
                    scale = min(target_width / width, target_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # 调整尺寸
                    resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # 创建画布并居中图片
                    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    
                    # 计算偏移量
                    x_offset = (target_width - new_width) // 2
                    y_offset = (target_height - new_height) // 2
                    
                    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
                    result_img = canvas
                else:
                    # 直接调整到目标尺寸
                    result_img = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
                
                # 保存图片
                output_path = output_folder / Path(image_path).name
                cv2.imwrite(str(output_path), result_img)
                output_paths.append(str(output_path))
                
            except Exception as e:
                self.logger.warning(f"调整图片尺寸失败: {image_path}, 错误: {e}")
        
        return output_paths