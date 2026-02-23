"""
类别管理器
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

from src.utils.logger import get_logger_simple


class ClassManager:
    """类别管理器"""
    
    def __init__(self):
        self._classes: Dict[int, Dict] = {}
        self._next_class_id = 0
        
        # 日志记录器
        self.logger = get_logger_simple(__name__)
        
        # 不初始化默认类别，从空状态开始
        # 用户可以根据需要添加自定义类别
    
    def add_class(self, name: str, color: Optional[Tuple[int, int, int]] = None) -> int:
        """添加新类别"""
        # 检查名称是否已存在
        for class_id, class_info in self._classes.items():
            if class_info["name"] == name:
                return class_id
        
        # 生成或使用提供的颜色
        if color is None:
            color = self._generate_color()
        
        # 添加类别
        class_id = self._next_class_id
        self._classes[class_id] = {
            "name": name,
            "color": color
        }
        
        self._next_class_id += 1
        return class_id
    
    def update_class(self, class_id: int, name: str, color: Tuple[int, int, int]):
        """更新类别信息"""
        if class_id in self._classes:
            self._classes[class_id] = {
                "name": name,
                "color": color
            }
    
    def delete_class(self, class_id: int):
        """删除类别"""
        if class_id in self._classes:
            del self._classes[class_id]
            
            # 如果删除的是最大ID，更新下一个ID
            if class_id == self._next_class_id - 1:
                self._update_next_class_id()
    
    def _update_next_class_id(self):
        """更新下一个可用的类别ID"""
        if self._classes:
            self._next_class_id = max(self._classes.keys()) + 1
        else:
            self._next_class_id = 0
    
    def add_or_update_class(self, class_id: int, name: str, color: Tuple[int, int, int]) -> int:
        """添加或更新类别（如果ID已存在则更新，否则添加）"""
        if class_id in self._classes:
            # 更新现有类别
            self._classes[class_id] = {
                "name": name,
                "color": color
            }
            return class_id
        else:
            # 添加新类别
            self._classes[class_id] = {
                "name": name,
                "color": color
            }
            # 更新下一个可用的ID
            if class_id >= self._next_class_id:
                self._next_class_id = class_id + 1
            return class_id
    
    def get_class(self, class_id: int) -> Optional[Dict]:
        """获取类别信息"""
        return self._classes.get(class_id)
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        class_info = self.get_class(class_id)
        if class_info:
            return class_info["name"]
        return f"Unknown({class_id})"
    
    def get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色"""
        class_info = self.get_class(class_id)
        if class_info:
            return class_info["color"]
        return (128, 128, 128)  # 默认灰色
    
    def get_classes(self) -> Dict[int, Dict]:
        """获取所有类别"""
        return self._classes.copy()
    
    def get_classes_list(self) -> List[Dict]:
        """获取类别列表（用于保存）"""
        classes_list = []
        for class_id in sorted(self._classes.keys()):
            class_info = self._classes[class_id].copy()
            class_info["id"] = class_id
            classes_list.append(class_info)
        return classes_list
    
    def load_from_list(self, classes_list: List[Dict]):
        """从列表加载类别"""
        self._classes.clear()
        
        for class_info in classes_list:
            class_id = class_info.get("id", len(self._classes))
            name = class_info.get("name", f"class_{class_id}")
            color = class_info.get("color", self._generate_color())
            
            self._classes[class_id] = {
                "name": name,
                "color": color
            }
        
        self._update_next_class_id()
    
    def get_class_count(self) -> int:
        """获取类别数量"""
        return len(self._classes)
    
    def find_class_by_name(self, name: str) -> Optional[int]:
        """根据名称查找类别ID"""
        for class_id, class_info in self._classes.items():
            if class_info["name"] == name:
                return class_id
        return None
    
    def _generate_color(self) -> Tuple[int, int, int]:
        """生成随机但可区分的颜色"""
        # 避免使用太暗或太亮的颜色
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        return (r, g, b)
    
    def export_to_yaml(self, output_path: str, dataset_path: str = "./dataset"):
        """导出为YOLO格式的data.yaml文件"""
        # 构建正确格式的names字典
        names_dict = {}
        for class_id in sorted(self._classes.keys()):
            names_dict[class_id] = self.get_class_name(class_id)
        
        yaml_data = {
            "path": dataset_path,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": self.get_class_count(),
            "names": names_dict  # 使用字典格式
        }
        
        # 保存YAML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
    def import_from_yaml(self, yaml_path: str):
        """从YAML文件导入类别"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # 清空现有类别
            self._classes.clear()
            
            # 解析类别
            names = yaml_data.get("names", [])
            for class_id, name in enumerate(names):
                color = self._generate_color()
                self._classes[class_id] = {
                    "name": name,
                    "color": color
                }
            
            self._update_next_class_id()
            return True
            
        except Exception as e:
            self.logger.error(f"导入YAML文件失败: {e}")
            return False
    
    def save_to_json(self, json_path: str):
        """保存类别到JSON文件"""
        classes_data = self.get_classes_list()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(classes_data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, json_path: str):
        """从JSON文件加载类别"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                classes_data = json.load(f)
            
            self.load_from_list(classes_data)
            return True
            
        except Exception as e:
            self.logger.error(f"加载JSON文件失败: {e}")
            return False
    
    def get_class_statistics(self, annotations: Dict[str, List]) -> Dict[int, int]:
        """获取类别统计信息（每个类别的标注数量）"""
        class_counts = {}
        
        for image_path, image_annotations in annotations.items():
            for ann in image_annotations:
                if hasattr(ann, 'class_id'):
                    class_id = ann.class_id
                else:
                    class_id = ann.get('class_id', 0)
                
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        return class_counts
    
    def get_class_names(self) -> List[str]:
        """获取所有类别名称"""
        names = []
        for class_id in sorted(self._classes.keys()):
            names.append(self.get_class_name(class_id))
        return names
    
    def validate_class_id(self, class_id: int) -> bool:
        """验证类别ID是否有效"""
        return class_id in self._classes
    
    def get_next_available_class_id(self) -> int:
        """获取下一个可用的类别ID"""
        return self._next_class_id
    
    def clear_all(self) -> None:
        """清空所有类别"""
        self._classes.clear()
        self._next_class_id = 0
        self.logger.info("所有类别已清空")
    
    def merge_classes(self, other_class_manager: 'ClassManager') -> Dict[int, int]:
        """
        合并两个类别管理器
        返回映射关系：原ID -> 新ID
        """
        id_mapping = {}
        
        for other_class_id, other_class_info in other_class_manager.get_classes().items():
            other_name = other_class_info["name"]
            other_color = other_class_info["color"]
            
            # 查找是否有相同名称的类别
            existing_id = self.find_class_by_name(other_name)
            
            if existing_id is not None:
                # 使用现有类别
                id_mapping[other_class_id] = existing_id
            else:
                # 添加新类别
                new_id = self.add_class(other_name, other_color)
                id_mapping[other_class_id] = new_id
        
        return id_mapping
