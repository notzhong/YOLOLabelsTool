"""
YOLO格式导出器
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import yaml

from ..core.annotation import AnnotationManager
from ..core.image_manager import ImageManager
from ..core.class_manager import ClassManager
from ..utils.logger import get_logger_simple


class YOLOExporter:
    """YOLO格式导出器"""
    
    def __init__(self):
        self.output_structure = {
            "images": ["train", "val", "test"],
            "labels": ["train", "val", "test"]
        }
        
        # 日志记录器
        self.logger = get_logger_simple(__name__)
    
    def export(
        self,
        image_manager: ImageManager,
        annotation_manager: AnnotationManager,
        class_manager: ClassManager,
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        copy_images: bool = True
    ):
        """
        导出为YOLO格式
        
        Args:
            image_manager: 图片管理器
            annotation_manager: 标注管理器
            class_manager: 类别管理器
            output_dir: 输出目录
            split_ratios: 训练集/验证集/测试集比例
            copy_images: 是否复制图片到输出目录
        """
        # 验证分割比例
        if abs(sum(split_ratios) - 1.0) > 0.01:
            raise ValueError(f"分割比例之和应为1.0，当前为{sum(split_ratios)}")
        
        # 创建输出目录结构
        self._create_output_structure(output_dir)
        
        # 获取所有图片路径
        image_paths = image_manager.get_all_image_paths()
        if not image_paths:
            raise ValueError("没有图片可导出")
        
        # 划分数据集
        train_paths, val_paths, test_paths = self._split_dataset(image_paths, split_ratios)
        
        # 导出各个子集
        self._export_subset(
            "train", train_paths, image_manager, annotation_manager, 
            class_manager, output_dir, copy_images
        )
        self._export_subset(
            "val", val_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images
        )
        self._export_subset(
            "test", test_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images
        )
        
        # 导出数据集划分文件
        self._export_split_files(output_dir, train_paths, val_paths, test_paths)
        
        # 导出data.yaml配置文件
        self._export_yaml_config(class_manager, output_dir)
    
    def _create_output_structure(self, output_dir: str):
        """创建输出目录结构"""
        output_path = Path(output_dir)
        
        # 创建主目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        for main_dir, sub_dirs in self.output_structure.items():
            main_dir_path = output_path / main_dir
            main_dir_path.mkdir(exist_ok=True)
            
            for sub_dir in sub_dirs:
                sub_dir_path = main_dir_path / sub_dir
                sub_dir_path.mkdir(exist_ok=True)
    
    def _split_dataset(
        self, 
        image_paths: List[str], 
        split_ratios: Tuple[float, float, float]
    ) -> Tuple[List[str], List[str], List[str]]:
        """划分数据集"""
        total_count = len(image_paths)
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # 计算各集合数量
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        # 划分数据集
        train_paths = image_paths[:train_count]
        val_paths = image_paths[train_count:train_count + val_count]
        test_paths = image_paths[train_count + val_count:]
        
        return train_paths, val_paths, test_paths
    
    def _export_subset(
        self,
        subset_name: str,
        image_paths: List[str],
        image_manager: ImageManager,
        annotation_manager: AnnotationManager,
        class_manager: ClassManager,
        output_dir: str,
        copy_images: bool
    ):
        """导出子集"""
        output_path = Path(output_dir)
        
        for image_path in image_paths:
            try:
                # 获取图片信息
                image_info = image_manager.get_image_info(image_path)
                if image_info is None:
                    self.logger.warning(f"无法获取图片信息: {image_path}")
                    continue
                
                image_width, image_height = image_info
                
                # 获取标注
                annotations = annotation_manager.get_annotations(image_path)
                
                # 导出标注文件（YOLO格式）
                self._export_yolo_labels(
                    image_path, annotations, image_width, image_height,
                    output_path / "labels" / subset_name
                )
                
                # 复制图片
                if copy_images:
                    self._copy_image(
                        image_path, output_path / "images" / subset_name
                    )
                    
            except Exception as e:
                self.logger.error(f"导出图片失败 {image_path}: {e}")
    
    def _export_yolo_labels(
        self,
        image_path: str,
        annotations: List,
        image_width: int,
        image_height: int,
        output_labels_dir: Path
    ):
        """导出YOLO格式标注文件"""
        image_name = Path(image_path).stem
        output_file = output_labels_dir / f"{image_name}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for ann in annotations:
                # 转换为YOLO格式
                if hasattr(ann, 'to_yolo_format'):
                    yolo_data = ann.to_yolo_format(image_width, image_height)
                else:
                    # 如果是字典格式
                    x = ann.get('x', 0)
                    y = ann.get('y', 0)
                    width = ann.get('width', 0)
                    height = ann.get('height', 0)
                    class_id = ann.get('class_id', 0)
                    
                    # 计算YOLO格式
                    x_center = (x + width / 2) / image_width
                    y_center = (y + height / 2) / image_height
                    norm_width = width / image_width
                    norm_height = height / image_height
                    
                    yolo_data = [class_id, x_center, y_center, norm_width, norm_height]
                
                # 写入文件
                line = " ".join(f"{val:.6f}" for val in yolo_data)
                f.write(line + "\n")
    
    def _copy_image(self, image_path: str, output_images_dir: Path):
        """复制图片到输出目录"""
        image_name = Path(image_path).name
        output_file = output_images_dir / image_name
        
        try:
            shutil.copy2(image_path, output_file)
        except Exception as e:
            self.logger.error(f"复制图片失败 {image_path}: {e}")
    
    def _export_split_files(
        self,
        output_dir: str,
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str]
    ):
        """导出数据集划分文件"""
        output_path = Path(output_dir)
        
        # 导出训练集列表
        self._export_path_list(
            output_path / "train.txt",
            train_paths,
            output_dir
        )
        
        # 导出验证集列表
        self._export_path_list(
            output_path / "val.txt",
            val_paths,
            output_dir
        )
        
        # 导出测试集列表
        self._export_path_list(
            output_path / "test.txt",
            test_paths,
            output_dir
        )
    
    def _export_path_list(
        self,
        output_file: Path,
        image_paths: List[str],
        output_dir: str
    ):
        """导出路径列表文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for image_path in image_paths:
                image_name = Path(image_path).name
                
                # 构建相对路径
                rel_path = str(Path("images") / image_name)
                f.write(rel_path + "\n")
    
    def _export_yaml_config(self, class_manager: ClassManager, output_dir: str):
        """导出data.yaml配置文件"""
        output_path = Path(output_dir)
        yaml_file = output_path / "data.yaml"
        
        yaml_data = {
            "path": str(output_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": class_manager.get_class_count(),
            "names": class_manager.get_class_names()
        }
        
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
    def export_single_image(
        self,
        image_path: str,
        annotations: List,
        output_dir: str,
        class_manager: ClassManager
    ):
        """导出单张图片的YOLO格式"""
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取图片信息
        from PIL import Image
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # 导出标注
        image_name = Path(image_path).stem
        label_file = output_path / f"{image_name}.txt"
        
        with open(label_file, 'w', encoding='utf-8') as f:
            for ann in annotations:
                if hasattr(ann, 'to_yolo_format'):
                    yolo_data = ann.to_yolo_format(image_width, image_height)
                else:
                    x = ann.get('x', 0)
                    y = ann.get('y', 0)
                    width = ann.get('width', 0)
                    height = ann.get('height', 0)
                    class_id = ann.get('class_id', 0)
                    
                    x_center = (x + width / 2) / image_width
                    y_center = (y + height / 2) / image_height
                    norm_width = width / image_width
                    norm_height = height / image_height
                    
                    yolo_data = [class_id, x_center, y_center, norm_width, norm_height]
                
                line = " ".join(f"{val:.6f}" for val in yolo_data)
                f.write(line + "\n")
        
        # 复制图片
        image_output = output_path / Path(image_path).name
        shutil.copy2(image_path, image_output)
        
        # 导出简化的配置文件
        config_file = output_path / "data.yaml"
        config_data = {
            "path": str(output_path.absolute()),
            "train": ".",
            "val": ".",
            "test": ".",
            "nc": class_manager.get_class_count(),
            "names": class_manager.get_class_names()
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def export_with_custom_split(
        self,
        image_manager: ImageManager,
        annotation_manager: AnnotationManager,
        class_manager: ClassManager,
        output_dir: str,
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str],
        copy_images: bool = True
    ):
        """使用自定义划分导出数据集"""
        # 创建输出目录结构
        self._create_output_structure(output_dir)
        
        # 导出各个子集
        self._export_subset(
            "train", train_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images
        )
        self._export_subset(
            "val", val_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images
        )
        self._export_subset(
            "test", test_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images
        )
        
        # 导出数据集划分文件
        self._export_split_files(output_dir, train_paths, val_paths, test_paths)
        
        # 导出data.yaml配置文件
        self._export_yaml_config(class_manager, output_dir)
    
    def validate_export(self, output_dir: str) -> Dict:
        """验证导出结果"""
        output_path = Path(output_dir)
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # 检查目录结构
            required_dirs = ["images/train", "images/val", "images/test", 
                           "labels/train", "labels/val", "labels/test"]
            
            for dir_name in required_dirs:
                dir_path = output_path / dir_name
                if not dir_path.exists():
                    validation_result["errors"].append(f"目录不存在: {dir_name}")
                    validation_result["valid"] = False
            
            # 检查data.yaml文件
            yaml_file = output_path / "data.yaml"
            if not yaml_file.exists():
                validation_result["errors"].append("data.yaml文件不存在")
                validation_result["valid"] = False
            else:
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        yaml_data = yaml.safe_load(f)
                    
                    required_keys = ["path", "train", "val", "test", "nc", "names"]
                    for key in required_keys:
                        if key not in yaml_data:
                            validation_result["errors"].append(f"data.yaml缺少必需字段: {key}")
                            validation_result["valid"] = False
                            
                except Exception as e:
                    validation_result["errors"].append(f"解析data.yaml失败: {e}")
                    validation_result["valid"] = False
            
            # 统计信息
            stats = {}
            for subset in ["train", "val", "test"]:
                images_dir = output_path / "images" / subset
                labels_dir = output_path / "labels" / subset
                
                if images_dir.exists():
                    image_count = len(list(images_dir.glob("*")))
                else:
                    image_count = 0
                
                if labels_dir.exists():
                    label_count = len(list(labels_dir.glob("*.txt")))
                else:
                    label_count = 0
                
                stats[subset] = {
                    "images": image_count,
                    "labels": label_count,
                    "matched": image_count == label_count
                }
                
                if image_count != label_count:
                    validation_result["warnings"].append(
                        f"{subset}子集图片数({image_count})与标注数({label_count})不匹配"
                    )
            
            validation_result["statistics"] = stats
            
        except Exception as e:
            validation_result["errors"].append(f"验证过程中发生错误: {e}")
            validation_result["valid"] = False
        
        return validation_result