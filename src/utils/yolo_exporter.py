"""
YOLO格式导出器
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import yaml

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.annotation import AnnotationManager
    from src.core.image_manager import ImageManager
    from src.core.class_manager import ClassManager
else:
    from src.core.image_manager import ImageManager
    from src.core.class_manager import ClassManager
    # AnnotationManager will be imported locally where needed
from src.utils.logger import get_logger_simple


logger = get_logger_simple(__name__)


def _polygon_points_from(annotation: Any) -> Optional[List[Tuple[float, float]]]:
    """提取标注的多边形顶点（像素坐标）；非多边形返回 None"""
    points = getattr(annotation, 'points', None)
    if points is None and isinstance(annotation, dict):
        if annotation.get('type') == 'polygon' or 'points' in annotation:
            points = annotation.get('points')
    if points is None:
        return None
    return [(float(point[0]), float(point[1])) for point in points]


def _box_fields_from(annotation: Any) -> Tuple[float, float, float, float, Any]:
    """提取标注的矩形字段 (x, y, width, height, class_id)"""
    if isinstance(annotation, dict):
        return (
            annotation.get('x', 0),
            annotation.get('y', 0),
            annotation.get('width', 0),
            annotation.get('height', 0),
            annotation.get('class_id', 0),
        )
    return (
        annotation.x,
        annotation.y,
        annotation.width,
        annotation.height,
        annotation.class_id,
    )


def annotation_to_yolo_lines(
    annotations: List[Any],
    image_width: int,
    image_height: int,
    task: str = "detect",
) -> List[str]:
    """
    将标注列表转换为 YOLO 格式文本行。

    task="detect"：`class x_center y_center width height`，多边形取其外接矩形；
    task="segment"：`class x1 y1 x2 y2 ...`，矩形展开为 4 顶点矩形。

    支持 dataclass Annotation/PolygonAnnotation 和普通 dict 两种输入。
    顶点数不足 3 的多边形会被跳过（记录警告）。
    """
    lines = []
    for ann in annotations:
        polygon_points = _polygon_points_from(ann)

        if task == "segment":
            if polygon_points is None:
                x, y, width, height, class_id = _box_fields_from(ann)
                polygon_points = [
                    (x, y), (x + width, y), (x + width, y + height), (x, y + height),
                ]
            else:
                class_id = ann.get('class_id', 0) if isinstance(ann, dict) else ann.class_id

            if len(polygon_points) < 3:
                logger.warning(f"跳过无效多边形标注（顶点数不足 3）: class_id={class_id}")
                continue

            parts = [str(int(class_id))]
            for x, y in polygon_points:
                parts.append(f"{x / image_width:.6f}")
                parts.append(f"{y / image_height:.6f}")
            lines.append(" ".join(parts))
        else:
            if polygon_points is not None:
                if len(polygon_points) < 3:
                    logger.warning("跳过无效多边形标注（顶点数不足 3）")
                    continue
                xs = [point[0] for point in polygon_points]
                ys = [point[1] for point in polygon_points]
                x, y = min(xs), min(ys)
                width, height = max(xs) - x, max(ys) - y
                class_id = ann.get('class_id', 0) if isinstance(ann, dict) else ann.class_id
            else:
                x, y, width, height, class_id = _box_fields_from(ann)

            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height
            norm_width = width / image_width
            norm_height = height / image_height
            lines.append(
                f"{int(class_id)} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )
    return lines


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
        annotation_manager: 'AnnotationManager',
        class_manager: ClassManager,
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        copy_images: bool = True,
        yaml_filename: str = "data.yaml",
        random_seed: int = 42,
        task: str = "detect"
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
            random_seed: 随机种子
            task: 任务类型（detect 输出检测框格式，segment 输出多边形分割格式）
        """
        # 验证分割比例
        if abs(sum(split_ratios) - 1.0) > 0.01:
            self.logger.error(f"导出失败: 分割比例和为 {sum(split_ratios)}，应为 1.0")
            raise ValueError(f"分割比例之和应为1.0，当前为{sum(split_ratios)}")

        # 创建输出目录结构
        self._create_output_structure(output_dir)

        # 获取所有图片路径
        image_paths = image_manager.get_all_image_paths()
        if not image_paths:
            self.logger.error("导出失败: 没有图片可导出")
            raise ValueError("没有图片可导出")
        
        # 划分数据集
        train_paths, val_paths, test_paths = self._split_dataset(image_paths, split_ratios, random_seed)
        
        # 导出各个子集
        self._export_subset(
            "train", train_paths, image_manager, annotation_manager, 
            class_manager, output_dir, copy_images, task
        )
        self._export_subset(
            "val", val_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images, task
        )
        self._export_subset(
            "test", test_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images, task
        )
        
        # 导出数据集划分文件
        self._export_split_files(output_dir, train_paths, val_paths, test_paths)
        
        # 导出data.yaml配置文件
        self._export_yaml_config(class_manager, output_dir, yaml_filename, task=task)
    
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
        split_ratios: Tuple[float, float, float],
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """划分数据集（委托给 DatasetSplitter.random_split）"""
        from src.utils.dataset_splitter import random_split
        return random_split(image_paths, split_ratios, seed)
    
    def _export_subset(
        self,
        subset_name: str,
        image_paths: List[str],
        image_manager: ImageManager,
        annotation_manager: 'AnnotationManager',
        class_manager: ClassManager,
        output_dir: str,
        copy_images: bool,
        task: str = "detect"
    ):
        """导出子集（委托给 _export_single_subset_image）"""
        output_path = Path(output_dir)
        images_dir = output_path / "images" / subset_name
        labels_dir = output_path / "labels" / subset_name

        for image_path in image_paths:
            if copy_images:
                self._export_single_subset_image(
                    image_path, image_manager, annotation_manager,
                    images_dir, labels_dir, task=task,
                )
            else:
                self._export_single_subset_image(
                    image_path, image_manager, annotation_manager,
                    None, labels_dir, task=task,
                )
    
    def _copy_image(self, image_path: str, output_images_dir: Path):
        """复制图片到输出目录"""
        image_name = Path(image_path).name
        output_file = output_images_dir / image_name

        try:
            shutil.copy2(image_path, output_file)
        except Exception as e:
            self.logger.error(f"复制图片失败 {image_path}: {e}")

    def _export_single_subset_image(
        self,
        image_path: str,
        image_manager,
        annotation_manager,
        output_images_dir: Path,
        output_labels_dir: Path,
        task: str = "detect",
    ):
        """导出单张图片及其 YOLO 标注到指定子集目录（供 DatasetSplitter 和内部使用）"""
        try:
            image_info = image_manager.get_image_info(image_path)
            if image_info is None:
                self.logger.warning(f"无法获取图片信息: {image_path}")
                return
            image_width, image_height = image_info

            if output_images_dir is not None:
                shutil.copy2(image_path, output_images_dir / Path(image_path).name)

            annotations = annotation_manager.get_annotations(image_path)
            self._write_yolo_file(
                annotations, image_width, image_height,
                output_labels_dir / f"{Path(image_path).stem}.txt",
                task=task,
            )
        except Exception as e:
            self.logger.error(f"导出失败 {image_path}: {e}")

    def _write_yolo_file(
        self, annotations, image_width, image_height, output_file: Path, task: str = "detect"
    ):
        """写入 YOLO 标注文件"""
        lines = annotation_to_yolo_lines(annotations, image_width, image_height, task=task)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    
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
            "train"
        )
        
        # 导出验证集列表
        self._export_path_list(
            output_path / "val.txt",
            val_paths,
            "val"
        )
        
        # 导出测试集列表
        self._export_path_list(
            output_path / "test.txt",
            test_paths,
            "test"
        )
    
    def _export_path_list(
        self,
        output_file: Path,
        image_paths: List[str],
        subset: str
    ):
        """导出路径列表文件。

        注意：列表文件路径必须使用**正斜杠**——ultralytics/YOLO 按 POSIX 风格
        解析 train.txt/val.txt/test.txt，Windows 上若写成 `images\\train\\a.png`
        会导致训练时找不到图片。
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for image_path in image_paths:
                image_name = Path(image_path).name
                rel_path = "/".join(["images", subset, image_name])
                f.write(rel_path + "\n")

    def _build_names_config(self, class_manager: ClassManager):
        """构建 names 字段（委托给 ClassManager）"""
        return class_manager.build_names_config()

    def _export_yaml_config(
        self,
        class_manager: ClassManager,
        output_dir: str,
        yaml_filename: str = "data.yaml",
        task: str = "detect",
    ):
        """导出 data.yaml 配置文件（分割任务追加 task: segment 字段）"""
        output_path = Path(output_dir)
        yaml_file = output_path / yaml_filename

        names_value = self._build_names_config(class_manager)

        yaml_data = {
            "path": str(output_path.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": class_manager.get_class_count(),
            "names": names_value
        }
        if task == "segment":
            yaml_data["task"] = "segment"

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    def export_single_image(
        self,
        image_path: str,
        annotations: List,
        output_dir: str,
        class_manager: ClassManager,
        yaml_filename: str = "data.yaml",
        task: str = "detect"
    ):
        """导出单张图片的 YOLO 格式"""
        output_path = Path(output_dir)

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取图片尺寸
        from PIL import Image
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # 导出标注
        image_name = Path(image_path).stem
        label_file = output_path / f"{image_name}.txt"

        lines = annotation_to_yolo_lines(annotations, image_width, image_height, task=task)
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        # 复制图片
        image_output = output_path / Path(image_path).name
        shutil.copy2(image_path, image_output)

        # 导出简化配置
        config_file = output_path / yaml_filename

        names_value = self._build_names_config(class_manager)

        config_data = {
            "path": str(output_path.resolve()),
            "train": ".",
            "val": ".",
            "test": ".",
            "nc": class_manager.get_class_count(),
            "names": names_value
        }
        if task == "segment":
            config_data["task"] = "segment"

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    def export_with_custom_split(
        self,
        image_manager: ImageManager,
        annotation_manager: 'AnnotationManager',
        class_manager: ClassManager,
        output_dir: str,
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str],
        copy_images: bool = True,
        yaml_filename: str = "data.yaml",
        task: str = "detect"
    ):
        """使用自定义划分导出数据集"""
        # 创建输出目录结构
        self._create_output_structure(output_dir)
        
        # 导出各个子集
        self._export_subset(
            "train", train_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images, task
        )
        self._export_subset(
            "val", val_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images, task
        )
        self._export_subset(
            "test", test_paths, image_manager, annotation_manager,
            class_manager, output_dir, copy_images, task
        )
        
        # 导出数据集划分文件
        self._export_split_files(output_dir, train_paths, val_paths, test_paths)
        
        # 导出data.yaml配置文件
        self._export_yaml_config(class_manager, output_dir, yaml_filename, task=task)
    
    def validate_export(self, output_dir: str, yaml_filename: str = "data.yaml") -> Dict:
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
                    self.logger.error(f"导出验证: 目录不存在 {dir_name}")
                    validation_result["errors"].append(f"目录不存在: {dir_name}")
                    validation_result["valid"] = False

            # 检查data.yaml文件
            yaml_file = output_path / yaml_filename
            if not yaml_file.exists():
                self.logger.error("导出验证: data.yaml 文件不存在")
                validation_result["errors"].append("data.yaml文件不存在")
                validation_result["valid"] = False
            else:
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        yaml_data = yaml.safe_load(f)

                    required_keys = ["path", "train", "val", "test", "nc", "names"]
                    for key in required_keys:
                        if key not in yaml_data:
                            self.logger.error(f"导出验证: data.yaml 缺少字段 {key}")
                            validation_result["errors"].append(f"data.yaml缺少必需字段: {key}")
                            validation_result["valid"] = False

                except Exception as e:
                    self.logger.exception(f"导出验证: 解析 data.yaml 失败")
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
                    self.logger.warning(f"导出验证: {subset} 图片({image_count})与标注({label_count})不匹配")
                    validation_result["warnings"].append(
                        f"{subset}子集图片数({image_count})与标注数({label_count})不匹配"
                    )

            validation_result["statistics"] = stats

        except Exception as e:
            self.logger.exception(f"导出验证过程中发生错误: {e}")
            validation_result["errors"].append(f"验证过程中发生错误: {e}")
            validation_result["valid"] = False
        
        return validation_result
