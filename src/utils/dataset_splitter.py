"""
数据集划分器 — 划分逻辑在此，文件导出委托给 YOLOExporter
"""

import random
from pathlib import Path
from typing import List, Tuple

from src.utils.logger import get_logger_simple
from src.utils.yolo_exporter import YOLOExporter


def random_split(
    image_paths: List[str],
    split_ratios: Tuple[float, float, float],
) -> Tuple[List[str], List[str], List[str]]:
    """打乱并划分数据集为 train/val/test。

    先随机打乱再按比例连续切分，消除文件系统顺序带来的划分偏差。
    """
    if abs(sum(split_ratios) - 1.0) > 0.01:
        raise ValueError(f"分割比例之和应为1.0，当前为{sum(split_ratios)}")

    total = len(image_paths)
    train_r, val_r, _ = split_ratios

    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    train_end = int(total * train_r)
    val_end = train_end + int(total * val_r)

    return (
        shuffled[:train_end],
        shuffled[train_end:val_end],
        shuffled[val_end:],
    )


class DatasetSplitter:
    """数据集划分器 — 划分逻辑在此，文件导出委托给 YOLOExporter"""

    def __init__(self):
        self.logger = get_logger_simple(__name__)

    def split_and_export(
        self,
        image_manager,
        annotation_manager,
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        copy_images: bool = True,
        random_seed: int = 42
    ):
        """
        划分并导出数据集

        Args:
            image_manager: 图片管理器
            annotation_manager: 标注管理器
            output_dir: 输出目录
            split_ratios: 训练集/验证集/测试集比例
            copy_images: 是否复制图片
            random_seed: 随机种子
        """
        # 获取所有图片路径
        image_paths = image_manager.get_all_image_paths()
        if not image_paths:
            raise ValueError("没有图片可划分")

        # 打乱并划分
        random.seed(random_seed)
        train_paths, val_paths, test_paths = random_split(image_paths, split_ratios)

        # 创建输出目录
        output_path = Path(output_dir)
        self._create_split_dirs(output_path)

        # 导出划分信息（绝对路径 + 统计）
        self._export_split_info(
            output_path, train_paths, val_paths, test_paths
        )

        # 委托给 YOLOExporter 处理图片复制 + YOLO 标注导出
        if copy_images:
            exporter = YOLOExporter()
            for subset_name, paths in [
                ("train", train_paths),
                ("val", val_paths),
                ("test", test_paths),
            ]:
                for image_path in paths:
                    exporter._export_single_subset_image(
                        image_path, image_manager, annotation_manager,
                        output_path / "images" / subset_name,
                        output_path / "labels" / subset_name,
                    )
    
    def _export_split_info(
        self,
        output_path: Path,
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str]
    ):
        """导出划分信息文件"""
        # 导出训练集列表
        self._export_path_list(
            output_path / "train.txt",
            train_paths
        )
        
        # 导出验证集列表
        self._export_path_list(
            output_path / "val.txt",
            val_paths
        )
        
        # 导出测试集列表
        self._export_path_list(
            output_path / "test.txt",
            test_paths
        )
        
        # 导出统计信息
        self._export_statistics(
            output_path / "split_statistics.txt",
            train_paths, val_paths, test_paths
        )
    
    def _export_path_list(self, output_file: Path, paths: List[str]):
        """导出路径列表"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for path in paths:
                f.write(f"{path}\n")
    
    def _export_statistics(
        self,
        output_file: Path,
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str]
    ):
        """导出统计信息"""
        total = len(train_paths) + len(val_paths) + len(test_paths)
        
        statistics = {
            "total_images": total,
            "train_images": len(train_paths),
            "val_images": len(val_paths),
            "test_images": len(test_paths),
            "train_ratio": len(train_paths) / total if total > 0 else 0,
            "val_ratio": len(val_paths) / total if total > 0 else 0,
            "test_ratio": len(test_paths) / total if total > 0 else 0
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("数据集划分统计信息\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in statistics.items():
                if "ratio" in key:
                    f.write(f"{key}: {value:.2%}\n")
                else:
                    f.write(f"{key}: {value}\n")
    
    def _create_split_dirs(self, output_path: Path):
        """创建划分目录"""
        output_path.mkdir(parents=True, exist_ok=True)
        for subset in ["train", "val", "test"]:
            (output_path / "images" / subset).mkdir(parents=True, exist_ok=True)
            (output_path / "labels" / subset).mkdir(parents=True, exist_ok=True)

    def create_cross_validation_splits(
        self,
        image_paths: List[str],
        n_folds: int = 5,
        random_seed: int = 42
    ) -> List[Tuple[List[str], List[str]]]:
        """
        创建交叉验证划分
        
        Returns:
            List of (train_paths, val_paths) for each fold
        """
        # 设置随机种子
        random.seed(random_seed)
        
        # 打乱顺序
        shuffled_paths = image_paths.copy()
        random.shuffle(shuffled_paths)
        
        # 创建K折交叉验证划分
        folds = []
        fold_size = len(shuffled_paths) // n_folds
        
        for fold_idx in range(n_folds):
            # 计算验证集范围
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size
            
            # 如果是最后一折，包含剩余所有数据
            if fold_idx == n_folds - 1:
                val_paths = shuffled_paths[val_start:]
            else:
                val_paths = shuffled_paths[val_start:val_end]
            
            # 训练集是除了验证集之外的所有数据
            train_paths = shuffled_paths[:val_start] + shuffled_paths[val_end:]
            
            folds.append((train_paths, val_paths))
        
        return folds
    
    def export_cross_validation(
        self,
        image_manager,
        annotation_manager,
        output_dir: str,
        n_folds: int = 5,
        random_seed: int = 42
    ):
        """导出交叉验证数据集"""
        # 获取所有图片路径
        image_paths = image_manager.get_all_image_paths()
        if not image_paths:
            raise ValueError("没有图片可划分")
        
        # 创建交叉验证划分
        folds = self.create_cross_validation_splits(
            image_paths, n_folds, random_seed
        )
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 导出每个折
        for fold_idx, (train_paths, val_paths) in enumerate(folds):
            fold_dir = output_path / f"fold_{fold_idx + 1}"
            fold_dir.mkdir(exist_ok=True)
            
            # 导出训练集和验证集列表
            with open(fold_dir / "train.txt", 'w', encoding='utf-8') as f:
                for path in train_paths:
                    f.write(f"{path}\n")
            
            with open(fold_dir / "val.txt", 'w', encoding='utf-8') as f:
                for path in val_paths:
                    f.write(f"{path}\n")
            
            # 导出统计信息
            stats = {
                "fold": fold_idx + 1,
                "total_folds": n_folds,
                "train_images": len(train_paths),
                "val_images": len(val_paths),
                "train_ratio": len(train_paths) / len(image_paths),
                "val_ratio": len(val_paths) / len(image_paths)
            }
            
            with open(fold_dir / "statistics.txt", 'w', encoding='utf-8') as f:
                for key, value in stats.items():
                    if "ratio" in key:
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value}\n")
    
    def balance_classes(
        self,
        image_paths: List[str],
        annotation_manager,
        max_samples_per_class: int = None,
        min_samples_per_class: int = None
    ) -> List[str]:
        """
        平衡类别分布
        
        Args:
            image_paths: 图片路径列表
            annotation_manager: 标注管理器
            max_samples_per_class: 每个类别的最大样本数
            min_samples_per_class: 每个类别的最小样本数
        
        Returns:
            平衡后的图片路径列表
        """
        # 统计每个类别的图片数量
        class_counts = {}
        class_to_images = {}
        
        for image_path in image_paths:
            annotations = annotation_manager.get_annotations(image_path)
            class_ids = set()
            
            for ann in annotations:
                if hasattr(ann, 'class_id'):
                    class_ids.add(ann.class_id)
                else:
                    class_ids.add(ann.get('class_id', 0))
            
            for class_id in class_ids:
                if class_id not in class_to_images:
                    class_to_images[class_id] = []
                    class_counts[class_id] = 0
                
                class_to_images[class_id].append(image_path)
                class_counts[class_id] += 1
        
        # 确定目标样本数
        if max_samples_per_class is None:
            max_samples_per_class = max(class_counts.values()) if class_counts else 0
        
        if min_samples_per_class is None:
            min_samples_per_class = min(class_counts.values()) if class_counts else 0
        
        # 平衡类别
        balanced_paths = []
        processed_images = set()
        
        for class_id, images in class_to_images.items():
            # 如果样本数太多，进行下采样
            if len(images) > max_samples_per_class:
                selected_images = random.sample(images, max_samples_per_class)
            # 如果样本数太少，进行上采样
            elif len(images) < min_samples_per_class:
                # 重复采样直到达到最小样本数
                selected_images = []
                while len(selected_images) < min_samples_per_class:
                    selected_images.append(random.choice(images))
                selected_images = selected_images[:min_samples_per_class]
            else:
                selected_images = images
            
            # 添加图片，避免重复
            for image_path in selected_images:
                if image_path not in processed_images:
                    balanced_paths.append(image_path)
                    processed_images.add(image_path)
        
        # 打乱顺序
        random.shuffle(balanced_paths)
        return balanced_paths
    
    def stratified_split(
        self,
        image_paths: List[str],
        annotation_manager,
        split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        random_seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        分层划分，保持每个类别的比例
        
        Args:
            image_paths: 图片路径列表
            annotation_manager: 标注管理器
            split_ratios: 划分比例
            random_seed: 随机种子
        
        Returns:
            (train_paths, val_paths, test_paths)
        """
        # 设置随机种子
        random.seed(random_seed)
        
        # 按类别分组图片
        class_to_images = {}
        
        for image_path in image_paths:
            annotations = annotation_manager.get_annotations(image_path)
            class_ids = set()
            
            for ann in annotations:
                if hasattr(ann, 'class_id'):
                    class_ids.add(ann.class_id)
                else:
                    class_ids.add(ann.get('class_id', 0))
            
            # 对于多类别图片，添加到所有相关类别
            for class_id in class_ids:
                if class_id not in class_to_images:
                    class_to_images[class_id] = []
                class_to_images[class_id].append(image_path)
        
        # 分层划分
        train_paths, val_paths, test_paths = [], [], []
        processed_images = set()
        
        for class_id, images in class_to_images.items():
            # 打乱当前类别的图片
            random.shuffle(images)
            
            # 计算各集合数量
            total = len(images)
            train_count = int(total * split_ratios[0])
            val_count = int(total * split_ratios[1])
            
            # 划分
            class_train = images[:train_count]
            class_val = images[train_count:train_count + val_count]
            class_test = images[train_count + val_count:]
            
            # 添加到总列表，避免重复
            for img in class_train:
                if img not in processed_images:
                    train_paths.append(img)
                    processed_images.add(img)
            
            for img in class_val:
                if img not in processed_images:
                    val_paths.append(img)
                    processed_images.add(img)
            
            for img in class_test:
                if img not in processed_images:
                    test_paths.append(img)
                    processed_images.add(img)
        
        # 最后打乱顺序
        random.shuffle(train_paths)
        random.shuffle(val_paths)
        random.shuffle(test_paths)
        
        return train_paths, val_paths, test_paths