"""src/utils/dataset_splitter.py 单元测试"""

import pytest

from src.utils.dataset_splitter import DatasetSplitter, random_split

IMAGES = [f"/data/img_{i:02d}.png" for i in range(10)]


class TestRandomSplit:
    def test_counts_match_ratios(self):
        train, val, test = random_split(IMAGES, (0.7, 0.2, 0.1))
        assert (len(train), len(val), len(test)) == (7, 2, 1)

    def test_no_loss_no_overlap(self):
        train, val, test = random_split(IMAGES, (0.6, 0.3, 0.1))
        combined = train + val + test
        assert sorted(combined) == sorted(IMAGES)
        assert len(combined) == len(set(combined))

    def test_deterministic_with_seed(self):
        r1 = random_split(IMAGES, (0.7, 0.2, 0.1), seed=123)
        r2 = random_split(IMAGES, (0.7, 0.2, 0.1), seed=123)
        assert r1 == r2

    def test_input_list_not_mutated(self):
        original = list(IMAGES)
        random_split(IMAGES, (0.7, 0.2, 0.1))
        assert IMAGES == original

    def test_ratios_do_not_sum_to_one_raises(self):
        with pytest.raises(ValueError, match="分割比例"):
            random_split(IMAGES, (0.5, 0.2, 0.1))

    def test_tiny_dataset_still_covers_all(self):
        # int(1*0.7)=0, int(1*0.2)=0 -> 全部落入 test
        train, val, test = random_split(["/only.png"], (0.7, 0.2, 0.1))
        assert train == [] and val == []
        assert test == ["/only.png"]

    def test_empty_input(self):
        assert random_split([], (0.7, 0.2, 0.1)) == ([], [], [])


@pytest.fixture
def real_dataset_factory(tmp_path, real_image_factory):
    """创建 N 张真实图片，返回 (路径列表, ImageManager)"""
    def _create(n=10, prefix="img"):
        from src.core.image_manager import ImageManager

        paths = [real_image_factory(f"{prefix}_{i}.png", size=(32, 24)) for i in range(n)]
        im = ImageManager()
        assert im.load_folder(str(tmp_path)) is True
        return paths, im

    return _create


@pytest.fixture
def annotation_manager_in_tmp(tmp_path):
    """标注目录指向 tmp_path 的 AnnotationManager"""
    import os

    from src.core.annotation import AnnotationManager

    am = AnnotationManager()
    am._annotation_dir = str(tmp_path / "annotations")
    os.makedirs(am._annotation_dir, exist_ok=True)
    return am


class TestDatasetSplitterSplitAndExport:
    def test_full_export_with_copy(self, tmp_path, real_dataset_factory, annotation_manager_in_tmp):
        from src.core.annotation import Annotation
        from src.core.class_manager import ClassManager

        paths, im = real_dataset_factory(10)
        am = annotation_manager_in_tmp
        cm = ClassManager()
        cm.add_class("person")

        for p in paths:
            am.save_annotations(p, [Annotation(1, 1, 5, 5, 0)])

        out = tmp_path / "dataset_out"
        DatasetSplitter().split_and_export(im, am, str(out), class_manager=cm)

        for sub in ("train", "val", "test"):
            assert (out / "images" / sub).is_dir()
            assert (out / "labels" / sub).is_dir()

        total_images = sum(
            len(list((out / "images" / s).glob("*.png"))) for s in ("train", "val", "test")
        )
        total_labels = sum(
            len(list((out / "labels" / s).glob("*.txt"))) for s in ("train", "val", "test")
        )
        assert total_images == 10
        assert total_labels == 10

        # 划分列表文件为绝对路径（DatasetSplitter._export_path_list 的行为；
        # 相对路径格式 images/<subset>/ 仅在 YOLOExporter._export_path_list 中使用）
        train_list = (out / "train.txt").read_text(encoding="utf-8").split()
        assert train_list and all(n.startswith(str(tmp_path)) for n in train_list)

        assert (out / "data.yaml").exists()
        assert (out / "split_statistics.txt").exists()

    def test_split_files_consistent_with_counts(self, tmp_path, real_dataset_factory, annotation_manager_in_tmp):
        from src.core.annotation import Annotation

        paths, im = real_dataset_factory(10)
        am = annotation_manager_in_tmp
        for p in paths:
            am.save_annotations(p, [Annotation(1, 1, 5, 5, 0)])

        out = tmp_path / "out2"
        DatasetSplitter().split_and_export(im, am, str(out))

        counts = {
            sub: len(list((out / "labels" / sub).glob("*.txt")))
            for sub in ("train", "val", "test")
        }
        listed = {
            sub: len((out / f"{sub}.txt").read_text(encoding="utf-8").splitlines())
            for sub in ("train", "val", "test")
        }
        assert counts == listed
        assert sum(counts.values()) == 10

    def test_no_images_raises(self, tmp_path):
        from src.core.image_manager import ImageManager

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        im = ImageManager()
        im.load_folder(str(empty_dir))
        with pytest.raises(ValueError, match="没有图片"):
            DatasetSplitter().split_and_export(im, None, str(tmp_path / "out"))

    def test_without_class_manager_no_yaml(self, tmp_path, real_dataset_factory, annotation_manager_in_tmp):
        from src.core.annotation import Annotation

        paths, im = real_dataset_factory(4)
        am = annotation_manager_in_tmp
        for p in paths:
            am.save_annotations(p, [Annotation(1, 1, 5, 5, 0)])

        out = tmp_path / "out3"
        DatasetSplitter().split_and_export(im, am, str(out), class_manager=None)
        assert not (out / "data.yaml").exists()


class TestCrossValidation:
    def test_folds_partition_dataset(self):
        folds = DatasetSplitter().create_cross_validation_splits(IMAGES, n_folds=5, random_seed=1)
        assert len(folds) == 5
        all_val = []
        for train, val in folds:
            assert len(set(train) & set(val)) == 0
            assert len(train) + len(val) == 10
            all_val.extend(val)
        # 每个样本恰好作为验证集一次
        assert sorted(all_val) == sorted(IMAGES)

    def test_deterministic(self):
        s = DatasetSplitter()
        f1 = s.create_cross_validation_splits(IMAGES, 3, 7)
        f2 = s.create_cross_validation_splits(IMAGES, 3, 7)
        assert f1 == f2


class TestStratifiedAndBalance:
    def test_stratified_split_partitions(self, tmp_path, annotation_manager_in_tmp):
        from src.core.annotation import Annotation

        am = annotation_manager_in_tmp
        # 6 张只含 class 0，4 张只含 class 1
        for i in range(10):
            cls = 0 if i < 6 else 1
            am.save_annotations(IMAGES[i], [Annotation(0, 0, 5, 5, cls)])

        train, val, test = DatasetSplitter().stratified_split(
            IMAGES, am, (0.5, 0.25, 0.25), random_seed=3
        )
        combined = train + val + test
        assert sorted(combined) == sorted(IMAGES)
        assert len(combined) == len(set(combined))

    def test_balance_classes_downsamples(self, tmp_path, annotation_manager_in_tmp):
        from src.core.annotation import Annotation

        am = annotation_manager_in_tmp
        # class 0 有 8 张，class 1 有 2 张
        for i in range(10):
            cls = 0 if i < 8 else 1
            am.save_annotations(IMAGES[i], [Annotation(0, 0, 5, 5, cls)])

        balanced = DatasetSplitter().balance_classes(
            IMAGES, am, max_samples_per_class=2, min_samples_per_class=2
        )
        assert len(balanced) == 4  # 每类下采样到 2
