"""src/utils/yolo_exporter.py 单元测试"""

from pathlib import Path

import pytest
import yaml

from src.core.annotation import Annotation
from src.core.class_manager import ClassManager
from src.utils.yolo_exporter import YOLOExporter, annotation_to_yolo_lines


class TestAnnotationToYoloLines:
    def test_dataclass_input(self, sample_annotations):
        lines = annotation_to_yolo_lines(sample_annotations, 200, 300)
        assert len(lines) == 3
        parts = lines[0].split()
        assert parts[0] == "0"  # class_id 整数化
        assert len(parts) == 5
        assert all(len(p.split(".")[1]) == 6 for p in parts[1:])  # 6 位小数

    def test_dict_input(self):
        dicts = [{"x": 10, "y": 20, "width": 100, "height": 200, "class_id": 2}]
        from_dict = annotation_to_yolo_lines(dicts, 200, 400)
        from_ann = annotation_to_yolo_lines([Annotation(10, 20, 100, 200, 2)], 200, 400)
        assert from_dict == from_ann

    def test_empty_annotations(self):
        assert annotation_to_yolo_lines([], 100, 100) == []


@pytest.fixture
def annotation_manager_in_tmp(tmp_path):
    import os

    from src.core.annotation import AnnotationManager

    am = AnnotationManager()
    am._annotation_dir = str(tmp_path / "annotations")
    os.makedirs(am._annotation_dir, exist_ok=True)
    return am


@pytest.fixture
def class_manager_person():
    cm = ClassManager()
    cm.add_class("person")
    return cm


class TestYOLOExporterValidation:
    def test_bad_split_ratio_raises(self, image_manager, annotation_manager_in_tmp):
        with pytest.raises(ValueError, match="分割比例"):
            YOLOExporter().export(
                image_manager, annotation_manager_in_tmp, ClassManager(),
                "/tmp/unused", split_ratios=(0.5, 0.2, 0.1),
            )

    def test_no_images_raises(self, tmp_path, annotation_manager_in_tmp):
        from src.core.image_manager import ImageManager

        empty = tmp_path / "empty"
        empty.mkdir()
        im = ImageManager()
        im.load_folder(str(empty))
        with pytest.raises(ValueError, match="没有图片"):
            YOLOExporter().export(
                im, annotation_manager_in_tmp, ClassManager(), str(tmp_path / "o")
            )


class TestYOLOExporterFullExport:
    def test_export_structure_and_yaml(
        self, tmp_path, real_image_factory, annotation_manager_in_tmp, class_manager_person
    ):
        from src.core.image_manager import ImageManager

        img = real_image_factory("photo.png", size=(40, 30))
        annotation_manager_in_tmp.save_annotations(img, [Annotation(5, 5, 20, 15, 0)])

        im = ImageManager()
        im.load_folder(str(tmp_path))

        out = tmp_path / "export"
        YOLOExporter().export(im, annotation_manager_in_tmp, class_manager_person, str(out))

        for sub in ("train", "val", "test"):
            assert (out / "images" / sub).is_dir()
            assert (out / "labels" / sub).is_dir()

        # 图片与标签一一对应且同主干名
        label_files = list((out / "labels").glob("*/*.txt"))
        assert len(label_files) == 1
        image_files = list((out / "images").glob("*/*.png"))
        assert len(image_files) == 1
        assert image_files[0].stem == label_files[0].stem == "photo"

        # 标签内容：单行 YOLO 格式
        parts = label_files[0].read_text(encoding="utf-8").strip().split()
        assert len(parts) == 5 and parts[0] == "0"

        # train.txt / val.txt / test.txt 列表文件
        for sub in ("train", "val", "test"):
            assert (out / f"{sub}.txt").exists()

        # data.yaml
        data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        assert data["nc"] == 1
        assert data["names"] == {0: "person"}
        for key in ("path", "train", "val", "test"):
            assert key in data
        assert data["path"] == str(out.resolve())

    def test_export_split_lists_use_relative_paths(
        self, tmp_path, real_image_factory, annotation_manager_in_tmp, class_manager_person
    ):
        from src.core.image_manager import ImageManager

        real_image_factory("a.png")
        real_image_factory("b.png")
        im = ImageManager()
        im.load_folder(str(tmp_path))

        out = tmp_path / "export2"
        YOLOExporter().export(im, annotation_manager_in_tmp, class_manager_person, str(out))

        lines = (out / "train.txt").read_text(encoding="utf-8").splitlines()
        assert lines, "train.txt 不应为空"
        for line in lines:
            # 列表文件必须是 POSIX 风格相对路径：ultralytics 按正斜杠解析，
            # Windows 上写成反斜杠会导致训练找不到图片（回归防护）
            assert line.startswith("images/train/")
            assert "\\" not in line, f"列表文件出现反斜杠（Windows 回归）: {line!r}"
            assert line == f"images/train/{Path(line).name}"


class TestPathListPosixSeparator:
    """列表文件路径分隔符不变量。

    锚点：`images` 与 `<subset>` 之间的分隔符必须是 `/`。旧实现用
    `str(Path("images") / subset / name)`，在 Windows 上产出 `images\\train\\a.png`，
    ultralytics 按 POSIX 风格解析列表文件 → 训练时找不到图片；本断言在
    Windows runner 上直接钉死该行为，Linux 上亦不误报。
    """

    @pytest.mark.parametrize(
        "image_path",
        [
            "a.png",
            r"D:\data\train\b.png",   # Windows 形态（Windows runner 上会按分隔符解析）
            "c d.png",                # 文件名含空格
        ],
    )
    def test_path_list_uses_posix_separator(self, tmp_path, image_path):
        out = tmp_path / "lists"
        YOLOExporter()._export_path_list(out, [image_path], "train")

        line = out.read_text(encoding="utf-8").strip()
        # 文件名提取遵循宿主平台语义，断言只锁定我们拼接的分隔符
        expected = f"images/train/{Path(image_path).name}"
        assert line == expected
        assert line.startswith("images/train/")


class TestYOLOExporterSingleAndCustom:
    def test_export_single_image(self, tmp_path, real_image_factory, class_manager_person):
        img = real_image_factory("single.png", size=(50, 40))
        annotations = [Annotation(5, 5, 25, 20, 0), Annotation(10, 10, 5, 5, 0)]

        out = tmp_path / "single_out"
        YOLOExporter().export_single_image(img, annotations, str(out), class_manager_person)

        label = (out / "single.txt").read_text(encoding="utf-8").splitlines()
        assert len(label) == 2
        assert all(len(line.split()) == 5 for line in label)

        assert (out / "single.png").exists()

        data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        assert data["nc"] == 1 and data["train"] == "."

    def test_export_single_image_empty_annotations(
        self, tmp_path, real_image_factory, class_manager_person
    ):
        img = real_image_factory("blank.png")
        out = tmp_path / "blank_out"
        YOLOExporter().export_single_image(img, [], str(out), class_manager_person)
        label = out / "blank.txt"
        assert label.exists()
        assert label.read_text(encoding="utf-8") == ""

    def test_export_with_custom_split(
        self, tmp_path, real_image_factory, annotation_manager_in_tmp, class_manager_person
    ):
        from src.core.image_manager import ImageManager

        a = real_image_factory("a.png", size=(32, 32))
        b = real_image_factory("b.png", size=(32, 32))
        c = real_image_factory("c.png", size=(32, 32))
        for p in (a, b, c):
            annotation_manager_in_tmp.save_annotations(p, [Annotation(1, 1, 5, 5, 0)])

        im = ImageManager()
        im.load_folder(str(tmp_path))

        out = tmp_path / "custom_out"
        YOLOExporter().export_with_custom_split(
            im, annotation_manager_in_tmp, class_manager_person, str(out),
            train_paths=[a, b], val_paths=[c], test_paths=[],
        )

        assert len(list((out / "images" / "train").glob("*.png"))) == 2
        assert len(list((out / "images" / "val").glob("*.png"))) == 1
        assert len(list((out / "images" / "test").glob("*.png"))) == 0
        assert (out / "data.yaml").exists()

    def test_export_without_copy_images(
        self, tmp_path, real_image_factory, annotation_manager_in_tmp, class_manager_person
    ):
        from src.core.image_manager import ImageManager

        img = real_image_factory("nocopy.png", size=(32, 32))
        annotation_manager_in_tmp.save_annotations(img, [Annotation(1, 1, 5, 5, 0)])

        im = ImageManager()
        im.load_folder(str(tmp_path))

        out = tmp_path / "nocopy_out"
        YOLOExporter().export(
            im, annotation_manager_in_tmp, class_manager_person, str(out), copy_images=False
        )

        # 不复制图片，只生成标签
        assert list((out / "images").glob("*/*.png")) == []
        assert len(list((out / "labels").glob("*/*.txt"))) == 1


class TestValidateExport:
    def test_valid_export_passes(
        self, tmp_path, real_image_factory, annotation_manager_in_tmp, class_manager_person
    ):
        from src.core.image_manager import ImageManager

        img = real_image_factory("v.png", size=(32, 32))
        annotation_manager_in_tmp.save_annotations(img, [Annotation(1, 1, 5, 5, 0)])
        im = ImageManager()
        im.load_folder(str(tmp_path))

        out = tmp_path / "v_out"
        YOLOExporter().export(im, annotation_manager_in_tmp, class_manager_person, str(out))

        result = YOLOExporter().validate_export(str(out))
        assert result["valid"] is True
        assert result["errors"] == []
        total = sum(s["images"] for s in result["statistics"].values())
        assert total == 1

    def test_invalid_dir_fails(self, tmp_path):
        out = tmp_path / "nothing"
        out.mkdir()
        result = YOLOExporter().validate_export(str(out))
        assert result["valid"] is False
        assert any("目录不存在" in e for e in result["errors"])
        assert any("data.yaml" in e for e in result["errors"])
