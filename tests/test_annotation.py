"""src/core/annotation.py 单元测试：Annotation 数据结构 + 命令模式撤销/重做 + AnnotationManager 持久化"""

import json
import os
from pathlib import Path

import pytest

from src.core.annotation import (
    AddAnnotationCommand,
    Annotation,
    AnnotationManager,
    Command,
    DeleteAnnotationCommand,
)

# ==================== Annotation 数据结构 ====================

class TestAnnotation:
    def test_to_dict_contains_all_fields(self, sample_annotation):
        d = sample_annotation.to_dict()
        assert d == {"x": 10, "y": 20, "width": 100, "height": 200, "class_id": 0}

    def test_from_dict_roundtrip(self, sample_annotation):
        restored = Annotation.from_dict(sample_annotation.to_dict())
        assert restored == sample_annotation

    def test_from_dict_missing_keys_default_to_zero(self):
        ann = Annotation.from_dict({})
        assert (ann.x, ann.y, ann.width, ann.height, ann.class_id) == (0, 0, 0, 0, 0)

    def test_to_yolo_format_exact_values(self):
        ann = Annotation(x=10, y=20, width=100, height=200, class_id=2)
        data = ann.to_yolo_format(image_width=200, image_height=400)
        assert data[0] == 2
        assert data[1] == pytest.approx((10 + 50) / 200)  # x_center
        assert data[2] == pytest.approx((20 + 100) / 400)  # y_center
        assert data[3] == pytest.approx(100 / 200)  # norm_width
        assert data[4] == pytest.approx(200 / 400)  # norm_height

    def test_from_yolo_format_roundtrip(self):
        ann = Annotation(x=10, y=20, width=100, height=200, class_id=2)
        yolo = ann.to_yolo_format(200, 400)
        restored = Annotation.from_yolo_format(yolo, 200, 400)
        assert restored.x == pytest.approx(10)
        assert restored.y == pytest.approx(20)
        assert restored.width == pytest.approx(100)
        assert restored.height == pytest.approx(200)
        assert restored.class_id == 2

    def test_from_yolo_format_truncates_class_id(self):
        ann = Annotation.from_yolo_format([2.9, 0.5, 0.5, 0.1, 0.1], 100, 100)
        assert ann.class_id == 2

    def test_from_yolo_format_wrong_length_raises(self):
        with pytest.raises(ValueError, match="长度应为5"):
            Annotation.from_yolo_format([0, 0.5, 0.5, 0.1], 100, 100)

    def test_to_yolo_format_zero_size_box(self):
        ann = Annotation(x=50, y=50, width=0, height=0, class_id=0)
        data = ann.to_yolo_format(100, 100)
        assert data == [0, 0.5, 0.5, 0.0, 0.0]


# ==================== 命令模式撤销/重做 ====================

class TestCommandUndoRedo:
    def test_command_base_is_noop(self, annotation_manager):
        cmd = Command(annotation_manager, "img.png")
        cmd.execute()
        cmd.undo()
        cmd.redo()
        assert cmd.old_state is None and cmd.new_state is None

    def test_add_command_execute_and_undo(self, annotation_manager):
        ann = Annotation(0, 0, 10, 10, 0)
        cmd = AddAnnotationCommand(annotation_manager, "img.png", ann)
        annotation_manager.execute_command(cmd)
        assert len(annotation_manager.get_annotations("img.png")) == 1

        assert annotation_manager.undo() is True
        assert annotation_manager.get_annotations("img.png") == []
        assert annotation_manager.redo() is True
        assert annotation_manager.get_annotations("img.png") == [ann]

    def test_delete_command_execute_and_undo(self, annotation_manager):
        a1, a2 = Annotation(0, 0, 10, 10, 0), Annotation(5, 5, 10, 10, 1)
        annotation_manager.save_annotations("img.png", [a1, a2])

        cmd = DeleteAnnotationCommand(annotation_manager, "img.png", 0)
        annotation_manager.execute_command(cmd)
        assert annotation_manager.get_annotations("img.png") == [a2]

        annotation_manager.undo()
        assert annotation_manager.get_annotations("img.png") == [a1, a2]

    def test_delete_command_out_of_range_is_noop(self, annotation_manager):
        annotation_manager.save_annotations("img.png", [])
        cmd = DeleteAnnotationCommand(annotation_manager, "img.png", 5)
        annotation_manager.execute_command(cmd)
        assert annotation_manager.get_annotations("img.png") == []

    def test_new_command_clears_redo_stack(self, annotation_manager):
        annotation_manager.execute_command(
            AddAnnotationCommand(annotation_manager, "img.png", Annotation(0, 0, 10, 10, 0))
        )
        annotation_manager.undo()
        assert annotation_manager.can_redo() is True
        annotation_manager.execute_command(
            AddAnnotationCommand(annotation_manager, "img.png", Annotation(1, 1, 10, 10, 1))
        )
        assert annotation_manager.can_redo() is False

    def test_undo_redo_empty_stacks_return_false(self, annotation_manager):
        assert annotation_manager.undo() is False
        assert annotation_manager.redo() is False
        assert annotation_manager.can_undo() is False
        assert annotation_manager.can_redo() is False

    def test_multi_step_history(self, annotation_manager):
        a1, a2 = Annotation(0, 0, 10, 10, 0), Annotation(5, 5, 10, 10, 1)
        annotation_manager.execute_command(AddAnnotationCommand(annotation_manager, "img.png", a1))
        annotation_manager.execute_command(AddAnnotationCommand(annotation_manager, "img.png", a2))

        annotation_manager.undo()
        assert annotation_manager.get_annotations("img.png") == [a1]
        annotation_manager.undo()
        assert annotation_manager.get_annotations("img.png") == []
        annotation_manager.redo()
        assert annotation_manager.get_annotations("img.png") == [a1]
        annotation_manager.redo()
        assert annotation_manager.get_annotations("img.png") == [a1, a2]
        assert annotation_manager.can_redo() is False

    def test_undo_stack_size_limited(self, annotation_manager):
        for i in range(AnnotationManager.MAX_UNDO_SIZE + 5):
            annotation_manager.execute_command(
                AddAnnotationCommand(annotation_manager, "img.png", Annotation(i, i, 5, 5, 0))
            )
        assert len(annotation_manager._undo_stack) == AnnotationManager.MAX_UNDO_SIZE

    def test_clear_history(self, annotation_manager):
        annotation_manager.execute_command(
            AddAnnotationCommand(annotation_manager, "img.png", Annotation(0, 0, 10, 10, 0))
        )
        annotation_manager.undo()
        annotation_manager.clear_history()
        assert annotation_manager.can_undo() is False
        assert annotation_manager.can_redo() is False


# ==================== AnnotationManager 持久化 ====================

class TestAnnotationManagerPersistence:
    def test_annotation_path_has_short_hash(self, annotation_manager):
        p = annotation_manager.get_annotation_path("/data/images/cat.png")
        name = Path(p).name
        assert name.startswith("cat_") and name.endswith(".json")

    def test_save_empty_deletes_file(self, annotation_manager, sample_annotations):
        image = "/data/img1.png"
        annotation_manager.save_annotations(image, sample_annotations)
        path = annotation_manager.get_annotation_path(image)
        assert os.path.exists(path)

        annotation_manager.save_annotations(image, [])
        assert not os.path.exists(path)
        assert annotation_manager.get_annotations(image) == []

    def test_load_corrupt_json_returns_empty(self, annotation_manager):
        image = "/data/broken.png"
        Path(annotation_manager.get_annotation_path(image)).write_text(
            "{not valid json", encoding="utf-8"
        )
        assert annotation_manager.load_annotations(image) == []

    def test_legacy_file_discovered_and_migrated(self, annotation_manager, sample_annotations):
        """旧版仅用文件名存储的标注应能被找到，并在加载后迁移到带 hash 的新路径"""
        image = "/data/legacy.png"
        legacy_path = Path(annotation_manager._legacy_annotation_path(image))
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(
            json.dumps([ann.to_dict() for ann in sample_annotations]), encoding="utf-8"
        )

        assert annotation_manager.has_annotations(image) is True
        loaded = annotation_manager.load_annotations(image)
        assert loaded == sample_annotations
        # 迁移后新路径存在且内容一致
        assert Path(annotation_manager.get_annotation_path(image)).exists()

    def test_get_annotations_prefers_cache(self, annotation_manager, sample_annotations):
        image = "/data/img1.png"
        annotation_manager.save_annotations(image, sample_annotations)
        # 绕过缓存直接篡改磁盘文件，缓存仍应生效
        Path(annotation_manager.get_annotation_path(image)).write_text("[]", encoding="utf-8")
        assert annotation_manager.get_annotations(image) == sample_annotations

    def test_add_annotation_appends(self, annotation_manager, sample_annotations):
        image = "/data/img1.png"
        # 注意：save_annotations 按引用缓存列表，这里传入副本以免断言列表被原地修改
        annotation_manager.save_annotations(image, list(sample_annotations))
        extra = Annotation(1, 2, 3, 4, 9)
        annotation_manager.add_annotation(image, extra)
        assert annotation_manager.get_annotations(image) == sample_annotations + [extra]

    def test_delete_annotation_valid_index(self, annotation_manager, sample_annotations):
        image = "/data/img1.png"
        annotation_manager.save_annotations(image, list(sample_annotations))
        annotation_manager.delete_annotation(image, 1)
        result = annotation_manager.get_annotations(image)
        assert len(result) == 2
        assert sample_annotations[1].to_dict() not in [a.to_dict() for a in result]
        assert sample_annotations[0].to_dict() in [a.to_dict() for a in result]

    @pytest.mark.parametrize("bad_index", [-1, 99])
    def test_delete_annotation_invalid_index_noop(
        self, annotation_manager, sample_annotations, bad_index
    ):
        image = "/data/img1.png"
        annotation_manager.save_annotations(image, list(sample_annotations))
        annotation_manager.delete_annotation(image, bad_index)
        assert [a.to_dict() for a in annotation_manager.get_annotations(image)] == [
            a.to_dict() for a in sample_annotations
        ]

    def test_clear_annotations_removes_files(self, annotation_manager, sample_annotations):
        image = "/data/legacy.png"
        annotation_manager.save_annotations(image, sample_annotations)
        legacy = Path(annotation_manager._legacy_annotation_path(image))
        legacy.write_text("[]", encoding="utf-8")

        annotation_manager.clear_annotations(image)
        assert not Path(annotation_manager.get_annotation_path(image)).exists()
        assert not legacy.exists()
        assert annotation_manager.get_annotations(image) == []

    def test_get_all_annotations_returns_copy(self, annotation_manager, sample_annotations):
        annotation_manager.save_annotations("/a.png", sample_annotations)
        all_anns = annotation_manager.get_all_annotations()
        all_anns["/injected.png"] = []
        assert "/injected.png" not in annotation_manager.get_all_annotations()


# ==================== YOLO 导入/导出与统计 ====================

class TestAnnotationManagerYoloIO:
    def test_export_to_yolo_format_lines(self, annotation_manager, sample_annotations):
        image = "/data/img.png"
        annotation_manager.save_annotations(image, sample_annotations)
        lines = annotation_manager.export_to_yolo_format(image, 200, 300)
        assert len(lines) == 3
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            # 归一化坐标均在 [0, 1] 内
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])

    def test_import_export_roundtrip(self, annotation_manager, sample_annotations):
        image = "/data/img.png"
        annotation_manager.save_annotations(image, sample_annotations)
        lines = annotation_manager.export_to_yolo_format(image, 200, 300)

        target = "/data/other.png"
        annotation_manager.import_from_yolo_format(target, lines, 200, 300)
        restored = annotation_manager.get_annotations(target)
        assert len(restored) == 3
        for orig, got in zip(sample_annotations, restored, strict=False):
            assert got.x == pytest.approx(orig.x, abs=1e-3)
            assert got.y == pytest.approx(orig.y, abs=1e-3)
            assert got.class_id == orig.class_id

    def test_import_skips_blank_and_malformed_lines(self, annotation_manager):
        lines = [
            "",
            "   ",
            "1 2 3 4",  # 字段数不足
            "not a number line",
            "0 0.5 0.5 0.1 0.1",  # 唯一合法行
        ]
        annotation_manager.import_from_yolo_format("/a.png", lines, 100, 100)
        assert len(annotation_manager.get_annotations("/a.png")) == 1

    def test_get_statistics(self, annotation_manager, sample_annotations):
        annotation_manager.save_annotations("/a.png", sample_annotations)
        annotation_manager.save_annotations("/b.png", [Annotation(0, 0, 1, 1, 1)])
        stats = annotation_manager.get_statistics()
        assert stats["total_images"] == 2
        assert stats["total_annotations"] == 4
        assert stats["class_counts"] == {0: 2, 1: 2}

