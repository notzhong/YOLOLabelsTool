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
    EditAnnotationsCommand,
    PolygonAnnotation,
    annotation_from_dict,
)

# ==================== Annotation 数据结构 ====================

class TestAnnotation:
    def test_to_dict_contains_all_fields(self, sample_annotation):
        d = sample_annotation.to_dict()
        assert d == {"x": 10, "y": 20, "width": 100, "height": 200, "class_id": 0, "type": "box"}

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


# ==================== 未加载图片时的防御 ====================
#
# 回归：未加载任何图片就标注 → AddAnnotationCommand.execute 拿 None 当
# image_path 查标注 → 一路炸到 Path(None) 的 TypeError（traceback 里全是
# pathlib 内部细节，看不出真正原因是"没加载图片"）。约定：
#   读路径（get/has/load）容忍空路径，返回空结果，不抛异常；
#   写路径（save/clear）与 get_annotation_path 快速失败，抛 ValueError。

class TestNoImageLoadedDefense:
    """image_path 为 None/空串（未加载图片）时的行为契约"""

    def test_get_annotation_path_rejects_empty(self, annotation_manager):
        with pytest.raises(ValueError, match="image_path 不能为空"):
            annotation_manager.get_annotation_path(None)

    def test_get_annotation_path_rejects_empty_string(self, annotation_manager):
        with pytest.raises(ValueError, match="image_path 不能为空"):
            annotation_manager.get_annotation_path("")

    def test_legacy_annotation_path_rejects_empty(self, annotation_manager):
        with pytest.raises(ValueError, match="image_path 不能为空"):
            annotation_manager._legacy_annotation_path(None)

    def test_get_annotations_returns_empty_for_none(self, annotation_manager):
        assert annotation_manager.get_annotations(None) == []

    def test_load_annotations_returns_empty_for_none(self, annotation_manager):
        assert annotation_manager.load_annotations(None) == []

    def test_has_annotations_false_for_none(self, annotation_manager):
        assert annotation_manager.has_annotations(None) is False

    def test_save_annotations_raises_for_none(self, annotation_manager):
        with pytest.raises(ValueError, match="拒绝保存标注"):
            annotation_manager.save_annotations(None, [Annotation(0, 0, 10, 10, 0)])

    def test_clear_annotations_raises_for_none(self, annotation_manager):
        with pytest.raises(ValueError, match="拒绝清除标注"):
            annotation_manager.clear_annotations(None)

    def test_add_command_with_none_path_raises_valueerror_not_typeerror(
        self, annotation_manager
    ):
        """原始崩溃路径：命令执行必须抛 ValueError，而不是 pathlib 的 TypeError"""
        cmd = AddAnnotationCommand(annotation_manager, None, Annotation(0, 0, 10, 10, 0))
        with pytest.raises(ValueError):
            annotation_manager.execute_command(cmd)

    def test_failed_add_command_leaves_no_undo_entry(self, annotation_manager):
        """执行失败的命令不得进撤销栈，否则 Ctrl+Z 会重放到无效命令"""
        cmd = AddAnnotationCommand(annotation_manager, None, Annotation(0, 0, 10, 10, 0))
        with pytest.raises(ValueError):
            annotation_manager.execute_command(cmd)
        assert annotation_manager.can_undo() is False

    def test_none_path_does_not_pollute_cache_or_statistics(self, annotation_manager):
        """空路径不得写入 _annotations，否则统计会把 None 当成一张图"""
        annotation_manager.get_annotations(None)
        assert None not in annotation_manager._annotations
        assert annotation_manager.get_statistics()["total_images"] == 0


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


# ==================== PolygonAnnotation 数据结构 ====================

class TestPolygonAnnotation:
    def test_to_dict_format(self):
        poly = PolygonAnnotation(points=[(1, 2), (3.5, 4), (5, 6)], class_id=2)
        assert poly.to_dict() == {
            "type": "polygon",
            "points": [[1.0, 2.0], [3.5, 4.0], [5.0, 6.0]],
            "class_id": 2,
        }

    def test_from_dict_roundtrip(self):
        poly = PolygonAnnotation(points=[(1.5, 2.5), (3.0, 4.0), (5.0, 6.0)], class_id=1)
        assert PolygonAnnotation.from_dict(poly.to_dict()) == poly

    def test_from_dict_missing_points_defaults_empty(self):
        poly = PolygonAnnotation.from_dict({})
        assert poly.points == [] and poly.class_id == 0

    def test_bounding_box(self):
        poly = PolygonAnnotation(points=[(10, 20), (40, 20), (40, 70), (10, 70)], class_id=0)
        assert poly.bounding_box() == (10, 20, 30, 50)

    def test_bounding_box_empty(self):
        assert PolygonAnnotation(points=[], class_id=0).bounding_box() == (0.0, 0.0, 0.0, 0.0)

    def test_to_yolo_seg_format_exact_values(self):
        poly = PolygonAnnotation(points=[(10, 20), (30, 20), (30, 60)], class_id=3)
        data = poly.to_yolo_seg_format(100, 200)
        assert data[0] == 3
        assert data[1:] == pytest.approx([0.1, 0.1, 0.3, 0.1, 0.3, 0.3])

    def test_from_yolo_seg_format_roundtrip(self):
        poly = PolygonAnnotation(points=[(10.0, 20.0), (30.0, 20.0), (30.0, 60.0)], class_id=3)
        yolo = poly.to_yolo_seg_format(100, 200)
        restored = PolygonAnnotation.from_yolo_seg_format(yolo, 100, 200)
        assert restored.class_id == 3
        for (rx, ry), (ox, oy) in zip(restored.points, poly.points, strict=False):
            assert (rx, ry) == pytest.approx((ox, oy))

    def test_from_yolo_seg_format_truncates_class_id(self):
        poly = PolygonAnnotation.from_yolo_seg_format(
            [2.9, 0.1, 0.1, 0.3, 0.1, 0.3, 0.3], 100, 100
        )
        assert poly.class_id == 2

    @pytest.mark.parametrize(
        "bad",
        [
            [0, 0.1, 0.1, 0.3, 0.1],  # 5 字段（框格式）
            [0, 0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.5],  # 偶数长度
        ],
    )
    def test_from_yolo_seg_format_wrong_length_raises(self, bad):
        with pytest.raises(ValueError, match=r"1\+2k"):
            PolygonAnnotation.from_yolo_seg_format(bad, 100, 100)

    def test_from_box_creates_rect_corners(self):
        box = Annotation(x=10, y=20, width=30, height=40, class_id=1)
        poly = PolygonAnnotation.from_box(box)
        assert poly.points == [(10, 20), (40, 20), (40, 60), (10, 60)]
        assert poly.class_id == 1

    def test_annotation_from_polygon_bbox(self):
        poly = PolygonAnnotation(points=[(10, 20), (40, 20), (40, 70), (10, 70)], class_id=2)
        box = Annotation.from_polygon(poly)
        assert (box.x, box.y, box.width, box.height, box.class_id) == (10, 20, 30, 50, 2)


class TestAnnotationFromDict:
    def test_polygon_dispatched(self):
        ann = annotation_from_dict(
            {"type": "polygon", "points": [[1, 2], [3, 4], [5, 6]], "class_id": 1}
        )
        assert isinstance(ann, PolygonAnnotation)
        assert ann.class_id == 1

    def test_legacy_file_without_type_is_box(self):
        ann = annotation_from_dict({"x": 1, "y": 2, "width": 3, "height": 4, "class_id": 0})
        assert isinstance(ann, Annotation)

    def test_explicit_box_type(self):
        ann = annotation_from_dict(
            {"type": "box", "x": 1, "y": 2, "width": 3, "height": 4, "class_id": 0}
        )
        assert isinstance(ann, Annotation)


class TestEditAnnotationsCommand:
    def test_execute_undo_redo(self, annotation_manager):
        old = [Annotation(0, 0, 10, 10, 0)]
        new = [
            Annotation(5, 5, 10, 10, 0),
            PolygonAnnotation(points=[(1, 1), (2, 1), (2, 2)], class_id=1),
        ]
        annotation_manager.save_annotations("img.png", list(old))
        cmd = EditAnnotationsCommand(annotation_manager, "img.png", old, new)
        annotation_manager.execute_command(cmd)
        assert [a.to_dict() for a in annotation_manager.get_annotations("img.png")] == [
            a.to_dict() for a in new
        ]

        assert annotation_manager.undo() is True
        assert [a.to_dict() for a in annotation_manager.get_annotations("img.png")] == [
            a.to_dict() for a in old
        ]

        assert annotation_manager.redo() is True
        assert [a.to_dict() for a in annotation_manager.get_annotations("img.png")] == [
            a.to_dict() for a in new
        ]

    def test_snapshots_isolated_from_later_mutations(self, annotation_manager):
        old = [PolygonAnnotation(points=[(0, 0), (10, 0), (10, 10)], class_id=0)]
        new = [PolygonAnnotation(points=[(0, 0), (20, 0), (20, 20)], class_id=0)]
        cmd = EditAnnotationsCommand(annotation_manager, "img.png", old, new)
        # 构造后原地修改传入对象，不应影响命令内部快照
        new[0].points[0] = (999, 999)
        old[0].points[0] = (888, 888)
        annotation_manager.execute_command(cmd)
        assert annotation_manager.get_annotations("img.png")[0].points[0] == (0, 0)
        annotation_manager.undo()
        assert annotation_manager.get_annotations("img.png")[0].points[0] == (0, 0)

    def test_empty_new_state_clears_file_undoable(self, annotation_manager):
        old = [Annotation(0, 0, 10, 10, 0)]
        annotation_manager.save_annotations("img.png", list(old))
        cmd = EditAnnotationsCommand(annotation_manager, "img.png", old, [])
        annotation_manager.execute_command(cmd)
        assert annotation_manager.get_annotations("img.png") == []
        annotation_manager.undo()
        assert [a.to_dict() for a in annotation_manager.get_annotations("img.png")] == [
            a.to_dict() for a in old
        ]


# ==================== 多边形持久化 / 导入导出 / 统计 ====================

class TestSegmentationPersistence:
    def test_save_and_load_mixed_types(self, annotation_manager):
        image = "/data/mixed.png"
        annotations = [
            Annotation(1, 2, 3, 4, 0),
            PolygonAnnotation(points=[(10, 10), (50, 10), (50, 50)], class_id=1),
        ]
        annotation_manager.save_annotations(image, annotations)

        data = json.loads(
            Path(annotation_manager.get_annotation_path(image)).read_text(encoding="utf-8")
        )
        assert data[0]["type"] == "box"
        assert data[1]["type"] == "polygon"

        loaded = annotation_manager.load_annotations(image)
        assert isinstance(loaded[0], Annotation)
        assert isinstance(loaded[1], PolygonAnnotation)
        assert loaded == annotations

    def test_statistics_merges_box_and_polygon(self, annotation_manager):
        annotation_manager.save_annotations(
            "/a.png",
            [Annotation(0, 0, 1, 1, 0), PolygonAnnotation(points=[(0, 0), (1, 0), (1, 1)], class_id=0)],
        )
        annotation_manager.save_annotations(
            "/b.png", [PolygonAnnotation(points=[(0, 0), (1, 0), (1, 1)], class_id=2)]
        )
        stats = annotation_manager.get_statistics()
        assert stats["total_images"] == 2
        assert stats["total_annotations"] == 3
        assert stats["class_counts"] == {0: 2, 2: 1}


class TestSegmentationYoloIO:
    def test_import_seg_lines(self, annotation_manager):
        lines = [
            "0 0.1 0.2 0.3 0.2 0.3 0.6",           # 3 顶点多边形
            "1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6",   # 4 顶点多边形
            "0 0.5 0.5 0.1 0.1",                    # 框
        ]
        imported, skipped = annotation_manager.import_from_yolo_format("/a.png", lines, 100, 200)
        assert (imported, skipped) == (3, 0)

        anns = annotation_manager.get_annotations("/a.png")
        assert isinstance(anns[0], PolygonAnnotation) and anns[0].class_id == 0
        assert len(anns[0].points) == 3
        assert anns[0].points[0] == pytest.approx((10.0, 40.0))
        assert isinstance(anns[1], PolygonAnnotation) and len(anns[1].points) == 4
        assert isinstance(anns[2], Annotation)

    def test_import_invalid_line_counts(self, annotation_manager):
        lines = [
            "",                                 # 空行：忽略且不计入跳过
            "0 0.1 0.2",                         # 3 字段：无效
            "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7",     # 偶数长度：无效
            "a b c d e f g",                     # 非数字：无效
            "0 0.5 0.5 0.2 0.2",                 # 合法框
        ]
        imported, skipped = annotation_manager.import_from_yolo_format("/b.png", lines, 100, 100)
        assert (imported, skipped) == (1, 3)

    def test_seg_export_roundtrip(self, annotation_manager):
        image = "/data/poly.png"
        polys = [
            PolygonAnnotation(points=[(10.0, 20.0), (30.0, 20.0), (30.0, 60.0)], class_id=1),
        ]
        annotation_manager.save_annotations(image, polys)
        lines = annotation_manager.export_to_yolo_format(image, 100, 200, task="segment")
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "1" and len(parts) == 7
        assert all(len(p.split(".")[1]) == 6 for p in parts[1:])

        target = "/data/poly2.png"
        annotation_manager.import_from_yolo_format(target, lines, 100, 200)
        restored = annotation_manager.get_annotations(target)[0]
        assert isinstance(restored, PolygonAnnotation)
        for (rx, ry), (ox, oy) in zip(restored.points, polys[0].points, strict=False):
            assert (rx, ry) == pytest.approx((ox, oy), abs=1e-3)

    def test_detect_export_of_polygon_uses_bbox(self, annotation_manager):
        image = "/data/poly.png"
        annotation_manager.save_annotations(
            image,
            [PolygonAnnotation(points=[(10, 20), (40, 20), (40, 70), (10, 70)], class_id=0)],
        )
        lines = annotation_manager.export_to_yolo_format(image, 100, 100)
        parts = lines[0].split()
        assert len(parts) == 5
        # 外接框中心 (0.25, 0.45)，宽高 (0.30, 0.50)
        assert float(parts[1]) == pytest.approx(0.25)
        assert float(parts[2]) == pytest.approx(0.45)
        assert float(parts[3]) == pytest.approx(0.30)
        assert float(parts[4]) == pytest.approx(0.50)

    def test_seg_export_expands_box_to_rect(self, annotation_manager):
        image = "/data/box.png"
        annotation_manager.save_annotations(image, [Annotation(10, 20, 30, 40, 0)])
        lines = annotation_manager.export_to_yolo_format(image, 100, 200, task="segment")
        parts = lines[0].split()
        # class + 4 顶点 × 2 坐标 = 9 字段
        assert len(parts) == 9
        assert [float(p) for p in parts[1:]] == pytest.approx(
            [0.10, 0.10, 0.40, 0.10, 0.40, 0.30, 0.10, 0.30]
        )

