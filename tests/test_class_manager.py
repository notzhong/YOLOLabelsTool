"""src/core/class_manager.py 单元测试"""

import json

import yaml

from src.core.class_manager import ClassManager


class TestClassCRUD:
    def test_initial_state_empty(self, class_manager):
        assert class_manager.get_class_count() == 0
        assert class_manager.get_classes() == {}
        assert class_manager.get_next_available_class_id() == 0

    def test_add_class_assigns_sequential_ids(self, class_manager):
        assert class_manager.add_class("person") == 0
        assert class_manager.add_class("car") == 1
        assert class_manager.get_next_available_class_id() == 2

    def test_add_class_same_name_returns_existing_id(self, class_manager):
        first = class_manager.add_class("person")
        again = class_manager.add_class("person")
        assert first == again == 0
        assert class_manager.get_class_count() == 1

    def test_add_class_generates_palette_color(self, class_manager):
        class_id = class_manager.add_class("person")
        assert class_manager.get_class_color(class_id) == ClassManager._COLOR_PALETTE[0]

    def test_add_class_generated_colors_do_not_repeat(self, class_manager):
        seen = set()
        for i in range(len(ClassManager._COLOR_PALETTE)):
            color = class_manager.get_class_color(class_manager.add_class(f"c{i}"))
            assert color not in seen
            seen.add(color)

    def test_update_class(self, class_manager_with_classes):
        cm = class_manager_with_classes
        cm.update_class(1, "truck", (9, 9, 9))
        info = cm.get_class(1)
        assert info["name"] == "truck" and info["color"] == (9, 9, 9)

    def test_update_missing_class_is_noop(self, class_manager):
        cm = class_manager
        cm.update_class(42, "ghost", (0, 0, 0))
        assert cm.get_class(42) is None

    def test_delete_class_updates_next_id(self, class_manager_with_classes):
        cm = class_manager_with_classes
        cm.delete_class(2)  # 删除最大 ID
        assert cm.get_class(2) is None
        assert cm.get_next_available_class_id() == 2

    def test_delete_missing_class_is_noop(self, class_manager):
        class_manager.delete_class(99)
        assert class_manager.get_class_count() == 0

    def test_add_or_update_class_existing_updates(self, class_manager_with_classes):
        cm = class_manager_with_classes
        assert cm.add_or_update_class(0, "human", (1, 2, 3)) == 0
        assert cm.get_class_name(0) == "human"
        assert cm.get_class_color(0) == (1, 2, 3)

    def test_add_or_update_class_new_bumps_next_id(self, class_manager):
        cm = class_manager
        assert cm.add_or_update_class(5, "extra", (1, 2, 3)) == 5
        assert cm.get_next_available_class_id() == 6

    def test_add_or_update_class_new_lower_id_keeps_next_id(self, class_manager_with_classes):
        cm = class_manager_with_classes
        cm.delete_class(2)
        cm.delete_class(1)
        # 此时 _next_class_id = max({0}) + 1 = 1
        assert cm.add_or_update_class(1, "reused", (1, 2, 3)) == 1
        # 重新添加走"新增"分支：1 >= 1 -> next_id 升到 2
        assert cm.get_next_available_class_id() == 2


class TestClassQueries:
    def test_get_class_missing_returns_none(self, class_manager):
        assert class_manager.get_class(7) is None

    def test_get_class_name_unknown(self, class_manager):
        assert class_manager.get_class_name(42) == "Unknown(42)"

    def test_get_class_color_default_gray(self, class_manager):
        assert class_manager.get_class_color(42) == (128, 128, 128)

    def test_find_class_by_name(self, class_manager_with_classes):
        assert class_manager_with_classes.find_class_by_name("car") == 1
        assert class_manager_with_classes.find_class_by_name("bird") is None

    def test_get_class_names_sorted_by_id(self, class_manager_with_classes):
        cm = class_manager_with_classes
        cm.delete_class(0)
        cm.add_class("bird")
        assert cm.get_class_names() == ["car", "dog", "bird"]

    def test_validate_class_id(self, class_manager_with_classes):
        assert class_manager_with_classes.validate_class_id(0) is True
        assert class_manager_with_classes.validate_class_id(99) is False

    def test_get_classes_returns_shallow_copy(self, class_manager_with_classes):
        cm = class_manager_with_classes
        snapshot = cm.get_classes()
        snapshot[99] = {"name": "x", "color": (0, 0, 0)}
        assert cm.get_class(99) is None

    def test_build_names_config_preserves_ids(self, class_manager_with_classes):
        cm = class_manager_with_classes
        cm.delete_class(1)  # 制造 ID 空洞
        assert cm.build_names_config() == {0: "person", 2: "dog"}


class TestClassSerialization:
    def test_get_classes_list_includes_ids(self, class_manager_with_classes):
        data = class_manager_with_classes.get_classes_list()
        assert [d["id"] for d in data] == [0, 1, 2]
        assert [d["name"] for d in data] == ["person", "car", "dog"]

    def test_load_from_list_roundtrip(self, class_manager_with_classes):
        cm = class_manager_with_classes
        data = cm.get_classes_list()
        cm.clear_all()
        cm.load_from_list(data)
        assert cm.get_class_count() == 3
        assert cm.get_class_name(2) == "dog"
        assert cm.get_next_available_class_id() == 3

    def test_load_from_list_missing_ids_and_colors(self, class_manager):
        class_manager.load_from_list([{"name": "only-name"}])
        assert class_manager.get_class(0)["name"] == "only-name"
        assert class_manager.get_class_color(0) in ClassManager._COLOR_PALETTE

    def test_json_roundtrip(self, class_manager_with_classes, tmp_path):
        path = str(tmp_path / "classes.json")
        class_manager_with_classes.save_to_json(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 3

        fresh = ClassManager()
        assert fresh.load_from_json(path) is True
        assert fresh.get_class_count() == 3
        assert fresh.get_class_name(1) == "car"

    def test_load_from_json_bad_file_returns_false(self, class_manager, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{{{", encoding="utf-8")
        assert class_manager.load_from_json(str(bad)) is False

    def test_load_from_json_missing_file_returns_false(self, class_manager, tmp_path):
        assert class_manager.load_from_json(str(tmp_path / "nope.json")) is False

    def test_export_to_yaml_content(self, class_manager_with_classes, tmp_path):
        path = str(tmp_path / "data.yaml")
        class_manager_with_classes.export_to_yaml(path, dataset_path="/my/dataset")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["path"] == "/my/dataset"
        assert data["nc"] == 3
        assert data["names"] == {0: "person", 1: "car", 2: "dog"}
        assert data["train"] == "images/train"

    def test_import_from_yaml_list_names(self, class_manager, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text(
            yaml.dump({"names": ["a", "b", "c"]}, allow_unicode=True), encoding="utf-8"
        )
        assert class_manager.import_from_yaml(str(path)) is True
        assert class_manager.get_class_count() == 3
        assert class_manager.get_class_names() == ["a", "b", "c"]

    def test_import_from_yaml_dict_names_preserves_ids(self, class_manager, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text(
            yaml.dump({"names": {0: "a", 5: "b", 2: "c"}}, allow_unicode=True), encoding="utf-8"
        )
        assert class_manager.import_from_yaml(str(path)) is True
        assert class_manager.get_class_name(5) == "b"
        assert class_manager.get_next_available_class_id() == 6

    def test_import_from_yaml_invalid_returns_false(self, class_manager, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("names: {not: [valid", encoding="utf-8")
        assert class_manager.import_from_yaml(str(bad)) is False


class TestClassMergeAndStats:
    def test_merge_classes_reuses_same_name(self, class_manager_with_classes):
        other = ClassManager()
        other.add_class("car", (10, 10, 10))   # 名称重复 -> 映射到已有 ID 1
        other.add_class("bird", (20, 20, 20))  # 新名称 -> 新 ID 3
        mapping = class_manager_with_classes.merge_classes(other)
        assert mapping == {0: 1, 1: 3}
        assert class_manager_with_classes.get_class_name(3) == "bird"

    def test_merge_empty_manager(self, class_manager_with_classes):
        assert class_manager_with_classes.merge_classes(ClassManager()) == {}

    def test_get_class_statistics(self, class_manager):
        from src.core.annotation import Annotation

        annotations = {
            "/a.png": [Annotation(0, 0, 1, 1, 0), Annotation(0, 0, 1, 1, 1)],
            "/b.png": [Annotation(0, 0, 1, 1, 0)],
            "/c.png": [{"class_id": 7}],  # dict 形式也能统计
        }
        assert class_manager.get_class_statistics(annotations) == {0: 2, 1: 1, 7: 1}

    def test_get_class_statistics_empty(self, class_manager):
        assert class_manager.get_class_statistics({}) == {}

    def test_clear_all(self, class_manager_with_classes):
        cm = class_manager_with_classes
        cm.clear_all()
        assert cm.get_class_count() == 0
        assert cm.get_next_available_class_id() == 0

