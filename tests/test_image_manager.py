"""src/core/image_manager.py 单元测试（真实 PNG 图片）"""

from pathlib import Path

import pytest

from src.core.image_manager import ImageManager


class TestLoadFolder:
    def test_load_folder_finds_images(self, tmp_path, real_image_factory):
        real_image_factory("a.png")
        real_image_factory("b.JPG")
        im = ImageManager()
        assert im.load_folder(str(tmp_path)) is True
        assert im.get_image_count() == 2
        assert im.get_folder_path() == str(tmp_path)

    def test_load_folder_ignores_non_images(self, tmp_path):
        (tmp_path / "note.txt").write_text("x")
        im = ImageManager()
        assert im.load_folder(str(tmp_path)) is False
        assert im.get_image_count() == 0

    def test_load_missing_folder_returns_false(self, tmp_path):
        assert ImageManager().load_folder(str(tmp_path / "nope")) is False

    def test_paths_sorted(self, tmp_path, real_image_factory):
        real_image_factory("b.png")
        real_image_factory("a.png")
        im = ImageManager()
        im.load_folder(str(tmp_path))
        names = [Path(p).name for p in im.get_all_image_paths()]
        assert names == sorted(names)


class TestIndexAndLookup:
    def test_get_image_path_bounds(self, tmp_path, real_image_factory):
        real_image_factory("a.png")
        im = ImageManager()
        im.load_folder(str(tmp_path))
        assert im.get_image_path(0) is not None
        assert im.get_image_path(1) is None
        assert im.get_image_path(-1) is None

    def test_next_prev_index_wraparound(self, tmp_path, real_image_factory):
        real_image_factory("a.png")
        real_image_factory("b.png")
        im = ImageManager()
        im.load_folder(str(tmp_path))
        assert im.get_next_image_index(1) == 0
        assert im.get_prev_image_index(0) == 1

    def test_next_index_empty_list(self):
        im = ImageManager()
        assert im.get_next_image_index(0) == 0
        assert im.get_prev_image_index(0) == 0

    def test_find_image_by_name_case_insensitive(self, tmp_path, real_image_factory):
        real_image_factory("Photo.PNG")
        im = ImageManager()
        im.load_folder(str(tmp_path))
        assert im.find_image_by_name("hoto") == 0
        assert im.find_image_by_name("zzz") is None

    def test_remove_image(self, tmp_path, real_image_factory):
        p = real_image_factory("a.png")
        im = ImageManager()
        im.load_folder(str(tmp_path))
        im.load_image(p)  # 进入缓存
        assert im.remove_image(0) is True
        assert im.get_image_count() == 0
        assert p not in im._image_cache
        assert im.remove_image(0) is False


class TestImageInfoAndCache:
    def test_get_image_info(self, tmp_path, real_image_factory):
        p = real_image_factory("a.png", size=(64, 48))
        im = ImageManager()
        assert im.get_image_info(p) == (64, 48)

    def test_get_image_info_invalid_returns_none(self, tmp_path):
        assert ImageManager().get_image_info(str(tmp_path / "x.png")) is None

    def test_load_image_returns_array(self, tmp_path, real_image_factory):
        p = real_image_factory("a.png", size=(64, 48))
        im = ImageManager()
        arr = im.load_image(p)
        assert arr is not None
        assert arr.shape == (48, 64, 3)

    def test_load_missing_returns_none(self, tmp_path):
        assert ImageManager().load_image(str(tmp_path / "x.png")) is None

    def test_cache_hit_returns_copy(self, tmp_path, real_image_factory):
        p = real_image_factory("a.png")
        im = ImageManager()
        a1 = im.load_image(p)
        assert p in im._image_cache
        a2 = im.load_image(p)
        a1[:] = 0  # 修改第一个返回值
        assert a2.sum() > 0  # 第二个不受影响（副本）

    def test_use_cache_false_skips_cache(self, tmp_path, real_image_factory):
        p = real_image_factory("a.png")
        im = ImageManager()
        im.load_image(p, use_cache=False)
        assert p not in im._image_cache

    def test_lru_evicts_oldest(self, tmp_path, real_image_factory):
        p1 = real_image_factory("1.png")
        p2 = real_image_factory("2.png")
        p3 = real_image_factory("3.png")
        im = ImageManager()
        im._max_cache_size = 2

        im.load_image(p1)
        im.load_image(p2)
        im.load_image(p1)  # p1 变为最新
        im.load_image(p3)  # 应淘汰 p2
        assert p2 not in im._image_cache
        assert p1 in im._image_cache and p3 in im._image_cache

    def test_clear_cache(self, tmp_path, real_image_factory):
        p = real_image_factory("a.png")
        im = ImageManager()
        im.load_image(p)
        im.clear_cache()
        assert im._image_cache == {}


class TestThumbnail:
    def test_thumbnail_respects_max_size(self, tmp_path, real_image_factory):
        p = real_image_factory("a.png", size=(200, 100))
        im = ImageManager()
        thumb = im.get_image_thumbnail(p, max_size=(50, 50))
        assert thumb.shape[0] <= 50 and thumb.shape[1] <= 50
        # 保持宽高比 2:1
        assert thumb.shape[1] / thumb.shape[0] == pytest.approx(2.0, rel=0.1)

    def test_thumbnail_missing_returns_none(self, tmp_path):
        assert ImageManager().get_image_thumbnail(str(tmp_path / "x.png")) is None


class TestExportAndResize:
    def test_export_image_with_annotations(self, tmp_path, real_image_factory):
        from src.core.annotation import Annotation

        p = real_image_factory("a.png", size=(64, 48))
        im = ImageManager()
        out = tmp_path / "out.png"
        anns = [Annotation(5, 5, 20, 15, 0)]

        assert im.export_image_with_annotations(p, anns, str(out)) is True
        assert out.exists()

    def test_export_missing_image_fails(self, tmp_path):
        im = ImageManager()
        assert im.export_image_with_annotations(
            str(tmp_path / "x.png"), [], str(tmp_path / "o.png")
        ) is False

    def test_batch_resize_keep_ratio(self, tmp_path, real_image_factory):
        real_image_factory("a.png", size=(100, 50))
        im = ImageManager()
        im.load_folder(str(tmp_path))

        out_dir = tmp_path / "resized"
        results = im.batch_resize_images(str(out_dir), target_size=(64, 64), keep_aspect_ratio=True)
        assert len(results) == 1
        from PIL import Image

        with Image.open(results[0]) as img:
            assert img.size == (64, 64)  # 画布尺寸

    def test_batch_resize_direct(self, tmp_path, real_image_factory):
        real_image_factory("a.png", size=(100, 50))
        im = ImageManager()
        im.load_folder(str(tmp_path))

        out_dir = tmp_path / "resized2"
        results = im.batch_resize_images(str(out_dir), target_size=(32, 32), keep_aspect_ratio=False)
        assert len(results) == 1
        from PIL import Image

        with Image.open(results[0]) as img:
            assert img.size == (32, 32)

    def test_batch_resize_empty_list(self, tmp_path):
        assert ImageManager().batch_resize_images(str(tmp_path / "o")) == []
