"""src/utils/i18n.py 单元测试（monkeypatch 翻译目录，不依赖 CWD、不触碰真实 translations/）"""

import pytest

import src.utils.i18n as i18n_module
from src.utils.i18n import TranslationManager

ZH_INI = "[translations]\ngreeting = 你好\nShared = 中文\n"
EN_INI = "[translations]\ngreeting = Hello\n"


@pytest.fixture
def tm(tmp_path, monkeypatch):
    """把翻译目录指到 tmp_path 并重置单例"""
    tdir = tmp_path / "translations"
    tdir.mkdir()
    (tdir / "zh_CN.ini").write_text(ZH_INI, encoding="utf-8")
    (tdir / "en_US.ini").write_text(EN_INI, encoding="utf-8")

    monkeypatch.setattr(
        TranslationManager, "_translation_dir", staticmethod(lambda: tdir)
    )
    i18n_module.TranslationManager._instance = None
    manager = TranslationManager()
    # 直接实例化后登记为单例，使 instance() 返回同一对象
    i18n_module.TranslationManager._instance = manager
    yield manager
    i18n_module.TranslationManager._instance = None


class TestLoadTranslations:
    def test_loads_both_languages(self, tm):
        assert tm.translations["zh_CN"] == {"greeting": "你好", "Shared": "中文"}
        assert tm.translations["en_US"] == {"greeting": "Hello"}

    def test_missing_dir_no_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            TranslationManager,
            "_translation_dir",
            staticmethod(lambda: tmp_path / "no_such_dir"),
        )
        i18n_module.TranslationManager._instance = None
        manager = TranslationManager()
        assert manager.translations == {}
        i18n_module.TranslationManager._instance = None

    def test_file_without_section_skipped(self, tmp_path, monkeypatch):
        tdir = tmp_path / "translations"
        tdir.mkdir()
        (tdir / "zh_CN.ini").write_text("[wrong]\nk = v\n", encoding="utf-8")
        monkeypatch.setattr(
            TranslationManager, "_translation_dir", staticmethod(lambda: tdir)
        )
        i18n_module.TranslationManager._instance = None
        manager = TranslationManager()
        assert "zh_CN" not in manager.translations
        i18n_module.TranslationManager._instance = None

    def test_case_sensitive_keys(self, tm):
        # optionxform 被覆盖为保留大小写
        assert "Shared" in tm.translations["zh_CN"]
        assert "shared" not in tm.translations["zh_CN"]


class TestTr:
    def test_current_language_hit(self, tm):
        assert tm.tr("greeting") == "你好"

    def test_fallback_to_en_us(self, tm):
        # zh_CN 缺 key（但字典非空，避免触发自动重载）-> 回退 en_US
        tm.translations["zh_CN"] = {"other": "x"}
        assert tm.tr("greeting") == "Hello"

    def test_fallback_to_default_then_key(self, tm):
        assert tm.tr("missing", "默认值") == "默认值"
        assert tm.tr("missing") == "missing"

    def test_switch_then_tr(self, tm):
        tm.switch_language("en_US")
        assert tm.tr("greeting") == "Hello"
        # en_US 没有 Shared，当前语言即 en_US，无进一步回退
        assert tm.tr("Shared", "fallback") == "fallback"


class TestSwitchLanguage:
    def test_switch_supported(self, tm):
        assert tm.switch_language("en_US") is True
        assert tm.get_current_language() == "en_US"

    def test_switch_unsupported(self, tm):
        assert tm.switch_language("fr_FR") is False
        assert tm.get_current_language() == "zh_CN"

    def test_switch_same_language_noop_true(self, tm):
        assert tm.switch_language("zh_CN") is True

    def test_switch_reloads_files(self, tm):
        # 切走再切回，翻译仍能加载
        tm.switch_language("en_US")
        tm.switch_language("zh_CN")
        assert tm.tr("greeting") == "你好"

    def test_get_supported_languages(self, tm):
        assert tm.get_supported_languages() == ["zh_CN", "en_US"]


class TestSaveAndSingleton:
    def test_save_merges_with_existing(self, tm, tmp_path):
        tm.translations["zh_CN"]["brand_new"] = "新增"
        assert tm.save_translation_file("zh_CN") is True

        content = (tmp_path / "translations" / "zh_CN.ini").read_text(encoding="utf-8")
        assert "brand_new" in content
        assert "greeting" in content  # 旧条目保留（合并不覆盖）

    def test_save_new_language_creates_file(self, tm, tmp_path):
        (tmp_path / "translations" / "en_US.ini").unlink()
        tm.translations["en_US"] = {"k": "V"}
        assert tm.save_translation_file("en_US") is True
        assert (tmp_path / "translations" / "en_US.ini").exists()

    def test_singleton(self, tm):
        assert TranslationManager.instance() is tm

    def test_global_tr_function(self, tm):
        from src.utils.i18n import tr

        assert tr("greeting") == "你好"


class TestTranslationDirResolution:
    def test_source_mode_points_to_repo_root(self):
        # 源码模式：解析到仓库根的 translations/（存在性由仓库本身保证）
        d = TranslationManager._translation_dir()
        assert d.name == "translations"
        assert (d / "zh_CN.ini").exists()
