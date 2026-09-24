"""src/utils/export_deps.py 单元测试（纯逻辑，无 Qt / ultralytics 依赖）"""

import sys

import src.utils.export_deps as export_deps


def _fake_env(monkeypatch, versions, modules):
    """替换检测环境：versions 为已装包→版本，modules 为可导入模块名集合"""
    monkeypatch.setattr(
        export_deps, "_installed_version", lambda name: versions.get(name)
    )
    monkeypatch.setattr(
        export_deps, "_module_present", lambda module: module in modules
    )


def test_format_requirements_cover_export_dialog_formats():
    assert set(export_deps.FORMAT_REQUIREMENTS) == {
        "ONNX",
        "TensorRT",
        "OpenVINO",
        "CoreML",
        "TFLite",
        "TF SavedModel",
        "PaddlePaddle",
        "ncnn",
    }


def test_missing_requirements_empty_when_all_satisfied(monkeypatch):
    _fake_env(
        monkeypatch,
        versions={
            "onnx": "1.23.0",
            "onnxslim": "0.1.96",
            "onnxruntime": "1.30.0",
            "openvino": "2026.4.0",
        },
        modules={"onnx", "onnxslim", "onnxruntime", "openvino"},
    )
    assert export_deps.missing_requirements("ONNX") == []
    assert export_deps.missing_requirements("TensorRT") == []
    assert export_deps.missing_requirements("OpenVINO") == []


def test_missing_requirements_reports_low_version(monkeypatch):
    _fake_env(
        monkeypatch,
        versions={"onnx": "1.23.0", "onnxslim": "0.1.50", "onnxruntime": "1.30.0"},
        modules={"onnx", "onnxslim", "onnxruntime"},
    )
    assert export_deps.missing_requirements("ONNX") == ["onnxslim>=0.1.82"]


def test_onnxruntime_group_accepts_gpu_variant(monkeypatch):
    """只装了 onnxruntime-gpu 时不应再要求 onnxruntime（组内 OR）"""
    _fake_env(
        monkeypatch,
        versions={
            "onnx": "1.23.0",
            "onnxslim": "0.1.96",
            "onnxruntime-gpu": "1.30.0",
        },
        modules={"onnx", "onnxslim", "onnxruntime"},
    )
    assert export_deps.missing_requirements("ONNX") == []


def test_missing_requirements_prefers_cpu_candidate(monkeypatch):
    _fake_env(monkeypatch, versions={"onnx": "1.23.0"}, modules={"onnx"})
    assert export_deps.missing_requirements("ONNX") == [
        "onnxslim>=0.1.82",
        "onnxruntime>=1.16",
    ]
    assert "onnxruntime-gpu" not in export_deps.default_packages("ONNX")


def test_ncnn_and_unknown_format_require_nothing():
    assert export_deps.missing_requirements("ncnn") == []
    assert export_deps.default_packages("ncnn") == []
    assert export_deps.missing_requirements("NotAFormat") == []


def test_version_check_degrades_when_packaging_missing(monkeypatch):
    _fake_env(
        monkeypatch,
        versions={"onnx": "1.23.0", "onnxslim": "0.1.50", "onnxruntime": "1.30.0"},
        modules={"onnx", "onnxslim", "onnxruntime"},
    )
    monkeypatch.setattr(export_deps, "_SpecifierSet", None)
    assert export_deps.missing_requirements("ONNX") == []


def test_metadata_missing_falls_back_to_module_presence(monkeypatch):
    _fake_env(monkeypatch, versions={}, modules={"onnx", "onnxslim", "onnxruntime"})
    assert export_deps.missing_requirements("ONNX") == []


def test_module_missing_and_metadata_missing_is_unsatisfied(monkeypatch):
    _fake_env(monkeypatch, versions={}, modules=set())
    assert export_deps.missing_requirements("OpenVINO") == ["openvino>=2024.0.0"]


def test_split_requirement():
    assert export_deps.split_requirement("onnx>=1.12.0,<2.0.0") == (
        "onnx",
        ">=1.12.0,<2.0.0",
    )
    assert export_deps.split_requirement("onnxslim") == ("onnxslim", "")
    assert export_deps.split_requirement("coremltools==9.0") == ("coremltools", "==9.0")


def test_install_commands():
    packages = ["onnxslim>=0.1.82"]
    assert export_deps.pip_install_args(packages) == packages
    command = export_deps.pip_install_command(packages)
    assert command.startswith(sys.executable)
    assert "-m pip install onnxslim>=0.1.82" in command
    assert export_deps.manual_pip_command(packages) == "pip install onnxslim>=0.1.82"


def test_is_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert export_deps.is_frozen() is False
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert export_deps.is_frozen() is True
