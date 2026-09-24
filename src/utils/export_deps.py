"""模型导出依赖声明与检测（纯逻辑，不依赖 Qt / ultralytics）

版本区间与 ultralytics 的 ``check_requirements`` 对齐：包"存在但版本过低"同样
视为缺失，避免放行后在导出过程中触发 ultralytics 的 AutoUpdate 静默 pip 安装。

依赖组内为 OR（任选其一即可满足），组间为 AND。
"""

import importlib
import importlib.metadata
import importlib.util
import sys

try:  # packaging 通常随 pip/setuptools 存在，缺失时降级为"仅存在性检查"
    from packaging.specifiers import SpecifierSet as _SpecifierSet
except ImportError:  # pragma: no cover - 取决于运行环境
    _SpecifierSet = None

# 格式 → 依赖组列表；组内 OR（任一满足即可），组间 AND。
# 每组的首个候选为默认安装目标（ONNX 类默认装 CPU 版 onnxruntime）。
FORMAT_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "ONNX": (
        ("onnx>=1.12.0,<2.0.0",),
        ("onnxslim>=0.1.82",),
        ("onnxruntime>=1.16", "onnxruntime-gpu>=1.16"),
    ),
    "TensorRT": (
        ("onnx>=1.12.0,<2.0.0",),
        ("onnxslim>=0.1.82",),
        ("onnxruntime>=1.16", "onnxruntime-gpu>=1.16"),
    ),
    "OpenVINO": (("openvino>=2024.0.0",),),
    "CoreML": (("coremltools>=9.0",),),
    "TFLite": (("tensorflow",),),
    "TF SavedModel": (("tensorflow",),),
    "PaddlePaddle": (("paddlepaddle",),),
    "ncnn": (),
}

# pip 包名 → 可导入模块名（包名含 '-' 无法直接 import；onnxruntime 各变体共用同一模块）
PKG_IMPORT_NAME: dict[str, str] = {
    "onnxruntime-gpu": "onnxruntime",
    "onnxruntime-directml": "onnxruntime",
    "onnxruntime-qnn": "onnxruntime",
}

_SPEC_CHARS = "<>=!~ "


def is_frozen() -> bool:
    """是否运行在 PyInstaller 等冻结环境中（冻结环境无法可靠地运行时 pip install）"""
    return bool(getattr(sys, "frozen", False))


def split_requirement(requirement: str) -> tuple[str, str]:
    """拆分 "onnx>=1.12.0,<2.0.0" → ("onnx", ">=1.12.0,<2.0.0")"""
    for index, char in enumerate(requirement):
        if char in _SPEC_CHARS:
            return requirement[:index].strip(), requirement[index:].strip()
    return requirement.strip(), ""


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _installed_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except Exception:  # PackageNotFoundError 及冻结环境下 metadata 失效
        return None


def _version_matches(version: str, spec: str) -> bool:
    """无版本区间或无 packaging 时降级为"视为满足"（仅存在性检查）"""
    if not spec or _SpecifierSet is None:
        return True
    try:
        return _SpecifierSet(spec).contains(version)
    except Exception:  # 版本串非 PEP 440（如本地构建号）时不做拦截
        return True


def is_satisfied(requirement: str) -> bool:
    """判断单条依赖是否已满足（存在 + 版本区间）"""
    name, spec = split_requirement(requirement)
    module = PKG_IMPORT_NAME.get(name, name)
    version = _installed_version(name)
    if version is not None:
        return _version_matches(version, spec)
    # metadata 缺失（冻结环境常见）→ 退化为"能找到模块即视为满足"
    return _module_present(module)


def missing_requirements(fmt: str) -> list[str]:
    """返回该导出格式尚未满足的依赖（每组返回首个候选，可直接交给 pip）"""
    missing = []
    for group in FORMAT_REQUIREMENTS.get(fmt, ()):
        if not any(is_satisfied(requirement) for requirement in group):
            missing.append(group[0])
    return missing


def default_packages(fmt: str) -> list[str]:
    """该格式的默认安装候选包（用于展示，不校验是否已安装）"""
    return [group[0] for group in FORMAT_REQUIREMENTS.get(fmt, ())]


def pip_install_args(packages: list[str] | tuple[str, ...]) -> list[str]:
    """生成 pip install 的参数列表"""
    return [*packages]


def pip_install_command(packages: list[str] | tuple[str, ...]) -> str:
    """可执行的安装命令（源码/conda 环境）"""
    return " ".join([sys.executable, "-m", "pip", "install", *pip_install_args(packages)])


def manual_pip_command(packages: list[str] | tuple[str, ...]) -> str:
    """供用户手动执行的安装命令（冻结环境展示用）"""
    return "pip install " + " ".join(pip_install_args(packages))
