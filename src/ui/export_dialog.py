"""
模型导出对话框
"""
import importlib
import importlib.metadata
import shutil
from pathlib import Path
from typing import Dict, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QFormLayout, QComboBox, QSpinBox,
    QFileDialog, QMessageBox, QTextEdit, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal

from src.utils.i18n import tr
from src.utils.logger import get_logger_simple

logger = get_logger_simple(__name__)

# 格式 → 所需 pip 包名列表（可能与 import 名不同，如 onnxruntime-gpu）
FORMAT_DEPENDENCIES: Dict[str, List[str]] = {
    "ONNX": ["onnx", "onnxslim", "onnxruntime-gpu", "onnxruntime"],
    "TensorRT": ["onnx", "onnxslim", "onnxruntime-gpu", "onnxruntime"],
    "OpenVINO": ["openvino"],
    "CoreML": ["coremltools"],
    "TFLite": ["tensorflow"],
    "TF SavedModel": ["tensorflow"],
    "PaddlePaddle": ["paddlepaddle"],
    "ncnn": [],
}

EXPORT_FORMATS: Dict[str, str] = {
    "ONNX": "onnx",
    "TensorRT": "engine",
    "OpenVINO": "openvino",
    "CoreML": "coreml",
    "TFLite": "tflite",
    "TF SavedModel": "saved_model",
    "PaddlePaddle": "paddle",
    "ncnn": "ncnn",
}

EXPORT_EXT: Dict[str, str] = {
    "ONNX": ".onnx",
    "TensorRT": ".engine",
    "OpenVINO": ".xml",
    "CoreML": ".mlpackage",
    "TFLite": ".tflite",
    "TF SavedModel": "",
    "PaddlePaddle": "",
    "ncnn": "",
}


class ExportWorker(QThread):
    """导出工作线程"""
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, model_path: str, fmt: str, output_dir: str,
                 output_name: str, imgsz: int):
        super().__init__()
        self.model_path = model_path
        self.fmt = fmt
        self.output_dir = output_dir
        self.output_name = output_name
        self.imgsz = imgsz

    def run(self):
        try:
            from ultralytics import YOLO

            self.progress.emit(f"加载模型: {self.model_path}")
            model = YOLO(self.model_path)

            self.progress.emit(f"开始导出为 {self.fmt} 格式...")
            result = model.export(format=self.fmt, imgsz=self.imgsz, device=0)

            # ultralytics 默认导出到源模型所在目录，需移动到用户指定位置
            if isinstance(result, str):
                src = Path(result)
            else:
                # 推导默认导出路径：与源模型同目录、同 stem、扩展名对应
                src = Path(self.model_path).with_suffix(EXPORT_EXT.get(self.fmt, ""))

            suffix = src.suffix  # e.g. ".engine", ".onnx"
            # 对目录格式导出（saved_model, mlpackage），保留整个目录名
            if suffix and suffix in {".onnx", ".engine", ".tflite", ".xml"}:
                # 单文件导出
                dest_name = self.output_name + suffix
            else:
                # 目录导出（saved_model 等无扩展名，用整个目录名）
                dest_name = self.output_name + suffix if suffix else self.output_name

            dest_dir = Path(self.output_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / dest_name

            if src.exists():
                if src.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(src), str(dest))
                else:
                    shutil.move(str(src), str(dest))
                self.progress.emit("导出完成")
                self.finished.emit(True, f"模型已成功导出到:\n{dest}")
            else:
                self.finished.emit(False, f"导出完成但找不到输出文件:\n{src}")
        except Exception as e:
            import traceback
            self.progress.emit(f"导出失败: {e}")
            self.finished.emit(False, f"导出失败:\n{traceback.format_exc()}")


class ExportDialog(QDialog):
    """模型导出对话框"""

    _last_browse_path = None

    def __init__(self, parent=None, default_model_path: str = ""):
        super().__init__(parent)
        self.default_model_path = default_model_path
        self.worker: ExportWorker | None = None

        if ExportDialog._last_browse_path is None:
            ExportDialog._last_browse_path = str(Path.cwd())

        self.init_ui()
        self.setWindowTitle(tr("export_model", "导出模型"))
        self.setModal(True)
        self.resize(550, 460)

        # 连接模型路径变更以自动检测 imgsz
        self.model_edit.textChanged.connect(self._on_model_edit_changed)

        # 自动检测已加载模型的训练尺寸
        if default_model_path:
            self._on_model_path_changed(default_model_path)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 模型选择
        model_group = QGroupBox(tr("model_file", "模型文件"))
        model_layout = QFormLayout(model_group)

        self.model_edit = QLineEdit(self.default_model_path)
        self.model_edit.setPlaceholderText(tr("select_model_file", "选择模型文件 (.pt)"))
        model_layout.addRow(tr("model_path_label", "模型路径:"), self.model_edit)

        browse_btn = QPushButton(tr("browse_button_label", "浏览..."))
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addRow("", browse_btn)

        layout.addWidget(model_group)

        # 导出格式
        fmt_group = QGroupBox(tr("export_format", "导出格式"))
        fmt_layout = QFormLayout(fmt_group)

        self.format_combo = QComboBox()
        self.format_combo.addItems(list(EXPORT_FORMATS.keys()))
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        fmt_layout.addRow(tr("format_label", "目标格式:"), self.format_combo)

        self.format_hint = QLabel()
        self.format_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.on_format_changed(self.format_combo.currentText())
        fmt_layout.addRow(self.format_hint)

        layout.addWidget(fmt_group)

        # 导出选项
        options_group = QGroupBox(tr("export_options", "导出选项"))
        options_layout = QFormLayout(options_group)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(tr("select_output_dir", "选择导出目录"))
        options_layout.addRow(tr("output_dir_label", "输出目录:"), self.output_dir_edit)

        browse_output_btn = QPushButton(tr("browse_button_label", "浏览..."))
        browse_output_btn.clicked.connect(self.browse_output_dir)
        options_layout.addRow("", browse_output_btn)

        self.output_name_edit = QLineEdit("model")
        self.output_name_edit.setPlaceholderText(tr("export_name_placeholder", "导出的文件名（不含扩展名）"))
        options_layout.addRow(tr("export_name", "文件名:"), self.output_name_edit)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setToolTip(tr("export_imgsz_tip", "导出模型的输入尺寸，应与训练时使用的尺寸一致"))
        options_layout.addRow(tr("export_imgsz", "输入尺寸:"), self.imgsz_spin)

        layout.addWidget(options_group)

        # 进度和日志
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.export_btn = QPushButton(tr("start_export", "导出"))
        self.export_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.export_btn.clicked.connect(self.start_export)
        btn_layout.addWidget(self.export_btn)

        self.cancel_btn = QPushButton(tr("cancel_btn", "取消"))
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def on_format_changed(self, fmt: str):
        if fmt == "TensorRT":
            self.format_hint.setText(tr("format_tensorrt_hint", "需要 NVIDIA CUDA 环境，导出时间较长"))
        elif fmt == "CoreML":
            self.format_hint.setText(tr("format_coreml_hint", "仅限 Apple 平台推理"))
        elif fmt == "ONNX":
            self.format_hint.setText(tr("format_onnx_hint", "通用跨平台格式，推荐用于部署"))
        else:
            self.format_hint.setText("")

    def _detect_model_imgsz(self, model_path: str) -> int | None:
        """从模型 checkpoint 中读取训练时的 imgsz，失败返回 None"""
        try:
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return None

        if ckpt is None:
            return None

        # 尝试多个可能的存储位置（兼容不同 ultralytics 版本）
        candidates = []
        if hasattr(ckpt, "overrides") and isinstance(ckpt.overrides, dict):
            candidates.append(ckpt.overrides)
        if isinstance(ckpt, dict):
            if "overrides" in ckpt and isinstance(ckpt["overrides"], dict):
                candidates.append(ckpt["overrides"])
            if "train_args" in ckpt and isinstance(ckpt["train_args"], dict):
                candidates.append(ckpt["train_args"])
            if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
                candidates.append(ckpt["cfg"])

        for d in candidates:
            imgsz = d.get("imgsz")
            if imgsz is not None:
                return int(imgsz)

        return None

    def _on_model_edit_changed(self, path: str):
        """模型路径文本变更时自动检测训练尺寸"""
        if path and path.endswith(".pt") and Path(path).exists():
            self._on_model_path_changed(path)

    def _on_model_path_changed(self, path: str):
        """模型路径变更时自动检测训练尺寸"""
        if path and Path(path).exists() and path.endswith(".pt"):
            detected = self._detect_model_imgsz(path)
            if detected is not None:
                self.imgsz_spin.setValue(int(detected))
            # 自动填入默认文件名（源模型 stem，去掉 best/last 后缀）
            stem = Path(path).stem
            self.output_name_edit.setText(stem)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("browse_model_file_dialog", "选择模型文件"),
            ExportDialog._last_browse_path, "PyTorch模型文件 (*.pt)")
        if path:
            self.model_edit.setText(path)
            self._on_model_path_changed(path)
            ExportDialog._last_browse_path = str(Path(path).parent)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, tr("browse_output_dir_dialog", "选择导出目录"),
            ExportDialog._last_browse_path)
        if path:
            self.output_dir_edit.setText(path)
            ExportDialog._last_browse_path = path

    def check_dependencies(self, fmt: str) -> List[str]:
        """检查目标格式所需的依赖包，返回缺失的包名列表。

        先尝试 import 模块，失败时再通过 importlib.metadata 按 pip 包名检测，
        以支持 onnxruntime-gpu 等包名含 '-' 无法直接 import 的包。
        """
        missing = []
        pkgs = FORMAT_DEPENDENCIES.get(fmt, [])
        for pkg in pkgs:
            try:
                importlib.import_module(pkg)
                continue
            except ImportError:
                pass
            try:
                importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                missing.append(pkg)
        return missing

    def validate(self) -> bool:
        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, tr("warning", "警告"), tr("no_model_selected", "请选择模型文件"))
            return False
        if not Path(model_path).exists():
            QMessageBox.warning(self, tr("warning", "警告"), tr("model_file_not_exists", "模型文件不存在"))
            return False

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, tr("warning", "警告"), tr("no_output_dir", "请选择导出目录"))
            return False
        if not Path(output_dir).exists():
            QMessageBox.warning(self, tr("warning", "警告"), tr("output_dir_not_exists", "导出目录不存在"))
            return False

        output_name = self.output_name_edit.text().strip()
        if not output_name:
            QMessageBox.warning(self, tr("warning", "警告"), tr("no_export_name", "请输入导出文件名"))
            return False

        return True

    def start_export(self):
        if not self.validate():
            return

        fmt = self.format_combo.currentText()
        missing = self.check_dependencies(fmt)
        if missing:
            reply = QMessageBox.question(
                self,
                tr("missing_deps_title", "缺少依赖"),
                tr("missing_deps_msg", "导出 {fmt} 需要以下依赖包，是否继续？\n\n{packages}\n\n可运行以下命令安装：\n{cmd}\n\n如已安装请忽略，导出可能仍会失败。").format(
                    fmt=fmt,
                    packages=", ".join(missing),
                    cmd="pip install " + " ".join(missing)
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        model_path = self.model_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        output_name = self.output_name_edit.text().strip()
        imgsz = self.imgsz_spin.value()

        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.setVisible(True)
        self.log_text.clear()

        # 将显示名称映射为 ultralytics 格式键（如 "TensorRT" → "engine"）
        fmt_key = EXPORT_FORMATS.get(fmt, fmt.lower())
        self.worker = ExportWorker(model_path, fmt_key, output_dir, output_name, imgsz)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, msg: str):
        self.log_text.append(msg)

    def on_finished(self, success: bool, message: str):
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, tr("success", "成功"), message)
            self.accept()
        else:
            QMessageBox.critical(self, tr("error", "错误"), message)
