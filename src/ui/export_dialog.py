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

    def __init__(self, model_path: str, fmt: str, output_file: str, imgsz: int):
        super().__init__()
        self.model_path = model_path
        self.fmt = fmt
        self.output_file = output_file
        self.imgsz = imgsz

    def run(self):
        try:
            from ultralytics import YOLO

            self.progress.emit(f"加载模型: {self.model_path}")
            model = YOLO(self.model_path)

            self.progress.emit(f"开始导出为 {self.fmt} 格式...")
            dest = Path(self.output_file)

            # ultralytics 默认导出到源模型所在目录，导出后移动到用户指定路径
            result = model.export(format=self.fmt, imgsz=self.imgsz, device=0)

            if isinstance(result, str) and Path(result).exists():
                src = Path(result)
            else:
                # 回退：推导默认导出路径
                src = Path(self.model_path).with_suffix(EXPORT_EXT.get(self.fmt, ""))

            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
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
        self.resize(550, 420)

        # 连接模型路径变更以自动检测 imgsz
        self.model_edit.textChanged.connect(self._on_model_edit_changed)
        # 格式变更时同步更新输出文件扩展名
        self.format_combo.currentTextChanged.connect(self._on_format_changed_sync_ext)

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

        self.output_file_edit = QLineEdit()
        self.output_file_edit.setPlaceholderText(
            tr("select_output_file", "选择导出文件路径，如 D:/model.engine"))
        options_layout.addRow(tr("output_file_label", "输出文件:"), self.output_file_edit)

        browse_output_btn = QPushButton(tr("browse_button_label", "浏览..."))
        browse_output_btn.clicked.connect(self.browse_output_file)
        options_layout.addRow("", browse_output_btn)

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

    def _on_format_changed_sync_ext(self, fmt: str):
        """格式变更时同步更新输出文件路径的扩展名"""
        ext = EXPORT_EXT.get(fmt, "")
        if not ext:
            return
        current = self.output_file_edit.text().strip()
        if not current:
            return
        p = Path(current)
        self.output_file_edit.setText(str(p.with_suffix(ext)))

    def _detect_model_imgsz(self, model_path: str) -> int | None:
        """从模型 checkpoint 中读取训练时的 imgsz，失败返回 None"""
        try:
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return None

        if ckpt is None:
            return None

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
        """模型路径变更时自动检测训练尺寸并更新输出路径"""
        if not path or not Path(path).exists() or not path.endswith(".pt"):
            return

        stem = Path(path).stem
        detected = self._detect_model_imgsz(path)
        if detected is not None:
            self.imgsz_spin.setValue(int(detected))

        # 自动填入输出文件路径（与源模型同目录、同名、扩展名按格式）
        ext = EXPORT_EXT.get(self.format_combo.currentText(), "")
        default_output = str(Path(path).with_name(stem + ext))
        self.output_file_edit.setText(default_output)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("browse_model_file_dialog", "选择模型文件"),
            ExportDialog._last_browse_path, "PyTorch模型文件 (*.pt)")
        if path:
            self.model_edit.setText(path)
            self._on_model_path_changed(path)
            ExportDialog._last_browse_path = str(Path(path).parent)

    def browse_output_file(self):
        fmt = self.format_combo.currentText()
        ext = EXPORT_EXT.get(fmt, "")
        filter_str = f"{fmt} 文件 (*{ext})" if ext else f"{fmt} 文件"
        default_dir = ExportDialog._last_browse_path

        path, _ = QFileDialog.getSaveFileName(
            self, tr("browse_output_file_dialog", "选择导出文件路径"),
            default_dir, filter_str)
        if path:
            # 确保扩展名匹配当前格式
            if ext and not path.endswith(ext):
                path += ext
            self.output_file_edit.setText(path)
            ExportDialog._last_browse_path = str(Path(path).parent)

    def check_dependencies(self, fmt: str) -> List[str]:
        """检查目标格式所需的依赖包，返回缺失的包名列表。"""
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

        output_file = self.output_file_edit.text().strip()
        if not output_file:
            QMessageBox.warning(self, tr("warning", "警告"), tr("no_output_file", "请选择输出文件路径"))
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
        output_file = self.output_file_edit.text().strip()
        imgsz = self.imgsz_spin.value()

        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.setVisible(True)
        self.log_text.clear()

        fmt_key = EXPORT_FORMATS.get(fmt, fmt.lower())
        self.worker = ExportWorker(model_path, fmt_key, output_file, imgsz)
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
