# YOLO Label Tool

[简体中文](README.md) | **English**

A PySide6-based YOLO annotation and training tool with manual annotation, model-assisted annotation, realtime window detection, and model training.

## Features

### 🎯 Core
- **Image management**: load/close folders, auto-restore last folder, image preview, image switching
- **Annotation drawing**: draw/edit/delete rectangular bounding boxes
- **Class management**: add/edit/delete classes, clear all classes, custom class colors, 16-color palette auto-assignment, double-click/right-click quick rename
- **Data export**: one-step export of YOLO-format datasets (data.yaml + labels folders) with custom yaml filename
- **Auto-save on switch**: annotations are saved automatically when switching images
- **Keyboard shortcuts**: ←/→ to switch images, Tab/Shift+Tab to cycle classes

### 🚀 Advanced
- **Model-assisted annotation**: load a trained YOLO model for automatic annotation
- **Model parameter tuning**: adjust confidence threshold and IoU threshold in realtime
- **Dataset splitting**: automatic train/val/test split with stratified and balanced sampling
- **Model training**: full YOLO training support — normal training, resumed training, and incremental training
- **Internationalization**: Chinese/English UI with dynamic language switching
- **Keyboard shortcuts**: boost annotation efficiency
- **Annotation statistics**: per-class annotation counts

### 🎨 User Experience
- **Four themes**: dark / light / colorful / eye-care themes
- **Image zoom**: zoom in/out/fit-window, mouse-wheel zoom
- **Status display**: realtime annotation status and info
- **Undo/Redo**: annotation undo/redo via the command pattern (up to 100 steps)
- **Logging**: automatic runtime and exception logging for debugging

### 🖥️ Realtime Verification
- **Window capture detection**: capture a specific on-screen window in realtime for YOLO detection
- **Region capture detection**: pick any screen region for realtime detection
- **Image detection**: detect on a single image file
- **DXCam high-performance capture**: low-latency screen capture via dxcam
- **Window highlighting**: captured windows are outlined automatically
- **DPI awareness**: multi-monitor, per-monitor DPI scaling support
- **Display controls**: adjustable label font size, confidence display toggle

### 🔧 Image List Management
- **Context menu**: delete all unlabeled images (also deletes local files), remove all unlabeled images (keeps files), remove selected image from list
- **Multi-select**: Ctrl/Shift multi-select images for batch removal
- **Image statistics**: realtime labeled/unlabeled counts

### 🔧 Developer Enhancements
- **Exception capture**: unhandled exceptions are caught and written to log files
- **Modular design**: clean structure, easy to extend and maintain
- **Configuration files**: user preferences persisted (window size, theme, classes)
- **i18n architecture**: multi-language ready, easy to add new languages

## Requirements

### Environment
- Python 3.10+ (developed and tested on Python 3.12)
- PySide6
- OpenCV
- ultralytics (YOLO model support and training)

### Installation

1. **Clone the repository**
```bash
GitHub: git clone https://github.com/notzhong/YOLOLabelsTool.git
or
Gitee:  git clone https://gitee.com/notzhong/YOLOLabelsTool.git
cd YOLOLabelsTool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Optional — only needed for the model-export feature (~540MB; skip if you never export):
```bash
pip install -r requirements-export.txt   # CPU onnxruntime (default)
# For GPU: pip uninstall -y onnxruntime && pip install onnxruntime-gpu
```
You may skip this as well — the export dialog will **ask before installing** on first export (the packaged build offers a copyable install command instead).

3. **Run the app**
```bash
python main.py
```

## Packaging

### Build a standalone executable
YOLO Label Tool can be packaged with PyInstaller into a standalone executable for Windows machines without a Python environment.

#### Build steps
1. **Install build dependencies**
```bash
pip install -r requirements-build.txt
```

2. **Download and install UPX** (optional, recommended for compression)
   - Download UPX from https://upx.github.io/
   - Extract and add `upx.exe` to your system PATH

3. **Run the build**
```bash
pyinstaller YoloLabelsTrainTool.spec
```

#### Packaging notes
- **Spec file**: `YoloLabelsTrainTool.spec` contains the complete build configuration
- **Bundled content**:
  - Main executable
  - Translation files (Chinese/English)
  - Theme stylesheets (dark / light / colorful / eye-care)
  - Icon file
  - Config files and directories
- **Output**: the executable lands in the `dist/YoloLabelsTrainTool/` directory
- **Size**: expect 300MB–500MB (Python runtime + PySide6 + OpenCV + ultralytics)
- **First launch**: may be slow due to extraction and loading
- **Model files**: put your YOLO model files in the `model/` folder next to the program
- **CUDA**: GPU acceleration requires proper NVIDIA drivers and CUDA libraries on the target machine
- **Platform**: window/region capture modes are Windows-only (Win32 API + dxcam); annotation, export, and training are cross-platform

## Quick Start

### 1. Launch
```bash
python main.py
```

### 2. Load images
- Click "Load Folder" and pick a folder containing images
- Supported formats: jpg, jpeg, png, bmp, tiff, tif, gif

### 3. Annotate
1. **Pick a class** in the right panel
2. **Draw a box**: hold the left mouse button and drag on the image
3. **Edit annotations**: adjust or delete drawn boxes

### 4. Export data
1. **Save annotations**: automatic — switching images saves current annotations
2. **Export YOLO format**: click "Export YOLO", choose the output path and data.yaml filename, done in one step
3. **Dataset split**: automatic train/val/test splitting is supported

### 5. Train a model
1. **Prepare a dataset**: annotate and export a YOLO dataset with this tool
2. **Configure training**: menu "Model → Train Model"
3. **Start training**: set parameters and click "Start Training"

### 6. Realtime detection
1. **Load a model**: menu "Model → Load Model"
2. **Open the verification window**: menu "Model → Verify Model"
3. **Pick a capture source**:
   - **Window mode**: hover a target window, click to confirm capture
   - **Region mode**: drag to select any screen region
   - **Image mode**: pick a single image file

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open folder |
| Ctrl+S | Save annotations |
| Ctrl+E | Export YOLO format |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Delete | Delete selected annotation |
| A / ← | Previous image |
| D / → | Next image |
| Tab / Shift+Tab | Previous/next annotation class |
| Ctrl+F | Fit window |
| Ctrl++ | Zoom in |
| Ctrl+- | Zoom out |
| Ctrl+A | Auto-annotate current image |
| Ctrl+Shift+A | Batch auto-annotate |
| Ctrl+M | Load model |
| Ctrl+T | Train model |

## Annotation Format

### YOLO format
Exported file structure:
```
output/
├── images/
│   ├── train/      # training images
│   ├── val/        # validation images
│   └── test/       # test images
├── labels/
│   ├── train/      # training labels (txt)
│   ├── val/        # validation labels
│   └── test/       # test labels
├── train.txt       # training image path list (images/train/xxx.jpg)
├── val.txt         # validation image path list (images/val/xxx.jpg)
├── test.txt        # test image path list (images/test/xxx.jpg)
└── data.yaml       # dataset config (filename customizable at export)
```

### data.yaml format
```yaml
path: /absolute/path/to/output  # absolute path
train: images/train
val: images/val
test: images/test

nc: 0  # class count (computed from added classes)
names: []  # class names (list when IDs are contiguous; dict otherwise)
```

### Label file format (.txt)
One annotation per line:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized to 0–1.

Unlabeled images get empty `.txt` files so that images/ and labels/ stay one-to-one for YOLO training compatibility.

## Model-Assisted Annotation

### Supported models
Model loading and inference are built on the [ultralytics](https://github.com/ultralytics/ultralytics) library. Supported YOLO versions include:

- **YOLOv5** (.pt) — the classic
- **YOLOv6** (.pt) — efficient detection
- **YOLOv8** (.pt) — mainstream; detection, instance segmentation, pose estimation
- **YOLOv9** (.pt)
- **YOLOv10** (.pt)
- **YOLOv11** (.pt)
- **YOLOv26** (.pt) — 2025 release

> **Note**: ultralytics adapts to different YOLO versions automatically — any standard Ultralytics-architecture `.pt` model can be loaded.

### Usage
1. **Load a model**:
   - Menu "Model → Load Model" or the toolbar
   - The model info panel appears in the right panel (with confidence/IoU threshold controls)
   - Optionally import model classes into the class manager

2. **Auto-annotate**:
   - **Single image**: select an image, click "Auto Annotate" or press `Ctrl+A`
   - **Batch**: select multiple images, click "Batch Auto Annotate" or press `Ctrl+Shift+A`
   - **Progress**: a progress bar is shown during batch annotation

3. **Model parameters**:
   - **Confidence threshold**: 0.01–1.00, default 0.25, slider + spinbox realtime sync
   - **IoU threshold**: 0.01–1.00, default 0.45, slider + spinbox realtime sync

4. **Model management**:
   - **Unload**, **refresh info**, and a status indicator (green "● loaded" / red "● not loaded")

### Notes
- Requires `ultralytics`: `pip install ultralytics`
- First inference may take extra time to download weights
- Batch annotation can be cancelled at any time

## Realtime Detection Verification

### Overview
Three realtime/offline detection modes for verifying model performance:

1. **Window capture mode**:
   - Hover any visible window — it gets a red highlight border
   - Click to confirm, then the window content is captured continuously for realtime detection
   - Detection stops automatically when the window closes

2. **Region capture mode**:
   - Enter a fullscreen region-selection overlay
   - Drag to select any rectangular screen region
   - The region is continuously detected in realtime

3. **Image detection mode**:
   - Pick a single image file to detect
   - Results shown as green boxes + class names + confidence

4. **Display controls**:
   - **Label font size**: slider/spinbox realtime adjustment (0.1–2.0)
   - **Confidence toggle**: show/hide confidence scores on labels
   - **Class names**: labels show class names (e.g. `person:0.85`) instead of numeric IDs

### Dependencies
- Window/region capture modes require `dxcam` (optional, graceful degradation when missing)
- Windows-only (Win32 API)
- Multi-monitor, per-monitor DPI scaling supported

## Model Training

### Training configuration
Full training support with detailed parameter configuration:

1. **Basics**: pretrained model, dataset config, output dir, resume, incremental training
2. **Training params**: epochs, image size, batch size, device, early-stopping patience
3. **Optimizer**: optimizer choice, initial lr, lr schedule
4. **Augmentation**: Mixup, rotation, shear, perspective, HSV, etc.
5. **Advanced**: weight decay, momentum, warmup, loss weights, label smoothing

### Training UI
- **Training config dialog**: five tabs for different parameter groups
- **Training progress dialog**: realtime progress, loss curves, log output
- **Asynchronous training**: runs in a background thread, never blocks the UI
- **Progress monitoring**: current epoch, elapsed time, loss values

### Market-proven defaults
Verified default parameters with one-click reset:

- Optimizer: AdamW (fits most scenarios)
- Learning rate: 0.01 (with cosine annealing)
- Epochs: 300
- Image size: 640
- Batch size: 4 (fits 8GB VRAM; increase on larger GPUs)
- Workers: 4 (auto-capped at ≤2 on Windows to avoid I/O conflicts)
- Augmentation: combined strategy (HSV, rotation, shear, etc.)

### Training modes
| Mode | Starting model | Optimizer state | Use case |
|------|---------------|-----------------|----------|
| Normal | Pretrained (yolo26n.pt) | Fresh | First training |
| Resume | Checkpoint (last.pt) | Restored | Continue after interruption |
| Incremental | Trained weights (best.pt) | Fresh | Keep learning on new data/classes |

Incremental training aggressively strips stale training params (data/project/nc, etc.) from old checkpoints to avoid conflicts. On Windows, workers are auto-capped at ≤2 to avoid the `torch.save` "I/O operation on closed file" error.

## Model Export

### Overview
Export trained `.pt` models to multiple inference formats via menu "Model → Export Model".

### Supported export formats and dependencies

| Format | Extension | Extra dependencies |
|--------|-----------|--------------------|
| ONNX | `.onnx` | `pip install onnx onnxslim onnxruntime-gpu` |
| TensorRT | `.engine` | `pip install onnx onnxslim onnxruntime-gpu` (+ NVIDIA CUDA + TensorRT libs) |
| OpenVINO | `.xml` + `.bin` | `pip install openvino` |
| CoreML | `.mlpackage` | `pip install coremltools` (macOS only) |
| TFLite | `.tflite` | `pip install tensorflow` |
| TF SavedModel | directory | `pip install tensorflow` |
| PaddlePaddle | directory | `pip install paddlepaddle` |
| ncnn | directory | none |

> **Tip**: the export dialog auto-detects missing dependencies after you pick a format and shows the install command. TensorRT requires an NVIDIA GPU with matching CUDA/TensorRT versions and takes longer.

## Internationalization

### Supported languages
- **Simplified Chinese** (zh_CN) — default
- **English** (en_US)

### Switching
- Menu "Language → 中文/English"
- Takes effect immediately, no restart needed
- All UI text, dialogs, buttons, and menus are translatable

### Architecture
- INI-file based translation system
- Dynamic loading and switching
- Easy to extend: just add a new translation file in `translations/`

## Project Structure

```
YOLOLabelsTool/
├── main.py                 # entry point
├── pyproject.toml          # project config and dependencies
├── requirements.txt        # runtime dependencies
├── requirements-export.txt # optional export dependencies (~540MB)
├── requirements-dev.txt    # development dependencies
├── requirements-build.txt  # packaging dependencies (full set)
├── README.md               # docs (Chinese)
├── README_EN.md            # docs (English)
├── docs/                   # analysis reports
├── tools/                  # dev verification scripts (see tools/README.md)
│   ├── verify_split.py     # Mixin split fidelity (AST byte comparison + whitelist)
│   ├── verify_*.py         # operation-flow / decision regression checks (offscreen)
│   ├── smoke_*.py          # UI offscreen smokes (main window / dialogs / export deps)
│   └── mixin_split/        # ⚠️ one-shot refactor scripts (rewrite sources; for reproducing the split)
├── YoloLabelsTrainTool.spec # PyInstaller build config
├── icon.ico                # app icon
├── yolo26n.pt              # pretrained YOLOv26n weights (optional)
├── src/
│   ├── __init__.py         # version info
│   ├── core/               # core modules
│   │   ├── annotation.py   # annotation data + manager (undo/redo)
│   │   ├── class_manager.py # class manager (YAML import/export/merge/stats)
│   │   ├── image_manager.py # image manager (LRU cache, thumbnails, batch scaling)
│   │   └── model_manager.py # YOLO model manager
│   ├── ui/                 # user interface
│   │   ├── annotation_canvas.py # annotation canvas (QGraphicsView)
│   │   ├── main_window.py  # main window skeleton (menus, shortcuts, panel wiring)
│   │   ├── main_window_mixins/ # main window responsibility mixins (theme/panels/images/classes/model)
│   │   ├── panels.py       # stats panel + model info panel
│   │   ├── class_dialog.py # class edit dialog
│   │   ├── region_selector.py # window highlighter + screen region selector
│   │   ├── train_dialog.py # training config dialog (skeleton)
│   │   ├── train_dialog_mixins/ # 4 training config mixins (tabs/browse/config/lifecycle)
│   │   ├── train_progress_dialog.py # progress dialog (blocking + reconfigure back to config)
│   │   ├── export_dialog.py # model export dialog (ask-before-install dependencies)
│   │   ├── validation_dialog.py # realtime verification dialog (skeleton)
│   │   ├── validation_dialog_mixins/ # 3 verification mixins (drawing helper: utils/unicode_text.py)
│   └── utils/              # utilities
│       ├── dataset_splitter.py # train/val/test splitter
│       ├── export_deps.py   # export dependency declarations + version-aware check
│       ├── unicode_text.py  # Unicode text drawing (pure Pillow/OpenCV, no Qt)
│       ├── i18n.py         # translation manager (pre-cache + en_US fallback)
│       ├── logger.py       # logging module
│       ├── widget_helpers.py # slider/spinbox sync helpers
│       ├── win32_helpers.py # Win32 API helpers
│       └── yolo_exporter.py # YOLO format exporter
├── yolo_tool/              # YOLO training module
│   ├── __init__.py
│   └── yolo_train.py       # YOLO trainer (async training)
├── translations/           # i18n files
│   ├── zh_CN.ini           # Simplified Chinese
│   └── en_US.ini           # English
├── qss/                    # theme stylesheets
│   ├── dark_theme.qss      # dark theme
│   ├── light_theme.qss     # light theme
│   ├── colorful_theme.qss  # colorful theme
│   └── eyecare_theme.qss   # eye-care theme
├── config/                 # config directory
│   └── config.ini          # user preferences
├── annotations/            # annotation files (auto-created)
│   └── *_{hash}.json       # annotation data (name + short hash)
├── logs/                   # log directory (auto-created)
│   └── YYYY-MM-DD.log      # date-named log files
├── runs/                   # training results (created by ultralytics)
│   └── detect/
└── model/                  # model files (user-created, optional)
```

## Development

### Adding features
1. **Annotation logic**: modify `src/core/annotation.py`
2. **UI**: modify `src/ui/main_window.py` or the panel components (`src/ui/panels.py`, `src/ui/annotation_canvas.py`)
3. **Export formats**: create a new exporter under `src/utils/`
4. **Training**: modify `yolo_tool/yolo_train.py` and `src/ui/train_dialog.py`
5. **New languages**: add a translation file under `translations/`
6. **Logging**: `from src.utils.logger import get_logger_simple`

### Running tests
```bash
pytest tests/
```

### Verification scripts (UI layer, not in CI)
```bash
conda activate yolo
export QT_QPA_PLATFORM=offscreen
python tools/verify_split.py         # Mixin split fidelity (run after touching mixin methods)
python tools/verify_flow_fixes.py    # operation-flow regression
python tools/verify_decision_fixes.py
python tools/smoke_mainwindow.py
```
See [tools/README.md](tools/README.md).

## FAQ

**Q: Images fail to load**
A: Check the format is supported and the path has no unusual/special characters.

**Q: Export fails**
A: Check write permissions on the output directory.

**Q: Model fails to load**
A: Verify the model path and that `ultralytics` is installed.

**Q: UI renders incorrectly**
A: Try resizing the window or check your PySide6 version.

**Q: What hardware does training need?**
A: A GPU with 8GB+ VRAM and a decent dataset is recommended. CPU training works but is slow.

**Q: What does realtime detection need?**
A: Window/region capture modes require `dxcam` (`pip install dxcam`) and are Windows-only. Image detection mode has no extra dependencies.

**Q: How do I switch the UI language?**
A: Menu "Language → 中文/English". Takes effect immediately, no restart.

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Questions or suggestions:
- Open an Issue
- Email developer@example.com

## Changelog

### v2.4.0 (2026-09-24)
- **Quality infrastructure**: 196 unit tests (83% coverage) + GitHub Actions CI (ruff + pytest, Python 3.10/3.12 × Linux/Windows) and pre-commit hooks; the test suite is finally version-controlled
- **Dependency slimming**: export dependencies became optional (`requirements-export.txt` / `.[export]`); missing packages are now offered for one-click install in the export dialog (the packaged build copies the command instead); ultralytics' silent auto-install is disabled
- **Refactoring**: main window 2,053 → 822 lines; train/validation dialogs 1,142/856 → 143/81 lines (split into responsibility mixins, method bodies byte-preserved)
- **Robustness**: config / QSS / icon / annotations / training outputs are anchored to the app root (settings no longer "forget", annotations no longer appear lost when launched from another directory)
- **Bug fixes**: broken relative imports in mixins (add-class / train / export buttons), wrong image after removing earlier images, batch annotations overwritten by stale canvas, export failure on CPU-only machines, undo stack surviving folder close, dead "reconfigure training" flow, cross-image undo scope
- **Dev tooling**: new `tools/` verification scripts (mixin fidelity, operation-flow regression, offscreen smokes)

### v2.3.0 (2026-05-10)
- **Model export**: new export feature supporting ONNX / TensorRT / OpenVINO / CoreML / TFLite / TF SavedModel / PaddlePaddle / ncnn (8 formats)
- **Export enhancements**: auto-detect training image size, custom filenames, extension sync per format, auto-move output to a chosen path, missing-dependency popup with install commands
- **Incremental training**: continue learning on trained weights with new data; mutually exclusive with resume training
- **Fixes**: data.yaml export now uses absolute paths, Windows workers auto-capped ≤2 to avoid `torch.save` I/O errors, export file-move handles Path/str returns, OpenVINO companion files (.bin) moved together

### v2.2.0 (2026-05-04)
- **Context menu**: delete/remove unlabeled images, Ctrl/Shift multi-select batch ops
- **Annotation matching**: index-based instead of coordinate-based, eliminating accidental deletions
- **Class management**: double-click/right-click quick rename and edit
- **Eye-care theme**: new warm dark theme
- **Verification window**: dxcam resource leak fix, Unicode batch drawing performance
- **Training fixes**: signal-accumulation popup, lr0 default mismatch, TQDM/GBK/epoch display and 6 more
- **Packaging fixes**: removed unittest/pydoc exclusions that broke torch/scipy imports

### v2.1.0 (2026-05-01)
- **Component extraction**: AnnotationCanvas, StatsPanel, ModelInfoPanel, WindowHighlighter, RegionSelector split into standalone modules
- **Unified dataset export**: DatasetSplitter delegates to YOLOExporter, removing duplicate methods
- **LRU image cache**, **i18n en_US fallback chain**, **auto-save on image switch**
- **176 unit tests** covering core data classes, command pattern, class management, dataset split/export, i18n and platform guards (80%+ coverage; the originally announced 80 tests were never committed — restoration tracked in docs/optimization-analysis-2026-09-24.md)

### v2.0.0 (2026-04-26)
- **Refactor**: extracted `annotation_to_yolo_lines()`, `_process_boxes()`, merged browse methods
- **Optimization**: stats panel cache dict, crosshair render reuse, i18n dead code cleanup
- **Fixes**: undo stack depth limit 100, class ID holes, undo/redo button state

### v1.5.0 (2026-04-26)
- Label font/confidence display controls, simplified window picking, region selection deadlock fix

### v1.0.0~1.4.0 (2026-02-22~2026-04-26)
- Initial release: manual annotation, YOLO export, model-assisted annotation, dark theme
- Logging system, theme switching, enhanced model info panel
- Model training, i18n (Chinese/English dynamic switching)
- PyInstaller packaging support
- Realtime verification (window capture, region capture, image detection)





