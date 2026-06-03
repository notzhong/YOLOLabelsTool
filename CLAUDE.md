# CLAUDE.md

## YOLO Labels Tool

PySide6-based YOLO labeling & training desktop app. Supports manual annotation, model-assisted annotation, real-time detection, model training, and model export.

### Architecture

```
src/
├── core/           # Data layer
│   ├── annotation.py      # Annotation data + undo/redo command pattern
│   ├── class_manager.py   # Class CRUD + YAML import/export
│   ├── image_manager.py   # Image list + LRU thumbnail cache
│   └── model_manager.py   # YOLO model load/unload/inference
├── ui/             # Presentation layer
│   ├── main_window.py           # Main window, menus, signal hub
│   ├── train_dialog.py          # Training config (5 tabs)
│   ├── train_progress_dialog.py # Training progress + logs
│   ├── export_dialog.py         # Model export (8 formats)
│   └── validation_dialog.py     # Realtime detection camera
└── utils/          # Shared utilities
    ├── yolo_exporter.py     # YOLO dataset export + validation
    ├── dataset_splitter.py  # Train/val/test random split
    ├── i18n.py              # Translation manager (zh_CN/en_US INI files)
    ├── logger.py            # File-based logging
    └── widget_helpers.py    # Slider/SpinBox sync helpers

yolo_tool/
└── yolo_train.py   # Async training backend (YOLOTrainer QObject)

translations/
├── zh_CN.ini       # Chinese translations
└── en_US.ini       # English translations

YoloLabelsTrainTool.spec  # PyInstaller build config
```

### Key Patterns

- **Translation**: `from src.utils.i18n import tr` -> `tr("key", "fallback")`
- **Logging**: `from src.utils.logger import get_logger_simple` -> `logger = get_logger_simple(__name__)`
- **Training**: `YOLOTrainer` emits Qt Signals; `TrainProgressDialog` connects and displays
- **Export**: `ExportDialog` -> `ExportWorker(QThread)` -> ultralytics `model.export()`

---

## /audit-project - Project Code Audit

Run a comprehensive audit of the entire codebase. Check each file for:

### 1. Logger Coverage
- Every `try/except` block must log the exception with `logger.exception()` or `logger.error()`
- Every `QMessageBox.warning()` / `QMessageBox.critical()` must be accompanied by `logger.warning()` / `logger.error()`
- Every significant state transition (training start/stop, model load/unload, export begin/end) must log

### 2. Silent Exception Swallowing
- Flag all `except Exception: pass` or `except: pass` blocks
- Flag all `except Exception:` blocks that don't call logger
- Flag all `except:` without specifying exception type

### 3. Translation Completeness
- Scan all `.py` files for `tr("key")` calls, extract keys
- Verify each key exists in BOTH `translations/zh_CN.ini` AND `translations/en_US.ini`
- Flag `tr("key", "fallback")` calls where the key is missing from INI files

### 4. PyInstaller Spec Completeness
- Check `YoloLabelsTrainTool.spec` hiddenimports against all import statements
- Flag any `from .xxx import` (relative imports in methods) that PyInstaller might miss
- Verify `collect_all()` covers all third-party packages with C extensions

### 5. Hardcoded Strings
- Flag Chinese/English user-visible strings not wrapped in `tr()`
- Allow: log messages, debug strings, internal variable names

### 6. Signal Connection Hygiene
- Flag `connect()` without preceding `disconnect()` in re-connectable slots
- Flag potential signal accumulation in dialogs that can be reopened

### 7. Resource Cleanup
- Flag missing `closeEvent()` overrides in dialogs using resources
- Flag threads/cameras/timers not stopped in close/destroy
- Flag `QThread` instances not cleaned up

### Output Format
Report findings grouped by file with line numbers, severity (CRITICAL/HIGH/MEDIUM/LOW), and suggested fix.
