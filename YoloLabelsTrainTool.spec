# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_all

sys.setrecursionlimit(5000)

project_root = os.getcwd()

# ===============================
# 数据文件收集
# ===============================
datas = []

# translations 文件夹
translations_dir = os.path.join(project_root, "translations")
if os.path.exists(translations_dir):
    for root, dirs, files in os.walk(translations_dir):
        for file in files:
            if file.endswith('.ini'):
                src_path = os.path.join(root, file)
                dest_dir = os.path.relpath(root, project_root)
                datas.append((src_path, dest_dir))

# qss 文件夹
qss_dir = os.path.join(project_root, "qss")
if os.path.exists(qss_dir):
    for root, dirs, files in os.walk(qss_dir):
        for file in files:
            if file.endswith('.qss'):
                src_path = os.path.join(root, file)
                dest_dir = os.path.relpath(root, project_root)
                datas.append((src_path, dest_dir))

# icon 文件
icon_path = os.path.join(project_root, "icon.ico")
if os.path.exists(icon_path):
    datas.append((icon_path, '.'))

# ===============================
# 自动收集 ultralytics 及其动态导入
# ===============================
ultralytics_datas, ultralytics_binaries, ultralytics_hidden = collect_all('ultralytics')
datas.extend(ultralytics_datas)
binaries = list(ultralytics_binaries)
hiddenimports = list(ultralytics_hidden)

# 补充 ultralytics / torch 可能需要的动态子模块
hiddenimports += [
    'torch',
    'torchvision',
    'ultralytics.nn.tasks',
    'ultralytics.engine.trainer',
    'ultralytics.engine.validator',
    'ultralytics.engine.predictor',
    'ultralytics.engine.model',
    'ultralytics.engine.results',
    'ultralytics.models.yolo.model',
    'ultralytics.models.yolo.detect.train',
    'ultralytics.models.yolo.detect.val',
    'ultralytics.models.yolo.detect.predict',
    'ultralytics.trackers',
    'ultralytics.utils',
    'ultralytics.utils.ops',
    'ultralytics.utils.torch_utils',
    'ultralytics.utils.loss',
    'ultralytics.utils.metrics',
    'ultralytics.utils.files',
    'ultralytics.utils.plotting',
    'ultralytics.data',
    'ultralytics.data.augment',
    'ultralytics.data.dataset',
    'ultralytics.data.utils',
    'ultralytics.data.loaders',
    'ultralytics.cfg',
    'ultralytics.solutions',
]

# ===============================
# 排除不必要的模块，减小打包体积
# ===============================
excludes = [
    'tkinter',
    'test',
    'distutils',
    'venv',
    'ensurepip',
    'pydoc',
    'lib2to3',
    '__pycache__',
    '*.pyc',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YoloLabelsTrainTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
    contents_directory='.',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YoloLabelsTrainTool',
)
