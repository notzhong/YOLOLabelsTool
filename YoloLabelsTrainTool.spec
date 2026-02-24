# -*- mode: python ; coding: utf-8 -*-


import os
import sys
from PyInstaller.utils.hooks import collect_submodules

# 防止 Qt 插件被自动删除
sys.setrecursionlimit(5000)

project_root = os.getcwd()

# ===============================
# 数据文件（ini/qss/icon）收集
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


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    console=False,
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
