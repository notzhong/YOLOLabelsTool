@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PROJECT_ROOT=%~dp0
set OUTPUT_DIR=%PROJECT_ROOT%install

echo ========================================
echo  YOLO Label Tool - Build Script
echo ========================================
echo.

:: Step 1: Clean previous build artifacts
echo [1/4] Cleaning previous build artifacts...
if exist "%PROJECT_ROOT%build"       rmdir /s /q "%PROJECT_ROOT%build"
if exist "%OUTPUT_DIR%"              rmdir /s /q "%OUTPUT_DIR%"

:: Step 2: Install build dependencies
echo [2/4] Installing build dependencies...
pip install -q -r "%PROJECT_ROOT%requirements-build.txt"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install build dependencies
    pause
    exit /b 1
)

:: Step 3: Run PyInstaller with --contents-directory .
echo [3/4] Running PyInstaller (this may take several minutes)...
pyinstaller ^
    --distpath "%OUTPUT_DIR%" ^
    --noconfirm ^
    "%PROJECT_ROOT%YoloLabelsTrainTool.spec"
if %errorlevel% neq 0 (
    echo ERROR: PyInstaller build failed
    pause
    exit /b 1
)

:: Step 4: Clean up temporary build files
echo [4/4] Cleaning up...
if exist "%PROJECT_ROOT%build" rmdir /s /q "%PROJECT_ROOT%build"
if exist "%PROJECT_ROOT%__pycache__" rmdir /s /q "%PROJECT_ROOT%__pycache__"

:: Remove .pyc files generated during build
for /r "%PROJECT_ROOT%" %%i in (*.pyc) do (
    if not "%%i"=="%PROJECT_ROOT%build\*" (
        del "%%i" 2>nul
    )
)

echo.
echo ========================================
echo  Build complete!
echo  Output: %OUTPUT_DIR%YoloLabelsTrainTool\
echo ========================================
echo.
echo  To run:  install\YoloLabelsTrainTool\YoloLabelsTrainTool.exe
echo.
echo  Note: The first startup may be slow due to
echo  PySide6 and PyTorch initialization.
echo.

endlocal
