@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo   APB 一键训练
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 找不到 python。请先安装 Python 3.10+ 并勾选 "Add to PATH"。
    echo        下载: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [错误] Python 版本太旧，需要 3.10 或更高。
    python --version
    echo.
    pause
    exit /b 1
)

python -c "import numpy, pandas" >nul 2>nul
if errorlevel 1 (
    echo [提示] 缺少依赖，正在安装 numpy / pandas ...
    python -m pip install --quiet numpy pandas
    if errorlevel 1 (
        echo [错误] 依赖安装失败。请手动执行: pip install numpy pandas
        echo.
        pause
        exit /b 1
    )
)

echo 启动训练（首次运行需要下载约 545MB 数据，请耐心等待）...
echo.
python tools\train_one_click.py %*

echo.
echo ============================================================
echo   训练结束。把 dist\models\ 里的 zip 拷到 NAS 即可。
echo ============================================================
echo.
pause
