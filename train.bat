@echo off
chcp 65001 >nul 2>nul
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo   APB Training Kit
echo ============================================================
echo.

rem --- locate a Python interpreter ---
set "PY="
python --version >nul 2>nul && set "PY=python"
if not defined PY py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY if exist "C:\Python312\python.exe" set "PY=C:\Python312\python.exe"
if not defined PY if exist "C:\Python311\python.exe" set "PY=C:\Python311\python.exe"

if not defined PY goto NOPYTHON

echo Using: %PY%
%PY% --version

rem --- version check ---
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 goto OLDPYTHON

rem --- dependencies ---
%PY% -c "import numpy, pandas" >nul 2>nul
if errorlevel 1 (
    echo [..] Installing numpy / pandas ...
    %PY% -m pip install --quiet numpy pandas
    if errorlevel 1 goto NODEP
)

echo.
echo Starting training. First run downloads ~545MB of data, please wait.
echo.
%PY% tools	rain_one_click.py %*
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo   Done. Copy the zip under dist\models\ to your NAS.
echo ============================================================
echo.
pause
exit /b %RC%

:NOPYTHON
echo [ERROR] Python not found.
echo         Install Python 3.10+ and check "Add python.exe to PATH".
echo         Download: https://www.python.org/downloads/
echo.
pause
exit /b 1

:OLDPYTHON
echo [ERROR] Python is too old. Need 3.10 or newer.
%PY% --version
echo.
pause
exit /b 1

:NODEP
echo [ERROR] Failed to install numpy / pandas.
echo         Run manually: %PY% -m pip install numpy pandas
echo.
pause
exit /b 1
