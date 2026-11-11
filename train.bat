@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo   APB Training Kit
echo ============================================================
echo.

rem ---------- 1. system python ----------
set "PY="
python --version >nul 2>nul && set "PY=python"
if not defined PY ( py -3 --version >nul 2>nul && set "PY=py -3" )

rem ---------- 2. bundled portable runtime ----------
rem IMPORTANT: "python.exe exists" does NOT mean the runtime works.
rem A failed bootstrap leaves python.exe behind with no pip and no numpy --
rem then this check would skip the repair forever (measured: the kit got
rem stuck on "No module named pip" and could never recover).
rem So verify it can actually import what training needs.
set "PYOK="
if exist "runtime\python\python.exe" (
    set "PY=runtime\python\python.exe"
    %PY% -c "import numpy, pandas" >nul 2>nul
    if not errorlevel 1 set "PYOK=1"
)

rem ---------- 3. bootstrap (also repairs a half-built runtime) ----------
if not defined PYOK (
    echo [..] No usable Python found on this machine.
    echo [..] Setting up a portable runtime inside this folder ^(no install, ~50MB^).
    echo [..] If a previous attempt failed, this will repair it.
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -File "tools\bootstrap.ps1"
    if exist "runtime\python\python.exe" set "PY=runtime\python\python.exe"
    if exist "runtime\python\python.exe" (
        %PY% -c "import numpy, pandas" >nul 2>nul
        if not errorlevel 1 set "PYOK=1"
    )
    if not defined PYOK goto BOOTFAIL
)

if not defined PY goto NOPYTHON

echo Using: %PY%
%PY% --version

rem ---------- 4. version check ----------
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 goto OLDPYTHON

rem ---------- 5. dependencies ----------
%PY% -c "import numpy, pandas" >nul 2>nul
if errorlevel 1 (
    echo [..] Installing numpy / pandas ...
    rem Bundled wheels first: no network needed, takes seconds.
    if exist "wheels" (
        %PY% -m pip install --quiet --disable-pip-version-check --no-index --find-links wheels numpy pandas >nul 2>nul
    )
    %PY% -c "import numpy, pandas" >nul 2>nul
    if errorlevel 1 (
        rem pip itself may be missing (half-built runtime) -- repair via bootstrap
        %PY% -m pip --version >nul 2>nul
        if errorlevel 1 (
            echo [..] pip is missing, repairing the runtime ...
            powershell -NoProfile -ExecutionPolicy Bypass -File "tools\bootstrap.ps1"
        ) else (
            %PY% -m pip install --quiet --disable-pip-version-check numpy pandas
        )
    )
    %PY% -c "import numpy, pandas" >nul 2>nul
    if errorlevel 1 goto NODEP
)

echo.
echo Starting. First run downloads ~545MB of market data, please wait.
echo.
%PY% tools\train_one_click.py %*
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo   Done. Models were uploaded to your NAS automatically.
echo.
echo   Want this machine to keep working as a training worker?
echo   Run APB.bat and pick 2 - it stays online and picks up tasks.
echo   (This window ran a one-time training.)
echo   If the network was down, find the fallback zip in dist\models\
echo ============================================================
echo.
pause
exit /b %RC%

:BOOTFAIL
echo [ERROR] Failed to set up the portable runtime.
echo         You can still install Python 3.10+ manually:
echo         https://www.python.org/downloads/
echo         Remember to check "Add python.exe to PATH".
echo.
pause
exit /b 1

:NOPYTHON
echo [ERROR] No usable Python found.
echo         Install Python 3.10+ from https://www.python.org/downloads/
echo         and check "Add python.exe to PATH".
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