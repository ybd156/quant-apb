@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo   APB Training Daemon  (keeps running)
echo ============================================================
echo.
echo   This machine becomes a training worker:
echo     1. checks which horizons are missing on the NAS
echo     2. trains the missing ones, uploads them
echo     3. then keeps running, picking up queued tasks
echo     4. repeats forever (Ctrl+C to stop)
echo.
echo   Leave this window open. Closing it stops the worker.
echo.

rem ---------- locate python ----------
set "PY="
python --version >nul 2>nul && set "PY=python"
if not defined PY ( py -3 --version >nul 2>nul && set "PY=py -3" )
if not defined PY if exist "runtime\python\python.exe" set "PY=runtime\python\python.exe"

if not defined PY (
    echo [..] No Python found. Setting up the portable runtime first...
    powershell -NoProfile -ExecutionPolicy Bypass -File "tools\bootstrap.ps1"
    if errorlevel 1 goto BOOTFAIL
    if exist "runtime\python\python.exe" set "PY=runtime\python\python.exe"
)
if not defined PY goto NOPYTHON

echo Using: %PY%
%PY% --version

rem ---------- dependencies ----------
%PY% -c "import numpy, pandas" >nul 2>nul
if errorlevel 1 (
    echo [..] Installing numpy / pandas from bundled wheels...
    %PY% -m pip install --quiet --disable-pip-version-check --no-index --find-links wheels numpy pandas
    if errorlevel 1 (
        echo [..] Bundled wheels failed, trying the internet...
        %PY% -m pip install --quiet --disable-pip-version-check numpy pandas
    )
)

rem ---------- NAS url ----------
set "NAS="
for /f "usebackq delims=" %%u in (`%PY% -c "import sys,os;sys.path.insert(0,os.path.join(os.getcwd(),'tools'));import defaults;print(defaults.nas())" 2^>nul`) do set "NAS=%%u"
if not defined NAS set "NAS=https://apbcdn.683868.xyz"

rem ---------- token ----------
if not defined APB_TOKEN (
    echo.
    echo NAS is protected by APB_TOKEN. Paste it below ^(input is hidden^).
    set /p APB_TOKEN=Token: 
)
if not defined APB_TOKEN (
    echo [!] No token. Uploading models will fail.
    echo     Set it with:  set APB_TOKEN=your-token
)

echo.
echo Target NAS: %NAS%
echo.

rem ---------- 1) train whatever is missing ----------
echo ============================================================
echo   Step 1: check which horizons are missing, train them
echo ============================================================
%PY% tools\train_missing.py --nas "%NAS%" --token "%APB_TOKEN%"
echo.
echo   (If every horizon already had a model, nothing is trained.)

rem ---------- 2) then keep running as a worker ----------
echo.
echo ============================================================
echo   Step 2: stay online, pick up queued tasks every 60s
echo ============================================================
echo.

:LOOP
%PY% tools\sync_agent.py --nas "%NAS%" --loop 60 --auto-update
echo.
echo [!] Worker exited (code %ERRORLEVEL%). Restarting in 30 seconds...
echo     Press Ctrl+C now to stop, or wait to continue.
timeout /t 30 /nobreak >nul
goto LOOP

:BOOTFAIL
echo [ERROR] Could not set up the portable Python runtime.
echo         Install Python 3.10+ manually: https://www.python.org/downloads/
pause
exit /b 1

:NOPYTHON
echo [ERROR] No usable Python found.
pause
exit /b 1
