@echo off
chcp 65001 >nul 2>nul
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo   APB Agent Control
echo ============================================================
echo.

rem ---------- pick python ----------
set "PY="
if exist "runtime\python\python.exe" set "PY=runtime\python\python.exe"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    py -3 --version >nul 2>nul && set "PY=py -3"
)
if not defined PY goto NOPYTHON

if /i "%~1"=="stop" goto STOP
if /i "%~1"=="status" goto STATUS
if /i "%~1"=="start" goto START

echo   Usage:
echo     start_agent.bat start    - run agent in background (silent)
echo     start_agent.bat stop     - stop the agent
echo     start_agent.bat status   - show status and recent log
echo.
echo   Tip: double-click start_agent_hidden.vbs for a silent launch
echo        with no console window at all.
echo.
goto END

:START
echo   [..] Starting agent in background ...
start "" /b %PY% "tools\agent_daemon.py"
echo   [OK] Launched. Use "start_agent.bat status" to check.
goto END

:STOP
echo   [..] Requesting stop ...
%PY% "tools\agent_daemon.py" --stop
echo   [OK] Done.
goto END

:STATUS
%PY% "tools\agent_daemon.py" --status
goto END

:NOPYTHON
echo   [X] No Python found.
echo       Run train.bat once first (it bootstraps a portable runtime),
echo       or install Python 3.10+ and retry.
goto END

:END
echo.
if not "%~1"=="" pause
endlocal
