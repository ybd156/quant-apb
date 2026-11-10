@echo off
rem ============================================================
rem  APB training machine - single entry point.
rem
rem  Keep this file PURE ASCII. The console on Chinese Windows is
rem  GBK, so Chinese text inside a .bat turns into garbage.
rem  All user-facing text is rendered by tools/kit_menu.py, which
rem  controls its own stdout encoding.
rem
rem  Why one file: the kit used to have 6 root scripts
rem  (train.bat / train.sh / daemon.bat / start_agent.bat /
rem  start_agent_hidden.vbs / setup.bat) and users could not tell
rem  which one to click, nor see any status.
rem ============================================================

rem Console codepage: Python writes UTF-8 (see tools/kit_menu.py),
rem so the console must interpret UTF-8 or Chinese turns to garbage.
chcp 65001 >nul 2>nul

cd /d "%~dp0"

set "PY="
if exist "runtime\python\python.exe" set "PY=runtime\python\python.exe"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    py -3 --version >nul 2>nul && set "PY=py -3"
)

if not defined PY goto NOPYTHON

rem Non-interactive shortcuts for scripts / task scheduler.
if /i "%~1"=="start"    goto PASS
if /i "%~1"=="stop"     goto PASS
if /i "%~1"=="status"   goto PASS
if /i "%~1"=="setup"    goto PASS
if /i "%~1"=="train"    goto PASS
if /i "%~1"=="selftest" goto PASS
if /i "%~1"=="update"   goto PASS
if /i "%~1"=="hidden"   goto PASS

%PY% "tools\kit_menu.py"
goto END

:PASS
%PY% "tools\kit_menu.py" %*
goto END

:NOPYTHON
echo.
echo   [X] Python not found.
echo       Run train.bat once to bootstrap the portable runtime,
echo       or install Python 3.10+ and retry.
echo.
pause

:END
endlocal
