@echo off
chcp 65001 >nul 2>nul
setlocal

cd /d "%~dp0"

rem Pick a Python that has tkinter (the bundled portable runtime may not).
set "PY="
if exist "runtime\python\python.exe" set "PY=runtime\python\python.exe"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    py -3 --version >nul 2>nul && set "PY=py -3"
)
if not defined PY goto NOPYTHON

rem If the chosen Python has no tkinter, fall back to the console wizard.
%PY% -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo   [!] This Python has no tkinter, using console mode.
    echo.
    %PY% "tools\kit_setup.py" --cli
    goto END
)

%PY% "tools\kit_setup.py"
goto END

:NOPYTHON
echo   [X] No Python found.
echo       Run train.bat once first (it bootstraps a portable runtime).
goto END

:END
rem Keep the window open on double-click failure,
rem otherwise the user sees nothing at all.
pause
endlocal
