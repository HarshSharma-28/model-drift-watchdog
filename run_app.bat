@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Model Drift Watchdog
set "LOG=%~dp0setup_log.txt"
echo [%date% %time%] Model Drift Watchdog setup started > "%LOG%"

echo.
echo  ===============================================
echo   Model Drift Watchdog - one-click launcher
echo  ===============================================
echo.

rem ---- 1. Find a working Python 3.10+ ---------------------------------
set "PY="
set "CHECK=import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
py -3 -c "%CHECK%" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "%CHECK%" >nul 2>&1 && set "PY=python"
for %%P in ("C:\Anaconda\python.exe" "%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "C:\ProgramData\anaconda3\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe") do (
  if not defined PY if exist "%%~P" "%%~P" -c "%CHECK%" >nul 2>&1 && set PY="%%~P"
)
if not defined PY (
  echo  Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/
  echo Python not found >> "%LOG%"
  pause
  exit /b 1
)
echo  Using Python: %PY%
echo Using Python: %PY% >> "%LOG%"
%PY% --version >> "%LOG%" 2>&1

rem ---- 2. Virtual environment + dependencies (first run only) ---------
if not exist ".venv\Scripts\python.exe" (
  echo  Creating virtual environment ^(first run only^)...
  %PY% -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 goto :fail
)
if not exist ".venv\installed.ok" (
  echo  Installing packages - this takes a few minutes the first time...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt >> "%LOG%" 2>&1
  if errorlevel 1 goto :fail
  echo ok > ".venv\installed.ok"
)
echo [%date% %time%] Packages ready >> "%LOG%"

rem ---- 3. Start the app and open the browser --------------------------
echo  Starting the app at http://localhost:8501  (close this window to stop it)
echo [%date% %time%] Starting Streamlit >> "%LOG%"
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
goto :eof

:fail
echo.
echo  Setup failed - see setup_log.txt in this folder.
echo [%date% %time%] Setup FAILED >> "%LOG%"
pause
exit /b 1
