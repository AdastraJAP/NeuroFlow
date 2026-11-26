@echo off
setlocal
set "project=%~dp0"
set "venv_python=%project%.venv\Scripts\python.exe"
title Neuro-Flow - Automatic Rep Detection

echo =========================================================
echo   NEURO-FLOW - AUTOMATIC REP DETECTION
echo =========================================================
echo.
echo Project folder: "%~dp0"
echo Checking your Python environment...
if exist "%venv_python%" (
  "%venv_python%" -c "import cv2, mediapipe, numpy, matplotlib" >nul 2>&1
  if not errorlevel 1 (
    "%venv_python%" "%~dp0menu.py"
    goto end
  )
)
python -c "import cv2, mediapipe, numpy, matplotlib" >nul 2>&1
if not errorlevel 1 (
  echo Required packages found.
  python "%~dp0menu.py"
  goto end
)
echo.
echo Installing dependencies in a local .venv (first run only)...
if not exist "%venv_python%" (
  py -3.14 -m venv "%project%.venv" 2>nul
  if not exist "%venv_python%" python -m venv "%project%.venv"
)
if not exist "%venv_python%" (
  echo ERROR: Python not found. Install Python or run from VS Code.
  goto end
)
"%venv_python%" -m pip install -r "%project%requirements.txt"
if errorlevel 1 (
  echo ERROR: Installation failed. Check internet connection and Python compatibility.
  goto end
)
"%venv_python%" "%~dp0menu.py"
:end
echo.
pause
