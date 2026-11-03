@echo off
title Mow My Lawn 2 - Smart Vision Macro
cd /d "%~dp0"

echo ====================================================================
echo   Mow My Lawn 2 - Macro Launcher ^& Dependency Checker
echo ====================================================================

:: Check if Python is installed and accessible
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python was not found on your PATH.
    echo     Attempting to install Python 3.12 via Windows Package Manager ^(winget^)...
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    echo.
    echo [!] If Python just finished installing, please close and re-open this script.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist yet
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating local Python virtual environment ^(.venv^)...
    python -m venv .venv
)

echo [*] Verifying required libraries ^(OpenCV, mss, pydirectinput, keyboard^)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo [*] Launching Mow My Lawn 2 Macro...
echo     ^(Note: Run as Administrator if hotkeys do not register inside Roblox^)
echo.
".venv\Scripts\python.exe" main.py
pause
