@echo off
title PagePilot Launcher
cd /d "%~dp0"
echo Activating virtual environment...
call .\.venv\Scripts\activate.bat
echo Starting PagePilot Desktop...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo PagePilot exited with error code %ERRORLEVEL%
    pause
)
