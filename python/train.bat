@echo off
REM C&C Generals AI Training Launcher
REM This script starts training the AI against the real game
REM
REM Prerequisites:
REM   1. Game built with auto-skirmish support
REM   2. Python packages: pip install torch pywin32
REM
REM Usage:
REM   train.bat                    - Train 100 episodes vs Easy AI
REM   train.bat --episodes 500     - Train 500 episodes
REM   train.bat --headless         - Train without graphics (faster)
REM   train.bat --ai 2             - Train vs Hard AI

echo ============================================================
echo   C&C Generals AI Training
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check PyTorch
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch
)

REM Check pywin32
python -c "import win32pipe" >nul 2>&1
if errorlevel 1 (
    echo Installing pywin32...
    pip install pywin32
)

echo Starting training...
echo.

REM Run training with all arguments passed through
python train_with_game.py %*

echo.
echo Training complete.
pause
