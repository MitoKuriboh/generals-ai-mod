@echo off
REM ============================================================================
REM deploy.bat - Build and deploy Learning AI to C&C Generals Zero Hour
REM ============================================================================
REM
REM This script builds the game with the Learning AI mod and deploys it
REM to your Steam installation.
REM
REM Usage:
REM   deploy.bat          - Build and deploy (default)
REM   deploy.bat build    - Build only
REM   deploy.bat deploy   - Deploy only (assumes already built)
REM   deploy.bat clean    - Clean build directory
REM
REM Environment variables:
REM   GENERALS_BUILD_DIR  - Build directory (default: C:\dev\generals\build\win32)
REM   STEAM_DIR           - Steam game directory (auto-detected)
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
if not defined GENERALS_BUILD_DIR set GENERALS_BUILD_DIR=C:\dev\generals\build\win32
set BUILD_CONFIG=Release
set TARGET=z_generals

REM Steam directory detection
if not defined STEAM_DIR (
    set STEAM_DIR=C:\Program Files (x86)\Steam\steamapps\common\Command and Conquer Generals - Zero Hour
)

REM Parse command
set CMD=%1
if "%CMD%"=="" set CMD=all

echo.
echo ============================================================
echo   C&C Generals Zero Hour - Learning AI Deployment
echo ============================================================
echo   Build Dir: %GENERALS_BUILD_DIR%
echo   Steam Dir: %STEAM_DIR%
echo   Command:   %CMD%
echo ============================================================
echo.

if /i "%CMD%"=="clean" goto :clean
if /i "%CMD%"=="build" goto :build
if /i "%CMD%"=="deploy" goto :deploy
if /i "%CMD%"=="all" goto :all
goto :usage

:all
call :build
if errorlevel 1 exit /b 1
call :deploy
if errorlevel 1 exit /b 1
goto :success

:build
echo [BUILD] Building game with Learning AI...
echo.

REM Check if build directory exists
if not exist "%GENERALS_BUILD_DIR%" (
    echo [ERROR] Build directory not found: %GENERALS_BUILD_DIR%
    echo         Run CMake configure first.
    exit /b 1
)

REM Build the game
cmake --build "%GENERALS_BUILD_DIR%" --config %BUILD_CONFIG% --target %TARGET%
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo [BUILD] Build successful!
exit /b 0

:deploy
echo [DEPLOY] Deploying to Steam...
echo.

REM Check source file exists
set SRC_EXE=%GENERALS_BUILD_DIR%\GeneralsMD\%BUILD_CONFIG%\generalszh.exe
if not exist "%SRC_EXE%" (
    echo [ERROR] Source executable not found: %SRC_EXE%
    echo         Run build first.
    exit /b 1
)

REM Check Steam directory exists
if not exist "%STEAM_DIR%" (
    echo [ERROR] Steam directory not found: %STEAM_DIR%
    echo         Set STEAM_DIR environment variable to the correct path.
    exit /b 1
)

REM Backup original if this is first deployment
set DST_EXE=%STEAM_DIR%\generalszh.exe
if not exist "%STEAM_DIR%\generalszh.exe.original" (
    if exist "%DST_EXE%" (
        echo [DEPLOY] Backing up original executable...
        copy /Y "%DST_EXE%" "%STEAM_DIR%\generalszh.exe.original" >nul
    )
)

REM Copy the new executable
echo [DEPLOY] Copying executable...
copy /Y "%SRC_EXE%" "%DST_EXE%" >nul
if errorlevel 1 (
    echo [ERROR] Copy failed! Game might be running.
    exit /b 1
)

REM Copy Python training files if they exist
set PY_SRC=%GENERALS_BUILD_DIR%\..\python
set PY_DST=%STEAM_DIR%\python
if exist "%PY_SRC%" (
    echo [DEPLOY] Copying Python training files...
    if not exist "%PY_DST%" mkdir "%PY_DST%"
    xcopy /Y /E /Q "%PY_SRC%\*" "%PY_DST%\" >nul 2>&1
)

echo.
echo [DEPLOY] Deployment successful!
exit /b 0

:clean
echo [CLEAN] Cleaning build directory...
echo.

if exist "%GENERALS_BUILD_DIR%\GeneralsMD\%BUILD_CONFIG%" (
    rmdir /S /Q "%GENERALS_BUILD_DIR%\GeneralsMD\%BUILD_CONFIG%"
)

echo [CLEAN] Clean complete!
exit /b 0

:success
echo.
echo ============================================================
echo   DEPLOYMENT COMPLETE!
echo ============================================================
echo.
echo   The Learning AI is now available in the game.
echo   To use it:
echo     1. Start C&C Generals Zero Hour
echo     2. Go to Skirmish
echo     3. Select "Learning AI" as opponent (after Hard AI)
echo.
echo   For training:
echo     python train_with_game.py --episodes 100
echo.
exit /b 0

:usage
echo Usage: deploy.bat [command]
echo.
echo Commands:
echo   all     - Build and deploy (default)
echo   build   - Build only
echo   deploy  - Deploy only
echo   clean   - Clean build directory
echo.
exit /b 1
