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
REM   deploy.bat check    - Verify environment (FIX D1)
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
set ERROR_COUNT=0

REM FIX D1: Enhanced Steam directory detection with multiple fallback paths
if not defined STEAM_DIR (
    REM Try common Steam installation paths
    if exist "C:\Program Files (x86)\Steam\steamapps\common\Command and Conquer Generals - Zero Hour" (
        set "STEAM_DIR=C:\Program Files (x86)\Steam\steamapps\common\Command and Conquer Generals - Zero Hour"
    ) else if exist "C:\Steam\steamapps\common\Command and Conquer Generals - Zero Hour" (
        set "STEAM_DIR=C:\Steam\steamapps\common\Command and Conquer Generals - Zero Hour"
    ) else if exist "D:\Steam\steamapps\common\Command and Conquer Generals - Zero Hour" (
        set "STEAM_DIR=D:\Steam\steamapps\common\Command and Conquer Generals - Zero Hour"
    ) else if exist "D:\SteamLibrary\steamapps\common\Command and Conquer Generals - Zero Hour" (
        set "STEAM_DIR=D:\SteamLibrary\steamapps\common\Command and Conquer Generals - Zero Hour"
    ) else if exist "E:\SteamLibrary\steamapps\common\Command and Conquer Generals - Zero Hour" (
        set "STEAM_DIR=E:\SteamLibrary\steamapps\common\Command and Conquer Generals - Zero Hour"
    ) else (
        set "STEAM_DIR=C:\Program Files (x86)\Steam\steamapps\common\Command and Conquer Generals - Zero Hour"
    )
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
if /i "%CMD%"=="check" goto :check
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

REM FIX D1: Copy checkpoints if they exist with validation
set CKPT_SRC=%GENERALS_BUILD_DIR%\..\checkpoints
set CKPT_DST=%STEAM_DIR%\checkpoints
if exist "%CKPT_SRC%" (
    REM Validate checkpoint files before copying (must be non-empty)
    set CKPT_VALID=0
    if exist "%CKPT_SRC%\best_agent.pt" (
        for %%A in ("%CKPT_SRC%\best_agent.pt") do if %%~zA GTR 1000 set CKPT_VALID=1
    )
    if exist "%CKPT_SRC%\final_agent.pt" (
        for %%A in ("%CKPT_SRC%\final_agent.pt") do if %%~zA GTR 1000 set CKPT_VALID=1
    )

    if !CKPT_VALID!==1 (
        echo [DEPLOY] Copying model checkpoints...
        if not exist "%CKPT_DST%" mkdir "%CKPT_DST%"
        REM Only copy the best/latest models to avoid bloat
        if exist "%CKPT_SRC%\best_agent.pt" copy /Y "%CKPT_SRC%\best_agent.pt" "%CKPT_DST%\" >nul
        if exist "%CKPT_SRC%\final_agent.pt" copy /Y "%CKPT_SRC%\final_agent.pt" "%CKPT_DST%\" >nul
    ) else (
        echo [WARN] Checkpoint files found but appear invalid ^(too small^). Skipping.
    )
)

REM Also copy from unified checkpoints location
set CKPT_UNIFIED=%GENERALS_BUILD_DIR%\..\checkpoints\unified
if exist "%CKPT_UNIFIED%" (
    echo [DEPLOY] Copying unified checkpoints...
    if not exist "%CKPT_DST%\unified" mkdir "%CKPT_DST%\unified"
    if exist "%CKPT_UNIFIED%\best_agent.pt" copy /Y "%CKPT_UNIFIED%\best_agent.pt" "%CKPT_DST%\unified\" >nul
    if exist "%CKPT_UNIFIED%\final_agent.pt" copy /Y "%CKPT_UNIFIED%\final_agent.pt" "%CKPT_DST%\unified\" >nul
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

:check
REM FIX D1: Verify environment before deployment
echo [CHECK] Verifying environment...
echo.

set CHECK_ERRORS=0

REM Check build directory
if exist "%GENERALS_BUILD_DIR%" (
    echo [OK] Build directory found: %GENERALS_BUILD_DIR%
) else (
    echo [WARN] Build directory not found: %GENERALS_BUILD_DIR%
    set /a CHECK_ERRORS+=1
)

REM Check Steam directory
if exist "%STEAM_DIR%" (
    echo [OK] Steam directory found: %STEAM_DIR%
) else (
    echo [ERROR] Steam directory not found: %STEAM_DIR%
    echo         Set STEAM_DIR environment variable to correct path.
    set /a CHECK_ERRORS+=1
)

REM Check for built executable
set SRC_EXE=%GENERALS_BUILD_DIR%\GeneralsMD\%BUILD_CONFIG%\generalszh.exe
if exist "%SRC_EXE%" (
    echo [OK] Built executable found: %SRC_EXE%
) else (
    echo [WARN] Built executable not found - run build first
    set /a CHECK_ERRORS+=1
)

REM Check for Python training files
set PY_SRC=%GENERALS_BUILD_DIR%\..\python
if exist "%PY_SRC%\train_manual.py" (
    echo [OK] Python training files found
) else (
    echo [WARN] Python training files not found at %PY_SRC%
)

REM Check for checkpoints
set CKPT_SRC=%GENERALS_BUILD_DIR%\..\checkpoints
if exist "%CKPT_SRC%" (
    if exist "%CKPT_SRC%\best_agent.pt" (
        echo [OK] Best checkpoint found: %CKPT_SRC%\best_agent.pt
    ) else if exist "%CKPT_SRC%\final_agent.pt" (
        echo [OK] Final checkpoint found: %CKPT_SRC%\final_agent.pt
    ) else (
        echo [WARN] No checkpoint files found in %CKPT_SRC%
    )
) else (
    echo [INFO] No checkpoints directory found (optional)
)

echo.
if %CHECK_ERRORS% GTR 0 (
    echo [CHECK] Found %CHECK_ERRORS% issue(s). Fix before deploying.
    exit /b 1
) else (
    echo [CHECK] Environment looks good!
    exit /b 0
)

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
echo   check   - Verify environment before deployment (FIX D1)
echo.
exit /b 1
