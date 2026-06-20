@echo off
cd /d "%~dp0"
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%

echo ========================================
echo   LiveTranslate Updater
echo ========================================
echo.

:: Check git
git --version >nul 2>&1
if errorlevel 1 (
    echo Git not found, attempting to install via winget...
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Git not found and winget is not available.
        echo Please install Git from https://git-scm.com/downloads
        pause
        exit /b 1
    )
    winget install Git.Git --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Git installation failed.
        pause
        exit /b 1
    )
    :: Refresh PATH
    set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%ProgramFiles%\Git\cmd;%PATH%"
    git --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Git installed but not found in PATH. Please restart and try again.
        pause
        exit /b 1
    )
    echo Git installed successfully.
    echo.
)

:: Pull latest code
echo Pulling latest changes...
git pull
if errorlevel 1 (
    echo.
    echo [ERROR] git pull failed. Check for local conflicts.
    pause
    exit /b 1
)

:: Check venv
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Virtual environment not found, running install.bat...
    call install.bat
    exit /b %errorlevel%
)

:: Check uv
uv --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo uv not found, running install.bat...
    call install.bat
    exit /b %errorlevel%
)

:: Update dependencies
echo.
echo Syncing dependencies...
uv sync --python .venv\Scripts\python.exe --locked --inexact --no-install-package torch --no-install-package torchaudio --quiet
if errorlevel 1 (
    echo [WARN] Some dependencies failed to sync.
)

if exist "repair_torch_metadata.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "repair_torch_metadata.ps1" -PythonExe ".venv\Scripts\python.exe"
)

uv pip install --python .venv\Scripts\python.exe funasr --no-deps --quiet

echo.
echo ========================================
echo   Update complete!
echo ========================================
echo.
echo Double-click start.bat to launch.
echo.
pause
