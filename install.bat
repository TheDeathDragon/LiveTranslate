@echo off
cd /d "%~dp0"
where pwsh >nul 2>&1
if not errorlevel 1 (
    pwsh -ExecutionPolicy Bypass -File "%~dp0install.ps1"
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"
)
