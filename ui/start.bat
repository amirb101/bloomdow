@echo off
REM Wrapper for start.ps1 — use if PowerShell execution policy blocks scripts
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"
pause
