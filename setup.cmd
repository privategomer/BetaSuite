@echo off
rem setup.cmd - double-click entry point for setup.ps1 on Windows.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set STATUS=%ERRORLEVEL%
echo.
pause
exit /b %STATUS%
