@echo off
REM Double-click launcher. Starts the app and opens it in your browser.
cd /d "%~dp0"
title Saand Efectos
echo Starting Saand Efectos...
echo.
echo Keep this window open while you work. Close it to stop the app.
echo.
uv run saand gui
if errorlevel 1 (
  echo.
  echo Something went wrong. Press any key to close.
  pause >nul
)
