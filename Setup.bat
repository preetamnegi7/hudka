@echo off
REM One-time setup. Double-click this once, then use "Hudka.bat" to run the app.
setlocal
cd /d "%~dp0"
title Hudka - Setup

echo ============================================
echo   Hudka - first-time setup
echo ============================================
echo.

REM ---------------------------------------------------------------- prerequisites
where uv >nul 2>&1
if errorlevel 1 (
  echo [X] uv is not installed. It manages Python for this project.
  echo.
  echo     Install it, then run this again:
  echo     powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
  echo.
  pause
  exit /b 1
)
echo [OK] uv found

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [X] ffmpeg is not on PATH. It does all the video and audio handling.
  echo.
  echo     winget install Gyan.FFmpeg
  echo     then close and reopen this window.
  echo.
  pause
  exit /b 1
)
echo [OK] ffmpeg found
echo.

REM ---------------------------------------------------------------- core install
echo [1/4] Installing the core pipeline...
call uv sync --extra dev
if errorlevel 1 goto failed
echo.

REM ---------------------------------------------------------------- engine
echo [2/4] Installing Stable Audio 3 (the generator)...
call uv pip install "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3" huggingface-hub
if errorlevel 1 goto failed
echo.

REM ---------------------------------------------------------------- torch
REM Order matters and --reinstall is required. stable-audio-3 pins torch==2.7.1 and
REM pulls the CPU-only wheel, which silently makes generation crawl on a good GPU.
REM Plain "uv pip install torch==2.7.1" will NOT fix it: uv sees the version already
REM satisfied and skips, because the "+cpu" local tag does not affect version matching.
echo [3/4] Checking for an NVIDIA GPU...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo      No NVIDIA GPU detected - staying on the CPU build.
  echo      Generation will work, but far more slowly.
) else (
  echo      NVIDIA GPU found. Installing the CUDA build of torch...
  call uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 --torch-backend=cu128
  if errorlevel 1 (
    echo      CUDA install failed; leaving the CPU build in place.
  )
)
echo.

REM ---------------------------------------------------------------- weights
echo [4/4] Model weights
echo.
echo   The weights are gated. Two one-off steps, both free:
echo.
echo     1. Sign in to Hugging Face and click Agree on BOTH pages:
echo        https://huggingface.co/stabilityai/stable-audio-3-small-sfx
echo        https://huggingface.co/stabilityai/stable-audio-3-small-music
echo.
echo     2. Paste an access token when prompted below.
echo        Create one at https://huggingface.co/settings/tokens (read access is enough)
echo.
echo   The first render then downloads about 3.3 GB. After that it all runs offline.
echo.
choice /c YN /m "Log in to Hugging Face now"
if errorlevel 2 goto skiplogin
call uv run hf auth login
:skiplogin
echo.

echo ============================================
call uv run hudka doctor
echo ============================================
echo.
echo Setup finished. Launch the app with "Hudka.bat".
echo.
echo Note: running "uv sync" later removes the engine and the CUDA build.
echo       If generation stops working, just run this script again.
echo.
pause
exit /b 0

:failed
echo.
echo Setup failed above. Fix the reported error and run this again.
pause
exit /b 1
