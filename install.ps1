<#
    Hudka - one-command installer.

    Run this in PowerShell:

        irm https://raw.githubusercontent.com/preetamnegi7/hudka/main/install.ps1 | iex

    It installs the prerequisites you are missing, clones the repo, sets up the
    environment, and installs the audio engine. Everything it installs is free.

    Safe to run more than once - it skips whatever is already present.
#>

$ErrorActionPreference = 'Stop'

function Say([string]$text, [string]$colour = 'Gray') { Write-Host $text -ForegroundColor $colour }
function Step([string]$text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Have([string]$exe) { $null -ne (Get-Command $exe -ErrorAction SilentlyContinue) }

Write-Host ""
Say "  Hudka - AI sound design from video" 'Yellow'
Say "  Everything runs on your own machine." 'DarkGray'
Write-Host ""

# --------------------------------------------------------------------- uv
Step "Checking for uv (manages Python)"
if (Have 'uv') {
    Say "    already installed" 'Green'
} else {
    Say "    installing..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Have 'uv')) {
        Say "    uv installed but not on PATH yet." 'Yellow'
        Say "    Close this window, open a new PowerShell, and run this again." 'Yellow'
        return
    }
    Say "    done" 'Green'
}

# ------------------------------------------------------------------ ffmpeg
Step "Checking for ffmpeg (does the video and audio work)"
if (Have 'ffmpeg') {
    Say "    already installed" 'Green'
} elseif (Have 'winget') {
    Say "    installing via winget..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements --silent
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
    if (Have 'ffmpeg') {
        Say "    done" 'Green'
    } else {
        Say "    ffmpeg installed but not on PATH yet." 'Yellow'
        Say "    Close this window, open a new PowerShell, and run this again." 'Yellow'
        return
    }
} else {
    Say "    winget is unavailable, so ffmpeg cannot be installed automatically." 'Red'
    Say "    Install it from https://ffmpeg.org/download.html and run this again." 'Red'
    return
}

# --------------------------------------------------------------------- repo
Step "Getting Hudka"
$target = Join-Path (Get-Location) 'hudka'
if (Test-Path (Join-Path $target '.git')) {
    Say "    already here, updating..."
    Push-Location $target
    git pull --ff-only 2>&1 | Out-Null
} else {
    if (-not (Have 'git')) {
        Say "    git is not installed. Get it from https://git-scm.com and run this again." 'Red'
        return
    }
    git clone --quiet https://github.com/preetamnegi7/hudka.git $target
    Push-Location $target
}
Say "    $target" 'Green'

# ------------------------------------------------------------------- python
Step "Setting up Python and the pipeline"
uv sync --extra dev
Say "    done" 'Green'

# ------------------------------------------------------------------- engine
Step "Installing the audio engine (Stable Audio 3)"
uv pip install --quiet "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3" huggingface-hub
Say "    done" 'Green'

# --------------------------------------------------------------------- torch
# Order matters. stable-audio-3 pins torch==2.7.1 and pulls the CPU-only wheel, and a
# plain reinstall will NOT fix it: uv treats the version as already satisfied, because
# the "+cpu" local tag does not affect version matching. Hence --reinstall.
Step "Checking for an NVIDIA GPU"
if (Have 'nvidia-smi') {
    Say "    found - installing the CUDA build of PyTorch (this one is large)"
    uv pip install --quiet --reinstall torch==2.7.1 torchaudio==2.7.1 --torch-backend=cu128
    Say "    done" 'Green'
} else {
    Say "    none found - staying on the CPU build." 'Yellow'
    Say "    Generation will work, just more slowly." 'DarkGray'
}

# ------------------------------------------------------------------- weights
Step "Model access (one-off, free)"
Write-Host ""
Say "  The model weights are gated. Two steps:" 'White'
Write-Host ""
Say "    1. Sign in to Hugging Face and click Agree on BOTH pages:" 'White'
Say "       https://huggingface.co/stabilityai/stable-audio-3-small-sfx" 'Cyan'
Say "       https://huggingface.co/stabilityai/stable-audio-3-small-music" 'Cyan'
Write-Host ""
Say "    2. Paste an access token when prompted." 'White'
Say "       Create one at https://huggingface.co/settings/tokens (read access)" 'Cyan'
Write-Host ""

$answer = Read-Host "  Open those pages and log in now? [Y/n]"
if ($answer -eq '' -or $answer -match '^[Yy]') {
    Start-Process "https://huggingface.co/stabilityai/stable-audio-3-small-sfx"
    Start-Process "https://huggingface.co/stabilityai/stable-audio-3-small-music"
    Say "  Click Agree on both, then come back here." 'DarkGray'
    Read-Host "  Press Enter once you have accepted both"
    uv run hf auth login
}

# --------------------------------------------------------------------- done
Write-Host ""
uv run hudka doctor
Write-Host ""
Say "  Ready. Start it with:" 'Green'
Say "      cd $target" 'White'
Say "      uv run hudka gui" 'White'
Say "  ...or just double-click Hudka.bat in that folder." 'DarkGray'
Write-Host ""
Say "  The first render downloads about 3.3 GB of weights. After that it works offline." 'DarkGray'
Write-Host ""
Pop-Location
