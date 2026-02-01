<#
.SYNOPSIS
    Setup script for AI Lab Server (llama.cpp backend)

.DESCRIPTION
    Creates a virtual environment, installs dependencies, detects hardware,
    and optionally starts the server.

.PARAMETER StartServer
    If specified, starts the AI server after setup

.PARAMETER Port
    Override the default server port (default: 8080)

.PARAMETER Firewall
    If specified, creates a Windows Firewall inbound rule to allow remote
    access to the server. Requires elevated (admin) privileges.

.EXAMPLE
    .\setup.ps1
    # Sets up the environment only

.EXAMPLE
    .\setup.ps1 -StartServer
    # Sets up the environment and starts the server

.EXAMPLE
    .\setup.ps1 -StartServer -Port 9000
    # Sets up and starts the server on port 9000

.EXAMPLE
    .\setup.ps1 -Firewall
    # Sets up the environment and creates a firewall rule for remote access
#>

param(
    [switch]$StartServer,
    [switch]$Firewall,
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  AI Lab Server - Setup Script" -ForegroundColor Cyan
Write-Host "  (llama.cpp backend)" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check Python installation
Write-Host "[1/7] Checking Python installation..." -ForegroundColor Yellow
$PythonCmd = $null

# Try different Python commands
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $version = & $cmd --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 9) {
                $PythonCmd = $cmd
                Write-Host "  Found: $version" -ForegroundColor Green
                break
            }
        }
    } catch {
        continue
    }
}

if (-not $PythonCmd) {
    Write-Host "  ERROR: Python 3.9+ not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.9 or later from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Check for NVIDIA GPU
Write-Host ""
Write-Host "[2/7] Checking for NVIDIA GPU..." -ForegroundColor Yellow
$HasGPU = $false

try {
    $nvidiaCheck = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  NVIDIA GPU detected!" -ForegroundColor Green
        $HasGPU = $true

        # Try to detect CUDA version
        if ($nvidiaCheck -match "CUDA Version: (\d+\.\d+)") {
            $cudaVersion = $Matches[1]
            Write-Host "  CUDA Version: $cudaVersion" -ForegroundColor Cyan
        }
    } else {
        Write-Host "  No NVIDIA GPU detected (CPU-only mode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  No NVIDIA GPU detected (CPU-only mode)" -ForegroundColor Yellow
}

# Check for Visual C++ Redistributable (required by llama-server.exe)
Write-Host ""
Write-Host "[3/7] Checking Visual C++ Redistributable..." -ForegroundColor Yellow

$VCRedistInstalled = $false
$VCRegPaths = @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"
)

foreach ($regPath in $VCRegPaths) {
    if (Test-Path $regPath) {
        $VCRedistInstalled = $true
        break
    }
}

if ($VCRedistInstalled) {
    Write-Host "  Visual C++ Redistributable found" -ForegroundColor Green
} else {
    Write-Host "  Visual C++ Redistributable (x64) NOT found!" -ForegroundColor Red
    Write-Host "  llama-server.exe requires this runtime and will crash without it." -ForegroundColor Yellow
    Write-Host "  Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Cyan
    Write-Host ""

    $installChoice = Read-Host "  Would you like to download and install it now? (Y/N)"
    if ($installChoice -eq 'Y' -or $installChoice -eq 'y') {
        Write-Host "  Downloading VC++ Redistributable..." -ForegroundColor Yellow
        $vcRedistPath = Join-Path $ScriptDir "vc_redist.x64.exe"
        try {
            Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vcRedistPath
            Write-Host "  Installing (may require admin privileges)..." -ForegroundColor Yellow
            Start-Process -FilePath $vcRedistPath -ArgumentList "/install", "/quiet", "/norestart" -Wait
            Write-Host "  Visual C++ Redistributable installed" -ForegroundColor Green
            Remove-Item $vcRedistPath -ErrorAction SilentlyContinue
        } catch {
            Write-Host "  Failed to download/install automatically." -ForegroundColor Red
            Write-Host "  Please install manually from the URL above." -ForegroundColor Yellow
        }
    }
}

# Create virtual environment
$VenvPath = Join-Path $ScriptDir "venv"
Write-Host ""
Write-Host "[4/7] Setting up virtual environment..." -ForegroundColor Yellow

if (Test-Path $VenvPath) {
    Write-Host "  Virtual environment already exists at: $VenvPath" -ForegroundColor Green
} else {
    Write-Host "  Creating virtual environment..."
    & $PythonCmd -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Created: $VenvPath" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[5/7] Activating and installing dependencies..." -ForegroundColor Yellow
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $ActivateScript)) {
    Write-Host "  ERROR: Activation script not found!" -ForegroundColor Red
    exit 1
}

# Source the activation script
. $ActivateScript
Write-Host "  Activated virtual environment" -ForegroundColor Green

# Upgrade pip
Write-Host "  Upgrading pip..."
& pip install --upgrade pip --quiet

# Install dependencies
$RequirementsPath = Join-Path $ScriptDir "requirements.txt"

if (Test-Path $RequirementsPath) {
    Write-Host "  Installing from requirements.txt..."
    & pip install -r $RequirementsPath --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to install dependencies!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  WARNING: requirements.txt not found, installing core packages..." -ForegroundColor Yellow
    & pip install fastapi uvicorn pydantic requests psutil huggingface_hub httpx --quiet
}

# Hardware detection
Write-Host ""
Write-Host "[6/7] Detecting hardware configuration..." -ForegroundColor Yellow

$HardwareDetector = Join-Path $ScriptDir "hardware_detector.py"
if (Test-Path $HardwareDetector) {
    try {
        & python -c "from hardware_detector import detect_and_save; detect_and_save('.hardware_profile.json')" 2>&1 | Out-Null

        $ProfilePath = Join-Path $ScriptDir ".hardware_profile.json"
        if (Test-Path $ProfilePath) {
            $profile = Get-Content $ProfilePath | ConvertFrom-Json

            Write-Host "  System Type: $($profile.system_type)" -ForegroundColor Cyan

            if ($profile.gpu.has_gpu) {
                Write-Host "  GPU: $($profile.gpu.name) ($($profile.gpu.vram_gb)GB VRAM)" -ForegroundColor Green
                Write-Host "  Mode: GPU-Accelerated" -ForegroundColor Green
            } else {
                Write-Host "  GPU: None detected" -ForegroundColor Yellow
                Write-Host "  Mode: CPU-Only" -ForegroundColor Yellow
            }

            Write-Host "  CPU: $($profile.cpu.name)" -ForegroundColor Cyan
            Write-Host "  RAM: $([math]::Round($profile.memory.total_gb, 1))GB" -ForegroundColor Cyan
            Write-Host "  Recommended GPU Layers: $($profile.recommended_config.n_gpu_layers)" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "  Hardware detection skipped (run manually with hardware_detector.py)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Hardware detector not found, skipping..." -ForegroundColor Yellow
}

# Check for llama-server
Write-Host ""
Write-Host "[7/7] Checking llama-server" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

$LlamaServer = Join-Path $ScriptDir "llama-server.exe"
if (Test-Path $LlamaServer) {
    Write-Host "  llama-server.exe found" -ForegroundColor Green
} else {
    Write-Host "  llama-server.exe NOT found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  You need to download or build llama-server:" -ForegroundColor Yellow
    Write-Host "    Option 1: Download pre-built from llama.cpp releases" -ForegroundColor White
    Write-Host "    Option 2: Build from source with CUDA support" -ForegroundColor White
    Write-Host "    https://github.com/ggerganov/llama.cpp/releases" -ForegroundColor Cyan
}

# Check for models directory
$ModelsDir = Join-Path $ScriptDir "models"
if (Test-Path $ModelsDir) {
    $ModelCount = (Get-ChildItem -Path $ModelsDir -Filter "*.gguf" -ErrorAction SilentlyContinue).Count
    if ($ModelCount -gt 0) {
        Write-Host "  Found $ModelCount GGUF model(s) in models/" -ForegroundColor Green
    } else {
        Write-Host "  No GGUF models found in models/" -ForegroundColor Yellow
    }
} else {
    Write-Host "  models/ directory not found" -ForegroundColor Yellow
}

# Firewall rule (opt-in only)
if ($Firewall) {
    Write-Host ""
    Write-Host "Configuring Windows Firewall..." -ForegroundColor Yellow

    $ruleName = "AI Lab Server (TCP $Port)"
    $existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

    if ($existingRule) {
        Write-Host "  Firewall rule '$ruleName' already exists" -ForegroundColor Green
    } else {
        try {
            New-NetFirewallRule -DisplayName $ruleName `
                -Direction Inbound `
                -LocalPort $Port `
                -Protocol TCP `
                -Action Allow `
                -Description "Allow remote access to AI Lab Server API" | Out-Null
            Write-Host "  Created inbound rule: $ruleName" -ForegroundColor Green
        } catch {
            Write-Host "  ERROR: Failed to create firewall rule." -ForegroundColor Red
            Write-Host "  Re-run setup as Administrator: .\setup.ps1 -Firewall" -ForegroundColor Yellow
        }
    }
}

# Setup complete
Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the server manually:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python ai_server.py" -ForegroundColor White
Write-Host ""
Write-Host "The server will be available at: http://localhost:$Port" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $LlamaServer)) {
    Write-Host "IMPORTANT: Download llama-server.exe before starting!" -ForegroundColor Yellow
    Write-Host ""
}

# Start server if requested
if ($StartServer) {
    if (-not (Test-Path $LlamaServer)) {
        Write-Host "Cannot start server: llama-server.exe not found" -ForegroundColor Red
        exit 1
    }

    Write-Host "Starting AI Lab Server on port $Port..." -ForegroundColor Cyan
    Write-Host ""

    $ServerScript = Join-Path $ScriptDir "ai_server.py"
    & python $ServerScript
}
