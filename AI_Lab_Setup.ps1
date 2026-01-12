# AI_Lab_Setup.ps1 (llama.cpp edition)
# Setup script for llama.cpp-based AI experimentation environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Lab Setup Script (llama.cpp)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking for Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check Python version (need 3.9+)
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
        Write-Host "ERROR: Python 3.9+ required. You have Python $major.$minor" -ForegroundColor Red
        exit 1
    }
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "ai_lab_cpp") {
    Write-Host "Virtual environment 'ai_lab_cpp' already exists." -ForegroundColor Yellow
    $response = Read-Host "Delete and recreate? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ai_lab_cpp
        python -m venv ai_lab_cpp
        Write-Host "New virtual environment created." -ForegroundColor Green
    } else {
        Write-Host "Using existing environment." -ForegroundColor Green
    }
} else {
    python -m venv ai_lab_cpp
    Write-Host "Virtual environment created." -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\ai_lab_cpp\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Check for NVIDIA GPU
Write-Host ""
Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaCheck = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "NVIDIA GPU detected!" -ForegroundColor Green
        $installGPU = $true
        
        # Try to detect CUDA version
        if ($nvidiaCheck -match "CUDA Version: (\d+\.\d+)") {
            $cudaVersion = $Matches[1]
            Write-Host "CUDA Version: $cudaVersion" -ForegroundColor Green
        }
    } else {
        Write-Host "No NVIDIA GPU detected. Installing CPU-only version." -ForegroundColor Yellow
        $installGPU = $false
    }
} catch {
    Write-Host "No NVIDIA GPU detected. Installing CPU-only version." -ForegroundColor Yellow
    $installGPU = $false
}

# Install llama-cpp-python
<#
Write-Host ""
Write-Host "Installing llama-cpp-python..." -ForegroundColor Yellow
if ($installGPU) {
    Write-Host "Installing with CUDA support..." -ForegroundColor Cyan
    Write-Host "(This may take several minutes as it compiles with CUDA)" -ForegroundColor Yellow
    
    # Set environment variables for CUDA build
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    $env:FORCE_CMAKE = "1"
    
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
    
    # Clear environment variables
    $env:CMAKE_ARGS = ""
    $env:FORCE_CMAKE = ""
} else {
    Write-Host "Installing CPU-only version..." -ForegroundColor Cyan
    pip install llama-cpp-python
}
#>

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: llama-cpp-python installation failed." -ForegroundColor Red
    Write-Host "You may need to install Visual Studio Build Tools." -ForegroundColor Yellow
    Write-Host "Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
    exit 1
}

# Install server/client dependencies
Write-Host ""
Write-Host "Installing server/client dependencies..." -ForegroundColor Yellow
pip install fastapi uvicorn pydantic requests rich psutil httpx

# Install optional but useful packages
Write-Host ""
Write-Host "Installing additional tools..." -ForegroundColor Yellow
pip install huggingface_hub nvidia-ml-py  # HuggingFace downloads + GPU detection (replaces deprecated pynvml)

# Install web search and scraping tools
Write-Host ""
Write-Host "Installing web search and scraping tools..." -ForegroundColor Yellow
pip install duckduckgo-search beautifulsoup4 lxml

# Verify installation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Create verification script
$verifyScript = @'
import sys

print("\n=== Installation Verification ===\n")

# Python version
print(f"Python version: {sys.version.split()[0]}")

# llama-cpp-python
try:
    from llama_cpp import Llama
    import llama_cpp
    print(f"llama-cpp-python version: {llama_cpp.__version__}")
    
    # Check for CUDA support
    try:
        # Try to check CUDA availability
        # This is a bit tricky as llama.cpp doesn't expose this directly
        print("\nâœ" llama-cpp-python installed successfully")
        print("  Note: CUDA support detection requires loading a model")
    except Exception as e:
        print(f"\nâœ— Issue checking CUDA: {e}")
except ImportError as e:
    print(f"\nâœ— llama-cpp-python not installed properly: {e}")

print("\n=== Installed Packages ===\n")
import pkg_resources
packages = ['llama-cpp-python', 'fastapi', 'uvicorn', 'pydantic', 'requests', 'rich', 'psutil', 'huggingface-hub', 'httpx', 'duckduckgo-search', 'beautifulsoup4', 'lxml']
for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"  {package}: {version}")
    except:
        print(f"  {package}: Not installed")

print("\n=== Setup Complete! ===\n")
print("To activate this environment in the future, run:")
print("  .\\ai_lab_cpp\\Scripts\\Activate.ps1")
print("\nNext steps:")
print("  1. Download GGUF models (see README.md)")
print("  2. Update server_config.py with model paths")
print("  3. Run: python ai_server.py")
'@

$verifyScript | Out-File -FilePath "verify_setup.py" -Encoding UTF8

# Run verification
python verify_setup.py

# Clean up verification script
Remove-Item verify_setup.py

# Run hardware detection
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Hardware Detection" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Detecting system hardware..." -ForegroundColor Yellow

try {
    python -c "from hardware_detector import detect_and_save; detect_and_save('.hardware_profile.json')" 2>&1 | Out-Null

    if ($LASTEXITCODE -eq 0 -and (Test-Path ".hardware_profile.json")) {
        Write-Host "Hardware profile created successfully!" -ForegroundColor Green
        Write-Host ""

        # Load and display the profile
        $profile = Get-Content .hardware_profile.json | ConvertFrom-Json

        Write-Host "System Type: " -NoNewline -ForegroundColor White
        Write-Host $profile.system_type -ForegroundColor Cyan

        # GPU info
        if ($profile.gpu.has_gpu) {
            Write-Host "GPU: " -NoNewline -ForegroundColor White
            Write-Host "$($profile.gpu.name) ($($profile.gpu.vram_gb)GB VRAM)" -ForegroundColor Green
            Write-Host "  CUDA: " -NoNewline -ForegroundColor White
            Write-Host $profile.gpu.cuda_version -ForegroundColor Cyan
            Write-Host "  Mode: GPU-Accelerated" -ForegroundColor Green
            Write-Host "  Recommended: 7B-14B models with Q4/Q5 quantization" -ForegroundColor Yellow
        } else {
            Write-Host "GPU: " -NoNewline -ForegroundColor White
            Write-Host "None detected" -ForegroundColor Yellow
            Write-Host "  Mode: CPU-Only" -ForegroundColor Yellow
            Write-Host "  Recommended: 3B-7B models with Q4 quantization" -ForegroundColor Yellow
        }

        # CPU info
        Write-Host "CPU: " -NoNewline -ForegroundColor White
        Write-Host "$($profile.cpu.name)" -ForegroundColor Cyan
        Write-Host "  Threads: " -NoNewline -ForegroundColor White
        Write-Host "$($profile.cpu.logical_cores)" -ForegroundColor Cyan

        # RAM info
        Write-Host "RAM: " -NoNewline -ForegroundColor White
        Write-Host "$([math]::Round($profile.memory.total_gb, 1))GB" -ForegroundColor Cyan

        # Configuration
        Write-Host ""
        Write-Host "Recommended Configuration:" -ForegroundColor Yellow
        Write-Host "  Context Size: " -NoNewline -ForegroundColor White
        Write-Host "$($profile.recommended_config.ctx_size) tokens" -ForegroundColor Cyan
        Write-Host "  GPU Layers: " -NoNewline -ForegroundColor White
        Write-Host $profile.recommended_config.n_gpu_layers -ForegroundColor Cyan
        Write-Host "  Reasoning: " -NoNewline -ForegroundColor White
        Write-Host $profile.recommended_config.reasoning -ForegroundColor Gray

    } else {
        Write-Host "Hardware detection completed with warnings. Check logs for details." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Hardware detection failed: $_" -ForegroundColor Red
    Write-Host "Continuing with setup - you can run detection manually later." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Download GGUF models from HuggingFace" -ForegroundColor White
Write-Host "     Example: Qwen2.5-7B-Instruct Q4_K_M quantization" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Create a 'models' directory and place GGUF files there:" -ForegroundColor White
Write-Host "     .\models\qwen2.5-7b-instruct-q4_k_m.gguf" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Update server_config.py with your model paths" -ForegroundColor White
Write-Host ""
Write-Host "  4. Start the server:" -ForegroundColor White
Write-Host "     python ai_server.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  5. In another terminal, run the client:" -ForegroundColor White
Write-Host "     python ai_client.py" -ForegroundColor Cyan
Write-Host ""
