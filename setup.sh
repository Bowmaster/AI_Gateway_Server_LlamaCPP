#!/bin/bash
#
# AI Lab Server - Setup Script (llama.cpp backend)
#
# Creates a virtual environment, installs dependencies, detects hardware,
# and optionally starts the server.
#
# Usage:
#   ./setup.sh              # Setup only
#   ./setup.sh --start      # Setup and start server
#   ./setup.sh --start --port 9000  # Start on custom port
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
START_SERVER=false
PORT=8080

while [[ $# -gt 0 ]]; do
    case $1 in
        --start|-s)
            START_SERVER=true
            shift
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--start] [--port PORT]"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}======================================"
echo -e "  AI Lab Server - Setup Script"
echo -e "  (llama.cpp backend)"
echo -e "======================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python installation
echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"
PYTHON_CMD=""

for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1)
        if [[ $VERSION =~ Python\ ([0-9]+)\.([0-9]+) ]]; then
            MAJOR=${BASH_REMATCH[1]}
            MINOR=${BASH_REMATCH[2]}
            if [[ $MAJOR -ge 3 && $MINOR -ge 9 ]]; then
                PYTHON_CMD=$cmd
                echo -e "  ${GREEN}Found: $VERSION${NC}"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "  ${RED}ERROR: Python 3.9+ not found!${NC}"
    echo -e "  ${YELLOW}Please install Python 3.9 or later${NC}"
    exit 1
fi

# Check for NVIDIA GPU
echo ""
echo -e "${YELLOW}[2/5] Checking for NVIDIA GPU...${NC}"
HAS_GPU=false

if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo -e "  ${GREEN}NVIDIA GPU detected!${NC}"
        HAS_GPU=true

        # Try to detect CUDA version
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$CUDA_VERSION" ]; then
            echo -e "  ${CYAN}Driver Version: $CUDA_VERSION${NC}"
        fi
    else
        echo -e "  ${YELLOW}No NVIDIA GPU detected (CPU-only mode)${NC}"
    fi
else
    echo -e "  ${YELLOW}nvidia-smi not found (CPU-only mode)${NC}"
fi

# Create virtual environment
VENV_PATH="$SCRIPT_DIR/venv"
echo ""
echo -e "${YELLOW}[3/5] Setting up virtual environment...${NC}"

if [ -d "$VENV_PATH" ]; then
    echo -e "  ${GREEN}Virtual environment already exists at: $VENV_PATH${NC}"
else
    echo "  Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_PATH"
    echo -e "  ${GREEN}Created: $VENV_PATH${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[4/5] Activating and installing dependencies...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "  ${GREEN}Activated virtual environment${NC}"

# Upgrade pip
echo "  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
REQUIREMENTS_PATH="$SCRIPT_DIR/requirements.txt"

if [ -f "$REQUIREMENTS_PATH" ]; then
    echo "  Installing from requirements.txt..."
    pip install -r "$REQUIREMENTS_PATH" --quiet
    echo -e "  ${GREEN}Dependencies installed${NC}"
else
    echo -e "  ${YELLOW}WARNING: requirements.txt not found, installing core packages...${NC}"
    pip install fastapi uvicorn pydantic requests psutil huggingface_hub httpx --quiet
fi

# Hardware detection
echo ""
echo -e "${YELLOW}[5/5] Detecting hardware configuration...${NC}"

HARDWARE_DETECTOR="$SCRIPT_DIR/hardware_detector.py"
if [ -f "$HARDWARE_DETECTOR" ]; then
    if python -c "from hardware_detector import detect_and_save; detect_and_save('.hardware_profile.json')" 2>/dev/null; then
        PROFILE_PATH="$SCRIPT_DIR/.hardware_profile.json"
        if [ -f "$PROFILE_PATH" ]; then
            # Parse JSON with python
            python << 'PYEOF'
import json
with open('.hardware_profile.json') as f:
    p = json.load(f)

print(f"  System Type: {p.get('system_type', 'unknown')}")

gpu = p.get('gpu', {})
if gpu.get('has_gpu'):
    print(f"  GPU: {gpu.get('name', 'Unknown')} ({gpu.get('vram_gb', 0):.1f}GB VRAM)")
    print("  Mode: GPU-Accelerated")
else:
    print("  GPU: None detected")
    print("  Mode: CPU-Only")

cpu = p.get('cpu', {})
print(f"  CPU: {cpu.get('name', 'Unknown')}")

mem = p.get('memory', {})
print(f"  RAM: {mem.get('total_gb', 0):.1f}GB")

rec = p.get('recommended_config', {})
print(f"  Recommended GPU Layers: {rec.get('n_gpu_layers', 0)}")
PYEOF
        fi
    else
        echo -e "  ${YELLOW}Hardware detection skipped (run manually with hardware_detector.py)${NC}"
    fi
else
    echo -e "  ${YELLOW}Hardware detector not found, skipping...${NC}"
fi

# Check for llama-server
echo ""
echo -e "${CYAN}======================================"
echo -e "  Checking llama-server"
echo -e "======================================${NC}"

LLAMA_SERVER="$SCRIPT_DIR/llama-server"
if [ -f "$LLAMA_SERVER" ]; then
    echo -e "  ${GREEN}llama-server found${NC}"
elif command -v llama-server &> /dev/null; then
    echo -e "  ${GREEN}llama-server found in PATH${NC}"
else
    echo -e "  ${YELLOW}llama-server NOT found${NC}"
    echo ""
    echo -e "  ${YELLOW}You need to download or build llama-server:${NC}"
    echo -e "    Option 1: Download pre-built from llama.cpp releases"
    echo -e "    Option 2: Build from source with CUDA support"
    echo -e "    ${CYAN}https://github.com/ggerganov/llama.cpp/releases${NC}"
fi

# Check for models directory
MODELS_DIR="$SCRIPT_DIR/models"
if [ -d "$MODELS_DIR" ]; then
    MODEL_COUNT=$(find "$MODELS_DIR" -name "*.gguf" 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo -e "  ${GREEN}Found $MODEL_COUNT GGUF model(s) in models/${NC}"
    else
        echo -e "  ${YELLOW}No GGUF models found in models/${NC}"
    fi
else
    echo -e "  ${YELLOW}models/ directory not found${NC}"
fi

# Setup complete
echo ""
echo -e "${GREEN}======================================"
echo -e "  Setup Complete!"
echo -e "======================================${NC}"
echo ""
echo -e "${CYAN}To start the server manually:${NC}"
echo -e "  source venv/bin/activate"
echo -e "  python ai_server.py"
echo ""
echo -e "${CYAN}The server will be available at: http://localhost:$PORT${NC}"
echo ""

if [ ! -f "$LLAMA_SERVER" ] && ! command -v llama-server &> /dev/null; then
    echo -e "${YELLOW}IMPORTANT: Download/build llama-server before starting!${NC}"
    echo ""
fi

# Start server if requested
if [ "$START_SERVER" = true ]; then
    if [ ! -f "$LLAMA_SERVER" ] && ! command -v llama-server &> /dev/null; then
        echo -e "${RED}Cannot start server: llama-server not found${NC}"
        exit 1
    fi

    echo -e "${CYAN}Starting AI Lab Server on port $PORT...${NC}"
    echo ""

    python "$SCRIPT_DIR/ai_server.py"
fi
