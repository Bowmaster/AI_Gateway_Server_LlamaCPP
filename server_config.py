"""
server_config.py - Configuration for AI Lab Server (llama.cpp edition)
"""

import os
from pathlib import Path

# ============================================================================
# Server Settings
# ============================================================================

# Client-facing API server
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8080

# ============================================================================
# llama-server Configuration
# ============================================================================

LLAMA_SERVER_CONFIG = {
    # Path to llama-server executable
    # Default assumes it's in the same directory as this script
    "executable": os.getenv("LLAMA_SERVER_PATH", "./llama.cpp/llama-server.exe"),
    
    "cache_dir": os.getenv("LLAMA_CACHE", "./models"),

    # Internal server host/port (Python will proxy to this)
    "host": "127.0.0.1",
    "port": 8081,
    
    # GPU configuration
    # -1 = all layers on GPU (full GPU mode)
    # 0 = all layers on CPU (CPU-only mode)
    # N = offload N layers to GPU (hybrid mode)
    "n_gpu_layers": int(os.getenv("N_GPU_LAYERS", "-1")),
    
    # Context window size (in tokens)
    "ctx_size": 12288,
    
    # CPU threads (None = auto-detect)
    "threads": None,
    
    # Auto-start llama-server when Python server starts
    "auto_start": True,
    
    # Additional command-line arguments (optional)
    # Example: ["--verbose", "--mlock"]
    "additional_args": [],
}

# ============================================================================
# Model Registry
# ============================================================================

# Directory where GGUF models are stored
MODELS_DIR = "./models"

# Model Registry
# Define all available GGUF models with their properties
MODELS = {
    # Qwen2.5 models - Excellent all-rounders
    "qwen2.5-7b-q4": {
        "name": "Qwen2.5-7B-Instruct-Q4_K_M",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",  # Local file
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",  # NEW: HuggingFace fallback
        "description": "Qwen2.5 7B: Best balance of quality and speed (Q4_K_M quantization)",
        "context_length": 32768,
        "vram_estimate": "~5GB",
        "recommended": True,
        "download_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "usage": "Best all-rounder: General chat, coding, reasoning. Fast and efficient.",
    },
    
    "qwen2.5-7b-q5": {
        "name": "Qwen2.5-7B-Instruct-Q5_K_M",
        "filename": "qwen2.5-7b-instruct-q5_k_m.gguf",
        "description": "Qwen2.5 7B: Higher quality quantization (Q5_K_M)",
        "context_length": 32768,
        "vram_estimate": "~6GB",
        "recommended": False,
        "download_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q5_k_m.gguf",
        "usage": "Higher quality: Better responses, slightly slower. Use when quality matters more than speed.",
    },
    
    "qwen2.5-7b-q8": {
        "name": "Qwen2.5-7B-Instruct-Q8_0",
        "filename": "qwen2.5-7b-instruct-q8_0.gguf",
        "description": "Qwen2.5 7B: Maximum quality quantization (Q8_0)",
        "context_length": 32768,
        "vram_estimate": "~8GB",
        "recommended": False,
        "download_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q8_0.gguf",
        "usage": "Maximum quality: Near-FP16 quality. Use when you have VRAM to spare.",
    },
    
    # Qwen2.5-Coder - Code specialist
    "qwen2.5-coder-7b-q4": {
        "name": "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-Coder 7B: Specialized for coding (Q4_K_M)",
        "context_length": 32768,
        "vram_estimate": "~5GB",
        "recommended": False,
        "download_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "usage": "Code-focused: Writing code, debugging, code review, explaining algorithms.",
    },
    
    # Larger models
    "qwen2.5-14b-q4": {
        "name": "Qwen2.5-14B-Instruct-Q4_K_M",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5 14B: More capable model (Q4_K_M)",
        "context_length": 32768,
        "vram_estimate": "~9GB",
        "recommended": False,
        "download_url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf",
        "usage": "Advanced tasks: Complex reasoning, nuanced analysis, technical writing.",
    },
    
    # Mistral Ministral-3-8B
    "ministral-3-8b-q8": {
        "name": "Ministral-3-8B-Instruct-Q8_0",
        "filename": "mistralai_Ministral-3-8B-Instruct-2512-GGUF_Ministral-3-8B-Instruct-2512-Q8_0.gguf",
        "description": "Mistral Ministral-3 8B: Latest model with excellent instruction following (Q8_0)",
        "context_length": 32768,
        "hf_repo": "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
        "vram_estimate": "~9GB",
        "recommended": False,
        "download_url": "https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/resolve/main/Ministral-3-8B-Instruct-2512-Q8_0.gguf",
        "usage": "Latest Mistral model: Excellent instruction following, general chat, coding, reasoning. High quality Q8 quantization.",
    },
    
    # DeepSeek-R1 - Reasoning specialist
    "deepseek-r1-7b-q4": {
        "name": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M",
        "filename": "deepseek-r1-distill-qwen-7b-q4_k_m.gguf",
        "description": "DeepSeek-R1 7B: Reasoning specialist with thinking traces (Q4_K_M)",
        "context_length": 32768,
        "vram_estimate": "~5GB",
        "recommended": False,
        "download_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B.Q4_K_M.gguf",
        "usage": "Deep reasoning: Math problems, logic puzzles, complex analysis. Shows reasoning process.",
    },
    
    # Llama models
    "llama3.2-3b-q4": {
        "name": "Llama-3.2-3B-Instruct-Q4_K_M",
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "description": "Llama 3.2 3B: Small, fast model for testing (Q4_K_M)",
        "context_length": 131072,
        "vram_estimate": "~3GB",
        "recommended": False,
        "download_url": "https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "usage": "Quick testing: Rapid prototyping, basic chat, verifying setup.",
    },
}

# Update default to Qwen (better context handling)
#DEFAULT_MODEL_KEY = "qwen2.5-7b-q4"
DEFAULT_MODEL_KEY = "ministral-3-8b-q8"

# ============================================================================
# Generation Defaults
# ============================================================================

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
DEFAULT_TEMPERATURE = 0.7
#DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.1

# ============================================================================
# Conversation Management
# ============================================================================

# Maximum tokens to keep in conversation history
# (helps prevent context overflow)
MAX_CONVERSATION_TOKENS = 3000

# Maximum number of messages to keep in history
# (0 = unlimited, relies on token limit instead)
MAX_CONVERSATION_MESSAGES = 0

# ============================================================================
# Tool/Function Calling
# ============================================================================

# Enable tool/function calling by default
ENABLE_TOOLS = True

# Maximum number of tool iterations per request
# (prevents infinite tool loops)
MAX_TOOL_ITERATIONS = 5

# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# Helper Functions
# ============================================================================

def get_model_path(model_key: str) -> str:
    """
    Get the full path to a model file.
    
    Args:
        model_key: The model key from MODELS registry
        
    Returns:
        Full path to the model file
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    filename = MODELS[model_key]["filename"]
    return os.path.join(MODELS_DIR, filename)


def model_exists(model_key: str) -> bool:
    """
    Check if a model is available (either locally or via HuggingFace).
    """
    try:
        get_model_source(model_key)
        return True
    except ValueError:
        return False


def get_model_info(model_key: str) -> dict:
    """
    Get model information from registry.
    
    Args:
        model_key: The model key from MODELS registry
        
    Returns:
        Model info dictionary, or None if not found
    """
    return MODELS.get(model_key)


def list_available_models(include_missing: bool = False) -> list:
    """
    List all models from registry, optionally filtering out missing files.
    
    Args:
        include_missing: If False, only return models that exist on disk
        
    Returns:
        List of model keys
    """
    if include_missing:
        return list(MODELS.keys())
    else:
        return [key for key in MODELS.keys() if model_exists(key)]


def validate_config() -> list:
    """
    Validate configuration and return list of warnings/errors.
    
    Returns:
        List of warning/error messages (empty if all OK)
    """
    issues = []
    
    # Check if llama-server executable exists
    executable = LLAMA_SERVER_CONFIG["executable"]
    if not os.path.exists(executable):
        issues.append(f"llama-server executable not found: {executable}")
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        issues.append(f"Models directory not found: {MODELS_DIR}")
    
    # Check if default model exists
    if DEFAULT_MODEL_KEY not in MODELS:
        issues.append(f"Default model key not in registry: {DEFAULT_MODEL_KEY}")
    elif not model_exists(DEFAULT_MODEL_KEY):
        issues.append(f"Default model file not found: {get_model_path(DEFAULT_MODEL_KEY)}")
    
    # Check for at least one available model
    available = list_available_models(include_missing=False)
    if not available:
        issues.append("No model files found in models directory")
    
    return issues


def get_model_source(model_key: str) -> tuple:
    """
    Determine model source and path/identifier.
    Returns (source_type, identifier) where:
    - source_type: "local" or "huggingface"
    - identifier: file path for local, repo:quant for HF
    
    Args:
        model_key: The model key from MODELS registry
        
    Returns:
        Tuple of (source_type, identifier)
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model_info = MODELS[model_key]
    
    # Check for local file first
    if "filename" in model_info:
        local_path = os.path.join(MODELS_DIR, model_info["filename"])
        if os.path.exists(local_path):
            return ("local", local_path)
    
    # Fall back to HuggingFace if available
    if "hf_repo" in model_info:
        return ("huggingface", model_info["hf_repo"])
    
    # If we have a filename but file doesn't exist, and no HF fallback
    if "filename" in model_info:
        return ("local", os.path.join(MODELS_DIR, model_info["filename"]))
    
    raise ValueError(f"Model {model_key} has no local file or HuggingFace repo defined")