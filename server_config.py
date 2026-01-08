"""
server_config.py - Configuration for AI Lab Server (llama.cpp edition)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import hardware detector for auto-configuration
try:
    from hardware_detector import load_hardware_profile, detect_and_save
    HAS_HARDWARE_DETECTOR = True
except ImportError:
    HAS_HARDWARE_DETECTOR = False
    logging.warning("hardware_detector not available - using static configuration")

logger = logging.getLogger(__name__)

# ============================================================================
# Server Settings
# ============================================================================

# Client-facing API server
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8080

# ============================================================================
# Hardware Detection & Auto-Configuration
# ============================================================================

HARDWARE_PROFILE_PATH = ".hardware_profile.json"

def load_or_detect_hardware() -> Dict[str, Any]:
    """
    Load existing hardware profile or run detection if not found.

    Returns:
        Hardware profile dictionary
    """
    if not HAS_HARDWARE_DETECTOR:
        # Fallback: No hardware detector available
        logger.warning("Hardware detector not available - using fallback profile")
        return {
            "version": "1.0",
            "system_type": "unknown",
            "gpu": {"has_gpu": False},
            "cpu": {"logical_cores": os.cpu_count() or 1},
            "memory": {"total_gb": 0},
            "recommended_config": {
                "n_gpu_layers": -1,  # Try GPU by default
                "ctx_size": 12288,
                "threads": None,
                "mode": "fallback"
            },
            "manual_overrides": {}
        }

    # Try to load existing profile
    profile = load_hardware_profile(HARDWARE_PROFILE_PATH)

    if profile:
        logger.info(f"Loaded hardware profile from {HARDWARE_PROFILE_PATH}")
        return profile

    # No profile found - run detection
    logger.info("No hardware profile found, running detection...")
    profile = detect_and_save(HARDWARE_PROFILE_PATH)

    return profile


def get_runtime_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get runtime configuration with override priority:
    1. Environment variables (highest priority)
    2. Manual overrides in profile
    3. Recommended config from hardware detection
    4. Hardcoded defaults (fallback)

    Returns:
        Tuple of (config_dict, hardware_profile)
    """
    # Load or detect hardware
    profile = load_or_detect_hardware()

    # Start with recommended config from hardware detection
    config = profile.get("recommended_config", {}).copy()

    # Apply manual overrides from profile (if present)
    manual_overrides = profile.get("manual_overrides", {})
    if manual_overrides:
        logger.info("Applying manual overrides from hardware profile")
        config.update(manual_overrides)

    # Apply environment variable overrides (highest priority)
    env_overrides = {}
    if os.getenv("N_GPU_LAYERS"):
        env_overrides["n_gpu_layers"] = int(os.getenv("N_GPU_LAYERS"))
        logger.info(f"Environment override: N_GPU_LAYERS={env_overrides['n_gpu_layers']}")

    if os.getenv("CTX_SIZE"):
        env_overrides["ctx_size"] = int(os.getenv("CTX_SIZE"))
        logger.info(f"Environment override: CTX_SIZE={env_overrides['ctx_size']}")

    if os.getenv("THREADS"):
        env_overrides["threads"] = int(os.getenv("THREADS"))
        logger.info(f"Environment override: THREADS={env_overrides['threads']}")

    config.update(env_overrides)

    # Ensure required keys exist with fallback defaults
    config.setdefault("n_gpu_layers", -1)
    config.setdefault("ctx_size", 12288)
    config.setdefault("threads", None)

    return config, profile


# ============================================================================
# llama-server Configuration
# ============================================================================

# Load runtime configuration based on detected hardware
_runtime_config, _hardware_profile = get_runtime_config()

LLAMA_SERVER_CONFIG = {
    # Path to llama-server executable
    # Default assumes it's in the same directory as this script
    "executable": os.getenv("LLAMA_SERVER_PATH", "./llama.cpp/llama-server.exe"),

    "cache_dir": os.getenv("LLAMA_CACHE", "./models"),

    # Internal server host/port (Python will proxy to this)
    "host": "127.0.0.1",
    "port": 8081,

    # DYNAMIC VALUES from hardware detection
    # These are auto-configured based on detected GPU, CPU, and RAM
    # Override with environment variables: N_GPU_LAYERS, CTX_SIZE, THREADS
    "n_gpu_layers": _runtime_config["n_gpu_layers"],
    "ctx_size": _runtime_config["ctx_size"],
    "threads": _runtime_config.get("threads"),

    # Auto-start llama-server when Python server starts
    "auto_start": True,

    # Additional command-line arguments (optional)
    # Example: ["--verbose", "--mlock"]
    "additional_args": [],
}

# Expose hardware profile globally for introspection
HARDWARE_PROFILE = _hardware_profile

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


def get_recommended_models_for_hardware() -> list:
    """
    Get list of recommended models based on detected hardware.

    Returns models that will run well on the current system:
    - High VRAM (>=16GB): 7B-14B models with Q4/Q5/Q8
    - Medium VRAM (>=8GB): 7B models with Q4/Q5
    - Low/No GPU: 3B-7B models with Q4 (smaller, faster on CPU)

    Returns:
        List of model keys recommended for current hardware
    """
    if not HARDWARE_PROFILE:
        # No profile available, return all models
        return list(MODELS.keys())

    gpu_info = HARDWARE_PROFILE.get("gpu", {})
    has_gpu = gpu_info.get("has_gpu", False)
    vram_gb = gpu_info.get("vram_gb", 0.0)

    recommended = []

    if has_gpu and vram_gb >= 16:
        # High VRAM - can handle larger models
        for key, info in MODELS.items():
            if "14b" in key or ("7b" in key and ("q5" in key or "q8" in key)):
                recommended.append(key)

    elif has_gpu and vram_gb >= 8:
        # Medium VRAM - 7B models work well
        for key in MODELS.keys():
            if "7b" in key:
                recommended.append(key)

    else:
        # CPU mode or low VRAM - smaller models, Q4 quantization
        for key in MODELS.keys():
            if "3b" in key or ("7b" in key and "q4" in key):
                recommended.append(key)

    # If no matches, return all models
    if not recommended:
        return list(MODELS.keys())

    return recommended