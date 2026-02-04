"""
ai_server.py - AI Lab Server (llama.cpp edition)
FastAPI server that manages llama-server and provides chat interface
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, AsyncGenerator, Union
import logging
import os
import sys
import signal
import atexit
import time
import uvicorn
import json
import requests
import httpx
import asyncio
import uuid

import server_config as config
from llama_manager import LlamaServerManager
import tools



# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Uvicorn reconfigures the root logger on startup, which can swallow our
# app-level messages. Attach a StreamHandler directly to the loggers we
# care about so startup/crash diagnostics are always visible.
_app_handler = logging.StreamHandler(sys.stderr)
_app_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
for _logger_name in (__name__, "llama_manager", "server_config", "hardware_detector"):
    _l = logging.getLogger(_logger_name)
    if not _l.handlers:
        _l.addHandler(_app_handler)
        _l.setLevel(config.LOG_LEVEL)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="AI Lab Server (llama.cpp)",
    version="2.0.0",
    description="FastAPI wrapper for llama-server with model management and tool calling"
)

# ============================================================================
# Global State
# ============================================================================

class ServerState:
    def __init__(self):
        self.llama_manager: Optional[LlamaServerManager] = None
        self.current_model_key: str = config.DEFAULT_MODEL_KEY
        self.system_prompt: str = config.DEFAULT_SYSTEM_PROMPT
        self.conversation_history: List[Dict[str, str]] = []
        self.is_generating: bool = False
        self.tools_enabled: bool = config.ENABLE_TOOLS
        self.shutdown_requested: bool = False

        # Approval system state
        self.pending_tool_calls: List[Dict] = []  # Tool calls awaiting approval
        self.pending_messages: List[Dict] = []  # Message history when approval was requested
        self.pending_generation_params: Dict = {}  # Temperature, max_tokens, etc. for continuation

        # Context management state
        self.context_summary: Optional[str] = None  # LLM-generated summary of older messages
        self.summarized_message_count: int = 0      # Number of messages included in summary
        self.current_context_size: int = 0          # Current model's effective ctx_size
        self.current_history_tokens: int = 0        # Cached token count (updated on changes)

        # Idle model unloading
        self.last_activity_time: float = time.time()
        self.idle_check_task: Optional[asyncio.Task] = None

state = ServerState()

# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = None
    temperature: Optional[float] = config.DEFAULT_TEMPERATURE
    max_tokens: Optional[int] = config.DEFAULT_MAX_TOKENS
    top_p: Optional[float] = config.DEFAULT_TOP_P
    top_k: Optional[int] = config.DEFAULT_TOP_K
    repeat_penalty: Optional[float] = config.DEFAULT_REPEAT_PENALTY
    enable_tools: Optional[bool] = None

class ChatResponse(BaseModel):
    response: str
    tokens_input: int
    tokens_generated: int
    tokens_total: int
    generation_time: float
    tokens_per_second: float
    device: str
    tools_used: Optional[List[str]] = None

class CommandRequest(BaseModel):
    command: str
    value: Optional[str] = None

class CommandResponse(BaseModel):
    status: str
    message: str
    current_value: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    model_key: str
    system_prompt: str
    is_generating: bool
    n_gpu_layers: int
    tools_enabled: bool
    # Context management fields
    context_size: int
    context_used_tokens: int
    context_used_percent: float
    context_available_output: int
    has_summary: bool
    summarized_messages: int
    # Context with tools (shows true llama-server pressure)
    tool_tokens: int
    context_with_tools_tokens: int
    context_with_tools_percent: float
    # Idle model unloading
    model_idle_unloaded: bool
    idle_timeout_seconds: int
    idle_seconds: float
    # Crash diagnostics
    crash_exit_code: Optional[int] = None
    crash_message: Optional[str] = None

class ModelInfo(BaseModel):
    key: str
    name: str
    description: str
    vram_estimate: str
    context_length: int
    recommended: bool
    is_current: bool
    exists: bool
    source: str  # NEW: "local", "huggingface", or "unavailable"

class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    current_model_key: str

class ModelSwitchRequest(BaseModel):
    model_key: str

class ModelSwitchResponse(BaseModel):
    status: str
    message: str
    previous_model: str
    new_model: str
    model_key: str

class HardwareInfoResponse(BaseModel):
    profile: Dict[str, Any]
    current_config: Dict[str, Any]
    device_string: str

class ToolApprovalRequest(BaseModel):
    """Model for tool approval requests"""
    tool_name: str
    arguments: Dict[str, Any]
    tool_call_id: str

class ApprovalRequiredResponse(BaseModel):
    """Response when tools require user approval"""
    approval_required: bool = True
    tools_pending: List[ToolApprovalRequest]
    message: str

class ToolApprovalDecision(BaseModel):
    """User's decision on a specific tool call"""
    tool_call_id: str
    approved: bool

class ChatApprovalRequest(BaseModel):
    """Request to approve/deny pending tool calls"""
    decisions: List[ToolApprovalDecision]

# ============================================================================
# OpenAI-Compatible Request/Response Models (/v1/ endpoints)
# ============================================================================

class OAIMessage(BaseModel):
    """OpenAI-compatible message supporting all role types."""
    role: str
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class OAIStreamOptions(BaseModel):
    include_usage: Optional[bool] = None

class OAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[OAIMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[OAIStreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    # Extensions (passed through to llama-server)
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None

class OAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OAIResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    refusal: Optional[str] = None

class OAIChoice(BaseModel):
    index: int
    message: OAIResponseMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None

class OAIChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OAIChoice]
    usage: OAIUsage
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None

class OAIModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class OAIModelList(BaseModel):
    object: str = "list"
    data: List[OAIModelObject]

# ============================================================================
# Helper Functions & Decorators
# ============================================================================

from functools import wraps

def require_llama_server(func):
    """Decorator to ensure llama-server is healthy before endpoint execution.

    If the model was idle-unloaded, this will automatically reload it
    before proceeding with the request.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not state.llama_manager:
            raise HTTPException(status_code=503, detail="llama-server is not initialized")

        if not state.llama_manager.is_healthy():
            if state.llama_manager.idle_unloaded:
                # Auto-reload: model was unloaded due to idle timeout
                logger.info("Auto-reloading model for incoming request (was idle-unloaded)")
                success = await asyncio.to_thread(state.llama_manager.reload_after_idle)
                if not success:
                    raise HTTPException(
                        status_code=503,
                        detail="Failed to reload model after idle unload"
                    )
            else:
                raise HTTPException(status_code=503, detail="llama-server is not running")

        # Update activity timestamp for idle tracking
        state.last_activity_time = time.time()

        return await func(*args, **kwargs)
    return wrapper

def require_not_generating(func):
    """Decorator to ensure server is not currently generating"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if state.is_generating:
            raise HTTPException(status_code=503, detail="Server busy - already generating response")
        return await func(*args, **kwargs)
    return wrapper

def get_device_string() -> str:
    """Get human-readable device string"""
    n_gpu_layers = config.LLAMA_SERVER_CONFIG['n_gpu_layers']
    if n_gpu_layers == -1:
        return "GPU (all layers)"
    elif n_gpu_layers == 0:
        return "CPU"
    else:
        return f"Hybrid (GPU: {n_gpu_layers} layers)"

def build_messages_for_llama(system_prompt: Optional[str] = None) -> List[Dict]:
    """
    Build messages array with system prompt, context summary, and conversation history.

    Order of messages:
    1. System prompt (if any)
    2. Context summary (if conversation has been summarized)
    3. Recent conversation history
    """
    messages = []

    # Add system prompt if provided
    prompt = system_prompt or state.system_prompt
    if prompt:
        messages.append({"role": "system", "content": prompt})

    # Add context summary if we have one (from previous summarization)
    if state.context_summary:
        messages.append({
            "role": "system",
            "content": f"[Previous conversation summary: {state.context_summary}]"
        })

    # Add conversation history
    messages.extend(state.conversation_history)

    return messages


def summarize_conversation_history(messages_to_summarize: List[Dict], max_summary_tokens: int = 500) -> str:
    """
    Use loaded LLM to generate concise summary of older messages.

    Args:
        messages_to_summarize: List of older messages to condense
        max_summary_tokens: Maximum tokens for the summary

    Returns:
        Summary string, or empty string if summarization fails
    """
    if not messages_to_summarize:
        return ""

    if not state.llama_manager or not state.llama_manager.is_healthy():
        logger.warning("Cannot summarize: llama-server not healthy")
        return ""

    # Build summarization prompt
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in messages_to_summarize
    ])

    summarization_messages = [
        {
            "role": "system",
            "content": "You are a conversation summarizer. Summarize the following conversation concisely, preserving key facts, decisions, code snippets, and important context. Be brief but complete."
        },
        {
            "role": "user",
            "content": f"Summarize this conversation:\n\n{conversation_text}\n\nProvide a concise summary:"
        }
    ]

    try:
        llama_url = state.llama_manager.server_url
        payload = {
            "messages": summarization_messages,
            "temperature": 0.3,  # Low temperature for factual summary
            "max_tokens": max_summary_tokens,
            "stream": False
        }

        response = requests.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            summary_tokens = config.count_tokens(summary)
            logger.info(f"Generated summary: {summary_tokens} tokens from {len(messages_to_summarize)} messages")
            return summary.strip()
        else:
            logger.error(f"Summarization failed: {response.status_code}")
            return ""

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return ""


def check_and_summarize_if_needed() -> bool:
    """
    Check if conversation history needs summarization and perform it if so.

    Returns:
        True if summarization was performed, False otherwise
    """
    if not state.conversation_history:
        return False

    # Get current context limit
    ctx_limit = state.current_context_size or config.LLAMA_SERVER_CONFIG.get('ctx_size', 12288)

    # Calculate threshold
    threshold_tokens = int(ctx_limit * config.CONTEXT_SUMMARIZATION_THRESHOLD)

    # Calculate current usage with tiktoken (accurate)
    system_tokens = config.count_tokens(state.system_prompt) if state.system_prompt else 0
    summary_tokens = config.count_tokens(state.context_summary) if state.context_summary else 0
    history_tokens = config.count_messages_tokens(state.conversation_history)

    total_tokens = system_tokens + summary_tokens + history_tokens
    state.current_history_tokens = total_tokens  # Cache for health endpoint

    logger.debug(f"Context usage: {total_tokens}/{ctx_limit} tokens ({100*total_tokens/ctx_limit:.1f}%)")

    # Check if we need to summarize
    if total_tokens < threshold_tokens:
        return False

    logger.info(f"Context threshold exceeded: {total_tokens}/{threshold_tokens}. Summarizing...")

    # Determine how many messages to keep in full
    min_recent = config.CONTEXT_MINIMUM_RECENT_MESSAGES
    if len(state.conversation_history) <= min_recent:
        logger.warning("Too few messages to summarize, skipping")
        return False

    # Find split point: keep recent messages, summarize older ones
    messages_to_keep = state.conversation_history[-min_recent:]
    messages_to_summarize = state.conversation_history[:-min_recent]

    # Include any existing summary context in what we're summarizing
    if state.context_summary:
        messages_to_summarize = [
            {"role": "system", "content": f"Previous context summary: {state.context_summary}"}
        ] + messages_to_summarize

    # Generate summary
    summary = summarize_conversation_history(
        messages_to_summarize,
        max_summary_tokens=config.CONTEXT_SUMMARY_MAX_TOKENS
    )

    if summary:
        # Update state
        state.context_summary = summary
        state.summarized_message_count += len(messages_to_summarize)
        state.conversation_history = messages_to_keep

        # Recalculate and cache token counts
        new_history_tokens = config.count_messages_tokens(state.conversation_history)
        new_summary_tokens = config.count_tokens(summary)
        state.current_history_tokens = system_tokens + new_summary_tokens + new_history_tokens

        logger.info(
            f"Summarization complete. "
            f"Reduced from {total_tokens} to {state.current_history_tokens} tokens."
        )
        return True
    else:
        logger.error("Summarization failed, context may exceed limits")
        return False


def validate_context_fits(messages: List[Dict], max_tokens: int) -> bool:
    """
    Validate that messages + expected output will fit in context.

    Args:
        messages: List of message dictionaries to send
        max_tokens: Maximum tokens expected for output

    Returns:
        True if safe, False if would overflow
    """
    ctx_limit = state.current_context_size or config.LLAMA_SERVER_CONFIG.get('ctx_size', 12288)
    hard_limit = int(ctx_limit * config.CONTEXT_HARD_LIMIT_PERCENT)

    messages_tokens = config.count_messages_tokens(messages)
    total_needed = messages_tokens + max_tokens

    if total_needed > hard_limit:
        logger.warning(f"Context overflow prevented: {total_needed} > {hard_limit}")
        return False
    return True


def call_llama_server(messages: List[Dict], **kwargs) -> Dict:
    """
    Call llama-server's /v1/chat/completions endpoint.
    
    Args:
        messages: List of message dictionaries
        **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
    Returns:
        Response dictionary from llama-server
    """
    if not state.llama_manager or not state.llama_manager.is_healthy():
        raise HTTPException(status_code=503, detail="llama-server is not running")
    
    llama_url = state.llama_manager.server_url
    
    # Build request payload
    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", config.DEFAULT_TEMPERATURE),
        "max_tokens": kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
        "top_p": kwargs.get("top_p", config.DEFAULT_TOP_P),
        "top_k": kwargs.get("top_k", config.DEFAULT_TOP_K),
        "repeat_penalty": kwargs.get("repeat_penalty", config.DEFAULT_REPEAT_PENALTY),
        "stream": False,
    }
    
    # Add tools if enabled and provided
    if kwargs.get("tools"):
        payload["tools"] = kwargs["tools"]
        payload["tool_choice"] = "auto"
    
    try:
        response = requests.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            timeout=300
        )
        
        if response.status_code != 200:
            logger.error(f"llama-server returned error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"llama-server error: {response.text}"
            )
        
        return response.json()
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to llama-server timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to reach llama-server: {str(e)}")


async def call_llama_server_streaming(
    messages: List[Dict],
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Call llama-server with streaming enabled, yielding SSE events.

    Args:
        messages: List of message dictionaries
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Yields:
        SSE-formatted strings: "data: {...}\n\n"
    """
    if not state.llama_manager or not state.llama_manager.is_healthy():
        yield f'data: {{"error": "llama-server is not running"}}\n\n'
        return

    llama_url = state.llama_manager.server_url

    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", config.DEFAULT_TEMPERATURE),
        "max_tokens": kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
        "top_p": kwargs.get("top_p", config.DEFAULT_TOP_P),
        "top_k": kwargs.get("top_k", config.DEFAULT_TOP_K),
        "repeat_penalty": kwargs.get("repeat_penalty", config.DEFAULT_REPEAT_PENALTY),
        "stream": True,
        # Request usage stats in the final streaming chunk
        "stream_options": {"include_usage": True},
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{llama_url}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f'data: {{"error": "llama-server error: {error_text.decode()}"}}\n\n'
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        # Don't forward [DONE] — generate_stream() sends its
                        # own after the stream_end metadata event
                        if line == "data: [DONE]":
                            break

                        # Forward SSE event from llama-server
                        yield f"{line}\n\n"

    except httpx.TimeoutException:
        yield f'data: {{"error": "Request to llama-server timed out"}}\n\n'
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield f'data: {{"error": "Streaming error: {str(e)}"}}\n\n'


# ============================================================================
# Idle Model Unloading
# ============================================================================

async def idle_check_loop():
    """Background task that periodically checks for idle timeout, unloads the model,
    and proactively detects unexpected process crashes."""
    while True:
        await asyncio.sleep(config.IDLE_CHECK_INTERVAL_SECONDS)

        if state.shutdown_requested:
            break

        # Proactive crash detection — call is_healthy() so it captures
        # crash diagnostics (exit code + stderr) immediately rather than
        # waiting for a client request to trigger the check.
        if (state.llama_manager and state.llama_manager.is_running
                and not state.is_generating):
            if not state.llama_manager.is_healthy():
                crash = state.llama_manager.last_crash_info
                if crash:
                    logger.error(
                        f"Background check: llama-server crashed "
                        f"(exit code: {crash.get('exit_code')})"
                    )

        # Skip idle unload if disabled, generating, or not running
        if config.IDLE_TIMEOUT_SECONDS <= 0:
            continue
        if state.is_generating:
            continue
        if not state.llama_manager or not state.llama_manager.is_running:
            continue

        idle_seconds = time.time() - state.last_activity_time
        if idle_seconds >= config.IDLE_TIMEOUT_SECONDS:
            idle_minutes = idle_seconds / 60
            logger.info(f"Model idle for {idle_minutes:.1f} minutes (threshold: {config.IDLE_TIMEOUT_SECONDS / 60:.0f}m). Unloading...")
            state.llama_manager.idle_unload()


# ============================================================================
# Process Cleanup (signal handlers + atexit)
# ============================================================================

_cleanup_done = False

def cleanup_llama_process():
    """Ensure llama-server process is stopped. Safe to call multiple times."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    if state.llama_manager and state.llama_manager.is_running:
        logger.info("Cleanup: stopping llama-server process tree")
        state.llama_manager.stop()

# Register atexit handler as a safety net
atexit.register(cleanup_llama_process)


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT to ensure clean process shutdown."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name} - cleaning up llama-server")
    cleanup_llama_process()
    # Re-raise for default behavior (Uvicorn needs to see the signal too)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register signal handlers for clean shutdown
# Only set handlers in the main process (not in Uvicorn workers)
try:
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
except (OSError, ValueError):
    # May fail if not on main thread (e.g., during testing)
    pass


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize llama-server on startup"""
    logger.info("=" * 60)
    logger.info("AI Lab Server (llama.cpp) Starting...")
    logger.info("=" * 60)

    # Display hardware configuration
    logger.info("=" * 60)
    logger.info("Hardware Configuration")
    logger.info("=" * 60)
    hw = config.HARDWARE_PROFILE

    if hw.get("gpu", {}).get("has_gpu"):
        gpu = hw["gpu"]
        logger.info(f"GPU: {gpu['name']} ({gpu['vram_gb']}GB VRAM)")
        logger.info(f"  CUDA: {gpu.get('cuda_version', 'unknown')}")
        logger.info(f"  Driver: {gpu.get('driver_version', 'unknown')}")
    else:
        logger.info("GPU: None detected")

    cpu = hw.get("cpu", {})
    logger.info(f"CPU: {cpu.get('name', 'Unknown')}")
    logger.info(f"  Cores: {cpu.get('physical_cores', 'unknown')} physical, {cpu.get('logical_cores', 'unknown')} logical")

    memory = hw.get("memory", {})
    logger.info(f"RAM: {memory.get('total_gb', 0):.1f}GB total ({memory.get('available_gb', 0):.1f}GB available)")

    recommended = hw.get("recommended_config", {})
    logger.info(f"System Type: {hw.get('system_type', 'unknown')}")
    logger.info(f"Inference Mode: {recommended.get('mode', 'unknown')}")
    logger.info(f"  GPU Layers: {config.LLAMA_SERVER_CONFIG['n_gpu_layers']}")
    logger.info(f"  Context Size: {config.LLAMA_SERVER_CONFIG['ctx_size']} tokens")
    logger.info(f"  CPU Threads: {config.LLAMA_SERVER_CONFIG['threads'] or 'auto'}")
    # CPU optimization flags
    if config.LLAMA_SERVER_CONFIG.get('numa_mode'):
        logger.info(f"  NUMA Mode: {config.LLAMA_SERVER_CONFIG['numa_mode']}")
    if config.LLAMA_SERVER_CONFIG.get('batch_size'):
        logger.info(f"  Batch Size: {config.LLAMA_SERVER_CONFIG['batch_size']}")
    if config.LLAMA_SERVER_CONFIG.get('ubatch_size'):
        logger.info(f"  UBatch Size: {config.LLAMA_SERVER_CONFIG['ubatch_size']}")
    if config.LLAMA_SERVER_CONFIG.get('mlock'):
        logger.info(f"  Memory Lock: enabled")
    if config.LLAMA_SERVER_CONFIG.get('threads_batch'):
        logger.info(f"  Threads (batch/prefill): {config.LLAMA_SERVER_CONFIG['threads_batch']}")
    if config.LLAMA_SERVER_CONFIG.get('flash_attn'):
        logger.info(f"  Flash Attention: enabled")
    if config.LLAMA_SERVER_CONFIG.get('no_mmap'):
        logger.info(f"  No MMap (preload): enabled")
    logger.info("=" * 60)

    # Validate configuration
    issues = config.validate_config()
    if issues:
        logger.warning("Configuration notes:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    # Initialize llama manager
    state.llama_manager = LlamaServerManager(config.LLAMA_SERVER_CONFIG)

    # Set system prompt for default model
    state.system_prompt = config.get_system_prompt_for_model(config.DEFAULT_MODEL_KEY)
    if state.system_prompt != config.DEFAULT_SYSTEM_PROMPT:
        logger.info(f"Using model-specific system prompt for {config.DEFAULT_MODEL_KEY}")

    # Auto-start if configured
    if config.LLAMA_SERVER_CONFIG.get('auto_start', True):
        logger.info(f"Auto-starting llama-server with model: {config.DEFAULT_MODEL_KEY}")

        try:
            # Use get_model_source to determine local vs HF
            source_type, identifier = config.get_model_source(config.DEFAULT_MODEL_KEY)

            logger.info(f"  Source: {source_type}")
            logger.info(f"  Identifier: {identifier}")

            use_hf = (source_type == "huggingface")

            # Calculate model-specific context size
            model_info = config.get_model_info(config.DEFAULT_MODEL_KEY)
            model_ctx_length = model_info.get('context_length', 8192) if model_info else 8192
            hw_ctx_limit = config.LLAMA_SERVER_CONFIG.get('ctx_size', 12288)
            effective_ctx_size = min(model_ctx_length, hw_ctx_limit)

            logger.info(f"  Model native context: {model_ctx_length}")
            logger.info(f"  Hardware context limit: {hw_ctx_limit}")
            logger.info(f"  Effective context size: {effective_ctx_size}")

            # Start with model-specific context size
            if state.llama_manager.start(identifier, use_hf=use_hf, ctx_size=effective_ctx_size):
                logger.info(f"✓ llama-server started successfully")
                logger.info(f"  Model: {config.DEFAULT_MODEL_KEY}")
                logger.info(f"  Device: {get_device_string()}")
                logger.info(f"  Context: {effective_ctx_size} tokens")

                # Initialize context management state
                state.current_context_size = effective_ctx_size

                if use_hf:
                    logger.info(f"  Downloaded from HuggingFace")
                    logger.info(f"  Cache: {config.LLAMA_SERVER_CONFIG['cache_dir']}")
            else:
                logger.error("✗ Failed to start llama-server")

        except Exception as e:
            logger.error(f"Error during llama-server startup: {e}")
            logger.error("Server will start but chat will not work")

    # Start idle check background task
    if config.IDLE_TIMEOUT_SECONDS > 0:
        state.idle_check_task = asyncio.create_task(idle_check_loop())
        logger.info(f"Idle model unloading enabled: {config.IDLE_TIMEOUT_SECONDS / 60:.0f} minute timeout")
    else:
        logger.info("Idle model unloading disabled")

    # Set initial activity time
    state.last_activity_time = time.time()

    logger.info("=" * 60)
    logger.info(f"AI Lab Server ready on {config.HOST}:{config.PORT}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Lab Server...")

    # Cancel idle check task
    if state.idle_check_task and not state.idle_check_task.done():
        state.idle_check_task.cancel()
        try:
            await state.idle_check_task
        except asyncio.CancelledError:
            pass

    # Stop llama-server process tree
    cleanup_llama_process()

    logger.info("Shutdown complete")

# ============================================================================
# OpenAI-Compatible Error Handling
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Return OpenAI-format errors for /v1/ routes, default format otherwise."""
    if request.url.path.startswith("/v1/"):
        error_type_map = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            429: "rate_limit_error",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": error_type_map.get(exc.status_code, "server_error"),
                    "param": None,
                    "code": None,
                }
            }
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with context status"""
    model_info = config.get_model_info(state.current_model_key)
    model_name = model_info["name"] if model_info else "Unknown"

    is_loaded = state.llama_manager and state.llama_manager.is_healthy()
    is_idle_unloaded = state.llama_manager.idle_unloaded if state.llama_manager else False
    idle_seconds = time.time() - state.last_activity_time

    # Check for crash info
    crash_info = state.llama_manager.last_crash_info if state.llama_manager else None

    # Determine status string
    if is_loaded:
        health_status = "ok"
    elif is_idle_unloaded:
        health_status = "idle"
    elif crash_info:
        health_status = "crashed"
    else:
        health_status = "degraded"

    # Calculate context usage with tiktoken
    ctx_size = state.current_context_size or config.LLAMA_SERVER_CONFIG.get('ctx_size', 12288)
    system_tokens = config.count_tokens(state.system_prompt) if state.system_prompt else 0
    summary_tokens = config.count_tokens(state.context_summary) if state.context_summary else 0
    history_tokens = config.count_messages_tokens(state.conversation_history)

    used_tokens = system_tokens + summary_tokens + history_tokens
    used_percent = (used_tokens / ctx_size) * 100 if ctx_size > 0 else 0

    available_output = config.calculate_available_output_tokens(
        ctx_size=ctx_size,
        history_tokens=history_tokens,
        system_tokens=system_tokens,
        summary_tokens=summary_tokens
    )

    # Calculate tool tokens (if tools enabled)
    tool_tokens = 0
    if state.tools_enabled:
        tool_definitions = tools.get_available_tools()
        # Format as OpenAI-style tool objects (how they're sent to llama-server)
        formatted_tools = [{"type": "function", "function": tool} for tool in tool_definitions]
        tool_json = json.dumps(formatted_tools)
        tool_tokens = config.count_tokens(tool_json)

    # Context with tools = what llama-server actually sees (excluding chat template overhead)
    context_with_tools = used_tokens + tool_tokens
    context_with_tools_percent = (context_with_tools / ctx_size) * 100 if ctx_size > 0 else 0

    return HealthResponse(
        status=health_status,
        model_loaded=is_loaded,
        device=get_device_string(),
        model_name=model_name,
        model_key=state.current_model_key,
        system_prompt=state.system_prompt,
        is_generating=state.is_generating,
        n_gpu_layers=config.LLAMA_SERVER_CONFIG['n_gpu_layers'],
        tools_enabled=state.tools_enabled,
        # Context management fields
        context_size=ctx_size,
        context_used_tokens=used_tokens,
        context_used_percent=round(used_percent, 1),
        context_available_output=available_output,
        has_summary=state.context_summary is not None,
        summarized_messages=state.summarized_message_count,
        # Context with tools
        tool_tokens=tool_tokens,
        context_with_tools_tokens=context_with_tools,
        context_with_tools_percent=round(context_with_tools_percent, 1),
        # Idle model unloading
        model_idle_unloaded=is_idle_unloaded,
        idle_timeout_seconds=config.IDLE_TIMEOUT_SECONDS,
        idle_seconds=round(idle_seconds, 1),
        # Crash diagnostics
        crash_exit_code=crash_info.get("exit_code") if crash_info else None,
        crash_message=crash_info.get("stderr", "")[:200] if crash_info else None,
    )

@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List all available models"""
    models_list = []

    for key, info in config.MODELS.items():
        try:
            source_type, _ = config.get_model_source(key)
        except ValueError:
            source_type = "unavailable"

        models_list.append(ModelInfo(
            key=key,
            name=info["name"],
            description=info["description"],
            vram_estimate=info["vram_estimate"],
            context_length=info["context_length"],
            recommended=info["recommended"],
            is_current=(key == state.current_model_key),
            exists=config.model_exists(key),
            source=source_type
        ))

    return ModelsListResponse(
        models=models_list,
        current_model_key=state.current_model_key
    )

@app.post("/model/switch", response_model=ModelSwitchResponse)
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model (local or HuggingFace)"""
    
    # Check if model exists in registry
    model_info = config.get_model_info(request.model_key)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {request.model_key}")
    
    # Check if model is available
    if not config.model_exists(request.model_key):
        raise HTTPException(status_code=404, detail=f"Model not available: {request.model_key}")
    
    # Already using this model
    if request.model_key == state.current_model_key:
        return ModelSwitchResponse(
            status="success",
            message=f"Already using {request.model_key}",
            previous_model=state.current_model_key,
            new_model=state.current_model_key,
            model_key=state.current_model_key
        )

    # Check if currently generating
    if state.is_generating:
        raise HTTPException(
            status_code=503,
            detail="Cannot switch models while generating response"
        )

    previous_key = state.current_model_key
    logger.info(f"Switching from {previous_key} to {request.model_key}")

    try:
        # Determine source
        source_type, identifier = config.get_model_source(request.model_key)
        use_hf = (source_type == "huggingface")

        if use_hf:
            logger.info(f"Downloading from HuggingFace: {identifier}")

        # Calculate model-specific context size
        model_ctx_length = model_info.get('context_length', 8192)
        hw_ctx_limit = config.LLAMA_SERVER_CONFIG.get('ctx_size', 12288)
        effective_ctx_size = min(model_ctx_length, hw_ctx_limit)

        logger.info(f"Model native context: {model_ctx_length}, Hardware limit: {hw_ctx_limit}")
        logger.info(f"Using effective context size: {effective_ctx_size}")

        # Restart with new model and model-specific context size
        if state.llama_manager.restart(identifier, use_hf=use_hf, ctx_size=effective_ctx_size):
            state.current_model_key = request.model_key
            state.conversation_history = []

            # Update context management state
            state.current_context_size = effective_ctx_size
            state.context_summary = None
            state.summarized_message_count = 0
            state.current_history_tokens = 0

            # Update system prompt for new model
            state.system_prompt = config.get_system_prompt_for_model(request.model_key)
            if state.system_prompt != config.DEFAULT_SYSTEM_PROMPT:
                logger.info(f"Using model-specific system prompt for {request.model_key}")

            logger.info(f"✓ Switched to {request.model_key} with {effective_ctx_size} context")

            return ModelSwitchResponse(
                status="success",
                message=f"Switched to {request.model_key} (ctx: {effective_ctx_size})" +
                       (" (downloaded from HF)" if use_hf else ""),
                previous_model=previous_key,
                new_model=request.model_key,
                model_key=state.current_model_key
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to switch model")
            
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=Union[ChatResponse, ApprovalRequiredResponse])
@require_llama_server
@require_not_generating
async def chat(request: ChatRequest):
    """Main chat endpoint with tool calling support"""

    if state.shutdown_requested:
        raise HTTPException(status_code=503, detail="Server shutting down")
    
    try:
        state.is_generating = True
        start_time = time.time()

        # Update conversation history from request
        state.conversation_history = [{"role": m.role, "content": m.content} for m in request.messages]

        # Check if summarization is needed and perform it
        check_and_summarize_if_needed()

        # Determine if tools should be enabled for this request
        tools_enabled = request.enable_tools if request.enable_tools is not None else state.tools_enabled

        # Prepare tool definitions if enabled
        tools_payload = None
        if tools_enabled:
            tools_payload = tools.get_available_tools()
            # Wrap in OpenAI format
            formatted_tools = [
                {
                    "type": "function",
                    "function": tool
                }
                for tool in tools_payload
            ]

            #llama_payload["tools"] = formatted_tools
            #llama_payload["tool_choice"] = "auto"

        # Build messages with system prompt
        messages = build_messages_for_llama(request.system_prompt)

        # Calculate dynamic max_tokens based on available context
        ctx_size = state.current_context_size or config.LLAMA_SERVER_CONFIG.get('ctx_size', 12288)
        system_tokens = config.count_tokens(state.system_prompt) if state.system_prompt else 0
        summary_tokens = config.count_tokens(state.context_summary) if state.context_summary else 0
        history_tokens = config.count_messages_tokens(state.conversation_history)

        available_output = config.calculate_available_output_tokens(
            ctx_size=ctx_size,
            history_tokens=history_tokens,
            system_tokens=system_tokens,
            summary_tokens=summary_tokens
        )

        # Use the minimum of: user request, available space
        effective_max_tokens = min(
            request.max_tokens or config.DEFAULT_MAX_TOKENS,
            available_output
        )

        logger.debug(f"Output tokens: requested={request.max_tokens}, available={available_output}, using={effective_max_tokens}")

        # Validate context fits (hard limit check)
        if not validate_context_fits(messages, effective_max_tokens):
            raise HTTPException(
                status_code=400,
                detail="Request would exceed context limit. Try a shorter conversation or lower max_tokens."
            )

        # Generate response (with potential tool iterations)
        tools_used = []
        iterations = 0
        max_iterations = config.MAX_TOOL_ITERATIONS

        total_tokens_in = 0
        total_tokens_out = 0

        # This is the complete tool loop section that goes in your /chat endpoint
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Generation iteration {iterations}/{max_iterations}")

            # Call llama-server
            llama_response = call_llama_server(
                messages=messages,
                temperature=request.temperature,
                max_tokens=effective_max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                tools=formatted_tools if tools_enabled else None
            )
            
            # Extract response
            choice = llama_response["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "stop")
            
            # Track tokens
            usage = llama_response.get("usage", {})
            total_tokens_in += usage.get("prompt_tokens", 0)
            total_tokens_out += usage.get("completion_tokens", 0)
            
            # Check for tool calls
            tool_calls = message.get("tool_calls", [])

            if not tool_calls or not tools_enabled:
                # No tool calls - this is the final response
                response_text = message.get("content", "")

                # Update conversation history
                state.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })

                break

            # We have tool calls - check if any require approval
            if config.TOOL_APPROVAL_MODE:
                tools_needing_approval = []
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    if function_name in config.TOOLS_REQUIRING_APPROVAL:
                        arguments = json.loads(tool_call["function"]["arguments"])
                        tools_needing_approval.append(
                            ToolApprovalRequest(
                                tool_name=function_name,
                                arguments=arguments,
                                tool_call_id=tool_call["id"]
                            )
                        )

                # If any tools need approval, pause and request user confirmation
                if tools_needing_approval:
                    logger.info(f"{len(tools_needing_approval)} tool(s) require approval - pausing execution")

                    # Save state for resumption after approval
                    state.pending_tool_calls = tool_calls
                    state.pending_messages = messages.copy()
                    state.pending_generation_params = {
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                        "repeat_penalty": request.repeat_penalty,
                        "tools_enabled": tools_enabled,
                        "iterations": iterations,
                        "total_tokens_in": total_tokens_in,
                        "total_tokens_out": total_tokens_out,
                        "tools_used": tools_used.copy(),
                        "formatted_tools": formatted_tools if tools_enabled else None
                    }

                    # Return approval request to client
                    state.is_generating = False
                    return ApprovalRequiredResponse(
                        tools_pending=tools_needing_approval,
                        message=f"The AI wants to use {len(tools_needing_approval)} tool(s) that require your approval."
                    )

            # Execute tool calls (either no approval needed, or approval mode disabled)
            logger.info(f"Executing {len(tool_calls)} tool call(s)")
            
            # Add the assistant's message with tool calls to the conversation
            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls
            })
            
            # Execute each tool call and collect results
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
                
                logger.info(f"Tool call: {function_name}({arguments})")
                
                # Execute the tool using the decorator registry
                try:
                    result = tools.execute_tool(function_name, arguments)

                    # Build logging string for tools_used
                    key_param = tools.get_tool_key_param(function_name)
                    if key_param and key_param in arguments:
                        # Special formatting for move/copy operations
                        if function_name in ["move_file", "copy_file"] and "destination" in arguments:
                            tools_used.append(f"{function_name}({arguments[key_param]}→{arguments['destination']})")
                        else:
                            tools_used.append(f"{function_name}({arguments[key_param]})")
                    else:
                        tools_used.append(function_name)

                    logger.info(f"Tool result: {result}")

                except Exception as e:
                    result = {"error": str(e)}
                    logger.error(f"Tool execution error: {e}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result)
                })
            
            # Continue loop to get next response from the model
            # The model will now see the tool results and can either:
            # 1. Call more tools
            # 2. Provide a final answer
        
        # Calculate final stats
        total_time = time.time() - start_time
        tokens_per_second = total_tokens_out / total_time if total_time > 0 else 0
        
        logger.info(f"Generation complete: {total_tokens_out} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return ChatResponse(
            response=response_text,
            tokens_input=total_tokens_in,
            tokens_generated=total_tokens_out,
            tokens_total=total_tokens_in + total_tokens_out,
            generation_time=total_time,
            tokens_per_second=tokens_per_second,
            device=get_device_string(),
            tools_used=tools_used if tools_used else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        state.is_generating = False


@app.post("/chat/approve", response_model=ChatResponse)
@require_llama_server
@require_not_generating
async def approve_tools(approval_request: ChatApprovalRequest):
    """
    Handle approval/denial of pending tool calls and continue generation.

    This endpoint is called after the client receives an ApprovalRequiredResponse
    and the user has made decisions on which tools to approve.
    """
    if state.shutdown_requested:
        raise HTTPException(status_code=503, detail="Server shutting down")

    if not state.pending_tool_calls:
        raise HTTPException(status_code=400, detail="No pending tool calls to approve")

    try:
        state.is_generating = True
        start_time = time.time()

        # Restore saved state
        messages = state.pending_messages.copy()
        tool_calls = state.pending_tool_calls.copy()
        params = state.pending_generation_params

        iterations = params.get("iterations", 1)
        max_iterations = config.MAX_TOOL_ITERATIONS
        total_tokens_in = params.get("total_tokens_in", 0)
        total_tokens_out = params.get("total_tokens_out", 0)
        tools_used = params.get("tools_used", [])
        formatted_tools = params.get("formatted_tools")

        # Create approval lookup
        approvals = {d.tool_call_id: d.approved for d in approval_request.decisions}

        # Add the assistant's message with tool calls to the conversation
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        })

        # Execute approved tool calls, skip denied ones
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]

            # Check if this tool was approved
            if tool_call_id in approvals and not approvals[tool_call_id]:
                logger.info(f"Tool call DENIED by user: {function_name}({arguments})")
                result = {
                    "error": "Tool execution denied by user",
                    "approved": False
                }
            else:
                logger.info(f"Tool call APPROVED: {function_name}({arguments})")

                # Execute the tool
                try:
                    result = tools.execute_tool(function_name, arguments)

                    # Build logging string for tools_used
                    key_param = tools.get_tool_key_param(function_name)
                    if key_param and key_param in arguments:
                        if function_name in ["move_file", "copy_file"] and "destination" in arguments:
                            tools_used.append(f"{function_name}({arguments[key_param]}→{arguments['destination']})")
                        else:
                            tools_used.append(f"{function_name}({arguments[key_param]})")
                    else:
                        tools_used.append(function_name)

                    logger.info(f"Tool result: {result}")

                except Exception as e:
                    result = {"error": str(e)}
                    logger.error(f"Tool execution error: {e}")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result)
            })

        # Clear pending state
        state.pending_tool_calls = []
        state.pending_messages = []
        state.pending_generation_params = {}

        # Continue generation loop with tool results
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Generation iteration {iterations}/{max_iterations} (post-approval)")

            # Call llama-server with updated messages
            llama_response = call_llama_server(
                messages=messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                top_k=params.get("top_k"),
                repeat_penalty=params.get("repeat_penalty"),
                tools=formatted_tools
            )

            # Extract response
            choice = llama_response["choices"][0]
            message = choice["message"]

            # Track tokens
            usage = llama_response.get("usage", {})
            total_tokens_in += usage.get("prompt_tokens", 0)
            total_tokens_out += usage.get("completion_tokens", 0)

            # Check for more tool calls
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # Final response
                response_text = message.get("content", "")
                state.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                break

            # More tool calls - check for approval again
            if config.TOOL_APPROVAL_MODE:
                tools_needing_approval = []
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    if function_name in config.TOOLS_REQUIRING_APPROVAL:
                        arguments = json.loads(tool_call["function"]["arguments"])
                        tools_needing_approval.append(
                            ToolApprovalRequest(
                                tool_name=function_name,
                                arguments=arguments,
                                tool_call_id=tool_call["id"]
                            )
                        )

                if tools_needing_approval:
                    logger.info(f"{len(tools_needing_approval)} more tool(s) require approval")

                    # Save state again
                    state.pending_tool_calls = tool_calls
                    state.pending_messages = messages.copy()
                    state.pending_generation_params = {
                        "temperature": params.get("temperature"),
                        "max_tokens": params.get("max_tokens"),
                        "top_p": params.get("top_p"),
                        "top_k": params.get("top_k"),
                        "repeat_penalty": params.get("repeat_penalty"),
                        "tools_enabled": True,
                        "iterations": iterations,
                        "total_tokens_in": total_tokens_in,
                        "total_tokens_out": total_tokens_out,
                        "tools_used": tools_used.copy(),
                        "formatted_tools": formatted_tools
                    }

                    state.is_generating = False
                    return ApprovalRequiredResponse(
                        tools_pending=tools_needing_approval,
                        message=f"The AI wants to use {len(tools_needing_approval)} more tool(s)."
                    )

            # Execute non-approval-required tools
            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls
            })

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]

                try:
                    result = tools.execute_tool(function_name, arguments)
                    key_param = tools.get_tool_key_param(function_name)
                    if key_param and key_param in arguments:
                        tools_used.append(f"{function_name}({arguments[key_param]})")
                    else:
                        tools_used.append(function_name)
                except Exception as e:
                    result = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result)
                })

        # Calculate final stats
        total_time = time.time() - start_time
        tokens_per_second = total_tokens_out / total_time if total_time > 0 else 0

        logger.info(f"Generation complete (post-approval): {total_tokens_out} tokens in {total_time:.2f}s")

        return ChatResponse(
            response=response_text,
            tokens_input=total_tokens_in,
            tokens_generated=total_tokens_out,
            tokens_total=total_tokens_in + total_tokens_out,
            generation_time=total_time,
            tokens_per_second=tokens_per_second,
            device=get_device_string(),
            tools_used=tools_used if tools_used else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in approval endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        state.is_generating = False


@app.post("/chat/stream")
@require_llama_server
@require_not_generating
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with SSE responses.

    When tools are enabled, the first response is buffered to check for tool calls.
    If tool calls are detected, tool loop executes non-streaming, then final
    response is streamed character-by-character.

    When tools are disabled, pure streaming from first token.
    """
    if state.shutdown_requested:
        raise HTTPException(status_code=503, detail="Server shutting down")

    tools_enabled = request.enable_tools if request.enable_tools is not None else state.tools_enabled

    async def generate_stream() -> AsyncGenerator[str, None]:
        start_time = time.time()

        try:
            state.is_generating = True

            # Update conversation history from request
            state.conversation_history = [
                {"role": m.role, "content": m.content} for m in request.messages
            ]

            messages = build_messages_for_llama(request.system_prompt)
            tools_used = []
            accumulated_content = ""
            tokens_in = 0
            tokens_out = 0

            if tools_enabled:
                # Hybrid mode: buffer first response for tool detection
                formatted_tools = [
                    {"type": "function", "function": tool}
                    for tool in tools.get_available_tools()
                ]

                iterations = 0
                max_iterations = config.MAX_TOOL_ITERATIONS

                while iterations < max_iterations:
                    iterations += 1
                    logger.info(f"Streaming hybrid mode: iteration {iterations}/{max_iterations}")

                    # Use non-streaming for tool loop
                    llama_response = call_llama_server(
                        messages=messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repeat_penalty=request.repeat_penalty,
                        tools=formatted_tools
                    )

                    choice = llama_response["choices"][0]
                    message = choice["message"]
                    tool_calls = message.get("tool_calls", [])

                    # Track token usage from batch response
                    usage = llama_response.get("usage", {})
                    tokens_in += usage.get("prompt_tokens", 0)
                    tokens_out += usage.get("completion_tokens", 0)

                    if not tool_calls:
                        # No tools - we have the final response
                        accumulated_content = message.get("content", "")
                        break

                    # Execute tools
                    messages.append({
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": tool_calls
                    })

                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        tool_call_id = tool_call["id"]

                        try:
                            result = tools.execute_tool(function_name, arguments)
                            key_param = tools.get_tool_key_param(function_name)
                            if key_param and key_param in arguments:
                                tools_used.append(f"{function_name}({arguments[key_param]})")
                            else:
                                tools_used.append(function_name)
                        except Exception as e:
                            result = {"error": str(e)}

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(result)
                        })

                # Stream the buffered response character-by-character
                for i, char in enumerate(accumulated_content):
                    chunk = {
                        "choices": [{
                            "delta": {"content": char},
                            "index": 0
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # Small delay every 20 chars for smoother streaming effect
                    if i % 20 == 0:
                        await asyncio.sleep(0.001)

            else:
                # Pure streaming mode (no tools)
                async for event in call_llama_server_streaming(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repeat_penalty=request.repeat_penalty
                ):
                    yield event

                    # Parse SSE events to accumulate content and extract usage
                    if event.startswith("data: ") and "data: [DONE]" not in event:
                        try:
                            chunk_data = json.loads(event[6:].strip())

                            # Accumulate content from deltas
                            # Note: the final usage chunk has "choices": [] (empty),
                            # so we must guard against that before indexing.
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    accumulated_content += delta["content"]

                            # Extract usage stats (typically in final event before [DONE])
                            if "usage" in chunk_data:
                                usage = chunk_data["usage"]
                                tokens_in = usage.get("prompt_tokens", 0)
                                tokens_out = usage.get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            pass

            # Update conversation history with accumulated response
            if accumulated_content:
                state.conversation_history.append({
                    "role": "assistant",
                    "content": accumulated_content
                })

            # Send final metadata with token stats
            total_time = time.time() - start_time
            tokens_per_sec = round(tokens_out / total_time, 1) if total_time > 0 else 0

            final_meta = {
                "type": "stream_end",
                "generation_time": round(total_time, 2),
                "tokens_input": tokens_in,
                "tokens_generated": tokens_out,
                "tokens_total": tokens_in + tokens_out,
                "tokens_per_second": tokens_per_sec,
                "tools_used": tools_used if tools_used else None,
                "device": get_device_string()
            }
            yield f"data: {json.dumps(final_meta)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f'data: {{"error": "{str(e)}"}}\n\n'
        finally:
            state.is_generating = False

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/command", response_model=CommandResponse)
async def execute_command(request: CommandRequest):
    """Execute server commands"""
    
    command = request.command.lower()
    
    # System prompt command
    if command == "system":
        if request.value is None:
            return CommandResponse(
                status="ok",
                message="Current system prompt",
                current_value=state.system_prompt
            )
        elif request.value.lower() == "reset":
            # Reset to model-specific prompt (or default if no override)
            state.system_prompt = config.get_system_prompt_for_model(state.current_model_key)
            is_custom = state.system_prompt != config.DEFAULT_SYSTEM_PROMPT
            msg = f"System prompt reset to {'model-specific' if is_custom else 'default'}"
            return CommandResponse(
                status="ok",
                message=msg,
                current_value=state.system_prompt
            )
        else:
            state.system_prompt = request.value
            return CommandResponse(
                status="ok",
                message="System prompt updated",
                current_value=state.system_prompt
            )
    
    # GPU layers command
    elif command == "layers":
        if request.value is None:
            return CommandResponse(
                status="ok",
                message="Current GPU layers",
                current_value=str(config.LLAMA_SERVER_CONFIG['n_gpu_layers'])
            )
        else:
            return CommandResponse(
                status="info",
                message="GPU layer changes require restarting llama-server (not implemented yet)",
                current_value=str(config.LLAMA_SERVER_CONFIG['n_gpu_layers'])
            )
    
    # Memory stats command
    elif command == "mem":
        stats = state.llama_manager.get_stats() if state.llama_manager else {}
        
        if stats.get("running"):
            mem_info = f"llama-server: {stats['memory_mb']:.0f}MB RAM, {stats['cpu_percent']:.1f}% CPU"
        else:
            mem_info = "llama-server not running"
        
        return CommandResponse(
            status="ok",
            message="Memory status",
            current_value=mem_info
        )
    
    # Idle timeout command
    elif command == "idle":
        if request.value is None:
            idle_seconds = time.time() - state.last_activity_time
            is_unloaded = state.llama_manager.idle_unloaded if state.llama_manager else False
            status_str = "unloaded (idle)" if is_unloaded else "loaded"
            return CommandResponse(
                status="ok",
                message=f"Model {status_str}. Idle: {idle_seconds:.0f}s. Timeout: {config.IDLE_TIMEOUT_SECONDS}s ({config.IDLE_TIMEOUT_SECONDS / 60:.0f}m)",
                current_value=str(config.IDLE_TIMEOUT_SECONDS)
            )
        else:
            try:
                new_timeout = int(request.value)
                if new_timeout < 0:
                    raise ValueError("Timeout must be >= 0")
                config.IDLE_TIMEOUT_SECONDS = new_timeout
                msg = f"Idle timeout set to {new_timeout}s ({new_timeout / 60:.0f}m)" if new_timeout > 0 else "Idle unloading disabled"
                return CommandResponse(
                    status="ok",
                    message=msg,
                    current_value=str(new_timeout)
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid value: {e}. Use seconds (e.g., '900' for 15 minutes, '0' to disable)")

    # Tools command
    elif command == "tools":
        if request.value is None:
            return CommandResponse(
                status="ok",
                message="Current tools status",
                current_value="enabled" if state.tools_enabled else "disabled"
            )
        elif request.value.lower() in ["on", "enable", "enabled", "true", "1"]:
            state.tools_enabled = True
            return CommandResponse(
                status="ok",
                message="Tools enabled",
                current_value="enabled"
            )
        elif request.value.lower() in ["off", "disable", "disabled", "false", "0"]:
            state.tools_enabled = False
            return CommandResponse(
                status="ok",
                message="Tools disabled",
                current_value="disabled"
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid value. Use 'on' or 'off'")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command}")

@app.get("/hardware", response_model=HardwareInfoResponse)
async def get_hardware_info():
    """Get detected hardware information and current configuration"""
    return HardwareInfoResponse(
        profile=config.HARDWARE_PROFILE,
        current_config={
            "n_gpu_layers": config.LLAMA_SERVER_CONFIG["n_gpu_layers"],
            "ctx_size": config.LLAMA_SERVER_CONFIG["ctx_size"],
            "threads": config.LLAMA_SERVER_CONFIG["threads"],
            "numa_mode": config.LLAMA_SERVER_CONFIG.get("numa_mode"),
            "batch_size": config.LLAMA_SERVER_CONFIG.get("batch_size"),
            "ubatch_size": config.LLAMA_SERVER_CONFIG.get("ubatch_size"),
            "mlock": config.LLAMA_SERVER_CONFIG.get("mlock", False),
            "threads_batch": config.LLAMA_SERVER_CONFIG.get("threads_batch"),
            "flash_attn": config.LLAMA_SERVER_CONFIG.get("flash_attn", False),
            "no_mmap": config.LLAMA_SERVER_CONFIG.get("no_mmap", False),
        },
        device_string=get_device_string()
    )

@app.post("/hardware/redetect")
async def redetect_hardware():
    """Force hardware re-detection (requires server restart to apply)"""
    try:
        from hardware_detector import detect_and_save
        profile = detect_and_save(config.HARDWARE_PROFILE_PATH)

        return {
            "status": "ok",
            "message": "Hardware re-detected successfully. Restart server to apply new configuration.",
            "profile": profile
        }
    except Exception as e:
        logger.error(f"Hardware re-detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to re-detect hardware: {str(e)}")

@app.post("/shutdown")
async def shutdown():
    """Shutdown server"""
    state.shutdown_requested = True
    
    if state.is_generating:
        return JSONResponse(
            content={"status": "ok", "message": "Shutdown requested - waiting for current generation"},
            status_code=200
        )
    
    # Trigger shutdown
    import asyncio
    asyncio.create_task(shutdown_server())
    
    return JSONResponse(
        content={"status": "ok", "message": "Server shutting down"},
        status_code=200
    )

async def shutdown_server():
    """Actually shut down the server after a delay"""
    import asyncio
    await asyncio.sleep(1)
    import os
    os.kill(os.getpid(), signal.SIGTERM)

# ============================================================================
# OpenAI-Compatible Endpoints (/v1/)
# ============================================================================

def _make_completion_id() -> str:
    """Generate a unique chat completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def _convert_oai_messages(request_messages: List[OAIMessage]) -> List[Dict]:
    """Convert OAIMessage objects to dicts for llama-server."""
    messages = []
    for msg in request_messages:
        m = {"role": msg.role}
        m["content"] = msg.content
        if msg.name is not None:
            m["name"] = msg.name
        if msg.tool_calls is not None:
            m["tool_calls"] = msg.tool_calls
        if msg.tool_call_id is not None:
            m["tool_call_id"] = msg.tool_call_id
        messages.append(m)
    return messages


def _build_v1_payload(
    request: OAIChatRequest,
    messages: List[Dict],
    tools_list: Optional[List] = None,
) -> Dict:
    """Build the payload to send to llama-server from an OpenAI-compatible request."""
    payload = {
        "messages": messages,
        "temperature": request.temperature if request.temperature is not None else config.DEFAULT_TEMPERATURE,
        "max_tokens": (
            request.max_completion_tokens
            or request.max_tokens
            or config.DEFAULT_MAX_TOKENS
        ),
        "top_p": request.top_p if request.top_p is not None else config.DEFAULT_TOP_P,
        "stream": False,
    }

    # Extensions (llama-server supports these)
    if request.top_k is not None:
        payload["top_k"] = request.top_k
    if request.repeat_penalty is not None:
        payload["repeat_penalty"] = request.repeat_penalty

    # Standard OpenAI params forwarded to llama-server
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    if request.stop is not None:
        payload["stop"] = request.stop
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.logprobs is not None:
        payload["logprobs"] = request.logprobs
    if request.top_logprobs is not None:
        payload["top_logprobs"] = request.top_logprobs
    if request.response_format is not None:
        payload["response_format"] = request.response_format
    if request.n is not None and request.n != 1:
        payload["n"] = request.n

    # Tools
    if tools_list:
        payload["tools"] = tools_list
        payload["tool_choice"] = request.tool_choice if request.tool_choice is not None else "auto"

    return payload


@app.post("/v1/chat/completions")
@require_llama_server
@require_not_generating
async def v1_chat_completions(request: OAIChatRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=true) and non-streaming modes.

    Tool calling modes:
    - Client-side: When 'tools' is provided in the request, tool_calls are
      returned in the response for the client to execute (standard OpenAI flow).
    - Server-side: When no 'tools' in request but server tools are enabled,
      tools are executed server-side and the final response is returned.
    """
    if state.shutdown_requested:
        raise HTTPException(status_code=503, detail="Server shutting down")

    messages = _convert_oai_messages(request.messages)

    # Determine tool mode
    client_side_tools = request.tools is not None
    server_side_tools = not client_side_tools and state.tools_enabled

    active_tools = None
    if client_side_tools:
        active_tools = request.tools
    elif server_side_tools:
        active_tools = [
            {"type": "function", "function": t}
            for t in tools.get_available_tools()
        ]

    # --- Streaming ---
    if request.stream:
        return StreamingResponse(
            _v1_generate_stream(request, messages, active_tools, server_side_tools),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-streaming ---
    try:
        state.is_generating = True

        if server_side_tools:
            return await _v1_server_tool_loop(request, messages, active_tools)

        # Single call: client-side tools or no tools — proxy to llama-server
        payload = _build_v1_payload(request, messages, active_tools)
        llama_url = state.llama_manager.server_url

        response = requests.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Upstream inference error: {response.text}",
            )

        # llama-server returns OpenAI-format — forward directly
        return JSONResponse(content=response.json())

    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to inference server timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to reach inference server: {str(e)}")
    except Exception as e:
        logger.error(f"Error in /v1/chat/completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        state.is_generating = False


async def _v1_server_tool_loop(
    request: OAIChatRequest,
    messages: List[Dict],
    tools_list: List[Dict],
) -> JSONResponse:
    """
    Execute server-side tool loop and return an OpenAI-format response.

    The model may call tools up to MAX_TOOL_ITERATIONS times.  Tools are
    executed server-side and the results fed back to the model until it
    produces a final text response (or the iteration limit is reached).
    """
    llama_url = state.llama_manager.server_url
    completion_id = _make_completion_id()
    iterations = 0
    max_iterations = config.MAX_TOOL_ITERATIONS
    total_prompt_tokens = 0
    total_completion_tokens = 0
    finish_reason = "stop"
    last_content = ""

    while iterations < max_iterations:
        iterations += 1
        logger.info(f"[v1] Server tool loop iteration {iterations}/{max_iterations}")

        payload = _build_v1_payload(request, messages, tools_list)

        resp = requests.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Upstream inference error: {resp.text}",
            )

        result = resp.json()
        choice = result["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        usage = result.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        tool_calls = message.get("tool_calls")
        last_content = message.get("content") or ""

        if not tool_calls:
            break

        # Execute tool calls server-side
        logger.info(f"[v1] Executing {len(tool_calls)} server-side tool call(s)")

        messages.append({
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            logger.info(f"[v1] Tool call: {fn_name}({arguments})")

            try:
                tool_result = tools.execute_tool(fn_name, arguments)
            except Exception as e:
                tool_result = {"error": str(e)}
                logger.error(f"[v1] Tool execution error: {e}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(tool_result),
            })

    return JSONResponse(content={
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": state.current_model_key,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": last_content,
                "refusal": None,
            },
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "system_fingerprint": None,
        "service_tier": None,
    })


async def _v1_generate_stream(
    request: OAIChatRequest,
    messages: List[Dict],
    active_tools: Optional[List[Dict]],
    server_side_tools: bool,
) -> AsyncGenerator[str, None]:
    """
    Generate an OpenAI-compatible streaming response.

    - Client-side tools / no tools: proxy SSE chunks from llama-server directly.
    - Server-side tools: run tool loop non-streaming, then stream the final text
      character-by-character with compliant chunk objects.
    """
    try:
        state.is_generating = True
        completion_id = _make_completion_id()
        created = int(time.time())
        model_name = state.current_model_key
        include_usage = (
            request.stream_options is not None
            and request.stream_options.include_usage
        )

        if server_side_tools:
            # ---- Server-side tool execution (buffered, then streamed) ----
            llama_url = state.llama_manager.server_url
            iterations = 0
            max_iterations = config.MAX_TOOL_ITERATIONS
            final_content = ""
            total_prompt_tokens = 0
            total_completion_tokens = 0

            while iterations < max_iterations:
                iterations += 1
                payload = _build_v1_payload(request, messages, active_tools)

                resp = requests.post(
                    f"{llama_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )

                if resp.status_code != 200:
                    yield f'data: {json.dumps({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"index": 0, "delta": {"content": "[inference error]"}, "finish_reason": "stop"}]})}\n\n'
                    yield "data: [DONE]\n\n"
                    return

                result = resp.json()
                choice = result["choices"][0]
                message = choice["message"]
                tool_calls = message.get("tool_calls")

                usage = result.get("usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

                if not tool_calls:
                    final_content = message.get("content", "")
                    break

                messages.append({
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                    try:
                        tool_result = tools.execute_tool(fn_name, arguments)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(tool_result),
                    })

            # Stream the final response character-by-character
            # First chunk: role
            yield f'data: {json.dumps({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})}\n\n'

            # Content chunks
            for char in final_content:
                yield f'data: {json.dumps({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}]})}\n\n'

            # Finish chunk
            yield f'data: {json.dumps({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'

            # Usage chunk (if requested)
            if include_usage:
                yield f'data: {json.dumps({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [], "usage": {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens, "total_tokens": total_prompt_tokens + total_completion_tokens}})}\n\n'

            yield "data: [DONE]\n\n"

        else:
            # ---- Client-side tools / no tools: proxy from llama-server ----
            llama_url = state.llama_manager.server_url

            payload = _build_v1_payload(request, messages, active_tools)
            payload["stream"] = True
            if include_usage:
                payload["stream_options"] = {"include_usage": True}

            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{llama_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f'data: {json.dumps({"error": {"message": f"Upstream inference error: {error_text.decode()}", "type": "server_error", "param": None, "code": None}})}\n\n'
                        yield "data: [DONE]\n\n"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                            if line == "data: [DONE]":
                                return

    except Exception as e:
        logger.error(f"[v1] Streaming error: {e}", exc_info=True)
        yield f'data: {json.dumps({"id": "chatcmpl-error", "object": "chat.completion.chunk", "created": int(time.time()), "model": state.current_model_key, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
        yield "data: [DONE]\n\n"
    finally:
        state.is_generating = False


# ---- /v1/models ----

@app.get("/v1/models")
async def v1_list_models():
    """OpenAI-compatible model listing."""
    models_data = []
    for key, info in config.get_all_models().items():
        if info.get("is_finetuned"):
            owned_by = "local-finetuned"
        elif "hf_repo" in info:
            owned_by = "huggingface"
        else:
            owned_by = "local"

        models_data.append({
            "id": key,
            "object": "model",
            "created": 0,
            "owned_by": owned_by,
        })

    return JSONResponse(content={
        "object": "list",
        "data": models_data,
    })


@app.get("/v1/models/{model_id}")
async def v1_retrieve_model(model_id: str):
    """OpenAI-compatible single model retrieval."""
    model_info = config.get_model_info(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if model_info.get("is_finetuned"):
        owned_by = "local-finetuned"
    elif "hf_repo" in model_info:
        owned_by = "huggingface"
    else:
        owned_by = "local"

    return JSONResponse(content={
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": owned_by,
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI Lab Server v2.0 (llama.cpp)")
    print("=" * 60)
    print(f"Client API: http://{config.HOST}:{config.PORT}")
    print(f"llama-server: {config.LLAMA_SERVER_CONFIG['host']}:{config.LLAMA_SERVER_CONFIG['port']}")
    print(f"Default Model: {config.DEFAULT_MODEL_KEY}")
    print(f"Device: {get_device_string()}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
