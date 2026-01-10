"""
ai_server.py - AI Lab Server (llama.cpp edition)
FastAPI server that manages llama-server and provides chat interface
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, AsyncGenerator
import logging
import sys
import signal
import uvicorn
import json
import requests
import httpx
import asyncio

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

# ============================================================================
# Helper Functions & Decorators
# ============================================================================

from functools import wraps

def require_llama_server(func):
    """Decorator to ensure llama-server is healthy before endpoint execution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not state.llama_manager or not state.llama_manager.is_healthy():
            raise HTTPException(status_code=503, detail="llama-server is not running")
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
    """Build messages array with system prompt and conversation history"""
    messages = []
    
    # Add system prompt if provided
    prompt = system_prompt or state.system_prompt
    if prompt:
        messages.append({"role": "system", "content": prompt})
    
    # Add conversation history
    messages.extend(state.conversation_history)
    
    return messages

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
                        # Forward SSE event from llama-server
                        yield f"{line}\n\n"

                        # Check for completion signal
                        if line == "data: [DONE]":
                            break

    except httpx.TimeoutException:
        yield f'data: {{"error": "Request to llama-server timed out"}}\n\n'
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield f'data: {{"error": "Streaming error: {str(e)}"}}\n\n'


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
    logger.info("=" * 60)

    # Validate configuration
    issues = config.validate_config()
    if issues:
        logger.warning("Configuration notes:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    # Initialize llama manager
    state.llama_manager = LlamaServerManager(config.LLAMA_SERVER_CONFIG)

    # Auto-start if configured
    if config.LLAMA_SERVER_CONFIG.get('auto_start', True):
        logger.info(f"Auto-starting llama-server with model: {config.DEFAULT_MODEL_KEY}")

        try:
            # NEW: Use get_model_source to determine local vs HF
            source_type, identifier = config.get_model_source(config.DEFAULT_MODEL_KEY)

            logger.info(f"  Source: {source_type}")
            logger.info(f"  Identifier: {identifier}")

            use_hf = (source_type == "huggingface")

            # Start with use_hf flag
            if state.llama_manager.start(identifier, use_hf=use_hf):
                logger.info(f"✓ llama-server started successfully")
                logger.info(f"  Model: {config.DEFAULT_MODEL_KEY}")
                logger.info(f"  Device: {get_device_string()}")

                if use_hf:
                    logger.info(f"  Downloaded from HuggingFace")
                    logger.info(f"  Cache: {config.LLAMA_SERVER_CONFIG['cache_dir']}")
            else:
                logger.error("✗ Failed to start llama-server")

        except Exception as e:
            logger.error(f"Error during llama-server startup: {e}")
            logger.error("Server will start but chat will not work")

    logger.info("=" * 60)
    logger.info(f"AI Lab Server ready on {config.HOST}:{config.PORT}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Lab Server...")
    
    if state.llama_manager:
        state.llama_manager.stop()
    
    logger.info("Shutdown complete")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = config.get_model_info(state.current_model_key)
    model_name = model_info["name"] if model_info else "Unknown"
    
    is_loaded = state.llama_manager and state.llama_manager.is_healthy()
    
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        device=get_device_string(),
        model_name=model_name,
        model_key=state.current_model_key,
        system_prompt=state.system_prompt,
        is_generating=state.is_generating,
        n_gpu_layers=config.LLAMA_SERVER_CONFIG['n_gpu_layers'],
        tools_enabled=state.tools_enabled
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

        # Restart with new model
        if state.llama_manager.restart(identifier, use_hf=use_hf):
            state.current_model_key = request.model_key
            state.conversation_history = []

            logger.info(f"✓ Switched to {request.model_key}")

            return ModelSwitchResponse(
                status="success",
                message=f"Switched to {request.model_key}" +
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

@app.post("/chat", response_model=ChatResponse)
@require_llama_server
@require_not_generating
async def chat(request: ChatRequest):
    """Main chat endpoint with tool calling support"""

    if state.shutdown_requested:
        raise HTTPException(status_code=503, detail="Server shutting down")
    
    try:
        state.is_generating = True
        import time
        start_time = time.time()
        
        # Update conversation history from request
        state.conversation_history = [{"role": m.role, "content": m.content} for m in request.messages]
        
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
                max_tokens=request.max_tokens,
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
            
            # We have tool calls - execute them
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
        import time
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

                    # Parse to accumulate content for history
                    if event.startswith("data: ") and "data: [DONE]" not in event:
                        try:
                            chunk_data = json.loads(event[6:].strip())
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                accumulated_content += delta["content"]
                        except json.JSONDecodeError:
                            pass

            # Update conversation history with accumulated response
            if accumulated_content:
                state.conversation_history.append({
                    "role": "assistant",
                    "content": accumulated_content
                })

            # Send final metadata
            total_time = time.time() - start_time
            final_meta = {
                "type": "stream_end",
                "generation_time": round(total_time, 2),
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
            state.system_prompt = config.DEFAULT_SYSTEM_PROMPT
            return CommandResponse(
                status="ok",
                message="System prompt reset to default",
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
