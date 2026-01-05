"""
ai_server.py - AI Lab Server (llama.cpp edition)
FastAPI server that manages llama-server and provides chat interface
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import sys
import signal
import uvicorn
import json
import requests

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

# ============================================================================
# Helper Functions & Decorators
# ============================================================================

def require_llama_server(func):
    """Decorator to ensure llama-server is healthy before endpoint execution"""
    async def wrapper(*args, **kwargs):
        if not state.llama_manager or not state.llama_manager.is_healthy():
            raise HTTPException(status_code=503, detail="llama-server is not running")
        return await func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

def require_not_generating(func):
    """Decorator to ensure server is not currently generating"""
    async def wrapper(*args, **kwargs):
        if state.is_generating:
            raise HTTPException(status_code=503, detail="Server busy - already generating response")
        return await func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
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

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize llama-server on startup"""
    logger.info("=" * 60)
    logger.info("AI Lab Server (llama.cpp) Starting...")
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
