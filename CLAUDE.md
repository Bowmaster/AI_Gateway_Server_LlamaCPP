# CLAUDE.md - AI Lab Llama.CPP Developer Guide

**Purpose**: Optimize Claude AI assistance for agentic coding, architectural improvements, and extensibility of the AI Lab server ecosystem.

---

## Quick Context

**What is this?**
- A FastAPI wrapper around `llama-server` (llama.cpp inference engine)
- Stateful chat server with conversation history, tool calling, and dynamic model switching
- Designed to be consumed by client applications (like `ai_client.py`)
- Foundation for fine-tuning local models with conversation data

**Architecture**:
```
Client Apps (ai_client.py + custom clients)
        ↓ HTTP/REST
FastAPI Server (ai_server.py) :8080
        ↓ Process Management
LlamaServerManager (llama_manager.py)
        ↓ Subprocess
llama-server.exe :8081 (OpenAI-compatible)
        ↓ Inference
GGUF Model (CPU/GPU/Hybrid)
```

**Key Features**:
- ✅ Hot-swap models (restart not needed)
- ✅ Stateful conversations with history
- ✅ Tool calling with 15 built-in tools
- ✅ OpenAI-compatible API
- ✅ GPU/CPU/Hybrid inference modes
- ✅ HuggingFace model download support
- ✅ Streaming responses (SSE token-by-token)
- ✅ Fine-tuned model support (local GGUF files)

---

## Code Organization

### File Structure and Responsibility

| File | Role | Key Responsibilities |
|------|------|---------------------|
| `ai_server.py` (27KB) | API Server | FastAPI endpoints, message routing, tool loop orchestration, state management |
| `llama_manager.py` (13KB) | Process Manager | Subprocess lifecycle (start/stop/restart), health checks, HF downloads |
| `server_config.py` (12KB) | Configuration | Model registry, generation settings, device config, helper functions |
| `tools.py` (41KB) | Tool Library | 15 tools for file I/O, networking, directory ops, content search |
| `ai_client.py` (28KB) | Reference Client | Rich CLI, model selection, conversation export, slash commands |
| `test_server.py` (8KB) | Test Suite | Validation tests for health, models, chat, tools |

### Dependency Graph

```
ai_server.py
  ├─ imports: llama_manager, server_config, tools, FastAPI
  ├─ uses: LlamaServerManager (lifecycle), MODELS config, tools.get_available_tools()
  └─ stateful: ServerState (global)

llama_manager.py
  ├─ imports: requests, psutil, subprocess, HF huggingface_hub
  └─ manages: llama-server process, health checks, downloads

server_config.py
  ├─ defines: MODELS, device config, generation defaults
  └─ utilities: model validation, path resolution, info lookups

tools.py
  ├─ defines: 15 tool functions
  └─ exports: get_available_tools() (OpenAI format)

ai_client.py
  ├─ imports: requests, Rich
  └─ consumes: all /chat, /models, /command, /health endpoints
```

---

## Core Data Flows

### 1. Chat Request → Response Flow

```
User sends: POST /chat { messages, temperature, max_tokens, enable_tools }
                ↓
ai_server validates request
                ↓
build_messages_for_llama() [adds system prompt + history]
                ↓
call_llama_server() [proxy to llama-server:8081/v1/chat/completions]
                ↓
[TOOL LOOP - repeated max 5 times]
  ├─ Check response for tool_calls[]
  ├─ If tools requested:
  │   ├─ Add assistant message to history
  │   ├─ Execute each tool function (with error handling)
  │   ├─ Add tool results to history (role="tool")
  │   └─ Re-call llama-server with updated history
  └─ If no tools or max iterations reached: exit loop
                ↓
Return ChatResponse { response, tokens, timing, device, tools_used }
```

**Key insight**: Conversation history is **accumulated on server** across requests. This enables multi-turn conversations and stateful tool execution.

### 2. Model Switching Flow

```
User/client requests: POST /model/switch { model_key }
                ↓
Validate model exists: get_model_source(model_key)
                ↓
If local: llama_manager.restart(local_path)
If HF:    llama_manager.restart(hf_repo, use_hf=True)
                ↓
  ├─ Stop old llama-server process
  ├─ Start new llama-server with model
  ├─ Wait for health endpoint ready (timeout: 1min local, 5min HF)
  ├─ Poll stderr for download progress if HF
  └─ Return success
                ↓
Server state updated: current_model_key
                ↓
Client can resume conversation or reset via /chat endpoint
```

**Key insight**: Model switching restarts the subprocess but **preserves conversation history** in server state. Clients can continue or reset as needed.

### 3. Tool Execution Flow

```
Model returns: { message, tool_calls: [{ id, function: { name, arguments } }] }
                ↓
For each tool_call:
  ├─ Parse function.name → "lookup_hostname"
  ├─ Parse function.arguments JSON → { "hostname": "google.com" }
  ├─ Route to tools.lookup_hostname(**args)
  ├─ Capture result or exception
  ├─ Add tool result to message history: { role: "tool", tool_call_id, content }
  └─ Increment tool iteration counter
                ↓
If iteration < MAX_TOOL_ITERATIONS (5):
  └─ Re-call llama-server with updated history
Else:
  └─ Return to client
```

**Key insight**: Tool results are **added to history** for the next inference pass, allowing models to refine responses based on tool output.

---

## API Specification

### Endpoints Cheat Sheet

```
GET  /health              → HealthResponse
GET  /models              → ModelsListResponse
POST /model/switch        → ModelSwitchResponse
POST /chat                → ChatResponse         [MAIN ENDPOINT]
POST /command             → CommandResponse
POST /shutdown            → JSONResponse
```

### POST /chat (Main Endpoint)

**Request**:
```python
{
    "messages": [
        {"role": "user", "content": "What is google.com's IP?"}
    ],
    "system_prompt": "You are a helpful assistant",      # Optional
    "temperature": 0.7,                                   # 0.0-2.0
    "max_tokens": 2048,                                   # Default from config
    "top_p": 0.9,                                         # Nucleus sampling
    "top_k": 40,                                          # Top-K sampling
    "repeat_penalty": 1.1,                                # Repetition penalty
    "enable_tools": true                                  # Enable tool calling
}
```

**Response**:
```python
{
    "response": "The IP address of google.com is 142.250.185.46",
    "tokens_input": 42,
    "tokens_generated": 15,
    "tokens_total": 57,
    "generation_time": 2.3,                               # Seconds
    "tokens_per_second": 6.5,
    "device": "GPU (all layers)",                         # Device status
    "tools_used": ["lookup_hostname(google.com)"]         # Tools invoked
}
```

**Key behaviors**:
- If `system_prompt` in request, it overrides server default
- Conversation history is **accumulated on server** per session
- Tools are called automatically if `enable_tools: true` and model requests them
- Max 5 tool iterations per request (configurable as `MAX_TOOL_ITERATIONS`)

### POST /command (Server Control)

**Available Commands**:

| Command | Value | Effect |
|---------|-------|--------|
| `system` | (none) | Get current system prompt |
| `system` | "new prompt" | Set system prompt |
| `layers` | (none) | Get GPU layer count |
| `layers` | N (int) | Set GPU layers (-1=all, 0=CPU, N=hybrid) |
| `mem` | (none) | Get CPU/memory usage |
| `tools` | (none) | Get tools enabled status |
| `tools` | "on"\|"off" | Enable/disable tool calling |

---

## Configuration System

### Environment Variables (Override Defaults)

```bash
export LLAMA_SERVER_PATH=/path/to/llama-server
export LLAMA_CACHE=/path/to/hf/cache
export N_GPU_LAYERS=-1              # -1=all, 0=cpu, N=hybrid
```

### server_config.py Registry

**Model Definition Template** (Base Models in `MODELS`):
```python
"qwen2.5-7b-q4": {
    "name": "Qwen2.5-7B-Instruct-Q4_K_M",
    "filename": "qwen2_5-7b-instruct-q4_k_m.gguf",
    "description": "...",
    "context_length": 32768,
    "vram_estimate": "~5GB",
    "recommended": True,              # Highlighted in client UI
    "download_url": "https://...",
    "hf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",  # Optional HF source
    "usage": "Instruction following, coding, general tasks"
}
```

**Fine-Tuned Model Template** (in `FINETUNED_MODELS`):
```python
# Fine-tuned models are stored in ./models/custom/ (FINETUNED_MODELS_DIR)
"my-finetuned-3b": {
    "name": "My Fine-tuned Qwen 3B",
    "filename": "output_3b_test_q4_k_m.gguf",
    "description": "Fine-tuned on Dolly dataset",
    "context_length": 32768,
    "base_model": "qwen2.5-3b",
    "training_data": "dolly_tiny (50 examples)",
    "is_finetuned": True,
}
```

**Key Functions**:
- `get_model_source(model_key)` - Returns `("local_finetuned", path)` for fine-tuned models
- `get_all_models()` - Returns combined MODELS + FINETUNED_MODELS dictionary
- `list_available_models(include_finetuned=True)` - Lists all available model keys

**Device Configuration**:
```python
LLAMA_SERVER_CONFIG = {
    "n_gpu_layers": -1,               # -1=all GPU, 0=CPU, N=hybrid
    "n_ctx": 32768,                   # Context window size
    "n_threads": 8,                   # CPU threads (if hybrid/CPU mode)
    # ... other llama-server options
}
```

**Generation Defaults**:
```python
GENERATION_DEFAULTS = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
}
```

---

## Tool System

### Tool Architecture

Tools are defined in OpenAI-compatible format:

```python
{
    "type": "function",
    "function": {
        "name": "lookup_hostname",
        "description": "Look up the IP address of a hostname/domain using DNS",
        "parameters": {
            "type": "object",
            "properties": {
                "hostname": {
                    "type": "string",
                    "description": "The hostname or domain name to look up"
                }
            },
            "required": ["hostname"]
        }
    }
}
```

### Available Tools (15 Total)

**Networking (2)**:
- `lookup_hostname(hostname)` → `{ip, ping_ms}`
- `measure_http_latency(hostname)` → `{latency_ms}`

**File I/O (2)**:
- `read_file(path)` → `{content, lines}`
- `write_file(path, lines)` → `{status, bytes_written}`

**Directory Operations (3)**:
- `list_contents(path, show_hidden)` → `{items, total_items}`
- `search_files(path, pattern, recursive)` → `{files, count}`
- `calculate_directory_size(path, max_depth)` → `{size_bytes, size_human}`

**File Metadata (1)**:
- `get_file_info(path)` → `{size, created, modified, permissions}`

**File Management (5)**:
- `create_directory(path, parents)` → `{status, path}`
- `move_file(source, dest, overwrite)` → `{status}`
- `copy_file(source, dest, overwrite)` → `{status}`
- `delete_file(path, recursive)` → `{status}`
- `get_current_directory()` → `{path}`

**Content Search (1)**:
- `find_in_files(path, search_text, file_pattern)` → `{matches, count}`

### Safety Mechanisms

**Protected Paths** (cannot be accessed):
- Windows: `C:\`, `C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`
- Linux: `/`, `/etc`, `/proc`, `/sbin`, `/boot`, `/sys`, `/dev`

**Constraints**:
- All file operations **require absolute paths**
- Write operations **require parent directory to exist**
- Search results **limited to max_results** (default 100)
- File read/write text files only

---

## Extension Points: Adding Features

### Adding a New Tool

**Example: Add a `get_weather` tool**

1. **Create function in tools.py**:
```python
def get_weather(location: str) -> dict:
    """Get weather for a location"""
    try:
        # Implementation: call weather API
        return {"status": "success", "temp": 72, "condition": "sunny"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

2. **Add to tool registry in tools.py**:
```python
{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates"
            }
        },
        "required": ["location"]
    }
}
```

3. **Add handler in ai_server.py** (in /chat endpoint, tool loop):
```python
elif function_name == "get_weather":
    result = tools.get_weather(**arguments)
    tools_used.append(f"get_weather({arguments.get('location')})")
```

4. **Test**:
```python
# In test_server.py or interactive test
POST /chat { "messages": [{"role": "user", "content": "What's the weather in NYC?"}], "enable_tools": true }
```

### Adding a New Model

**Step 1: Register in server_config.py**:
```python
MODELS = {
    # ... existing models ...
    "new-model-q4": {
        "name": "New Model Q4",
        "filename": "new-model-q4.gguf",
        "description": "Latest model with feature X",
        "context_length": 32768,
        "vram_estimate": "~6GB",
        "recommended": False,
        "download_url": "https://huggingface.co/...",
        "hf_repo": "org/repo:Q4_K_M",  # Optional
        "usage": "General purpose, coding"
    }
}
```

**Step 2: Download model (if local)**:
```bash
# Download GGUF file to models/ directory
wget https://huggingface.co/.../resolve/main/new-model-q4.gguf -O models/new-model-q4.gguf
```

**Step 3: Switch via API**:
```bash
curl -X POST http://localhost:8080/model/switch \
  -H "Content-Type: application/json" \
  -d '{"model_key": "new-model-q4"}'
```

**Step 4: Verify**:
```bash
curl http://localhost:8080/health
# Should show: "model_key": "new-model-q4"
```

### Adding a New API Endpoint

**Template**:
```python
from pydantic import BaseModel

class MyRequest(BaseModel):
    param1: str
    param2: int = 10

class MyResponse(BaseModel):
    status: str
    result: str

@app.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    """Describe what this endpoint does"""

    # Check server is healthy
    if not state.llama_manager.is_healthy():
        raise HTTPException(status_code=503, detail="Llama server unavailable")

    try:
        # Implementation
        result = await do_something(request.param1, request.param2)
        return MyResponse(status="success", result=result)

    except HTTPException:
        raise  # Re-raise HTTP errors
    except Exception as e:
        logger.error(f"Error in my_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
```

### Adding a New Client Command

**In ai_client.py** (add to `handle_slash_command()`):
```python
elif cmd == "new-command":
    if value is None:
        # Get operation
        result = self.send_command("new-command")
        console.print(f"[cyan]Result: {result.get('message')}[/cyan]")
    else:
        # Set operation
        result = self.send_command("new-command", value)
        if result and result.get("status") == "ok":
            console.print(f"[green]✓ {result.get('message')}[/green]")
        else:
            console.print(f"[red]✗ Error[/red]")
    return True
```

**In ai_server.py** (add to `/command` endpoint):
```python
elif command == "new-command":
    if request.value is None:
        # Read operation
        return CommandResponse(
            status="ok",
            message="Current value is X",
            current_value=str(state.some_value)
        )
    else:
        # Write operation
        state.some_value = request.value
        return CommandResponse(
            status="ok",
            message="Updated successfully",
            current_value=str(request.value)
        )
```

---

## Development for Client Applications

### Building a Custom Client

**Minimal Client Example** (Python):
```python
import requests
import json

class MyClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.history = []

    def chat(self, user_message, enable_tools=False):
        self.history.append({"role": "user", "content": user_message})

        response = requests.post(
            f"{self.server_url}/chat",
            json={
                "messages": self.history,
                "enable_tools": enable_tools,
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        # Add assistant response to history
        self.history.append({
            "role": "assistant",
            "content": data["response"]
        })

        return {
            "response": data["response"],
            "tokens": data["tokens_total"],
            "tools_used": data.get("tools_used", [])
        }

# Usage
client = MyClient()
result = client.chat("What is 2+2?")
print(result["response"])
```

### Using ai_client.py as Reference

`ai_client.py` provides:
- **Request/response patterns** for all endpoints
- **Error handling** and retry logic
- **Rich console UI** formatting
- **Conversation export** in 3 formats (text, JSON, JSONL for fine-tuning)
- **Interactive model selection** UI
- **Slash command** implementation pattern

Key methods to reference:
- `send_message()` - Chat request pattern
- `send_command()` - Command pattern
- `switch_model()` - Model switching pattern
- `export_for_finetuning()` - Export format for model training

### Fine-Tuning Integration

The server supports exporting conversations for fine-tuning:

**Export Format** (JSONL for model training):
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Client Code**:
```python
# ai_client.py has this implemented
client.export_for_finetuning("training_data.jsonl")
```

**Integration Points for Fine-Tuning**:
1. Accumulate conversations via `/chat` endpoint
2. Export to JSONL via client's `export_for_finetuning()`
3. Use JSONL to fine-tune a local model (e.g., with llama.cpp's fine-tuning script)
4. Convert fine-tuned model to GGUF format
5. Register in `server_config.py` MODELS
6. Load via `/model/switch` endpoint

---

## Common Architectural Decisions

### Stateful vs Stateless

**Decision**: Server maintains conversation history (stateful).

**Why**:
- Enables natural multi-turn conversations
- Tools can reference earlier messages
- Simpler for clients (no need to manage history)

**Trade-off**:
- Memory grows with conversation length (mitigated by `MAX_CONVERSATION_TOKENS` limit)
- Cannot share state across server instances

**For multi-client scenarios**: Implement client-side history management and use `/chat` with full message array.

### Tool Loop with Max Iterations

**Decision**: Tool calling loops with max 5 iterations per request.

**Why**:
- Prevents infinite loops (e.g., tool fails, model keeps calling it)
- Gives models multiple chances to refine responses
- Returns to client for feedback if needed

**Trade-off**:
- Some complex workflows may need >5 iterations (rare)
- Client must re-request to continue

**For extended workflows**: Build multi-turn conversation with explicit user guidance between rounds.

### Subprocess Management

**Decision**: Each model switch restarts `llama-server` subprocess.

**Why**:
- llama-server doesn't support model hotswapping
- Ensures clean state for each model
- Prevents memory leaks across model switches

**Trade-off**:
- Model switching takes 1-5 minutes (depending on model size)
- Conversation history is preserved but subprocess is new

**Optimization**: For rapid model testing, batch tests instead of per-request switching.

### OpenAI API Compatibility

**Decision**: Expose OpenAI-compatible `/v1/chat/completions` endpoint (via llama-server proxy).

**Why**:
- Allows use of existing OpenAI clients
- Future migration flexibility
- Standard tooling ecosystem

**Trade-off**:
- Slightly more overhead in proxying
- Some OpenAI features may not be supported

---

## Gotchas & Best Practices

### Gotcha 1: Conversation History Growth

**Problem**: Conversation history accumulates; token count grows.

**Solution**:
- Monitor `MAX_CONVERSATION_TOKENS` (currently 3000)
- Client can reset history via `/chat` with empty messages array
- Server automatically truncates oldest messages when limit exceeded

### Gotcha 2: Tool Failures Don't Stop Processing

**Problem**: If a tool fails, the model still gets the error and may retry.

**Solution**:
- Tools return `{"status": "error", "error": "..."}` on failure
- Model sees error in history and may try a different approach
- This is usually desired behavior (graceful degradation)

### Gotcha 3: GPU Memory Management

**Problem**: Large models (14B+) can run out of VRAM.

**Solution**:
- Use quantization (Q4_K_M, Q5_K_M) to reduce memory
- Try hybrid mode: `n_gpu_layers: 20` (20 layers on GPU, rest on CPU)
- Monitor via `/command layers` endpoint

### Gotcha 4: HuggingFace Downloads Are Slow

**Problem**: First load of HF model can take 5+ minutes.

**Solution**:
- HuggingFace cache location: `~/.cache/huggingface/hub/` (or `LLAMA_CACHE` env var)
- Download manually if possible: `huggingface-cli download org/repo --include "*.gguf" --cache-dir path`
- Consider using local models for development

### Gotcha 5: Tool Path Security

**Problem**: Tools can only access absolute paths in non-protected directories.

**Solution**:
- Always provide absolute paths in tool calls
- Tools include path validation and will error on protected paths
- This is intentional for security

### Best Practice 1: Validate Model Exists Before Switching

```python
# Good
models = requests.get(f"{server_url}/models").json()
available_keys = [m["key"] for m in models["models"] if m["exists"]]
if "new-model" in available_keys:
    switch_model("new-model")

# Bad
switch_model("nonexistent-model")  # Will fail after 1 minute timeout
```

### Best Practice 2: Implement Exponential Backoff

```python
# For long-running operations (model switch, HF download)
import time
for attempt in range(5):
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.Timeout:
        wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
        time.sleep(wait_time)
```

### Best Practice 3: Maintain Conversation as List of Objects

```python
# Good - preserve history across requests
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]
response = requests.post("/chat", json={"messages": messages})

# Not recommended - lose history
response = requests.post("/chat", json={"messages": [{"role": "user", "content": "Next query"}]})
```

### Best Practice 4: Check Health Before Sending Requests

```python
# Always check before critical operations
try:
    health = requests.get(f"{server_url}/health", timeout=2).json()
    if health["status"] != "ok":
        print(f"Server degraded: {health.get('message')}")
except:
    print("Server unreachable")
    return
```

---

## Development Workflow

### Setup for Development

```bash
# 1. Verify Python + dependencies
python --version  # 3.9+
pip install fastapi uvicorn requests psutil

# 2. Verify llama-server exists
ls ./llama-server  # or wherever it's configured

# 3. Verify models directory
ls ./models/

# 4. Start server
python ai_server.py

# 5. In another terminal, test
python test_server.py
```

### Testing Checklist

Before deploying changes:

- [ ] All 5 tests in `test_server.py` pass
- [ ] Health check returns model
- [ ] Chat with tools enabled works
- [ ] Model switching works and preserves history
- [ ] Tool calling returns results correctly
- [ ] File operations fail gracefully on protected paths
- [ ] No server crashes on malformed requests

### Common Debugging

**Server won't start**:
```bash
# Check executable exists
file ./llama-server

# Check ports are free
lsof -i :8080  # FastAPI server
lsof -i :8081  # llama-server

# Check default model exists
ls models/ministral*
```

**Model won't load**:
```bash
# Check VRAM
nvidia-smi  # or your GPU tool

# Try CPU mode
curl -X POST http://localhost:8080/command \
  -H "Content-Type: application/json" \
  -d '{"command": "layers", "value": "0"}'
```

**Tools not being called**:
```bash
# Check tools are enabled
curl http://localhost:8080/command -d '{"command": "tools"}'

# Check model supports tools
# (Qwen2.5, Llama 3.2, etc. generally do)

# Enable in request
curl -X POST http://localhost:8080/chat \
  -d '{"messages": [...], "enable_tools": true}'
```

---

## Architecture Improvement Opportunities

### Potential Enhancements

1. **Concurrent Requests**: Currently processes one chat at a time (generation lock). Could queue requests.

2. **Persistent Conversation Storage**: Save conversation history to disk (SQLite/JSON) across server restarts.

3. **Batch Tool Execution**: Execute multiple tools in parallel instead of sequentially.

4. **Model Preloading**: Background preload next model while current is in use (requires dual llama-server instances).

5. **Streaming Responses**: Use SSE (Server-Sent Events) for token-by-token streaming instead of full response.

6. **Multi-GPU Support**: Load different models on different GPUs simultaneously.

7. **Token Budget System**: Limit conversation history by token count instead of message count.

8. **Tool Caching**: Cache tool results (e.g., DNS lookups, file reads) to avoid redundant calls.

9. **Request Queuing**: Queue incoming requests instead of blocking on generation lock.

10. **Model Benchmarking**: Built-in performance profiling for different models/quantizations.

---

## Integration with Fine-Tuning Workflow

A complete fine-tuning toolkit has been developed and is maintained in a separate repository. The toolkit uses QLoRA (Quantized Low-Rank Adaptation) for efficient training on consumer GPUs.

### Fine-Tuning Toolkit (Separate Repository)

The training toolkit includes:
- `Training_Setup.ps1` - Environment setup (handles RTX 50 series with PyTorch nightly cu128)
- `prepare_dataset.py` - Download, inspect, filter, and convert datasets
- `finetune.py` - QLoRA fine-tuning with configurable parameters
- `evaluate_model.py` - Before/after comparison of model responses
- `convert_to_gguf.py` - Convert LoRA adapters to GGUF format

### Data Pipeline for Model Training

```
1. Use Server + Client to Generate Data
   └─ User interactions via ai_client.py
   └─ Export via /export-ft command

2. Prepare Dataset (in training repo)
   └─ python prepare_dataset.py download --dataset dolly
   └─ Or use custom exported data

3. Fine-Tune Model
   └─ python finetune.py --dataset data/training.jsonl --base-model qwen2.5-3b
   └─ Output: LoRA adapter in output/final/

4. Evaluate Results
   └─ python evaluate_model.py --model qwen2.5-3b --lora output/final

5. Convert to GGUF
   └─ python convert_to_gguf.py --lora output/final --quantize q4_k_m
   └─ Output: models/<name>_q4_k_m.gguf

6. Integrate Back
   └─ Copy GGUF to ./models/custom/
   └─ Register in server_config.py FINETUNED_MODELS
   └─ Load via /model/switch

7. Iterate
   └─ Generate more data with improved model
   └─ Fine-tune again
```

### Training Time Estimates

| Dataset Size | Model | GPU (RTX 5070 Ti) | CPU (56 threads) |
|--------------|-------|-------------------|------------------|
| 50 examples | 3B | ~20 seconds | ~1 hour |
| 500 examples | 3B | ~3 minutes | ~6 hours |
| 500 examples | 7B | ~10 minutes | ~12 hours |

GPU training is **30-50x faster** than CPU.

### Export Format for Fine-Tuning

```jsonl
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
{"messages": [{"role": "user", "content": "How does fine-tuning work?"}, {"role": "assistant", "content": "Fine-tuning adjusts..."}]}
```

Each line is a complete conversation (messages array).

---

## Summary: Decision Tree for Development

```
I want to...

├─ Add a new tool?
│  └─ See "Adding a New Tool" section
│
├─ Add a new model?
│  └─ See "Adding a New Model" section
│
├─ Create a custom client?
│  └─ Use POST /chat endpoint, see "Building a Custom Client"
│
├─ Build a web UI?
│  └─ Use /chat, /models, /command endpoints (CORS may need enabling)
│
├─ Improve inference speed?
│  ├─ Use smaller model or lower quantization
│  ├─ Increase n_gpu_layers for GPU acceleration
│  └─ Batch requests on client side
│
├─ Support concurrent requests?
│  └─ Remove generation lock, add request queue
│
├─ Stream token-by-token?
│  └─ Use SSE with llama-server's streaming capability
│
├─ Fine-tune a model?
│  └─ Export via client.export_for_finetuning(), follow workflow above
│
└─ Debug an issue?
   └─ Check "Common Debugging" section
```

---

## Quick Links & References

- **Server Source**: `ai_server.py` - Main FastAPI app and endpoints
- **Process Management**: `llama_manager.py` - Model loading and lifecycle
- **Configuration**: `server_config.py` - Models, defaults, helpers
- **Tools**: `tools.py` - All 15 tool implementations
- **Client Reference**: `ai_client.py` - Request/response patterns, CLI UX
- **Tests**: `test_server.py` - Example API usage
- **Setup**: `README.md` - Initial setup instructions
- **User Docs**: `README.md` - Feature overview for end users

---

## Testing Systems
**System 1 (Custom gaming PC)**
CPU: AMD 5800 XT (8 core)
GPU: Nvidia RTX 5070 Ti (16 GB VRAM, Blackwell/sm_120 architecture)
RAM: 32 GB

**System 2 (DL380 Gen9)**
CPU:  Dual CPU E5-2680 v4 (28 total cores, 56 total threads)
GPU: None (But considering purchasing a low profile card that will fit in a 2u chassis)
Memory: 256
Extra Tools: Docker

## Code Style
1. We should focus on safe, reliable, guardrailed executuion first and foremost.
2. Code should be clear, succinct, and avoid being needlessly complex, so long as the style item from 1. is adhered to.

**Last Updated**: 2026-01-13
**Scope**: Optimized for Claude Code agentic development, architectural improvements, and extensibility.

